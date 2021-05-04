import argparse
import os
import time
import traceback
from collections import OrderedDict
from itertools import chain, product
from types import FunctionType
from typing import Callable, List
import gc
import json
from filelock import FileLock

from apex.optimizers import FusedAdam
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import sys

import mpu
from utils import set_random_seed, timeout, TimeoutError, uniform_slice
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, grid_slice_batch_and_sequence,
    ModelParallelTransformerLayer,
)
from memory_model import peak_memory_per_gpu
import checkpoint

LOSS_SCALE_FACTOR = 128.0


def initialize_distributed_env(distributed_init_method, rank, local_rank, world_size, model_parallel_size, pipeline_parallel_size):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=distributed_init_method,
        world_size=world_size,
        rank=rank,
    )
    # A small all_reduce for warmup.
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(model_parallel_size, pipeline_parallel_size)
    set_random_seed(0)
    mpu.model_parallel_cuda_manual_seed(0)


def loss_func(y):
    return torch.mean(y)


class TransformerLayers(nn.Module):
    def __init__(self, n_layers, embedding_dim, ffn_embedding_dim, num_attention_heads, mixed_precision=True):
        super().__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.mixed_precision = mixed_precision
        self.layers = []
        for _ in range(self.n_layers):
            layer = ModelParallelTransformerLayer(
                self.embedding_dim,
                self.ffn_embedding_dim,
                self.num_attention_heads,
            ).to('cuda')
            self.layers.append(layer.half() if self.mixed_precision else layer)
        self.layers = nn.ModuleList(self.layers)

    def create_cache(self, batch_size, seq_len):
        cache = []
        for layer_id in range(self.n_layers):
            k, v = self.layers[layer_id].attn.create_attn_cache(
                batch_size, seq_len, device='cuda',
                dtype=torch.float16 if self.mixed_precision else torch.float32)
            cache += [k, v]
        return cache

    def forward(self, x, cache, cache_len):
        cache_outputs = []
        for layer_id in range(self.n_layers):
            k_input = cache[2 * layer_id]
            v_input = cache[2 * layer_id + 1]
            x, (k_output, v_output) = self.layers[layer_id](x, (k_input, v_input), cache_len)
            cache_outputs += [k_output, v_output]
        return x, cache_outputs

    def create_inputs(self, batch_size, seq_len, device='cuda', random=False):
        if not random:
            x = torch.empty((seq_len, batch_size, self.embedding_dim), device=device,
                            dtype=torch.float16 if self.mixed_precision else torch.float32)
        else:
            x = torch.randn((seq_len, batch_size, self.embedding_dim), device=device,
                            dtype=torch.float16 if self.mixed_precision else torch.float32)
        return x


class PipelineSendOperator(torch.autograd.Function):
    """Send activations to the next pipeline stage"""

    @staticmethod
    def forward(ctx, x):
        rank = torch.distributed.get_rank()
        dst_rank = mpu.get_model_parallel_dst_rank()
        next_src_rank = mpu.get_model_parallel_next_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_pred_group()
        if rank == dst_rank and next_src_rank is not None:
            assert pipeline_group is not None
            dist.broadcast(x, dst_rank, group=pipeline_group)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        rank = torch.distributed.get_rank()
        dst_rank = mpu.get_model_parallel_dst_rank()
        next_src_rank = mpu.get_model_parallel_next_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_pred_group()
        model_parallel_group = mpu.get_model_parallel_group()
        if next_src_rank is not None:
            if rank == dst_rank:
                assert pipeline_group is not None
                dist.broadcast(grad_x, next_src_rank, group=pipeline_group)
            dist.broadcast(grad_x, dst_rank, group=model_parallel_group)
        return grad_x


class PipelineRecvOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rank = torch.distributed.get_rank()
        prev_dst_rank = mpu.get_model_parallel_prev_dst_rank()
        src_rank = mpu.get_model_parallel_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_succ_group()
        model_parallel_group = mpu.get_model_parallel_group()
        if prev_dst_rank is not None:
            if rank == src_rank:
                assert pipeline_group is not None
                dist.broadcast(x, prev_dst_rank, group=pipeline_group)
            dist.broadcast(x, src_rank, group=model_parallel_group)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        rank = torch.distributed.get_rank()
        prev_dst_rank = mpu.get_model_parallel_prev_dst_rank()
        src_rank = mpu.get_model_parallel_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_succ_group()
        if prev_dst_rank is not None and rank == src_rank:
            assert pipeline_group is not None
            dist.broadcast(grad_x, src_rank, group=pipeline_group)
        return None


class TeraPipeBackwardPassHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, outputs, cache_inputs, cache_outputs, batch_slices, seq_slices, batch_dim, sequence_dim, cat_outputs):
        ctx.outputs = outputs
        ctx.cache_inputs = cache_inputs
        ctx.cache_outputs = cache_outputs
        ctx.batch_slices = batch_slices
        ctx.seq_slices = seq_slices
        ctx.batch_dim = batch_dim
        ctx.sequence_dim = sequence_dim
        ctx.cat_outputs = cat_outputs
        return y

    @staticmethod
    def backward(ctx, grad_y):
        n_batch_slices = len(ctx.batch_slices)
        n_input_slices = len(ctx.seq_slices)
        if ctx.cat_outputs:
            grad_outputs = grid_slice_batch_and_sequence(
                grad_y, batch_slices=ctx.batch_slices, seq_slices=ctx.seq_slices,
                batch_dim=ctx.batch_dim, sequence_dim=ctx.sequence_dim, requires_grad=False)
        else:
            grad_outputs = np.empty((n_batch_slices, n_input_slices), dtype='O')
            for i in range(n_batch_slices):
                for j in range(n_input_slices):
                    grad_outputs[i, j] = torch.empty_like(ctx.outputs[i, j])

        for batch_id in reversed(range(n_batch_slices)):
            da = []
            for input_id in reversed(range(n_input_slices)):
                y = ctx.outputs[batch_id, input_id]
                dy = grad_outputs[batch_id, input_id]
                if input_id < n_input_slices - 1:
                    a = ctx.cache_outputs[batch_id, input_id]
                else:
                    a = []
                torch.autograd.backward([y] + a, [dy] + da)
                da = [t.grad for t in ctx.cache_inputs[batch_id, input_id]]

        del ctx.outputs
        del ctx.cache_inputs
        del ctx.cache_outputs
        del ctx.batch_slices
        del ctx.seq_slices
        del ctx.batch_dim
        del ctx.sequence_dim
        del ctx.cat_outputs
        return None, None, None, None, None, None, None, None, None


def terapipe_backward_hook(outputs, cache_inputs, cache_outputs, batch_slices, seq_slices, batch_dim=1, sequence_dim=0, cat_outputs=False):
    with torch.no_grad():
        if cat_outputs:
            y = torch.cat([torch.cat(s.tolist(), dim=sequence_dim) for s in outputs], dim=batch_dim)
        else:
            y = torch.tensor(0.0)
    y.requires_grad_()
    y = TeraPipeBackwardPassHook.apply(y, outputs, cache_inputs, cache_outputs, batch_slices, seq_slices, batch_dim, sequence_dim, cat_outputs)
    return y


pipeline_send = PipelineSendOperator.apply
pipeline_recv = PipelineRecvOperator.apply


class TeraPipe(nn.Module):
    def __init__(self, layers, batch_size, seq_len, batch_slices, seq_slices, batch_dim=1, sequence_dim=0):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_slices = batch_slices
        self.seq_slices = seq_slices
        self.batch_dim = batch_dim
        self.sequence_dim = sequence_dim
        self.layers = layers

    @property
    def n_batch_slices(self):
        return len(self.batch_slices)

    @property
    def n_input_slices(self):
        return len(self.seq_slices)

    def forward(self, inputs=None):
        if inputs is None:
            assert mpu.get_pipeline_parallel_group_rank() > 0
            inputs = self.layers.create_inputs(self.batch_size, self.seq_len)
        inputs = grid_slice_batch_and_sequence(
            inputs, batch_slices=self.batch_slices, seq_slices=self.seq_slices,
            batch_dim=self.batch_dim, sequence_dim=self.sequence_dim, requires_grad=True)
        cache_inputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        cache_outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')

        for batch_id in range(self.n_batch_slices):
            slice_batch_size = inputs[batch_id, 0].size(self.batch_dim)
            cache = self.layers.create_cache(slice_batch_size, self.seq_len)
            cache_len = 0
            for input_id in range(self.n_input_slices):
                x = inputs[batch_id, input_id]
                x = pipeline_recv(x)
                slice_seq_len = x.size(self.sequence_dim)
                cache = [c.detach().requires_grad_() for c in cache]
                y, cache_output = self.layers(x, cache, cache_len)
                y = pipeline_send(y)
                cache_inputs[batch_id, input_id] = cache
                cache_outputs[batch_id, input_id] = cache_output
                cache = cache_output
                outputs[batch_id, input_id] = y
                cache_len += slice_seq_len

        y = terapipe_backward_hook(
            outputs, cache_inputs, cache_outputs, self.batch_slices,
            self.seq_slices, self.batch_dim, self.sequence_dim,
            cat_outputs=(mpu.get_pipeline_parallel_group_rank() == mpu.get_pipeline_parallel_world_size() - 1))

        return y


def initialize_model(args):
    config = TransformerConfig.from_predefined_model(args.model, batch_size=args.batch_size)
    config.n_total_layers = config.n_layers
    config.n_layers = (config.n_total_layers // mpu.get_pipeline_parallel_world_size()
                       + int(mpu.get_pipeline_parallel_group_rank()
                             < config.n_total_layers % mpu.get_pipeline_parallel_world_size()))

    layers = TransformerLayers(
        config.n_layers, config.embedding_dim, config.ffn_embedding_dim,
        config.num_attention_heads, mixed_precision=args.mixed_precision)
    seq_slices = uniform_slice(config.seq_len, args.n_input_slices)
    batch_slices = uniform_slice(config.batch_size, args.n_batch_slices)
    pipelined_layers = TeraPipe(layers, config.batch_size, config.seq_len, batch_slices, seq_slices)
    return config, layers, pipelined_layers


def measure_iteration_time(args, n_warmup_steps=2):
    config, layers, pipelined_layers = initialize_model(args)
    optimizer = torch.optim.Adam(pipelined_layers.parameters(), lr=1e-10)
    step_times = []
    for _ in range(args.n_steps + n_warmup_steps):
        start_time = time.time()
        optimizer.zero_grad()
        if mpu.get_pipeline_parallel_group_rank() == 0:
            x = layers.create_inputs(config.batch_size, config.seq_len, random=True)
        else:
            x = None
        try:
            y = pipelined_layers(x)
            if mpu.get_pipeline_parallel_group_rank() == mpu.get_pipeline_parallel_world_size() - 1:
                loss = loss_func(y)
                loss.backward()
            else:
                y.backward()
        except:
            print(f"rank={args.rank}", traceback.format_exc())
            raise
        optimizer.step()
        step_time = time.time() - start_time
        step_times.append(step_time)
    step_times = np.array(step_times)[n_warmup_steps:]
    return np.mean(step_times), np.std(step_times)


def slice_state_dict(config, loaded_state_dict):
    sliced_state_dict = OrderedDict()
    start_layer_id = (config.n_total_layers // mpu.get_pipeline_parallel_world_size() * mpu.get_pipeline_parallel_group_rank()
                      + min(mpu.get_pipeline_parallel_group_rank(),
                            config.n_total_layers % mpu.get_pipeline_parallel_world_size()))
    end_layer_id = start_layer_id + config.n_layers
    for key, value in loaded_state_dict.items():
        keys = key.split('.')
        global_layer_id = int(keys[2])
        if start_layer_id <= global_layer_id < end_layer_id:
            local_layer_id = global_layer_id - start_layer_id
            new_key = '.'.join(keys[:2] + [str(local_layer_id)] + keys[3:])
            if keys[3] == 'attn' and keys[4] == 'in_proj':
                in_size = mpu.divide(value.size(0), mpu.get_model_parallel_world_size())
                if keys[5] in ('weight', 'bias'):
                    new_value = value[mpu.get_model_parallel_rank() * in_size
                                      :(mpu.get_model_parallel_rank() + 1) * in_size]
                else:
                    raise NotImplementedError(f"Unknown key {key}")
            elif keys[3] == 'attn' and keys[4] == 'out_proj':
                if keys[5] == 'weight':
                    out_size = mpu.divide(value.size(1), mpu.get_model_parallel_world_size())
                    new_value = value[:, mpu.get_model_parallel_rank() * out_size
                                         :(mpu.get_model_parallel_rank() + 1) * out_size]
                elif keys[5] == 'bias':
                    new_value = value
                else:
                    raise NotImplementedError(f"Unknown key {key}")
            elif keys[3] == 'fc1':
                in_size = mpu.divide(value.size(0), mpu.get_model_parallel_world_size())
                if keys[4] in ('weight', 'bias'):
                    new_value = value[mpu.get_model_parallel_rank() * in_size
                                      :(mpu.get_model_parallel_rank() + 1) * in_size]
                else:
                    raise NotImplementedError(f"Unknown key {key}")
            elif keys[3] == 'fc2':
                if keys[4] == 'weight':
                    out_size = mpu.divide(value.size(1), mpu.get_model_parallel_world_size())
                    new_value = value[:, mpu.get_model_parallel_rank() * out_size
                                         :(mpu.get_model_parallel_rank() + 1) * out_size]
                elif keys[4] == 'bias':
                    new_value = value
                else:
                    raise NotImplementedError(f"Unknown key {key}")
            else:
                new_value = value
            sliced_state_dict[new_key] = new_value
    return sliced_state_dict


def verify_one_step(args):
    if args.verify == "save":
        assert dist.get_world_size() == 1
        assert mpu.get_pipeline_parallel_world_size() == 1
        assert mpu.get_model_parallel_world_size() == 1
        assert args.n_input_slices == 1
        assert args.n_batch_slices == 1
        os.makedirs(args.verify_path, exist_ok=True)
        config, layers, pipelined_layers = initialize_model(args)
        if mpu.get_pipeline_parallel_group_rank() == 0:
            x = layers.create_inputs(config.batch_size, config.seq_len, random=True)
            torch.save(x, os.path.join(args.verify_path, 'input.pt'))
        else:
            x = None
        try:
            y = pipelined_layers(x)
            if mpu.get_pipeline_parallel_group_rank() == mpu.get_pipeline_parallel_world_size() - 1:
                loss = loss_func(y)
                loss.backward()
            else:
                y.backward()
        except:
            print(f"rank={args.rank}", traceback.format_exc())
            raise
        torch.save(pipelined_layers.state_dict(), os.path.join(args.verify_path, 'model.ckpt'))
        grad_dic = OrderedDict((x[0], x[1].grad) for x in pipelined_layers.named_parameters())
        torch.save(grad_dic, os.path.join(args.verify_path, 'model.grad.ckpt'))
    else:
        assert args.verify == "load"
        config, layers, pipelined_layers = initialize_model(args)
        with FileLock(os.path.join(args.verify_path, 'model.ckpt.lock')):
            loaded_state_dict = torch.load(os.path.join(args.verify_path, 'model.ckpt'), map_location=torch.device('cuda'))
        sliced_state_dict = slice_state_dict(config, loaded_state_dict)
        pipelined_layers.load_state_dict(sliced_state_dict)
        if mpu.get_pipeline_parallel_group_rank() == 0:
            with FileLock(os.path.join(args.verify_path, 'input.pt.lock')):
                x = torch.load(os.path.join(args.verify_path, 'input.pt'), map_location=torch.device('cuda'))
        else:
            x = None
        try:
            y = pipelined_layers(x)
            if mpu.get_pipeline_parallel_group_rank() == mpu.get_pipeline_parallel_world_size() - 1:
                loss = loss_func(y)
                loss.backward()
            else:
                y.backward()
        except:
            print(f"rank={args.rank}", traceback.format_exc())
            raise
        grad_dic = OrderedDict((x[0], x[1].grad) for x in pipelined_layers.named_parameters())
        with FileLock(os.path.join(args.verify_path, 'model.grad.ckpt.lock')):
            loaded_grad_dic = torch.load(os.path.join(args.verify_path, 'model.grad.ckpt'), map_location=torch.device('cuda'))
        sliced_grad_dic = slice_state_dict(config, loaded_grad_dic)
        assert grad_dic.keys() == sliced_grad_dic.keys()
        for k in grad_dic.keys():
            assert torch.allclose(grad_dic[k], sliced_grad_dic[k])


def main():
    parser = argparse.ArgumentParser(description='TeraPipe')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--local-rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)
    parser.add_argument('--model-parallel-size', metavar='N', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', metavar='N', type=int, default=1)
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--batch-size', metavar='N', type=int, default=1)
    parser.add_argument('--n-batch-slices', metavar='N', type=int, default=1)
    parser.add_argument('--n-input-slices', metavar='N', type=int, default=1)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--use-mpi', action='store_true', default=False)
    parser.add_argument('--verify', metavar='[save/load]', type=str, default=None)
    parser.add_argument('--verify-path', metavar='PATH', type=str, default=None)

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'

    initialize_distributed_env(
        distributed_init_method, args.rank, args.local_rank, args.world_size,
        args.model_parallel_size, args.pipeline_parallel_size)
    if args.verify is not None:
        verify_one_step(args)
    else:
        time_mean, time_std = measure_iteration_time(args)
        print(f"rank={args.rank} Time (s): mean={time_mean}, std={time_std}")


if __name__ == "__main__":
    main()
