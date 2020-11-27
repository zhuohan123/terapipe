import argparse
import os
import time
from itertools import chain

from apex import optimizers
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import mpu
import nccl
from utils import set_random_seed
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, uniform_slice_batch_and_input,
    ModelParallelTransformerLayer,
)

WARM_UP_ROUNDS = 5
LOSS_SCALE_FACTOR = 128.0

class LocalTransformer(nn.Module):
    def __init__(self, config, n_layers, mixed_precision):
        self.config = config
        self.n_layers = n_layers
        self.mixed_precision = mixed_precision

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ModelParallelTransformerLayer(
                config.embedding_dim,
                config.ffn_embedding_dim,
                config.num_attention_heads,
                device="cuda"
            )
            if self.mixed_precision:
                layer = layer.half()
            self.layers.append(layer)
        
        self.all_parameters = list(self.parameters())
        self.n_params = len(self.all_parameters)

        if self.mixed_precision:
            # '.float()' returns a copy automatically
            self.master_parameters = [
                p.detach().float().requires_grad_() for p in self.all_parameters]
            self.optimizer = optimizers.FusedAdam(self.master_parameters, lr=1e-10)
        else:
            self.optimizer = torch.optim.Adam(self.all_parameters, lr=1e-10)

    def forward(self, x, attn_caches):
        new_attn_caches = []
        if attn_caches:
            for layer, attn_cache in zip(self.layers, attn_caches):
                x, new_attn_cache = layer(x, attn_cache)
                new_attn_caches.append(new_attn_cache)
        else:
            for layer in self.layers:
                x, new_attn_cache = layer(x, None)
                new_attn_caches.append(new_attn_cache)
        return x, new_attn_caches

    def update_weights(self):
        # copy FP16 model gradients to FP32 master before comparing/optimizing
        if self.mixed_precision:
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                if master_param.grad is None:
                    master_param.grad = master_param.new(*master_param.size())
                # descale master weights
                master_param.grad.copy_(model_param.grad).mul_(1. / LOSS_SCALE_FACTOR)
                del model_param.grad

        self.optimizer.step()
        if self.mixed_precision:
            # apex set 'set_to_none=True' by default and do not have such a parameter.
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)

        if self.mixed_precision:
            # copy master updated FP32 parameters back to FP16
            with torch.no_grad():
                for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                    model_param.copy_(master_param)


class NCCLTransformer:
    def __init__(self, config, n_batch_slices, n_input_slices, distributed_init_method, world_size, data_parallel_size,
                 model_parallel_size, pipeline_parallel_size, rank, local_rank,
                 n_steps, mixed_precision=False, use_mpi=False):
        # Initialize communication.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        dist.all_reduce(torch.zeros(1).cuda())
        mpu.initialize_model_parallel(model_parallel_size, pipeline_parallel_size)
        set_random_seed(0)
        mpu.model_parallel_cuda_manual_seed(0)
        self.comm = nccl.get_nccl_communicator(local_rank, rank, world_size, use_mpi)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_group_rank = mpu.get_pipeline_parallel_group_rank()
        self.data_parallel_group = mpu.get_data_parallel_group()
        self.model_parallel_group = mpu.get_model_parallel_group()
        self.model_parallel_src_rank = mpu.get_model_parallel_src_rank()
        self.model_parallel_dst_rank = mpu.get_model_parallel_dst_rank()
        self.model_parallel_next_src_rank = (
            self.model_parallel_src_rank + self.model_parallel_size
            if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1
            else None)
        self.model_parallel_prev_dst_rank = (
            self.model_parallel_dst_rank - self.model_parallel_size
            if self.pipeline_parallel_group_rank > 0 else None)

        # Initialize model.
        self.config = config
        self.n_batch_slices = n_batch_slices
        self.n_input_slices = n_input_slices
        self.n_steps = n_steps
        n_layers = (config.n_layers // pipeline_parallel_size
                    + int(rank < config.n_layers % pipeline_parallel_size))
        self.local_transformer = LocalTransformer(config, n_layers, mixed_precision)

    def create_placeholder(self):
        return np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')

    def forward_step(self, all_batch_inputs):
        all_batch_attn_hiddens = self.create_placeholder()
        all_batch_attn_hiddens_detached = self.create_placeholder()
        all_batch_outputs = self.create_placeholder()

        for batch_id in range(self.n_batch_slices):
            # forward
            new_attn_caches = []
            for input_id in range(self.n_input_slices):
                x = all_batch_inputs[batch_id, input_id]
                if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                    self.comm.recv_tensor(x, self.model_parallel_prev_dst_rank)
                if self.model_parallel_size > 1:
                    dist.broadcast(x, self.model_parallel_src_rank, group=self.model_parallel_group)
                attn_caches = [a.detach() for a in new_attn_caches]
                all_batch_attn_hiddens[batch_id, input_id] = list(chain.from_iterable(new_attn_caches))
                all_batch_attn_hiddens_detached[batch_id, input_id] = list(chain.from_iterable(attn_caches))
                y, new_attn_caches = self.local_transformer.forward(x, attn_caches)
                all_batch_outputs[batch_id, input_id] = y
                if (self.rank == self.model_parallel_dst_rank
                        and self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1):
                    self.comm.send_tensor(y, self.model_parallel_next_src_rank)
        return all_batch_outputs, all_batch_attn_hiddens, all_batch_attn_hiddens_detached

    def backward_step(self, sliced_grad_x, all_batch_inputs, all_batch_outputs, all_batch_attn_hiddens, all_batch_attn_hiddens_detached):
        for batch_id in reversed(range(self.n_batch_slices)):
            a = []
            da = []
            for input_id in reversed(range(self.n_input_slices)):
                dy = sliced_grad_x[batch_id, input_id]
                if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1:
                    if self.rank == self.model_parallel_dst_rank:
                        self.comm.recv_tensor(dy, self.model_parallel_next_src_rank)
                    if self.model_parallel_size > 1:
                        dist.broadcast(dy, self.model_parallel_dst_rank, group=self.model_parallel_group)
                x = all_batch_inputs[batch_id, input_id]
                y = all_batch_outputs[batch_id, input_id]
                outputs = [y] + a
                grad_outputs = [dy] + da
                inputs = self.all_parameters + [x] + all_batch_attn_hiddens_detached[batch_id, input_id]
                all_grads = torch.autograd.grad(outputs, inputs, grad_outputs)
                dw, dx, da = all_grads[:self.n_params], all_grads[self.n_params], list(all_grads[self.n_params + 1:])
                a = all_batch_attn_hiddens[batch_id, input_id]
                if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                    self.comm.send_tensor(dx, self.model_parallel_prev_dst_rank)
                for grad_w, w in zip(dw, self.all_parameters):
                    if w.grad is None:
                        w.grad = grad_w.detach()
                    else:
                        w.grad += grad_w

    def create_inputs(self):
        if self.rank != 0:
            input_x = self.config.create_inputs_empty()
        else:
            input_x = self.config.create_inputs()

        if self.mixed_precision:
            input_x = input_x.half()
        sliced_x = uniform_slice_batch_and_input(input_x, self.n_batch_slices, self.n_input_slices)
        for batch_id in range(self.n_batch_slices):
            for input_id in range(self.n_input_slices):
                sliced_x[batch_id, input_id].requires_grad_(True)
        return sliced_x

    def create_grad_inputs(self, all_batch_outputs):
        if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1:
            grad_x = self.config.create_inputs_empty()
            if self.mixed_precision:
                grad_x = grad_x.half()
            return uniform_slice_batch_and_input(grad_x, self.n_batch_slices, self.n_input_slices)

        concated_outputs = torch.cat([torch.cat(all_outputs.tolist(), dim=0) for all_outputs in all_batch_outputs], dim=1)
        if self.mixed_precision:
            # cast reductions to FP32
            concated_outputs = concated_outputs.float()
        loss = torch.mean(concated_outputs)
        # scale up the loss at the source for FP16, then de-scale when each
        # worker performs step() or correctness checks
        if self.mixed_precision:
            loss = loss.float().mul(LOSS_SCALE_FACTOR).half()
        sliced_grad_x = torch.autograd.grad(loss, all_batch_outputs.ravel())
        return np.array(sliced_grad_x, dtype='O').reshape(
            self.n_batch_slices, self.n_input_slices)

    def allreduce_params(self, reduce_after=True, no_scale=False, fp32_allreduce=False):
        # adopted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/distributed.py
        buckets = {}
        for param in self.local_transformer.parameters():
            if param.requires_grad and param.grad is not None:
                tp = (param.data.type())
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = _flatten_dense_tensors(grads)
            if fp32_allreduce:
                coalesced = coalesced.float()
            if not no_scale and not reduce_after:
                coalesced /= dist.get_world_size(group=self.data_parallel_group)
            dist.all_reduce(coalesced, group=self.data_parallel_group)
            torch.cuda.synchronize()
            if not no_scale and reduce_after:
                coalesced /= dist.get_world_size(group=self.data_parallel_group)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


class NCCLTransformerRunner(NCCLTransformer):
    def step(self):
        all_batch_inputs = self.create_inputs()
        all_batch_outputs, all_batch_attn_hiddens, all_batch_attn_hiddens_detached = self.forward_step(all_batch_inputs)

        sliced_grad_x = self.create_grad_inputs(all_batch_outputs)
        self.backward_step(
            sliced_grad_x,
            all_batch_inputs,
            all_batch_outputs,
            all_batch_attn_hiddens,
            all_batch_attn_hiddens_detached)
        del sliced_grad_x, all_batch_inputs, all_batch_outputs
        del all_batch_attn_hiddens, all_batch_attn_hiddens_detached

        if self.data_parallel_size > 1:
            # data parallel allreduce
            self.allreduce_params()

        self.local_transformer.update_weights()

    def run(self):
        all_step_times = []
        for _ in range(self.n_steps):
            start_time = time.time()
            self.step()
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            all_step_times.append(step_time)
            print("rank", self.rank, "step_time:", step_time, flush=True)
        if len(all_step_times) > WARM_UP_ROUNDS:
            print("rank", self.rank,
                  "step_time_mean:", np.mean(all_step_times[WARM_UP_ROUNDS:]),
                  "step_time_std:", np.std(all_step_times[WARM_UP_ROUNDS:]),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM')
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

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    config = TransformerConfig.from_predefined_model(
        args.model, n_devices=args.world_size, batch_size=args.batch_size)

    data_parallel_size = args.world_size // (args.model_parallel_size * args.pipeline_parallel_size)
    assert args.world_size == data_parallel_size * args.model_parallel_size * args.pipeline_parallel_size
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = NCCLTransformerRunner(
        config, args.n_batch_slices, args.n_input_slices, distributed_init_method, args.world_size,
        data_parallel_size, args.model_parallel_size, args.pipeline_parallel_size,
        args.rank, args.local_rank, args.n_steps, mixed_precision=args.mixed_precision,
        use_mpi=args.use_mpi
    )
    runner.run()


if __name__ == "__main__":
    main()
