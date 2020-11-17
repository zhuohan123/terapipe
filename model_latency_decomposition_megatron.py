import argparse
import os
import time

from apex import optimizers
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

import mpu
import nccl
from utils import set_random_seed
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, ModelParallelTransformerLayer,
)

WARM_UP_ROUNDS = 5
LOSS_SCALE_FACTOR = 128.0


class NCCLTransformerRunner:
    def __init__(self, config, distributed_init_method, world_size,
                 model_parallel_size, pipeline_parallel_size, rank, local_rank,
                 n_steps, mixed_precision=False, use_mpi=False):
        self.config = config
        self.n_layers = self.config.n_layers // self.config.n_devices
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
        self.model_parallel_size = model_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_group_rank = mpu.get_pipeline_parallel_group_rank()
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
        self.n_steps = n_steps
        self.n_layers = (config.n_layers // pipeline_parallel_size
                         + int(rank < config.n_layers % pipeline_parallel_size))
        self.layers = [
            ModelParallelTransformerLayer(
                config.embedding_dim,
                config.ffn_embedding_dim,
                config.num_attention_heads,
                device="cuda",
            )
            for _ in range(self.n_layers)
        ]

        self.all_parameters = []
        for layer in self.layers:
            self.all_parameters += list(layer.parameters())
        self.n_params = len(self.all_parameters)

        self.mixed_precision = mixed_precision

        if self.mixed_precision:
            for i in range(len(self.layers)):
                self.layers[i] = self.layers[i].half()

            self.all_parameters = []
            for layer in self.layers:
                self.all_parameters += list(layer.parameters())

            self.master_parameters = [p.clone().detach().float() for p in self.all_parameters]
            for p in self.master_parameters:
                p.requires_grad_()

        if self.mixed_precision:
            self.optimizer = optimizers.FusedSGD(self.master_parameters, lr=1e-10)
        else:
            self.optimizer = torch.optim.SGD(self.all_parameters, lr=1e-10)

    def step(self, seqlen):
        x = self.config.create_inputs()

        if self.mixed_precision:
            x = x.half()

        x = x[:seqlen]
        # forward
        attn_caches = [None] * len(self.layers)
        start_time = time.time()

        dist.broadcast(x, self.model_parallel_src_rank, group=self.model_parallel_group)
        x.requires_grad_()
        new_attn_caches_detached = []
        attn_hiddens = []
        attn_hiddens_detached = []
        for layer, attn_cache in zip(self.layers, attn_caches):
            x, new_attn_cache = layer(x, attn_cache)
            attn_hiddens += [v for k, v in new_attn_cache.items()]
            new_attn_cache_detached = {k: v.detach().requires_grad_() for k, v in new_attn_cache.items()}
            attn_hiddens_detached += [v for k, v in new_attn_cache_detached.items()]
            new_attn_caches_detached.append(new_attn_cache_detached)
        attn_caches = new_attn_caches_detached
        y = x
        py_forward_time = time.time() - start_time
        torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # backward
        start_time = time.time()
        if self.mixed_precision:
            for layer in self.layers:
                layer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # print("rank", self.rank, "calculate loss", flush=True)
        if self.mixed_precision:
            # cast reductions to FP32
            y = y.float()
        loss = torch.mean(y)

        # scale up the loss at the source for FP16, then de-scale when each
        # worker performs step() or correctness checks
        if self.mixed_precision:
            loss = loss.float() * LOSS_SCALE_FACTOR
            loss = loss.half()
        dy = torch.autograd.grad(loss, y)

        a = []
        da = []

        outputs = [y] + a
        grad_outputs = [dy[0]] + da
        inputs = self.all_parameters + [x] + attn_hiddens_detached
        all_grads = torch.autograd.grad(outputs, inputs, grad_outputs, allow_unused=True)
        dw = all_grads[:self.n_params]
        dx = all_grads[self.n_params]
        # 'dx' is to be sent out
        torch.cuda.synchronize()
        backward_time = time.time() - start_time

        start_time = time.time()
        da = list(all_grads[self.n_params + 1:])
        a = attn_hiddens
        for grad_w, w in zip(dw, self.all_parameters):
            if w.grad is None:
                w.grad = grad_w.detach()
            else:
                w.grad += grad_w
    
        # copy FP16 model gradients to FP32 master before comparing/optimizing
        if self.mixed_precision:
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                if master_param.grad is None:
                    master_param.grad = master_param.new(*master_param.size())
                master_param.grad.copy_(model_param.grad)

                # descale master weights
                master_param.grad.mul_(1. / LOSS_SCALE_FACTOR)

        self.optimizer.step()
        if self.mixed_precision:
            # copy master updated FP32 parameters back to FP16
            with torch.no_grad():
                for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                    model_param.copy_(master_param)

        torch.cuda.synchronize()
        update_time = time.time() - start_time
        return py_forward_time, forward_time, backward_time, update_time

    def run(self, seqlen):
        py_forward_durations = []
        forward_durations = []
        backward_durations = []
        update_durations = []
        for _ in range(self.n_steps):
            MPI.COMM_WORLD.Barrier()
            py_forward_time, forward_time, backward_time, update_time = self.step(seqlen)
            py_forward_durations.append(py_forward_time)
            forward_durations.append(forward_time)
            backward_durations.append(backward_time)
            update_durations.append(update_time)
        py_forward_durations = py_forward_durations[WARM_UP_ROUNDS:]
        forward_durations = forward_durations[WARM_UP_ROUNDS:]
        backward_durations = backward_durations[WARM_UP_ROUNDS:]
        update_durations = update_durations[WARM_UP_ROUNDS:]
        if self.rank == 0:
            print(f"seqlen {seqlen:5}, {np.mean(py_forward_durations):8.6f} {np.mean(forward_durations):8.6f}, "
                  f"{np.mean(backward_durations):8.6f}, {np.mean(update_durations):8.6f}",
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
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--use-mpi', action='store_true', default=False)

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    config = TransformerConfig.from_predefined_model(args.model, args.world_size)
    assert args.world_size == args.model_parallel_size * args.pipeline_parallel_size, \
        "Data parallel is not implemented yet"
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = NCCLTransformerRunner(
        config, distributed_init_method, args.world_size,
        args.model_parallel_size, args.pipeline_parallel_size,
        args.rank, args.local_rank, args.n_steps, mixed_precision=args.mixed_precision,
        use_mpi=args.use_mpi
    )
    for seqlen in [1, 2, 4, 8, 12, 16, 24, 32, 64] + list(range(128, 2049, 128)):
        runner.run(seqlen)


if __name__ == "__main__":
    main()
