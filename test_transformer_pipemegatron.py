import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import mpu
import nccl
from utils import set_random_seed
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, uniform_slice_x,
    ModelParallelTransformerLayer,
)
import terapipe_trainer

WARM_UP_ROUNDS = 5


class NCCLTransformerRunner:
    def __init__(self, config, n_slices, distributed_init_method, world_size,
                 model_parallel_size, pipeline_parallel_size, rank, local_rank,
                 n_steps, mixed_precision=False):
        self.config = config
        self.n_layers = self.config.n_layers // self.config.n_devices
        self.n_slices = n_slices
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
        self.comm = nccl.get_nccl_communicator(local_rank, rank, world_size)
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
        self.mixed_precision = mixed_precision
        self.trainer = terapipe_trainer.GPTTrainer(
            ModelParallelTransformerLayer,
            self.n_layers,
            config.embedding_dim,
            config.ffn_embedding_dim,
            config.num_attention_heads,
            mixed_precision)

    def step(self):
        if self.rank != 0:
            input_x = self.config.create_inputs_empty()
        else:
            input_x = self.config.create_inputs()
        if self.mixed_precision:
            input_x = input_x.half()
        sliced_x = uniform_slice_x(input_x, self.n_slices)

        # forward
        forward_co = self.trainer.forward_corotine(self.n_slices)
        start_time = time.time()
        for i in forward_co:
            x = sliced_x[i]
            if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                self.comm.recv_tensor(x, self.model_parallel_prev_dst_rank)
            dist.broadcast(x, self.model_parallel_src_rank, group=self.model_parallel_group)
            y = forward_co.send(x)
            if (self.rank == self.model_parallel_dst_rank
                    and self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1):
                self.comm.send_tensor(y, self.model_parallel_next_src_rank)
        print("rank", self.rank, "forward_time", time.time() - start_time, flush=True)

        # backward
        start_time = time.time()
        self.trainer.zero_grad()

        if self.pipeline_parallel_group_rank == self.pipeline_parallel_size - 1:
            print("rank", self.rank, "calculate loss", flush=True)
            _, grad_all_outputs = self.trainer.compute_loss()
            print("rank", self.rank, "finish calculating loss", flush=True)

        if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1:
            grad_x = self.config.create_inputs_empty()
            if self.mixed_precision:
                grad_x = grad_x.half()
            sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)

        backward_co = self.trainer.backward_coroutine(self.n_slices)
        for i in backward_co:
            if self.pipeline_parallel_group_rank == self.pipeline_parallel_size - 1:
                dy = grad_all_outputs[i]
            else:
                dy = sliced_grad_x[i]
                if self.rank == self.model_parallel_dst_rank:
                    self.comm.recv_tensor(dy, self.model_parallel_next_src_rank)
                dist.broadcast(dy, self.model_parallel_dst_rank, group=self.model_parallel_group)
            dx = backward_co.send(dy)
            if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                self.comm.send_tensor(dx, self.model_parallel_prev_dst_rank)

        self.trainer.update()

        print("rank", self.rank, "backward_time", time.time() - start_time, flush=True)
        torch.cuda.synchronize()

    def run(self):
        all_step_times = []
        for _ in range(self.n_steps):
            start_time = time.time()
            self.step()
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
    parser.add_argument('--n-slices', metavar='N', type=int, default=8)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--use-mpi', action='store_true', default=False)

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        n_devices=args.world_size,
        model_name=args.model,
    )
    assert args.world_size == args.model_parallel_size * args.pipeline_parallel_size, \
        "Data parallel is not implemented yet"
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = NCCLTransformerRunner(
        config, args.n_slices, distributed_init_method, args.world_size,
        args.model_parallel_size, args.pipeline_parallel_size,
        args.rank, args.local_rank, args.n_steps, mixed_precision=args.mixed_precision
    )
    runner.run()


if __name__ == "__main__":
    main()
