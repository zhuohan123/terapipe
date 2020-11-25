import argparse
import json
import time

import numpy as np
import torch
import torch.distributed as dist

from transformer_models import TransformerConfig, uniform_slice_x, MODEL_CONFIGS
import nccl


class NCCLLatencyTest:
    def __init__(self, world_size, rank, local_rank, n_steps, n_warmup_steps=5):
        torch.cuda.set_device(local_rank)
        self.comm = nccl.get_nccl_communicator(local_rank, rank, world_size)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.n_steps = n_steps
        self.n_warmup_steps = n_warmup_steps

    def step(self, config, n_slices):
        input_x = config.create_inputs()
        sliced_x = uniform_slice_x(input_x, n_slices)
        grad_x = config.create_inputs()
        sliced_grad_x = uniform_slice_x(grad_x, n_slices)
        torch.cuda.synchronize()

        start = time.time()
        # forward
        for i in range(n_slices):
            x = sliced_x[i]
            if self.rank > 0:
                self.comm.recv_tensor(x, self.rank - 1)
            if self.rank < self.world_size - 1:
                self.comm.send_tensor(x, self.rank + 1)

        # backward
        for i in reversed(range(n_slices)):
            dx = sliced_grad_x[i]
            if self.rank < self.world_size - 1:
                self.comm.recv_tensor(dx, self.rank + 1)
            if self.rank > 0:
                self.comm.send_tensor(dx, self.rank - 1)
        torch.cuda.synchronize()
        return time.time() - start

    def run(self, config, n_slices):
        all_step_times = []
        for _ in range(self.n_warmup_steps + self.n_steps):
            step_time = self.step(config, n_slices)
            all_step_times.append(step_time)
        durations = all_step_times[self.n_warmup_steps:]
        print("rank", self.rank,
              "step_time_mean:", np.mean(durations),
              "step_time_std:", np.std(durations),
              flush=True)
        return {
            'step_time_mean': np.mean(durations),
            'step_time_std': np.std(durations),
        }


def nccl_pipelining_latency_benchmark_worker(model_names, rank, local_rank, world_size, n_steps):
    torch.cuda.set_device(local_rank)
    workload = NCCLLatencyTest(world_size, rank, local_rank, n_steps)
    results = {'rank': rank}
    for model_name in model_names:
        config = TransformerConfig.from_predefined_model(model_name)
        results[model_name] = {}
        for n_slices in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            print(f"n_slices={n_slices}", flush=True)
            results[model_name][n_slices] = workload.run(config, n_slices)
    return results


def nccl_allreduce_latency_benchmark_worker(model_names, rank, local_rank, world_size, ip_address, port, n_steps):
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip_address}:{port}',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    results = {'rank': rank}
    for model_name in model_names:
        config = TransformerConfig.from_predefined_model(model_name, n_devices=1)
        x = config.create_inputs(device='cuda')

        dist.barrier()
        torch.cuda.synchronize()

        n_warmup_steps = 5
        durations = []
        for _ in range(n_warmup_steps + n_steps):
            dist.barrier()
            start = time.time()
            # 2 * n_layers forward, 2 * n_layers backward
            for i in range(config.n_layers * 4):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            durations.append(time.time() - start)
        durations = durations[n_warmup_steps:]  # drop off warmup timing
        results[model_name] = {
            'step_time_mean': np.mean(durations),
            'step_time_std': np.std(durations),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCCL based transformer')
    parser.add_argument('--comm', metavar='NAME', type=str, choices=['pipeline', 'allreduce'],
                        help="communication methods")
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--local-rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)
    parser.add_argument('--models', metavar='CSV', type=str, default=None)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--ip-address', metavar='IP_ADDRESS', type=str, default=None)
    parser.add_argument('--port', metavar='PORT', type=int, default=16794)

    args = parser.parse_args()
    models = args.models.split(',')

    if args.comm == 'pipeline':
        result = nccl_pipelining_latency_benchmark_worker(
            models, args.rank, args.local_rank, args.world_size, args.n_steps)
    elif args.comm == 'allreduce':
        assert args.ip_address is not None
        result = nccl_allreduce_latency_benchmark_worker(
            models, args.rank, args.local_rank, args.world_size, args.ip_address, args.port, args.n_steps)
    else:
        raise ValueError('Unknown communication type.')
    with open(f'{args.world_size}_{args.rank}-{args.comm}-communication_latency.json', 'w') as f:
        json.dump(result, f, indent=2)
