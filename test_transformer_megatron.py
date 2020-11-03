import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import mpu
from utils import set_random_seed, suppress_output
from transformer_models import (
    TransformerConfig,
    SingleDeviceTransformer,
    ModelParallelTransformerLayer,
    MODEL_CONFIGS,
)


def megatron_main(local_rank, distributed_init_method, world_size,
                  world_rank, config: TransformerConfig, n_testing_steps=10, profile=False):
    local_size = torch.cuda.device_count()
    distributed_world_size = world_size * local_size
    distributed_world_rank = world_rank * local_size + local_rank
    print("distributed_world_size", distributed_world_size, "distributed_world_rank", distributed_world_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=distributed_init_method,
        world_size=distributed_world_size,
        rank=distributed_world_rank,
    )
    torch.cuda.set_device(local_rank)
    suppress_output(distributed_world_rank)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(distributed_world_size)
    set_random_seed(0)
    mpu.model_parallel_cuda_manual_seed(0)
    print("init_models")
    transformer_layers = [
        ModelParallelTransformerLayer(
            config.embedding_dim,
            config.ffn_embedding_dim,
            config.num_attention_heads,
            device="cuda",
        )
        for _ in range(config.n_layers)
    ]
    transformer = SingleDeviceTransformer(transformer_layers)
    x = torch.randn(config.seq_len, config.batch_size, config.embedding_dim, device="cuda")
    print("warmup rounds")
    for t in range(2):
        transformer.zero_grad()
        y, _ = transformer(x)
        loss = torch.mean(y)
        loss.backward()
    torch.cuda.synchronize()
    print("start testing")
    start = time.time()

    all_step_times = []
    with torch.autograd.profiler.profile(enabled=profile, use_cuda=True) as prof:
        for t in range(n_testing_steps):
            start_time = time.time()
            transformer.zero_grad()
            y, _ = transformer(x)
            loss = torch.mean(y)
            loss.backward()
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            all_step_times.append(step_time)
            print("step_time:", step_time, flush=True)
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace("gpipe.gtrace")
        print(prof.key_averages().table())
    duration = time.time() - start
    print("megatron (s/it):", np.mean(all_step_times), np.std(all_step_times))


def megatron_spawn_tasks(world_size, world_rank, ip_address, port, config, n_testing_steps=10):
    distributed_init_method = f'tcp://{ip_address}:{port}'
    local_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        megatron_main,
        args=(distributed_init_method, world_size, world_rank, config, n_testing_steps),
        nprocs=local_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Megatron-LM baseline')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    args = parser.parse_args()
    set_random_seed(0)

    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        placement_orders=[0, 3, 2, 1, 5, 6, 7, 4],
        model_name=args.model,
    )

    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    print("world_size", world_size)

    megatron_spawn_tasks(world_size, world_rank, args.ip_address, args.port, config, n_testing_steps=args.n_steps)
