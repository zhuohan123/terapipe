import os
import sys
import time

from mpi4py import MPI
import numpy as np
import torch
import nccl


def benchmark(payload_size, comm, nccl_comm, rank):
    buf_send = torch.ones(payload_size // 4, dtype=torch.float32, device='cuda')
    buf_recv = torch.zeros(payload_size // 4, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    comm.Barrier()

    warmup_steps = 10
    n_steps = 30
    n_accum = min(max(1, int(1024*1024*1024 / payload_size)), 10)
    single_direction_durations = []
    for _ in range(warmup_steps + n_steps):
        start = time.time()
        for a in range(n_accum):
            if rank == 0:
                nccl_comm.send_tensor(buf_send, 2)
            elif rank == 2:
                nccl_comm.recv_tensor(buf_recv, 0)
        torch.cuda.synchronize()
        single_direction_durations.append(time.time() - start)
    single_direction_durations = np.array(single_direction_durations[warmup_steps:]) / n_accum
    mean = np.mean(single_direction_durations)
    std = np.std(single_direction_durations)
    if rank == 0:
        print(f"Send finished in {mean*1000:.6f} ± {std*1000:.6f}ms", flush=True)
    elif rank == 2:
        print(f"Recv finished in {mean*1000:.6f} ± {std*1000:.6f}ms", flush=True)

    comm.Barrier()
    return float(mean), float(std)

    double_direction_durations = []
    for _ in range(warmup_steps + n_steps):
        start = time.time()
        for a in range(n_accum):
            if rank == 0:
                nccl_comm.send_tensor(buf_send, 2)
            elif rank == 1:
                nccl_comm.recv_tensor(buf_send, 3)
            elif rank == 2:
                nccl_comm.recv_tensor(buf_recv, 0)
            elif rank == 3:
                nccl_comm.send_tensor(buf_send, 1)
        torch.cuda.synchronize()
        double_direction_durations.append(time.time() - start)
    double_direction_durations = np.array(double_direction_durations[warmup_steps:]) / n_accum * 1000
    mean = np.mean(double_direction_durations)
    std = np.std(double_direction_durations)
    if rank == 0:
        print(f"Send 0 finished in {mean:.6f} ± {std:.6f}ms", flush=True)
    elif rank == 1:
        print(f"Recv 1 finished in {mean:.6f} ± {std:.6f}ms", flush=True)
    elif rank == 2:
        print(f"Recv 2 finished in {mean:.6f} ± {std:.6f}ms", flush=True)
    elif rank == 3:
        print(f"Send 3 finished in {mean:.6f} ± {std:.6f}ms", flush=True)


if __name__ == "__main__":
    payload_sizes = np.arange(8, 2048 + 1, 8) * 12288 * 2 #+ list(range(12288, 12288 * 2 * 2048 + 1, 12288))

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    torch.cuda.set_device(local_rank)
    nccl_comm = nccl.get_nccl_communicator(local_rank, rank, world_size)
    results = {}
    for payload_size in payload_sizes:
        if rank == 0:
            print(f"\nPayload size = {payload_size}\n")
        results[int(payload_size) // (12288 * 2)] = benchmark(payload_size, comm, nccl_comm, rank)
    import json
    if rank == 0:
        with open("communication_latency.json", "w") as f:
            json.dump(results, f)
