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

    warmup_steps = 5
    n_steps = 10
    n_accum = max(1, int(2**30 / payload_size))
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
    single_direction_durations = np.array(single_direction_durations[warmup_steps:]) / n_accum * 1000
    mean = np.mean(single_direction_durations)
    std = np.std(single_direction_durations)
    if rank == 0:
        print(f"Send finished in {mean:.6f} ± {std:.6f}ms")
    elif rank == 2:
        print(f"Recv finished in {mean:.6f} ± {std:.6f}ms")

    comm.Barrier()

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
        print(f"Send 0 finished in {mean:.6f} ± {std:.6f}ms")
    elif rank == 1:
        print(f"Recv 1 finished in {mean:.6f} ± {std:.6f}ms")
    elif rank == 2:
        print(f"Recv 2 finished in {mean:.6f} ± {std:.6f}ms")
    elif rank == 3:
        print(f"Send 3 finished in {mean:.6f} ± {std:.6f}ms")


if __name__ == "__main__":
    payload_sizes = [1] + list(range(12288, 12288, 12288 * 2 * 2048 + 1))

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    torch.cuda.set_device(local_rank)
    nccl_comm = nccl.get_nccl_communicator(local_rank, rank, world_size)
    for payload_size in payload_sizes:
        print(f"\nPayload size = {payload_size}\n")
        benchmark(payload_size, comm, nccl_comm, rank)
