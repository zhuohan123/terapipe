import json
import os
import time

from mpi4py import MPI
import numpy as np
import torch
import torch.distributed as dist

from transformer_models import MODEL_CONFIGS, BATCH_CONFIGS


def p2p_communication_latency(payload_size, group):
    buf_send = torch.ones(payload_size, dtype=torch.float16, device='cuda')
    buf_recv = torch.zeros(payload_size, dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()

    # 3-30 steps
    n_steps = int(np.clip(32 / np.log2(payload_size), 3, 30))
    warmup_steps = n_steps
    durations = []
    for _ in range(warmup_steps + n_steps):
        start = time.time()
        dist.broadcast(buf_send, 7, group=group)
        dist.broadcast(buf_recv, 8, group=group)
        torch.cuda.synchronize()
        durations.append((time.time() - start) / 2)  # round trip time
    durations = np.array(durations[warmup_steps:])
    mean = np.mean(durations)
    std = np.std(durations)    
    return float(mean), float(std)


def benchmark_p2p_communication(mpi_comm, rank, model_name, size_gap=8):
    _,  token_size, seqlen,  _ = MODEL_CONFIGS[model_name]
    max_tokens = seqlen * BATCH_CONFIGS[model_name]
    # we assume 2 8-GPU machines are used
    send_recv_group = torch.distributed.new_group([7, 8])
    durations_mean = []
    durations_std = []

    tensor_sizes = list(range(size_gap, max_tokens + 1, size_gap))
    for tokens in range(size_gap, max_tokens + 1, size_gap):
        payload_size = tokens * token_size
        mpi_comm.Barrier()
        if rank in (7, 8):
            mean, std = p2p_communication_latency(payload_size, send_recv_group)
            if rank == 7:
                print(f"\npayload size = {payload_size}, " 
                    f"communication finished in {mean*1000:.6f} ± {std*1000:.6f}ms",
                    flush=True)
                durations_mean.append(mean)
                durations_std.append(std)

    if rank == 7:
        results = {
            "model_name": model_name,
            "size_gap": size_gap,
            "tensor_sizes": tensor_sizes,
            "mean": durations_mean,
            "std": durations_std,
        }
        with open(f"{model_name}.communication_latency.json", "w") as f:
            json.dump(results, f)
    mpi_comm.Barrier()


def get_local_ip_address():
    import socket
    socket.gethostbyname(socket.gethostname())


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    if rank == 0:
        local_ip_address = get_local_ip_address()
    else:
        local_ip_address = None
    local_ip_address = MPI.COMM_WORLD.bcast(local_ip_address, root=0)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{local_ip_address}:14399',
        world_size=world_size,
        rank=rank)
    dist.all_reduce(torch.zeros(1).cuda())

    benchmark_p2p_communication(comm, rank, "gpt3-1b")
    benchmark_p2p_communication(comm, rank, "gpt3-13b")
    benchmark_p2p_communication(comm, rank, "gpt3-44b")
    benchmark_p2p_communication(comm, rank, "gpt3-175b")
