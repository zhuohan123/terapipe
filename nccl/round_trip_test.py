import os
import sys
import socket
import time

import ray

import torch
from nccl_utils import get_nccl_communicator

def pingpong(rank):
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("start pingpong", flush=True)
    nccl = get_nccl_communicator(0, rank, 2)
    print("after init", flush=True)
    for _ in range(10):
        print("rank", rank, flush=True)
        if rank == 0:
            tensor = torch.ones(4 * 2**30//4, dtype=torch.float32, device='cuda')
            print("rank", rank, "tensor", tensor, flush=True)
        else:
            tensor = torch.zeros(4 * 2**30//4, dtype=torch.float32, device='cuda')
            print("rank", rank, "tensor", tensor, flush=True)
        torch.cuda.synchronize()
        start = time.time()
        if rank == 0:
            print("rank", rank, "send", flush=True)
            nccl.send_tensor(tensor, 1)
        else:
            print("rank", rank, "recv", flush=True)
            nccl.recv_tensor(tensor, 0)
        torch.cuda.synchronize()
        duration = time.time() - start
        print(f"Time used: {duration}s Bandwidth {4/duration}GB/s")
        assert abs((tensor.sum()/2**28 / 4).item() - 1) < 1e-9


if __name__ == "__main__":
    print("start", flush=True)
    pingpong(int(sys.argv[1]))
