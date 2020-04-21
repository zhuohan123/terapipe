import torch
import torch.distributed as dist
import mpu
import argparse
import numpy as np
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)


def distributed_main(process_idx, args):
    args.distributed_rank = process_idx
    dist.init_process_group(
        backend='nccl',
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank,
    )
    torch.cuda.set_device(args.distributed_rank)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(args.distributed_world_size)
    set_random_seed(1337)
    x = torch.rand(5)
    print(x, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model parallel speed with program Y=XAB')
    parser.add_argument('distributed_world_size', metavar='N', type=int, help='Model parallel size')
    args = parser.parse_args()
    port = random.randint(10000, 20000)
    args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    assert torch.cuda.is_available()
    torch.multiprocessing.spawn(
        distributed_main,
        args=(args,),
        nprocs=args.distributed_world_size
    )
