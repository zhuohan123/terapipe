import torch
import torch.distributed as dist
import mpu
import argparse
import random


def distributed_main(process_idx, args):
    args.distributed_rank = process_idx
    dist.init_process_group(
        backend='nccl',
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank,
    )
    mpu.initialize_model_parallel(args.distributed_world_size)
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model parallel speed with program Y=XAB')
    parser.add_argument('distributed_world_size', metavar='N', type=int, help='Model parallel size')
    args = parser.parse_args()
    port = random.randint(10000, 20000)
    args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)

    torch.multiprocessing.spawn(
        distributed_main,
        args=(args,),
        nprocs=args.distributed_world_size
    )
