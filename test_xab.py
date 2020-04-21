import torch
import torch.distributed as dist
import mpu
import argparse
import numpy as np
import random
import time

def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)


def measure_time(b, n, m, d, repeat_times=100):
    """Test the running speed of Y=XAB where X is b*n, A is n*m and B is m*d"""
    mul_A = mpu.ColumnParallelLinear(n, m, bias=False, gather_output=False).cuda()
    mul_B = mpu.RowParallelLinear(m, d, bias=False, input_is_parallel=True).cuda()
    X = torch.randn(b, n).cuda()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(repeat_times):
            Y = mul_B(mul_A(X))
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        print("total_time", total_time, force=True)



def distributed_main(process_idx, args):
    args.distributed_rank = process_idx
    dist.init_process_group(
        backend='nccl',
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank,
    )
    torch.cuda.set_device(args.distributed_rank)
    suppress_output(args.distributed_rank)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(args.distributed_world_size)
    set_random_seed(1337)
    measure_time(16, 16, 16, 16)


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
