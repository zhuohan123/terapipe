import torch
import torch.distributed as dist
import mpu
import argparse
import numpy as np
import random
import time
import itertools

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


def measure_time(b, n, m, d, repeat_times=1000):
    """Test the running speed of Y=XAB where X is b*n, A is n*m and B is m*d"""
    mul_A = mpu.ColumnParallelLinear(n, m, bias=False, gather_output=False).cuda()
    mul_B = mpu.RowParallelLinear(m, d, bias=False, input_is_parallel=True).cuda()
    X = torch.randn(b, n).cuda()
    with torch.no_grad():
        # warm up runs
        for _ in range(2):
            Y = mul_B(mul_A(X))
        torch.cuda.synchronize()
        start_time = time.time()
        total_reduce_time = 0.0
        for _ in range(repeat_times):
            Y, reduce_time = mul_B(mul_A(X), return_reduce_time=True)
            total_reduce_time += reduce_time
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        print(b, n, m, d, total_time / repeat_times, total_reduce_time / repeat_times)



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
    choice_list = [32, 128, 512, 2048, 8192]
    for b, n, m, d in itertools.product(choice_list, choice_list, choice_list, choice_list):
        measure_time(b, n, m, d)


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
