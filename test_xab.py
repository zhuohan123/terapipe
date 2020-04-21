import torch
import mpu
import argparse
import random


def distributed_main(process_idx, args):
    mpu.initialize_model_parallel(args.model_parallel_size)
    args.device_id = process_idx
    print(process_idx, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model parallel speed with program Y=XAB')
    parser.add_argument('model_parallel_size', metavar='N', type=int, nargs='+',
                        help='Model parallel size')
    args = parser.parse_args()
    port = random.randint(10000, 20000)
    args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    args.distributed_rank = None

    torch.multiprocessing.spawn(
        distributed_main,
        args=(args,),
        nprocs=args.model_parallel_size
    )
