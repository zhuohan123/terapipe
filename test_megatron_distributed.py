import random
import os

from test_transformer import megatron_main, TransformerConfig, set_random_seed

import argparse
import torch

parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('ip_address', type=str, help='the IP address of the head node')
parser.add_argument('-p', '--port', type=int, help='the port of the head node')


def megatron_spawn_tasks(world_size, world_rank, ip_address, port, config):
    assert torch.cuda.device_count() == 1
    distributed_init_method = f'tcp://{ip_address}:{port}'
    local_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        megatron_main,
        args=(distributed_init_method, world_size, world_rank, config),
        nprocs=local_size,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    set_random_seed(0)

    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        placement_orders=[0, 3, 2, 1, 5, 6, 7, 4],
    )

    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))

    megatron_spawn_tasks(world_size, world_rank, args.ip_address, args.port, config)
