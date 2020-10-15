import argparse
import os

from test_transformer import megatron_spawn_tasks, TransformerConfig, set_random_seed, MODEL_CONFIGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    args = parser.parse_args()
    set_random_seed(0)

    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        placement_orders=[0, 3, 2, 1, 5, 6, 7, 4],
        model_name=args.model,
    )

    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))

    megatron_spawn_tasks(world_size, world_rank, args.ip_address, args.port, config, n_testing_steps=args.n_steps)
