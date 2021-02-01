import argparse
import gc
import json
import os
import time

from transformer_models import TransformerConfig, MODEL_CONFIGS
from test_transformer_pipemegatron import NCCLTransformerRunner
from memory_model import peak_memory_per_gpu


def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--model-name', metavar='NAME', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))

    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--checkpoint-gradients', action='store_true', default=False)

    args = parser.parse_args()
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    model_name = args.model_name
    with open('dp_results.json', 'r') as f:
        data = json.load(f)

    experiment_results = []
    should_initialize_dist_group = True

    for experiment in data:
        if experiment['model_name'] != model_name:
            continue
        curr_time = time.time()
        batch_size = experiment['batch_size']
        model_parallel_size = experiment['model_parallel_size']
        pipeline_parallel_size = experiment['pipeline_length']
        data_parallel_size = experiment['data_parallel_size']
        batch_slices = experiment['batch_slices']
        input_slices = experiment['input_slices']
        assert data_parallel_size == world_size // (model_parallel_size * pipeline_parallel_size)
        assert world_size == data_parallel_size * model_parallel_size * pipeline_parallel_size
        result = {
            "model": args.model,
            "n_gpus": world_size,
            "batch_size": batch_size,
            "batch_slices": batch_slices,
            "input_slices": input_slices,
            "n_steps": args.n_steps,
            "mixed_precision": args.mixed_precision,
            "model_parallel_size": model_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "rank": rank,
            "data_parallel_size": data_parallel_size,
        }

        memory_usage = peak_memory_per_gpu(
            model_name, batch_size, world_size // 8, n_data_parallel_replicas=data_parallel_size, gpus_per_megatronlm_shard=model_parallel_size)
        MEMORY_LIMIT = 14.0
        if args.checkpoint_gradients:
            MEMORY_LIMIT *= 2
        if memory_usage > MEMORY_LIMIT:
            result["mean_time"] = "OOM"
            result["std_time"] = "OOM"
            experiment_results.append(result)
            continue

        config = TransformerConfig.from_predefined_model(model_name, n_devices=world_size, batch_size=batch_size)

        distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
        runner = NCCLTransformerRunner(
            config, batch_slices, input_slices, distributed_init_method, world_size,
            data_parallel_size, model_parallel_size, pipeline_parallel_size,
            rank, local_rank, mixed_precision=args.mixed_precision,
            use_mpi=True, init_process_group=should_initialize_dist_group,
            checkpoint_gradients=args.checkpoint_gradients
        )
        should_initialize_dist_group = False
        # GC the last experiment run to prevent memory leaks.
        gc.collect()
        if rank == 0:
            print(f"-------- Beginning run for model {args.model}; using {world_size} GPUs; batch size {batch_size}; "
                  f"batch slices {batch_slices}; input slices {input_slices}; steps {args.n_steps}; "
                  f"mixed precision {args.mixed_precision}; model parallel size {model_parallel_size}; "
                  f"pipeline parallel size {pipeline_parallel_size} --------", flush=True)
            print(f"-------- Experiment setup took {(time.time() - curr_time) * 1000} ms --------", flush=True)

        mean_time, std_time = runner.run(args.n_steps, verbose=args.verbose)
        result["mean_time"] = mean_time
        result["std_time"] = std_time
        experiment_results.append(result)
    if rank == 0:
        with open("dp_evaluation_results.json", "w") as f:
            json.dump(experiment_results, f)


if __name__ == "__main__":
    main()
