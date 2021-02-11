import argparse
import gc
import json
import os
import time

from transformer_models import TransformerConfig, MODEL_CONFIGS
from test_transformer_terapipe import NCCLTransformerRunner
from memory_model import peak_memory_per_gpu


def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--model-name', metavar='NAME', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--index', type=int, required=True, help="The index of the setting we want to run.")
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
    data = [x for x in data if x['model_name'] == model_name]
    should_initialize_dist_group = True
    experiment = data[args.index]

    filename = f"dp_evaluation_results/dp_eval-{model_name}-{args.index}.json"
    if os.path.exists(filename):
        print(f"The experiment {model_name}-{args.index} already exists. Evaluation skipped.")
        return

    curr_time = time.time()
    batch_size = experiment['batch_size']
    model_parallel_size = experiment['model_parallel_size']
    pipeline_parallel_size = experiment['pipeline_length']
    data_parallel_size = experiment['data_parallel_size']
    batch_slices = experiment['batch_slices']
    input_slices = experiment['input_slices']
    if rank == 0:
        print(experiment, f"index={args.index}")
    assert data_parallel_size == world_size // (model_parallel_size * pipeline_parallel_size), (data_parallel_size, world_size, model_parallel_size * pipeline_parallel_size)
    assert world_size == data_parallel_size * model_parallel_size * pipeline_parallel_size
    result = {
        "model": model_name,
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
        raise ValueError("The setting will cause OOM. Skipped.")

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
        print(f"-------- Beginning run for model {model_name}; using {world_size} GPUs; batch size {batch_size}; "
                f"batch slices {batch_slices}; input slices {input_slices}; steps {args.n_steps}; "
                f"mixed precision {args.mixed_precision}; model parallel size {model_parallel_size}; "
                f"pipeline parallel size {pipeline_parallel_size} --------", flush=True)
        print(f"-------- Experiment setup took {(time.time() - curr_time) * 1000} ms --------", flush=True)

    mean_time, std_time = runner.run(args.n_steps, verbose=args.verbose)
    result["mean_time"] = mean_time
    result["std_time"] = std_time
    if rank == 0:
        with open(filename, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
