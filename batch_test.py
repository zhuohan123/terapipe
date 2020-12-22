import os
import subprocess
import argparse

from itertools import product

from transformer_models import (
    MODEL_CONFIGS
)

def parse_comma_delimited_arg(arg, cast_fn):
    list_form = arg.split(',')
    return list(map(cast_fn, list_form))

def run_experiment(n_nodes, n_gpus_per_node, model_parallel_size, pipeline_parallel_size, 
                model, batch_size, n_batch_slices, n_input_slices, n_steps, mixed_precision):
    run_cmd = [
        "bash",
        "mpirun_pipemegatron.sh",
        str(n_nodes),
        str(n_gpus_per_node),
        str(model_parallel_size),
        str(pipeline_parallel_size),
        str(model),
        str(batch_size),
        str(n_batch_slices),
        str(n_input_slices),
        str(n_steps),
        (lambda x: "--mixed-precision" if x else '')(mixed_precision)
    ]

    try:
        # 2min timeout
        subprocess.run(run_cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        os.system("pgrep -fl python | awk '!/batch_test\.py/{print $1}' | xargs kill")
    except RuntimeError as e:
        print(e)
        os.system("pgrep -fl python | awk '!/batch_test\.py/{print $1}' | xargs kill")

"""
python batch_test.py --n-nodes 8 --n-gpus-per-node 8 --model-parallel-size 1,2,4,8 --pipeline-parallel-size 1,2,4,8 --model gpt3-1b --batch-size 1,4,16 --n-batch-slices 1,4,16 --n-input-slices 1,8,16,32,64 --n-steps 10 --mixed-precision
"""
def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM runner')
    parser.add_argument('--n-nodes', metavar='N', type=int, default=8)
    parser.add_argument('--n-gpus-per-node', metavar='N', type=int, default=8)
    parser.add_argument('--model-parallel-size', metavar='N', type=str, default='1')
    parser.add_argument('--pipeline-parallel-size', metavar='N', type=str, default='1')
    parser.add_argument('--model', metavar='NAME', type=str, default=None)
    parser.add_argument('--batch-size', metavar='N', type=str, default='1')
    parser.add_argument('--n-batch-slices', metavar='N', type=str, default='1')
    parser.add_argument('--n-input-slices', metavar='N', type=str, default='1')
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=False)

    args = parser.parse_args()

    args.model_parallel_size = parse_comma_delimited_arg(args.model_parallel_size, lambda x: int(x))
    args.pipeline_parallel_size = parse_comma_delimited_arg(args.pipeline_parallel_size, lambda x: int(x))
    args.batch_size = parse_comma_delimited_arg(args.batch_size, lambda x: int(x))
    args.n_batch_slices = parse_comma_delimited_arg(args.n_batch_slices, lambda x: int(x))
    args.n_input_slices = parse_comma_delimited_arg(args.n_input_slices, lambda x: int(x))
    args.model = parse_comma_delimited_arg(args.model, lambda x: str(x))

    for experiment in product(args.model_parallel_size, args.pipeline_parallel_size, args.batch_size, args.n_batch_slices, args.n_input_slices, args.model):
        model_parallel_size, pipeline_parallel_size, batch_size, n_batch_slices, n_input_slices, model = experiment
        run_experiment(args.n_nodes, args.n_gpus_per_node, model_parallel_size, pipeline_parallel_size,
            model, batch_size, n_batch_slices, n_input_slices, args.n_steps, args.mixed_precision)