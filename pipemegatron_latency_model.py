import argparse
import json
import os
import time

import numpy as np
import torch
import tqdm

from test_transformer_pipemegatron import NCCLTransformer
from transformer_models import TransformerConfig, MODEL_CONFIGS
from latency_model import SCAN_GRID, STEP_GAP


class TerapipeLatencyModel(NCCLTransformer):
    def step(self, attn_cache_len):
        all_inputs = self.create_inputs()
        torch.cuda.synchronize()

        # forward
        start = time.time()
        all_outputs, all_cache_inputs, all_cache_outputs = self.forward_step(all_inputs, attn_cache_len)
        py_forward_time = time.time() - start
        torch.cuda.synchronize()
        forward_time = time.time() - start

        sliced_grad_x = self.prepare_grad_x(all_outputs)

        # backward
        start = time.time()
        self.backward_step(sliced_grad_x, all_inputs, all_outputs, all_cache_inputs, all_cache_outputs)
        py_backward_time = time.time() - start
        torch.cuda.synchronize()
        backward_time = time.time() - start

        del sliced_grad_x
        del all_inputs
        del all_outputs
        del all_cache_inputs
        del all_cache_outputs

        # if self.data_parallel_size > 1:
        #     # data parallel allreduce
        #     self.allreduce_params()

        start = time.time()
        self.update_weights()
        torch.cuda.synchronize()
        update_time = time.time() - start

        return py_forward_time, forward_time, py_backward_time, backward_time, update_time

    def run(self, seqlen, attn_cache_len, n_steps, warmup_steps):
        # overwrite the original seqlen
        self.config.seq_len = seqlen

        py_forward_durations = []
        forward_durations = []
        py_backward_durations = []
        backward_durations = []
        update_durations = []

        for _ in range(n_steps + warmup_steps):
            py_forward_time, forward_time, py_backward_time, backward_time, update_time = \
                self.step(attn_cache_len)
            py_forward_durations.append(py_forward_time)
            forward_durations.append(forward_time)
            py_backward_durations.append(py_backward_time)
            backward_durations.append(backward_time)
            update_durations.append(update_time)

        py_forward_durations = py_forward_durations[warmup_steps:]
        forward_durations = forward_durations[warmup_steps:]
        py_backward_durations = py_backward_durations[warmup_steps:]
        backward_durations = backward_durations[warmup_steps:]
        update_durations = update_durations[warmup_steps:]

        return {
            'py_forward_mean': np.mean(py_forward_durations),
            'forward_mean': np.mean(forward_durations),
            'py_backward_mean': np.mean(py_backward_durations),
            'backward_mean': np.mean(backward_durations),
            'update_mean': np.mean(update_durations),
            'py_forward_std': np.std(py_forward_durations),
            'forward_std': np.std(forward_durations),
            'py_backward_std': np.std(py_backward_durations),
            'backward_std': np.std(backward_durations),
            'update_std': np.std(update_durations),
        }


def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('--port', type=int, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--local-rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)

    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))

    parser.add_argument('--model-parallel-size', metavar='N', type=int, default=8)
    parser.add_argument('--batch-size', metavar='N', type=int, default=1)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--warmup-steps', metavar='N', type=int, default=5)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--use-mpi', action='store_true', default=False)

    # These are fixed during the measurement.
    parser.add_argument('--n-batch-slices', metavar='N', type=int, default=1)
    parser.add_argument('--n-input-slices', metavar='N', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', metavar='N', type=int, default=1)

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    config = TransformerConfig.from_predefined_model(
        args.model, n_devices=args.world_size, batch_size=args.batch_size)
    # We just test a single layer, since all of them are identical.
    config.n_layers = 1

    data_parallel_size = args.world_size // (args.model_parallel_size * args.pipeline_parallel_size)
    assert args.world_size == data_parallel_size * args.model_parallel_size * args.pipeline_parallel_size
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = TerapipeLatencyModel(
        config, args.n_batch_slices, args.n_input_slices, distributed_init_method, args.world_size,
        data_parallel_size, args.model_parallel_size, args.pipeline_parallel_size,
        args.rank, args.local_rank, mixed_precision=args.mixed_precision,
        use_mpi=args.use_mpi
    )
    full_seqlen = config.seq_len
    results = []
    for attn_cache_len in tqdm.tqdm(range(full_seqlen // SCAN_GRID[1], full_seqlen + 1, full_seqlen // SCAN_GRID[1])):
        for seqlen in range(full_seqlen // SCAN_GRID[0], full_seqlen + 1, full_seqlen // SCAN_GRID[0]):
            r = runner.run(seqlen, attn_cache_len, args.n_steps, args.warmup_steps)
            results.append(r)
    if args.rank == 0:
        with open(f'{args.model}.latency_model.attn_cache_len.json', 'w') as f:
            json.dump(results, f)

    results = []
    for seqlen in tqdm.tqdm(range(STEP_GAP, full_seqlen + 1, STEP_GAP)):
         r = runner.run(seqlen, 0, args.n_steps, args.warmup_steps)
         results.append(r)
    if args.rank == 0:
        with open(f'{args.model}.latency_model.seqlen.json', 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
