import time

import torch
import torch.nn as nn
import itertools
import random
import numpy as np


def gpipe_forward_and_backward(modules, chunks):
    n_modules = len(modules)
    n_chunks = len(chunks)
    n_timesteps = n_modules + n_chunks - 1
    activations = list(chunks)
    # forward
    for t in range(n_timesteps):
        for i in range(n_modules):
            if 0 <= t - i < n_chunks:
                c = activations[t - i].to('cuda:' + str(i), non_blocking=True)
                activations[t - i] = modules[i](c)

    # backward
    activations[-1].sum().backward()


def run_gpipe(inputs, modules, pipeline_depth, repeat_times):
    n = torch.cuda.device_count()
    n_modules = len(modules)
    splits = [n_modules // n + (n_modules % n <= i) for i in range(n)]
    stages = []
    for i in range(n):
        s = n_modules // n + (i <= n_modules % n)
        seq = nn.Sequential(*modules[:s])
        seq.to('cuda:' + str(i))
        stages.append(seq)
        modules = modules[s:]
    assert inputs.size(0) % pipeline_depth == 0
    chunks = inputs.chunk(pipeline_depth, dim=0)
    start = time.time()
    torch.cuda.synchronize()
    for _ in range(repeat_times):
        gpipe_forward_and_backward(stages, chunks)
    torch.cuda.synchronize()
    return time.time() - start



def benchmark(batch_size, input_dimension, hidden_size, pipeline_depth, n_modules, repeat_times):
    x = torch.rand(batch_size, input_dimension).to('cuda:0')
    modules = []
    for _ in range(n_modules):
        modules.append(nn.Sequential(
            nn.Linear(input_dimension, hidden_size, bias=False),
            nn.Linear(hidden_size, input_dimension, bias=False),
        ))
    return run_gpipe(x, modules, pipeline_depth, repeat_times) / repeat_times


if __name__ == "__main__":
    for i in range(12):
        r = benchmark(batch_size=2048, input_dimension=4096, hidden_size=4096, pipeline_depth=2**i, n_modules=40, repeat_times=10)
        print(2**i, r)