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
    activations = [[chunk] for chunk in chunks]
    # forward
    for t in range(n_timesteps):
        for i in range(n_modules):
            if 0 <= t - i < n_chunks:
                c = activations[t - i][-1].to('cuda:' + str(i), non_blocking=True)
                activations[t - i].append(modules[i](c))

    # backward
    grad_activations = [torch.ones_like(a[-1]) for a in activations]
    for t in reversed(range(n_timesteps)):
        for i in range(n_modules):
            if 0 <= t - i < n_chunks:
                for p in modules[i].parameters():
                    assert p.requires_grad
                if activations[t - i][i].requires_grad:
                    input_nodes = itertools.chain(modules[i].parameters(), [activations[t - i][i]])
                else:
                    input_nodes = modules[i].parameters()

                all_grads = torch.autograd.grad(
                    outputs=activations[t - i][i + 1],
                    inputs=input_nodes,
                    grad_outputs=grad_activations[t - i]
                )
                if activations[t - i][i].requires_grad:
                    grad_activations[t - i] = all_grads[-1]
                # zip() will drop the last item of all_grads
                for param, grad in zip(modules[i].parameters(), all_grads):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad


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
    r = benchmark(batch_size=128,
                  input_dimension=512,
                  hidden_size=1024,
                  pipeline_depth=8,
                  n_modules=20,
                  repeat_times=1)
    print(r)
