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
                activations[t - i].append(modules[i](activations[t - i][-1]))
    grad_activations = [torch.ones_like(a[-1]) for a in activations]
    # backward
    for t in reversed(range(n_timesteps)):
        for i in range(n_modules):
            if 0 <= t - i < n_chunks:
                all_grads = torch.autograd.grad(
                    outputs=activations[t - i][i + 1],
                    inputs=itertools.chain(
                        [activations[t - i][i]],
                        modules[i].parameters()
                    ),
                    grad_outputs=grad_activations[t - i]
                )
                grad_activations[t - i] = all_grads[0]
                for param, grad in zip(modules[i].parameters(), all_grads[1:]):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad


if __name__ == "__main__":
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    linear1 = nn.Linear(10, 10)
    linear2 = nn.Linear(10, 1)
    x = torch.randn(2, 10, requires_grad=True)
    # y = linear2(linear1(x)).sum()
    # y.backward()
    gpipe_forward_and_backward([linear1, linear2], [x[0], x[1]])
    for param in linear1.parameters():
        print(param, param.grad)
