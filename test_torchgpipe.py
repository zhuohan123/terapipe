import time

import torch
import torch.nn as nn
import itertools
import random
import numpy as np

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time


def benchmark(batch_size, input_dimension, hidden_size, pipeline_depth, n_modules, repeat_times):
    x = torch.rand(batch_size, input_dimension).to('cuda:0')
    modules = []
    for _ in range(n_modules):
        modules.append(nn.Sequential(
            nn.Linear(input_dimension, hidden_size, bias=False),
            nn.Linear(hidden_size, input_dimension, bias=False),
        ))

    model = nn.Sequential(*modules)


    partitions = torch.cuda.device_count()
    sample = x
    balance = balance_by_time(partitions, model, sample)
    model = GPipe(model, balance)
    # model = GPipe(model, balance, chunks=8)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat_times):
        model(x).sum().backward()
    torch.cuda.synchronize()
    return (time.time() - start) / repeat_times


r = benchmark(batch_size=512,
              input_dimension=4096,
              hidden_size=4096,
              pipeline_depth=8,
              n_modules=40,
              repeat_times=50)
print(r)
