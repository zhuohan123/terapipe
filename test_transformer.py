import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mpu
import itertools
import sys
from transformer_models import (
    TransformerConfig, TransformerLayer,
    SingleDeviceTransformer, PipelinedTransformer
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mpu.model_parallel_cuda_manual_seed(seed)


def uniform_slice_x(x, n_slices):
    seq_len = x.size()[0]
    sliced_x = []
    start_index = 0
    for i in range(n_slices):
        seq_len_slice = seq_len // n_slices + int(i < seq_len % n_slices)
        sliced_x.append(x[start_index:start_index + seq_len_slice].detach())
        start_index += seq_len_slice
    assert start_index == seq_len
    return sliced_x


def uniform_slice_layers(transformer_layers, n_devices=None):
    n_layers = len(transformer_layers)
    n_devices = n_devices if n_devices else torch.cuda.device_count()
    nested_layers = []
    layer_idx = 0
    for i in range(n_devices):
        n_layers_device = n_layers // n_devices + int(i < n_layers % n_devices)
        nested_layers.append(transformer_layers[layer_idx:layer_idx + n_layers_device])
        layer_idx += n_layers_device
    assert layer_idx == n_layers
    return nested_layers


def unit_test_forward_time(
    time_testing_steps=10,
    batch_size=1,
    n_layers=12,
    embedding_dim=768,
    seq_len=128,
    cache_len=896,
):
    ffn_embedding_dim = embedding_dim * 4
    assert embedding_dim % 64 == 0
    num_attention_heads = embedding_dim // 64
    transformer_layers = [
        TransformerLayer(embedding_dim, ffn_embedding_dim, num_attention_heads)
        for _ in range(n_layers)
    ]
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    if cache_len > 0:
        x_cache = torch.randn(cache_len, batch_size, embedding_dim).cuda(0)
        _, cache = single_device_transformer(x_cache)
    else:
        cache = None
    x = torch.randn(seq_len, batch_size, embedding_dim).cuda(0)
    # warm up
    for t in range(2):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x, cache)
    torch.cuda.synchronize()
    start = time.time()
    for t in range(time_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x, cache)
    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / time_testing_steps


def grid_search_forward_time():
    print("grid_search_forward_time")
    n_layers_list = [3, 6, 12, 24]
    embedding_dim_list = [768, 1024, 1536, 2048, 3072, 4096]
    seq_len_list = [128, 256, 512, 1024]
    cache_times_list = [0, 1, 2, 4, 8]
    for n_layers, embedding_dim, seq_len, cache_times in itertools.product(
        n_layers_list, embedding_dim_list, seq_len_list, cache_times_list
    ):
        try:
            this_step_time = unit_test_forward_time(
                n_layers=n_layers,
                embedding_dim=embedding_dim,
                seq_len=seq_len,
                cache_len=cache_times * seq_len,
            )
        except:
            this_step_time = None
        print(n_layers, embedding_dim, seq_len, cache_times, this_step_time)


def single_device_time(config: TransformerConfig, n_testing_steps=10):
    transformer_layers, x = config.create_layers_and_inputs()
    x = x.cuda(0)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    for t in range(2):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        loss.backward()
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def gpipe_time(config: TransformerConfig, n_testing_steps=10):
    transformer_layers, x = config.create_layers_and_inputs()
    x = x.cuda(0)
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers)
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer([x])
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer([x])
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def seqpipe_time(config: TransformerConfig, n_testing_steps=10, n_slices=8):
    transformer_layers, x = config.create_layers_and_inputs()
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers)
    sliced_x = uniform_slice_x(x.cuda(0), config.seq_len, n_slices)
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


if __name__ == "__main__":
    set_random_seed(0)
    # grid_search_forward_time()
    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=72,
        embedding_dim=2048,
    )
    assert len(sys.argv) > 1
    if sys.argv[1] == "gridsearch":
        grid_search_forward_time()
    elif sys.argv[1] == "single":
        print("single_device (s/it):", single_device_time(config))
    elif sys.argv[1] == "gpipe":
        print("gpipe (s/it):", gpipe_time(config))
    elif sys.argv[1] == "seqpipe":
        print("seqpipe (s/it):", seqpipe_time(config))
