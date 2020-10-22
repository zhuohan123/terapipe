#!/usr/bin/env python
import time
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import mpu
import itertools
import sys
import argparse
from transformer_models import (
    TransformerConfig, TransformerLayer,
    SingleDeviceTransformer, PipelinedTransformer,
    ModelParallelTransformerLayer, save_layers_and_inputs,
    MODEL_CONFIGS, load_layers, load_grads, load_inputs
)


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def uniform_slice_x(x, n_slices):
    seq_len = x.size()[0]
    sliced_x = []
    start_index = 0
    for i in range(n_slices):
        seq_len_slice = seq_len // n_slices + int(i < seq_len % n_slices)
        sliced_x.append(x[start_index:start_index + seq_len_slice])
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


def grid_search_seq_length_forward_time():
    print("grid_search_seq_length_forward_time")
    n_layers = 24
    embedding_dim = 1024
    seq_len_list = range(1, 1025)
    ffn_embedding_dim = embedding_dim * 4
    num_attention_heads = embedding_dim // 64
    transformer_layers = [
        TransformerLayer(embedding_dim, ffn_embedding_dim, num_attention_heads)
        for _ in range(n_layers)
    ]
    batch_size = 1
    assert embedding_dim % 64 == 0
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    time_testing_steps = 10
    with torch.no_grad():
        for seq_len in itertools.product(seq_len_list):
            try:
                x = torch.randn(seq_len, batch_size, embedding_dim).cuda(0)
                # warm up
                for t in range(2):
                    single_device_transformer.zero_grad()
                    y, _ = single_device_transformer(x)
                torch.cuda.synchronize()
                start = time.time()
                for t in range(time_testing_steps):
                    single_device_transformer.zero_grad()
                    y, _ = single_device_transformer(x)
                torch.cuda.synchronize()
                duration = time.time() - start
                this_step_time = duration / time_testing_steps
            except:
                this_step_time = None
            print(seq_len, this_step_time, flush=True)



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


def single_device_correctness(config: TransformerConfig, checkpoint_path: str, n_testing_steps=10):
    set_random_seed(2)
    transformer_layers, x = config.create_layers_and_inputs()
    load_layers(transformer_layers, range(config.n_layers), checkpoint_path)
    x = x.cuda(0)
    x = load_inputs(checkpoint_path)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        loss.backward()
        all_ref_grads = load_grads(range(config.n_layers), checkpoint_path)
        for layer, ref_grads in zip(transformer_layers, all_ref_grads):
            for param, ref_grad in zip(layer.parameters(), ref_grads):
                assert param.grad.size() == ref_grad.size()
                print(torch.mean(torch.abs(param.grad - ref_grad)))

    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def check_correctness(config: TransformerConfig, checkpoint_path: str):
    transformer_layers, x = config.create_layers_and_inputs()
    transformer_layers = [layer.cuda(0) for layer in transformer_layers]
    x = x.cuda(0)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    single_device_transformer.zero_grad()
    y, _ = single_device_transformer(x)
    loss = torch.mean(y)
    loss.backward()
    torch.cuda.synchronize()
    grad_layers = []
    for layer in transformer_layers:
        grad_layer = []
        for param in layer.parameters():
            grad_layer.append(param.grad)
        grad_layers.append(grad_layer)
    save_layers_and_inputs(transformer_layers, grad_layers,
                           range(len(transformer_layers)), x, checkpoint_path)


def gpipe_time(config: TransformerConfig, n_testing_steps=10, profile=False):
    print("gpipe_time")
    print("preparing layers and inputs")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    print("warmup rounds")
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer([x])
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    print("start testing")
    start = time.time()
    with torch.autograd.profiler.profile(enabled=profile, use_cuda=True) as prof:
        for t in range(n_testing_steps):
            print("step", t)
            pipelined_transformer.zero_grad()
            y_pipelined = pipelined_transformer([x])
            loss = torch.mean(torch.cat(y_pipelined, dim=0))
            loss.backward()
        torch.cuda.synchronize()
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace("gpipe.gtrace")
        print(prof.key_averages().table())
    duration = time.time() - start
    return duration / n_testing_steps


def seqpipe_time(config: TransformerConfig, n_testing_steps=10, n_slices=8, profile=False):
    print("seqpipe_time")
    print("preparing layers and inputs")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    sliced_x = uniform_slice_x(x, n_slices)
    print("warmup rounds")
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    print("start testing")
    start = time.time()
    with torch.autograd.profiler.profile(enabled=profile, use_cuda=True) as prof:
        for t in range(n_testing_steps):
            print("step", t)
            pipelined_transformer.zero_grad()
            y_pipelined = pipelined_transformer(sliced_x)
            loss = torch.mean(torch.cat(y_pipelined, dim=0))
            loss.backward()
        torch.cuda.synchronize()
    duration = time.time() - start
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace(f"seqpipe_{n_slices}.gtrace")
        print(prof.key_averages().table())
    return duration / n_testing_steps


def seqpipe_correctness(config: TransformerConfig, checkpoint_path, n_testing_steps=10, n_slices=8):
    print("seqpipe_correctness")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    load_layers(transformer_layers, range(config.n_layers), checkpoint_path)
    x = load_inputs(checkpoint_path)
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    sliced_x = uniform_slice_x(x, n_slices)
    start = time.time()
    for t in range(n_testing_steps):
        print("step", t)
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
        all_ref_grads = load_grads(range(config.n_layers), checkpoint_path)
        for layer, ref_grads in zip(transformer_layers, all_ref_grads):
            for param, ref_grad in zip(layer.parameters(), ref_grads):
                assert param.grad.size() == ref_grad.size()
                print(torch.mean(torch.abs(param.grad - ref_grad.to(param.grad))))

    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def megatron_main(local_rank, distributed_init_method, world_size,
                  world_rank, config: TransformerConfig, n_testing_steps=10, profile=False):
    local_size = torch.cuda.device_count()
    distributed_world_size = world_size * local_size
    distributed_world_rank = world_rank * local_size + local_rank
    print("distributed_world_size", distributed_world_size, "distributed_world_rank", distributed_world_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=distributed_init_method,
        world_size=distributed_world_size,
        rank=distributed_world_rank,
    )
    torch.cuda.set_device(local_rank)
    suppress_output(distributed_world_rank)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(distributed_world_size)
    set_random_seed(0)
    mpu.model_parallel_cuda_manual_seed(0)
    print("init_models")
    transformer_layers = [
        ModelParallelTransformerLayer(
            config.embedding_dim,
            config.ffn_embedding_dim,
            config.num_attention_heads,
            device="cuda",
        )
        for _ in range(config.n_layers)
    ]
    transformer = SingleDeviceTransformer(transformer_layers)
    x = torch.randn(config.seq_len, config.batch_size, config.embedding_dim, device="cuda")
    print("warmup rounds")
    for t in range(2):
        transformer.zero_grad()
        y, _ = transformer(x)
        loss = torch.mean(y)
        loss.backward()
    torch.cuda.synchronize()
    print("start testing")
    start = time.time()

    all_step_times = []
    with torch.autograd.profiler.profile(enabled=profile, use_cuda=True) as prof:
        for t in range(n_testing_steps):
            start_time = time.time()
            transformer.zero_grad()
            y, _ = transformer(x)
            loss = torch.mean(y)
            loss.backward()
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            all_step_times.append(step_time)
            print("step_time:", step_time, flush=True)
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace("gpipe.gtrace")
        print(prof.key_averages().table())
    duration = time.time() - start
    print("megatron (s/it):", np.mean(all_step_times), np.std(all_step_times))


def megatron_spawn_tasks(world_size, world_rank, ip_address, port, config, n_testing_steps=10):
    distributed_init_method = f'tcp://{ip_address}:{port}'
    local_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        megatron_main,
        args=(distributed_init_method, world_size, world_rank, config, n_testing_steps),
        nprocs=local_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Different parallel methods for the Transformer')
    parser.add_argument('--type', metavar='NAME', type=str, default=None,
                        choices=["gridsearch", "gridseqlen", "single", "correctness", "single_correctness", "gpipe",
                                 "seqpipe", "seqpipe_correctness", "megatron"])
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-slices', metavar='N', type=int, default=8)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--checkpoint-path', metavar='PATH', type=str, default=None)
    args = parser.parse_args()
    set_random_seed(0)
    config = TransformerConfig(
        batch_size=1,
        seq_len=128,
        n_layers=24,
        embedding_dim=256,
        placement_orders=[0, 3, 2, 1, 5, 6, 7, 4],
        model_name=args.model,
    )
    if args.type == "gridsearch":
        grid_search_forward_time()
    elif args.type == "gridseqlen":
        grid_search_forward_time()
    elif args.type == "single":
        print("single_device (s/it):", single_device_time(config, n_testing_steps=args.n_steps))
    elif args.type == "correctness":
        assert args.checkpoint_path is not None
        check_correctness(config, args.checkpoint_path)
    elif args.type == "single_correctness":
        assert args.checkpoint_path is not None
        single_device_correctness(config, args.checkpoint_path, n_testing_steps=args.n_steps)
    elif args.type == "gpipe":
        print("gpipe (s/it):", gpipe_time(config, n_testing_steps=args.n_steps))
    elif args.type == "seqpipe":
        print("seqpipe (s/it):", seqpipe_time(config, n_testing_steps=args.n_steps, n_slices=args.n_slices))
    elif args.type == "seqpipe_correctness":
        assert args.checkpoint_path is not None
        seqpipe_correctness(config, args.checkpoint_path, n_testing_steps=args.n_steps, n_slices=args.n_slices)
    elif sys.argv[1] == "megatron":
        megatron_spawn_tasks(1, 0, 'localhost', random.randint(10000, 20000), config, n_testing_steps=args.n_steps)
