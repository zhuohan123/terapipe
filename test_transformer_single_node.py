import time
import torch
import torch.nn as nn
import itertools
import traceback
import argparse
from transformer_models import (
    TransformerConfig, TransformerLayer,
    SingleDeviceTransformer, PipelinedTransformer, save_layers_and_inputs,
    MODEL_CONFIGS, load_layers, load_grads, load_inputs,
    uniform_slice_x, uniform_slice_layers,
)
from utils import set_random_seed
from apex import amp


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
        for seq_len in seq_len_list:
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
                track = traceback.format_exc()
                print(track, flush=True)
                this_step_time = None
            print(seq_len, this_step_time, flush=True)



def single_device_time(config: TransformerConfig, n_testing_steps=10, mixed_precision=False):
    transformer_layers, x = config.create_layers_and_inputs()
    x = x.cuda(0)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    if mixed_precision:
        single_device_transformer = amp.initialize(single_device_transformer, opt_level='O2', loss_scale=128.0)
    for t in range(2):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def single_device_correctness(config: TransformerConfig, checkpoint_path: str, n_testing_steps=10, mixed_precision=False):
    set_random_seed(2)
    transformer_layers, x = config.create_layers_and_inputs()
    load_layers(transformer_layers, range(config.n_layers), checkpoint_path)
    x = x.cuda(0)
    x = load_inputs(checkpoint_path)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    if mixed_precision:
        single_device_transformer = amp.initialize(single_device_transformer, opt_level='O2', loss_scale=128.0)
    torch.cuda.synchronize()
    start = time.time()
    for t in range(n_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        all_ref_grads = load_grads(range(config.n_layers), checkpoint_path)
        for layer, ref_grads in zip(transformer_layers, all_ref_grads):
            for param, ref_grad in zip(layer.parameters(), ref_grads):
                assert param.grad.size() == ref_grad.size()
                print(torch.mean(torch.abs(param.grad - ref_grad)))

    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


def check_correctness(config: TransformerConfig, checkpoint_path: str, mixed_precision=False):
    transformer_layers, x, target = config.create_layers_and_inputs_with_embedding()
    transformer_layers = [layer.cuda(0) for layer in transformer_layers]
    x = x.cuda(0)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    if mixed_precision:
        single_device_transformer = amp.initialize(single_device_transformer, opt_level='O2', loss_scale=128.0)
    single_device_transformer.zero_grad()
    y, _ = single_device_transformer(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, target)

    if mixed_precision:
        with amp.scale_loss(loss, []) as scaled_loss:
            scaled_loss.backward()
    else:
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


def gpipe_time(config: TransformerConfig, n_testing_steps=10, profile=False, mixed_precision=False):
    print("gpipe_time")
    print("preparing layers and inputs")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    if mixed_precision:
        pipelined_transformer = amp.initialize(pipelined_transformer, opt_level='O2', loss_scale=128.0)
    print("warmup rounds")
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer([x])
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
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
            if mixed_precision:
                with amp.scale_loss(loss, []) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        torch.cuda.synchronize()
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace("gpipe.gtrace")
        print(prof.key_averages().table())
    duration = time.time() - start
    return duration / n_testing_steps


def seqpipe_time(config: TransformerConfig, n_testing_steps=10, n_slices=8, profile=False, mixed_precision=False):
    print("seqpipe_time")
    print("preparing layers and inputs")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    if mixed_precision:
        pipelined_transformer = amp.initialize(pipelined_transformer, opt_level='O2', loss_scale=128.0)
    sliced_x = uniform_slice_x(x, n_slices)
    print("warmup rounds")
    for t in range(2):
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
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
            if mixed_precision:
                with amp.scale_loss(loss, []) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        torch.cuda.synchronize()
    duration = time.time() - start
    if profile:
        print("writing trace to disk")
        prof.export_chrome_trace(f"seqpipe_{n_slices}.gtrace")
        print(prof.key_averages().table())
    return duration / n_testing_steps


def seqpipe_correctness(config: TransformerConfig, checkpoint_path, n_testing_steps=10, n_slices=8, mixed_precision=False):
    print("seqpipe_correctness")
    transformer_layers, x = config.create_layers_and_inputs_on_gpu()
    load_layers(transformer_layers, range(config.n_layers), checkpoint_path)
    x = load_inputs(checkpoint_path)
    nested_layers = uniform_slice_layers(transformer_layers)
    pipelined_transformer = PipelinedTransformer(nested_layers, config)
    if mixed_precision:
        pipelined_transformer = amp.initialize(pipelined_transformer, opt_level='O2', loss_scale=128.0)
    sliced_x = uniform_slice_x(x, n_slices)
    start = time.time()
    for t in range(n_testing_steps):
        print("step", t)
        pipelined_transformer.zero_grad()
        y_pipelined = pipelined_transformer(sliced_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        if mixed_precision:
            with amp.scale_loss(loss, []) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        all_ref_grads = load_grads(range(config.n_layers), checkpoint_path)
        for layer, ref_grads in zip(transformer_layers, all_ref_grads):
            for param, ref_grad in zip(layer.parameters(), ref_grads):
                assert param.grad.size() == ref_grad.size()
                print(torch.mean(torch.abs(param.grad - ref_grad.to(param.grad))))

    torch.cuda.synchronize()
    duration = time.time() - start
    return duration / n_testing_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Different parallel methods for the Transformer')
    parser.add_argument('--type', metavar='NAME', type=str, default=None,
                        choices=["gridsearch", "gridseqlen", "single", "correctness", "single_correctness", "gpipe",
                                 "seqpipe", "seqpipe_correctness", "megatron"])
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-slices', metavar='N', type=int, default=8)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
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
        grid_search_seq_length_forward_time()
    elif args.type == "single":
        print("single_device (s/it):", single_device_time(config, n_testing_steps=args.n_steps, mixed_precision=args.mixed_precision))
    elif args.type == "correctness":
        assert args.checkpoint_path is not None
        check_correctness(config, args.checkpoint_path, mixed_precision=args.mixed_precision)
    elif args.type == "single_correctness":
        assert args.checkpoint_path is not None
        single_device_correctness(config, args.checkpoint_path, n_testing_steps=args.n_steps, mixed_precision=args.mixed_precision)
    elif args.type == "gpipe":
        print("gpipe (s/it):", gpipe_time(config, n_testing_steps=args.n_steps, mixed_precision=args.mixed_precision))
    elif args.type == "seqpipe":
        print("seqpipe (s/it):", seqpipe_time(config, n_testing_steps=args.n_steps, n_slices=args.n_slices, mixed_precision=args.mixed_precision))
    elif args.type == "seqpipe_correctness":
        assert args.checkpoint_path is not None
        seqpipe_correctness(config, args.checkpoint_path, n_testing_steps=args.n_steps, n_slices=args.n_slices, mixed_precision=args.mixed_precision)
