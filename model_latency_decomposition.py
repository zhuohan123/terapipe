import json
import time

import numpy as np
import torch
from torch import optim
import tqdm

from transformer_models import TransformerConfig, SingleDeviceTransformer, NEG_INF


def single_device_model_latency(config: TransformerConfig, presets, n_timing_steps=10, n_warmup_steps=5):
    transformer_layers = config.create_layers_gpu('cuda:0')
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    # disable optimizer from actually optimizing
    optimizer = optim.SGD(single_device_transformer.parameters(), lr=1e-11)

    x, y = config.create_pseudo_inputs_outputs('cuda:0', inputs_requires_grad=True)

    # warmup before timing
    for _ in range(n_warmup_steps):
        optimizer.zero_grad()
        pred_y, _ = single_device_transformer(x, None)
        loss = config.compute_pseudo_loss(pred_y, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    timing_results = []

    # measure the time
    for p in tqdm.tqdm(presets):
        zero_grad_durations = []
        forward_durations = []
        backward_durations = []
        optimizer_update_durations = []
        seq_len, att_cache_len = p['seq_len'], p['att_cache_len']
        x_i, y_i = x[:seq_len], y[:seq_len]
        if att_cache_len <= 0:
            cache_i = None
        else:
            cache_i = config.create_pseudo_attention_cache(att_cache_len, 'cuda:0', requires_grad=True)

        for _ in range(n_timing_steps):
            start = time.time()

            optimizer.zero_grad()
            torch.cuda.synchronize()
            ts_zero_grad = time.time()

            pred_y, _ = single_device_transformer(x_i, cache_i)
            loss = config.compute_pseudo_loss(pred_y, y_i)
            torch.cuda.synchronize()
            ts_forward = time.time()

            loss.backward()
            torch.cuda.synchronize()
            ts_backward = time.time()

            optimizer.step()
            torch.cuda.synchronize()
            ts_optimizer_update = time.time()

            zero_grad_durations.append(ts_zero_grad - start)
            forward_durations.append(ts_forward - ts_zero_grad)
            backward_durations.append(ts_backward - ts_forward)
            optimizer_update_durations.append(ts_optimizer_update - ts_backward)

        zero_grad_durations = np.array(zero_grad_durations)
        forward_durations = np.array(forward_durations)
        backward_durations = np.array(backward_durations)
        optimizer_update_durations = np.array(optimizer_update_durations)
        timing_results.append({
            'zero_grad_mean': zero_grad_durations.mean(),
            'zero_grad_std': zero_grad_durations.std(),
            'forward_mean': forward_durations.mean(),
            'forward_std': forward_durations.std(),
            'backward_mean': backward_durations.mean(),
            'backward_std': backward_durations.std(),
            'optimizer_update_mean': optimizer_update_durations.mean(),
            'optimizer_update_std': optimizer_update_durations.std(),
        })
    return timing_results


def attention_cache_latency(config, presets, n_timing_steps=10, n_warmup_steps=5):
    timing_results = []
    # measure the time
    for p in tqdm.tqdm(presets):
        forward_script_durations = []
        backward_script_durations = []
        forward_stream_durations = []
        backward_stream_durations = []

        seq_len, att_cache_len = p['seq_len'], p['att_cache_len']
        bsz = config.batch_size
        num_heads = config.num_attention_heads

        x = torch.randn(
            seq_len, bsz, config.embedding_dim, device='cuda:0').requires_grad_(True)
        q, k, v = torch.rand(bsz * num_heads, seq_len, config.attention_heads_dim*3,
                             device='cuda:0').requires_grad_(True).chunk(3, dim=-1)
        attn_mask = x.new_full((seq_len, seq_len), NEG_INF).triu(1)
        src_len = seq_len + att_cache_len
        if att_cache_len > 0:
            cache_k = torch.randn(
                bsz * num_heads, att_cache_len, config.attention_heads_dim, device='cuda:0').requires_grad_(True)
            cache_v = torch.randn(
                bsz * num_heads, att_cache_len, config.attention_heads_dim, device='cuda:0').requires_grad_(True)

        torch.cuda.synchronize()

        def forward_step():
            if att_cache_len > 0:
                ak = torch.cat([cache_k, k], dim=1)
                av = torch.cat([cache_v, v], dim=1)
                a_attn_mask = torch.cat([x.new_zeros(seq_len, att_cache_len), attn_mask], dim=1)
            else:
                ak, av, a_attn_mask = k, v, attn_mask

            attn_weights = torch.bmm(q, ak.transpose(1, 2))
            assert attn_weights.size() == (bsz * num_heads, seq_len, src_len)
            attn_weights += a_attn_mask[None, :, :]
            attn_probs = attn_weights.softmax(dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn = torch.bmm(attn_probs, av)
            return attn

        accum = 10

        for _ in range(n_warmup_steps + n_timing_steps):
            start = time.time()
            # start timing here
            for n in range(accum):
                attn = forward_step()
            forward_script_durations.append((time.time() - start) / accum)
            torch.cuda.synchronize()
            forward_stream_durations.append((time.time() - start) / accum)

            start = time.time()
            for n in range(accum):
                attn.backward(1.0, retain_graph=True)
            backward_script_durations.append((time.time() - start) / accum)
            torch.cuda.synchronize()
            backward_stream_durations.append((time.time() - start) / accum)

        forward_script_durations = np.array(forward_script_durations[n_warmup_steps:])
        forward_stream_durations = np.array(forward_stream_durations[n_warmup_steps:])
        backward_script_durations = np.array(backward_script_durations[n_warmup_steps:])
        backward_stream_durations = np.array(backward_stream_durations[n_warmup_steps:])

        timing_results.append({
            'forward_script_mean': forward_script_durations.mean(),
            'forward_script_std': forward_script_durations.std(),
            'forward_stream_mean': forward_stream_durations.mean(),
            'forward_stream_std': forward_stream_durations.std(),
            'backward_script_mean': backward_script_durations.mean(),
            'backward_script_std': backward_script_durations.std(),
            'backward_stream_mean': backward_stream_durations.mean(),
            'backward_stream_std': backward_stream_durations.std(),
        })


def model_latency_decomposition(model_names):
    presets = []
    for att_cache_len in range(0, 2049, 32):
        for seq_len in list(range(1, 32)) + list(range(32, 2049, 32)):
            presets.append({'seq_len': seq_len, 'att_cache_len': att_cache_len})
    for model_name in model_names:
        model_config = TransformerConfig.from_predefined_model(model_name, n_devices=1)
        results = single_device_model_latency(model_config, presets)
        with open(f'{model_name}-latency.json', 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    model_names = ('gpt3-13b-single-device', 'gpt2-3hm-single-device')
    model_latency_decomposition(model_names)
