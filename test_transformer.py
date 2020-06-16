import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mpu
import itertools

NEG_INF = -1e10


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mpu.model_parallel_cuda_manual_seed(seed)


class MultiheadLMAttentionWithCache(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()

        self.embed_dim = embed_dim

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

    def forward(self, x, cache=None):
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q *= self.scaling
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        attn_mask = x.new_full((tgt_len, tgt_len), NEG_INF).triu(1)
        src_len = tgt_len
        if cache is not None:
            cache_len = cache["k"].size()[1]
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            attn_mask = torch.cat([x.new_zeros(tgt_len, cache_len), attn_mask], dim=1)
            src_len += cache_len

        new_cache = {
            "k": k,
            "v": v,
        }

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        attn_weights += attn_mask[None, :, :]
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, new_cache


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads):
        super().__init__()
        self.attn_ln = nn.LayerNorm(embedding_dim)
        self.attn = MultiheadLMAttentionWithCache(embedding_dim, num_attention_heads)
        self.fc_ln = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x, attn_cache=None):
        seq_len, batch_size, _ = x.size()
        y = x
        x = self.attn_ln(x)
        x, new_attn_cache = self.attn(x, attn_cache)
        x = y + x
        y = x
        x = self.fc_ln(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = y + x
        return x, new_attn_cache


class SingleDeviceTransformer(nn.Module):
    def __init__(self, transformer_layers):
        super().__init__()
        self.layers = nn.ModuleList(transformer_layers)

    def forward(self, x, attn_caches=None):
        if attn_caches is None:
            attn_caches = [None] * len(self.layers)
        new_attn_caches = []
        for layer, attn_cache in zip(self.layers, attn_caches):
            x, new_attn_cache = layer(x, attn_cache)
            new_attn_caches.append(new_attn_cache)
        return x, new_attn_caches


class PipelinedTransformer(nn.Module):
    def __init__(self, nested_transformer_layers):
        super().__init__()
        self.n_devices = len(nested_transformer_layers)
        assert self.n_devices <= torch.cuda.device_count()
        self.single_device_transformers = nn.ModuleList([
            SingleDeviceTransformer(transformer_layers).to(torch.device(i))
            for i, transformer_layers in enumerate(nested_transformer_layers)
        ])

    def forward(self, segmented_xs):
        n_segments = len(segmented_xs)
        n_timesteps = len(segmented_xs) + self.n_devices - 1
        caches = [None] * self.n_devices
        for t in range(n_timesteps):
            for i in range(self.n_devices):
                if 0 <= t - i < n_segments:
                    x = segmented_xs[t - i].to(torch.device(i))
                    x, cache = self.single_device_transformers[i](x, caches[i])
                    caches[i] = cache
                    segmented_xs[t - i] = x

        return segmented_xs


def uniform_slice(x, seq_len, n_slices):
    slice_x = []
    start_index = 0
    for i in range(n_slices):
        seq_len_slice = seq_len // n_slices + int(i < seq_len % n_slices)
        slice_x.append(x[start_index:start_index + seq_len_slice])
        start_index += seq_len_slice
    assert start_index == seq_len
    return slice_x


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
    x_cache = torch.randn(cache_len, batch_size, embedding_dim).cuda(0)
    _, cache = single_device_transformer(x_cache)
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


def compare_single_device_and_pipeline():
    set_random_seed(0)
    time_testing_steps = 100
    batch_size = 1
    seq_len = 1024
    n_layers = 12
    embedding_dim = 768
    ffn_embedding_dim = embedding_dim * 4
    num_attention_heads = 12
    transformer_layers = [
        TransformerLayer(embedding_dim, ffn_embedding_dim, num_attention_heads)
        for _ in range(n_layers)
    ]
    x = torch.randn(seq_len, batch_size, embedding_dim).cuda(0)
    single_device_transformer = SingleDeviceTransformer(transformer_layers).cuda(0)
    torch.cuda.synchronize()
    start = time.time()
    for t in range(time_testing_steps):
        single_device_transformer.zero_grad()
        y, _ = single_device_transformer(x)
        loss = torch.mean(y)
        loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    print("single_device_time (per step):", duration / time_testing_steps)

    n_devices = torch.cuda.device_count()
    nested_layers = []
    layer_idx = 0
    for i in range(n_devices):
        n_layers_device = n_layers // n_devices + int(i < n_layers % n_devices)
        nested_layers.append(transformer_layers[layer_idx:layer_idx + n_layers_device])
        layer_idx += n_layers_device
    pipelined_transformer = PipelinedTransformer(nested_layers)

    slice_x = uniform_slice(x, seq_len, 16)
    torch.cuda.synchronize()
    start = time.time()
    for t in range(time_testing_steps):
        single_device_transformer.zero_grad()
        y_pipelined = pipelined_transformer(slice_x)
        loss = torch.mean(torch.cat(y_pipelined, dim=0))
        loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - start
    print("pipelined_time (per step):", duration / time_testing_steps)


if __name__ == "__main__":
    set_random_seed(0)
    grid_search_forward_time()
