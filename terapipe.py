import argparse
import os
import time
from itertools import chain, product
from types import FunctionType
from typing import Callable, List
import gc
import json

from apex import optimizers
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import sys

import mpu
from utils import set_random_seed, timeout, TimeoutError, uniform_slicemo
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, grid_slice_batch_and_sequence,
    ModelParallelTransformerLayer,
)
from memory_model import peak_memory_per_gpu
import checkpoint

LOSS_SCALE_FACTOR = 128.0


def initialize_distributed_environment(distributed_init_method, rank, local_rank, world_size, model_parallel_size, pipeline_parallel_size):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=distributed_init_method,
        world_size=world_size,
        rank=rank,
    )
    # A small all_reduce for warmup.
    dist.all_reduce(torch.zeros(1).cuda())
    mpu.initialize_model_parallel(model_parallel_size, pipeline_parallel_size)
    set_random_seed(0)
    mpu.model_parallel_cuda_manual_seed(0)


class TransformerLayers(nn.Module):
    def __init__(self, n_layers, embedding_dim, ffn_embedding_dim, num_attention_heads, mixed_precision=True):
        super().__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.mixed_precision = mixed_precision
        self.layers = []
        for _ in range(self.n_layers):
            layer = ModelParallelTransformerLayer(
                self.embedding_dim,
                self.ffn_embedding_dim,
                self.num_attention_heads,
                device="cuda",
            )
            self.layers.append(layer.half() if self.mixed_precision else layer)

    def create_cache(self, batch_size, seq_len):
        cache = []
        for layer_id in range(self.n_layers):
            k, v = self.layers[layer_id].attn.create_attn_cache(
                batch_size, seq_len, device='cuda',
                dtype=torch.float16 if self.mixed_precision else torch.float32)
            cache += [k, v]
        return cache

    def forward(self, x, cache, cache_len):
        new_cache = []
        cache_outputs = []
        for layer_id in range(self.n_layers):
            k_input = cache[2 * layer_id].detach()
            v_input = cache[2 * layer_id + 1].detach()
            new_cache += [k_input, v_input]
            x, (k_output, v_output) = self.layers[layer_id](x, (k_input, v_input), cache_len)
            cache_outputs += [k_output, v_output]
        return x, new_cache, cache_outputs


class TeraPipe:
    def __init__(self, layers, batch_size, seq_len, batch_slices, seq_slices, world_size, data_parallel_size,
                 model_parallel_size, pipeline_parallel_size, rank, local_rank):
        # Some assumptions about layers:
        #   forward maps from (x, cache, cache_len) to (x, new_cache, cache_outputs):
        #   x and cache's shape are (batch, sequence, ...):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_slices = batch_slices
        self.seq_slices = seq_slices
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_group_rank = mpu.get_pipeline_parallel_group_rank()
        self.data_parallel_group = mpu.get_data_parallel_group()
        self.model_parallel_group = mpu.get_model_parallel_group()
        self.pipeline_parallel_pred_group = mpu.get_pipeline_parallel_pred_group()
        self.pipeline_parallel_succ_group = mpu.get_pipeline_parallel_succ_group()
        self.model_parallel_src_rank = mpu.get_model_parallel_src_rank()
        self.model_parallel_dst_rank = mpu.get_model_parallel_dst_rank()
        self.model_parallel_next_src_rank = (
            self.model_parallel_src_rank + self.model_parallel_size
            if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1
            else None)
        self.model_parallel_prev_dst_rank = (
            self.model_parallel_dst_rank - self.model_parallel_size
            if self.pipeline_parallel_group_rank > 0 else None)

        self.layers = layers

    @property
    def n_batch_slices(self):
        return len(self.batch_slices)

    @property
    def n_input_slices(self):
        return len(self.seq_slices)

    def forward_step(self, all_inputs, initial_cache_len=0):
        all_cache_inputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        all_cache_outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        all_outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')

        for batch_id in range(self.n_batch_slices):
            slice_batch_size = all_inputs[batch_id, 0].size(1)
            cache = self.layers.create_cache(slice_batch_size, self.seq_len)
            cache_len = initial_cache_len
            for input_id in range(self.n_input_slices):
                x = all_inputs[batch_id, input_id]
                slice_seq_len = x.size(0)
                if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                    assert self.pipeline_parallel_succ_group is not None
                    dist.broadcast(x, self.model_parallel_prev_dst_rank, group=self.pipeline_parallel_succ_group)
                if self.model_parallel_size > 1:
                    dist.broadcast(x, self.model_parallel_src_rank, group=self.model_parallel_group)
                x, cache, cache_output = self.layers(x, cache, cache_len)
                all_cache_inputs[batch_id, input_id] = cache
                all_cache_outputs[batch_id, input_id] = cache_output
                all_outputs[batch_id, input_id] = x
                cache_len += slice_seq_len
                if (self.rank == self.model_parallel_dst_rank
                        and self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1):
                    assert self.pipeline_parallel_pred_group is not None
                    dist.broadcast(x, self.model_parallel_dst_rank, group=self.pipeline_parallel_pred_group)
        return all_outputs, all_cache_inputs, all_cache_outputs

    def backward_step(self, sliced_grad_x, all_inputs, all_outputs, all_cache_inputs, all_cache_outputs):
        for batch_id in reversed(range(self.n_batch_slices)):
            slice_batch_size = all_inputs[batch_id, 0].size(1)
            cache_grad = self.layers.create_cache(slice_batch_size, self.seq_len)
            cache_len = self.seq_len
            for input_id in reversed(range(self.n_input_slices)):
                dy = sliced_grad_x[batch_id, input_id]
                if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1:
                    if self.rank == self.model_parallel_dst_rank:
                        assert self.pipeline_parallel_pred_group is not None
                        dist.broadcast(dy, self.model_parallel_next_src_rank, group=self.pipeline_parallel_pred_group)
                    if self.model_parallel_size > 1:
                        dist.broadcast(dy, self.model_parallel_dst_rank, group=self.model_parallel_group)
                x = all_inputs[batch_id, input_id]
                y = all_outputs[batch_id, input_id]
                slice_seq_len = x.size(0)
                if input_id < self.n_input_slices - 1:
                    a = all_cache_outputs[batch_id, input_id]
                    da = [x[:, cache_len - slice_seq_len:cache_len] for x in cache_grad]
                else:
                    a = []
                    da = []
                outputs = [y] + a
                grad_outputs = [dy] + da
                parameters = list(self.layers.parameters())
                n_params = len(parameters)
                inputs = parameters + [x] + all_cache_inputs[batch_id, input_id]
                all_grads = torch.autograd.grad(outputs, inputs, grad_outputs)
                dw, dx, dcache = all_grads[:n_params], all_grads[n_params], all_grads[n_params + 1:]
                cache_len -= slice_seq_len
                if self.rank == self.model_parallel_src_rank and self.pipeline_parallel_group_rank > 0:
                    assert self.pipeline_parallel_succ_group is not None
                    dist.broadcast(dx, self.model_parallel_src_rank, group=self.pipeline_parallel_succ_group)
                if cache_len > 0:
                    for grad, update in zip(cache_grad, dcache):
                        grad[:, :cache_len] += update[:, :cache_len]
                for grad_w, w in zip(dw, parameters):
                    if w.grad is None:
                        w.grad = grad_w.detach()
                    else:
                        w.grad += grad_w

    def create_slices(self, x, requires_grad):
        # This function will be overrided by other classes. Do not delete it.
        return grid_slice_batch_and_sequence(x, self.batch_slices, self.seq_slices, requires_grad=requires_grad)



