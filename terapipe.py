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
from utils import set_random_seed, timeout, TimeoutError, uniform_slice
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, grid_slice_batch_and_sequence,
    ModelParallelTransformerLayer,
)
from memory_model import peak_memory_per_gpu
import checkpoint

LOSS_SCALE_FACTOR = 128.0


def initialize_distributed_env(distributed_init_method, rank, local_rank, world_size, model_parallel_size, pipeline_parallel_size):
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
        cache_outputs = []
        for layer_id in range(self.n_layers):
            k_input = cache[2 * layer_id]
            v_input = cache[2 * layer_id + 1]
            x, (k_output, v_output) = self.layers[layer_id](x, (k_input, v_input), cache_len)
            cache_outputs += [k_output, v_output]
        return x, cache_outputs

    def create_inputs_empty(self, batch_size, seq_len, device='cuda'):
        x = torch.empty((seq_len, batch_size, self.embedding_dim), device=device, dtype=torch.float16 if self.mixed_precision else torch.float32)
        return x


class PipelineSendOperator(torch.autograd.Function):
    """Send activations to the next pipeline stage"""

    @staticmethod
    def forward(ctx, x):
        rank = torch.distributed.get_rank()
        dst_rank = mpu.get_model_parallel_dst_rank()
        next_src_rank = mpu.get_model_parallel_next_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_pred_group()
        if rank == dst_rank and next_src_rank is not None:
            assert pipeline_group is not None
            dist.broadcast(x, dst_rank, group=pipeline_group)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        rank = torch.distributed.get_rank()
        dst_rank = mpu.get_model_parallel_dst_rank()
        next_src_rank = mpu.get_model_parallel_next_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_pred_group()
        model_parallel_group = mpu.get_model_parallel_group()
        if next_src_rank is not None:
            if rank == dst_rank:
                assert pipeline_group is not None
                dist.broadcast(grad_x, next_src_rank, group=pipeline_group)
            dist.broadcast(grad_x, dst_rank, group=model_parallel_group)
        return grad_x


class PipelineRecvOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rank = torch.distributed.get_rank()
        prev_dst_rank = mpu.get_model_parallel_prev_dst_rank()
        src_rank = mpu.get_model_parallel_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_succ_group()
        model_parallel_group = mpu.get_model_parallel_group()
        if prev_dst_rank is not None:
            if rank == src_rank:
                assert pipeline_group is not None
                dist.broadcast(x, prev_dst_rank, group=pipeline_group)
            dist.broadcast(x, src_rank, group=model_parallel_group)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        rank = torch.distributed.get_rank()
        prev_dst_rank = mpu.get_model_parallel_prev_dst_rank()
        src_rank = mpu.get_model_parallel_src_rank()
        pipeline_group = mpu.get_pipeline_parallel_succ_group()
        if prev_dst_rank is not None and rank == src_rank:
            assert pipeline_group is not None
            dist.broadcast(grad_x, src_rank, group=pipeline_group)
        return None


pipeline_send = PipelineSendOperator.apply
pipeline_recv = PipelineRecvOperator.apply


def terapipe_backward(outputs, grad_outputs, cache_inputs, cache_outputs):
    n_batch_slices, n_input_slices = outputs.size()
    for batch_id in reversed(range(n_batch_slices)):
        da = []
        for input_id in reversed(n_input_slices):
            y = outputs[batch_id, input_id]
            dy = grad_outputs[batch_id, input_id]
            if input_id < n_input_slices - 1:
                a = cache_outputs[batch_id, input_id]
            else:
                a = []
            torch.autograd.backward([y] + a, [dy] + da)
            da = [t.grad for t in cache_inputs[batch_id, input_id]]


class TeraPipeBackwardPassWithCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, cache_inputs, cache_outputs):
        # FIXME: may need a phony input x here
        ctx.outputs = outputs
        ctx.cache_inputs = cache_inputs
        ctx.cache_outputs = cache_outputs
        y = torch.cat([torch.cat(s.tolist(), dim=0) for s in outputs], dim=1)
        return y

    @staticmethod
    def backward(ctx, grad_x):
        del ctx.outputs
        del ctx.cache_inputs
        del ctx.cache_outputs
        # TODO: finish backward here


class TeraPipe:
    def __init__(self, layers, batch_size, seq_len, batch_slices, seq_slices):
        # Some assumptions about layers:
        #   forward maps from (x, cache, cache_len) to (x, new_cache, cache_outputs):
        #   x's shape is (sequence, batch, ...)
        #   caches' shape are (batch, sequence, ...)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_slices = batch_slices
        self.seq_slices = seq_slices

        self.layers = layers

    @property
    def n_batch_slices(self):
        return len(self.batch_slices)

    @property
    def n_input_slices(self):
        return len(self.seq_slices)

    def forward(self, inputs):
        cache_inputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        cache_outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')
        outputs = np.empty((self.n_batch_slices, self.n_input_slices), dtype='O')

        for batch_id in range(self.n_batch_slices):
            slice_batch_size = inputs[batch_id, 0].size(1)
            cache = self.layers.create_cache(slice_batch_size, self.seq_len)
            cache_len = 0
            for input_id in range(self.n_input_slices):
                x = inputs[batch_id, input_id]
                x = pipeline_recv(x)
                slice_seq_len = x.size(0)
                cache = [c.detach().requires_grad_() for c in cache]
                y, cache_output = self.layers(x, cache, cache_len)
                y = pipeline_send(y)
                cache_inputs[batch_id, input_id] = cache
                cache_outputs[batch_id, input_id] = cache_output
                cache = cache_output
                outputs[batch_id, input_id] = y
                cache_len += slice_seq_len
        return outputs, (cache_inputs, cache_outputs)
