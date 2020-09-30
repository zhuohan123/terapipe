#!/usr/bin/env python
import numpy as np
from tensor_p2p import Communicator
import threading
import queue
import asyncio
import argparse
import torch
from transformer_models import (
    TransformerConfig, TransformerLayer,
    SingleDeviceTransformer, PipelinedTransformer,
    ModelParallelTransformerLayer
)


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


class UCXTransformerRunner:
    def __init__(self, config, n_slices, my_address, my_port, prev_address,
                 prev_port, rank, world_size):
        self.config = config
        self.n_slices = n_slices
        self.my_address = my_address
        self.my_port = my_port
        self.prev_address = prev_address
        self.prev_port = prev_port
        self.comm = Communicator(self.ucx_main, my_address, my_port,
                                 prev_address, prev_port)
        self.loop = self.comm.loop
        self.q_in = queue.Queue()
        self.q_out = asyncio.Queue(loop=self.loop)
        self.rank = rank
        self.world_size = world_size
        self.layers = self.config.create_layers_gpu()

    async def ucx_main(self, prev_ep, next_ep):
        await asyncio.gather(
            self.recv_coroutine(prev_ep, next_ep),
            self.send_coroutine(prev_ep, next_ep)
        )

    async def recv_coroutine(self, prev_ep=None, next_ep=None):
        x = (self.config.create_inputs_empty()
             if prev_ep is not None else self.config.create_inputs())
        sliced_x = uniform_slice_x(x, self.n_slices)
        for s in sliced_x:
            if prev_ep is not None:
                await prev_ep.recv(s)
            self.q_in.put(s)

        grad_x = self.config.create_inputs_empty()
        sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)
        for s in reversed(sliced_grad_x):
            if next_ep is not None:
                await next_ep.recv(s)
            self.q_in.put(s)

    async def send_coroutine(self, prev_ep=None, next_ep=None):
        for i in range(self.n_slices):
            y = await self.q_out.get()
            if next_ep is None:
                print("forward i:", i, "y:", y, flush=True)
            else:
                await next_ep.send(y.detach())

        for i in reversed(range(self.n_slices)):
            y = await self.q_out.get()
            if prev_ep is not None:
                await prev_ep.send(y)
            else:
                print("back i:", i, "y:", y, flush=True)

    async def put_stuff_to_q_out(self, x):
        await self.q_out.put(x)

    def calc(self):
        # forward
        attn_caches = [None] * len(self.layers)
        for i in range(self.n_slices):
            x = self.q_in.get()
            new_attn_caches = []
            for layer, attn_cache in zip(self.layers, attn_caches):
                x, new_attn_cache = layer(x, attn_cache)
                new_attn_caches.append(new_attn_cache)
            attn_caches = new_attn_caches
            asyncio.run_coroutine_threadsafe(self.put_stuff_to_q_out(x), loop=self.loop)

        # backward
        for i in reversed(range(self.n_slices)):
            x = self.q_in.get()
            y = x + 1
            asyncio.run_coroutine_threadsafe(self.put_stuff_to_q_out(y), loop=self.loop)

    def run(self):
        t = threading.Thread(target=self.calc)
        t.start()
        self.comm.run()
        t.join()


def main():
    parser = argparse.ArgumentParser(description='UCX based transformer')
    parser.add_argument('--my-address', metavar='IP', type=str, default=None)
    parser.add_argument('--my-port', metavar='PORT', type=int, default=None)
    parser.add_argument('--prev-address', metavar='IP', type=str, default=None)
    parser.add_argument('--prev-port', metavar='PORT', type=int, default=None)
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)
    args = parser.parse_args()

    config = TransformerConfig(
        batch_size=1,
        seq_len=64,
        n_layers=24,
        embedding_dim=128,
        n_devices=args.world_size,
    )
    n_slices = 8

    runner = UCXTransformerRunner(
        config, n_slices, args.my_address, args.my_port, args.prev_address,
        args.prev_port, args.rank, args.world_size)
    runner.run()


if __name__ == "__main__":
    main()
