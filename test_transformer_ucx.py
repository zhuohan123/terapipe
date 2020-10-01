#!/usr/bin/env python
import numpy as np
from tensor_p2p import Communicator
import threading
import queue
import asyncio
import argparse
import traceback
import time
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
                 prev_port, rank, world_size, n_steps=10):
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
        self.all_paramters = []
        for layer in self.layers:
            self.all_paramters += list(layer.parameters())
        self.n_params = len(self.all_paramters)
        self.optimizer = torch.optim.SGD(self.all_paramters, lr=1e-10)
        self.n_steps = n_steps

    async def ucx_main(self, prev_ep, next_ep):
        await asyncio.gather(
            self.recv_coroutine(prev_ep, next_ep),
            self.send_coroutine(prev_ep, next_ep)
        )

    async def recv_coroutine(self, prev_ep=None, next_ep=None):
        for _ in range(self.n_steps):
            x = (self.config.create_inputs_empty()
                 if prev_ep is not None else self.config.create_inputs())
            sliced_x = uniform_slice_x(x, self.n_slices)
            for s in sliced_x:
                if prev_ep is not None:
                    await prev_ep.recv(s)
                self.q_in.put(s)

            grad_x = (self.config.create_inputs_empty()
                      if next_ep is not None else self.config.create_inputs())
            sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)
            for s in reversed(sliced_grad_x):
                if next_ep is not None:
                    await next_ep.recv(s)
                self.q_in.put(s)

    async def send_coroutine(self, prev_ep=None, next_ep=None):
        for _ in range(self.n_steps):
            for _ in range(self.n_slices):
                y = await self.q_out.get()
                if next_ep is not None:
                    await next_ep.send(y.detach())

            for _ in reversed(range(self.n_slices)):
                dx = await self.q_out.get()
                if prev_ep is not None:
                    await prev_ep.send(dx.detach())

    async def put_stuff_to_q_out(self, x):
        await self.q_out.put(x)

    def step(self):
        # forward
        attn_caches = [None] * len(self.layers)
        all_attn_hiddens = [[]]
        all_attn_hiddens_detached = [[]]
        all_inputs = []
        all_outputs = []
        start_time = time.time()
        for i in range(self.n_slices):
            x = self.q_in.get()
            x.requires_grad = True
            all_inputs.append(x)
            new_attn_caches_detached = []
            attn_hiddens = []
            attn_hiddens_detached = []
            for layer, attn_cache in zip(self.layers, attn_caches):
                x, new_attn_cache = layer(x, attn_cache)
                attn_hiddens += [v for k, v in new_attn_cache.items()]
                new_attn_cache_detached = {k: v.detach() for k, v in new_attn_cache.items()}
                attn_hiddens_detached += [v for k, v in new_attn_cache_detached.items()]
                new_attn_caches_detached.append(new_attn_cache_detached)
            attn_caches = new_attn_caches_detached
            all_attn_hiddens.append(attn_hiddens)
            all_attn_hiddens_detached.append(attn_hiddens_detached)
            all_outputs.append(x)
            asyncio.run_coroutine_threadsafe(self.put_stuff_to_q_out(x), loop=self.loop)
        print("rank", self.rank, "forward_time", time.time() - start_time)

        # backward
        start_time = time.time()
        self.optimizer.zero_grad()
        a = []
        da = []
        for i in reversed(range(self.n_slices)):
            dy = self.q_in.get()
            y = all_outputs[i]
            x = all_inputs[i]
            outputs = [y] + a
            grad_outputs = [dy] + da
            inputs = self.all_paramters + [x] + all_attn_hiddens_detached[i]
            # TODO: also calculate the grad to the weights, check why retain_graph is necessary.
            all_grads = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)
            dw = all_grads[:self.n_params]
            dx = all_grads[self.n_params]
            da = list(all_grads[self.n_params + 1:])
            a = all_attn_hiddens[i]
            asyncio.run_coroutine_threadsafe(self.put_stuff_to_q_out(dx), loop=self.loop)
            for grad_w, w in zip(dw, self.all_paramters):
                if w.grad is None:
                    w.grad = grad_w.detach()
                else:
                    w.grad += grad_w
        self.optimizer.step()
        print("rank", self.rank, "backward_time", time.time() - start_time)

    def calc(self):
        for _ in range(self.n_steps):
            start_time = time.time()
            try:
                self.step()
            except Exception as e:
                track = traceback.format_exc()
                print(f"rank = {self.rank}", track, flush=True)
                exit(1)
            step_time = time.time() - start_time
            print("rank", self.rank, "step_time:", step_time, flush=True)

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
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        n_devices=args.world_size,
    )
    n_slices = 8

    runner = UCXTransformerRunner(
        config, n_slices, args.my_address, args.my_port, args.prev_address,
        args.prev_port, args.rank, args.world_size)
    runner.run()


if __name__ == "__main__":
    main()
