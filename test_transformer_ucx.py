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


n_slices = 8
q_in = queue.Queue()
q_out = None
loop = None
config = TransformerConfig(
    batch_size=1,
    seq_len=1024,
    n_layers=72,
    embedding_dim=2048,
    placement_orders=[0, 3, 2, 1, 5, 6, 7, 4],
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


async def put_stuff_to_q_out(x):
    await q_out.put(x)


def calc():
    for _ in range(n_slices):
        x = q_in.get()
        y = x + 1
        asyncio.run_coroutine_threadsafe(put_stuff_to_q_out(y), loop=loop)


async def generate_data_and_put_to_q_in():
    x = config.create_inputs()
    sliced_x = uniform_slice_x(x, n_slices)
    for s in sliced_x:
        print("s:", s)
        q_in.put(s)


async def receive_data_and_put_to_q_in(prev_ep):
    x = config.create_inputs_empty()
    sliced_x = uniform_slice_x(x, n_slices)
    for s in sliced_x:
        await prev_ep.recv(s)
        q_in.put(s)


async def get_outputs_and_send_them_out(next_ep):
    for _ in range(n_slices):
        y = await q_out.get()
        await next_ep.send(y)


async def get_outputs_and_print_the_results():
    for i in range(n_slices):
        y = await q_out.get()
        print("i:", i, "y:", y)


async def ucx_main(prev_ep, next_ep):
    if prev_ep:
        prev_aw = receive_data_and_put_to_q_in(prev_ep)
    else:
        prev_aw = generate_data_and_put_to_q_in()

    if next_ep:
        next_aw = get_outputs_and_send_them_out(next_ep)
    else:
        next_aw = get_outputs_and_print_the_results()
    await asyncio.wait([prev_aw, next_aw])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UCX based transformer')
    parser.add_argument('--my-address', metavar='IP', type=str, default=None)
    parser.add_argument('--my-port', metavar='PORT', type=int, default=None)
    parser.add_argument('--prev-address', metavar='IP', type=str, default=None)
    parser.add_argument('--prev-port', metavar='PORT', type=int, default=None)
    args = parser.parse_args()
    comm = Communicator(ucx_main, args.my_address, args.my_port,
                        args.prev_address, args.prev_port)
    loop = comm.loop
    q_out = asyncio.Queue(loop=loop)
    t = threading.Thread(target=calc, args=config)
    t.start()
    comm.run()
    t.join()
