#!/usr/bin/env python
import numpy as np
from tensor_p2p import Communicator
import threading
import queue
import asyncio
import argparse

n_bytes = 100
seq_len = 10
q_in = queue.Queue()
q_out = asyncio.Queue()
loop = None


async def put_stuff_to_q_out(x):
    await q_out.put(x)


def calc():
    for _ in range(seq_len):
        t = q_in.get()
        t = t + 1
        asyncio.run_coroutine_threadsafe(put_stuff_to_q_out(t), loop=loop)


async def generate_data_and_put_to_q_in():
    for _ in range(seq_len):
        msg = np.zeros(n_bytes, dtype='u1')  # create some data to send
        q_in.put(msg)


async def receive_data_and_put_to_q_in(prev_ep):
    for _ in range(seq_len):
        msg = np.zeros(n_bytes, dtype='u1')  # create some data to send
        msg_size = np.array([msg.nbytes], dtype=np.uint64)
        await prev_ep.recv(msg, msg_size)
        q_in.put(msg)


async def get_outputs_and_send_them_out(next_ep):
    for _ in range(seq_len):
        msg_new = await q_out.get()
        await next_ep.send(msg_new, n_bytes)


async def get_outputs_and_print_the_results():
    for i in range(seq_len):
        msg_new = await q_out.get()
        print("i:", i, "msg:", msg_new)


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
    t = threading.Thread(target=calc)
    t.start()
    comm = Communicator(ucx_main, args.my_address, args.my_port,
                        args.prev_address, args.prev_port)
    loop = comm.loop
    comm.run()
    t.join()
