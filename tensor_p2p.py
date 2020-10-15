import asyncio
import time
import sys
import ucp
import numpy as np


class Communicator:
    def __init__(self, func, my_address=None, my_port=13337, prev_address=None, prev_port=13338):
        self.func = func
        self.my_address = my_address
        self.my_port = my_port
        self.prev_address = prev_address
        self.prev_port = prev_port
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def get_prev_ep(self):
        if self.prev_address is not None:
            await asyncio.sleep(2.0)
            print("has previous node", flush=True)
            prev_ep = await ucp.create_endpoint(self.prev_address, self.prev_port)
        else:
            print("does not have previous node", flush=True)
            prev_ep = None
        return prev_ep

    async def call_func(self, next_ep=None):
        prev_ep = await self.get_prev_ep()
        await self.func(prev_ep, next_ep)
        if prev_ep:
            await prev_ep.close()

    async def start(self):
        self.lf = ucp.create_listener(self.call_back, self.my_port) if self.my_address else None
        if self.lf:
            print("has next node", flush=True)
            while not self.lf.closed():
                await asyncio.sleep(0.1)
        else:
            print("does not have next node", flush=True)
            await self.call_func()

    async def call_back(self, next_ep):
        await self.call_func(next_ep)
        await next_ep.close()
        self.lf.close()

    def run(self):
        try:
            return self.loop.run_until_complete(self.start())
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)


n_bytes = 100


async def _client(prev_ep, next_ep):
    assert next_ep is None
    msg = np.zeros(n_bytes, dtype='u1')  # create some data to send
    msg_size = np.array([msg.nbytes], dtype=np.uint64)
    await prev_ep.recv(msg, msg_size)
    print("client:", msg)


async def _server(prev_ep, next_ep):
    assert prev_ep is None
    msg = np.ones(n_bytes, dtype='u1')  # create some data to send
    print("server:", msg)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)
    await next_ep.send(msg, msg_size)


if __name__ == "__main__":
    if sys.argv[1] == "c":
        comm = Communicator(_client, None, "127.0.0.1")
    else:
        assert sys.argv[1] == "s"
        comm = Communicator(_server, "127.0.0.1", None)
    comm.run()
