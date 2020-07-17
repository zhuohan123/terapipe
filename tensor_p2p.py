import asyncio
import time
import sys
import ucp
import numpy as np


def run(aw):
    if sys.version_info >= (3, 7):
        return asyncio.run(aw)

    # Emulate asyncio.run() on older versions
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(aw)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class Communicator:
    def __init__(self, func, my_address=None, prev_address=None, port=13337):
        self.func = func
        self.my_address = my_address
        self.prev_address = prev_address
        self.port = port

    async def get_prev_ep(self):
        if self.prev_address is not None:
            print("has previous node")
            prev_ep = await ucp.create_endpoint(self.prev_address, self.port)
        else:
            print("does not have previous node")
            prev_ep = None
        return prev_ep

    async def call_func(self, next_ep=None):
        prev_ep = await self.get_prev_ep()
        await self.func(prev_ep, next_ep)
        if prev_ep:
            await prev_ep.close()

    async def start(self):
        lf = ucp.create_listener(self.call_back, self.port) if self.my_address else None
        if lf:
            print("has next node")
            while not lf.closed():
                await asyncio.sleep(0.1)
        else:
            print("does not have next node")
            await self.call_func()

    async def call_back(self, next_ep):
        await self.call_func(next_ep)
        await next_ep.close()

    def run(self):
        run(self.start())


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
