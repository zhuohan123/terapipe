import socket
import time

import torch
import ray
import py_nccl_sendrecv


@ray.remote
def pingpong(unique_id, rank):
    nccl = py_nccl_sendrecv.NCCL(unique_id, 2)
    with py_nccl_sendrecv.NCCLGroup():
        nccl.init_rank(0, rank)
    if rank == 0:
        tensor = torch.ones(2**30//4, dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(2**30//4, dtype=torch.float32).cuda()
    start = time.time()
    if rank == 0:
        nccl.send_tensor(tensor, 1)
    else:
        nccl.recv_tensor(tensor, 0)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Time used: {duration}s")


if __name__ == "__main__":
    ray.init(address="auto")
    my_addr = socket.gethostbyname(socket.gethostname())

    addr_list = [f"node:{my_addr}"]
    for k in ray.cluster_resources():
        if k.startswith('node') and k.split(':')[1] != my_addr:
            addr_list.append(k)
    unique_id = py_nccl_sendrecv.get_unique_id()
    tasks = []
    for i in range(2):
        tasks.append(pingpong.options(resources={addr_list[i]: 1}).remote(unique_id, rank=i))
    ray.get(tasks)
    ray.shutdown()
