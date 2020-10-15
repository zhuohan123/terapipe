#!/usr/bin/env python
import os
import socket

import ray


def get_ip_addresses():
    ray.init(address="auto")
    my_addr = socket.gethostbyname(socket.gethostname())
    addr_list = [my_addr]
    for k in ray.cluster_resources():
        if k.startswith('node'):
            ip = k.split(':')[1]
            if ip != my_addr:
                addr_list.append(ip)
    ray.shutdown()
    return addr_list


def mpirun(exe_abspath,
           host_list,
           map_by='ppr:1:node',
           exclude_tcp_devs=('lo', 'docker0'),
           env_vars=None):
    cmd = ["/home/ubuntu/anaconda3/bin/mpirun"]
    if exclude_tcp_devs:
        cmd.append(f"--mca btl_tcp_if_exclude {','.join(exclude_tcp_devs)}")
    cmd.append(f"--map-by {map_by}")
    cmd.append(f"-H {','.join(host_list)}")
    if env_vars:
        cmd.extend(f"-x {k}={v}" for k, v in env_vars.items())
    cmd.append(exe_abspath)
    os.system(' '.join(cmd))


if __name__ == "__main__":
    hosts = get_ip_addresses()
    print(hosts[:2])
    mpirun(os.path.abspath('round_trip_test'), hosts[:2], env_vars={'NCCL_DEBUG': 'WARN', 'NCCL_SOCKET_NTHREADS': 8, 'NCCL_NSOCKS_PERTHREAD': 8})
