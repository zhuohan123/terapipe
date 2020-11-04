import os
import sys
import time

# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from py_nccl_sendrecv import NCCLGroup, NCCL, get_unique_id


def get_nccl_communicator(device_id, rank, world_size):
    # NOTE: we must save id file to a shared filesystem like AWS efs!
    id_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "nccl_uniq_id")

    if rank == 0:
        nccl_uniq_id = get_unique_id()
        with open(id_file, "wb") as f:
            f.write(nccl_uniq_id)
    else:
        time.sleep(3)
        with open(id_file, "rb") as f:
            nccl_uniq_id = f.read()

    comm = NCCL(nccl_uniq_id, world_size)
    comm.init_rank(device_id, rank)
    return comm
