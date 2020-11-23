import os
import sys
import time
import torch

# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from py_nccl_sendrecv import NCCLGroup, NCCL, get_unique_id

@torch.jit.ignore
def get_nccl_communicator(device_id, rank, world_size, use_mpi=False):
    if use_mpi:
        try:
            from mpi4py import MPI
        except ImportError:
            print("To get NCCL communicator with MPI, please install 'mpi4py'.")
            use_mpi=False
    if use_mpi:
        assert MPI.COMM_WORLD.Get_rank() == rank
        if rank == 0:
            nccl_uniq_id = get_unique_id()
        else:
            nccl_uniq_id = None
        nccl_uniq_id = MPI.COMM_WORLD.bcast(nccl_uniq_id, root=0)
    else:
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
