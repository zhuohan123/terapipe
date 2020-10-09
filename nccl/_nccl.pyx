from libcpp cimport bool as c_bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string as c_string, memset, memcpy
from libcpp.unordered_map cimport unordered_map

from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t, intptr_t
from _nccl cimport *

from cython.operator cimport dereference, postincrement

import torch

# TODO(Siyuan): python error handling for NCCL

def _get_native_cuda_stream(stream=None, tensor=None):
    if stream is None:
        if tensor is None:
            raise ValueError("stream and tensor cannot both be None")
        if isinstance(tensor, torch.Tensor):
            assert tensor.device.type == 'cuda'
            stream = torch.cuda.current_stream(tensor.device)
        else:
            raise ValueError("Unknown tensor")
    if isinstance(stream, torch.cuda.streams.Stream):
        return stream.cuda_stream
    else:
        raise ValueError("Unknown stream")


cdef ncclDataType_t _type_from_string(s):
    if s == 'float32':
        return ncclFloat32
    elif s == 'float16':
        return ncclFloat16
    else:
        raise ValueError("Unsupported type string")


def _probe_tensor(tensor):
    if isinstance(torch.Tensor):
        assert tensor.device.type == 'cuda'
        device_id = tensor.device.index
        buff = a.data_ptr()
        count = a.numel()
        dtype = str(a.dtype).lstrip('torch.')
    else:
        raise ValueError("unknown tensor")
    return device_id, buff, count, dtype


def get_unique_id():
    cdef ncclUniqueId uid
    ncclGetUniqueId(&uid)
    return uid.internal


cdef class NCCLGroup:
    def __enter__(self):
        ncclGroupStart()

    def __exit__(self, exc_type, exc_value, traceback):
        ncclGroupEnd()


cdef class NCCL:
    cdef:
        ncclUniqueId _unique_id
        int nranks
        # map from rank to ncclComm_t
        unordered_map[int, ncclComm_t] comm_map

    def __cinit__(self, bytes unique_id, int nranks):
        cdef c_string id_string = unique_id
        memcpy(self._unique_id.internal, id_string.c_str(), id_string.length())
        self.nranks = nranks
        # map from device_id to rank
        self.dev_map = {}

    def init_rank(self, int device_id, int rank):
        cdef:
            int original_dev_id
            ncclComm_t comm
        if device_id in self.dev_map:
            raise KeyError("The device has been initialized.")
        cudaGetDevice(&original_dev_id)
        cudaSetDevice(device_id)
        ncclCommInitRank(&comm, self.nranks, self._unique_id, rank)
        self.dev_map[device_id] = rank
        self.comm_map[rank] = comm
        cudaSetDevice(original_dev_id)

    def send_tensor(self, tensor, int peer, stream=None):
        cdef:
            int cuda_stream
            ncclComm_t comm
            intptr_t sendbuff
            size_t count
        cuda_stream = _get_native_cuda_stream(stream, tensor)
        device_id, sendbuff, count, dtype = _probe_tensor(tensor)
        comm = self.comm_map[self.dev_map[device_id]]
        with nogil:
            ncclSend(<const void*>sendbuff, count, _type_from_string(dtype), peer, comm, cuda_stream)

    def recv_tensor(self, tensor, int peer, stream=None):
        cdef:
            int cuda_stream
            ncclComm_t comm
            intptr_t recvbuff
            size_t count
        cuda_stream = _get_native_cuda_stream(stream, tensor)
        device_id, recvbuff, count, dtype = _probe_tensor(tensor)
        comm = self.comm_map[self.dev_map[device_id]]
        with nogil:
            ncclRecv(<void*>recvbuff, count, _type_from_string(dtype), peer, comm, cuda_stream)

    def __dealloc__(self):
        cdef unordered_map[int, ncclComm_t].iterator it = comm_map.begin()
        # finalize NCCL
        while(it != comm_map.end()):
            ncclCommDestroy(dereference(it).second)
            postincrement(it)
