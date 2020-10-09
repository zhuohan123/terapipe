from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t, intptr_t
from libc.string cimport memcpy

from libcpp cimport bool as c_bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string as c_string
from libcpp.unordered_map cimport unordered_map

from _nccl cimport *

from cython.operator cimport dereference, postincrement

import torch

# TODO(Siyuan): python error handling for NCCL

cdef cudaStream_t _get_native_cuda_stream(stream=None, tensor=None):
    if stream is None:
        if tensor is None:
            raise ValueError("stream and tensor cannot both be None")
        if isinstance(tensor, torch.Tensor):
            assert tensor.device.type == 'cuda'
            stream = torch.cuda.current_stream(tensor.device)
        else:
            raise ValueError("Unknown tensor")
    if isinstance(stream, torch.cuda.streams.Stream):
        return <cudaStream_t><intptr_t>stream.cuda_stream
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
    if isinstance(tensor, torch.Tensor):
        assert tensor.device.type == 'cuda'
        device_id = tensor.device.index
        buff = tensor.data_ptr()
        count = tensor.numel()
        dtype = str(tensor.dtype).lstrip('torch.')
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
            ncclComm_t comm
            intptr_t sendbuff
            size_t count
            cudaStream_t cuda_stream
            ncclDataType_t nccltype
        cuda_stream = _get_native_cuda_stream(stream, tensor)
        device_id, sendbuff, count, dtype = _probe_tensor(tensor)
        comm = self.comm_map[self.dev_map[device_id]]
        nccltype = _type_from_string(dtype)
        with nogil:
            ncclSend(<const void*>sendbuff, count, nccltype, peer, comm, cuda_stream)

    def recv_tensor(self, tensor, int peer, stream=None):
        cdef:
            ncclComm_t comm
            intptr_t recvbuff
            size_t count
            cudaStream_t cuda_stream
            ncclDataType_t nccltype
        cuda_stream = _get_native_cuda_stream(stream, tensor)
        device_id, recvbuff, count, dtype = _probe_tensor(tensor)
        comm = self.comm_map[self.dev_map[device_id]]
        nccltype = _type_from_string(dtype)
        with nogil:
            ncclRecv(<void*>recvbuff, count, nccltype, peer, comm, cuda_stream)

    def __dealloc__(self):
        cdef unordered_map[int, ncclComm_t].iterator it = self.comm_map.begin()
        with nogil:
            # finalize NCCL
            while(it != self.comm_map.end()):
              ncclCommDestroy(dereference(it).second)
              postincrement(it)
