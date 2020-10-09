# cython: language_level = 3

cdef extern from "cuda_runtime.h" nogil:
    ctypedef struct CUstream_st:
        pass
    ctypedef CUstream_st* cudaStream_t
    ctypedef enum cudaError:
        cudaSuccess = 0
    ctypedef cudaError cudaError_t
    cdef cudaError_t cudaSetDevice(int device)
    cdef cudaError_t cudaGetDevice(int* device)
    cdef const char* cudaGetErrorString(cudaError_t error)


cdef extern from "nccl.h" nogil:
    ctypedef enum ncclResult_t:
        ncclSuccess                 =  0
        ncclUnhandledCudaError      =  1
        ncclSystemError             =  2
        ncclInternalError           =  3
        ncclInvalidArgument         =  4
        ncclInvalidUsage            =  5
        ncclNumResults              =  6 

    ctypedef enum ncclDataType_t:
        ncclInt8       = 0
        ncclChar       = 0
        ncclUint8      = 1
        ncclInt32      = 2
        ncclInt        = 2
        ncclUint32     = 3
        ncclInt64      = 4
        ncclUint64     = 5
        ncclFloat16    = 6
        ncclHalf       = 6
        ncclFloat32    = 7
        ncclFloat      = 7
        ncclFloat64    = 8
        ncclDouble     = 8
        ncclNumTypes   = 9

    # Opaque handle to communicator
    ctypedef struct ncclComm:
        pass
    ctypedef ncclComm* ncclComm_t

    cdef int NCCL_UNIQUE_ID_BYTES = 128
    DEF _NCCL_UNIQUE_ID_BYTES = 128

    ctypedef struct ncclUniqueId:
        char internal[_NCCL_UNIQUE_ID_BYTES]

    const char* ncclGetErrorString(ncclResult_t result)

    cdef ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)

    cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)

    cdef ncclResult_t ncclCommDestroy(ncclComm_t comm)

    cdef ncclResult_t ncclGroupStart()

    cdef ncclResult_t ncclGroupEnd()

    cdef ncclResult_t ncclSend(
        const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)

    cdef ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
        ncclComm_t comm, cudaStream_t stream)
