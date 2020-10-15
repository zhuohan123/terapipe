#include <cstdio>
#include <cstdint>

#include <chrono>
#include <iostream>

#include <unistd.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}


int main(int argc, char* argv[])
{
  // 1 GB buffer.
  int size = 1024*1024*1024/sizeof(float);
  int myRank, nRanks, localRank = 0;

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  // calculating localRank which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
    if (p == myRank) break;
    if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  // each process is using the first GPU in the node
  int nDev = 1;

  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // picking GPUs based on localRank
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(localRank*nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  ncclUniqueId id;
  ncclComm_t comms[nDev];

  // generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // initializing NCCL, group API is required around ncclCommInitRank as it is
  // called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(localRank*nDev + i));
    NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
  }
  NCCLCHECK(ncclGroupEnd());

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread/process

  /* NCCLCHECK(ncclGroupStart()); */
  if (myRank == 0) {
    NCCLCHECK(ncclSend((const void*)sendbuff[0], size, ncclFloat, /*peer=*/1, comms[0], s[0]));
  } else if (myRank == 1) {
    NCCLCHECK(ncclRecv((void*)recvbuff[0], size, ncclFloat, /*peer=*/0, comms[0], s[0]));
  }

  if (myRank == 0) {
    NCCLCHECK(ncclRecv((void*)recvbuff[0], size, ncclFloat, /*peer=*/1, comms[0], s[0]));
  } else if (myRank == 1) {
    NCCLCHECK(ncclSend((const void*)sendbuff[0], size, ncclFloat, /*peer=*/0, comms[0], s[0]));
  }
  /* NCCLCHECK(ncclGroupEnd()); */

  // synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "It takes " << time_span.count() << " seconds." << std::endl;

  // freeing device memory
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  // finalizing NCCL
  for (int i=0; i<nDev; i++) {
    ncclCommDestroy(comms[i]);
  }

  // finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
