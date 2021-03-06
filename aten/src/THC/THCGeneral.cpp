#include "THCGeneral.h"
#include "TH.h"
#include "THCAllocator.h"
#include "THCCachingHostAllocator.h"
#include "THCTensorRandom.h"
#include "THCGeneral.hpp"

#include "ATen/cuda/CUDAStream.h"

#include "THCCachingAllocator.h"
#include <stdlib.h>
#include <stdint.h>

/* Size of scratch space available in global memory per each SM + stream */
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

/* Minimum amount of scratch space per device. Total scratch memory per
 * device is either this amount, or the # of SMs * the space per SM defined
 * above, whichever is greater.*/
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE 32768 * sizeof(float)

/* Maximum number of P2P connections (if there are more than 9 then P2P is
 * enabled in groups of 8). */
#define THC_CUDA_MAX_PEER_SIZE 8

void THCState_free(THCState* state)
{
  free(state);
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) malloc(sizeof(THCState));
  memset(state, 0, sizeof(THCState));
  return state;
}

void THCudaInit(THCState* state)
{
  if (!state->cudaDeviceAllocator) {
    state->cudaDeviceAllocator = THCCachingAllocator_get();
  }
  if (!state->cudaHostAllocator) {
    state->cudaHostAllocator = getTHCCachingHostAllocator();
  }

  int numDevices = 0;
  THCudaCheck(cudaGetDeviceCount(&numDevices));
  state->numDevices = numDevices;

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    malloc(numDevices * sizeof(THCCudaResourcesPerDevice));
  memset(state->resourcesPerDevice, 0, numDevices * sizeof(THCCudaResourcesPerDevice));

  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(numDevices * sizeof(struct cudaDeviceProp));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, numDevices, device);

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) malloc(sizeof(int*) * numDevices);
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) malloc(sizeof(int) * numDevices);
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else if (j / THC_CUDA_MAX_PEER_SIZE != i / THC_CUDA_MAX_PEER_SIZE)
        state->p2pAccessEnabled[i][j] = 0;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaGetDeviceProperties(&state->deviceProperties[i], i));

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);

  free(state->rngState);
  free(state->deviceProperties);

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  /* cleanup per-device state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    THCudaCheck(cudaSetDevice(dev));
    THCCudaResourcesPerDevice* res = &(state->resourcesPerDevice[dev]);
    
    // Frees BLAS handle
    if (res->blasHandle) {
      THCublasCheck(cublasDestroy(res->blasHandle));
    }

    // Frees sparse handle
    if (res->sparseHandle) {
      THCusparseCheck(cusparseDestroy(res->sparseHandle));
    }
  }

  free(state->resourcesPerDevice);
  if (state->cudaDeviceAllocator == THCCachingAllocator_get()) {
    THCCachingAllocator_emptyCache();
  }
  if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_emptyCache();
  }

  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  if (dev < 0 || dev >= state->numDevices) {
    THError("%d is not a device", dev);
  }
  if (devToAccess < 0 || devToAccess >= state->numDevices) {
    THError("%d is not a device", devToAccess);
  }
  if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
    int prevDev = 0;
    THCudaCheck(cudaGetDevice(&prevDev));
    THCudaCheck(cudaSetDevice(dev));

    int access = 0;
    THCudaCheck(cudaDeviceCanAccessPeer(&access, dev, devToAccess));
    if (access) {
      cudaError_t err = cudaDeviceEnablePeerAccess(devToAccess, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        cudaGetLastError();
      } else {
        THCudaCheck(err);
      }
      state->p2pAccessEnabled[dev][devToAccess] = 1;
    } else {
      state->p2pAccessEnabled[dev][devToAccess] = 0;
    }

    THCudaCheck(cudaSetDevice(prevDev));
  }
  return state->p2pAccessEnabled[dev][devToAccess];
}

struct cudaDeviceProp* THCState_getCurrentDeviceProperties(THCState* state)
{
  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));

  return &(state->deviceProperties[curDev]);
}

struct cudaDeviceProp* THCState_getDeviceProperties(THCState* state, int device)
{
  THAssert(device >= 0 && device < state->numDevices);
  return &(state->deviceProperties[device]);
}

struct THCRNGState* THCState_getRngState(THCState *state)
{
  return state->rngState;
}

THAllocator* THCState_getCudaHostAllocator(THCState* state)
{
  return state->cudaHostAllocator;
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

THCStream* THCState_getStreamOnDevice(THCState* state, int device) {
  return at::cuda::detail::CUDAStream_getCurrentStreamOnDeviceUnsafe(device);
}

void THCState_setStreamOnDevice(THCState *state, int device, THCStream *stream) {
  at::cuda::detail::CUDAStream_setStreamOnDevice(device, stream);
}

cudaStream_t THCState_getCurrentStreamOnDevice(THCState *state, int device) {
  return at::cuda::detail::CUDAStream_stream(
    at::cuda::detail::CUDAStream_getCurrentStreamOnDeviceUnsafe(device));
}

cudaStream_t THCState_getCurrentStream(THCState *state) {
  return at::cuda::detail::CUDAStream_stream(
    at::cuda::detail::CUDAStream_getCurrentStreamUnsafe());
}

THCStream* THCState_getStream(THCState *state) {
  return at::cuda::detail::CUDAStream_getCurrentStreamUnsafe();
}

void THCState_setStream(THCState *state, THCStream *stream) {
  at::cuda::detail::CUDAStream_setStream(stream);
}

cublasHandle_t THCState_getCurrentBlasHandle(THCState *state)
{
  // Short-circuits if state is NULL
  // Note: possible in debugging code or improperly instrumented kernels
  if (!state) {
    THError("THCState and sparseHandles must be set as there is no default sparseHandle");
    return NULL;
  }
    
  int device;
  THCudaCheck(cudaGetDevice(&device));

  // Creates the BLAS handle if not created yet
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (!res->blasHandle) {
    THCublasCheck(cublasCreate(&res->blasHandle));
  }

  return res->blasHandle;
}

cusparseHandle_t THCState_getCurrentSparseHandle(THCState *state)
{
  // Short-circuits if state is NULL
  // Note: possible in debugging code or improperly instrumented kernels
  if (!state) {
    THError("THCState and sparseHandles must be set as there is no default sparseHandle");
    return NULL;
  }

  int device;
  THCudaCheck(cudaGetDevice(&device));
  
  // Creates the sparse handle if not created yet
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (!res->sparseHandle) {
    THCusparseCheck(cusparseCreate(&res->sparseHandle));
  }

  return res->sparseHandle;
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  return res->scratchSpacePerStream;
}

void __THCudaCheck(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THCudaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
      alreadyFailed = 1;
    }
    _THError(file, line, "cuda runtime error (%d) : %s", err,
             cudaGetErrorString(err));
  }
}

void __THCudaCheckWarn(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    fprintf(stderr, "THCudaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
  }
}

void __THCublasCheck(cublasStatus_t status, const char *file, const int line)
{
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case CUBLAS_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case CUBLAS_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case CUBLAS_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

#ifndef __HIP_PLATFORM_HCC__
      case CUBLAS_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUBLAS_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;
#endif

      case CUBLAS_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cublas runtime error : %s", errmsg);
  }
}

void __THCusparseCheck(cusparseStatus_t status, const char *file, const int line)
{
  if(status != CUSPARSE_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case CUSPARSE_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case CUSPARSE_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case CUSPARSE_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case CUSPARSE_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case CUSPARSE_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUSPARSE_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case CUSPARSE_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        errmsg = "the matrix type is not supported by this function";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cusparse runtime error : %s", errmsg);
  }
}

void* THCudaMalloc(THCState *state, size_t size)
{
  THCudaCheck(cudaGetLastError());
  THCDeviceAllocator* allocator = state->cudaDeviceAllocator;
  return allocator->raw_allocate(size);
}

void THCudaFree(THCState *state, void* ptr) {
  state->cudaDeviceAllocator->raw_deallocate(ptr);
}

at::DataPtr THCudaHostAlloc(THCState *state, size_t size)
{
  THCudaCheck(cudaGetLastError());
  THAllocator* allocator = state->cudaHostAllocator;
  return allocator->allocate(size);
}

void THCudaHostRecord(THCState *state, void *ptr) {
  if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
    THCStream* stream = THCState_getStream(state);
    THCCachingHostAllocator_recordEvent(ptr, stream);
  }
}

cudaError_t THCudaMemGetInfo(THCState *state,  size_t* freeBytes, size_t* totalBytes, size_t* largestBlock)
{
  size_t cachedBytes = 0;
  THCDeviceAllocator* allocator = state->cudaDeviceAllocator;

  *largestBlock = 0;
  /* get info from CUDA first */
  cudaError_t ret = cudaMemGetInfo(freeBytes, totalBytes);
  if (ret!= cudaSuccess)
    return ret;

  int device;
  ret = cudaGetDevice(&device);
  if (ret!= cudaSuccess)
    return ret;

  /* not always true - our optimistic guess here */
  *largestBlock = *freeBytes;

  if (allocator == THCCachingAllocator_get()) {
    THCCachingAllocator_cacheInfo(device, &cachedBytes, largestBlock);
  }

  /* Adjust resulting free bytes number. largesBlock unused for now */
  *freeBytes += cachedBytes;
  return cudaSuccess;
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include "THCStorage.cpp"
#include "THCAllocator.cpp"

/* from THCHalf.h */

half THC_float2half(float f)
{
#if CUDA_VERSION < 9000
  half h;
  TH_float2halfbits(&f, &h.x);
  return h;
#else
  __half_raw h_raw;
  TH_float2halfbits(&f, &h_raw.x);
  return half(h_raw);
#endif
}

float  THC_half2float(half h)
{
  float f;
#if CUDA_VERSION < 9000
  TH_halfbits2float(&h.x, &f);
#else
  __half_raw h_raw(h);
  TH_halfbits2float(&h_raw.x, &f);
#endif
  return f;
}
