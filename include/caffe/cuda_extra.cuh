#ifndef CAFFE_CUDA_EXTRA_CUH_
#define CAFFE_CUDA_EXTRA_CUH_

#include <cuda.h>


// CUDA: atomicAdd is not defined for data type double
static __inline__ __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
  return __longlong_as_double(old);
}

#endif  //  #ifndef CAFFE_CUDA_EXTRA_CUH_
