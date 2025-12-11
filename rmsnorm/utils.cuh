#include "ATen/core/interned_strings.h"
#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <pthread.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceFP32(float value){
  uint64_t mask = 0xffffffff;
  for(int clip = WARP_SIZE >> 1 ; clip >=1 ; clip >>=1){
    value += __shfl_xor_sync(mask , value , clip);
  }
  return value;
}

__device__ __forceinline__ float blockReduceFP32(float val , float* smem){
  
  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid % WARP_SIZE;
  
  val = warpReduceFP32(val);
  
  if(lane_id == 0){
    smem[warp_id] = val;
  }
  __syncthreads();
  
  if(warp_id == 0){
    val = smem[lane_id];
    val = warpReduceFP32(val);
    if(lane_id == 0){
      smem[0] = val;
    }
    __syncthreads();
    return smem[0];
  }
  
}



