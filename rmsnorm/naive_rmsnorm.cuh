#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.cuh"


__global__ void rmsnormNaiveKernel(
  half* inputMatrix, // M , N
  half* weightMatrix, // (1,N)
  half* outMatrix, // M , N
  int M, int N,
  float eps = 1e-6
){
  /*
   * x / rms(x)
   * rms(x) = sqrt(mean(x**2))
   * (x / rms(x)) * weight 
   */
  
  int tidx = threadIdx.x;
  int bidx  = blockIdx.x;
  
  int rowStart = bidx * N;
  
  extern __shared__ char smem[];
  
  float* smem_rows = reinterpret_cast<float*>(smem);
  float* smem_partial_sum = reinterpret_cast<float*>(smem + N);
  
  
  float partialSum = 0;
  for(int idx = tidx ; idx < N ; idx += blockDim.x){
    half element = inputMatrix[rowStart + idx];
    float eleFloat = __half2float(element);
    smem_rows[idx] = eleFloat;
    partialSum += eleFloat * eleFloat;
  }
  
  __syncthreads();
  
  float totalSum = blockReduceFP32(partialSum, smem_partial_sum);
  
  float inv_rms = rsqrtf((totalSum / float(N)) + eps);
  
  for(int idx = tidx ; idx < N ; idx += blockDim.x){
    float element = smem_rows[idx];
    half weight = weightMatrix[idx];
    float output = (element / inv_rms) * __half2float(weight);
    outMatrix[rowStart + idx] = output;
  }
}


void launchNaiveRms(
  half* inputMatrix,
  half* weightArray,
  half* outputMatrix,
  int M , int N
){
  int threadsPerBlock = 256;
  int blocksPerGrid = M;
  
  size_t smemArraySize = sizeof(float) * N;
  size_t smemPartialSize = sizeof(float) * threadsPerBlock;
  size_t smemSize = smemArraySize + smemPartialSize;
  rmsnormNaiveKernel<<<blocksPerGrid , threadsPerBlock , smemSize>>>(inputMatrix, weightArray, outputMatrix, M, N);
}
