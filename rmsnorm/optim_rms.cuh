#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.cuh"


__global__ void rmsnormKernel(
  half2* inputMatrix, // M , N
  half2* weightMatrix, // (1,N)
  half2* outMatrix, // M , N
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
  int ncols = N / 2;
  
  int rowStart = bidx * ncols;

  
  extern __shared__ char smem[];
  
  float* smem_rows = reinterpret_cast<float*>(smem);
  float* smem_partial_sum = reinterpret_cast<float*>(smem + N);
  
  
  float partialSum = 0;
  for(int idx = tidx ; idx < ncols ; idx += blockDim.x){
    half2 element = __ldg(&inputMatrix[rowStart + idx]);

    float eleFloat1 = __half2float(element.x);
    float eleFloat2 = __half2float(element.y);
    
    smem_rows[idx * 2] = eleFloat1;
    smem_rows[idx * 2 + 1] = eleFloat2;
    
    partialSum += eleFloat1 * eleFloat1;
    partialSum += eleFloat2 * eleFloat2;
  }
  
  __syncthreads();
  
  float totalSum = blockReduceFP32(partialSum, smem_partial_sum);
  
  float inv_rms = rsqrtf((totalSum / float(N)) + eps);
  
  for(int idx = tidx ; idx < ncols ; idx += blockDim.x){
    
    half2 weight = __ldg(&weightMatrix[idx]);

    float elem1 = smem[idx * 2]; 
    float out1 = (elem1 * inv_rms) * __half2float(weight.x);

    float elem2 = smem[idx * 2 + 1];
    float out2 = (elem2 * inv_rms) * __half2float(weight.y);

    half2 output = __floats2half2_rn(out1 , out2);
    outMatrix[rowStart + idx] = output;
  }
}


void launchOptimRms(
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

  half2* inputMatrix2 = reinterpret_cast<half2*>(inputMatrix);
  half2* weightArray2 = reinterpret_cast<half2*>(weightArray);
  half2* outputMatrix2 = reinterpret_cast<half2*>(outputMatrix);
  rmsnormKernel<<<blocksPerGrid , threadsPerBlock , smemSize>>>(inputMatrix2, weightArray2, outputMatrix2, M, N);
}
