#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpMax(float val){
    #pragma unroll
    for(int offset = warpSize/2 ; offset > 0 ; offset /= 2){
        float temp = __shfl_down_sync(0xffffffff , val , offset);
        val = fmaxf(val , temp);

    }
    return val;

}


__global__ void reductionMaxOptimized(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int gridStride = gridDim.x * blockDim.x;
    
    // Registers for multiple values per thread
    float localMax = -INFINITY;
    
    // Coalesced global memory access with grid-stride loop
    for (int i = gid; i < n; i += gridStride) {
        localMax = fmaxf(localMax, input[i]);
    }
    
    // Warp-level reduction using shuffle instructions
    localMax = warpMax(localMax);
    
    // Block-level reduction
    __shared__ float blockMax[32]; // Max 32 warps per block
    int laneId = tid % 32;
    int warpId = tid / 32;
    
    if (laneId == 0) {
        blockMax[warpId] = localMax;
    }
    __syncthreads();
    
    // Final warp reduces the warp results
    if (warpId == 0) {
        localMax = (laneId < (blockDim.x + 31) / 32) ? blockMax[laneId] : -INFINITY;
        localMax = warpMax(localMax);
        
        if (laneId == 0) {
            output[bid] = localMax;
        }
    }
}



void launch_max(
    float* input,
    float N
){
    float *output;
    cudaMalloc(&output, 1024 * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = min(1024, (N + threadsPerBlock - 1) / threadsPerBlock);

    reductionMaxOptimized<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();

    if(blocksPerGrid > 1){
        reductionMaxOptimized<<<1 , blocksPerGrid>>>(output, output, N);
    }

}