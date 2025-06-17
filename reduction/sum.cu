#include <cmath>
#include <iostream>

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>


__global__
void findSum(
    float* vector,
    float* output,
    int N
){
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;
    extern __shared__ float SHARED_MEM[];

    if(local_index < N){
        SHARED_MEM[local_index] = vector[global_index];
    }
    else{
        SHARED_MEM[local_index] = 0;
    }

    for(int stride = 1 ; stride < blockDim.x ; stride *= 2){

        if(local_index % (2 * stride) == 0){
            SHARED_MEM[local_index] += SHARED_MEM[local_index + stride];
        }

        __syncthreads();
    }

    if(local_index == 0){
        output[blockIdx.x] = SHARED_MEM[0];
    }

}




void launch_findSum(
    float *vector,
    int N
){
    int threadsPerBlock = 256;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t shared_mem_size = threadsPerBlock * sizeof(float);
    float* output;
    cudaMalloc((void**)&output , sizeof(float) * N);
    findSum<<<
        blockPerGrid,
        threadsPerBlock,
        shared_mem_size
        >>>(
            vector,
            output,
            N
        );

    if(blockPerGrid > 1){
        findSum<<<
            blockPerGrid,
            threadsPerBlock,
            shared_mem_size
            >>>(output, output, blockPerGrid);
    }


    float* h_output;
    h_output = new float[blockPerGrid];
    cudaMemcpy(h_output , output , blockPerGrid * sizeof(float) , cudaMemcpyDeviceToHost);

    std::cout << h_output[0] << std::endl;
}


int main(){
    int N = 1000;

    float* input;
    input = new float[N];
    
    //
    int sum_verify = 0;
    for(int i = 0 ; i < N ; i++){
        input[i] = i;
        sum_verify += i;
    }

    float* vector;
    cudaMalloc((void**)&vector , sizeof(float)* N);
    cudaMemcpy(vector , input , N * sizeof(float) , cudaMemcpyHostToDevice);

    launch_findSum(vector,  N);

    // std::cout << std::accumulate(input , input + N ,0) << std::endl;
    std::cout << sum_verify << std::endl;

    
}