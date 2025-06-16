#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "torch/types.h"
// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_complex_builtins.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <torch/torch.h>


//TO DO : optimize it 
__device__
void findMax(
    float* vector
){

    int local_idx = threadIdx.x; // range = 0 - blockSize

    //rest of loop
    for(int i = 1 ; i < blockDim.x ; i *=2){
        if(local_idx % (2 * i) == 0){
            if(vector[local_idx] < vector[local_idx + i]){
                vector[local_idx] = vector[local_idx + i];
            }
        }
        __syncthreads();
    }
}


__device__
void findSum(
    float* vector
){
    int local_idx = threadIdx.x;

    for(int stride = 0 ; stride < blockDim.x ; stride *= 2){
        if(local_idx % (2 * stride) == 0){
            vector[local_idx] += vector[local_idx + stride];
        }
        __syncthreads();
    }

}


__global__
void _softmax_kernel(
    float* matrix_A,
    float* max_arr,
    float* sum_arr,
    float* output_matrix,
    int N , int Dim
){


    /*
    Take 1 row at a time-> distribute acros SM
    then use a for loop for getting the next row
    */
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // range -> 0 - total threads launched i.e gridDim * blockDim
    int local_thread =  threadIdx.x; // range -> 0 - threadPerBlock

    extern __shared__ float SHARED_MEM[];
    //divide shared mem into 2 parts -> 
    //  first one carrying the original vector
    //  second one carrying the copy so that we can find maximum
    float* vector_sm = &SHARED_MEM[0];
    float* temp_sm = &SHARED_MEM[blockDim.x];

    //load part of one vector into shared memory
    for(int row_offset = 0 ; row_offset < N; row_offset += Dim){
        // load data into shared mem for each SM
        if(global_idx < Dim){
            vector_sm[local_thread] = matrix_A[global_idx + row_offset];
            temp_sm[local_thread] = matrix_A[global_idx + row_offset];
        }else{
            vector_sm[local_thread] = -INFINITY;
        }
        __syncthreads();

        //find maximum for each row
        findMax(temp_sm);
        __syncthreads();
        if(local_thread == 0){
            max_arr[blockIdx.x] = temp_sm[0];
            // printf("%f | " , max_arr[blockIdx.x]);
        }
        findMax(max_arr);
        __syncthreads();
        float MAX_OF_ROW = 0.0f;
        MAX_OF_ROW = max_arr[0];

        //numerator
        vector_sm[local_thread] = vector_sm[local_thread] - MAX_OF_ROW;

        if(local_thread < Dim){
            temp_sm[local_thread] = vector_sm[local_thread];
        }

        findSum(temp_sm);
        __syncthreads();
        if(local_thread == 0){
            sum_arr[blockIdx.x] = temp_sm[0];
        }
        findSum(sum_arr);
        float SUM = 0.0f;
        SUM = sum_arr[0];

        output_matrix[global_idx + row_offset] = vector_sm[local_thread] / SUM;

    }
    
    
}


void launch_softmax_kernel(
    float* matrix,
    float* output_matrix,
    int N, int Dim
){
    /*
    N = number of vectors in matrix
    Dim = dimension of vectors in matrix
    */

    int threadsPerBlock = 256;
    int blocksPerGrid = (Dim + threadsPerBlock -1) / threadsPerBlock;
    
    float* max_arr;
    float* sum_arr;
    cudaMalloc((void**)&max_arr , sizeof(float) * blocksPerGrid);
    cudaMalloc((void**)&sum_arr , sizeof(float) * blocksPerGrid); // total number of threads launched = Dim only 


    size_t shared_mem_size = 2 * threadsPerBlock * sizeof(float);
    
    // _softmax_kernel<<<blocksPerGrid , threadsPerBlock , shared_mem_size>>>(matrix, max_arr, output_matrix, N, Dim);
    std::cout << "Calling kernel" << std::endl;
    _softmax_kernel<<<
    blocksPerGrid ,
    threadsPerBlock,
    shared_mem_size
    >>>(
        matrix,
        max_arr,
        sum_arr,
        output_matrix,
        N,
        Dim
    );

    cudaDeviceSynchronize();
    std::

}


bool tester(
    float* input,
    int N,
    int Dim,
    float* kernel_output
){

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor input_tensor = torch::empty({N, Dim}, options);
    cudaMemcpy(
            input_tensor.data_ptr<float>(), 
            input, 
            N * Dim * sizeof(float), 
            cudaMemcpyHostToDevice);

    torch::Tensor output_tensor;
    output_tensor = torch::softmax(input_tensor, /*dim=*/1);
    output_tensor = torch::softmax(input_tensor, 1);
    output_tensor = output_tensor.cpu().contiguous();
    const float* output_ten = output_tensor.data_ptr<float>();

    for(int i = 0 ; i < N * Dim ; i ++){
        if(fabs(kernel_output[i] - output_ten[i]) > 0.0001){
            std::cout << "failed" << std::endl;
            std::cout << "At index position " << i << std::endl;

            return false;
        }
    }
    return true;
}


int main(){
    float* mat , *out;
    int N = 1000;
    int Dim = 1;
    std::cout << "allocating memory in Host" << std::endl;
    mat = (float*)malloc(N * Dim * sizeof(float));
    out = (float*)malloc(N * Dim  * sizeof(float));
    
    //init
    std::cout << "Pre-filling" << std::endl;
    for(int i = 0 ; i < N *Dim ; i++){
        mat[i] = i;   
    }

    float* device_mat , *device_out;
    std::cout << "allocating memory in Device" << std::endl;
    cudaMalloc((void**)&device_mat , N * Dim * sizeof(float));
    cudaMalloc((void**)&device_out , N * Dim * sizeof(float));

    cudaMemcpy(device_mat, mat, N * Dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, out, N * Dim * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Call launch function" << std::endl;
    launch_softmax_kernel(device_mat, device_out, N, Dim);
    cudaDeviceSynchronize();    
    cudaMemcpy(out, device_out, N * Dim * sizeof(float), cudaMemcpyDeviceToHost);



}
