#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cute/tensor.hpp>

#include "util.cuh"

__global__ void vector_add_kernel(
    float const* A,
    float const* B,
    float* C
){
    using namespace cute;

    auto layout = make_layout(Int<1024>{});
    Tensor tensor_A = make_tensor(make_gmem_ptr(A) , layout);
    Tensor tensor_B = make_tensor(make_gmem_ptr(B) , layout);
    Tensor tensor_C = make_tensor(make_gmem_ptr(C) , layout);

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_index < N){
        tensor_C(global_index) = tensor_A(global_index) + tensor_B(global_index);
    }
}

void launch_naive_kernel(float* da , float* db , float* dc){
  int thread_per_block = 256;
  int block_per_grid = (N + thread_per_block - 1) / thread_per_block;
  vector_add_kernel<<<block_per_grid, thread_per_block>>>(da, db, dc);
}

int main(){
    benchmark(launch_naive_kernel);
}