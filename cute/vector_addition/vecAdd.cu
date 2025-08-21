#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cute/tensor.hpp>


__global__ void vector_add_kernel(
    float const* A,
    float const* B,
    float* C,
    int N
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

int main(){
    int N = 1024;
    size_t bytes = N * sizeof(float);

    std::vector<float> ha(N) , hb(N) , hc(N);
    for (int i = 0; i < N; ++i) {
      ha[i] = static_cast<float>(i);
      hb[i] = static_cast<float>(i * 2);
    }

    float *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da , ha.data() , bytes , cudaMemcpyHostToDevice);
    cudaMemcpy(db , hb.data() , bytes , cudaMemcpyHostToDevice);

    int thread_per_block = 256;
    int block_per_grid = (N + thread_per_block - 1) / thread_per_block;
    vector_add_kernel<<<block_per_grid, thread_per_block>>>(da, db, dc, N);
    cudaEvent_t start ,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i = 0 ; i < 100 ; i++){
        vector_add_kernel<<<block_per_grid , thread_per_block>>>(da , db , dc , N);
    
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "TIME : " << ms / 100 << std::endl;

    cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (abs(hc[i] - (ha[i] + hb[i])) > 1e-5) {
            std::cout << "Verification failed at index " << i << "!" << std::endl;
            return -1;
        }
    }
    std::cout << "Kernel 1 (Vector Add): Verification successful!" << std::endl;
}