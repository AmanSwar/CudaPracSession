#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/memory.h>
#include <cutlass/gemm/threadblock/index_remat.h>

#include "util.cuh"

__global__ void matrix_add_kernel(
    float const* A,
    float const* B,
    float* C
){

    using namespace cute;

    auto layout = make_layout(
        Shape<int, int>{M , N},
        Stride<int , _1>{N , _1{}}
    );

    Tensor tensorA = make_tensor(make_gmem_ptr(A) , layout);
    Tensor tensorB = make_tensor(make_gmem_ptr(B) , layout);
    Tensor tensorC = make_tensor(make_gmem_ptr(C) , layout);

    int global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(global_index_x < M && global_index_y < N){
        tensorC(make_coord(global_index_x , global_index_y)) = tensorA(make_coord(global_index_x , global_index_y)) + tensorB(make_coord(global_index_x , global_index_y));
    }
}

void launch_kernel(float* A , float* B , float* C){
    dim3 threadsPerBlock(16 ,16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x , (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_add_kernel<<<blocksPerGrid , threadsPerBlock>>>(A , B , C);
}


int main(){
    benchmark(launch_kernel);
}