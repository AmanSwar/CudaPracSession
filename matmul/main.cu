#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#include "ATen/core/interned_strings.h"
#include "ATen/ops/glu_backward_ops.h"
#include "main.h"
#include <cuda_runtime.h>

/*
List of optimization
- naive
- tiled
- optimized tiled -> increase register pr
- tensor cores

*/



__global__
void naive_matmul(
    float* matrixA,
    float* matirxB,
    float* output
){
    int global_index_x = blockDim.x * blockIdx.x + threadIdx.x;
    // int global_index_y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = global_index_x  / M;
    int y = global_index_x % N;

    if(x < M && y < N){
        for(int k = 0 ; k < K ; k++){
            output[x * M + y] = matrixA[x*K +k] * matirxB[k*N + y];
        }
    }
    
}

void launch_naive_matmul(
    float* matrixA,
    float* matrixB,
    float* output
){

    int threadsPerBlock = 1024;
    int blocksPerGrid = ((M+N) + threadsPerBlock - 1) / threadsPerBlock;

    naive_matmul<<<blocksPerGrid ,  threadsPerBlock>>>(
        matrixA,
        matrixB,
        output
    );
    cudaDeviceSynchronize();
}



// -----------------------------------------------------------




__global__
void tiled_matmul(
    float* matrixA,
    float* matrixB,
    float* output
){

    const int TILE_SIZE = 32;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;    

    int block_col = bx * TILE_SIZE + tx;  // gives all cols in a block  | 
    //                                                                    ]-> row * row_offset + col = all address    
    int block_row = by * TILE_SIZE + ty;   // gives all rows in a block | 

    __shared__ float SM_A[TILE_SIZE][TILE_SIZE];
    __shared__ float SM_B[TILE_SIZE][TILE_SIZE];

    int total_tile_k = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int tile = 0 ; tile < total_tile_k; tile ++){

        if(block_row < M && block_col < K){
            // SM_A[ty][tx] = matrixA[tile * TILE_SIZE + (block_row * K + block_col)];
            SM_A[ty][tx] = matrixA[tile * TILE_SIZE + (ty * N + tx)];
        }
        if(block_row < K && block_col < N){
            SM_B[ty][tx] = matrixB[block_col * TILE_SIZE + block_row];
        }
    }
    

}   




//pybind function
void matmul(at::Tensor)


float time_it(void (*function)(float* , float* , float*) ,float* matrixA , float* matrixB , float* output){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //launch kernel
    function(matrixA , matrixB , output);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


int main(){

    float* matrixA_host = new float[M*K];
    float* matrixB_host = new float[K*N];
    float* matrix_output_host = new float[M*N];

    float* matrixA , *matrixB , *matrixO;
    cudaMalloc(&matrixA , M*K*sizeof(float));
    cudaMalloc(&matrixB , N*K*sizeof(float));
    cudaMalloc(&matrixO , N*M*sizeof(float));

    //init
    for(int i = 0 ; i < M*K ; i++){
        matrixA_host[i] = rand();
    }
    for(int i = 0 ; i < N*K ; i++){
        matrixB_host[i] = rand();
    }

    //copy
    cudaMemcpy(matrixA , matrixA_host , M*K*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB , matrixB_host , N*K*sizeof(float) , cudaMemcpyHostToDevice);

    



    
}