#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#include <cuda_runtime.h>
#define M 10000
#define N 10000
#define K 10000


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

    dim3 blockDim()

}







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