#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>


// ================================= NAIVE ===================
__global__ void relu_naive(
    half* matrixA,
    half* matrixOut,
    int M , int N
){
    int global_index_x = blockDim.x  * blockIdx.x + threadIdx.x; // total range -> 0 - M-1
    int local_index_x = threadIdx.x;
    int row_start = global_index_x * N;
    if(global_index_x < M){
        for(int idx = local_index_x ; idx < N ; idx += blockDim.x){
            matrixOut[row_start +idx] = __hmax(__float2half(0.0) , matrixA[row_start + idx]); 
        }
    }
}

void launch_naive(
    half* matrixA,
    half* matrixOut,
    int M , int N
){
    int block_dim = 1024;
    int grid_dim = M;
    relu_naive<<<grid_dim , block_dim>>>(matrixA , matrixOut , M , N);
}


// ===========================================================================================

__global__ void relu_coal(
    half* matrixA,
    half* matrixO,
    int M , int N
){
    int row_index = blockIdx.x;
    int local_index = threadIdx.x;

    int row_start = row_index * N;
    
    for(int idx = local_index ; idx < N ; idx+= blockDim.x){
        matrixO[row_start + idx] = __hmax(0.0 , matrixA[row_start + idx]);
    }
}

void launch_coal(half *matrixA, half *matrixOut, int M, int N) {
  int block_dim = 1024;
  int grid_dim = M;
  relu_coal<<<grid_dim, block_dim>>>(matrixA, matrixOut, M, N);
}

// ======================================= BENCHMARK ================================================





void benchmark(void (*function)(half *, half * , int , int) , std::string name, half* a , half* b , int M , int N) { 
    cudaEvent_t start, end;
    
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    function(a , b ,M , N);
    cudaEventRecord(start);
    for(int i = 0 ; i < 100 ; i++){
        function(a,b, M , N);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0;
    cudaEventElapsedTime(&ms , start , end);
    std::cout << name << " average time: " << (ms / 100) << " ms" << std::endl;
    std::cout << name << " GFLOPS : " << ((M*N) / ((ms / 100)/1000))/ 1e9 << std::endl;
}

int main(){
    int M = 1024;
    int N = 2048;

    size_t _size = sizeof(half) * M * N;

    half* ha = new half[M*N];
    half* hb = new half[M*N];
    half* hc = new half[M*N];

    half* da , *db;
    cudaMalloc(&da , _size);
    cudaMalloc(&db , _size);


    cudaMemcpy(da , ha , _size , cudaMemcpyHostToDevice);
    cudaMemcpy(db , hb , _size , cudaMemcpyHostToDevice);

    benchmark(launch_naive , "Naive" , da , db , M , N);
    benchmark(launch_coal , "Coal" , da , db , M , N);
    
    
}