#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>



// =============================================NAIVE==============================================

__global__ void relu_naive(
    half* matrixA,
    half* matrixO,
    int M , int N
){
    int row_index = blockIdx.x;
    int local_index = threadIdx.x;

    int row_start = row_index * N;
    
    for(int idx = local_index ; idx < N ; idx+= blockDim.x){
        matrixO[row_start + idx] = __hmax(__float2half(0.0) , matrixA[row_start + idx]);
    }
}

void launch_naive(half *matrixA, half *matrixOut, int M, int N) {
  int block_dim = 1024;
  int grid_dim = M;
  relu_naive<<<grid_dim, block_dim>>>(matrixA, matrixOut, M, N);
}


// ========================================== COALESCED =========================================
__global__ void relu_coal(
    half* matrixA,
    half* matrixO,
    int M , int N
){
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const half zero = __float2half(0.0f);
    
    for (int i = global_index; i < M*N; i += stride) {
        matrixO[i] = __hmax(zero, matrixA[i]);
    }
}

void launch_coal(half *matrixA, half *matrixOut, int M, int N) {
  int block_dim = 256;
  int min_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_size, &block_dim, relu_coal, 0, 0);
  int grid_dim = (block_dim + (M*N) - 1) / block_dim;
  relu_coal<<<grid_dim, block_dim>>>(matrixA, matrixOut, M, N);
}

// ====================================== VECTORIZED ===========================================

#define HALF2(val) (*reinterpret_cast<half2 *>(&val))

__global__ void relu_vec(
    half* matrixA,
    half* matrixO,
    int M , int N
){
    int row_index = blockIdx.x;
    int local_index = threadIdx.x;

    int row_start = row_index * N;

    for(int idx = local_index*2; idx < N ; idx += blockDim.x*2){
        
        if(idx + 1 < N){
            half2 acc;
            half2 val = __ldg(&HALF2(matrixA[row_start + idx]));
            acc = __hmax2(__float2half2_rn(0.0) , val);
            HALF2(matrixO[row_start + idx]) = acc;
        }
        else{
            half val = __ldg(&matrixA[row_start + idx]);
            half result = __hmax(__float2half(0.0) , val);
            matrixO[row_start] = result;
        }
    }
}

void launch_vec(half *matrixA, half *matrixOut, int M, int N) {
  int block_dim = 1024;
  int grid_dim = M;
  relu_vec<<<grid_dim, block_dim>>>(matrixA, matrixOut, M, N);
}


// ====================================== AGGRESIVE VECTORIZATION ============================

#define NUM_ELEMENTS_PER_THREAD 8

__global__ void relu_vecfp16x8(
    half* matrixA,
    half* matrixO,
    int M , int N
){
    int row_index = blockIdx.x;
    int row_start = row_index * N;

    int local_index = threadIdx.x;

    #pragma unroll
    for(int idx = local_index * NUM_ELEMENTS_PER_THREAD; idx < N ; idx += blockDim.x * NUM_ELEMENTS_PER_THREAD){
        #pragma unroll
        for(int i = 0 ; i < NUM_ELEMENTS_PER_THREAD;  i += 2){
            if(idx + i + 1 < N){
                half2 values = __ldca(&HALF2(matrixA[row_start + idx + i]));
                half2 acc = __hmax2(__float2half2_rn(0.0) , values);
                HALF2(matrixO[row_start + idx + i]) = acc;
        
            }
            else{
                half val = __ldca(&matrixA[row_start + idx + i]);
                half acc = __hmax(__float2half(0.0) , val);
                matrixO[row_start + idx + i] = acc;
            }
        }
    }

}

void launch_vecfp16x8(half *matrixA, half *matrixOut, int M, int N) {
  int block_dim = 1024;
  int grid_dim = M;
  int min_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_size, &block_dim, relu_vecfp16x8 , 0 , 0);
  relu_vecfp16x8<<<grid_dim, block_dim>>>(matrixA, matrixOut, M, N);
}

// ======================================= BENCHMARK ================================================

void inline cpu_relu(half *ha, half *hb, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    hb[i] = __float2half(fmaxf(0.0, __half2float(ha[i])));
  }
}


bool inline verify(half* a , half* b , int M , int N){
    for(int i = 0 ; i < M*N ; i++){
        if(std::abs(__half2float(a[i]) - __half2float(b[i])) > 1e-3){
            std::cout << "Fail" << std::endl;
            std::cout << __half2float(a[i]) << " : " << __half2float(b[i]) << std::endl;
            return false;
        }
    }
    std::cout << "Pass" << std::endl;
    return true;
}

void inline init(half* a , int size){
    for(int i = 0 ; i < size ; i++){
        float val = (i % 2 == 0) ? (float)(i+1) : -(float)(i + 1);
        a[i] = __float2half(val);
    }
}

void benchmark(void (*function)(half *, half * , int , int) , std::string name, half* a , half* b , int M , int N , half* hb , half* hc) { 
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
    
    cudaMemcpy(hb , b , sizeof(half) * M * N , cudaMemcpyDeviceToHost);
    bool result = verify(hb , hc , M , N);
}

int main(){
    int M = 1024;
    int N = 2048;

    size_t _size = sizeof(half) * M * N;

    half* ha = new half[M*N];
    half* hb = new half[M*N];
    half* hc = new half[M*N];

    init(ha , M * N);
    cpu_relu(ha , hc , M , N);


    half* da , *db;
    cudaMalloc(&da , _size);
    cudaMalloc(&db , _size);


    cudaMemcpy(da , ha , _size , cudaMemcpyHostToDevice);
    cudaMemcpy(db , hb , _size , cudaMemcpyHostToDevice);

    benchmark(launch_naive , "Naive" , da , db , M , N , hb , hc);
    benchmark(launch_coal , "Coal " , da , db , M , N , hb , hc);
    benchmark(launch_vec , "Vec" , da , db , M , N , hb , hc);
    benchmark(launch_vecfp16x8, "Vec fp16x8 ", da, db, M, N, hb, hc);
}