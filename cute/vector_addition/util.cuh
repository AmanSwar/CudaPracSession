#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>


#define N 4084

void cublas_vecAdd(
    float* da,
    float* db,
    float* dc
){
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(dc, db, sizeof(float) * N, cudaMemcpyDeviceToDevice);
    const float alpha = 1.0f;
    cublasSaxpy(handle , N , &alpha , da , 1 , dc , 1);


    cudaEvent_t start , end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i = 0 ; i < 100 ; i++){
      cublasSaxpy(handle, N, &alpha, da, 1, dc, 1);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms , start , end);
    std::cout << "CUBLAS TIME : " <<ms/ 100 << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);
}


bool checker(float* cublas_out , float* kernel_out){

    for(int i = 0 ; i < N ; i++){
        if(std::abs(cublas_out[i] - kernel_out[i]) > 1e-3){
            return false;
        }
    }

    return true;
}


void benchmark(void(launch_func)(float* , float* , float*)){
    size_t bytes = N * sizeof(float);

    std::vector<float> ha(N), hb(N), h_kernelout(N) , h_cublasout(N);
    for (int i = 0; i < N; ++i) {
        ha[i] = static_cast<float>(i);
        hb[i] = static_cast<float>(i * 2);
    }

    float *da, *db, *d_kernelout , *d_cublasout;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&d_kernelout, bytes);
    cudaMalloc(&d_cublasout, bytes);

    cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice);

    launch_func(da , db ,d_kernelout);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        launch_func(da , db ,d_kernelout);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "Kernel TIME : " << ms / 100 << std::endl;

    cublas_vecAdd(da , db ,d_cublasout);
    cudaMemcpy(d_kernelout , h_kernelout.data() , sizeof(float)* N , cudaMemcpyDeviceToHost);
    cudaMemcpy(d_cublasout , h_cublasout.data() , sizeof(float) * N , cudaMemcpyDeviceToHost);
    
    std::cout << checker(h_kernelout.data(),h_cublasout.data()) << std::endl;

    delete da;
    delete db;
    delete d_kernelout;
    delete d_cublasout;

}


