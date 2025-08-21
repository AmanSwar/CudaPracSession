#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define M 1024
#define N 1024

void cublas_matAdd(float *da, float *db, float *dc) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f;
  const float beta = 1.0f;


  const int m_geam = N;
  const int n_geam = M;
  const int lda = N;
  const int ldb = N;
  const int ldc = N;

  cublasSgeam(
    handle,
    CUBLAS_OP_T,
    CUBLAS_OP_T,
    m_geam,
    n_geam,
    &alpha ,
    da , lda,
    &beta,
    db , ldb,
    dc , ldc
  );

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m_geam, n_geam, &alpha, da,
                lda, &beta, db, ldb, dc, ldc);
    }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  std::cout << "CUBLAS TIME : " << ms / 100 << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy(handle);
}

bool checker(float *cublas_out, float *kernel_out) {

  for (int i = 0; i < M*N; i++) {
    if (std::abs(cublas_out[i] - kernel_out[i]) > 1e-3) {
      return false;
    }
  }

  return true;
}

void benchmark(void(launch_func)(float *, float *, float *)) {
  size_t bytes = M*N * sizeof(float);

  std::vector<float> ha(M*N), hb(M*N), h_kernelout(M*N), h_cublasout(M*N);
  for (int i = 0; i < M*N; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(i * 2);
  }

  float *da, *db, *d_kernelout, *d_cublasout;
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&d_kernelout, bytes);
  cudaMalloc(&d_cublasout, bytes);

  cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice);

  launch_func(da, db, d_kernelout);
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    launch_func(da, db, d_kernelout);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  std::cout << "Kernel TIME : " << ms / 100 << std::endl;

  cublas_matAdd(da, db, d_cublasout);
  cudaMemcpy(d_kernelout, h_kernelout.data(), sizeof(float) * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(d_cublasout, h_cublasout.data(), sizeof(float) * N,
             cudaMemcpyDeviceToHost);

  std::cout << checker(h_kernelout.data(), h_cublasout.data()) << std::endl;

  delete da;
  delete db;
  delete d_kernelout;
  delete d_cublasout;
}
