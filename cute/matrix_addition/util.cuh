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
