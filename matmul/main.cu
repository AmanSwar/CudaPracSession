#include <cstdlib>
#include <cuda_runtime.h>
#define M 10000
#define N 10000
#define K 10000


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

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    // //launch kernel
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop); 
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);



    
}