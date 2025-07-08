#include "ATen/core/interned_strings.h"
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;


//matrix dims for WMMA
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;


//actual mat dim
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;


__global__
void wmma_gemm_kernel(
    half* a,
    half* b,
    float* c,
    int m,
    int n,
    int k
){
    int warpM = (blockIdx.y *blockDim.y + threadIdx.y) / WARP_SIZE;

    int warpN = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;

    //fragments
    wmma::fragment<wmma::matrix_a , WMMA_M , WMMA_N , WMMA_K , half , wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b , WMMA_M , WMMA_N , WMMA_K , half , wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator , WMMA_M , WMMA_N , WMMA_K , half> c_frag;
    wmma::fragment<wmma::accumulator , WMMA_M , WMMA_N , WMMA_K , half> acc_frag;

    //init

    wmma::fill_fragment(acc_frag , 0.0f);

    for(int i = 0 ; i < k; i += WMMA_K){
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if(aRow < m && aCol < k && bRow < k && bCol < N){
            wmma::load_matrix_sync(a_frag , a + aRow * k + aCol , k);
            wmma::load_matrix_sync(b_frag , b + bRow * n + bCol , n);

            //matmul
            wmma::mma_sync(acc_frag , a_frag , b_frag , acc_frag);
        }


    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if(cRow < m && cCol < n){
        wmma::load_matrix_sync(c_frag , c + cRow * n + cCol , n , wmma::mem_row_major);

        for(int i = 0 ; i < c_frag.num_elements ; i++){
            c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
        }

        wmma::store_matrix_sync(c + cRow * n + cCol , c_frag , n , wmma::mem_row_major);
        
    }
    
}




