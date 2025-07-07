#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>

using namespace cute;


template <class ProblemShape , class CtaTiler ,
class TA , class AStride , class ASmemLayout , class AThreadLayout ,
class TB , class BStride , class BSmemLayout , class BThreadLayout,
class TC , class CStride , class CSmemLayout , class CThreadLayout,
class Alpha , class Beta >
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) 
void gemm_device(
    ProblemShape shape_MNK, // (M N K)
    CtaTiler cta_tiler, //tiles info 
    TA const *A, // ptr to A mat
    AStride dA, // layout stride
    ASmemLayout sA_layout, // layout for shared memory
    AThreadLayout tA, // layout of threads to be used for partitioning
    TB const *B, 
    BStride dB,
    BSmemLayout sB_layout,
    BThreadLayout tB,
    TC *C,
    CStride dC,
    CSmemLayout,
    CThreadLayout tC,
    Alpha alpha, // const
    Beta beta
){
    using namespace cute;
    //Tensor obj for each matrix ptsrs 
    Tensor matrixA = make_tensor(make_gmem_ptr(A) , select<0 , 2>(shape_MNK) , dA);
    Tensor matrixB = make_tensor(make_gmem_ptr(B) , select<1 , 2>(shape_MNK) , dB);
    Tensor matrixC = make_tensor(make_gmem_ptr(C) , select<0,1>(shape_MNK) , dC);

    //
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y ,_);

    //blocks in global
    Tensor gA = local_tile(matrixA , cta_tiler , cta_coord , Step<_1 ,X , _1>{});
    Tensor gB = local_tile(matrixB , cta_tiler , cta_coord , Step<_1 ,X , _1>{});
    Tensor gC = local_tile(matrixC , cta_tiler , cta_coord , Step<_1 ,X , _1>{});


    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TA smemB[cosize_v<ASmemLayout>];


    //Tensor obj for blocks in shared mem
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); 
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    
    Tensor tAgA = local_partition(gA , tA , threadIdx.x);
    Tensor tAsA = local_partition(sA , tA , threadIdx.x);

    Tensor tBgB = local_partition(gB , tB , threadIdx.x);
    Tensor tBsB = local_partition(sB , tB , threadIdx.x);

    Tensor tCsA = local_partition(sA , tC , threadIdx.x , Step<_1 , X>{});
    Tensor tCsB = local_partition(sB , tC , threadIdx.x , Step<X , _1>{});
    Tensor tCgC = local_partition(gC , tC , threadIdx.x , Step<_1 , _1>{});

    Tensor tCrC = make_tensor_like(tCgC);

    clear(tCrC);


    auto K_TILE_MAX = size<2>(tAgA);

    for(int k_tile = 0 ; k_tile < K_TILE_MAX ; ++k_tile){

        copy(tAgA(_ , _ , k_tile) , tAsA);
        copy(tBgB(_ , _ ,   k_tile) , tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA , tCsB , tCrC);
        __syncthreads();
    }

    axpby(alpha , tCrC, beta , tCgC);


}


template <class TA , class TB , class TC , class ALpha , class Beta>
void gemm_nt(
    int m , int n , int k, ALpha alpha,
    TA const *A,
    int ldA,
    TB const *B,
    int ldB,
    Beta beta,
    TC *C,
    int ldC,
    cudaStream_t stream=0
){
    using namespace cute;
    //dim M N K
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    // make the shape
    auto prob_shape = make_shape(M , N , K);
    
    // stride
    auto dA = make_stride(Int<1>{}, ldA); 
    auto dB = make_stride(Int<1>{} , ldB);
    auto dC = make_stride(Int<1>{} , ldC);


    // block dims
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    //tiler shape
    auto cta_tiler = make_shape(bM , bN , bK);

    //shared memory layout -> smem holds 1 block so bM x bK , bN
    auto sA = make_layout(make_shape(bM , bK));
    auto sB = make_layout(make_shape(bN , bK));
    auto sC = make_layout(make_shape(bM , bN));

    //thread layout
    auto tA = make_layout(make_shape(Int<32>{} , Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{} , Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{} , Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)) , size(ceil_div(n , bN)));

    gemm_device<<<dimGrid , dimBlock , 0 , stream>>>(
        prob_shape , cta_tiler , A , dA,
        sA,tA,B,dB,sB,tB,C,dC,sC,tC,alpha,beta
    );

}

