#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class TiledCopyA, class TB, class BStride,
          class BSmemLayout, class TiledCopyB, class TC, class CStride,
          class CSmemLayout, class TiledMma, class Alpha, class Beta>
__global__ static __launch_bounds__(
    decltype(size(TiledMma{})) :: value
) void gemm_device(
    ProblemShape shape_mnk,
    CtaTiler cta_tiler, TA const *A,
    AStride dA, ASmemLayout sA_layout,
    TiledCopyA copy_a, TB const *B,
    BStride dB, BSmemLayout sB_layout,
    TiledCopyB copy_b, TC *C, CStride dC,
    CSmemLayout, TiledMma mma,
    Alpha alpha, Beta beta
){
    using namespace cute;

    //declare tensor to global memory
    Tensor mA = make_tensor(make_gmem_ptr(A) , select<0 , 2>(shape_mnk) , dA);
    Tensor mB = make_tensor(make_gmem_ptr(B) , select<1 , 2>(shape_mnk) , dB);
    Tensor mC = make_tensor(make_gmem_ptr(C) , select<0 , 1>(shape_mnk) , dC);


    auto cta_coord = make_coord(blockIdx.x  ,blockIdx.y , _);

    Tensor gA = local_tile(mA , cta_tiler , cta_coord , Step<_1 , X , _1>{});
    Tensor gB = local_tile(mB , cta_tiler , cta_coord , Step<X , _1 , _1>{});
    Tensor gC = local_tile(mC , cta_tiler , cta_coord , Step<_1 , _1 ,X>{});


    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TA smemB[cosize_v<BSmemLayout>];

    //declare tensors to shared memoryu
    Tensor sA = make_tensor(make_smem_ptr(smemA) , sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB) , sB_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);

    Tensor tAgA = thr_copy_a.parition_S(gA);
    Tensor tAsA = thr_copy_a.parition_D(sA);

    Tensor tArA = make_fragment_like(tAsA);


    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    Tensor tBrB = make_fragment_like(tBsB);

    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

        gemm(mma, tCsA, tCsB, tCrC);
    }

    axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const *A, int ldA,
             TB const *B, int ldB, Beta beta, TC *C, int ldC,
             cudaStream_t stream = 0)
{
    using namespace cute;
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_32 , _8>>{},
        Layout<Shape<_4 , _1>>{}
    );
    TiledCopy copyB =
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                        Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                        Layout<Shape<_4, _1>>{}); // Val layout  4x1 n-major

    TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                   Layout<Shape<_16, _16, _1>>{});

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                  sA, copyA, B, dB, sB, copyB,
                                                  C, dC, sC, mmaC, alpha, beta);
}