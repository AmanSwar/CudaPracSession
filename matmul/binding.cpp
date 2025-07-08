#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "ATen/ops/zeros_like.h"
#include "main.h"
#include <torch/extension.h>




torch::Tensor matmul_naive(
    torch::Tensor matrixA,
    torch::Tensor matrixB
){
    TORCH_CHECK(matrixA.is_cuda() , "matrixA is not in CUDA");
    TORCH_CHECK(matrixB.is_cuda() , "matrixA is not in CUDA");

    auto output = torch::zeros((M , N));

    launch_naive_matmul(matrixA.data_ptr<float>(), matrixB.data_ptr<float>(), output.data_ptr<float>());
    return output;
}


PYBIND11_MODULE(CUDA_KERNEL , m){
    m.def("matmul_naive" , &matmul_naive , "naive matmul");
}