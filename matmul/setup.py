from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='cuda_kernel',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'cuda_kernel',
            ['binding.cpp', 'main.cu'],
                                         add             
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)