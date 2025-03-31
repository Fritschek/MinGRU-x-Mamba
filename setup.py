from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["CC"] = "/usr/bin/gcc-11"
os.environ["CXX"] = "/usr/bin/g++-11"

setup(
    name='fused_parallel_scan',
    ext_modules=[
        CUDAExtension(
            name='fused_parallel_scan',
            sources=['fused_parallel_scan_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-std=c++17',
                    '-ccbin=/usr/bin/gcc-11',
                    '-Xcompiler=-fPIC',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-I/usr/local/cuda-12.1/include',  # Explicit CUDA include path
                    '-isystem', '/usr/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

