from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_parallel_scan',
    ext_modules=[
        CUDAExtension(
            name='fused_parallel_scan',
            sources=['fused_parallel_scan_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
