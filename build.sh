#!/bin/bash

# Clean old builds (optional)
echo " Cleaning previous builds..."
rm -rf build fused_parallel_scan*.so

# Build the extension
echo " Building CUDA extension..."
python setup.py build_ext --inplace

