# StackedMambaMinGRU: Modular Sequence Modeling with minGRU

This repository provides the implementation of **minGRU**, as described in:

**Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadegh, H. (2024). Were RNNs All We Needed?**  
arXiv preprint [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)

---

## Components

### 1. **minGRU**
The described minimal GRU variant with simplified gates and log-space computations for efficiency.  
I only implemented the log-space variant.

### 2. **WrappedMinGRU**
A wrapper that enables `minGRU` to function like PyTorch's GRU, including:
- Multi-layer support.
- Bidirectional processing.
- Compatibility with standard RNN pipelines.
  
This is mostly for comparison with standard models.

### 3. **MambaModule**
A modular block combining:
- Residual connections.
- Two-strand processing.
- Convolutions and normalization.
  
For my application, this yielded the best results.

### 4. **StackedMambaMinGRU**
A stack of `MambaModule` layers for deep sequence modeling, with options for dimension projection.

---

## Additional Files and Optimizations

To enhance both performance and comparability, the following files have been added:

- **compare_kernel_numeric.py**  
  Provides numerical comparisons between different kernel implementations, ensuring that the fused version maintains numerical accuracy.

- **compare_minGRU.py**  
  Compares the performance of **minGRU** with PyTorch’s standard GRU. Benchmarks indicate a **3× speedup** with minGRU on similar workloads.

- **fused_parallel_scan_kernel.cu**  
  Contains a CUDA kernel that implements a fused parallel log-space scan.
  **Note:** This CUDA kernel must be compiled as part of the extension.

- **mingru_stacks_fused.py**  
  Implements a version of `minGRU` that leverages the fused CUDA kernel for the parallel log scan within a stacked architecture. This file demonstrates how to integrate the fused kernel into the sequence modeling pipeline, providing further performance improvements over the original implementation.

---

## Compilation Instructions

The fused CUDA kernel must be compiled before use. To build the extension, follow these steps:

1. Ensure you have PyTorch installed with CUDA support.
2. Navigate to the project directory.
3. Run the following command to compile the kernel:

```bash
python setup.py build_ext --inplace
```

---

## Running Benchmarks

After compilation, you can run performance comparisons with:

```bash
python compare_minGRU.py
python compare_kernel_numeric.py
```

---
Feel free to open an issue or submit a pull request if you find any bugs or want to contribute improvements. Contributions are welcome!



