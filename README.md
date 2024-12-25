# StackedMambaMinGRU: Modular Sequence Modeling with minGRU

This repository provides the implementation of **minGRU**, as described in:

**Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadegh, H. (2024). Were RNNs All We Needed?**  
arXiv preprint [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)

---

## Components

### 1. **minGRU**
The described minimal GRU variant with simplified gates and log-space computations for efficiency.
I only implemented the log space variant.

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


