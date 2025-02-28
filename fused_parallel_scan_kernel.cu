#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Each thread processes one (batch, hidden) series.
template <typename scalar_t>
__global__ void fused_parallel_scan_kernel(
    const scalar_t* __restrict__ log_coeffs,  // shape: (batch, T, hidden)
    const scalar_t* __restrict__ log_values,   // shape: (batch, T+1, hidden)
    scalar_t* __restrict__ output,             // shape: (batch, T, hidden)
    int T,                                     // sequence length (for log_coeffs and output)
    int hidden,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;
    int batch_idx = idx / hidden;
    int hid_idx = idx % hidden;
    
    // Compute pointer offsets.
    const scalar_t* coeff_ptr = log_coeffs + batch_idx * T * hidden;
    const scalar_t* value_ptr = log_values + batch_idx * (T+1) * hidden;
    scalar_t* out_ptr = output + batch_idx * T * hidden;
    
    // Initialize cumulative sum (a) and the running logsumexp (Y).
    scalar_t a = 0;
    scalar_t Y = -INFINITY;
    
    // Loop over t = 0 to T (inclusive). For t==0, a is zero.
    for (int t = 0; t <= T; t++) {
        if (t > 0) {
            // a[t] = a[t-1] + log_coeffs[t-1]
            scalar_t coeff = coeff_ptr[(t - 1) * hidden + hid_idx];
            a += coeff;
        }
        // x = log_values[t] - a
        scalar_t x = value_ptr[t * hidden + hid_idx] - a;
        // Update Y = log(exp(Y) + exp(x))
        if (Y == -INFINITY) {
            Y = x;
        } else {
            scalar_t m = fmaxf(Y, x);
            Y = m + logf(expf(Y - m) + expf(x - m));
        }
        scalar_t log_h = a + Y;
        if (t > 0) {
            out_ptr[(t - 1) * hidden + hid_idx] = expf(log_h);
        }
    }
}

torch::Tensor fused_parallel_scan_cuda(torch::Tensor log_coeffs, torch::Tensor log_values) {
    // log_coeffs: (batch, T, hidden)
    // log_values: (batch, T+1, hidden)
    auto batch = log_coeffs.size(0);
    auto T = log_coeffs.size(1);
    auto hidden = log_coeffs.size(2);
    auto output = torch::empty({batch, T, hidden}, log_coeffs.options());
    
    int total = batch * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(log_coeffs.scalar_type(), "fused_parallel_scan_cuda", ([&] {
        fused_parallel_scan_kernel<scalar_t><<<blocks, threads>>>(
            log_coeffs.data_ptr<scalar_t>(),
            log_values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            T,
            hidden,
            batch
        );
    }));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_parallel_scan_cuda", &fused_parallel_scan_cuda, "Fused Parallel Scan CUDA Kernel");
}
