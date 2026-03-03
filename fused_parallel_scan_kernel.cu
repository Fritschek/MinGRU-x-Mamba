#include <torch/extension.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t logaddexp_device(scalar_t a, scalar_t b) {
    scalar_t m = (a > b) ? a : b;
    return m + log(exp(a - m) + exp(b - m));
}

template <typename scalar_t>
struct LogAffine {
    scalar_t logA;
    scalar_t logB;
};

template <typename scalar_t>
struct ComposeRevOp {
    __device__ __forceinline__ LogAffine<scalar_t> operator()(const LogAffine<scalar_t>& u,
                                                               const LogAffine<scalar_t>& v) const {
        LogAffine<scalar_t> out;
        out.logA = v.logA + u.logA;
        out.logB = logaddexp_device(v.logB, v.logA + u.logB);
        return out;
    }
};

template <typename scalar_t, int BLOCK>
__global__ void fused_parallel_scan_tiled_kernel(
    const scalar_t* __restrict__ log_coeffs,   // (B,T,H)
    const scalar_t* __restrict__ log_values,   // (B,T+1,H): [log_h0, log_b1..log_bT]
    scalar_t* __restrict__ output,             // (B,T,H) in log-domain
    int B,
    int T,
    int H) {
    int r = blockIdx.x;
    int tid = threadIdx.x;

    int64_t R = static_cast<int64_t>(B) * static_cast<int64_t>(H);
    if (r >= R) return;

    int b = r / H;
    int h = r - b * H;

    using BlockScan = cub::BlockScan<LogAffine<scalar_t>, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    __shared__ LogAffine<scalar_t> carry_sh;
    __shared__ LogAffine<scalar_t> tile_last;

    if (tid == 0) {
        carry_sh.logA = scalar_t(0);
        carry_sh.logB = scalar_t(-INFINITY);
    }
    __syncthreads();

    const scalar_t log_h0 = log_values[static_cast<int64_t>(b) * (T + 1) * H + h];

    for (int base = 0; base < T; base += BLOCK) {
        int t = base + tid;

        LogAffine<scalar_t> elem;
        if (t < T) {
            const int64_t coeff_idx = (static_cast<int64_t>(b) * T + t) * H + h;
            const int64_t value_idx = (static_cast<int64_t>(b) * (T + 1) + (t + 1)) * H + h;
            elem.logA = log_coeffs[coeff_idx];
            elem.logB = log_values[value_idx];
        } else {
            elem.logA = scalar_t(0);
            elem.logB = scalar_t(-INFINITY);
        }

        LogAffine<scalar_t> scan_out;
        BlockScan(temp).InclusiveScan(elem, scan_out, ComposeRevOp<scalar_t>());
        __syncthreads();

        LogAffine<scalar_t> carry = carry_sh;

        LogAffine<scalar_t> prefix;
        prefix.logA = scan_out.logA + carry.logA;
        prefix.logB = logaddexp_device(scan_out.logB, scan_out.logA + carry.logB);

        if (t < T) {
            const scalar_t log_h = logaddexp_device(prefix.logA + log_h0, prefix.logB);
            const int64_t out_idx = (static_cast<int64_t>(b) * T + t) * H + h;
            output[out_idx] = log_h;
        }

        int last = min(BLOCK - 1, T - 1 - base);
        if (tid == last) {
            tile_last = prefix;
        }
        __syncthreads();

        if (tid == 0) {
            carry_sh = tile_last;
        }
        __syncthreads();
    }
}

torch::Tensor fused_parallel_scan_cuda(torch::Tensor log_coeffs, torch::Tensor log_values) {
    TORCH_CHECK(log_coeffs.is_cuda() && log_values.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(log_coeffs.is_contiguous() && log_values.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(log_coeffs.dim() == 3 && log_values.dim() == 3, "inputs must be 3D");
    TORCH_CHECK(log_values.size(0) == log_coeffs.size(0), "batch dimension must match");
    TORCH_CHECK(log_values.size(2) == log_coeffs.size(2), "hidden dimension must match");
    TORCH_CHECK(log_values.size(1) == log_coeffs.size(1) + 1, "log_values must have T+1 time steps");

    int B = static_cast<int>(log_coeffs.size(0));
    int T = static_cast<int>(log_coeffs.size(1));
    int H = static_cast<int>(log_coeffs.size(2));

    auto output = torch::empty({B, T, H}, log_coeffs.options());

    const at::cuda::CUDAGuard device_guard(log_coeffs.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    constexpr int BLOCK = 256;
    int64_t R = static_cast<int64_t>(B) * static_cast<int64_t>(H);

    AT_DISPATCH_FLOATING_TYPES(log_coeffs.scalar_type(), "fused_parallel_scan_cuda", [&] {
        fused_parallel_scan_tiled_kernel<scalar_t, BLOCK>
            <<<R, BLOCK, 0, stream>>>(
                log_coeffs.data_ptr<scalar_t>(),
                log_values.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                B, T, H);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_parallel_scan_cuda", &fused_parallel_scan_cuda, "Fused Parallel Scan CUDA Kernel");
}
