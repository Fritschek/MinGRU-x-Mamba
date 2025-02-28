import torch
import torch.nn.functional as F
import fused_parallel_scan  # the compiled extension

def fused_parallel_scan_fn(log_coeffs, log_values):
    return fused_parallel_scan.fused_parallel_scan_cuda(log_coeffs, log_values)

def reference_parallel_scan(log_coeffs, log_values):
    # log_coeffs: (batch, T, hidden)
    # log_values: (batch, T+1, hidden)
    a_star = torch.cumsum(log_coeffs, dim=1)
    a_star = F.pad(a_star, (0, 0, 1, 0)) 
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

def main():
    torch.manual_seed(42)
    
    batch = 20
    T = 1000
    hidden = 40

    log_coeffs = torch.randn(batch, T, hidden, device='cuda')
    log_values = torch.randn(batch, T+1, hidden, device='cuda')
    
    out_fused = torch.log(fused_parallel_scan_fn(log_coeffs, log_values))
    out_ref   = torch.log(reference_parallel_scan(log_coeffs, log_values))
    
    print("Fused output:")
    print(out_fused)
    print("\nReference output:")
    print(out_ref)
    
    # Compare using allclose with a tolerance.
    if torch.allclose(out_fused, out_ref, atol=1e-5, rtol=1e-5):
        print("\nOutputs are close!")
    else:
        diff = torch.abs(out_fused - out_ref).max()
        print("\nMaximum absolute difference:", diff.item())


if __name__ == '__main__':
    main()
