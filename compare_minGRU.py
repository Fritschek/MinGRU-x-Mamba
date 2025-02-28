import time  # Import time module for measuring execution time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

import fused_parallel_scan

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class minGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(minGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Linear layers for gates
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)

        self.linear_fused = nn.Linear(input_size, 2 * hidden_size, bias=bias)

        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Custom weight initialization for the GRU.
        """
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.linear_z.weight)
        nn.init.xavier_uniform_(self.linear_h.weight)

        # Zero initialization for biases
        if self.bias:
            nn.init.zeros_(self.linear_z.bias)
            nn.init.zeros_(self.linear_h.bias)

    def forward(self, x, h_0=None):
        if self.training:
            return self.forward_training(x, h_0)
        else:
            return self.forward_sequence_inference(x, h_0)
        
    def forward_sequence_inference(self, x, h_0):
        """
        Parameters:
        - x: (batch_size, seq_len, input_size) The input sequence.
        - h_0: (batch_size, hidden_size) The initial hidden state.

        Returns:
        - h_all: (batch_size, seq_len, hidden_size) The hidden states for the entire sequence.
        """
        _, seq_len, _ = x.size()
        h_all = []  # List to hold all hidden states
        
        def g(x): 
            return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

        # Precompute linear transformations for the entire sequence in one go
        z_t = torch.sigmoid(self.linear_z(x))  # (batch_size, seq_len, hidden_size)
        h_tilde_t = g(self.linear_h(x))   # (batch_size, seq_len, hidden_size)
        
        # Initialize the hidden state
        h_prev = g(h_0)

        # Vectorized computation through the sequence
        for t in range(seq_len):
            h_prev = (1 - z_t[:, t, :]) * h_prev + z_t[:, t, :] * h_tilde_t[:, t, :]
            h_all.append(h_prev.unsqueeze(1))  # Add sequence dimension back

        h_all = torch.cat(h_all, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return h_all

    def forward_training(self, x, h_0=None):
        # x: (batch_size, seq_len, input_size)
        if h_0 is None:
        # Initialize h_0 with zeros; adjust shape if necessary
            h_0 = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)

        #@torch.jit.script
        def log_g(x): 
            return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
        
        fused = self.linear_fused(x)
        k, h_candidate = torch.split(fused, self.hidden_size, dim=-1)
        # Compute k for z gate
        # k = self.linear_z(x)  # (batch_size, seq_len, hidden_size)
        log_z = -F.softplus(-k)  # log(z)
        log_coeffs = -F.softplus(k)  # log(1 - z)

        # Compute h_tilde
        log_h_0 = log_g(h_0)  # log(g(h_0))
        log_tilde_h = log_g(h_candidate)#self.linear_h(x)) # log(g(h_tilde))

        # Concatenate initial hidden state with inputs
        log_z.add_(log_tilde_h)
        log_values = torch.cat([log_h_0, log_z], dim=1) # (batch_size, seq_len + 1, hidden_size)
        #log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)  

        # Perform the parallel scan using log-space computations
        h = self.fused_parallel_scan_fn(log_coeffs, log_values)  # (batch_size, seq_len, hidden_size)
        return h

    #def parallel_scan_log(self, log_coeffs, log_values):
    #    # log_coeffs: (batch_size, seq_len, hidden_size)
    #    # log_values: (batch_size, seq_len + 1, hidden_size)
    #    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))  # (batch_size, seq_len + 1, hidden_size)
    #    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)  # (batch_size, seq_len + 1, hidden_size)
    #    log_h = a_star + log_h0_plus_b_star
    #    return torch.exp(log_h)[:, 1:]

    def fused_parallel_scan_fn(self,log_coeffs, log_values):
        return fused_parallel_scan.fused_parallel_scan_cuda(log_coeffs, log_values)


# Standard GRU Model
class StandardGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True):
        super(StandardGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False
        )

    def forward(self, x, h_0=None):
        output, h_n = self.gru(x, h_0)
        return output

# Profiling Function with Total Timing
def profile_model(model, x, model_name="Model"):
    model.to(device)
    model.train()  # Ensure the model is in training mode

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Warm-up iterations
    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        logits = output[:, -1, :]  # Use the last time step
        fc = nn.Linear(logits.size(-1), 10).to(device)
        logits = fc(logits)
        labels = torch.randint(0, 10, (x.size(0),)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Start timing
    start_time = time.time()

    # Start profiling
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            logits = output[:, -1, :]
            logits = fc(logits)
            labels = torch.randint(0, 10, (x.size(0),)).to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print the total execution time
    print(f"\nTotal execution time for {model_name}: {total_time:.4f} seconds")

    # Print the profiling results
    print(f"\nProfiling results for {model_name}:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

def generate_sample_data(batch_size, seq_len, input_size):
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 10, (batch_size,))
    return x, y

input_size = 10
hidden_size = 100
seq_len = 1000
batch_size = 128

x, y = generate_sample_data(batch_size, seq_len, input_size)
x = x.to(device)
y = y.to(device)

# Instantiate MinGRU Model
min_gru_model = minGRU(
    input_size=input_size,
    hidden_size=hidden_size
)

# Instantiate Standard GRU Model
standard_gru_model = StandardGRUModel(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    bias=True,
    batch_first=True
)

# Run Profiling for MinGRU Model
profile_model(min_gru_model, x, model_name="MinGRU Model")

# Run Profiling for Standard GRU Model
profile_model(standard_gru_model, x, model_name="Standard GRU Model")

