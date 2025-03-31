import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import fused_parallel_scan as fused_parallel_scan

# Import the non-fused minGRU
from mingru_stacks import minGRU as MinGRUImported
#from mingru_stacks_test import minGRU as MinGRUImported_test

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom fused minGRU
class MinGRUFused(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MinGRUFused, self).__init__()
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
        if h_0 is None:
            batch_size = x.size(0)
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
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
            batch_size = x.size(0)
            h_0 = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)

        #@torch.jit.script
        def log_g(x): 
            return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
        
        #fused = self.linear_fused(x)
        #k, h_candidate = torch.split(fused, self.hidden_size, dim=-1)
        # Compute k for z gate
        # k = self.linear_z(x)  # (batch_size, seq_len, hidden_size)
        #log_z = -F.softplus(-k)  # log(z)
        #log_coeffs = -F.softplus(k)  # log(1 - z)
        # Compute h_tilde
        #log_h_0 = log_g(h_0)  # log(g(h_0))
        #log_tilde_h = log_g(h_candidate)#self.linear_h(x)) # log(g(h_tilde))
        # Compute k for z gate
        k = self.linear_z(x)  # (batch_size, seq_len, hidden_size)
        log_z = -F.softplus(-k)  # log(z)
        log_coeffs = -F.softplus(k)  # log(1 - z)        
        
        

        # Compute h_tilde
        log_h_0 = log_g(h_0)  # log(g(h_0))
        log_tilde_h = log_g(self.linear_h(x)) # log(g(h_tilde))



        # Concatenate initial hidden state with inputs
        #log_z.add_(log_tilde_h)
        #log_values = torch.cat([log_h_0, log_z], dim=1) # (batch_size, seq_len + 1, hidden_size)
        log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)  

        # Perform the parallel scan using log-space computations
        h = self.fused_parallel_scan_fn(log_coeffs, log_values)  # (batch_size, seq_len, hidden_size)
        #h = self.parallel_scan_log(log_coeffs, log_values)  # (batch_size, seq_len, hidden_size)
        return torch.exp(h)


    def fused_parallel_scan_fn(self,log_coeffs, log_values):
        return fused_parallel_scan.fused_parallel_scan_cuda(log_coeffs, log_values)


# Standard GRU Model
class StandardGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(StandardGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x, h_0=None):
        output, h_n = self.gru(x, h_0)
        return output


# Profiling Function
def profile_model(model, x, model_name="Model"):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Warm-up iterations
    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        logits = output[:, -1, :]
        fc = nn.Linear(logits.size(-1), 10).to(device)
        logits = fc(logits)
        labels = torch.randint(0, 10, (x.size(0),)).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Start timing
    start_time = time.time()

    # Profiling loop
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

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal execution time for {model_name}: {total_time:.4f} seconds")
    print(f"\nProfiling results for {model_name}:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))


# Inference Profiling
def profile_inference(model, x, model_name="Model"):
    model.to(device)
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(50):
            output = model(x)
        end_time = time.time()

    print(f"Inference time for {model_name}: {end_time - start_time:.4f} seconds")


# Generate sample data
def generate_sample_data(batch_size, seq_len, input_size):
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


# Test Configuration
input_size = 10
hidden_size = 100
seq_len = 1000
batch_size = 128

x, y = generate_sample_data(batch_size, seq_len, input_size)
x = x.to(device)
y = y.to(device)

# Instantiate Models
min_gru_imported = MinGRUImported(input_size=input_size, hidden_size=hidden_size)
min_gru_fused = MinGRUFused(input_size=input_size, hidden_size=hidden_size)
standard_gru = StandardGRUModel(input_size=input_size, hidden_size=hidden_size)

# Run Profiling
print("\n------ Training Profiling ------")
profile_model(min_gru_imported, x, model_name="Imported MinGRU")
profile_model(min_gru_fused, x, model_name="Fused MinGRU")
profile_model(standard_gru, x, model_name="Standard GRU")

print("\n------ Inference Profiling ------")
profile_inference(min_gru_imported, x, model_name="Imported MinGRU")
profile_inference(min_gru_fused, x, model_name="Fused MinGRU")
profile_inference(standard_gru, x, model_name="Standard GRU")

# Function to compare outputs of two models
def compare_outputs(output1, output2, model1_name, model2_name, tolerance=1e-5):
    is_close = torch.allclose(output1, output2, atol=tolerance)
    mse = torch.mean((output1 - output2) ** 2).item()
    max_diff = torch.max(torch.abs(output1 - output2)).item()

    print(f"\nComparing {model1_name} vs. {model2_name}:")
    print(f"  All close within tolerance {tolerance}? {'Yes' if is_close else 'No'}")
    print(f"  Mean Squared Error (MSE): {mse:.6e}")
    print(f"  Max Absolute Difference: {max_diff:.6e}")

    return is_close

# Function to test all three models and compare results
def test_and_compare_models(models, x):
    results = {}

    for model_name, model in models.items():
        model.to(device)

        print(f"\n================== Testing {model_name} ==================")

        # Training Mode Output
        model.train()
        with torch.no_grad():
            train_output = model(x).detach()
        
        # Inference Mode Output
        model.eval()
        with torch.no_grad():
            inference_output = model(x).detach()

        # Store results
        results[model_name] = {"train": train_output, "inference": inference_output}

        # Compare Training and Inference Outputs for the Same Model
        compare_outputs(train_output, inference_output, f"{model_name} (Train)", f"{model_name} (Inference)")

    # Compare Fused vs. Non-Fused MinGRU
    compare_outputs(
        results["Imported MinGRU"]["inference"],
        results["Fused MinGRU"]["inference"],
        "Imported MinGRU",
        "Fused MinGRU"
    )

    compare_outputs(
        results["Imported MinGRU"]["train"],
        results["Fused MinGRU"]["train"],
        "Imported MinGRU (Train)",
        "Fused MinGRU (Train)"
    )


# Generate a test input batch
batch_size = 500
seq_len = 100
input_size = 10

x_test = torch.randn(batch_size, seq_len, input_size, device=device)
# Instantiate Models
min_gru_imported = MinGRUImported(input_size=input_size, hidden_size=100)
min_gru_fused = MinGRUFused(input_size=input_size, hidden_size=100)
standard_gru = StandardGRUModel(input_size=input_size, hidden_size=100)

min_gru_fused.linear_z.weight.data.copy_(min_gru_imported.linear_z.weight.data)
min_gru_fused.linear_h.weight.data.copy_(min_gru_imported.linear_h.weight.data)
min_gru_fused.linear_z.bias.data.copy_(min_gru_imported.linear_z.bias.data)
min_gru_fused.linear_h.bias.data.copy_(min_gru_imported.linear_h.bias.data)
# Instantiate Models
models = {
    "Imported MinGRU": min_gru_imported,  # Use the already created instance
    "Fused MinGRU": min_gru_fused,          # Use the model with copied weights
    "Standard GRU": standard_gru,         # No change needed
}
# Run Comparison


test_and_compare_models(models, x_test)

