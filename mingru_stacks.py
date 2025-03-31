import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from numpy import arange
from numpy.random import mtrand
import math
import numpy as np

class StackedMambaMinGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias=True, dropout=0.0, proj_down=True):
        super(StackedMambaMinGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.proj_down = proj_down

        self.layers = nn.ModuleList([
            MambaModule(
                input_size if i == 0 else (input_size if proj_down else hidden_size),  # First layer always input_size, others depend on proj_down
                hidden_size,
                bias=bias,
                dropout=dropout,
                proj_down=proj_down
            )
            for i in range(num_layers)
        ])
        
        # Final downprojection layer after all MambaModules, if proj_down is True
        if self.proj_down:
            self.final_down_projection = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_0=None):
        # x shape: (batch_size, seq_len, input_size)
        #batch_size = x.size(0)

        # Initialize hidden states as None if not provided
        if h_0 is None:
            h_0 = [None] * self.num_layers
            
        # Track updated hidden states across layers
        #updated_hiddens = []

        # Iterate through each layer, passing the last hidden state as input to the next layer
        for idx, layer in enumerate(self.layers):
            # Pass output hidden state of the current layer as the initial hidden state for the next
            x, h_0[idx] = layer(x, h_0=h_0[idx])
            
        return x

class MambaModule(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, proj_down=True):
        super(MambaModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_down = proj_down
        # Linear projections x2 for each strand with bias handling
        # Strand 1
        self.strand1_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.strand1_ln1 = nn.LayerNorm(hidden_size)
        
        # Strand 2
        self.strand2_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.strand2_ln1 = nn.LayerNorm(hidden_size)

        # Activation function
        self.activation = nn.SiLU()

        # Conv1D layer for strand1 with bias handling
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            bias=bias
        )
        self.conv_ln = nn.LayerNorm(hidden_size)

        # minGRU layer for strand1 with bias handling
        self.min_gru = minGRU(hidden_size, hidden_size, bias=bias)
        self.gru_ln = nn.LayerNorm(hidden_size)


        self.down_projection = nn.Linear(hidden_size, input_size, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        self.output_ln = nn.LayerNorm(input_size)

        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):
        # Initialize linear layers
        for layer in [
            self.strand1_proj, 
            self.strand2_proj, 
            self.down_projection
        ]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Initialize Conv1D layer
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize minGRU parameters
        self.min_gru.reset_parameters()

    def forward(self, x, h_0=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        residual = x
        
        # Final LayerNorm for output normalization
          

        # Initialize hidden state differently for training and inference
        if h_0 is None:
            if self.training:
                # For training, hidden state shape is (batch_size, 1, hidden_size)
                h_0 = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
            else:
                # For inference, hidden state shape is (batch_size, hidden_size)
                h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Split into two strands
        strand1 = x
        strand2 = x

        # Strand 1 processing
        strand1 = self.strand1_proj(strand1) 
        strand1 = self.strand1_ln1(strand1)
        

        # Conv1D expects input of shape (batch_size, channels, seq_len)
        strand1 = strand1.permute(0, 2, 1)
        strand1 = self.conv1d(strand1)
        strand1 = strand1.permute(0, 2, 1)
        strand1 = self.conv_ln(strand1)
        
        strand1 = self.activation(strand1)
        #strand1 = self.dropout(strand1)

        # Pass through minGRU
        strand1 = self.min_gru(strand1, h_0)
        strand1 = self.gru_ln(strand1)

        # Strand 2 processing
        strand2 = self.strand2_proj(strand2)
        strand2 = self.strand2_ln1(strand2)
        strand2 = self.activation(strand2)
        #strand2 = self.dropout(strand2)

        # Multiply the two strands element-wise
        x_out = strand1 * strand2
        
        

        
        # Down projection to original input size with bias handling
        if self.proj_down==True:
            x_out = self.down_projection(x_out)
            
        # Add the residual connection and normalize
        x_out = x_out + residual #self.output_ln(x_out + residual)

        return x_out, strand1[:, -1:]

class WrappedMinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, bias=True, batch_first=True):
        super(WrappedMinGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1

        # Create layers
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            if self.bidirectional:
                # For bidirectional, we have forward and backward minGRUs
                layer_module = nn.ModuleDict({
                    'fwd': minGRU(layer_input_size, hidden_size, bias=bias),
                    'bwd': minGRU(layer_input_size, hidden_size, bias=bias)
                })
            else:
                # Unidirectional
                layer_module = minGRU(layer_input_size, hidden_size, bias=bias)
            self.layers.append(layer_module)

    def forward(self, input, h_0=None):
        """
        input: (batch_size, seq_len, input_size) if batch_first=True
               (seq_len, batch_size, input_size) if batch_first=False
        h_0: (num_layers * num_directions, batch_size, hidden_size) or None
        Returns:
        output: (batch_size, seq_len, num_directions * hidden_size) if batch_first=True
                (seq_len, batch_size, num_directions * hidden_size) if batch_first=False
        h_n: (num_layers * num_directions, batch_size, hidden_size)
        """
        # Adjust input shape if batch_first=False
        if not self.batch_first:
            input = input.transpose(0, 1)  # Convert to (batch_size, seq_len, input_size)

        batch_size, seq_len, _ = input.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
        else:
            h_0 = h_0

        # Split h_0 into layers and directions
        h_0 = h_0.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Initialize the output
        layer_output = input
        h_n = []

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            # Get h_0 for this layer
            h_0_layer = h_0[layer_idx]  # Shape: (num_directions, batch_size, hidden_size)
            if self.bidirectional:
                # Bidirectional processing
                h_0_forward = h_0_layer[0]  # Shape: (batch_size, hidden_size)
                h_0_backward = h_0_layer[1]

                # Adjust h_0 shape based on training or inference mode
                if self.training:
                    h_0_forward = h_0_forward.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
                    h_0_backward = h_0_backward.unsqueeze(1)

                # Process forward direction
                forward_layer = self.layers[layer_idx]['fwd']
                h_forward = forward_layer(layer_input, h_0_forward)

                # Process backward direction
                backward_layer = self.layers[layer_idx]['bwd']
                reversed_input = torch.flip(layer_input, dims=[1])
                h_backward = backward_layer(reversed_input, h_0_backward)
                h_backward = torch.flip(h_backward, dims=[1])

                # Concatenate the outputs
                layer_output = torch.cat([h_forward, h_backward], dim=2)  # Shape: (batch_size, seq_len, 2 * hidden_size)

                # Collect the final hidden states
                h_n_forward = h_forward[:, -1, :] if not self.training else h_forward[:, -1, :]
                h_n_backward = h_backward[:, 0, :] if not self.training else h_backward[:, 0, :]
                h_n.append(h_n_forward)
                h_n.append(h_n_backward)
            else:
                # Unidirectional processing
                h_0_layer = h_0_layer.squeeze(0)  # Shape: (batch_size, hidden_size)
                if self.training:
                    h_0_layer = h_0_layer.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
                forward_layer = self.layers[layer_idx]
                h_forward = forward_layer(layer_input, h_0_layer)
                layer_output = h_forward
                h_n_forward = h_forward[:, -1, :] if not self.training else h_forward[:, -1, :]
                h_n.append(h_n_forward)

        # Stack h_n to get (num_layers * num_directions, batch_size, hidden_size)
        h_n = torch.stack(h_n, dim=0)

        # Adjust output shape if batch_first=False
        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)  # Convert back to (seq_len, batch_size, hidden_size)

        return layer_output, h_n

class minGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(minGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Linear layers for gates
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)

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

        def log_g(x): 
            return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
            
                # Initialize h_0 with zeros; adjust shape if necessary

        if h_0 is None:
        # Initialize h_0 with zeros; adjust shape if necessary
            batch_size = x.size(0)
            h_0 = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)

        # Compute k for z gate
        k = self.linear_z(x)  # (batch_size, seq_len, hidden_size)
        log_z = -F.softplus(-k)  # log(z)
        log_coeffs = -F.softplus(k)  # log(1 - z)

        # Compute h_tilde
        log_h_0 = log_g(h_0)  # log(g(h_0))
        log_tilde_h = log_g(self.linear_h(x)) # log(g(h_tilde))

        # Concatenate initial hidden state with inputs
        log_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)  # (batch_size, seq_len + 1, hidden_size)

        # Perform the parallel scan using log-space computations
        h = self.parallel_scan_log(log_coeffs, log_values)  # (batch_size, seq_len, hidden_size)
        return h

    def parallel_scan_log(self, log_coeffs, log_values):
        # log_coeffs: (batch_size, seq_len, hidden_size)
        # log_values: (batch_size, seq_len + 1, hidden_size)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))  # (batch_size, seq_len + 1, hidden_size)
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)  # (batch_size, seq_len + 1, hidden_size)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]
    