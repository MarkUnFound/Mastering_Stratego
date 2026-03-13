# LSTM Network - Piece Action Pattern Learning
# Drop-in replacement for AAREN (PieceActionAaren)

"""
PieceActionLSTM: Standard LSTM-based network to learn piece value patterns
from action sequences. Replaces AAREN with a simpler recurrent architecture
while maintaining the same interface for HistoryAggregator compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Import NUM_PIECE_TYPES from piece module
try:
    from piece import NUM_PIECE_TYPES
except ImportError:
    NUM_PIECE_TYPES = 12  # Fallback default


class PieceActionLSTM(nn.Module):
    """
    LSTM-based network to learn piece value patterns from action sequences.
    
    Drop-in replacement for PieceActionAaren, providing:
    - forward_parallel(x)    → Parallel training on full sequences
    - forward_sequential(x_t, prev_states) → O(1) inference step
    - forward_embedding(x)   → Hidden state embedding for DQN
    - forward(x)             → Full sequence → softmax probabilities
    
    Unlike AAREN's attention-based mechanism, this uses standard LSTM
    gating (forget, input, output gates) for sequential modeling.
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = NUM_PIECE_TYPES, device=None):
        """
        Initialize LSTM for piece value inference.
        
        Args:
            input_size: Size of action feature vector
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            output_size: Number of possible piece types
            device: PyTorch device
        """
        super(PieceActionLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection to hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability (applied to LSTM output)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers (classification head for piece type prediction)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward_parallel(self, x: torch.Tensor):
        """
        Forward pass for training (parallel processing of full sequences).
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size) with piece type logits
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden size
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Process through LSTM (processes full sequence in parallel via cuDNN)
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_size)
        
        # Apply layer norm to last timestep
        last_hidden = self.layer_norm(lstm_out[:, -1, :])  # (batch, hidden_size)
        
        # Classification head
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return logits (softmax applied in loss function)
        return x
    
    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return hidden state embedding instead of classification logits.
        
        Used at replay time to produce differentiable LSTM embeddings from
        stored action histories, enabling end-to-end gradient flow from
        the DQN loss through the LSTM.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch, hidden_size) — last timestep hidden state
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden size
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_size)
        
        # Return last timestep hidden state as embedding (no classification head)
        return self.layer_norm(lstm_out[:, -1, :])  # (batch, hidden_size)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass (defaults to parallel mode for training).
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size) with piece type probabilities
        """
        logits = self.forward_parallel(x)
        return F.softmax(logits, dim=1)
    
    def forward_sequential(self, x_t: torch.Tensor, prev_states: Optional[List[Tuple]] = None):
        """
        Inference Mode: O(1) step update.
        
        Processes a single timestep using the LSTM's recurrent hidden state.
        
        Args:
            x_t: Input tensor of shape (batch, 1, input_size) or (batch, input_size)
            prev_states: Optional list of (h, c) tuples for each layer.
                         Each h and c has shape (1, batch, hidden_size).
                         If None, LSTM initializes with zeros.
            
        Returns:
            probs: Output probabilities of shape (batch, output_size)
            new_states: List of (h, c) tuples for each layer
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # (batch, 1, input_size)
        
        batch_size = x_t.size(0)
        
        # Project input
        h = self.input_proj(x_t)  # (batch, 1, hidden_size)
        
        # Convert prev_states list-of-tuples to LSTM's expected format
        # LSTM expects: h_0 of shape (num_layers, batch, hidden_size)
        #               c_0 of shape (num_layers, batch, hidden_size)
        if prev_states is not None:
            # prev_states is a list of (h, c) tuples, one per layer
            h_list = [s[0] for s in prev_states]  # Each (1, batch, hidden_size)
            c_list = [s[1] for s in prev_states]  # Each (1, batch, hidden_size)
            h_0 = torch.cat(h_list, dim=0)  # (num_layers, batch, hidden_size)
            c_0 = torch.cat(c_list, dim=0)  # (num_layers, batch, hidden_size)
            hx = (h_0, c_0)
        else:
            hx = None  # LSTM will initialize with zeros
        
        # Single-step LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(h, hx)  # lstm_out: (batch, 1, hidden_size)
        
        # Apply layer norm
        output = self.layer_norm(lstm_out.squeeze(1))  # (batch, hidden_size)
        
        # Classification head
        x = F.relu(self.fc1(output))
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        
        # Convert hidden states back to list-of-tuples format
        # h_n, c_n shape: (num_layers, batch, hidden_size)
        new_states = []
        for layer_idx in range(self.num_layers):
            new_states.append((
                h_n[layer_idx:layer_idx+1],  # (1, batch, hidden_size)
                c_n[layer_idx:layer_idx+1]   # (1, batch, hidden_size)
            ))
        
        return probs, new_states
