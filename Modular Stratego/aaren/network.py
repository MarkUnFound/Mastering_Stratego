# AAREN Network - Piece Action Pattern Learning
# Extracted from probabilistic_belief_state.py for better modularity

"""
PieceActionAaren: AAREN-based network to learn piece value patterns from action sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .cell import AarenCell
from .kernel import aaren_scan_kernel

# Import NUM_PIECE_TYPES from piece module
try:
    from piece import NUM_PIECE_TYPES
except ImportError:
    NUM_PIECE_TYPES = 12  # Fallback default


class PieceActionAaren(nn.Module):
    """
    AAREN-based network to learn piece value patterns from action sequences.
    
    AAREN provides:
    - Parallel training (no sequential bottleneck)
    - Efficient O(1) inference updates
    - Constant memory usage
    - Better gradient flow
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = NUM_PIECE_TYPES, device=None):
        """
        Initialize Aaren for piece value inference.
        
        Args:
            input_size: Size of action feature vector (8 features)
            hidden_size: Hidden state size
            num_layers: Number of Aaren layers
            output_size: Number of possible piece types
            device: PyTorch device
        """
        super(PieceActionAaren, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection to hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Stack of Aaren cells (one per layer)
        self.aaren_cells = nn.ModuleList([
            AarenCell(hidden_size) for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward_parallel(self, x: torch.Tensor):
        """
        Forward pass for training (parallel processing of full sequences).
        
        Uses parallel prefix scan algorithm for efficient parallel computation.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, sequence_length, output_size) with piece type logits
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden size
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Process through each Aaren layer
        for layer_idx, (aaren_cell, layer_norm) in enumerate(zip(self.aaren_cells, self.layer_norms)):
            # Parallel prefix scan for this layer
            h = self._parallel_prefix_scan(h, aaren_cell, layer_norm)
        
        # Use last timestep for prediction
        last_hidden = h[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return logits (softmax applied in loss function)
        return x
    
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
        Inference Mode: O(1) step.
        
        Args:
            x_t: Input tensor of shape (batch, 1, input_size) or (batch, input_size)
            prev_states: Optional list of previous states for each layer
            
        Returns:
            probs: Output probabilities of shape (batch, output_size)
            new_states: List of new states for each layer
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        
        h = self.input_proj(x_t)
        new_states = []
        
        for layer_idx, (aaren_cell, layer_norm) in enumerate(zip(self.aaren_cells, self.layer_norms)):
            # Get state for this specific layer
            prev_state = prev_states[layer_idx] if prev_states is not None else None
            
            # Recurrent Update
            h, new_state = aaren_cell(h, prev_state)
            
            # Apply LayerNorm (Crucial: AAREN output is unnormalized sum ratio)
            h = layer_norm(h)
            new_states.append(new_state)
        
        # Heads
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        
        return probs, new_states
    
    def _parallel_prefix_scan(self, h: torch.Tensor, aaren_cell: AarenCell, layer_norm: nn.LayerNorm):
        """
        Uses the optimized JIT kernel.
        
        Args:
            h: Hidden states of shape (batch, seq_len, hidden_size)
            aaren_cell: AAREN cell for this layer
            layer_norm: Layer normalization for this layer
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # 1. Projections
        k = aaren_cell.W_k(h)
        v = aaren_cell.W_v(h)
        
        # 2. Parallel Scan
        u_final, w_final, _ = aaren_scan_kernel(k, v, aaren_cell.q)
        
        # 3. Output
        output = u_final / (w_final + 1e-8)
        output = layer_norm(output)
        
        return output
