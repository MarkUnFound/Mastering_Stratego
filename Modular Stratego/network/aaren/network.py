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
        
    def _last_step(self, h: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """Hidden state at each sequence's last real step.

        Sequences are padded at the end; the scan is causal, so padding after a sequence's last
        step cannot change it. Without `lengths` every sequence is taken to be full length.
        (The thesis code always took h[:, -1], i.e. the padding step of every shorter sequence.)
        """
        if lengths is None:
            return h[:, -1, :]
        lengths = torch.as_tensor(lengths, device=h.device, dtype=torch.long).clamp(min=1)
        rows = torch.arange(h.size(0), device=h.device)
        return h[rows, lengths - 1, :]

    def encode(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(batch, seq_len, input_size) end-padded sequences -> (batch, hidden_size) embeddings."""
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        for aaren_cell, layer_norm in zip(self.aaren_cells, self.layer_norms):
            h = self._parallel_prefix_scan(h, aaren_cell, layer_norm)
        return self._last_step(h, lengths)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Piece-type logits from embeddings (dropout only in training mode)."""
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        return self.fc2(x)

    def forward_parallel(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass for training (parallel processing of full sequences).
        
        Uses parallel prefix scan algorithm for efficient parallel computation.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size), padded at the end
            lengths: Real length of each sequence (default: all full length)
            
        Returns:
            (batch, output_size) piece type logits at each sequence's last real step
        """
        return self.head(self.encode(x, lengths))
    
    def forward_embedding(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Hidden state embedding (no classification head) at each sequence's last real step.
        Equal to the embedding the HistoryAggregator builds move by move with step().
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size), padded at the end
            lengths: Real length of each sequence (default: all full length)
            
        Returns:
            Tensor of shape (batch, hidden_size)
        """
        return self.encode(x, lengths)
    
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
    
    def step(self, x_t: torch.Tensor, prev_states: Optional[List[Tuple]] = None):
        """
        Inference mode: O(1) recurrent update.

        Args:
            x_t: (batch, input_size) or (batch, 1, input_size) features of one new action
            prev_states: per-layer recurrent states from the previous step (None at the start)

        Returns:
            h: (batch, hidden_size) embedding after this action (same as encode() on the
               whole sequence)
            new_states: per-layer recurrent states
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        h = self.input_proj(x_t)
        new_states = []
        for layer_idx, (aaren_cell, layer_norm) in enumerate(zip(self.aaren_cells, self.layer_norms)):
            prev_state = prev_states[layer_idx] if prev_states is not None else None
            h, new_state = aaren_cell(h, prev_state)
            h = layer_norm(h)  # AAREN output is an unnormalised ratio
            new_states.append(new_state)
        return h, new_states

    def forward_sequential(self, x_t: torch.Tensor, prev_states: Optional[List[Tuple]] = None):
        """
        Inference Mode: O(1) step that also applies the classification head.

        Returns:
            probs: Output probabilities of shape (batch, output_size)
            new_states: List of new states for each layer
        """
        h, new_states = self.step(x_t, prev_states)
        probs = F.softmax(self.head(h), dim=1)
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
