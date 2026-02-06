"""
AAREN: Attention as a Recurrent Neural Network
Single-layer implementation with RMSNorm for improved gradient flow.

Reference: AAREN formulation for implicit PBS (Probabilistic Belief State)

Key differences from standard LSTM:
- O(log N) parallel training via prefix scan
- O(1) recurrent inference
- RMSNorm (NOT LayerNorm) for better RNN gradient flow
- Single layer only (memory constraint)
- 64-dim output strictly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    Unlike LayerNorm, RMSNorm does not center the activations,
    which provides better gradient flow for recurrent architectures.
    
    Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class AARENCell(nn.Module):
    """
    AAREN Cell: O(1) Recurrent Update for Inference
    
    State: (a_t, c_t, m_t) where:
        - a_t: Weighted sum of encoded values (numerator)
        - c_t: Normalization constant (denominator)
        - m_t: Cumulative maximum (numerical stability)
    
    Query: Learned parameter q (what to attend to)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Learned query vector
        self.q = nn.Parameter(torch.randn(hidden_size) * 0.02)
        
        # Key and value projections (no bias for memory efficiency)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.1)
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        O(1) Recurrent Update for Inference.
        
        Args:
            x_t: Input tensor (batch, hidden_size)
            prev_state: Optional (a_prev, c_prev, m_prev)
            
        Returns:
            output: (batch, hidden_size)
            new_state: (a_t, c_t, m_t)
        """
        # Projections
        k_t = self.W_k(x_t)  # (batch, hidden_size)
        v_t = self.W_v(x_t)  # (batch, hidden_size)
        
        # Attention score: s_t = q · k_t
        s_t = torch.sum(self.q.unsqueeze(0) * k_t, dim=1, keepdim=True)  # (batch, 1)
        
        if prev_state is None:
            # Initialization
            m_t = s_t
            a_t = v_t
            c_t = torch.ones_like(s_t)
        else:
            a_prev, c_prev, m_prev = prev_state
            
            # Associative operator (matches parallel scan logic)
            m_t = torch.maximum(m_prev, s_t)
            
            exp_prev = torch.exp(m_prev - m_t)
            exp_curr = torch.exp(s_t - m_t)
            
            # Update numerator and denominator
            a_t = a_prev * exp_prev + v_t * exp_curr
            c_t = c_prev * exp_prev + exp_curr
        
        # Output = Numerator / Denominator
        output = a_t / (c_t + 1e-8)
        
        return output, (a_t, c_t, m_t)


@torch.jit.script
def aaren_parallel_scan(
    k: torch.Tensor, 
    v: torch.Tensor, 
    q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT-compiled Hillis-Steele Parallel Prefix Scan for AAREN.
    
    Args:
        k: Keys (batch, seq_len, hidden)
        v: Values (batch, seq_len, hidden)
        q: Query vector (hidden,)
    
    Returns:
        u_final: Numerator sums (batch, seq_len, hidden)
        w_final: Denominator sums (batch, seq_len, 1)
        m_final: Cumulative max (batch, seq_len, 1)
    """
    batch_size, seq_len, hidden_size = k.size()
    
    # Attention scores: s = q · k
    s = (k * q).sum(dim=2, keepdim=True)  # (batch, seq, 1)
    
    # Initialize leaf nodes
    curr_m = s
    curr_u = v
    curr_w = torch.ones_like(s)
    
    # Padding value for -inf
    pad_m_val = -1e9
    
    # Hillis-Steele: ceil(log2(L)) steps
    num_steps = int(math.ceil(math.log(float(seq_len)) / math.log(2.0)))
    
    for i in range(num_steps):
        offset = 1 << i  # 2^i
        
        # Shift and pad
        pad_m = torch.full((batch_size, offset, 1), pad_m_val, device=curr_m.device, dtype=curr_m.dtype)
        prev_m = torch.cat((pad_m, curr_m[:, :-offset, :]), dim=1)
        
        pad_u = torch.zeros((batch_size, offset, hidden_size), device=curr_u.device, dtype=curr_u.dtype)
        prev_u = torch.cat((pad_u, curr_u[:, :-offset, :]), dim=1)
        
        pad_w = torch.zeros((batch_size, offset, 1), device=curr_w.device, dtype=curr_w.dtype)
        prev_w = torch.cat((pad_w, curr_w[:, :-offset, :]), dim=1)
        
        # Associative operator
        m_new = torch.maximum(prev_m, curr_m)
        exp_prev = torch.exp(prev_m - m_new)
        exp_curr = torch.exp(curr_m - m_new)
        
        curr_u = prev_u * exp_prev + curr_u * exp_curr
        curr_w = prev_w * exp_prev + curr_w * exp_curr
        curr_m = m_new
    
    return curr_u, curr_w, curr_m


class AAREN(nn.Module):
    """
    AAREN: Attention-based Action-Relational Episodic Network
    
    Single-layer implementation for memory-constrained training.
    Output: 64-dim latent history embedding
    
    Architecture constraints:
    - Single layer (NOT multi-layer)
    - RMSNorm (NOT LayerNorm)
    - 64-dim output strictly
    - NO dropout (save memory)
    - Trained end-to-end via DQN loss only
    """
    
    HIDDEN_SIZE = 64  # Fixed per spec
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        """
        Initialize AAREN.
        
        Args:
            input_dim: Size of input feature vector (e.g., 15*10*10 for flattened board)
            hidden_dim: Hidden state size (must be 64 per spec)
            device: PyTorch device
        """
        super().__init__()
        
        if hidden_dim != 64:
            print(f"[WARNING] AAREN hidden_dim forced to 64 (was {hidden_dim})")
            hidden_dim = 64
        
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Single AAREN cell (NOT multiple layers)
        self.aaren_cell = AARENCell(hidden_dim)
        
        # RMSNorm (NOT LayerNorm - crucial for RNN gradient flow)
        self.rms_norm = RMSNorm(hidden_dim)
        
        # Output projection (maintains 64-dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable default embedding for when no action history is available
        # This prevents all-zeros output that triggers sparse death detection
        self.default_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # No dropout (memory constraint)
        
        # Initialize for stability
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)
    
    def get_default_embedding(self, batch_size: int) -> torch.Tensor:
        """
        Get the learnable default embedding for when no action history is available.
        
        Args:
            batch_size: Number of samples in batch
            
        Returns:
            embedding: (batch_size, 64) default embeddings
        """
        return self.default_embedding.unsqueeze(0).expand(batch_size, -1)
    
    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (parallel O(log N)).
        
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            
        Returns:
            embedding: 64-dim history embedding (batch, 64)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, 64)
        
        # Compute keys and values
        k = self.aaren_cell.W_k(h)
        v = self.aaren_cell.W_v(h)
        
        # Parallel prefix scan
        u_final, w_final, _ = aaren_parallel_scan(k, v, self.aaren_cell.q)
        
        # Output = Numerator / Denominator
        output = u_final / (w_final + 1e-8)
        
        # RMSNorm
        output = self.rms_norm(output)
        
        # Use last timestep
        last_hidden = output[:, -1, :]  # (batch, 64)
        
        # Output projection
        embedding = self.output_proj(last_hidden)
        
        return embedding
    
    def forward_sequential(
        self, 
        x_t: torch.Tensor, 
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for inference (recurrent O(1)).
        
        Args:
            x_t: Single input (batch, input_dim) or (batch, 1, input_dim)
            prev_state: Previous AAREN state
            
        Returns:
            embedding: 64-dim vector (batch, 64)
            new_state: Updated state tuple
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        
        # Project input
        h = self.input_proj(x_t)  # (batch, 64)
        
        # Recurrent update
        output, new_state = self.aaren_cell(h, prev_state)
        
        # RMSNorm
        output = self.rms_norm(output)
        
        # Output projection
        embedding = self.output_proj(output)
        
        return embedding, new_state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward (parallel mode for training).
        
        Args:
            x: Input (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            embedding: 64-dim vector (batch, 64)
        """
        if x.dim() == 2:
            # Single timestep - add sequence dimension
            x = x.unsqueeze(1)
        
        return self.forward_parallel(x)
    
    def get_initial_state(self, batch_size: int) -> None:
        """
        Get initial AAREN state (None for fresh start).
        """
        return None
    
    def reset_state(self):
        """Reset internal state for new episode."""
        pass  # State is passed explicitly, nothing to reset internally
