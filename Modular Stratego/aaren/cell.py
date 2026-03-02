# AAREN Cell - O(1) Recurrent Update for Inference
# Extracted from probabilistic_belief_state.py for better modularity

"""
AarenCell implements attention-based recurrent computation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AarenCell(nn.Module):
    """
    Aaren (Attention as a Recurrent Neural Network) Cell.
    
    Implements attention-based recurrent computation with:
    - State: (a_t, c_t, m_t) where:
      - a_t: Weighted sum of encoded values
      - c_t: Normalization constant
      - m_t: Cumulative maximum (for numerical stability)
    - Query vector q: Learned parameter (what to attend to)
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize Aaren cell.
        
        Args:
            hidden_size: Hidden state dimension
        """
        super(AarenCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Learned query vector (what to attend to)
        self.q = nn.Parameter(torch.randn(hidden_size))
        
        # Key and value projections
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x_t: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        O(1) Recurrent Update for Inference.
        
        Args:
            x_t: Input tensor of shape (batch, hidden_size)
            prev_state: Optional tuple of (a_prev, c_prev, m_prev)
            
        Returns:
            output: Output tensor of shape (batch, hidden_size)
            new_state: Tuple of (a_t, c_t, m_t)
        """
        # Compute projections
        k_t = self.W_k(x_t)  # (batch, hidden_size)
        v_t = self.W_v(x_t)  # (batch, hidden_size)
        
        # Attention score s_t = q * k_t
        s_t = torch.sum(self.q.unsqueeze(0) * k_t, dim=1, keepdim=True)  # (batch, 1)
        
        if prev_state is None:
            # Initialization
            m_t = s_t
            a_t = v_t 
            c_t = torch.ones_like(s_t)
        else:
            a_prev, c_prev, m_prev = prev_state
            
            # The exact same associative operator logic as the scan, but sequential
            m_t = torch.maximum(m_prev, s_t)
            
            exp_prev = torch.exp(m_prev - m_t)
            exp_curr = torch.exp(s_t - m_t)
            
            # Update Numerator (a/u) and Denominator (c/w)
            a_t = a_prev * exp_prev + v_t * exp_curr
            c_t = c_prev * exp_prev + exp_curr
        
        # Output = Numerator / Denominator
        output = a_t / (c_t + 1e-8)
        
        return output, (a_t, c_t, m_t)
