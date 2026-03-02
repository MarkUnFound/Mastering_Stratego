# AAREN Module - Attention as a Recurrent Neural Network
# Extracted from probabilistic_belief_state.py for better modularity

"""
JIT-compiled AAREN scan kernel implementing Hillis-Steele Parallel Prefix Scan.
"""

import torch
import math
from typing import Tuple


@torch.jit.script
def aaren_scan_kernel(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT-compiled implementation of the Hillis-Steele Parallel Prefix Scan 
    using the associative operator defined in the AAREN paper.
    
    Tuples are (m, u, w) where:
    m: Cumulative max (for stability)
    u: Numerator sum (weighted values) -> corresponds to 'w' in paper notation
    w: Denominator sum (normalization) -> corresponds to 'u' in paper notation
    
    Input:
      k: (batch, seq_len, hidden) - Keys
      v: (batch, seq_len, hidden) - Values
      q: (hidden) - Learned Query vector
    """
    batch_size, seq_len, hidden_size = k.size()
    
    # 1. Compute Scores s = q * k
    # s corresponds to the attention logits before softmax
    s = (k * q).sum(dim=2, keepdim=True)  # (Batch, Seq, 1)

    # 2. Initialize the tuple (m, u, w) at leaf nodes
    # m_i = s_i
    # u_i = v_i * 1 (Since exp(s_i - m_i) = exp(0) = 1)
    # w_i = 1 
    curr_m = s
    curr_u = v
    curr_w = torch.ones_like(s)

    # Padding identities for the associative operator
    pad_m_val = -1e9  # Effectively -inf
    
    # Calculate steps for Hillis-Steele: ceil(log2(L))
    num_steps = int(math.ceil(math.log(float(seq_len)) / math.log(2.0)))
    
    for i in range(num_steps):
        offset = 1 << i  # 2^i
        
        # Shift and Pad
        # Create 'previous' tensors by shifting right by 'offset'
        pad_m = torch.full((batch_size, offset, 1), pad_m_val, device=curr_m.device, dtype=curr_m.dtype)
        prev_m = torch.cat((pad_m, curr_m[:, :-offset, :]), dim=1)

        pad_u = torch.zeros((batch_size, offset, hidden_size), device=curr_u.device, dtype=curr_u.dtype)
        prev_u = torch.cat((pad_u, curr_u[:, :-offset, :]), dim=1)

        pad_w = torch.zeros((batch_size, offset, 1), device=curr_w.device, dtype=curr_w.dtype)
        prev_w = torch.cat((pad_w, curr_w[:, :-offset, :]), dim=1)

        # --- The Associative Operator (Paper Eq in Appendix B) ---
        # m_new = max(m_prev, m_curr)
        m_new = torch.maximum(prev_m, curr_m)
        
        # Stability factors: exp(m_prev - m_new) and exp(m_curr - m_new)
        # One of these will always be exp(0) = 1, the other <= 1
        exp_prev = torch.exp(prev_m - m_new)
        exp_curr = torch.exp(curr_m - m_new)
        
        # u_new = u_prev * exp_prev + u_curr * exp_curr
        # w_new = w_prev * exp_prev + w_curr * exp_curr
        curr_u = prev_u * exp_prev + curr_u * exp_curr
        curr_w = prev_w * exp_prev + curr_w * exp_curr
        curr_m = m_new

    # Final Output = u / w (Numerator / Denominator)
    # We return components to allow the caller to handle final normalization or layer norm
    return curr_u, curr_w, curr_m
