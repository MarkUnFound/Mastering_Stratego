"""
Hindsight-Aided Belief Refinement (HABR) Module for Stratego DRL Agent

This module implements:
1. SinkhornLayer: Differentiable normalization for piece-count constraints
2. RetrospectiveBuffer: Stores hidden states and PBS snapshots for retrospective learning
3. HABRLoss: Computes retrospective KL divergence loss with temporal decay
4. Information Gain reward calculation

Mathematical Foundations:
- Information Gain: R_gain(t) = H(PBS_{t-1}) - H(PBS_t)
- Retrospective KL: L_retro = sum_{k=1}^{t} lambda^{t-k} * D_KL(PBS_k(p) || delta_s*)
- Sinkhorn constraints: Row sums = 1, Column sums = piece counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

# Import piece types
try:
    from piece import PieceType, NUM_PIECE_TYPES
except ImportError:
    NUM_PIECE_TYPES = 12

# Standard Stratego piece counts
STANDARD_PIECE_COUNTS = {
    1: 1,   # Flag
    2: 1,   # Spy
    3: 8,   # Scout
    4: 5,   # Miner
    5: 4,   # Sergeant
    6: 4,   # Lieutenant
    7: 4,   # Captain
    8: 3,   # Major
    9: 2,   # Colonel
    10: 1,  # General
    11: 1,  # Marshal
    12: 6,  # Bomb
}


class SinkhornLayer(nn.Module):
    """
    Differentiable Sinkhorn normalization layer for enforcing piece-count constraints.
    
    This layer ensures:
    - Row constraint: Each piece has exactly one identity (probabilities sum to 1)
    - Column constraint: Total count of each piece type matches remaining counts
    
    Uses iterative proportional fitting (Sinkhorn-Knopp algorithm) which is
    differentiable and preserves gradients for backpropagation.
    
    The Sinkhorn operator acts as a differentiable version of the Hungarian Algorithm,
    pushing probability mass between pieces when constraints are violated.
    """
    
    def __init__(self, num_iterations: int = 10, temperature: float = 0.1, eps: float = 1e-8):
        """
        Initialize Sinkhorn normalization layer.
        
        Args:
            num_iterations: Number of Sinkhorn iterations (default: 10)
            temperature: Softmax temperature for differentiability (default: 0.1)
            eps: Small epsilon for numerical stability
        """
        super(SinkhornLayer, self).__init__()
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, logits: torch.Tensor, 
                col_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Sinkhorn normalization to belief logits.
        
        Args:
            logits: Raw logit matrix of shape (num_pieces, num_piece_types)
                    or (batch, num_pieces, num_piece_types)
            col_targets: Target column sums (remaining piece counts)
                        Shape: (num_piece_types,) or (batch, num_piece_types)
                        If None, uses uniform distribution
        
        Returns:
            Normalized probability matrix satisfying constraints
        """
        # Handle 2D and 3D inputs
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, num_pieces, num_types = logits.shape
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Initialize with softmax (satisfies row constraint)
        log_alpha = F.log_softmax(scaled_logits, dim=-1)
        
        # Set up column targets (normalized)
        if col_targets is None:
            # Uniform distribution if not specified
            col_targets = torch.ones(num_types, device=logits.device)
        
        if col_targets.dim() == 1:
            col_targets = col_targets.unsqueeze(0).expand(batch_size, -1)
            
        # Normalize column targets to sum to num_pieces
        col_targets = col_targets / (col_targets.sum(dim=-1, keepdim=True) + self.eps) * num_pieces
        
        # Sinkhorn iterations
        for _ in range(self.num_iterations):
            # Column normalization (to match piece counts)
            log_col_sums = torch.logsumexp(log_alpha, dim=1)  # (batch, num_types)
            log_col_targets = torch.log(col_targets + self.eps)
            log_alpha = log_alpha - (log_col_sums - log_col_targets).unsqueeze(1)
            
            # Row normalization (each piece sums to 1)
            log_row_sums = torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - log_row_sums
        
        # Convert back to probabilities
        probs = torch.exp(log_alpha)
        
        # Ensure valid probability distribution
        probs = torch.clamp(probs, min=self.eps, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        if squeeze_output:
            probs = probs.squeeze(0)
            
        return probs


class RetrospectiveBuffer:
    """
    Specialized buffer for storing PBS history for retrospective learning.
    
    Stores for each tracked piece:
    - PBS snapshots at each timestep
    - Hidden states from the belief tracker
    - Timestamps for temporal decay calculation
    
    This enables computing the retrospective KL loss when a piece is revealed:
    L_retro = sum_{k=1}^{t} lambda^{t-k} * D_KL(PBS_k(p) || delta_s*)
    """
    
    def __init__(self, max_pieces: int = 40, max_history: int = 500, device=None):
        """
        Initialize the retrospective buffer.
        
        Args:
            max_pieces: Maximum number of pieces to track
            max_history: Maximum history length per piece
            device: PyTorch device
        """
        self.max_pieces = max_pieces
        self.max_history = max_history
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Storage: piece_id -> list of (timestamp, pbs_snapshot, hidden_state)
        self.history: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]]] = defaultdict(list)
        
        # Current timestamp
        self.current_time = 0
        
    def reset(self):
        """Reset buffer for a new game."""
        self.history.clear()
        self.current_time = 0
        
    def add_observation(self, piece_pos: Tuple[int, int], 
                       pbs_snapshot: torch.Tensor,
                       hidden_state: Optional[torch.Tensor] = None):
        """
        Add a PBS observation for a piece.
        
        Args:
            piece_pos: Position (row, col) of the piece
            pbs_snapshot: PBS probability distribution of shape (num_piece_types,)
            hidden_state: Optional hidden state from belief tracker
        """
        # Ensure tensors are detached and on correct device
        pbs_snapshot = pbs_snapshot.detach().clone().to(self.device)
        if hidden_state is not None:
            hidden_state = hidden_state.detach().clone().to(self.device)
            
        self.history[piece_pos].append((self.current_time, pbs_snapshot, hidden_state))
        
        # Trim history if too long
        if len(self.history[piece_pos]) > self.max_history:
            self.history[piece_pos] = self.history[piece_pos][-self.max_history:]
            
    def update_position(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """Update piece position when it moves."""
        if old_pos in self.history:
            self.history[new_pos] = self.history.pop(old_pos)
            
    def get_history(self, piece_pos: Tuple[int, int]) -> List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]]:
        """Get the full history for a piece."""
        return self.history.get(piece_pos, [])
    
    def get_pbs_history(self, piece_pos: Tuple[int, int]) -> List[torch.Tensor]:
        """Get just the PBS snapshots for a piece."""
        return [entry[1] for entry in self.history.get(piece_pos, [])]
    
    def increment_time(self):
        """Increment the current timestamp."""
        self.current_time += 1
        
    def remove_piece(self, piece_pos: Tuple[int, int]):
        """Remove a piece from tracking (e.g., after capture)."""
        if piece_pos in self.history:
            del self.history[piece_pos]


class HABRLoss(nn.Module):
    """
    Hindsight-Aided Belief Refinement Loss computation.
    
    Computes the retrospective KL divergence loss when a piece is revealed:
    L_retro = sum_{k=1}^{t} lambda^{t-k} * D_KL(PBS_k(p) || delta_s*)
    
    The temporal decay factor lambda < 1 (e.g., 0.9) prioritizes recent,
    high-information behaviors while providing smoothed learning signal.
    
    This addresses non-stationary piece behavior (e.g., bluffing) by:
    - Treating early-game bluffs with lower weight
    - Prioritizing late-game tells with higher weight
    """
    
    def __init__(self, decay_factor: float = 0.9, eps: float = 1e-8):
        """
        Initialize HABR loss.
        
        Args:
            decay_factor: Temporal decay factor (lambda in the loss formula)
            eps: Small epsilon for numerical stability
        """
        super(HABRLoss, self).__init__()
        self.decay_factor = decay_factor
        self.eps = eps
        
    def forward(self, pbs_history: List[torch.Tensor], 
                revealed_type: int) -> torch.Tensor:
        """
        Compute retrospective KL divergence loss.
        
        Args:
            pbs_history: List of PBS snapshots for the piece
                        Each tensor has shape (num_piece_types,)
            revealed_type: The revealed piece type index (0-11)
            
        Returns:
            Scalar loss tensor
        """
        if not pbs_history:
            return torch.tensor(0.0, requires_grad=True)
            
        device = pbs_history[0].device
        num_types = pbs_history[0].shape[0]
        t = len(pbs_history)
        
        # Create one-hot target (Dirac delta at revealed type)
        target = torch.zeros(num_types, device=device)
        target[revealed_type] = 1.0
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for k, pbs_k in enumerate(pbs_history):
            # Temporal decay weight: lambda^{t-k}
            # k=0 is earliest, k=t-1 is most recent
            time_diff = t - 1 - k
            weight = self.decay_factor ** time_diff
            
            # KL divergence: D_KL(PBS_k || delta_s*)
            # Since target is one-hot, this simplifies to -log(PBS_k[revealed_type])
            # With proper KL: sum_s target[s] * log(target[s] / PBS_k[s])
            # = -log(PBS_k[revealed_type])  (since target is 1 only at revealed_type)
            
            pbs_k_clamped = torch.clamp(pbs_k, min=self.eps)
            kl_term = -torch.log(pbs_k_clamped[revealed_type])
            
            total_loss = total_loss + weight * kl_term
            
        return total_loss


class InformationGainReward:
    """
    Computes Information Gain reward for belief state updates.
    
    The Information Gain reward is:
    R_gain(t) = D_KL(PBS_{t-1} || Prior) - D_KL(PBS_t || Prior)
              = H(PBS_{t-1}) - H(PBS_t)
    
    This equals the reduction in entropy. A positive reward is only achieved
    if the observation increases certainty about piece identity.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize Information Gain reward calculator.
        
        Args:
            eps: Small epsilon for numerical stability
        """
        self.eps = eps
        
    def compute_entropy(self, pbs: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of a probability distribution.
        
        Args:
            pbs: Probability distribution of shape (num_piece_types,)
            
        Returns:
            Scalar entropy value
        """
        pbs_clamped = torch.clamp(pbs, min=self.eps)
        return -torch.sum(pbs_clamped * torch.log(pbs_clamped))
    
    def compute_kl_from_uniform(self, pbs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from uniform prior.
        
        D_KL(P || U) = log(N) - H(P)
        
        Args:
            pbs: Probability distribution of shape (num_piece_types,)
            
        Returns:
            KL divergence
        """
        n = pbs.shape[0]
        entropy = self.compute_entropy(pbs)
        return math.log(n) - entropy
    
    def compute_reward(self, pbs_prev: torch.Tensor, 
                      pbs_current: torch.Tensor) -> float:
        """
        Compute Information Gain reward.
        
        R_gain = H(PBS_{t-1}) - H(PBS_t)
        
        Args:
            pbs_prev: Previous PBS state
            pbs_current: Current PBS state
            
        Returns:
            Information gain reward (positive = increased certainty)
        """
        h_prev = self.compute_entropy(pbs_prev)
        h_current = self.compute_entropy(pbs_current)
        
        # Entropy reduction = information gain
        return (h_prev - h_current).item()


class HABRBeliefTracker(nn.Module):
    """
    HABR-enhanced belief tracking module.
    
    Integrates:
    - Sinkhorn normalization for global piece-count constraints
    - Retrospective buffer for history storage
    - HABR loss computation on piece revelation
    - Information Gain reward calculation
    
    This replaces the Instantaneous Supervised Loss approach with
    retrospective learning that considers the full history of a piece.
    """
    
    def __init__(self, 
                 num_piece_types: int = NUM_PIECE_TYPES,
                 sinkhorn_iterations: int = 10,
                 decay_factor: float = 0.9,
                 device=None):
        """
        Initialize HABR belief tracker.
        
        Args:
            num_piece_types: Number of piece types (default: 12)
            sinkhorn_iterations: Number of Sinkhorn iterations
            decay_factor: Temporal decay for retrospective loss
            device: PyTorch device
        """
        super(HABRBeliefTracker, self).__init__()
        
        self.num_piece_types = num_piece_types
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core components
        self.sinkhorn = SinkhornLayer(num_iterations=sinkhorn_iterations)
        self.retrospective_buffer = RetrospectiveBuffer(device=self.device)
        self.habr_loss = HABRLoss(decay_factor=decay_factor)
        self.info_gain_reward = InformationGainReward()
        
        # Remaining piece counts (updated as pieces are revealed)
        self.remaining_counts = dict(STANDARD_PIECE_COUNTS)
        
        # Track accumulated HABR losses for training
        self.accumulated_losses: List[torch.Tensor] = []
        
        # Previous PBS for info gain calculation
        self._prev_pbs_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        
    def reset(self):
        """Reset for a new game."""
        self.retrospective_buffer.reset()
        self.remaining_counts = dict(STANDARD_PIECE_COUNTS)
        self.accumulated_losses.clear()
        self._prev_pbs_cache.clear()
        
    def update_remaining_counts(self, revealed_type: int):
        """Update remaining piece counts when a piece is revealed."""
        if revealed_type in self.remaining_counts:
            self.remaining_counts[revealed_type] = max(0, self.remaining_counts[revealed_type] - 1)
            
    def get_column_targets(self) -> torch.Tensor:
        """Get current column targets for Sinkhorn based on remaining counts."""
        targets = torch.zeros(self.num_piece_types, device=self.device)
        for piece_type, count in self.remaining_counts.items():
            if 1 <= piece_type <= self.num_piece_types:
                targets[piece_type - 1] = count
        return targets
    
    def apply_sinkhorn_constraints(self, belief_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply Sinkhorn normalization to enforce global constraints.
        
        Args:
            belief_matrix: Raw belief matrix of shape (num_pieces, num_piece_types)
            
        Returns:
            Normalized belief matrix satisfying constraints
        """
        col_targets = self.get_column_targets()
        return self.sinkhorn(belief_matrix, col_targets)
    
    def record_pbs_snapshot(self, piece_pos: Tuple[int, int], 
                           pbs: torch.Tensor,
                           hidden_state: Optional[torch.Tensor] = None):
        """
        Record a PBS snapshot for retrospective learning.
        
        Args:
            piece_pos: Position of the piece
            pbs: Current PBS state for this piece
            hidden_state: Optional hidden state from AAREN
        """
        # Cache previous PBS for info gain
        if piece_pos in self._prev_pbs_cache:
            prev_pbs = self._prev_pbs_cache[piece_pos]
        else:
            prev_pbs = None
            
        self._prev_pbs_cache[piece_pos] = pbs.detach().clone()
        
        # Store in retrospective buffer
        self.retrospective_buffer.add_observation(piece_pos, pbs, hidden_state)
        
    def compute_info_gain_reward(self, piece_pos: Tuple[int, int], 
                                  pbs_current: torch.Tensor) -> float:
        """
        Compute Information Gain reward for a PBS update.
        
        Args:
            piece_pos: Position of the piece
            pbs_current: Current PBS state
            
        Returns:
            Information gain reward value
        """
        if piece_pos in self._prev_pbs_cache:
            pbs_prev = self._prev_pbs_cache[piece_pos]
            return self.info_gain_reward.compute_reward(pbs_prev, pbs_current)
        return 0.0
    
    def on_piece_revealed(self, piece_pos: Tuple[int, int], 
                          revealed_type: int) -> Optional[torch.Tensor]:
        """
        Compute retrospective HABR loss when a piece is revealed.
        
        This triggers the A1 Objective: apply retrospective KL divergence
        loss across the entire history of the revealed piece.
        
        Args:
            piece_pos: Position of the revealed piece
            revealed_type: The revealed piece type (1-12)
            
        Returns:
            The computed HABR loss tensor (or None if no history)
        """
        # Get PBS history for this piece
        pbs_history = self.retrospective_buffer.get_pbs_history(piece_pos)
        
        if not pbs_history:
            return None
            
        # Convert revealed_type to 0-indexed
        type_idx = revealed_type - 1 if revealed_type >= 1 else revealed_type
        
        # Compute retrospective loss
        loss = self.habr_loss(pbs_history, type_idx)
        
        # Accumulate for training
        if loss.requires_grad:
            self.accumulated_losses.append(loss)
            
        # Update remaining counts
        self.update_remaining_counts(revealed_type)
        
        # Clean up
        self.retrospective_buffer.remove_piece(piece_pos)
        if piece_pos in self._prev_pbs_cache:
            del self._prev_pbs_cache[piece_pos]
            
        return loss
    
    def get_accumulated_loss(self) -> Optional[torch.Tensor]:
        """
        Get the accumulated HABR loss for training.
        
        Returns:
            Sum of all accumulated losses, or None if no losses
        """
        if not self.accumulated_losses:
            return None
            
        total_loss = sum(self.accumulated_losses)
        return total_loss
    
    def clear_accumulated_losses(self):
        """Clear accumulated losses after training step."""
        self.accumulated_losses.clear()
        
    def increment_time(self):
        """Increment the internal timestamp."""
        self.retrospective_buffer.increment_time()
        
    def on_piece_moved(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """Update tracking when a piece moves."""
        self.retrospective_buffer.update_position(old_pos, new_pos)
        if old_pos in self._prev_pbs_cache:
            self._prev_pbs_cache[new_pos] = self._prev_pbs_cache.pop(old_pos)


# Utility functions for integration with existing PBS

def create_belief_matrix_from_distributions(
    belief_distributions: Dict[Tuple[int, int], Dict],
    positions: List[Tuple[int, int]],
    num_types: int = NUM_PIECE_TYPES,
    device=None
) -> torch.Tensor:
    """
    Convert position-based belief dictionaries to a matrix for Sinkhorn.
    
    Args:
        belief_distributions: Dict mapping positions to belief dicts
        positions: List of positions to include
        num_types: Number of piece types
        device: PyTorch device
        
    Returns:
        Tensor of shape (num_positions, num_types)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_positions = len(positions)
    
    matrix = torch.zeros(n_positions, num_types, device=device)
    
    for i, pos in enumerate(positions):
        if pos in belief_distributions:
            beliefs = belief_distributions[pos]
            for piece_type, prob in beliefs.items():
                if hasattr(piece_type, 'value'):
                    type_idx = piece_type.value - 1
                else:
                    type_idx = int(piece_type) - 1
                if 0 <= type_idx < num_types:
                    matrix[i, type_idx] = prob
                    
    return matrix


def update_distributions_from_matrix(
    belief_distributions: Dict[Tuple[int, int], Dict],
    matrix: torch.Tensor,
    positions: List[Tuple[int, int]],
    piece_type_enum
):
    """
    Update position-based belief dictionaries from a Sinkhorn-normalized matrix.
    
    Args:
        belief_distributions: Dict to update
        matrix: Normalized matrix of shape (num_positions, num_types)
        positions: List of positions
        piece_type_enum: PieceType enum class
    """
    piece_types = list(piece_type_enum)
    
    for i, pos in enumerate(positions):
        if pos in belief_distributions:
            for j, pt in enumerate(piece_types):
                belief_distributions[pos][pt] = matrix[i, j].item()
