"""
Intrinsic Curiosity Module for Stratego DQN Training

Provides novelty-based intrinsic rewards to encourage exploration 
in sparse-reward environments.
"""

import torch
from typing import Dict, Optional
from collections import defaultdict


class StateNoveltyTracker:
    """
    Tracks visited states using locality-sensitive hashing.
    Provides decaying novelty bonus for new state visits.
    
    Uses a simple hash-based approach that is memory efficient
    and GPU-compatible.
    """
    
    def __init__(self, 
                 bonus_scale: float = 0.01,
                 decay_rate: float = 0.5,
                 max_states: int = 100000,
                 device: str = 'cuda'):
        """
        Args:
            bonus_scale: Maximum bonus for a completely novel state
            decay_rate: How fast the bonus decays (0.5 = halves each visit)
            max_states: Maximum states to track (LRU eviction after this)
            device: Device for tensor operations
        """
        self.bonus_scale = bonus_scale
        self.decay_rate = decay_rate
        self.max_states = max_states
        self.device = device
        
        # Visit counts: hash -> count
        self.visit_counts: Dict[int, int] = defaultdict(int)
        self.access_order: list = []  # For LRU eviction
        
    def _compute_hash(self, state_tensor: torch.Tensor) -> int:
        """
        Compute a fast hash of the state tensor.
        Uses quantization + Python hash for speed.
        """
        # Quantize to reduce memory and speed up hashing
        # Round to 2 decimal places and convert to bytes
        if state_tensor.device.type == 'cuda':
            state_tensor = state_tensor.cpu()
            
        # Simple spatial hash: sample key positions
        # This is faster than hashing entire tensor
        flat = state_tensor.flatten()
        sample_size = min(100, len(flat))
        indices = torch.linspace(0, len(flat)-1, sample_size).long()
        sampled = flat[indices]
        
        # Quantize and hash
        quantized = (sampled * 100).int().tolist()
        return hash(tuple(quantized))
    
    def get_novelty_bonus(self, state_tensor: torch.Tensor) -> float:
        """
        Compute novelty bonus for a state.
        
        Returns:
            Bonus value in [0, bonus_scale], higher = more novel
        """
        h = self._compute_hash(state_tensor)
        count = self.visit_counts[h]
        
        # Update count
        self.visit_counts[h] = count + 1
        
        # LRU eviction if needed
        if h in self.access_order:
            self.access_order.remove(h)
        self.access_order.append(h)
        
        while len(self.access_order) > self.max_states:
            oldest = self.access_order.pop(0)
            if oldest in self.visit_counts:
                del self.visit_counts[oldest]
        
        # Decaying bonus: scale * decay^count
        # First visit (count=0): full bonus
        # Second visit (count=1): bonus * 0.5
        # Third visit (count=2): bonus * 0.25
        bonus = self.bonus_scale * (self.decay_rate ** count)
        
        return bonus
    
    def get_batch_novelty_bonus(self, state_tensors: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty bonus for a batch of states.
        
        Args:
            state_tensors: Tensor of shape (batch, channels, height, width)
            
        Returns:
            Tensor of shape (batch,) with bonus values
        """
        batch_size = state_tensors.shape[0]
        bonuses = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            bonuses[i] = self.get_novelty_bonus(state_tensors[i])
            
        return bonuses
    
    def reset(self):
        """Clear all visit tracking (call between episodes if desired)."""
        self.visit_counts.clear()
        self.access_order.clear()
        
    def get_stats(self) -> Dict:
        """Get tracking statistics for logging."""
        counts = list(self.visit_counts.values())
        return {
            'unique_states': len(self.visit_counts),
            'avg_visits': sum(counts) / max(1, len(counts)),
            'max_visits': max(counts) if counts else 0,
        }


def create_novelty_tracker(bonus_scale: float = 0.01, 
                          device: str = 'cuda') -> StateNoveltyTracker:
    """Factory function for creating novelty tracker."""
    return StateNoveltyTracker(
        bonus_scale=bonus_scale,
        decay_rate=0.5,
        max_states=100000,
        device=device
    )
