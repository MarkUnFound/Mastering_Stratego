
import numpy as np
import random
import torch
from collections import namedtuple

# Define Experience namedtuple to match DQNAgent
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """
    GPU-accelerated Prioritized Experience Replay Buffer.
    Uses PyTorch tensors for priorities to enable fast vectorized sampling.
    This eliminates the CPU bottleneck from the SumTree implementation.
    """
    def __init__(self, capacity, alpha=0.6, device='cuda'):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.epsilon = 0.01  # Small constant to ensure non-zero priority
        self.device = device
        
        # Storage
        self.buffer = [None] * capacity
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.position = 0
        self.size = 0
        
    def add(self, experience, error=None):
        """
        Add a new experience to the buffer.
        New experiences are given max priority to ensure they are seen at least once.
        """
        # Calculate priority
        if error is None:
            # Use max priority for new experiences
            if self.size > 0:
                max_priority = self.priorities[:self.size].max().item()
                if max_priority == 0:
                    max_priority = 1.0
            else:
                max_priority = 1.0
        else:
            max_priority = (abs(error) + self.epsilon) ** self.alpha
        
        # Store experience
        self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priority.
        FULLY GPU-ACCELERATED using PyTorch operations.
        Returns batch, indices (for update), and importance sampling weights.
        """
        # Get valid priorities (only up to current size)
        valid_priorities = self.priorities[:self.size]
        
        # Normalize to probabilities (on GPU)
        probabilities = valid_priorities / valid_priorities.sum()
        
        # Sample indices using multinomial (GPU operation)
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        
        # Get experiences
        batch = [self.buffer[idx.item()] for idx in indices]
        
        # Calculate importance sampling weights (GPU operations)
        sample_probs = probabilities[indices]
        is_weights = torch.pow(self.size * sample_probs, -beta)
        is_weights = is_weights / is_weights.max()  # Normalize
        
        # Convert to numpy for compatibility with existing training code
        is_weights_np = is_weights.cpu().numpy()
        indices_list = indices.cpu().tolist()
        
        return batch, indices_list, is_weights_np
    
    def update(self, indices, errors):
        """
        Update priorities of sampled experiences based on new TD errors.
        """
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(
            [(abs(error) + self.epsilon) ** self.alpha for error in errors],
            dtype=torch.float32,
            device=self.device
        )
        self.priorities[indices_tensor] = priorities
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = [None] * self.capacity
        self.priorities = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.position = 0
        self.size = 0
