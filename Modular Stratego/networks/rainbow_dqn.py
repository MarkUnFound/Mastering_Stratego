# Networks Module - Rainbow DQN Network
# Extracted from drqn_agent.py for better modularity

"""
RainbowDQN: Rainbow DQN Network with Dueling Architecture and C51 Distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .noisy_linear import NoisyLinear


class RainbowDQN(nn.Module):
    """
    Rainbow DQN Network
    - Feed-Forward (CNN)
    - Dueling Heads
    - Noisy Nets
    - C51 Distributional Output
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (15, 10, 10), output_size: int = 1000, num_atoms: int = 51):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_atoms = num_atoms
        
        # CNN Layers (Feature Extractor)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size: 64 * 10 * 10 = 6400
        self.flatten_size = 64 * 10 * 10
        
        # Dueling Architecture with Noisy Nets
        # Value stream: State -> Value Distribution
        self.value_fc = NoisyLinear(self.flatten_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)  # Output is distribution over atoms
        
        # Advantage stream: State -> Advantage Distribution
        self.advantage_fc = NoisyLinear(self.flatten_size, 512)
        self.advantage_out = NoisyLinear(512, output_size * num_atoms)  # Output is (Actions * Atoms)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (batch, C, H, W)
        Returns:
            log_probs: Log probabilities of shape (batch, action_size, num_atoms)
        """
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Flatten
        
        # Dueling Heads
        # Value stream
        val_hidden = F.relu(self.value_fc(x))
        val_out = self.value_out(val_hidden)  # (batch, num_atoms)
        val_out = val_out.view(batch_size, 1, self.num_atoms)  # Reshape for broadcasting
        
        # Advantage stream
        adv_hidden = F.relu(self.advantage_fc(x))
        adv_out = self.advantage_out(adv_hidden)  # (batch, action_size * num_atoms)
        adv_out = adv_out.view(batch_size, self.output_size, self.num_atoms)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # In Distributional RL, we combine logits/probs
        adv_mean = adv_out.mean(dim=1, keepdim=True)  # Mean over actions
        
        # Unnormalized logits
        q_logits = val_out + (adv_out - adv_mean)
        
        # Softmax to get probabilities (Log Softmax for stability with KL Div loss)
        log_probs = F.log_softmax(q_logits, dim=2)  # Softmax over atoms dimension
        
        return log_probs
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
