"""
Exploitability Critic for Stratego
Predicts the agent's action to penalize predictable behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

class ExploitabilityCritic(nn.Module):
    """
    A network that tries to predict the agent's action from the state.
    If this network has high accuracy, the agent is predictable (exploitable).
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 10, 10), output_size: int = 1000):
        """
        Initialize the Critic network (CNN-based, similar to the Agent)
        
        Args:
            input_shape: Shape of input (channels, height, width)
            output_size: Size of output (number of possible actions)
        """
        super(ExploitabilityCritic, self).__init__()
        
        # Same architecture as ConvDQN to ensure it has enough capacity to model the agent
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.flatten_size = 64 * 10 * 10
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
    def forward(self, x):
        """Forward pass - outputs logits for each action"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SetupExploitabilityCritic(nn.Module):
    """
    Critic for the Setup Agent (CNN-based)
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (41, 10, 10), output_size: int = 100):
        """
        Args:
            input_shape: (channels, 10, 10). 41 channels: 40 for pieces + 1 for board state? 
                         Actually for setup, we'll use the same input as the new ConvSetupDQN.
                         Let's assume 41 channels: 1 for board, 40 for available pieces mask?
                         Or just 1 channel with values?
                         Let's match the planned ConvSetupDQN input.
            output_size: Number of positions (100 for 10x10 board)
        """
        super(SetupExploitabilityCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.flatten_size = 64 * 10 * 10
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
