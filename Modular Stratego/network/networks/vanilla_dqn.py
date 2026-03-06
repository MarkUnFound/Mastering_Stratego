import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VanillaDQN(nn.Module):
    """
    Vanilla DQN Network for Stratego
    - Simple 3-layer CNN
    - Fully connected output
    - Standard outputs instead of distributional C51
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (15, 10, 10), output_size: int = 400):
        super(VanillaDQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        
        # Simple Convolutional Backbone
        # Input: (D, 10, 10) -> Output: (64, 4, 4)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn3 = nn.InstanceNorm2d(64)
        
        # Calculate flattened size: 10 -> 10 -> 8
        self.flatten_size = 64 * 8 * 8
        
        # Standard Fully Connected Head
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully Connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values
