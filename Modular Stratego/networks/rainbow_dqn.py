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


class ResidualBlock(nn.Module):
    """
    Simple Residual Block for Stratego ResNet
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels) # InstanceNorm is often better for batch-size 1 or small batches
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention Layer for global board reasoning.
    
    Each board position (cell) attends to ALL other positions,
    allowing the network to capture long-range piece relationships
    like "my Marshal vs their Spy" or "path to enemy flag".
    
    Input: (B, C, H, W) -> Output: (B, C, H, W)
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward network (standard transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, 100, 64)
        
        # Self-attention with residual connection
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        return x_flat.permute(0, 2, 1).view(B, C, H, W)

class RainbowDQN(nn.Module):
    """
    Rainbow DQN Network with ResNet Backbone
    - ResNet-Lite Backbone (6 Residual Blocks)
    - Dueling Heads
    - Noisy Nets
    - C51 Distributional Output
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (15, 10, 10), output_size: int = 400, num_atoms: int = 51):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_atoms = num_atoms
        
        # Initial Convolution
        # Input: (15/27, 10, 10) -> Output: (64, 10, 10)
        self.conv_in = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.bn_in = nn.InstanceNorm2d(64)
        
        # Residual Backbone (6 Blocks)
        # Keeps shape (64, 10, 10) but increases depth/abstraction
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(6)])
        
        # Spatial Self-Attention for global board reasoning
        # Applied AFTER ResBlocks to capture long-range piece relationships
        self.spatial_attention = SpatialAttention(channels=64, num_heads=4, dropout=0.1)
        
        # Value Head
        # (64, 10, 10) -> (1, 10, 10) -> Flatten -> 100 -> NoisyLinear
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.InstanceNorm2d(1)
        self.value_fc = NoisyLinear(10 * 10, 256)
        self.value_out = NoisyLinear(256, num_atoms)
        
        # Advantage Head
        # (64, 10, 10) -> (2, 10, 10) -> Flatten -> 200 -> NoisyLinear
        self.advantage_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.advantage_bn = nn.InstanceNorm2d(2)
        self.advantage_fc = NoisyLinear(2 * 10 * 10, 512)
        self.advantage_out = NoisyLinear(512, output_size * num_atoms)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # ResNet Backbone
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Spatial Self-Attention for global reasoning
        x = self.spatial_attention(x)
            
        # Dueling Heads
        
        # Value Stream
        val_x = F.relu(self.value_bn(self.value_conv(x)))
        val_x = val_x.view(batch_size, -1)
        val_hidden = F.relu(self.value_fc(val_x))
        val_out = self.value_out(val_hidden)
        val_out = val_out.view(batch_size, 1, self.num_atoms)
        
        # Advantage Stream
        adv_x = F.relu(self.advantage_bn(self.advantage_conv(x)))
        adv_x = adv_x.view(batch_size, -1)
        adv_hidden = F.relu(self.advantage_fc(adv_x))
        adv_out = self.advantage_out(adv_hidden)
        adv_out = adv_out.view(batch_size, self.output_size, self.num_atoms)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        q_logits = val_out + (adv_out - adv_mean)
        
        # Log Softmax for C51
        log_probs = F.log_softmax(q_logits, dim=2)
        
        return log_probs
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
