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
from action_encoding import ACTIONS_PER_SQUARE, NUM_ACTIONS
from observation import NUM_OBS_PLANES


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
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1,
                 board_height: int = 10, board_width: int = 10):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Learnable 2D positional encoding
        # Allows attention to distinguish board positions (e.g., flag row vs midfield)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, board_height * board_width, channels) * 0.02
        )
        
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
        
        # Add positional encoding so attention can reason about board positions
        x_flat = x_flat + self.pos_encoding
        
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

    Output: (batch, 3600, atoms) log-probabilities, one return distribution per action of the
    lossless encoding in environment/action_encoding.py (index = square * 36 + k, canonical
    frame). The advantage head is applied per square: every square's backbone features go
    through the same two NoisyLinear layers to that square's 36 actions. (The thesis network
    flattened a 2-channel map into one 512-unit layer feeding all 400 actions.)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (NUM_OBS_PLANES, 10, 10), output_size: int = NUM_ACTIONS,
                 num_atoms: int = 51, channels: int = None, res_blocks: int = None, attention_heads: int = None,
                 advantage_hidden: int = None):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_atoms = num_atoms
        board_squares = input_shape[1] * input_shape[2]
        if output_size != board_squares * ACTIONS_PER_SQUARE:
            raise ValueError(f"output_size must be {board_squares} squares x {ACTIONS_PER_SQUARE} actions "
                             f"= {board_squares * ACTIONS_PER_SQUARE} (got {output_size})")
        
        # Backbone size comes from training_config (defaults: 64 channels, 6 blocks, 4 heads)
        from training_config import NETWORK_CHANNELS, RESNET_BLOCKS, ATTENTION_HEADS, ADVANTAGE_HIDDEN
        channels = channels or NETWORK_CHANNELS
        res_blocks = RESNET_BLOCKS if res_blocks is None else res_blocks
        attention_heads = attention_heads or ATTENTION_HEADS
        advantage_hidden = advantage_hidden or ADVANTAGE_HIDDEN
        self.channels = channels
        
        # Initial Convolution
        # Input: (C_in, 10, 10) -> Output: (channels, 10, 10)
        self.conv_in = nn.Conv2d(input_shape[0], channels, kernel_size=3, padding=1)
        self.bn_in = nn.InstanceNorm2d(channels)
        
        # Residual Backbone
        # Keeps shape (channels, 10, 10) but increases depth/abstraction
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(res_blocks)])
        
        # Spatial Self-Attention for global board reasoning
        # Applied AFTER ResBlocks to capture long-range piece relationships
        self.spatial_attention = SpatialAttention(channels=channels, num_heads=attention_heads, dropout=0.1)
        
        # Value Head
        # (channels, 10, 10) -> (1, 10, 10) -> Flatten -> 100 -> NoisyLinear
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.InstanceNorm2d(1)
        self.value_fc = NoisyLinear(10 * 10, 256)
        self.value_out = NoisyLinear(256, num_atoms)
        
        # Advantage Head, per square (weights shared by all squares)
        # (channels,) per square -> NoisyLinear -> advantage_hidden -> NoisyLinear -> 36 actions x atoms
        self.advantage_fc = NoisyLinear(channels, advantage_hidden)
        self.advantage_out = NoisyLinear(advantage_hidden, ACTIONS_PER_SQUARE * num_atoms)
        
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
        
        # Advantage Stream: (B, C, H, W) -> (B, squares, C) -> (B, squares * 36, atoms)
        cells = x.flatten(2).transpose(1, 2)
        adv_hidden = F.relu(self.advantage_fc(cells))
        adv_out = self.advantage_out(adv_hidden)
        adv_out = adv_out.reshape(batch_size, self.output_size, self.num_atoms)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        q_logits = val_out + (adv_out - adv_mean)
        return F.log_softmax(q_logits, dim=2)
        
    def evaluate_state_value(self, x):
        """
        Extract the pure state-value V(s) from the dueling head.
        This provides a global evaluation of the board advantage.
        """
        batch_size = x.size(0)
        
        # ResNet Backbone
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.spatial_attention(x)
            
        # Value Stream
        val_x = F.relu(self.value_bn(self.value_conv(x)))
        val_x = val_x.view(batch_size, -1)
        val_hidden = F.relu(self.value_fc(val_x))
        val_out = self.value_out(val_hidden)
        
        # Softmax for probability distribution over atoms
        return F.softmax(val_out, dim=1)

    def set_noise(self, enabled: bool):
        """Keep the NoisyLinear noise on in eval mode too (see NoisyLinear.noisy)."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.noisy = enabled

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
