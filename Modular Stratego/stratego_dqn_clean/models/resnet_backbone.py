"""
ResNet Backbone for Stratego Double DQN
4-layer architecture (64 → 128 → 128 → 64) for 6GB VRAM constraint.

This is NOT ResNet-18/34/50 - it's a lightweight custom backbone.

Input: (B, 79, 10, 10) - 15 board channels + 64 AAREN embedding channels
Output: (B, 64, 10, 10) - Spatial feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Simple Residual Block with projection shortcut when dimensions change.
    Uses BatchNorm2d (better for batch sizes up to 32).
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class ResNetBackbone(nn.Module):
    """
    4-layer ResNet backbone for Stratego.
    
    Architecture: 79 → 64 → 128 → 128 → 64
    
    Designed for:
    - 6GB VRAM constraint
    - Batch size ≤ 32
    - 10x10 Stratego board (no spatial downsampling)
    
    Input: (B, 79, 10, 10) - 15 board + 64 AAREN = 79 channels
    Output: (B, 64, 10, 10) - Spatial features
    """
    
    def __init__(self, input_channels: int = 79):
        super().__init__()
        
        self.input_channels = input_channels
        
        # Initial convolution: 79 → 64
        self.conv_in = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(64)
        
        # Layer 1: 64 → 64 (maintain dimensions)
        self.layer1 = ResidualBlock(64, 64)
        
        # Layer 2: 64 → 128 (increase channels)
        self.layer2 = ResidualBlock(64, 128)
        
        # Layer 3: 128 → 128 (maintain)
        self.layer3 = ResidualBlock(128, 128)
        
        # Layer 4: 128 → 64 (reduce for output)
        self.layer4 = ResidualBlock(128, 64)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 79, 10, 10)
            
        Returns:
            features: (B, 64, 10, 10)
        """
        # Initial conv
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # 4 residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def get_output_size(self) -> Tuple[int, int, int]:
        """Return output dimensions (C, H, W)."""
        return (64, 10, 10)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetBackboneWithCheckpointing(ResNetBackbone):
    """
    ResNet backbone with gradient checkpointing for OOM prevention.
    
    Trades compute for memory by recomputing activations during backward pass.
    Automatically enabled by Memory Guardian when VRAM > 5.5GB.
    """
    
    def __init__(self, input_channels: int = 79):
        super().__init__(input_channels)
        self.use_checkpointing = False
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage."""
        self.use_checkpointing = True
        print("[Memory Guardian] Gradient checkpointing ENABLED for ResNet backbone")
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.use_checkpointing and self.training:
            return self._forward_with_checkpointing(x)
        return super().forward(x)
    
    def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        # Initial conv (small, no checkpoint needed)
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # Checkpoint each residual layer
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint(self.layer3, x, use_reentrant=False)
        x = checkpoint(self.layer4, x, use_reentrant=False)
        
        return x
