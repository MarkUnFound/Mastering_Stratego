"""
Double DQN + AAREN Models for Stratego
Memory-constrained implementation for 6GB VRAM
"""

from .aaren import AAREN, AARENCell
from .resnet_backbone import ResNetBackbone
from .double_dqn import DoubleDQN, DoubleDQNAgent

__all__ = ['AAREN', 'AARENCell', 'ResNetBackbone', 'DoubleDQN', 'DoubleDQNAgent']
