# AAREN Module - Attention as a Recurrent Neural Network
# 
# This module provides efficient attention-based recurrent computation:
# - Parallel training (like Transformers)
# - O(1) inference updates (like RNNs)
# - Constant memory usage
# - No vanishing gradients

"""
AAREN (Attention as a Recurrent Neural Network) module for Stratego PBS.
"""

from .kernel import aaren_scan_kernel
from .cell import AarenCell
from .network import PieceActionAaren

__all__ = [
    'aaren_scan_kernel',
    'AarenCell',
    'PieceActionAaren',
]
