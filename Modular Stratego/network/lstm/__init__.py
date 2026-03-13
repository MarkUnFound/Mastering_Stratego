# LSTM Module - Standard Recurrent Network for History Encoding
# 
# Replaces AAREN (Attention as a Recurrent Neural Network) with
# a standard PyTorch LSTM for piece action pattern learning.

"""
LSTM module for Stratego history encoding.
Drop-in replacement for the AAREN module.
"""

from .network import PieceActionLSTM

__all__ = [
    'PieceActionLSTM',
]
