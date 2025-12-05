# PBS Module - Probabilistic Belief State
# 
# This module provides probabilistic inference about opponent pieces:
# - Rule-based inference (e.g., multi-tile moves = Scout)
# - AAREN-based pattern learning from action sequences
# - Confidence scores for each possible piece value

"""
PBS (Probabilistic Belief State) module for Stratego.
"""

from .belief_state import ProbabilisticBeliefState, PBS_EVALUATOR_AVAILABLE
from .utils import extract_action_features, calculate_entropy, normalize_beliefs

__all__ = [
    'ProbabilisticBeliefState',
    'PBS_EVALUATOR_AVAILABLE',
    'extract_action_features',
    'calculate_entropy',
    'normalize_beliefs',
]
