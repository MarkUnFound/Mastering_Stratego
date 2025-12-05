# PBS Module - Probabilistic Belief State Utilities
# Extracted helper functions for better modularity

"""
Utility functions for PBS (Probabilistic Belief State).
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

try:
    from piece import PieceType, PIECE_RANKS, NUM_PIECE_TYPES
except ImportError:
    # Fallback if running standalone
    NUM_PIECE_TYPES = 12


def extract_action_features(action: Tuple[Tuple[int, int], Tuple[int, int]], 
                            game_state, 
                            player_id: int,
                            pos: Optional[Tuple[int, int]] = None,
                            belief_distributions: Optional[Dict] = None,
                            piece_action_history: Optional[Dict] = None,
                            piece_observation_times: Optional[Dict] = None,
                            turn_count: int = 0,
                            evaluator=None,
                            apply_feature_weights: bool = True) -> np.ndarray:
    """
    Extract enhanced features from an action for Aaren input.
    
    Original Features (0-7):
    0: Move distance (tiles)
    1: Is attack (1 if attacking, 0 otherwise)
    2: Direction (0=N, 1=S, 2=E, 3=W)
    3: Distance from center
    4: Is forward move (toward enemy)
    5: Is backward move (away from enemy)
    6: Is lateral move
    7: Aggressiveness score (based on position and action)
    
    Enhanced Features (8-23):
    8: Piece value estimate (from current beliefs)
    9: Confidence in prediction (1 - entropy)
    10: Number of previous moves by this piece
    11: Time since piece was first observed (normalized)
    12: Position row (normalized)
    13: Position column (normalized)
    14: Distance to own flag (normalized)
    15: Distance to enemy flag (normalized)
    16: Number of adjacent friendly pieces
    17: Number of adjacent enemy pieces
    18: Is piece in enemy territory
    19: Is piece in center (rows 4-5)
    20: Game phase (early=0, mid=0.5, end=1.0)
    21: Turn count (normalized)
    22: Piece mobility estimate
    23: Threat level (adjacent enemies)
    """
    (r_from, c_from), (r_to, c_to) = action
    
    # Original features (0-7)
    distance = max(abs(r_to - r_from), abs(c_to - c_from))
    
    is_attack = 0.0
    if hasattr(game_state, 'board'):
        board = game_state.board
        if isinstance(board, torch.Tensor):
            target_val = board[r_to, c_to].item()
            if player_id == 1:
                is_attack = 1.0 if target_val < 0 else 0.0
            else:
                is_attack = 1.0 if target_val > 0 else 0.0
    
    if r_to > r_from:
        direction = 1.0  # South
    elif r_to < r_from:
        direction = 0.0  # North
    elif c_to > c_from:
        direction = 2.0  # East
    else:
        direction = 3.0  # West
    
    center_r, center_c = 4.5, 4.5
    dist_from_center = np.sqrt((r_to - center_r)**2 + (c_to - center_c)**2) / 10.0
    
    if player_id == 1:
        # Player 1 is at bottom (rows 6-9), moves UP (decreasing r) to advance
        is_forward = 1.0 if r_to < r_from else 0.0
        is_backward = 1.0 if r_to > r_from else 0.0
    else:
        # Player 2 is at top (rows 0-3), moves DOWN (increasing r) to advance
        is_forward = 1.0 if r_to > r_from else 0.0
        is_backward = 1.0 if r_to < r_from else 0.0
    
    is_lateral = 1.0 if r_from == r_to or c_from == c_to else 0.0
    aggressiveness = is_attack * 0.5
    if is_forward:
        aggressiveness += 0.3
    if distance == 0:
        aggressiveness += 0.2
    
    # Enhanced features (8-23)
    if pos and belief_distributions and pos in belief_distributions:
        beliefs = belief_distributions[pos]
        # Feature 8: Piece value estimate
        piece_value_estimate = sum(PIECE_RANKS.get(pt, 0) * conf for pt, conf in beliefs.items()) / 12.0
        
        # Feature 9: Confidence (1 - entropy)
        entropy = -sum(conf * np.log(conf + 1e-10) for conf in beliefs.values())
        max_entropy = np.log(len(beliefs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        # Feature 10: Number of previous moves
        num_moves = len(piece_action_history.get(pos, [])) / 20.0 if piece_action_history else 0.0
        
        # Feature 11: Time since first observed
        if piece_observation_times and pos in piece_observation_times:
            time_since_observed = (turn_count - piece_observation_times[pos]) / 500.0
        else:
            time_since_observed = 0.0
    else:
        piece_value_estimate = 0.5
        confidence = 0.0
        num_moves = 0.0
        time_since_observed = 0.0
    
    # Feature 12-13: Position
    pos_row = r_from / 10.0
    pos_col = c_from / 10.0
    
    # Feature 14-15: Distance to flags (simplified)
    dist_to_own_flag = 0.5
    dist_to_enemy_flag = 0.5
    
    # Feature 16-17: Adjacent pieces (simplified)
    adjacent_friendly = 0.0
    adjacent_enemy = 0.0
    
    # Feature 18: Is in enemy territory
    if player_id == 1:
        is_in_enemy_territory = 1.0 if r_from <= 3 else 0.0
    else:
        is_in_enemy_territory = 1.0 if r_from >= 6 else 0.0
    
    # Feature 19: Is in center
    is_in_center = 1.0 if 4 <= r_from <= 5 else 0.0
    
    # Feature 20: Game phase
    if hasattr(game_state, 'turn_count'):
        turn = game_state.turn_count
    else:
        turn = turn_count
    if turn < 50:
        game_phase = 0.0  # Early
    elif turn < 200:
        game_phase = 0.5  # Mid
    else:
        game_phase = 1.0  # End
    
    # Feature 21: Turn count
    turn_count_norm = turn / 500.0
    
    # Feature 22: Mobility estimate
    mobility_estimate = 1.0 - (distance / 10.0) if distance > 0 else 0.5
    
    # Feature 23: Threat level
    threat_level = is_attack * 0.5 + (adjacent_enemy / 4.0)
    
    features = np.array([
        distance / 10.0,          # 0
        is_attack,                 # 1
        direction / 3.0,           # 2
        dist_from_center,          # 3
        is_forward,                # 4
        is_backward,               # 5
        is_lateral,                # 6
        aggressiveness,            # 7
        piece_value_estimate,      # 8
        confidence,                # 9
        num_moves,                 # 10
        time_since_observed,       # 11
        pos_row,                   # 12
        pos_col,                   # 13
        dist_to_own_flag,          # 14
        dist_to_enemy_flag,        # 15
        adjacent_friendly,         # 16
        adjacent_enemy,            # 17
        is_in_enemy_territory,     # 18
        is_in_center,              # 19
        game_phase,                # 20
        turn_count_norm,           # 21
        mobility_estimate,         # 22
        threat_level               # 23
    ], dtype=np.float32)
    
    # Apply feature importance weighting if evaluator is available
    if apply_feature_weights and evaluator is not None:
        try:
            importance_weights = evaluator.get_feature_importance(features)
            if importance_weights.shape == features.shape:
                features = features * importance_weights
        except Exception:
            pass
    
    return features


def calculate_entropy(beliefs: Dict) -> float:
    """Calculate normalized entropy of a belief distribution."""
    if not beliefs:
        return 1.0
    
    probs = [p for p in beliefs.values() if p > 0]
    if not probs:
        return 1.0
    
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    max_entropy = math.log(len(beliefs))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def normalize_beliefs(beliefs: Dict) -> Dict:
    """Normalize a belief distribution to sum to 1.0."""
    total = sum(beliefs.values())
    if total > 0:
        return {pt: p / total for pt, p in beliefs.items()}
    return beliefs
