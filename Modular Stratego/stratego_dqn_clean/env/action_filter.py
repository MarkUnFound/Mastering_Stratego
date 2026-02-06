"""
Heuristic Action Filter for Stratego Double DQN
Filters legal moves to Top-K using strategic heuristics.

Ported from heuristic_filter.py

Scoring criteria:
- Attack moves: +100 (always high priority)
- Forward advancement: +5 per row toward enemy
- Center control: +10 for center squares
- Retreat penalty: -20 for backward moves
- Scout distance bonus: +2 per square for long moves
"""

import torch
from typing import List, Tuple, Optional

# Board constants
BOARD_SIZE = 10
LAKE_SQUARE = -13


class ActionFilter:
    """
    Filters legal moves to Top-K using heuristic scoring.
    
    Designed to reduce action space from ~400 to 100 maximum.
    This is critical for memory-constrained training.
    """
    
    MAX_MOVES = 100  # Hard ceiling per spec
    
    # Scoring weights
    ATTACK_SCORE = 100
    ADVANCE_SCORE = 5
    CENTER_SCORE = 10
    RETREAT_PENALTY = -20
    SCOUT_DISTANCE_BONUS = 2
    MIN_SCORE_THRESHOLD = 50
    
    def __init__(
        self,
        attack_score: int = 100,
        advance_score: int = 5,
        center_score: int = 10,
        retreat_penalty: int = -20,
        scout_distance_bonus: int = 2,
        min_score_threshold: int = 50
    ):
        """
        Initialize filter with scoring parameters.
        
        Args:
            attack_score: Bonus for attacking enemy pieces
            advance_score: Points per row advanced toward enemy
            center_score: Bonus for controlling center squares
            retreat_penalty: Penalty for moving backward
            scout_distance_bonus: Bonus per square for Scout long moves
            min_score_threshold: Minimum score to auto-include
        """
        self.attack_score = attack_score
        self.advance_score = advance_score
        self.center_score = center_score
        self.retreat_penalty = retreat_penalty
        self.scout_distance_bonus = scout_distance_bonus
        self.min_score_threshold = min_score_threshold
    
    def score_move(
        self,
        move: Tuple[Tuple[int, int], Tuple[int, int]],
        board: torch.Tensor,
        player_id: int
    ) -> int:
        """
        Score a single move based on strategic heuristics.
        
        Args:
            move: ((r_from, c_from), (r_to, c_to))
            board: 10x10 board tensor
            player_id: 1 or -1
            
        Returns:
            Heuristic score (higher = better)
        """
        (r1, c1), (r2, c2) = move
        score = 0
        
        # 1. Attack detection
        target_piece = board[r2, c2].item() if board is not None else 0
        
        if player_id == 1:
            is_attack = target_piece < 0 and target_piece > LAKE_SQUARE
        else:
            is_attack = target_piece > 0
        
        if is_attack:
            score += self.attack_score
        
        # 2. Forward advancement (toward enemy back rank)
        if player_id == 1:
            # Player 1 wants to decrease row (move toward row 0)
            advancement = r1 - r2
        else:
            # Player 2 wants to increase row (move toward row 9)
            advancement = r2 - r1
        
        if advancement > 0:
            score += advancement * self.advance_score
        elif advancement < 0:
            score += self.retreat_penalty
        
        # 3. Center control bonus
        center_dist = abs(r2 - 4.5) + abs(c2 - 4.5)
        if center_dist < 3:
            score += self.center_score
        
        # 4. Scout distance bonus
        move_dist = abs(r2 - r1) + abs(c2 - c1)
        if move_dist > 1:
            score += (move_dist - 1) * self.scout_distance_bonus
        
        return score
    
    def filter_moves(
        self,
        board: torch.Tensor,
        legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        player_id: int,
        max_moves: int = None
    ) -> List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]]:
        """
        Filter legal moves to Top-K using heuristic scoring.
        
        Algorithm:
        1. Score all legal moves
        2. Keep all moves with score > min_score_threshold
        3. Fill remaining slots to max_moves with next-best moves
        
        Args:
            board: 10x10 board tensor
            legal_moves: List of valid moves
            player_id: 1 or -1
            max_moves: Maximum moves to return (default: MAX_MOVES)
            
        Returns:
            List of (move, score) tuples, sorted by score descending
        """
        if max_moves is None:
            max_moves = self.MAX_MOVES
        
        # Hard ceiling
        max_moves = min(max_moves, self.MAX_MOVES)
        
        if not legal_moves:
            return []
        
        if len(legal_moves) <= max_moves:
            # No filtering needed, but still score for sorting
            scored = [(move, self.score_move(move, board, player_id)) 
                      for move in legal_moves]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        
        # Score all moves
        scored_moves = []
        for move in legal_moves:
            score = self.score_move(move, board, player_id)
            scored_moves.append((move, score))
        
        # Sort by score descending
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Take top max_moves
        return scored_moves[:max_moves]
    
    def get_filtered_moves(
        self,
        board: torch.Tensor,
        legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        player_id: int,
        max_moves: int = None
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get filtered moves (without scores).
        
        Convenience method that returns just the moves.
        """
        scored = self.filter_moves(board, legal_moves, player_id, max_moves)
        return [move for move, score in scored]
    
    def get_action_mask(
        self,
        board: torch.Tensor,
        legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        player_id: int,
        action_size: int = 100,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Create action mask tensor for filtered moves.
        
        Args:
            board: 10x10 board tensor
            legal_moves: List of valid moves
            player_id: 1 or -1
            action_size: Size of action space (100)
            device: PyTorch device
            
        Returns:
            mask: (action_size,) boolean tensor
            filtered_moves: List of moves in mask order
        """
        # Get filtered moves
        filtered_moves = self.get_filtered_moves(board, legal_moves, player_id, action_size)
        
        # Create mask
        mask = torch.zeros(action_size, dtype=torch.bool, device=device)
        
        num_valid = min(len(filtered_moves), action_size)
        mask[:num_valid] = True
        
        return mask, filtered_moves


def create_action_filter(**kwargs) -> ActionFilter:
    """Factory function to create action filter with optional custom parameters."""
    return ActionFilter(**kwargs)
