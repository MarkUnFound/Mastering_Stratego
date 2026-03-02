"""
Heuristic Move Filter for Stratego DQN

Scores and filters legal moves to reduce action space noise.
Uses strategic heuristics to prioritize attacks, advancement, and center control.
"""

import torch
from typing import List, Tuple, Optional
from board import LAKE_SQUARE


class HeuristicMoveFilter:
    """
    Filters legal moves to Top-K using strategic heuristics.
    
    Scoring criteria:
    - Attack moves: +100 (always high priority)
    - Forward advancement: +5 per row toward enemy
    - Center control: +10 (columns 3-6)
    - Retreat penalty: -20
    - Scout distance bonus: +2 per square
    """
    
    def __init__(self, 
                 attack_score: int = 100,
                 advance_score: int = 5,
                 center_score: int = 10,
                 retreat_penalty: int = -20,
                 scout_distance_bonus: int = 2,
                 min_score_threshold: int = 50):
        """
        Initialize filter with scoring parameters.
        
        Args:
            attack_score: Bonus for attacking enemy pieces
            advance_score: Points per row advanced toward enemy
            center_score: Bonus for controlling center columns (3-6)
            retreat_penalty: Penalty for moving backward
            scout_distance_bonus: Bonus per square for long scout moves
            min_score_threshold: Moves above this are always kept
        """
        self.attack_score = attack_score
        self.advance_score = advance_score
        self.center_score = center_score
        self.retreat_penalty = retreat_penalty
        self.scout_distance_bonus = scout_distance_bonus
        self.min_score_threshold = min_score_threshold
    
    def score_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]], 
                   board: torch.Tensor, player_id: int) -> float:
        """
        Score a single move based on strategic heuristics.
        
        Args:
            move: ((r_from, c_from), (r_to, c_to))
            board: 10x10 board tensor
            player_id: 1 or -1
            
        Returns:
            Heuristic score (higher = better)
        """
        (r_from, c_from), (r_to, c_to) = move
        score = 0.0
        
        # Get target piece value
        if isinstance(board, torch.Tensor):
            target_piece = board[r_to, c_to].item()
        else:
            target_piece = board[r_to, c_to]
        
        # 1. ATTACK BONUS - Always high priority
        # Check if target is enemy (non-empty, non-lake, opposite sign)
        is_attack = False
        if target_piece != 0 and target_piece != LAKE_SQUARE:
            if player_id == 1 and target_piece < 0 and target_piece > LAKE_SQUARE:
                is_attack = True
            elif player_id == -1 and target_piece > 0:
                is_attack = True
        
        if is_attack:
            score += self.attack_score
        
        # 2. FORWARD ADVANCEMENT
        # Player 1 (rows 6-9) moves toward row 0
        # Player -1 (rows 0-3) moves toward row 9
        if player_id == 1:
            # Moving up (decreasing row) is advancing
            advancement = r_from - r_to  # Positive if moving up
            score += advancement * self.advance_score
        else:
            # Moving down (increasing row) is advancing
            advancement = r_to - r_from  # Positive if moving down
            score += advancement * self.advance_score
        
        # 3. CENTER CONTROL BONUS
        if 3 <= c_to <= 6:
            score += self.center_score
        
        # 4. RETREAT PENALTY
        if player_id == 1 and r_to > r_from:  # P1 moving backward (down)
            score += self.retreat_penalty
        elif player_id == -1 and r_to < r_from:  # P2 moving backward (up)
            score += self.retreat_penalty
        
        # 5. SCOUT DISTANCE BONUS (long moves)
        distance = abs(r_to - r_from) + abs(c_to - c_from)
        if distance > 1:
            score += distance * self.scout_distance_bonus
        
        return score
    
    def get_filtered_actions(self, board: torch.Tensor, 
                             legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                             player_id: int,
                             max_moves: int = 100) -> List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float]]:
        """
        Filter legal moves to Top-K using heuristic scoring.
        
        Algorithm:
        1. Score all legal moves
        2. Keep all moves with score > min_score_threshold
        3. Fill remaining slots to max_moves with next-best moves
        4. Cap at max_moves total
        
        Args:
            board: 10x10 board tensor
            legal_moves: List of valid moves
            player_id: 1 or -1
            max_moves: Maximum moves to return (default: 100)
            
        Returns:
            List of (move, score) tuples, sorted by score descending
        """
        if not legal_moves:
            return []
        
        # Score all moves
        scored_moves = []
        for move in legal_moves:
            score = self.score_move(move, board, player_id)
            scored_moves.append((move, score))
        
        # Sort by score descending
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy: Keep high-scoring moves + fill to max_moves
        # Step 1: Keep all moves above threshold
        high_priority = [m for m in scored_moves if m[1] > self.min_score_threshold]
        
        # Step 2: If we have fewer than max_moves, fill with next best
        if len(high_priority) >= max_moves:
            # Already have enough high-priority moves
            return high_priority[:max_moves]
        else:
            # Need to fill remaining slots
            remaining_slots = max_moves - len(high_priority)
            lower_priority = [m for m in scored_moves if m[1] <= self.min_score_threshold]
            
            # Combine: all high priority + top lower priority to fill
            result = high_priority + lower_priority[:remaining_slots]
            return result
    
    def get_action_mask(self, board: torch.Tensor,
                       legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       player_id: int,
                       action_size: int = 400,
                       max_moves: int = 100,
                       device: str = 'cuda') -> Tuple[torch.Tensor, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Create action mask tensor for Top-K filtered moves.
        
        Args:
            board: 10x10 board tensor
            legal_moves: List of valid moves
            player_id: 1 or -1
            action_size: Size of action space (400)
            max_moves: Maximum moves to allow (100)
            device: Torch device
            
        Returns:
            Tuple of:
            - mask tensor (action_size,): 0 for valid actions, -inf for masked
            - list of filtered moves (for action decoding)
        """
        # Get filtered moves
        filtered = self.get_filtered_actions(board, legal_moves, player_id, max_moves)
        
        # Create mask initialized to -inf (all masked)
        mask = torch.full((action_size,), float('-inf'), device=device)
        
        # Unmask the filtered moves
        filtered_moves = []
        for move, score in filtered:
            (r_from, c_from), (r_to, c_to) = move
            
            # Calculate distance
            distance = abs(r_to - r_from) + abs(c_to - c_from)
            
            # For Scout moves (dist > 1), map to 1-step direction action
            if distance > 1:
                if r_to != r_from:
                    dr = 1 if r_to > r_from else -1
                    dc = 0
                else:
                    dr = 0
                    dc = 1 if c_to > c_from else -1
            elif distance == 1:
                dr = r_to - r_from
                dc = c_to - c_from
            else:
                continue  # Invalid (distance 0)
                
            # Encode direction to index
            if (dr, dc) == (0, 1):
                dir_idx = 0  # Right
            elif (dr, dc) == (0, -1):
                dir_idx = 1  # Left
            elif (dr, dc) == (1, 0):
                dir_idx = 2  # Down
            elif (dr, dc) == (-1, 0):
                dir_idx = 3  # Up
            else:
                continue  # Invalid direction
            
            action_idx = (r_from * 10 + c_from) * 4 + dir_idx
            
            if 0 <= action_idx < action_size:
                mask[action_idx] = 0.0  # Unmask this action
                filtered_moves.append(move)
        
        return mask, filtered_moves


# Factory function
def create_heuristic_filter(**kwargs) -> HeuristicMoveFilter:
    """Create a HeuristicMoveFilter with optional custom parameters."""
    return HeuristicMoveFilter(**kwargs)
