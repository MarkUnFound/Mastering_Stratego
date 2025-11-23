# Random Starting Player System
#
# This module implements functionality to randomize which player goes first in Stratego games.
# This helps balance the action counts between Agent 1 and Agent 2 during training.
#
# Usage:
#   from random_starting_player import should_swap_players, swap_placements
#   
#   if should_swap_players():
#       p1_placement, p2_placement = swap_placements(p1_placement, p2_placement)

import random
from typing import List, Tuple
from piece import PieceType

def should_swap_players() -> bool:
    """
    Determine randomly if players should be swapped (50% chance).
    
    Returns:
        bool: True if players should be swapped, False otherwise
    """
    return random.random() < 0.5

def get_batch_swap_decisions(count: int) -> List[bool]:
    """
    Generate a batch of random swap decisions.
    
    Args:
        count: Number of decisions to generate
        
    Returns:
        List[bool]: List of booleans indicating whether to swap for each instance
    """
    return [random.random() < 0.5 for _ in range(count)]

def swap_placements(p1_placement: List[Tuple[PieceType, Tuple[int, int]]], 
                   p2_placement: List[Tuple[PieceType, Tuple[int, int]]]) -> Tuple[
                       List[Tuple[PieceType, Tuple[int, int]]], 
                       List[Tuple[PieceType, Tuple[int, int]]]]:
    """
    Swap player placements by mirroring positions across the center of the board.
    
    This effectively swaps which agent starts first by swapping their physical positions.
    Player 1's pieces (rows 6-9) become Player 2's (rows 0-3) and vice versa.
    
    Args:
        p1_placement: List of (piece_type, position) tuples for Player 1
        p2_placement: List of (piece_type, position) tuples for Player 2
        
    Returns:
        Tuple[p2_in_p1_zone, p1_in_p2_zone]: Swapped placements
    """
    def mirror_position(pos: Tuple[int, int]) -> Tuple[int, int]:
        """Mirror a position across the board center (row 4.5)."""
        r, c = pos
        # Mirror row across center: row 9 -> 0, row 8 -> 1, etc.
        mirrored_r = 9 - r
        return (mirrored_r, c)
    
    # Mirror P1's placement (rows 6-9) to P2's zone (rows 0-3)
    p1_mirrored = [(piece, mirror_position(pos)) for piece, pos in p1_placement]
    
    # Mirror P2's placement (rows 0-3) to P1's zone (rows 6-9)
    p2_mirrored = [(piece, mirror_position(pos)) for piece, pos in p2_placement]
    
    # Return swapped: P2's setup goes where P1 was, P1's setup goes where P2 was
    return p2_mirrored, p1_mirrored
