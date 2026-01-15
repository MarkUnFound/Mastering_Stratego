"""
Random Starting Player Utility
Provides functionality to swap player placements for randomizing who starts first.
"""


def swap_placements(p1_placement, p2_placement):
    """
    Swap the placements between two players.
    
    This is used to randomize the starting player by swapping which
    player gets which setup configuration.
    
    Args:
        p1_placement: Player 1's piece placement dictionary
        p2_placement: Player 2's piece placement dictionary
        
    Returns:
        Tuple of (new_p1_placement, new_p2_placement) with swapped values
    """
    return p2_placement, p1_placement
