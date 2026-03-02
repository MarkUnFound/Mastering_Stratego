# Starting Player Randomization
#
# This module implements true randomization of which player moves first.
# Player 1 (value=1) and Player 2 (value=-1) each have a 50% chance of starting.

import random
from typing import List


def get_random_starting_player() -> int:
    """
    Randomly determine which player moves first.
    
    Returns:
        int: 1 for Player 1 first, -1 for Player 2 first
    """
    return 1 if random.random() < 0.5 else -1


def get_batch_starting_players(count: int) -> List[int]:
    """
    Generate a batch of random starting players.
    
    Args:
        count: Number of starting players to generate (e.g., number of lanes)
        
    Returns:
        List[int]: List of starting players (1 or -1)
    """
    return [get_random_starting_player() for _ in range(count)]


def get_weighted_starting_player(p1_probability: float = 0.5) -> int:
    """
    Get a starting player with configurable probability.
    
    Args:
        p1_probability: Probability that Player 1 starts (default 0.5)
        
    Returns:
        int: 1 for Player 1 first, -1 for Player 2 first
    """
    return 1 if random.random() < p1_probability else -1
