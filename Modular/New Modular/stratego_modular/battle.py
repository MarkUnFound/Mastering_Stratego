# stratego_modular/battle.py

from .piece import PieceType, PIECE_RANKS, SPECIAL_BATTLES
from typing import Tuple

class BattleResolver:
    """Handles battle resolution between pieces."""
    
    @staticmethod
    def resolve_battle(attacker: PieceType, defender: PieceType, attacker_player: int = 1, defender_player: int = 2) -> int:
        """
        Resolve a battle between two pieces.
        Returns:
            1 if attacker wins
           -1 if defender wins
            0 if both are removed (draw)
        """
        # Prevent cannibalizing of pieces - pieces cannot eat each other if they belong to the same player
        if attacker_player == defender_player and attacker_player != 0:
            return 0  # Draw - both removed (prevent cannibalizing)
            
        # Check special battle rules
        if (attacker, defender) in SPECIAL_BATTLES:
            if SPECIAL_BATTLES[(attacker, defender)]:
                return 1  # Attacker wins
        elif (defender, attacker) in SPECIAL_BATTLES:
            if SPECIAL_BATTLES[(defender, attacker)]:
                return -1  # Defender wins (attacker loses)
                
        # BOMB vs non-MINER
        if attacker != PieceType.MINER and defender == PieceType.BOMB:
            return -1  # Defender (Bomb) wins
        elif attacker == PieceType.BOMB and defender != PieceType.MINER:
            return -1  # Defender wins (attacker Bomb loses)
            
        # Standard rank comparison
        attacker_rank = PIECE_RANKS[attacker]
        defender_rank = PIECE_RANKS[defender]
        
        if attacker_rank > defender_rank:
            return 1  # Attacker wins
        elif defender_rank > attacker_rank:
            return -1  # Defender wins
        else:
            return 0  # Draw - both removed