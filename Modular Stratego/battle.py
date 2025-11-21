from piece import PieceType, SPECIAL_BATTLES, PIECE_RANKS

class BattleResolver:
    def resolve_battle(self, attacker: PieceType, defender: PieceType, attacker_player: int = 1, defender_player: int = 2) -> int:
        """
        Resolve a battle between two pieces.
        Assumes attacker and defender are enemies (validated by get_valid_moves).
        Returns:
            1 if attacker wins
           -1 if defender wins
            0 if both are removed (draw)
        """
            
        # Check special battle rules (only apply when the special piece is the attacker)
        # SPY vs MARSHAL: SPY wins only when SPY attacks, not when MARSHAL attacks
        if (attacker, defender) in SPECIAL_BATTLES:
            if SPECIAL_BATTLES[(attacker, defender)]:
                return 1  # Attacker wins
        # We don't check (defender, attacker) because special rules only apply
        # when the special piece attacks (e.g., SPY must attack MARSHAL to win)
                
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