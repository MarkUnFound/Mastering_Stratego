from typing import Optional

from piece import PieceType, SPECIAL_BATTLES, PIECE_RANKS

HIDDEN_VALUE = -20  # board.HIDDEN_PIECE: an enemy piece whose rank the viewer cannot see
LAKE_VALUE = -13    # board.LAKE_SQUARE

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


def battle_from_values(attacker_value, defender_value) -> Optional[int]:
    """Outcome when the piece with board value `attacker_value` attacks the one with
    `defender_value` (signs ignored; values are PieceType values): 1 attacker wins, -1 defender
    wins, 0 both removed, None when the defender is HIDDEN (its rank is unknown)."""
    if int(defender_value) == HIDDEN_VALUE:
        return None
    return _RESOLVER.resolve_battle(PieceType(abs(int(attacker_value))), PieceType(abs(int(defender_value))))


def is_enemy_value(value, player) -> bool:
    """Whether a square with board value `value` holds an enemy piece of `player` (1 or -1) on
    `player`'s view: HIDDEN on either side's view, or a piece of the other sign (lakes are not)."""
    value = int(value)
    if value == HIDDEN_VALUE:
        return True
    if value == 0 or value == LAKE_VALUE or abs(value) > 12:
        return False
    return (value < 0) if player == 1 else (value > 0)


_RESOLVER = BattleResolver()
