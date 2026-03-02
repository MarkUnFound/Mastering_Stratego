"""
Stratego Piece Value Calculator - Analytical Method

Calculates piece values using probability theory and game theory,
without requiring brute-force simulation.

Mathematical Foundation:
1. Combat Dominance: Probability of winning against each enemy type
2. Expected Value: Weighted by how often you'll encounter each type
3. Special Abilities: Strategic value of unique capabilities
4. Scarcity: Fewer pieces = higher individual value
5. Mobility: Movement options increase tactical flexibility
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from enum import IntEnum
import math


class PieceType(IntEnum):
    """Stratego piece types with their ranks."""
    FLAG = 0       # F - Cannot move, game ends if captured
    SPY = 1        # S - Rank 1, kills Marshal when attacking
    SCOUT = 2      # 2 - Can move multiple squares
    MINER = 3      # 3 - Can defuse bombs
    SERGEANT = 4   # 4
    LIEUTENANT = 5 # 5
    CAPTAIN = 6    # 6
    MAJOR = 7      # 7
    COLONEL = 8    # 8
    GENERAL = 9    # 9
    MARSHAL = 10   # 10 - Highest rank
    BOMB = 11      # B - Immovable, kills attackers (except Miner)


# Standard Stratego piece counts per player
PIECE_COUNTS = {
    PieceType.FLAG: 1,
    PieceType.SPY: 1,
    PieceType.SCOUT: 8,
    PieceType.MINER: 5,
    PieceType.SERGEANT: 4,
    PieceType.LIEUTENANT: 4,
    PieceType.CAPTAIN: 4,
    PieceType.MAJOR: 3,
    PieceType.COLONEL: 2,
    PieceType.GENERAL: 1,
    PieceType.MARSHAL: 1,
    PieceType.BOMB: 6,
}

TOTAL_PIECES = sum(PIECE_COUNTS.values())  # 40 pieces per player


def get_combat_result(attacker: PieceType, defender: PieceType) -> Tuple[float, float, float]:
    """
    Returns (P(attacker_wins), P(defender_wins), P(both_die))
    
    Combat Rules:
    - Higher rank wins
    - Equal rank: both die
    - Spy beats Marshal ONLY when attacking
    - Miner beats Bomb
    - Bomb kills all attackers except Miner
    - Flag is captured by any attacker
    """
    # Flag is always captured
    if defender == PieceType.FLAG:
        return (1.0, 0.0, 0.0)
    
    # Bombs and Flags can't attack
    if attacker in [PieceType.FLAG, PieceType.BOMB]:
        return (0.0, 0.0, 0.0)  # Can't attack
    
    # Bomb defense
    if defender == PieceType.BOMB:
        if attacker == PieceType.MINER:
            return (1.0, 0.0, 0.0)  # Miner defuses
        else:
            return (0.0, 1.0, 0.0)  # Attacker dies
    
    # Spy special case - only when attacking
    if attacker == PieceType.SPY and defender == PieceType.MARSHAL:
        return (1.0, 0.0, 0.0)  # Spy kills Marshal
    
    # Normal combat - compare ranks
    if attacker.value > defender.value:
        return (1.0, 0.0, 0.0)  # Attacker wins
    elif attacker.value < defender.value:
        return (0.0, 1.0, 0.0)  # Defender wins
    else:
        return (0.0, 0.0, 1.0)  # Both die (equal rank)


@dataclass
class AnalyticalPieceValue:
    """Stores the calculated components of piece value."""
    piece_type: PieceType
    combat_dominance: float    # % of pieces it can beat
    expected_combat_value: float  # Expected value from random encounters
    special_ability: float     # Value of unique abilities
    scarcity_factor: float     # Rarity bonus
    mobility_factor: float     # Movement capability
    total_value: float         # Combined normalized value


def calculate_combat_dominance(piece: PieceType) -> float:
    """
    Calculate what percentage of enemy pieces this piece can beat.
    
    CDS = Σ P(beat enemy_i) × Count(enemy_i) / Total_enemy_pieces
    """
    if piece in [PieceType.FLAG, PieceType.BOMB]:
        return 0.0  # Can't move/attack
    
    total_beatable_weight = 0.0
    
    for enemy_type, count in PIECE_COUNTS.items():
        p_win, p_lose, p_draw = get_combat_result(piece, enemy_type)
        total_beatable_weight += p_win * count
    
    return total_beatable_weight / TOTAL_PIECES


def calculate_survival_rate(piece: PieceType) -> float:
    """
    Calculate expected survival rate when defending against random attacker.
    
    P(survive) = Σ P(win as defender) × P(enemy attacks with type_i)
    """
    if piece == PieceType.FLAG:
        return 0.0  # Always captured
    if piece == PieceType.BOMB:
        # Bombs kill most attackers, only Miner survives
        miner_attack_prob = PIECE_COUNTS[PieceType.MINER] / TOTAL_PIECES
        return 1.0 - miner_attack_prob  # Very high survival
    
    survival_weight = 0.0
    
    for attacker_type, count in PIECE_COUNTS.items():
        if attacker_type in [PieceType.FLAG, PieceType.BOMB]:
            continue  # Can't attack
        
        p_atk_win, p_def_win, p_draw = get_combat_result(attacker_type, piece)
        # Defender survives if attacker loses
        survival_weight += p_def_win * count
    
    # Normalize by movable pieces (exclude Flag and Bomb)
    movable_pieces = TOTAL_PIECES - PIECE_COUNTS[PieceType.FLAG] - PIECE_COUNTS[PieceType.BOMB]
    return survival_weight / movable_pieces


def calculate_expected_combat_value(piece: PieceType, base_values: Dict[PieceType, float]) -> float:
    """
    Calculate expected value from combat encounters.
    
    ECV = Σ [P(win) × Value(enemy) × Count(enemy)] 
        - Σ [P(lose) × Value(self) × Count(enemy)]
    
    This answers: "On average, do I gain or lose value from fighting?"
    """
    if piece in [PieceType.FLAG, PieceType.BOMB]:
        return 0.0
    
    expected_gain = 0.0
    expected_loss = 0.0
    self_value = base_values.get(piece, 1.0)
    
    for enemy_type, count in PIECE_COUNTS.items():
        enemy_value = base_values.get(enemy_type, 1.0)
        p_win, p_lose, p_draw = get_combat_result(piece, enemy_type)
        
        # Weight by encounter probability (count / total)
        encounter_prob = count / TOTAL_PIECES
        
        expected_gain += p_win * enemy_value * encounter_prob
        expected_loss += (p_lose + p_draw) * self_value * encounter_prob
    
    return expected_gain - expected_loss


def calculate_special_ability_value(piece: PieceType) -> float:
    """
    Calculate value of special abilities.
    
    - Miner: Can defuse bombs (strategic necessity)
    - Spy: Can kill Marshal (game-changing)
    - Scout: Extended movement (tactical flexibility)
    - Marshal: Highest combat power
    """
    special_values = {
        PieceType.MINER: 2.0,   # Essential for bomb defusal
        PieceType.SPY: 1.5,     # Marshal assassination potential
        PieceType.SCOUT: 1.0,   # Extended movement
        PieceType.MARSHAL: 0.5, # Prestige/power (but vulnerable to Spy)
    }
    return special_values.get(piece, 0.0)


def calculate_mobility_factor(piece: PieceType) -> float:
    """
    Calculate mobility value.
    
    Mobility = ability to move and control space
    - Scout: Can move multiple squares (high mobility)
    - Normal pieces: 1 square in 4 directions
    - Flag/Bomb: Cannot move (0 mobility)
    """
    if piece in [PieceType.FLAG, PieceType.BOMB]:
        return 0.0
    elif piece == PieceType.SCOUT:
        return 2.0  # Can slide across board
    else:
        return 1.0  # Standard movement


def calculate_scarcity_factor(piece: PieceType) -> float:
    """
    Calculate scarcity bonus.
    
    Fewer pieces = higher individual importance
    SF = log(max_count / count) + 1
    """
    count = PIECE_COUNTS[piece]
    max_count = max(PIECE_COUNTS.values())  # 8 (Scouts)
    
    # Logarithmic scaling: 1 piece is much more valuable than 8
    return math.log(max_count / count) + 1


def calculate_all_piece_values() -> Dict[PieceType, AnalyticalPieceValue]:
    """
    Calculate analytical piece values using iterative refinement.
    
    Method:
    1. Start with base values (combat dominance)
    2. Calculate expected combat value based on current values
    3. Iterate until values converge
    """
    # Phase 1: Calculate base metrics (no iteration needed)
    base_metrics = {}
    for piece in PieceType:
        base_metrics[piece] = {
            'combat_dominance': calculate_combat_dominance(piece),
            'survival_rate': calculate_survival_rate(piece),
            'special_ability': calculate_special_ability_value(piece),
            'mobility': calculate_mobility_factor(piece),
            'scarcity': calculate_scarcity_factor(piece),
        }
    
    # Phase 2: Initial values = combat dominance + special + mobility
    current_values = {}
    for piece in PieceType:
        m = base_metrics[piece]
        current_values[piece] = (
            m['combat_dominance'] * 5 +  # Scale to 0-5
            m['special_ability'] +
            m['mobility'] * 0.5
        )
    
    # Phase 3: Calculate final values (no iteration needed for analytical)
    for piece in PieceType:
        m = base_metrics[piece]
        
        # Combat Power: Higher rank = more enemies you can beat
        combat_power = m['combat_dominance'] * 10  # Scale 0-10
        
        # Defensive Power: How likely to survive attacks
        defensive_power = m['survival_rate'] * 5   # Scale 0-5
        
        # Special abilities add fixed value
        special = m['special_ability']
        
        # Mobility increases tactical options
        mobility = m['mobility'] * 0.5
        
        # Scarcity: fewer pieces = more valuable each one is
        scarcity = m['scarcity'] * 0.5
        
        # Combine: Combat dominance is most important in Stratego
        current_values[piece] = (
            combat_power * 0.40 +      # 40% - ability to beat others
            defensive_power * 0.25 +   # 25% - ability to survive
            special * 0.20 +           # 20% - unique abilities
            mobility * 0.10 +          # 10% - movement options  
            scarcity * 0.05            # 5% - rarity bonus
        )
    
    # Phase 4: Normalize so Scout = 1.0
    scout_value = current_values[PieceType.SCOUT]
    if scout_value == 0:
        scout_value = 1.0
    
    # Build result objects
    results = {}
    for piece in PieceType:
        m = base_metrics[piece]
        results[piece] = AnalyticalPieceValue(
            piece_type=piece,
            combat_dominance=m['combat_dominance'],
            expected_combat_value=calculate_expected_combat_value(piece, current_values),
            special_ability=m['special_ability'],
            scarcity_factor=m['scarcity'],
            mobility_factor=m['mobility'],
            total_value=current_values[piece] / scout_value
        )
    
    return results


def print_analytical_values():
    """Print a formatted report of analytical piece values."""
    values = calculate_all_piece_values()
    
    print("\n" + "=" * 70)
    print("STRATEGO ANALYTICAL PIECE VALUES")
    print("Calculated using probability theory (no simulation required)")
    print("=" * 70)
    
    print(f"\n{'Piece':<12} {'Value':>7} {'Combat%':>8} {'Survive%':>9} {'Special':>8} {'Scarce':>7}")
    print("-" * 55)
    
    # Sort by value
    sorted_pieces = sorted(values.items(), key=lambda x: x[1].total_value, reverse=True)
    
    for piece, data in sorted_pieces:
        survival = calculate_survival_rate(piece)
        print(f"{piece.name:<12} {data.total_value:>7.2f} {data.combat_dominance*100:>7.1f}% "
              f"{survival*100:>8.1f}% {data.special_ability:>8.1f} {data.scarcity_factor:>7.2f}")
    
    print("\n" + "-" * 55)
    print("\nChess-style notation (Scout = 1 point):")
    print("-" * 40)
    
    for piece, data in sorted_pieces:
        if piece not in [PieceType.FLAG, PieceType.BOMB]:
            print(f"  {piece.name:<12} = {data.total_value:.1f}")
    
    print("\n" + "=" * 70)
    print("MATHEMATICAL FORMULAS USED:")
    print("=" * 70)
    print("""
1. Combat Dominance (CD):
   CD(p) = Σ P(p beats enemy_i) × Count(enemy_i) / Total_pieces

2. Survival Rate (SR):
   SR(p) = Σ P(p survives attack from enemy_i) × P(encounter enemy_i)

3. Expected Combat Value (ECV):  
   ECV(p) = Σ [P(win) × Value(enemy)] - Σ [P(lose) × Value(self)]

4. Scarcity Factor (SF):
   SF(p) = log(max_count / count) + 1

5. Final Value:
   V(p) = 0.30×CD + 0.20×SR + 0.20×ECV + 0.15×Special + 0.05×Mobility + 0.10×SF
""")


if __name__ == "__main__":
    print_analytical_values()
