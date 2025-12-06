"""
Reward Shaping Module for Stratego DQN Training

Implements comprehensive reward shaping strategies:
- Material Rewards (Combat Signal)
- Epistemic Rewards (Information Signal)
- Positional Rewards (Potential-Based Reward Shaping)
- Prediction Rewards (Auxiliary Task Support)

Composite Reward: R_total = w1*R_outcome + w2*R_material + w3*R_info + w4*R_position
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

from piece import PieceType, PIECE_RANKS
from board import BOARD_SIZE, LAKE_SQUARE


# Piece value mapping for reward calculations
PIECE_VALUES = {
    PieceType.FLAG: 0,      # Capture = win, not a trade value
    PieceType.SPY: 1,
    PieceType.SCOUT: 2,
    PieceType.MINER: 3,
    PieceType.SERGEANT: 4,
    PieceType.LIEUTENANT: 5,
    PieceType.CAPTAIN: 6,
    PieceType.MAJOR: 7,
    PieceType.COLONEL: 8,
    PieceType.GENERAL: 9,
    PieceType.MARSHAL: 10,
    PieceType.BOMB: 0,      # Can't be traded
}


@dataclass
class RewardWeights:
    """Configurable reward weights for the composite function."""
    outcome: float = 1.0      # Win/Loss/Draw
    material: float = 0.5     # Combat outcomes
    epistemic: float = 0.3    # Information gain
    positional: float = 0.2   # Strategic positioning
    
    @classmethod
    def from_config(cls, config: Dict) -> 'RewardWeights':
        return cls(
            outcome=config.get('REWARD_WEIGHT_OUTCOME', 1.0),
            material=config.get('REWARD_WEIGHT_MATERIAL', 0.5),
            epistemic=config.get('REWARD_WEIGHT_EPISTEMIC', 0.3),
            positional=config.get('REWARD_WEIGHT_POSITIONAL', 0.2),
        )


@dataclass
class BattleInfo:
    """Information about a battle outcome."""
    attacker_type: PieceType
    defender_type: PieceType
    attacker_player: int
    result: int  # 1=attacker wins, -1=defender wins, 0=mutual destruction
    attacker_rank: int
    defender_rank: int


@dataclass
class MoveInfo:
    """Information about a move for reward calculation."""
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    moving_piece_type: PieceType
    moving_piece_rank: int
    current_player: int
    game_phase: str  # "early", "mid", "end"
    turn_count: int
    is_battle: bool
    battle_info: Optional[BattleInfo] = None
    revealed_pieces: List[Tuple[Tuple[int, int], PieceType]] = None


class RewardCalculator:
    """
    Modular reward calculator implementing comprehensive reward shaping.
    
    Usage:
        calculator = RewardCalculator(device, weights)
        reward = calculator.calculate_total_reward(move_info, board, pbs_state)
    """
    
    def __init__(self, device, weights: RewardWeights = None):
        self.device = device
        self.weights = weights or RewardWeights()
        
        # Track game state for epistemic rewards
        self._revealed_this_game = set()
        self._known_scouts = set()
        self._known_bombs = set()
        
    def reset(self):
        """Reset per-game tracking state."""
        self._revealed_this_game = set()
        self._known_scouts = set()
        self._known_bombs = set()
    
    # =========================================================================
    # MATERIAL REWARDS (Combat Signal)
    # =========================================================================
    
    def calculate_material_reward(self, battle_info: BattleInfo, current_player: int) -> float:
        """
        Calculate material reward from combat.
        
        Includes:
        - Rank delta (value difference)
        - Trade-up bonus
        - Marshal preservation
        - Spy/Marshal interactions
        """
        if battle_info is None:
            return 0.0
            
        reward = 0.0
        attacker_val = PIECE_VALUES.get(battle_info.attacker_type, 0)
        defender_val = PIECE_VALUES.get(battle_info.defender_type, 0)
        
        is_current_player_attacker = (battle_info.attacker_player == current_player)
        
        # --- Rank Delta ---
        if battle_info.result == 1:  # Attacker wins
            if is_current_player_attacker:
                reward += defender_val * 0.1  # Captured enemy piece
            else:
                reward -= attacker_val * 0.1  # Lost our defender
                
        elif battle_info.result == -1:  # Defender wins
            if is_current_player_attacker:
                reward -= attacker_val * 0.1  # Lost our attacker
            else:
                reward += attacker_val * 0.1  # Killed enemy attacker
                
        else:  # Mutual destruction
            if is_current_player_attacker:
                reward += (defender_val - attacker_val) * 0.05  # Net exchange
            else:
                reward += (attacker_val - defender_val) * 0.05
        
        # --- Trade-Up Bonus ---
        value_diff = defender_val - attacker_val
        if battle_info.result == 1 and is_current_player_attacker and value_diff > 0:
            reward += 0.1 * (value_diff / 10.0)  # Bonus for favorable trade
        
        # --- Marshal Preservation Penalty ---
        if battle_info.attacker_type == PieceType.MARSHAL:
            if battle_info.result != 1 and is_current_player_attacker:
                reward -= 0.3  # Heavy penalty for losing Marshal
        if battle_info.defender_type == PieceType.MARSHAL:
            if battle_info.result == 1 and not is_current_player_attacker:
                reward -= 0.3  # Heavy penalty for our Marshal being captured
                
        # --- Spy/Marshal Interactions ---
        # Slayer Bonus: Spy kills Marshal
        if (battle_info.attacker_type == PieceType.SPY and 
            battle_info.defender_type == PieceType.MARSHAL and 
            battle_info.result == 1):
            if is_current_player_attacker:
                reward += 0.5  # Massive bonus for Spy killing Marshal
            else:
                reward -= 0.5  # Massive penalty for losing Marshal to Spy
                
        # Suicide Penalty: Spy dies to non-Marshal
        if (battle_info.attacker_type == PieceType.SPY and 
            battle_info.defender_type != PieceType.MARSHAL and
            battle_info.result != 1):
            if is_current_player_attacker:
                reward -= 0.2  # Penalty for wasting Spy
                
        # --- Miner vs Bomb ---
        if (battle_info.attacker_type == PieceType.MINER and 
            battle_info.defender_type == PieceType.BOMB):
            if is_current_player_attacker:
                reward += 0.2  # Good: Miner defuses bomb
        elif (battle_info.attacker_type != PieceType.MINER and 
              battle_info.defender_type == PieceType.BOMB):
            if is_current_player_attacker:
                reward -= 0.15  # Bad: Non-miner hit bomb
        
        return reward
    
    # =========================================================================
    # EPISTEMIC REWARDS (Information Signal)
    # =========================================================================
    
    def calculate_epistemic_reward(self, move_info: MoveInfo, 
                                    pbs_state: Optional[Any] = None) -> float:
        """
        Calculate epistemic reward for information gain.
        
        Includes:
        - Entropy reduction (revealing unknowns)
        - Scout deduction
        - Bomb identification
        - Bluffing incentive
        """
        reward = 0.0
        
        # --- Entropy Reduction (Reveal Reward) ---
        if move_info.revealed_pieces:
            for pos, piece_type in move_info.revealed_pieces:
                if pos not in self._revealed_this_game:
                    self._revealed_this_game.add(pos)
                    # Reward based on piece value (more valuable = more info)
                    piece_val = PIECE_VALUES.get(piece_type, 0)
                    reward += 0.02 + 0.01 * (piece_val / 10.0)
                    
                    # Extra reward for revealing high-value pieces
                    if piece_val >= 8:  # Colonel, General, Marshal
                        reward += 0.03
        
        # --- Scout Deduction ---
        # Moving 2+ squares reveals Scout identity
        (r_from, c_from), (r_to, c_to) = move_info.from_pos, move_info.to_pos
        distance = abs(r_to - r_from) + abs(c_to - c_from)
        
        if distance > 1 and move_info.to_pos not in self._known_scouts:
            # Enemy piece identified as Scout via movement
            self._known_scouts.add(move_info.to_pos)
            reward += 0.03  # Small reward for Scout identification
            
        # --- Bomb Identification ---
        # Reward for identifying and remembering bomb locations
        if move_info.battle_info:
            if move_info.battle_info.defender_type == PieceType.BOMB:
                if move_info.to_pos not in self._known_bombs:
                    self._known_bombs.add(move_info.to_pos)
                    # Reward for discovery (even if we lost a piece)
                    reward += 0.02
                    
        # --- Bluffing Incentive ---
        # Moving low-rank piece near high-rank enemy (causing potential retreat)
        # This is harder to measure without opponent behavior, so we approximate
        if move_info.moving_piece_rank <= 4:  # Low-rank piece
            # Check if moving toward enemy territory aggressively
            if move_info.current_player == 1 and r_to < r_from:  # P1 advancing
                reward += 0.01
            elif move_info.current_player == -1 and r_to > r_from:  # P2 advancing
                reward += 0.01
        
        return reward
    
    # =========================================================================
    # POSITIONAL REWARDS (Potential-Based Reward Shaping)
    # =========================================================================
    
    def calculate_positional_reward(self, move_info: MoveInfo, 
                                     board: torch.Tensor,
                                     enemy_flag_pos: Optional[Tuple[int, int]] = None) -> float:
        """
        Calculate positional reward using PBRS principles.
        
        Includes:
        - Manhattan distance to enemy base
        - Center control
        - Miner positioning (late game)
        - Flag proximity curiosity
        """
        reward = 0.0
        (r_from, c_from), (r_to, c_to) = move_info.from_pos, move_info.to_pos
        
        # --- Forward Movement (Manhattan Distance to Base) ---
        if move_info.current_player == 1:
            # P1 wants to reduce row number (advance toward row 0)
            forward_progress = r_from - r_to
            enemy_base_row = 0
        else:
            # P2 wants to increase row number (advance toward row 9)
            forward_progress = r_to - r_from
            enemy_base_row = 9
            
        # Higher value pieces get more reward for advancing
        if forward_progress > 0:
            piece_val = move_info.moving_piece_rank / 10.0
            phase_mult = 1.0 if move_info.game_phase == "early" else 0.5
            reward += 0.02 * forward_progress * (1 + piece_val * 0.5) * phase_mult
            
        # --- Center Control ---
        # Center rows (4-5) excluding lakes
        if 4 <= r_to <= 5:
            is_lake = (c_to in [2, 3, 6, 7])
            if not is_lake:
                reward += 0.015
                
        # Center columns (3-6) bonus
        if 3 <= c_to <= 6:
            reward += 0.005
            
        # --- Miner Positioning (Late Game) ---
        if (move_info.game_phase == "end" and 
            move_info.moving_piece_type == PieceType.MINER):
            # Miners should be near enemy flag area in endgame
            if move_info.current_player == 1:
                # P1 miners want low rows
                if r_to <= 3:
                    reward += 0.03
            else:
                # P2 miners want high rows
                if r_to >= 6:
                    reward += 0.03
                    
        # --- Flag Proximity / Curiosity Reward ---
        if enemy_flag_pos:
            dist_before = abs(r_from - enemy_flag_pos[0]) + abs(c_from - enemy_flag_pos[1])
            dist_after = abs(r_to - enemy_flag_pos[0]) + abs(c_to - enemy_flag_pos[1])
            
            if dist_after < dist_before:
                reward += 0.02 * (dist_before - dist_after)
                
            # Bonus for being very close
            if dist_after <= 2:
                reward += 0.02
                
        # --- Attacking Stationary Pieces (Flag/Bomb Curiosity) ---
        # This would require tracking movement history, which we approximate
        # by giving bonus for attacking pieces in back rows
        if move_info.is_battle:
            if move_info.current_player == 1 and r_to <= 1:  # P1 attacking back row
                reward += 0.02
            elif move_info.current_player == -1 and r_to >= 8:  # P2 attacking back row
                reward += 0.02
        
        return reward
    
    # =========================================================================
    # PREDICTION REWARDS (Auxiliary Tasks)
    # =========================================================================
    
    def calculate_prediction_reward(self, predictions: Dict, actuals: Dict) -> float:
        """
        Calculate prediction-based reward (for auxiliary task training).
        
        This is typically used as an auxiliary loss rather than environment reward,
        but we provide a small reward signal for correct predictions.
        """
        reward = 0.0
        
        # Next-move prediction accuracy
        if 'next_move' in predictions and 'actual_move' in actuals:
            if predictions['next_move'] == actuals['actual_move']:
                reward += 0.01
                
        # Rank prediction accuracy (PBS integration)
        if 'rank_probs' in predictions and 'actual_rank' in actuals:
            predicted_rank = np.argmax(predictions['rank_probs'])
            if predicted_rank == actuals['actual_rank']:
                reward += 0.01
                
        return reward
    
    # =========================================================================
    # COMPOSITE REWARD
    # =========================================================================
    
    def calculate_total_reward(self, move_info: MoveInfo, 
                                board: torch.Tensor,
                                pbs_state: Optional[Any] = None,
                                enemy_flag_pos: Optional[Tuple[int, int]] = None,
                                outcome_reward: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total composite reward.
        
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        # Calculate components
        r_material = self.calculate_material_reward(
            move_info.battle_info, move_info.current_player
        )
        r_epistemic = self.calculate_epistemic_reward(move_info, pbs_state)
        r_positional = self.calculate_positional_reward(move_info, board, enemy_flag_pos)
        
        # Weighted sum
        total = (
            self.weights.outcome * outcome_reward +
            self.weights.material * r_material +
            self.weights.epistemic * r_epistemic +
            self.weights.positional * r_positional
        )
        
        # Clamp to reasonable range
        total = max(-2.0, min(2.0, total))
        
        breakdown = {
            'outcome': outcome_reward,
            'material': r_material,
            'epistemic': r_epistemic,
            'positional': r_positional,
            'total': total
        }
        
        return total, breakdown


# =============================================================================
# HELPER FUNCTION FOR ENVIRONMENT INTEGRATION
# =============================================================================

def create_move_info(action: Tuple, 
                     moving_piece_value: int,
                     target_piece_value: int,
                     current_player: int,
                     turn_count: int,
                     battle_result: Optional[int] = None,
                     revealed_pieces: List = None) -> MoveInfo:
    """
    Helper to create MoveInfo from environment step data.
    """
    (r_from, c_from), (r_to, c_to) = action
    
    moving_type = PieceType(abs(moving_piece_value))
    moving_rank = abs(moving_piece_value)
    
    # Determine game phase
    if turn_count < 50:
        phase = "early"
    elif turn_count < 200:
        phase = "mid"
    else:
        phase = "end"
    
    is_battle = target_piece_value != 0 and abs(target_piece_value) < 13
    
    battle_info = None
    if is_battle and battle_result is not None:
        target_type = PieceType(abs(target_piece_value))
        attacker_player = 1 if moving_piece_value > 0 else -1
        battle_info = BattleInfo(
            attacker_type=moving_type,
            defender_type=target_type,
            attacker_player=attacker_player,
            result=battle_result,
            attacker_rank=moving_rank,
            defender_rank=abs(target_piece_value)
        )
    
    return MoveInfo(
        from_pos=(r_from, c_from),
        to_pos=(r_to, c_to),
        moving_piece_type=moving_type,
        moving_piece_rank=moving_rank,
        current_player=current_player,
        game_phase=phase,
        turn_count=turn_count,
        is_battle=is_battle,
        battle_info=battle_info,
        revealed_pieces=revealed_pieces or []
    )
