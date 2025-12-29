"""
Centralized Reward System for Stratego DQN (C51-Compatible)

This module is the SINGLE SOURCE OF TRUTH for all reward calculations.
Logic previously in environment.py, reward_shaping.py, and legacy distributional modules
is now consolidated here for transparency and consistent scaling.
"""

import torch
from typing import Tuple, Optional, Dict, Any, Set, List
from dataclasses import dataclass
from piece import PieceType
from game_state import GameState

# Intrinsic Curiosity for exploration
from intrinsic_curiosity import StateNoveltyTracker

# Piece rank values for reward calculations (1-10 scale)
PIECE_RANKS = {
    PieceType.FLAG: 0,
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
    PieceType.BOMB: 0,
}

@dataclass
class StrategoRewardConfig:
    """Consolidated configuration for all reward components."""

    @classmethod
    def from_training_config(cls):
        """Creates a configuration instance from training_config.py constants."""
        try:
            import training_config
            # Create default instance
            config = cls()
            
            # Application of REWARD_SCALE if present
            if hasattr(training_config, 'REWARD_SCALE'):
                scale = getattr(training_config, 'REWARD_SCALE')
                # For now, we trust the defaults in this file as the source of truth,
                # but we respect the global scale if defined.
                pass 
                
            return config
        except ImportError:
            return cls()
    # Weights for component mixing
    outcome_weight: float = 1.0     # Win/Loss/Draw
    material_weight: float = 0.2    # Combat rewards (REDUCED from 0.5 to discourage grinding)
    epistemic_weight: float = 0.1   # Reduced to favor terminal outcome
    positional_weight: float = 0.05 # Strictly guidance, not a primary objective
    
    # Terminal rewards - DIFFERENTIATED by win type
    win_reward_flag: float = 15.0   # Flag capture = primary objective (BOOSTED)
    win_reward_depletion: float = 5.0  # Opponent immobilized = secondary
    win_reward: float = 10.0        # Fallback if win_type not specified
    loss_penalty: float = -10.0     # Symmetric loss penalty
    draw_penalty: float = -1.0      # Small penalty for draw (better than loss)
    
    # Per-step penalties (scaled for 1000 step max)
    step_penalty: float = -0.0001   # Slightly increased to encourage faster wins
    step_penalty_mid: float = -0.0001 # consistency
    step_penalty_late: float = -0.0001 # consistency
    stalemate_penalty: float = -0.05 # Penalty when mobility is suddenly restricted
    
    # Material rewards
    capture_scale: float = 0.05     # Enemy rank * scale (Normalized)
    loss_scale: float = -0.05       # Flat piece loss penalty (negated for opponent)
    spy_marsh_bonus: float = 0.3    # Extra for Spy killing Marshal
    miner_bomb_bonus: float = 0.15  # Extra for Miner defusing Bomb
    
    # Epistemic rewards
    reveal_bonus: float = 0.01      # Revealed any enemy rank
    first_reveal_bonus: float = 0.02 # First time seeing this piece type
    
    # Positional rewards
    territory_advance: float = 0.05 # One-time reward for reaching a new row
    center_control: float = 0.005
    flag_proximity_bonus: float = 0.05
    
    # Scout penetration bonus (encourage flag hunting)
    scout_penetration_bonus: float = 0.3  # Scout reaching enemy back rank
    
    # Intrinsic Curiosity (exploration bonus)
    curiosity_weight: float = 0.1   # Weight for novelty bonus
    curiosity_bonus_scale: float = 0.01  # Max bonus per novel state


class UnifiedRewardShaper:
    """Standardized reward calculator for all Stratego RL components."""
    
    def __init__(self, player_id: int, config: Optional[StrategoRewardConfig] = None, device: str = 'cuda'):
        self.player_id = player_id
        self.config = config or StrategoRewardConfig()
        self.device = device
        
        # Initialize novelty tracker for intrinsic curiosity
        self.novelty_tracker = StateNoveltyTracker(
            bonus_scale=self.config.curiosity_bonus_scale,
            decay_rate=0.5,
            device=device
        )
        self.reset()
        
    def reset(self):
        """Reset per-episode tracking."""
        self.revealed_types: Set[PieceType] = set()
        self.revealed_positions: Set[Tuple[int, int]] = set()
        self.max_row_reached: int = 10 if self.player_id == 1 else -1
        # Note: We don't reset novelty_tracker - it persists across episodes
        # to encourage exploration of truly novel states 
        
    def __call__(self, previous_state: GameState, action: Optional[Tuple], 
                 current_state: GameState, done: bool, 
                 winner: Optional[int], info: Dict[str, Any]) -> float:
        """Calculate the total normalized reward for this step."""
        
        # 1. Terminal Outcomes - differentiated by win type
        if done:
            if winner == self.player_id:
                win_type = info.get('win_type', 'unknown')
                if win_type == 'flag_capture':
                    return self.config.outcome_weight * self.config.win_reward_flag
                elif win_type == 'no_moves':
                    return self.config.outcome_weight * self.config.win_reward_depletion
                else:
                    return self.config.outcome_weight * self.config.win_reward
            elif winner == -self.player_id:
                return self.config.outcome_weight * self.config.loss_penalty
            elif winner == 0:
                return self.config.outcome_weight * self.config.draw_penalty
            return 0.0

        if action is None:
            return 0.0

        (r_from, c_from), (r_to, c_to) = action
        
        # 1.5 Linear Step Penalty
        current_step_penalty = self.config.step_penalty
            
        reward_components = {'step': current_step_penalty}
        
        # 2. Combat / Material Logic
        prev_board = previous_state.board
        curr_board = current_state.board
        
        # Get piece values safely
        moving_val = prev_board[r_from, c_from].item()
        target_val = prev_board[r_to, c_to].item()
        
        # If target was enemy piece, it's a battle
        was_battle = (target_val != 0 and target_val != 13) and \
                     ((self.player_id == 1 and target_val < 0) or \
                      (self.player_id == -1 and target_val > 0))
        
        if was_battle:
            # Safely get ranks (handling 0, HIDDEN, LAKE)
            def _get_rank(val):
                abs_val = abs(int(val))
                if 1 <= abs_val <= 12:
                    return PIECE_RANKS.get(PieceType(abs_val), 5)
                return 5 # Fallback rank
            
            defender_rank = _get_rank(target_val)
            attacker_rank = _get_rank(moving_val)
            
            # Determine outcome from current board
            result_val = curr_board[r_to, c_to].item()
            source_val = curr_board[r_from, c_from].item()
            
            we_won = (self.player_id == 1 and result_val > 0) or (self.player_id == -1 and result_val < 0)
            we_lost = (source_val == 0 and result_val != 0 and not we_won)
            
            material_r = 0.0
            if we_won:
                material_r += self.config.capture_scale * (defender_rank / 10.0)
                # Strategic bonuses
                if abs(int(moving_val)) == 1 and abs(int(target_val)) == 10: # Spy vs Marshal
                    material_r += self.config.spy_marsh_bonus
                if abs(int(moving_val)) == 3 and abs(int(target_val)) == 11: # Miner vs Bomb
                    material_r += self.config.miner_bomb_bonus
            elif we_lost:
                material_r += self.config.loss_scale
            
            reward_components['material'] = material_r * self.config.material_weight
            
            # 3. Epistemic (Info Gain)
            epistemic_r = 0.0
            if (r_to, c_to) not in self.revealed_positions:
                self.revealed_positions.add((r_to, c_to))
                epistemic_r += self.config.reveal_bonus
                
            def_abs = abs(int(target_val))
            if 1 <= def_abs <= 12:
                def_type = PieceType(def_abs)
                if def_type not in self.revealed_types:
                    self.revealed_types.add(def_type)
                    epistemic_r += self.config.first_reveal_bonus
                
            reward_components['epistemic'] = epistemic_r * self.config.epistemic_weight
            
        # 4. Positional / Strategic (State-based, not move-based)
        positional_r = 0.0
        
        # Advance toward enemy base (One-time reward per row)
        is_new_territory = False
        if self.player_id == 1: # Red (moves up, row indices decrease)
            if r_to < self.max_row_reached:
                self.max_row_reached = r_to
                is_new_territory = True
        else: # Blue (moves down, row indices increase)
            if r_to > self.max_row_reached:
                self.max_row_reached = r_to
                is_new_territory = True
        
        if is_new_territory:
            positional_r += self.config.territory_advance
            
        # Center control
        if 3 <= c_to <= 6 and 4 <= r_to <= 5:
            positional_r += self.config.center_control
            
        # Flag proximity (Corner focus)
        is_p1_near_top = (self.player_id == 1 and r_to <= 1)
        is_p2_near_bot = (self.player_id == -1 and r_to >= 8)
        if is_p1_near_top or is_p2_near_bot:
            corner_mult = 1.0 if c_to in (0, 1, 8, 9) else 0.5
            positional_r += self.config.flag_proximity_bonus * corner_mult
        
        # Scout penetration bonus (encourage flag hunting)
        moving_piece_type = abs(int(prev_board[r_from, c_from].item()))
        if moving_piece_type == PieceType.SCOUT.value:
            # P1 scouts reaching rows 0-3 (enemy back rank)
            if self.player_id == 1 and r_to <= 3:
                positional_r += self.config.scout_penetration_bonus
            # P2 scouts reaching rows 6-9 (enemy back rank)
            elif self.player_id == -1 and r_to >= 6:
                positional_r += self.config.scout_penetration_bonus
            
        # Stalemate prevention (Penalty only, no positive mobility bonus)
        curr_mob = info.get('num_valid_moves', 0)
        if curr_mob < 5: # Critical mobility threshold
            reward_components['stalemate'] = self.config.stalemate_penalty
            
        reward_components['positional'] = positional_r * self.config.positional_weight
        
        # 5. Intrinsic Curiosity (Novelty Bonus)
        state_tensor = current_state.board
        novelty_bonus = self.novelty_tracker.get_novelty_bonus(state_tensor)
        reward_components['curiosity'] = novelty_bonus * self.config.curiosity_weight
        
        total_reward = sum(reward_components.values())
        return total_reward

def create_unified_reward_shaper(player_id: int = 1, config: Optional[StrategoRewardConfig] = None, device: str = 'cuda'):
    """Factory function for creating the shaper."""
    return UnifiedRewardShaper(player_id=player_id, config=config, device=device)

# Legacy aliases for compatibility
DistributionalRewardConfig = StrategoRewardConfig
create_distributional_reward_wrapper = create_unified_reward_shaper
