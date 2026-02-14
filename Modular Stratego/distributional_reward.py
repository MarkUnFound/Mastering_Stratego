"""
Centralized Reward System for Stratego DQN (C51-Compatible)

This module is the SINGLE SOURCE OF TRUTH for all reward calculations.
Logic previously in environment.py, reward_shaping.py, and legacy distributional modules
is now consolidated here for transparency and consistent scaling.

DESIGN PRINCIPLES (v2 — 2026-02-14):
  1. Terminal and shaping rewards share the same order of magnitude so that
     C51 atoms (support [-10, +10], 51 atoms, ~0.4 per atom) can resolve both.
  2. Piece-loss penalties are rank-weighted (losing Marshal >> losing Scout).
  3. Forward-movement reward is gated on piece rank to prevent suicidal rushes.
  4. Scout penetration zone tightened to actual enemy back rank (2 rows).
  5. Curiosity tracker resets per episode for consistent exploration pressure.
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

# Maximum piece rank for gating forward-movement reward.
# Only pieces with rank <= this get the flag-distance incentive.
# Marshal(10), General(9), Colonel(8) should NOT rush forward.
OFFENSIVE_RANK_THRESHOLD = 6  # Captain and below

@dataclass
class StrategoRewardConfig:
    """Consolidated configuration for all reward components."""

    @classmethod
    def from_training_config(cls):
        """Creates a configuration instance from training_config.py constants."""
        try:
            import training_config
            config = cls()
            return config
        except ImportError:
            return cls()

    # ── Component Weights ───────────────────────────────────────────────
    outcome_weight: float = 1.0     # Win/Loss/Draw
    material_weight: float = 1.0    # Combat rewards (raised from 0.2 so captures are visible to C51)
    epistemic_weight: float = 0.5   # Reveal rewards
    positional_weight: float = 0.3  # Positional guidance

    # ── Terminal Rewards ────────────────────────────────────────────────
    # Scaled to fit C51 support [-10, +10] with room for accumulated shaping.
    win_reward_flag: float = 1.0        # Flag capture = primary objective
    win_reward_depletion: float = 0.5   # Opponent immobilized = secondary
    win_reward: float = 0.75            # Fallback if win_type not specified
    loss_penalty: float = -1.0          # Symmetric loss penalty
    draw_penalty: float = -0.3          # Discourage passive draws

    # Material-advantage draws: adjust draw reward based on piece count
    draw_material_bonus: float = 0.15   # Max bonus for having more pieces at draw

    # ── Per-Step Penalties ──────────────────────────────────────────────
    step_penalty: float = -0.002        # Encourage faster games
    stalemate_penalty: float = -0.05    # When mobility drops below threshold

    # ── Material Rewards (rank-weighted) ────────────────────────────────
    capture_scale: float = 0.1          # Reward = scale × (defender_rank / 10)
    loss_scale: float = -0.1            # Penalty = scale × (our_piece_rank / 10) — FIX #3
    attack_bonus: float = 0.02          # Small bonus for initiating combat
    spy_marsh_bonus: float = 0.3        # Spy killing Marshal
    miner_bomb_bonus: float = 0.15      # Miner defusing Bomb

    # ── Epistemic Rewards ───────────────────────────────────────────────
    reveal_bonus: float = 0.02          # Revealed any enemy piece position
    first_reveal_bonus: float = 0.05    # First time seeing this piece type

    # ── Positional Rewards ──────────────────────────────────────────────
    territory_advance: float = 0.05     # One-time per new row reached
    center_control: float = 0.005       # In center zone
    flag_proximity_bonus: float = 0.1   # Near enemy back rank

    # Flag distance reward — FIX #2 & #9: reduced, applied directly (not × positional_weight),
    # gated on piece rank (only offensive pieces rank ≤ 6)
    flag_distance_reward: float = 0.02

    # Scout penetration bonus — FIX #8: tightened to actual back rank (rows 0-1 / 8-9)
    scout_penetration_bonus: float = 0.1

    # ── Curiosity (exploration bonus) ───────────────────────────────────
    curiosity_weight: float = 0.1
    curiosity_bonus_scale: float = 0.01


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

        # FIX #6: Reset curiosity tracker each episode for consistent exploration
        self.novelty_tracker.reset()

    def __call__(self, previous_state: GameState, action: Optional[Tuple],
                 current_state: GameState, done: bool,
                 winner: Optional[int], info: Dict[str, Any]) -> float:
        """Calculate the total normalized reward for this step."""

        # ── 1. Terminal Outcomes ─────────────────────────────────────────
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
                # Material-advantage draws
                board = current_state.board
                my_pieces = ((board > 0) & (board < 13)).sum().item() if self.player_id == 1 else ((board < 0) & (board > -13)).sum().item()
                enemy_pieces = ((board < 0) & (board > -13)).sum().item() if self.player_id == 1 else ((board > 0) & (board < 13)).sum().item()

                piece_diff = my_pieces - enemy_pieces
                material_bonus = min(max(piece_diff / 10.0, -1.0), 1.0) * self.config.draw_material_bonus

                return self.config.outcome_weight * (self.config.draw_penalty + material_bonus)
            return 0.0

        if action is None:
            return 0.0

        (r_from, c_from), (r_to, c_to) = action

        # ── 1.5 Step Penalty ────────────────────────────────────────────
        reward_components = {'step': self.config.step_penalty}

        # ── 2. Combat / Material Logic (rank-weighted) ──────────────────
        prev_board = previous_state.board
        curr_board = current_state.board

        moving_val = prev_board[r_from, c_from].item()
        target_val = prev_board[r_to, c_to].item()

        was_battle = (target_val != 0 and target_val != 13) and \
                     ((self.player_id == 1 and target_val < 0) or
                      (self.player_id == -1 and target_val > 0))

        if was_battle:
            def _get_rank(val):
                abs_val = abs(int(val))
                if 1 <= abs_val <= 12:
                    return PIECE_RANKS.get(PieceType(abs_val), 5)
                return 5  # Fallback rank

            defender_rank = _get_rank(target_val)
            attacker_rank = _get_rank(moving_val)

            result_val = curr_board[r_to, c_to].item()
            source_val = curr_board[r_from, c_from].item()

            we_won = (self.player_id == 1 and result_val > 0) or (self.player_id == -1 and result_val < 0)
            we_lost = (source_val == 0 and result_val != 0 and not we_won)

            material_r = 0.0

            # Attack bonus: small reward for initiating combat
            material_r += self.config.attack_bonus

            if we_won:
                # Reward proportional to captured piece rank
                material_r += self.config.capture_scale * (defender_rank / 10.0)
                # Strategic bonuses
                if abs(int(moving_val)) == 1 and abs(int(target_val)) == 10:  # Spy vs Marshal
                    material_r += self.config.spy_marsh_bonus
                if abs(int(moving_val)) == 3 and abs(int(target_val)) == 11:  # Miner vs Bomb
                    material_r += self.config.miner_bomb_bonus
            elif we_lost:
                # FIX #3: Penalty proportional to OUR piece rank (losing Marshal >> losing Scout)
                material_r += self.config.loss_scale * (attacker_rank / 10.0)

            reward_components['material'] = material_r * self.config.material_weight

            # ── 3. Epistemic (Info Gain from combat) ────────────────────
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

        # ── 4. Positional / Strategic ───────────────────────────────────
        positional_r = 0.0

        # Territory advance (one-time per row)
        is_new_territory = False
        if self.player_id == 1:  # Red moves up (row indices decrease)
            if r_to < self.max_row_reached:
                self.max_row_reached = r_to
                is_new_territory = True
        else:  # Blue moves down (row indices increase)
            if r_to > self.max_row_reached:
                self.max_row_reached = r_to
                is_new_territory = True

        if is_new_territory:
            positional_r += self.config.territory_advance

        # FIX #9: Flag distance reward — only for offensive pieces (rank ≤ Captain)
        # High-value pieces (Marshal, General, Colonel) should NOT rush forward.
        moving_piece_type = abs(int(prev_board[r_from, c_from].item()))
        piece_rank = PIECE_RANKS.get(PieceType(moving_piece_type), 5) if 1 <= moving_piece_type <= 12 else 5

        if piece_rank <= OFFENSIVE_RANK_THRESHOLD:
            if self.player_id == 1:  # P1 wants to go UP
                if r_to < r_from:
                    # FIX #2: Applied directly, not multiplied by positional_weight
                    reward_components['flag_distance'] = self.config.flag_distance_reward
            else:  # P2 wants to go DOWN
                if r_to > r_from:
                    reward_components['flag_distance'] = self.config.flag_distance_reward

        # Center control
        if 3 <= c_to <= 6 and 4 <= r_to <= 5:
            positional_r += self.config.center_control

        # Flag proximity (corner focus)
        is_p1_near_top = (self.player_id == 1 and r_to <= 1)
        is_p2_near_bot = (self.player_id == -1 and r_to >= 8)
        if is_p1_near_top or is_p2_near_bot:
            corner_mult = 1.0 if c_to in (0, 1, 8, 9) else 0.5
            positional_r += self.config.flag_proximity_bonus * corner_mult

        # FIX #8: Scout penetration — tightened to actual enemy back rank (2 rows)
        if moving_piece_type == PieceType.SCOUT.value:
            if self.player_id == 1 and r_to <= 1:     # Back 2 rows only
                positional_r += self.config.scout_penetration_bonus
            elif self.player_id == -1 and r_to >= 8:  # Back 2 rows only
                positional_r += self.config.scout_penetration_bonus

        # Stalemate prevention
        curr_mob = info.get('num_valid_moves', 0)
        if curr_mob < 5:
            reward_components['stalemate'] = self.config.stalemate_penalty

        reward_components['positional'] = positional_r * self.config.positional_weight

        # ── 5. Intrinsic Curiosity (Novelty Bonus) ──────────────────────
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
