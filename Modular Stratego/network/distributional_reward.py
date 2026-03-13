"""
Centralized Reward System for Stratego DQN (C51-Compatible)

This module is the SINGLE SOURCE OF TRUTH for all reward calculations.

DESIGN PRINCIPLES (v3 — 2026-02-22):
  1. Potential-Based Reward Shaping (PBRS) replaces all additive shaping.
     Shaping reward = γ·Φ(s') − Φ(s) guarantees optimal-policy invariance
     and gives ZERO reward for stationary behavior (Ng et al., 1999;
     extended by Potential-Based Intrinsic Motivation, AAMAS 2024).
  2. Terminal rewards carry game-length modifiers (speed bonus / slow
     penalty) instead of a per-step penalty, preventing Q-value distortion.
  3. Piece oscillation (A→B→A patterns) receives escalating penalties.
  4. Move diversity is tracked — repeat-shuffle patterns are penalized.
  5. A small combat-initiation bonus remains outside PBRS to always
     encourage engagement over passivity.
"""

import torch
from collections import deque
from typing import Tuple, Optional, Dict, Any, Set, List
from dataclasses import dataclass
from piece import PieceType
from game_state import GameState
from training_config import GAMMA

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


# ═══════════════════════════════════════════════════════════════════════
# POTENTIAL FUNCTION — Φ(s): game-progress potential for PBRS
# ═══════════════════════════════════════════════════════════════════════

def compute_potential(board: torch.Tensor, player_id: int,
                      revealed_types: Set[PieceType]) -> float:
    """
    Compute a scalar game-progress potential Φ(s).

    Higher Φ ⟹ closer to winning.  Components:
      1. Material advantage (rank-weighted own − enemy pieces)
      2. Territorial penetration (deepest offensive piece toward enemy flag)
      3. Information gain (fraction of enemy piece types revealed)
      4. Offensive proximity (closest offensive piece to enemy back rank)

    All components are normalized to [0, 1] and combined with fixed weights.
    The potential is bounded ∈ [−1, +1] for C51 compatibility.
    """
    # ── 1. Material advantage ────────────────────────────────────────
    my_material = 0.0
    enemy_material = 0.0

    board_flat = board.flatten()
    for val_t in board_flat:
        val = val_t.item()
        if val == 0 or val == 13:  # Empty or lake
            continue
        abs_val = abs(int(val))
        if abs_val < 1 or abs_val > 12:
            continue
        rank = PIECE_RANKS.get(PieceType(abs_val), 0)
        is_mine = (player_id == 1 and val > 0) or (player_id == -1 and val < 0)
        if is_mine:
            my_material += rank
        else:
            enemy_material += rank

    # Normalize: max total material = 10+9+8+8+7+7+7+6+6+6+6+5+5+5+5+4+4+4+3+3+3+3+3+3+2*8+1 ≈ 148
    max_material = 148.0
    material_score = (my_material - enemy_material) / max_material  # ∈ [-1, 1]

    # ── 2. Territorial penetration ───────────────────────────────────
    # How deep has any offensive piece (rank ≤ Captain) pushed?
    best_penetration = 0.0
    for r in range(10):
        for c in range(10):
            val = board[r, c].item()
            if val == 0 or val == 13:
                continue
            abs_val = abs(int(val))
            if abs_val < 1 or abs_val > 12:
                continue
            is_mine = (player_id == 1 and val > 0) or (player_id == -1 and val < 0)
            if not is_mine:
                continue
            rank = PIECE_RANKS.get(PieceType(abs_val), 0)
            if rank > OFFENSIVE_RANK_THRESHOLD:
                continue  # Skip high-value pieces
            # Penetration depth: Player 1 starts at rows 6-9 and pushes toward row 0
            #                     Player -1 starts at rows 0-3 and pushes toward row 9
            if player_id == 1:
                depth = (9 - r) / 9.0  # row 0 = 1.0, row 9 = 0.0
            else:
                depth = r / 9.0         # row 9 = 1.0, row 0 = 0.0
            best_penetration = max(best_penetration, depth)

    # ── 3. Information gain ──────────────────────────────────────────
    # What fraction of the 12 enemy piece types have been revealed?
    revealed_score = len(revealed_types) / 12.0

    # ── 4. Offensive proximity to enemy back rank ────────────────────
    # Minimum distance of any offensive piece to enemy flag zone
    min_dist = 10.0
    for r in range(10):
        for c in range(10):
            val = board[r, c].item()
            if val == 0 or val == 13:
                continue
            abs_val = abs(int(val))
            if abs_val < 1 or abs_val > 12:
                continue
            is_mine = (player_id == 1 and val > 0) or (player_id == -1 and val < 0)
            if not is_mine:
                continue
            rank = PIECE_RANKS.get(PieceType(abs_val), 0)
            if rank > OFFENSIVE_RANK_THRESHOLD:
                continue
            if player_id == 1:
                dist = r  # Distance to row 0 (enemy back rank)
            else:
                dist = 9 - r  # Distance to row 9 (enemy back rank)
            min_dist = min(min_dist, dist)

    proximity_score = 1.0 - (min_dist / 10.0)  # ∈ [0, 1]

    # ── Combine ──────────────────────────────────────────────────────
    potential = (0.4 * material_score +
                 0.15 * best_penetration +
                 0.1 * revealed_score +
                 0.35 * proximity_score)

    return potential


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

    # ── Terminal Rewards ────────────────────────────────────────────────
    # Scaled to fit C51 support [-10, +10] with room for accumulated shaping.
    win_reward_flag: float = 1.5        # Flag capture = primary objective (raised for stronger signal)
    win_reward_depletion: float = 0.75  # Opponent immobilized = secondary (raised proportionally)
    win_reward: float = 1.0             # Fallback if win_type not specified (raised)
    loss_penalty: float = -1.0          # Symmetric loss penalty
    draw_penalty: float = -0.5          # Discourage passive draws (raised from -0.3 for stronger contrast)

    # Material-advantage draws: adjust draw reward based on piece count
    draw_material_bonus: float = 0.15   # Max bonus for having more pieces at draw

    # ── Terminal Game-Length Modifier (replaces per-step penalty) ───────
    speed_bonus_max: float = 0.3        # Max bonus for winning quickly
    slow_draw_penalty_max: float = -0.3 # Max additional penalty for long draws

    # ── PBRS Shaping Weight ─────────────────────────────────────────────
    pbrs_weight: float = 1.0            # Overall scale for potential-based shaping

    # ── Combat Bonus (outside PBRS — always encourages engagement) ──────
    attack_bonus: float = 0.02          # Small bonus for initiating combat
    spy_marsh_bonus: float = 0.3        # Spy killing Marshal (outside PBRS — rare strategic event)
    miner_bomb_bonus: float = 0.15      # Miner defusing Bomb (outside PBRS — rare strategic event)

    # ── Oscillation Penalty (anti-shuffle) ──────────────────────────────
    oscillation_penalty: float = -0.02  # Per oscillation beyond threshold
    oscillation_threshold: int = 2      # Number of A→B→A before penalty kicks in

    # ── Move Diversity Penalty ──────────────────────────────────────────
    diversity_window: int = 20          # Lookback window for move diversity
    diversity_min_unique: int = 3       # Min unique moves required in window
    low_diversity_penalty: float = -0.05

    # ── Stalemate (near-immobility) ─────────────────────────────────────
    stalemate_penalty: float = -0.1     # Severe penalty for near-immobility
    stalemate_mobility_threshold: int = 5  # Trigger when < this many valid moves

    # ── Legacy Fields (backward-compat for opponents.py GreedyAgent) ────
    # These are NOT used by the PBRS reward shaper but GreedyAgent._score_move()
    # references them for heuristic opponent behavior.
    capture_scale: float = 0.1          # Reward = scale × (defender_rank / 10)
    loss_scale: float = -0.1            # Penalty = scale × (our_piece_rank / 10)
    territory_advance: float = 0.05     # One-time per new row reached
    center_control: float = 0.005       # In center zone


class UnifiedRewardShaper:
    """
    Standardized reward calculator for all Stratego RL components.

    v3 architecture:
      - Terminal: win/loss/draw with game-length modifier
      - Shaping: PBRS via γΦ(s') − Φ(s)
      - Anti-stagnation: oscillation + move diversity penalties
      - Combat: small constant bonus for initiating battles (non-PBRS)
    """

    def __init__(self, player_id: int, config: Optional[StrategoRewardConfig] = None, device: str = 'cuda'):
        self.player_id = player_id
        self.config = config or StrategoRewardConfig()
        self.device = device
        self.current_phase = 1  # Default to Phase 1 (Physics of War)
        self.gamma = GAMMA
        self.reset()

    def reset(self):
        """Reset per-episode tracking."""
        self.revealed_types: Set[PieceType] = set()

        # PBRS: track previous potential for Φ(s') - Φ(s)
        self.previous_potential: float = 0.0
        self.potential_initialized: bool = False

        # Oscillation detection: per-piece position history
        # Key: current position of a piece, Value: list of past positions
        self.piece_position_history: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # Move diversity tracking
        self.recent_moves: deque = deque(maxlen=self.config.diversity_window)

    def set_phase(self, phase: int):
        """Update the current curriculum phase to scale shaping rewards."""
        self.current_phase = phase

    def get_shaping_multiplier(self) -> float:
        """
        Calculate the multiplier for shaping rewards based on the current phase.
        Phase 1: 1.0 (Full shaping to learn basic physics of war)
        Phase 2: 0.5 (Steep early drop to force flag capture focus)
        Phase 3: 0.2 (Fading out)
        Phase 4-5: 0.0 (Pure terminal reward only)
        """
        if self.current_phase == 1:
            return 1.0
        elif self.current_phase == 2:
            return 0.5
        elif self.current_phase == 3:
            return 0.2
        else:
            # Phase 4+: Residual shaping to guide toward flag without corrupting
            # the pure MARL objective. Raised from 0.05 to 0.2 to prevent Q-value collapse
            # over 1500-step games where terminal signal alone is too sparse and discounting zeroes it out.
            return 0.2
            
    def get_terminal_multiplier(self) -> float:
        """
        Calculates the multiplier for terminal rewards based on current phase.
        This provides Reward Scale Dampening for MARL in Phase 4+.
        """
        try:
            from training_config import MARL_REWARD_SCALE
        except ImportError:
            MARL_REWARD_SCALE = 0.5
            
        if self.current_phase >= 4:
            return MARL_REWARD_SCALE
        return 1.0

    def _count_oscillations(self, dest: Tuple[int, int], source: Tuple[int, int]) -> int:
        """
        Count how many times a piece has oscillated (A→B→A pattern).
        Returns the oscillation count for the piece arriving at dest from source.
        """
        # Get the history for the piece that was at 'source'
        history = self.piece_position_history.get(source, [])
        if len(history) < 2:
            return 0

        # Count A→B→A→B→A patterns: every time the piece returns to a
        # previously visited position
        oscillations = 0
        for past_pos in history:
            if past_pos == dest:
                oscillations += 1
        return oscillations

    def _update_piece_tracking(self, source: Tuple[int, int], dest: Tuple[int, int]):
        """Update per-piece position history after a move."""
        # Transfer history from source to dest (piece moved)
        history = self.piece_position_history.pop(source, [])
        history.append(source)  # Record where it came from
        # Keep only last 10 positions to bound memory
        if len(history) > 10:
            history = history[-10:]
        self.piece_position_history[dest] = history

    def __call__(self, previous_state: GameState, action: Optional[Tuple],
                 current_state: GameState, done: bool,
                 winner: Optional[int], info: Dict[str, Any]) -> float:
        """Calculate the total normalized reward for this step."""

        # ── 1. Terminal Outcomes (with game-length modifier) ─────────
        if done:
            turn_count = info.get('turn_count', current_state.turn_count)
            max_turns = info.get('max_turns', 200)
            game_fraction = min(turn_count / max(max_turns, 1), 1.0)
            
            terminal_mult = self.get_terminal_multiplier()

            if winner == self.player_id:
                win_type = info.get('win_type', 'unknown')
                if win_type == 'flag_capture':
                    base = self.config.win_reward_flag
                elif win_type == 'no_moves':
                    base = self.config.win_reward_depletion
                else:
                    base = self.config.win_reward

                # Quick wins are worth more — speed bonus decays linearly
                speed_bonus = self.config.speed_bonus_max * (1.0 - game_fraction)
                return self.config.outcome_weight * (base + speed_bonus) * terminal_mult

            elif winner == -self.player_id:
                return self.config.outcome_weight * self.config.loss_penalty * terminal_mult

            elif winner == 0:
                # Material-advantage draws
                board = current_state.board
                my_pieces = ((board > 0) & (board < 13)).sum().item() if self.player_id == 1 else ((board < 0) & (board > -13)).sum().item()
                enemy_pieces = ((board < 0) & (board > -13)).sum().item() if self.player_id == 1 else ((board > 0) & (board < 13)).sum().item()

                piece_diff = my_pieces - enemy_pieces
                material_bonus = min(max(piece_diff / 10.0, -1.0), 1.0) * self.config.draw_material_bonus

                # Long draws are punished more severely
                length_penalty = self.config.slow_draw_penalty_max * game_fraction

                return self.config.outcome_weight * (self.config.draw_penalty + material_bonus + length_penalty) * terminal_mult
            return 0.0

        if action is None:
            return 0.0

        (r_from, c_from), (r_to, c_to) = action
        source = (r_from, c_from)
        dest = (r_to, c_to)

        reward_components: Dict[str, float] = {}

        # ── 2. PBRS: γΦ(s') − Φ(s) ─────────────────────────────────
        shaping_mult = self.get_shaping_multiplier()
        if shaping_mult > 0:
            # Track revealed types from combat info
            if info.get('revealed_in_step'):
                for (pos, piece_type) in info['revealed_in_step']:
                    if isinstance(piece_type, PieceType):
                        self.revealed_types.add(piece_type)

            current_potential = compute_potential(
                current_state.board, self.player_id, self.revealed_types
            )

            if not self.potential_initialized:
                # First step: initialize previous potential from starting state
                self.previous_potential = compute_potential(
                    previous_state.board, self.player_id, self.revealed_types
                )
                self.potential_initialized = True

            pbrs_reward = self.gamma * current_potential - self.previous_potential
            self.previous_potential = current_potential

            reward_components['pbrs'] = pbrs_reward * self.config.pbrs_weight * shaping_mult

        # ── 3. Combat Bonus (outside PBRS — always encourage fighting) ──
        prev_board = previous_state.board
        target_val = prev_board[r_to, c_to].item()
        moving_val = prev_board[r_from, c_from].item()

        was_battle = (target_val != 0 and target_val != 13) and \
                     ((self.player_id == 1 and target_val < 0) or
                      (self.player_id == -1 and target_val > 0))

        if was_battle:
            # Small constant combat-initiation bonus (not annealed? actually we should anneal it)
            # Scaling it by shaping_mult so it zeroes out in Phase 4
            reward_components['attack'] = self.config.attack_bonus * shaping_mult

            # Strategic bonuses for rare high-impact captures (not annealed)
            if abs(int(moving_val)) == 1 and abs(int(target_val)) == 10:  # Spy vs Marshal
                reward_components['spy_marshal'] = self.config.spy_marsh_bonus * shaping_mult
            if abs(int(moving_val)) == 3 and abs(int(target_val)) == 11:  # Miner vs Bomb
                reward_components['miner_bomb'] = self.config.miner_bomb_bonus * shaping_mult

        # ── 4. Oscillation Penalty (A→B→A detection) ─────────────────
        oscillation_count = self._count_oscillations(dest, source)
        if oscillation_count >= self.config.oscillation_threshold:
            reward_components['oscillation'] = (
                self.config.oscillation_penalty * oscillation_count * shaping_mult
            )

        # Update piece tracking AFTER computing oscillation penalty
        self._update_piece_tracking(source, dest)

        # ── 5. Move Diversity Penalty ────────────────────────────────
        move_key = (source, dest)
        self.recent_moves.append(move_key)
        if len(self.recent_moves) >= self.config.diversity_window:
            unique_moves = len(set(self.recent_moves))
            if unique_moves <= self.config.diversity_min_unique:
                reward_components['low_diversity'] = self.config.low_diversity_penalty * shaping_mult

        # ── 6. Stalemate (near-immobility) ───────────────────────────
        curr_mob = info.get('num_valid_moves', 100)  # Default high when not provided
        if curr_mob < self.config.stalemate_mobility_threshold:
            reward_components['stalemate'] = self.config.stalemate_penalty * shaping_mult

        total_reward = sum(reward_components.values())
        return total_reward


def create_unified_reward_shaper(player_id: int = 1, config: Optional[StrategoRewardConfig] = None, device: str = 'cuda'):
    """Factory function for creating the shaper."""
    return UnifiedRewardShaper(player_id=player_id, config=config, device=device)

# Legacy aliases for compatibility
DistributionalRewardConfig = StrategoRewardConfig
create_distributional_reward_wrapper = create_unified_reward_shaper
