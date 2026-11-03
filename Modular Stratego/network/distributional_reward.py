"""
Centralized Reward System for Stratego DQN (C51-Compatible)

This module is the SINGLE SOURCE OF TRUTH for the shaping reward. The terminal reward (+1 win,
-1 loss, DRAW_REWARD for a draw) is network/training/transitions.outcome_reward; the trainer adds it to the
shaping of terminal transitions.

DESIGN PRINCIPLES (v4, cleanup branch):
  1. Potential-Based Reward Shaping (PBRS): shaping = γ·Φ(s') − Φ(s), with Φ(terminal) = 0, so
     the shaping of a whole game telescopes to −Φ(s_0) and the optimal policy is unchanged
     (Ng et al., 1999). The thesis code gave the terminal transition no −Φ(s) term.
  2. Φ is a function of the side's own view: an enemy piece the view shows as hidden counts with
     the mean rank of a full army, not as zero material (the thesis code ignored hidden pieces,
     so revealing an enemy piece in a lost battle lowered the potential and capturing a hidden
     piece gained no material).
  3. The non-potential terms (combat bonus, Spy-Marshal / Miner-Bomb bonuses, oscillation,
     move diversity, stalemate) can change the optimal policy and are off by default (weight 0).
"""

import torch
from collections import deque
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from piece import PieceType
from game_state import GameState
from board import HIDDEN_PIECE
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

# Standard army (40 pieces per side)
ARMY = {
    PieceType.FLAG: 1, PieceType.SPY: 1, PieceType.SCOUT: 8, PieceType.MINER: 5,
    PieceType.SERGEANT: 4, PieceType.LIEUTENANT: 4, PieceType.CAPTAIN: 4, PieceType.MAJOR: 3,
    PieceType.COLONEL: 2, PieceType.GENERAL: 1, PieceType.MARSHAL: 1, PieceType.BOMB: 6,
}
ARMY_MATERIAL = float(sum(PIECE_RANKS[t] * n for t, n in ARMY.items()))  # 148
# Material of an enemy piece whose rank the view does not show: the mean over a full army
UNKNOWN_PIECE_RANK = ARMY_MATERIAL / sum(ARMY.values())                 # 3.7

# Maximum piece rank for gating forward-movement reward.
# Only pieces with rank <= this get the flag-distance incentive.
# Marshal(10), General(9), Colonel(8) should NOT rush forward.
OFFENSIVE_RANK_THRESHOLD = 6  # Captain and below
_RANK_BY_VALUE = {pt.value: PIECE_RANKS.get(pt, 0) for pt in PieceType}


# ═══════════════════════════════════════════════════════════════════════
# POTENTIAL FUNCTION — Φ(s): game-progress potential for PBRS
# ═══════════════════════════════════════════════════════════════════════

def compute_potential(board: torch.Tensor, player_id: int) -> float:
    """
    Game-progress potential Φ(s) of `board` as `player_id` sees it (its own view: enemy pieces it
    cannot see are HIDDEN_PIECE; a fully observable board shows every rank).

    Higher Φ ⟹ closer to winning.  Components (weights 0.4 / 0.15 / 0.1 / 0.35):
      1. Material advantage: own − enemy rank-weighted material over a full army's material;
         a hidden enemy piece counts as UNKNOWN_PIECE_RANK
      2. Territorial penetration: deepest offensive piece (rank <= Captain) toward the enemy side
      3. Information: share of the enemy pieces on the board whose rank the view shows
      4. Offensive proximity: closest offensive piece to the enemy back rank

    Φ ∈ [−0.4, 1].
    """
    # One pass over the board as Python ints (per-cell tensor .item() calls were the cost here)
    my_material = 0.0
    enemy_material = 0.0
    enemy_known = 0
    enemy_hidden = 0
    best_penetration = 0.0  # 2. deepest offensive piece (rank <= Captain) toward the enemy side
    min_dist = 10.0         # 4. closest offensive piece to the enemy back rank
    for i, val in enumerate(board.flatten().tolist()):
        val = int(val)
        if val == HIDDEN_PIECE:
            # Only enemy pieces are ever hidden from the viewer (the marker has the same value
            # on both sides' views)
            enemy_hidden += 1
            enemy_material += UNKNOWN_PIECE_RANK
            continue
        abs_val = abs(val)
        if abs_val < 1 or abs_val > 12:
            continue  # empty or lake
        rank = _RANK_BY_VALUE[abs_val]
        is_mine = (player_id == 1 and val > 0) or (player_id == -1 and val < 0)
        if not is_mine:
            enemy_known += 1
            enemy_material += rank
            continue
        my_material += rank
        if rank > OFFENSIVE_RANK_THRESHOLD:
            continue
        r = i // 10
        # Player 1 starts at rows 6-9 and pushes toward row 0; Player -1 the other way
        if player_id == 1:
            depth, dist = (9 - r) / 9.0, r
        else:
            depth, dist = r / 9.0, 9 - r
        best_penetration = max(best_penetration, depth)
        min_dist = min(min_dist, dist)

    material_score = (my_material - enemy_material) / ARMY_MATERIAL  # ∈ [-1, 1]
    enemy_on_board = enemy_known + enemy_hidden
    info_score = enemy_known / enemy_on_board if enemy_on_board else 1.0  # ∈ [0, 1]
    proximity_score = 1.0 - (min_dist / 10.0)  # ∈ [0, 1]

    # ── Combine ──────────────────────────────────────────────────────
    potential = (0.4 * material_score +
                 0.15 * best_penetration +
                 0.1 * info_score +
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

    # ── PBRS Shaping Weight ─────────────────────────────────────────────
    pbrs_weight: float = 1.0            # Overall scale for potential-based shaping

    # ── Non-potential terms (0 = off; any non-zero weight can change the optimal policy) ──
    # Thesis values: attack 0.02, Spy-Marshal 0.3, Miner-Bomb 0.15, oscillation -0.02,
    # low diversity -0.05, stalemate -0.1
    attack_bonus: float = 0.0           # Bonus for initiating combat
    spy_marsh_bonus: float = 0.0        # Spy attacking the Marshal
    miner_bomb_bonus: float = 0.0       # Miner attacking a Bomb

    # ── Oscillation Penalty (anti-shuffle) ──────────────────────────────
    oscillation_penalty: float = 0.0    # Per oscillation beyond threshold
    oscillation_threshold: int = 2      # Number of A→B→A before penalty kicks in

    # ── Move Diversity Penalty ──────────────────────────────────────────
    diversity_window: int = 20          # Lookback window for move diversity
    diversity_min_unique: int = 3       # Min unique moves required in window
    low_diversity_penalty: float = 0.0

    # ── Stalemate (near-immobility) ─────────────────────────────────────
    stalemate_penalty: float = 0.0      # Penalty for near-immobility
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
    Shaping reward of one side's transitions (each from its own view to its own view).

      - PBRS: γΦ(s') − Φ(s), Φ(terminal) = 0; scaled by the curriculum phase
      - Optional non-potential terms (off by default): combat bonuses, oscillation, move
        diversity and stalemate penalties
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
            return 0.0

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

    def potential(self, state: GameState) -> float:
        """Φ of this side's view `state`."""
        return compute_potential(state.board, self.player_id)

    def __call__(self, previous_state: GameState, action: Optional[Tuple],
                 current_state: GameState, done: bool,
                 winner: Optional[int] = None, info: Optional[Dict[str, Any]] = None) -> float:
        """
        Shaping reward of the transition previous_state -> current_state (both this side's own
        view). A terminal transition gets −Φ(previous_state) (Φ of a finished game is 0); its
        outcome reward is added by the trainer. `winner` is not used.
        """
        info = info or {}
        if action is None:
            return 0.0

        reward_components: Dict[str, float] = {}

        # ── 1. PBRS: γΦ(s') − Φ(s), Φ(terminal) = 0 ─────────────────
        shaping_mult = self.get_shaping_multiplier()
        if shaping_mult > 0 and self.config.pbrs_weight:
            next_potential = 0.0 if done else self.potential(current_state)
            pbrs_reward = self.gamma * next_potential - self.potential(previous_state)
            reward_components['pbrs'] = pbrs_reward * self.config.pbrs_weight * shaping_mult

        (r_from, c_from), (r_to, c_to) = action
        source = (r_from, c_from)
        dest = (r_to, c_to)

        # ── 2. Combat bonuses (non-potential, off by default) ────────
        cfg = self.config
        if cfg.attack_bonus or cfg.spy_marsh_bonus or cfg.miner_bomb_bonus:
            prev_board = previous_state.board
            target_val = int(prev_board[r_to, c_to].item())
            moving_val = int(prev_board[r_from, c_from].item())
            was_battle = target_val == HIDDEN_PIECE or (
                target_val != 0 and abs(target_val) <= 12 and (target_val > 0) != (self.player_id == 1))
            if was_battle:
                reward_components['attack'] = cfg.attack_bonus
                # (the thesis code tested the values 1/10 and 3/11 here, i.e. Flag vs General and
                # Scout vs Marshal, so the Spy and Miner bonuses were never paid for their events)
                if abs(moving_val) == PieceType.SPY.value and abs(target_val) == PieceType.MARSHAL.value:
                    reward_components['spy_marshal'] = cfg.spy_marsh_bonus * shaping_mult
                if abs(moving_val) == PieceType.MINER.value and abs(target_val) == PieceType.BOMB.value:
                    reward_components['miner_bomb'] = cfg.miner_bomb_bonus * shaping_mult

        # ── 3. Oscillation Penalty (A→B→A detection) ─────────────────
        if cfg.oscillation_penalty:
            oscillation_count = self._count_oscillations(dest, source)
            if oscillation_count >= cfg.oscillation_threshold:
                reward_components['oscillation'] = cfg.oscillation_penalty * oscillation_count
            # Update piece tracking AFTER computing oscillation penalty
            self._update_piece_tracking(source, dest)

        # ── 4. Move Diversity Penalty ────────────────────────────────
        if cfg.low_diversity_penalty:
            self.recent_moves.append((source, dest))
            if len(self.recent_moves) >= cfg.diversity_window:
                if len(set(self.recent_moves)) <= cfg.diversity_min_unique:
                    reward_components['low_diversity'] = cfg.low_diversity_penalty

        # ── 5. Stalemate (near-immobility) ───────────────────────────
        if cfg.stalemate_penalty:
            curr_mob = info.get('num_valid_moves', 100)  # Default high when not provided
            if curr_mob < cfg.stalemate_mobility_threshold:
                reward_components['stalemate'] = cfg.stalemate_penalty

        return sum(reward_components.values())


def create_unified_reward_shaper(player_id: int = 1, config: Optional[StrategoRewardConfig] = None, device: str = 'cuda'):
    """Factory function for creating the shaper."""
    return UnifiedRewardShaper(player_id=player_id, config=config, device=device)

# Legacy aliases for compatibility
DistributionalRewardConfig = StrategoRewardConfig
create_distributional_reward_wrapper = create_unified_reward_shaper
