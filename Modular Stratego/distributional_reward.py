"""
Distributional RL-Compatible Reward Shaping for Rainbow DQN

This module implements reward shaping specifically designed for Rainbow DQN
with Categorical DQN (C51) distributional value estimation.

Key Design Principles for Distributional RL:
1. All rewards are SMALL and normalized to prevent V-max/V-min clipping
2. Terminal rewards are well WITHIN the support (±1.0, not ±10.0)
3. Intermediate rewards are tiny (0.01-0.1) so cumulative returns stay bounded
4. The value distribution can capture variance/risk rather than being truncated

Recommended V_min/V_max settings:
- With this reward function: V_min=-3.0, V_max=+3.0
- This provides headroom: max episode return ≈ +1.0 (win) + 0.5 (captures) - 0.5 (steps) ≈ +1.0
- And min episode return ≈ -1.0 (loss) - 0.3 (losses) - 0.5 (steps) ≈ -1.8

Reference: Bellemare et al. (2017) "A Distributional Perspective on RL"
"""

import torch
from typing import Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass

from piece import PieceType
from game_state import GameState


# Piece rank values for reward calculations (1-10 scale)
PIECE_RANKS = {
    PieceType.FLAG: 0,      # Capture = game over
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
    PieceType.BOMB: 0,      # Handled specially
}


@dataclass
class DistributionalRewardConfig:
    """
    Configuration for Distributional RL-compatible reward shaping.
    
    All values are normalized to keep cumulative returns within [-3, +3]
    for optimal C51 distribution learning with V_min=-3, V_max=+3.
    """
    # =========================================================================
    # TERMINAL REWARDS (Loudest signal, but WITHIN distribution support)
    # =========================================================================
    win_reward: float = 1.0           # Win (flag capture or elimination)
    loss_penalty: float = -1.0        # Loss (opponent wins)
    draw_penalty: float = -0.8        # Draw/Timeout - WORSE than fighting loss!
    
    # =========================================================================
    # ANTI-STALL GRADIENT (Time pressure)
    # =========================================================================
    step_penalty: float = -0.005      # Tiny per-step penalty
    # With ~100 steps average, this adds ≈ -0.5 per episode
    
    # =========================================================================
    # COMBAT REWARDS (Material signal)
    # =========================================================================
    capture_scale: float = 0.1        # Capture: +0.1 * (rank/10) = max +0.1
    loss_scale: float = -0.05         # Losing piece: flat -0.05
    
    # =========================================================================
    # INFORMATION GAIN (Crucial for Distributional RL!)
    # =========================================================================
    # This helps the VALUE DISTRIBUTION learn the variance/risk of attacks
    # Unknown pieces have high variance; revealed pieces have low variance
    reveal_bonus: float = 0.02        # Small bonus for revealing enemy rank
    first_reveal_bonus: float = 0.03  # Extra bonus for first reveal of a rank
    
    # =========================================================================
    # STRATEGIC BONUSES (Domain knowledge)
    # =========================================================================
    spy_kills_marshal: float = 0.15   # Spy kills Marshal (rare, valuable)
    miner_defuses_bomb: float = 0.08  # Miner removes Bomb (strategic)
    
    # =========================================================================
    # TERRITORY ADVANCEMENT (Forward progress bonus)
    # =========================================================================
    territory_advance: float = 0.02   # Bonus for moving toward enemy flag
    center_control: float = 0.01      # Bonus for occupying center positions
    
    @classmethod
    def from_training_config(cls) -> 'DistributionalRewardConfig':
        """Load configuration from training_config.py settings."""
        from training_config import (
            DIST_STEP_PENALTY, DIST_DRAW_PENALTY,
            DIST_WIN_REWARD, DIST_LOSS_PENALTY,
            DIST_CAPTURE_SCALE, DIST_LOSS_SCALE,
            DIST_REVEAL_BONUS, DIST_FIRST_REVEAL_BONUS,
            DIST_SPY_KILLS_MARSHAL, DIST_MINER_DEFUSES_BOMB,
            DIST_TERRITORY_ADVANCE, DIST_CENTER_CONTROL
        )
        return cls(
            step_penalty=DIST_STEP_PENALTY,
            draw_penalty=DIST_DRAW_PENALTY,
            win_reward=DIST_WIN_REWARD,
            loss_penalty=DIST_LOSS_PENALTY,
            capture_scale=DIST_CAPTURE_SCALE,
            loss_scale=DIST_LOSS_SCALE,
            reveal_bonus=DIST_REVEAL_BONUS,
            first_reveal_bonus=DIST_FIRST_REVEAL_BONUS,
            spy_kills_marshal=DIST_SPY_KILLS_MARSHAL,
            miner_defuses_bomb=DIST_MINER_DEFUSES_BOMB,
            territory_advance=DIST_TERRITORY_ADVANCE,
            center_control=DIST_CENTER_CONTROL
        )
    

class DistributionalRewardTracker:
    """
    Tracks per-episode state for information gain rewards.
    Reset at the start of each episode.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset tracking for new episode."""
        # Track revealed enemy piece types (for first-reveal bonus)
        self.revealed_enemy_types: Set[PieceType] = set()
        
        # Track revealed positions (for general reveal bonus)
        self.revealed_positions: Set[Tuple[int, int]] = set()
        
    def record_reveal(self, position: Tuple[int, int], piece_type: PieceType) -> Tuple[bool, bool]:
        """
        Record a piece reveal and return bonus flags.
        
        Returns:
            (is_new_position, is_first_of_type): Tuple of booleans
        """
        is_new_position = position not in self.revealed_positions
        is_first_of_type = piece_type not in self.revealed_enemy_types
        
        self.revealed_positions.add(position)
        self.revealed_enemy_types.add(piece_type)
        
        return is_new_position, is_first_of_type


def calculate_distributional_reward(
    previous_state: GameState,
    action: Tuple[Tuple[int, int], Tuple[int, int]],
    current_state: GameState,
    done: bool,
    winner: Optional[int],
    info: Dict[str, Any],
    player_id: int = 1,
    config: Optional[DistributionalRewardConfig] = None,
    tracker: Optional[DistributionalRewardTracker] = None
) -> float:
    """
    Calculate reward for Stratego with Distributional RL (C51) compatibility.
    
    This function produces rewards that:
    1. Stay well within the V_min/V_max support (recommended: -3 to +3)
    2. Provide information gain signals for variance learning
    3. Create an anti-stall gradient through tiny step penalties
    4. Make draws WORSE than losses to encourage aggression
    
    Args:
        previous_state: GameState before the action
        action: Tuple of ((from_row, from_col), (to_row, to_col))
        current_state: GameState after the action
        done: Whether the episode has ended
        winner: Winner (1, -1, or 0 for draw)
        info: Step info dictionary
        player_id: Player ID (1 or -1)
        config: Reward configuration
        tracker: Per-episode tracker for information gain
        
    Returns:
        float: Normalized reward value (small magnitude)
        
    V_min/V_max Guidance:
        - Recommended: V_min=-3.0, V_max=+3.0
        - Max possible episode return: ~+1.5 (win + captures + reveals)
        - Min possible episode return: ~-2.5 (loss + losses + steps)
        - This provides buffer room for the distribution to learn
    """
    if config is None:
        config = DistributionalRewardConfig()
        
    if tracker is None:
        tracker = DistributionalRewardTracker()
    
    reward = 0.0
    
    # =========================================================================
    # 1. ANTI-STALL GRADIENT (Always applied)
    # =========================================================================
    reward += config.step_penalty
    
    # =========================================================================
    # 2. TERMINAL REWARDS
    # =========================================================================
    if done:
        if winner == player_id:
            reward += config.win_reward
        elif winner == -player_id:
            reward += config.loss_penalty
        elif winner == 0:
            # CRITICAL: Draw is worse than a fighting loss
            # This creates a gradient TOWARD aggression
            # Agent learns: "If I'm going to lose anyway, at least fight"
            reward += config.draw_penalty
        return reward
    
    # =========================================================================
    # 3. COMBAT REWARDS (Extract battle info)
    # =========================================================================
    if action is None:
        return reward
        
    (r_from, c_from), (r_to, c_to) = action
    
    # Detect if this was a battle
    prev_board = previous_state.board
    curr_board = current_state.board
    
    # Get piece values
    moving_val = prev_board[r_from, c_from].item() if hasattr(prev_board[r_from, c_from], 'item') else prev_board[r_from, c_from]
    target_val = prev_board[r_to, c_to].item() if hasattr(prev_board[r_to, c_to], 'item') else prev_board[r_to, c_to]
    
    # Check if battle occurred (target was enemy piece)
    was_battle = False
    if target_val != 0 and target_val != 13:  # 13 = LAKE_SQUARE
        if (player_id == 1 and target_val < 0) or (player_id == -1 and target_val > 0):
            was_battle = True
    
    if was_battle:
        # Get piece types
        try:
            attacker_type = PieceType(abs(int(moving_val)))
            defender_type = PieceType(abs(int(target_val)))
            attacker_rank = PIECE_RANKS.get(attacker_type, 5)
            defender_rank = PIECE_RANKS.get(defender_type, 5)
        except (ValueError, KeyError):
            attacker_type = None
            defender_type = None
            attacker_rank = 5
            defender_rank = 5
        
        # Determine battle outcome from current board state
        result_val = curr_board[r_to, c_to].item() if hasattr(curr_board[r_to, c_to], 'item') else curr_board[r_to, c_to]
        source_val = curr_board[r_from, c_from].item() if hasattr(curr_board[r_from, c_from], 'item') else curr_board[r_from, c_from]
        
        # Determine who won
        if player_id == 1:
            we_won = result_val > 0
            we_lost = result_val <= 0 and source_val == 0 and result_val != 0
            mutual = source_val == 0 and result_val == 0
        else:
            we_won = result_val < 0
            we_lost = result_val >= 0 and source_val == 0 and result_val != 0
            mutual = source_val == 0 and result_val == 0
        
        # -----------------------------------------------------------------
        # 3a. COMBAT OUTCOME REWARDS
        # -----------------------------------------------------------------
        if we_won:
            # Capture reward scaled by enemy rank
            capture_reward = config.capture_scale * (defender_rank / 10.0)
            reward += capture_reward
            
            # Strategic bonus: Spy kills Marshal
            if attacker_type == PieceType.SPY and defender_type == PieceType.MARSHAL:
                reward += config.spy_kills_marshal
                
            # Strategic bonus: Miner defuses Bomb
            if attacker_type == PieceType.MINER and defender_type == PieceType.BOMB:
                reward += config.miner_defuses_bomb
                
        elif we_lost:
            # Flat penalty for losing a piece
            reward += config.loss_scale
            
        else:  # Mutual destruction
            # Net zero for mutual (already paid step cost)
            # Could add small penalty if desired
            pass
        
        # -----------------------------------------------------------------
        # 3b. INFORMATION GAIN (Crucial for Distributional RL!)
        # -----------------------------------------------------------------
        # This helps the value distribution learn VARIANCE
        # Attacking unknown pieces has high uncertainty
        # Revealing their rank reduces variance in future encounters
        
        if defender_type is not None:
            is_new_pos, is_first_type = tracker.record_reveal((r_to, c_to), defender_type)
            
            if is_new_pos:
                reward += config.reveal_bonus
                
            if is_first_type:
                # Extra bonus for first reveal of this piece type
                # Helps with opponent modeling (now we know they have/had this piece)
                reward += config.first_reveal_bonus
    
    # =========================================================================
    # 4. TERRITORY ADVANCEMENT (Forward progress signal)
    # =========================================================================
    # This is CRUCIAL for Phase 1 with full observability
    # Encourages the agent to push toward enemy flag
    if action is not None:
        (r_from, c_from), (r_to, c_to) = action
        
        if player_id == 1:
            # Player 1 advances by moving to lower rows (toward enemy at top)
            if r_to < r_from:
                reward += config.territory_advance
        else:
            # Player -1 advances by moving to higher rows (toward enemy at bottom)
            if r_to > r_from:
                reward += config.territory_advance
        
        # Center control bonus (columns 3-6 and rows 4-5 are strategic center)
        if 3 <= c_to <= 6 and 4 <= r_to <= 5:
            reward += config.center_control
    
    return reward


def create_distributional_reward_wrapper(player_id: int = 1, config: Optional[DistributionalRewardConfig] = None):
    """
    Create a reward wrapper with per-episode tracking.
    
    Usage:
        reward_fn = create_distributional_reward_wrapper(player_id=1)
        
        # At episode start:
        reward_fn.reset()
        
        # During step:
        reward = reward_fn(prev_state, action, curr_state, done, winner, info)
    """
    if config is None:
        # Load from training_config.py for latest reward values
        config = DistributionalRewardConfig.from_training_config()
    
    tracker = DistributionalRewardTracker()
    
    class RewardWrapper:
        def __init__(self):
            self.tracker = tracker
            self.config = config
            self.player_id = player_id
            
        def __call__(self, previous_state, action, current_state, done, winner, info):
            return calculate_distributional_reward(
                previous_state=previous_state,
                action=action,
                current_state=current_state,
                done=done,
                winner=winner,
                info=info,
                player_id=self.player_id,
                config=self.config,
                tracker=self.tracker
            )
            
        def reset(self):
            """Reset tracker for new episode."""
            self.tracker.reset()
    
    return RewardWrapper()


# =============================================================================
# V_MIN / V_MAX GUIDANCE FOR C51
# =============================================================================
"""
HYPERPARAMETER RECOMMENDATIONS:

With this reward function, the expected returns are:

WINNING GAMES (best case):
- Win terminal: +1.0
- Captures (5 pieces @ avg rank 5): +0.25
- Info reveals: +0.1
- Step penalty (~100 steps): -0.5
- TOTAL: ≈ +0.85

LOSING GAMES (worst case):
- Loss terminal: -1.0
- Piece losses: -0.25
- Step penalty (~100 steps): -0.5
- TOTAL: ≈ -1.75

DRAW GAMES (stall penalty):
- Draw terminal: -0.8
- Step penalty (~200 steps for stall): -1.0
- TOTAL: ≈ -1.8

RECOMMENDED C51 SETTINGS:
    V_MIN = -3.0   # Provides buffer below worst case
    V_MAX = +3.0   # Provides buffer above best case
    N_ATOMS = 51   # Standard C51 setting

This gives the distribution room to:
1. Learn the variance of different game outcomes
2. Capture the uncertainty of attacking unknown pieces
3. Not clip at the bounds (which loses information)

To change in your code (drqn_agent.py or similar):
    self.v_min = -3.0
    self.v_max = 3.0
    self.n_atoms = 51
    self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
"""
