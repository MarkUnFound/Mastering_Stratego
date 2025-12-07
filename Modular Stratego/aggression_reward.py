"""
Aggression-Based Dense Reward Shaping for Stratego DQN

This module implements a reward shaping strategy specifically designed to combat
reward hacking (stalling for draws) by incentivizing:
1. Time pressure (penalize every step, heavily penalize draws)
2. Aggressive combat (reward attacks, even losing ones that reveal info)
3. Territorial advancement (reward crossing into enemy territory)
4. Strategic piece captures (bonus for Spy/Miner when tactically valuable)

Reference: Dense reward shaping for imperfect information games to encourage
exploration and aggressive play over risk-averse stalling.
"""

import torch
from typing import Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from piece import PieceType
from game_state import GameState


# Piece values for rank-based calculations
PIECE_RANK_VALUES = {
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
    PieceType.BOMB: 11,  # Special handling
}


@dataclass
class AggressionRewardConfig:
    """Configuration for aggression-based reward shaping."""
    # Anti-stall penalties
    step_penalty: float = -0.01          # Small penalty every step
    draw_penalty: float = -0.5           # Heavy penalty for draw/timeout
    
    # Terminal rewards (loudest signal)
    win_reward: float = 10.0             # Flag capture or elimination
    loss_penalty: float = -10.0          # Opponent wins
    invalid_move_penalty: float = -1.0   # Invalid action (if not masked)
    
    # Combat rewards
    attack_win_base: float = 1.0         # Base reward for winning attack
    attack_lose_penalty: float = -0.5    # Less bad than doing nothing
    information_bonus: float = 0.1       # Bonus for revealing enemy piece
    rank_difference_scale: float = 0.1   # Scale factor for rank bonuses
    
    # Territorial rewards
    territory_advance: float = 0.05      # First time crossing to enemy half
    retreat_penalty: float = -0.02       # Moving backward into own territory
    
    # Strategic bonuses
    spy_capture_bonus: float = 0.3       # Killing Spy when enemy has Marshal
    miner_capture_bonus: float = 0.3     # Killing Miner when enemy has Bombs
    
    # Mutual destruction handling
    mutual_info_bonus: float = 0.05      # Both pieces revealed in tie
    
    @classmethod
    def from_training_config(cls) -> 'AggressionRewardConfig':
        """Load configuration from training_config.py settings."""
        try:
            from training_config import (
                AGGRESSION_STEP_PENALTY, AGGRESSION_DRAW_PENALTY,
                AGGRESSION_WIN_REWARD, AGGRESSION_LOSS_PENALTY,
                AGGRESSION_ATTACK_WIN_BASE, AGGRESSION_ATTACK_LOSE_PENALTY,
                AGGRESSION_INFO_BONUS, AGGRESSION_RANK_SCALE,
                AGGRESSION_TERRITORY_ADVANCE, AGGRESSION_RETREAT_PENALTY,
                AGGRESSION_SPY_CAPTURE, AGGRESSION_MINER_CAPTURE
            )
            return cls(
                step_penalty=AGGRESSION_STEP_PENALTY,
                draw_penalty=AGGRESSION_DRAW_PENALTY,
                win_reward=AGGRESSION_WIN_REWARD,
                loss_penalty=AGGRESSION_LOSS_PENALTY,
                attack_win_base=AGGRESSION_ATTACK_WIN_BASE,
                attack_lose_penalty=AGGRESSION_ATTACK_LOSE_PENALTY,
                information_bonus=AGGRESSION_INFO_BONUS,
                rank_difference_scale=AGGRESSION_RANK_SCALE,
                territory_advance=AGGRESSION_TERRITORY_ADVANCE,
                retreat_penalty=AGGRESSION_RETREAT_PENALTY,
                spy_capture_bonus=AGGRESSION_SPY_CAPTURE,
                miner_capture_bonus=AGGRESSION_MINER_CAPTURE
            )
        except ImportError:
            # Fall back to defaults if training_config doesn't have these
            return cls()



class AggressionRewardTracker:
    """
    Tracks per-game state for territorial rewards and piece awareness.
    Must be reset at the start of each episode.
    """
    
    def __init__(self, player_id: int = 1):
        self.player_id = player_id
        self.reset()
        
    def reset(self):
        """Reset tracking for a new game."""
        # Track pieces that have crossed into enemy territory
        self.pieces_in_enemy_territory: Set[Tuple[int, int]] = set()
        
        # Track known enemy pieces (revealed through combat)
        self.revealed_enemy_pieces: Dict[Tuple[int, int], PieceType] = {}
        
        # Track if enemy still has key pieces (for strategic bonuses)
        self.enemy_has_marshal: bool = True
        self.enemy_has_bombs: bool = True
        
        # Track previous positions for retreat detection
        self.piece_previous_rows: Dict[Tuple[int, int], int] = {}
        
    def update_enemy_piece_status(self, captured_type: PieceType):
        """Update knowledge about enemy pieces after a capture."""
        if captured_type == PieceType.MARSHAL:
            self.enemy_has_marshal = False
        # Note: Can't fully track bombs without counting, but this is a heuristic


def calculate_reward(
    previous_state: GameState,
    action: Tuple[Tuple[int, int], Tuple[int, int]],
    current_state: GameState,
    done: bool,
    winner: Optional[int],
    info: Dict[str, Any],
    player_id: int = 1,
    config: Optional[AggressionRewardConfig] = None,
    tracker: Optional[AggressionRewardTracker] = None
) -> float:
    """
    Calculate dense reward for a Stratego action with aggression-focused shaping.
    
    This function implements a reward structure designed to combat reward hacking
    where agents stall to force draws due to fear of unknown enemy ranks.
    
    Args:
        previous_state: GameState before the action
        action: Tuple of ((from_row, from_col), (to_row, to_col))
        current_state: GameState after the action
        done: Whether the episode has ended
        winner: Winner of the game (1, -1, or 0 for draw, None if not done)
        info: Dictionary containing battle info from environment step
            Expected keys:
            - 'battle_result': int (1=attacker wins, -1=defender wins, 0=mutual)
            - 'attacker_type': PieceType (type of attacking piece)
            - 'defender_type': PieceType (type of defending piece)
            - 'revealed_in_step': List of revealed pieces
            - 'game_phase': str ("early", "mid", "end")
            - 'turn_count': int
        player_id: The player for whom we're calculating reward (1 or -1)
        config: Reward configuration (uses default if None)
        tracker: Per-game state tracker (uses temporary one if None)
        
    Returns:
        float: The calculated reward value
        
    Design Philosophy:
        - Terminal rewards are the LOUDEST signal (+/-10)
        - Combat outcomes provide MEDIUM signal (+1 to -0.5)
        - Positional/information gains provide SMALL signal (+0.1 to -0.02)
        - Step penalties create constant TIME PRESSURE (-0.01)
    """
    if config is None:
        config = AggressionRewardConfig()
        
    if tracker is None:
        # Create temporary tracker (won't persist between calls)
        tracker = AggressionRewardTracker(player_id)
    
    reward = 0.0
    
    # =========================================================================
    # 1. ANTI-STALL PENALTIES (Time Pressure)
    # =========================================================================
    
    # Small negative reward for every timestep
    # Forces agent to prefer shorter games and active play
    reward += config.step_penalty
    
    # =========================================================================
    # 2. TERMINAL REWARDS (Loudest Signal)
    # =========================================================================
    
    if done:
        if winner == player_id:
            # WIN: Flag capture or elimination victory
            reward += config.win_reward
        elif winner == -player_id:
            # LOSS: Opponent won
            reward += config.loss_penalty
        elif winner == 0:
            # DRAW/TIMEOUT: Heavy penalty to discourage stalling
            # This is CRITICAL for breaking the draw-forcing behavior
            reward += config.draw_penalty
        
        # Invalid move penalty (handled separately if action was invalid)
        if info.get('invalid_move', False):
            reward += config.invalid_move_penalty
            
        return reward  # Terminal state, no further rewards needed
    
    # =========================================================================
    # 3. COMBAT REWARDS (Rank-Based)
    # =========================================================================
    
    (r_from, c_from), (r_to, c_to) = action
    
    # Check if this was a battle (info should contain battle details)
    battle_result = info.get('battle_result', None)
    attacker_type = info.get('attacker_type', None)
    defender_type = info.get('defender_type', None)
    
    # Detect battle from board state change if not in info
    if battle_result is None:
        # Check if there was a piece at the target position
        prev_board = previous_state.board
        target_val = prev_board[r_to, c_to].item() if hasattr(prev_board[r_to, c_to], 'item') else prev_board[r_to, c_to]
        
        # Battle occurred if target was an enemy piece (non-zero, opposite sign)
        if target_val != 0 and target_val != 13:  # 13 is LAKE_SQUARE
            was_battle = (target_val > 0 and player_id == -1) or (target_val < 0 and player_id == 1)
            if was_battle:
                # Infer battle result from current state
                curr_board = current_state.board
                our_piece_at_target = curr_board[r_to, c_to].item() if hasattr(curr_board[r_to, c_to], 'item') else curr_board[r_to, c_to]
                our_piece_at_source = curr_board[r_from, c_from].item() if hasattr(curr_board[r_from, c_from], 'item') else curr_board[r_from, c_from]
                
                if player_id == 1:
                    if our_piece_at_target > 0:
                        battle_result = 1  # We won
                    elif our_piece_at_source == 0 and our_piece_at_target == 0:
                        battle_result = 0  # Mutual destruction
                    else:
                        battle_result = -1  # We lost
                else:
                    if our_piece_at_target < 0:
                        battle_result = 1  # We won
                    elif our_piece_at_source == 0 and our_piece_at_target == 0:
                        battle_result = 0  # Mutual destruction
                    else:
                        battle_result = -1  # We lost
                
                # Get piece types from board values
                moving_val = prev_board[r_from, c_from].item() if hasattr(prev_board[r_from, c_from], 'item') else prev_board[r_from, c_from]
                try:
                    attacker_type = PieceType(abs(int(moving_val)))
                    defender_type = PieceType(abs(int(target_val)))
                except (ValueError, KeyError):
                    attacker_type = None
                    defender_type = None
    
    if battle_result is not None and attacker_type is not None and defender_type is not None:
        # Get rank values
        attacker_rank = PIECE_RANK_VALUES.get(attacker_type, 5)
        defender_rank = PIECE_RANK_VALUES.get(defender_type, 5)
        rank_difference = defender_rank - attacker_rank
        
        if battle_result == 1:  # Attacker (us) wins
            # Base reward + bonus for killing higher-ranked pieces
            reward += config.attack_win_base + (rank_difference * config.rank_difference_scale)
            
            # Information bonus: we revealed the enemy piece
            reward += config.information_bonus
            
            # Update tracker
            tracker.update_enemy_piece_status(defender_type)
            tracker.revealed_enemy_pieces[(r_to, c_to)] = defender_type
            
            # Strategic bonuses for high-value captures
            if defender_type == PieceType.SPY and tracker.enemy_has_marshal:
                # Killing enemy Spy when they still have Marshal is valuable
                reward += config.spy_capture_bonus
                
            if defender_type == PieceType.MINER and tracker.enemy_has_bombs:
                # Killing enemy Miner when they still have Bombs is valuable
                reward += config.miner_capture_bonus
                
        elif battle_result == -1:  # Attacker (us) loses
            # Penalty for losing, BUT less severe than doing nothing
            # This encourages probing attacks even at risk
            reward += config.attack_lose_penalty
            
            # CRUCIAL: Information bonus for revealing enemy piece
            # Even losing the battle gave us valuable intel
            reward += config.information_bonus
            
            # Record what we learned
            tracker.revealed_enemy_pieces[(r_to, c_to)] = defender_type
            
        else:  # Mutual destruction (battle_result == 0)
            # Both pieces die - small info bonus for both reveals
            reward += config.mutual_info_bonus * 2
            tracker.revealed_enemy_pieces[(r_to, c_to)] = defender_type
    
    # =========================================================================
    # 4. TERRITORIAL REWARDS (Crossing the River)
    # =========================================================================
    
    # Determine "enemy territory" based on player
    if player_id == 1:
        # Player 1 starts rows 6-9, enemy is rows 0-4 (row 5 is river)
        enemy_territory_threshold = 4  # Rows 0-4 are enemy territory
        is_forward = r_to < r_from
        in_enemy_territory = r_to <= enemy_territory_threshold
        in_own_territory = r_to >= 6
    else:
        # Player -1 starts rows 0-3, enemy is rows 5-9 (row 4 is river)
        enemy_territory_threshold = 5
        is_forward = r_to > r_from
        in_enemy_territory = r_to >= enemy_territory_threshold
        in_own_territory = r_to <= 3
    
    # First-time territory advancement bonus
    if in_enemy_territory and (r_to, c_to) not in tracker.pieces_in_enemy_territory:
        # First time this piece reached enemy territory
        reward += config.territory_advance
        tracker.pieces_in_enemy_territory.add((r_to, c_to))
    
    # Retreat penalty (moving backward into own territory)
    prev_row = tracker.piece_previous_rows.get((r_from, c_from), r_from)
    if player_id == 1:
        is_retreating = r_to > prev_row and in_own_territory
    else:
        is_retreating = r_to < prev_row and in_own_territory
        
    if is_retreating:
        reward += config.retreat_penalty
    
    # Update position tracking
    tracker.piece_previous_rows[(r_to, c_to)] = r_to
    if (r_from, c_from) in tracker.piece_previous_rows:
        del tracker.piece_previous_rows[(r_from, c_from)]
    
    return reward


def create_reward_wrapper(player_id: int = 1, config: Optional[AggressionRewardConfig] = None):
    """
    Create a reward calculation wrapper with persistent per-game tracking.
    
    Usage in training loop:
        reward_fn = create_reward_wrapper(player_id=1)
        
        # At episode start:
        reward_fn.reset()
        
        # At each step:
        shaped_reward = reward_fn(prev_state, action, curr_state, done, winner, info)
    
    Args:
        player_id: The player ID (1 or -1)
        config: Optional reward configuration
        
    Returns:
        A callable wrapper with reset() method
    """
    if config is None:
        # Load from training_config.py settings
        config = AggressionRewardConfig.from_training_config()
    
    tracker = AggressionRewardTracker(player_id)
    
    class RewardWrapper:
        def __init__(self):
            self.tracker = tracker
            self.config = config
            self.player_id = player_id
            
        def __call__(self, previous_state, action, current_state, done, winner, info):
            return calculate_reward(
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
# INTEGRATION HELPER for train_dqn.py
# =============================================================================

def integrate_aggression_rewards(
    env_reward: float,
    previous_state: GameState,
    action: Tuple[Tuple[int, int], Tuple[int, int]],
    current_state: GameState,
    done: bool,
    winner: Optional[int],
    info: Dict[str, Any],
    player_id: int = 1,
    aggression_weight: float = 1.0,
    env_weight: float = 0.5
) -> float:
    """
    Combine environment rewards with aggression-based rewards.
    
    This helper can be used to gradually integrate the aggression rewards
    with the existing environment rewards during training.
    
    Args:
        env_reward: Original reward from environment.step()
        previous_state: GameState before action
        action: The action taken
        current_state: GameState after action
        done: Episode termination flag
        winner: Game winner
        info: Step info dictionary
        player_id: Player ID
        aggression_weight: Weight for aggression rewards
        env_weight: Weight for original environment rewards
        
    Returns:
        Combined reward
    """
    aggression_reward = calculate_reward(
        previous_state=previous_state,
        action=action,
        current_state=current_state,
        done=done,
        winner=winner,
        info=info,
        player_id=player_id
    )
    
    return (env_weight * env_reward) + (aggression_weight * aggression_reward)
