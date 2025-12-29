"""
LaneManager: Manages parallel lane states for multi-environment training.

Extracted from train_dqn.py for better maintainability.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from distributional_reward import create_unified_reward_shaper, StrategoRewardConfig


class LaneManager:
    """
    Manages state for all parallel training lanes.
    
    Encapsulates:
    - Lane game states, valid moves, current player
    - Episode rewards and step counts
    - Opponent selection and tracking
    - Reward shapers per lane
    """
    
    def __init__(self, num_lanes: int, config: StrategoRewardConfig, device: str = 'cuda'):
        self.num_lanes = num_lanes
        self.config = config
        self.device = device
        
        # Per-lane state tracking
        self.game_states: List[Any] = [None] * num_lanes
        self.valid_moves: List[Any] = [None] * num_lanes
        self.current_player: List[int] = [1] * num_lanes
        
        # Episode metrics per lane
        self.episode_rewards_p1: List[float] = [0.0] * num_lanes
        self.episode_rewards_p2: List[float] = [0.0] * num_lanes
        self.step_counts: List[int] = [0] * num_lanes
        
        # Loss tracking per lane
        self.episode_loss_sum_p1 = np.zeros(num_lanes)
        self.episode_loss_count_p1 = np.zeros(num_lanes, dtype=int)
        
        # Opponent tracking per lane
        self.opponent_types: List[str] = ["self"] * num_lanes
        self.opponent_uses_pbs: List[bool] = [True] * num_lanes
        self.current_opponents: List[Any] = [None] * num_lanes
        
        # Pending transitions for P1 (when waiting for P2's move)
        self.pending_transitions: List[Optional[Dict]] = [None] * num_lanes
        
        # Reward shapers per lane
        self.reward_shapers_p1 = [
            create_unified_reward_shaper(player_id=1, config=config, device=device)
            for _ in range(num_lanes)
        ]
        self.reward_shapers_p2 = [
            create_unified_reward_shaper(player_id=-1, config=config, device=device)
            for _ in range(num_lanes)
        ]
    
    def reset_lane(self, lane_idx: int) -> None:
        """Reset a single lane's state after episode completion."""
        self.episode_rewards_p1[lane_idx] = 0.0
        self.episode_rewards_p2[lane_idx] = 0.0
        self.episode_loss_sum_p1[lane_idx] = 0.0
        self.episode_loss_count_p1[lane_idx] = 0
        self.step_counts[lane_idx] = 0
        self.current_player[lane_idx] = 1
        self.pending_transitions[lane_idx] = None
        
        # Reset reward shapers
        self.reward_shapers_p1[lane_idx].reset()
        self.reward_shapers_p2[lane_idx].reset()
    
    def reset_all(self) -> None:
        """Reset all lanes."""
        for i in range(self.num_lanes):
            self.reset_lane(i)
    
    def set_game_state(self, lane_idx: int, state: Any, valid_moves: Any) -> None:
        """Set game state for a lane."""
        self.game_states[lane_idx] = state
        self.valid_moves[lane_idx] = valid_moves
    
    def set_opponent(self, lane_idx: int, opp_type: str, uses_pbs: bool, opponent: Any) -> None:
        """Set opponent for a lane."""
        self.opponent_types[lane_idx] = opp_type
        self.opponent_uses_pbs[lane_idx] = uses_pbs
        self.current_opponents[lane_idx] = opponent
    
    def increment_step(self, lane_idx: int) -> None:
        """Increment step count for a lane."""
        self.step_counts[lane_idx] += 1
    
    def switch_player(self, lane_idx: int) -> None:
        """Switch current player for a lane."""
        self.current_player[lane_idx] *= -1
    
    def add_reward(self, lane_idx: int, reward: float, player: int) -> None:
        """Add reward to a lane's episode total."""
        if player == 1:
            self.episode_rewards_p1[lane_idx] += reward
        else:
            self.episode_rewards_p2[lane_idx] += reward
    
    def add_loss(self, lane_idx: int, loss: float) -> None:
        """Add training loss for a lane."""
        self.episode_loss_sum_p1[lane_idx] += loss
        self.episode_loss_count_p1[lane_idx] += 1
    
    def get_avg_loss(self, lane_idx: int) -> float:
        """Get average loss for a lane's current episode."""
        if self.episode_loss_count_p1[lane_idx] > 0:
            return self.episode_loss_sum_p1[lane_idx] / self.episode_loss_count_p1[lane_idx]
        return 0.0
    
    def set_pending_transition(self, lane_idx: int, transition: Dict) -> None:
        """Store a pending P1 transition while waiting for P2."""
        self.pending_transitions[lane_idx] = transition
    
    def get_pending_transition(self, lane_idx: int) -> Optional[Dict]:
        """Get and clear pending transition for a lane."""
        transition = self.pending_transitions[lane_idx]
        self.pending_transitions[lane_idx] = None
        return transition
    
    def get_p1_lanes(self) -> List[int]:
        """Get indices of lanes where P1 is acting."""
        return [i for i in range(self.num_lanes) if self.current_player[i] == 1]
    
    def get_p2_lanes(self) -> List[int]:
        """Get indices of lanes where P2 is acting."""
        return [i for i in range(self.num_lanes) if self.current_player[i] == -1]
    
    def get_reward_p1(self, lane_idx: int, prev_state: Any, action: Any, 
                      curr_state: Any, done: bool, winner: Any, info: Dict) -> float:
        """Calculate P1 reward for a lane."""
        return self.reward_shapers_p1[lane_idx](
            prev_state, action, curr_state, done, winner, info
        )
    
    def get_reward_p2(self, lane_idx: int, prev_state: Any, action: Any,
                      curr_state: Any, done: bool, winner: Any, info: Dict) -> float:
        """Calculate P2 reward for a lane."""
        return self.reward_shapers_p2[lane_idx](
            prev_state, action, curr_state, done, winner, info
        )
