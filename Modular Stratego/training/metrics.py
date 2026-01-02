"""
MetricsTracker: Centralized training metrics management.

Extracted from train_dqn.py for better maintainability.
"""

import json
import os
from typing import Dict, List, Optional, Any


class MetricsTracker:
    """
    Centralized training metrics tracking.
    
    Encapsulates all metrics collection, aggregation, and persistence.
    """
    
    def __init__(self, save_dir: str = "dqn_models"):
        self.save_dir = save_dir
        self.metrics = self._init_metrics()
    
    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize empty metrics dictionary."""
        return {
            # Core agent metrics
            'rewards_p1': [], 
            'rewards_p2': [],
            'wins_p1': 0, 
            'wins_p2': 0, 
            'draws': 0,
            'wins_p1_history': [], 
            'wins_p2_history': [],
            'lengths': [],
            'losses_p1': [],
            'loss_steps_p1': [],
            'avg_loss_p1_history': [],
            
            # Win type tracking
            'wins_by_flag': 0,
            'wins_by_depletion': 0,
            'wins_by_flag_history': [],
            'wins_by_depletion_history': [],
            
            # Loss type tracking (how P1 lost)
            'losses_by_flag': 0,
            'losses_by_depletion': 0,
            'losses_by_flag_history': [],
            'losses_by_depletion_history': [],
            
            # PBS evaluator metrics
            'pbs_eval1_losses': [],
            'pbs_eval1_buffer_sizes': [],
            'pbs_eval1_accuracy': [],
            
            # AAREN metrics
            'aaren_loss': [],
            'aaren_accuracy': [],
            'aaren_buffer_size': [],
            
            # Additional metrics
            'avg_q_values_p1': [],
            'avg_entropy_p1': [],
            'win_rate_100': [],
            'phase_history': [],
            'pbs_accuracy': [],
            'episode_end_steps': [],
            
            # State tracking
            'global_step': 0,
            'last_plot_episode': 0,
            'last_save_episode': 0,
        }
    
    def load(self) -> bool:
        """Load metrics from file. Returns True if loaded successfully."""
        path = os.path.join(self.save_dir, "training_history.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                self.metrics.update(loaded)
                return True
            except Exception:
                return False
        return False
    
    def save(self) -> None:
        """Save metrics to file."""
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def record_episode_end(
        self,
        winner: int,
        win_type: Optional[str],
        reward_p1: float,
        reward_p2: float,
        episode_length: int,
        avg_loss: float,
        global_step: int,
        phase: int = 1,
        pbs_metrics: Optional[Dict] = None,
        aaren_metrics: Optional[Dict] = None,
        avg_q: float = 0.0,
        avg_entropy: float = 0.0
    ) -> None:
        """Record all metrics for a completed episode."""
        # Win tracking
        if winner == 1:
            self.metrics['wins_p1'] += 1
            if win_type == 'flag_capture':
                self.metrics['wins_by_flag'] += 1
            elif win_type == 'no_moves':
                self.metrics['wins_by_depletion'] += 1
        elif winner == -1:
            self.metrics['wins_p2'] += 1
        else:
            self.metrics['draws'] += 1
        
        # Core metrics
        self.metrics['rewards_p1'].append(reward_p1)
        self.metrics['rewards_p2'].append(reward_p2)
        self.metrics['lengths'].append(episode_length)
        self.metrics['avg_loss_p1_history'].append(avg_loss)
        
        # History tracking
        self.metrics['wins_p1_history'].append(self.metrics['wins_p1'])
        self.metrics['wins_p2_history'].append(self.metrics['wins_p2'])
        self.metrics['wins_by_flag_history'].append(self.metrics['wins_by_flag'])
        self.metrics['wins_by_depletion_history'].append(self.metrics['wins_by_depletion'])
        self.metrics['phase_history'].append(phase)
        self.metrics['episode_end_steps'].append(global_step)
        
        # PBS metrics
        if pbs_metrics:
            self.metrics['pbs_eval1_losses'].append(pbs_metrics.get('loss', 0.0))
            self.metrics['pbs_eval1_buffer_sizes'].append(pbs_metrics.get('buffer_size', 0))
            self.metrics['pbs_eval1_accuracy'].append(pbs_metrics.get('accuracy', 0.0))
        else:
            self.metrics['pbs_eval1_losses'].append(0.0)
            self.metrics['pbs_eval1_buffer_sizes'].append(0)
            self.metrics['pbs_eval1_accuracy'].append(0.0)
        
        # AAREN metrics
        if aaren_metrics:
            self.metrics['aaren_loss'].append(aaren_metrics.get('loss', 0.0))
            self.metrics['aaren_accuracy'].append(aaren_metrics.get('accuracy', 0.0))
            self.metrics['aaren_buffer_size'].append(aaren_metrics.get('buffer_size', 0))
        
        # Q-value and entropy
        self.metrics['avg_q_values_p1'].append(avg_q)
        self.metrics['avg_entropy_p1'].append(avg_entropy)
    
    def record_loss(self, loss: float, global_step: int) -> None:
        """Record a training loss."""
        self.metrics['losses_p1'].append(loss)
        self.metrics['loss_steps_p1'].append(global_step)
    
    def update_win_rate(self) -> None:
        """Calculate and record sliding window win rate."""
        if len(self.metrics['wins_p1_history']) >= 100:
            wins_100 = self.metrics['wins_p1_history'][-1] - self.metrics['wins_p1_history'][-100]
            win_rate = wins_100 / 100.0
        else:
            total_games = self.metrics['wins_p1'] + self.metrics['wins_p2'] + self.metrics['draws']
            win_rate = self.metrics['wins_p1'] / max(total_games, 1)
        self.metrics['win_rate_100'].append(win_rate)
    
    def get_recent_reward(self, window: int = 10) -> float:
        """Get average reward over recent episodes."""
        if not self.metrics['rewards_p1']:
            return 0.0
        return sum(self.metrics['rewards_p1'][-window:]) / min(window, len(self.metrics['rewards_p1']))
    
    def get_global_step(self) -> int:
        """Get current global step."""
        return self.metrics.get('global_step', 0)
    
    def set_global_step(self, step: int) -> None:
        """Set current global step."""
        self.metrics['global_step'] = step
    
    def should_plot(self, episode: int, interval: int) -> bool:
        """Check if we should plot at this episode."""
        milestone = (episode // interval) * interval
        return episode > 0 and milestone > self.metrics.get('last_plot_episode', 0)
    
    def mark_plotted(self, episode: int) -> None:
        """Mark that we've plotted at this episode."""
        self.metrics['last_plot_episode'] = episode
    
    def should_save(self, episode: int, interval: int, start_episode: int) -> bool:
        """Check if we should save at this episode."""
        milestone = (episode // interval) * interval
        return (episode > 0 and 
                milestone > self.metrics.get('last_save_episode', 0) and 
                episode != start_episode)
    
    def mark_saved(self, episode: int) -> None:
        """Mark that we've saved at this episode."""
        self.metrics['last_save_episode'] = episode
    
    def get_plot_data(self) -> Dict[str, Any]:
        """Get data formatted for plotting."""
        num_episodes = len(self.metrics['rewards_p1'])
        return {
            'episode_history': list(range(1, num_episodes + 1)),
            'rewards_history': {
                'agent1': self.metrics['rewards_p1'],
                'agent2': self.metrics['rewards_p2']
            },
            'wins_history': {
                'agent1': self.metrics['wins_p1_history'],
                'agent2': self.metrics['wins_p2_history']
            },
            'policy_loss_history': {'agent1': self.metrics['losses_p1']},
            'phase_history': self.metrics.get('phase_history', []),
            'loss_steps': self.metrics.get('loss_steps_p1'),
            'episode_end_steps': self.metrics.get('episode_end_steps')
        }
