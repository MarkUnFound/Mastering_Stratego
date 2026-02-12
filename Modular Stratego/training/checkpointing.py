"""
Checkpointer: Model saving, loading, and plotting.

Extracted from train_dqn.py for better maintainability.
"""

import os
import glob
from typing import Optional, Dict, Any, Tuple

from training_visualizer import plot_training_progress, plot_additional_metrics


class Checkpointer:
    """
    Handles model checkpointing, loading, and plot generation.
    """
    
    def __init__(self, save_dir: str = "dqn_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    @staticmethod
    def extract_episode(filename: str) -> int:
        """Extract episode number from checkpoint filename."""
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1
    
    def find_latest_checkpoint(self, pattern: str) -> Optional[Tuple[str, int]]:
        """
        Find the latest checkpoint matching pattern.
        Returns (filepath, episode) or None.
        """
        files = glob.glob(os.path.join(self.save_dir, pattern))
        if not files:
            return None
        
        files.sort(key=self.extract_episode, reverse=True)
        latest = files[0]
        episode = self.extract_episode(latest)
        return (latest, episode)
    
    def load_agent_models(self, agent1, agent2) -> int:
        """
        Load latest agent models.
        Returns start_episode (0 if no models found).
        """
        start_episode = 0
        
        # Load Agent 1
        result = self.find_latest_checkpoint("agent1_rainbow_episode_*.pth")
        if result:
            path, episode = result
            try:
                agent1.load_model(path)
                start_episode = episode
                print(f"[OK] Loaded Agent 1 from {path}")
            except Exception as e:
                print(f"[WARN] Failed to load Agent 1: {e}")
        
        # Load Agent 2
        result = self.find_latest_checkpoint("agent2_rainbow_episode_*.pth")
        if result:
            path, _ = result
            try:
                agent2.load_model(path)
                print(f"[OK] Loaded Agent 2 from {path}")
            except Exception as e:
                print(f"[WARN] Failed to load Agent 2: {e}")
        
        return start_episode
    
    # load_pbs_evaluators removed — AAREN replaced PBS
    
    def save_checkpoint(
        self,
        episode: int,
        agent1,
        agent2,
        league_manager,
        curriculum,
        metrics_tracker,
        league_interval: int
    ) -> None:
        """Save all models and state."""
        # Save Agent 1
        agent1_path = os.path.join(self.save_dir, f"agent1_rainbow_episode_{episode}.pth")
        agent1.save_model(agent1_path)
        
        # Export to league
        if episode % league_interval == 0:
            league_manager.save_agent(agent1_path, episode)
        
        # PBS evaluator saving removed — AAREN replaced PBS
        
        # Save curriculum state
        if curriculum:
            curriculum.save_state()
        
        # Save metrics
        metrics_tracker.set_global_step(metrics_tracker.get_global_step())
        metrics_tracker.save()
    
    def plot_progress(
        self,
        episode: int,
        metrics_tracker,
        global_step: int,
        num_envs: int
    ) -> None:
        """Generate training progress plots."""
        try:
            plot_data = metrics_tracker.get_plot_data()
            
            # Main progress plot
            plot_training_progress(
                episode_history=plot_data['episode_history'],
                rewards_history=plot_data['rewards_history'],
                wins_history=plot_data['wins_history'],
                policy_loss_history=plot_data['policy_loss_history'],
                save_path=os.path.join(self.save_dir, f"training_progress_episode_{episode}.png"),
                total_episodes=episode,
                total_steps=global_step,
                num_envs=num_envs,
                phase_history=plot_data['phase_history'],
                loss_steps=plot_data['loss_steps'],
                episode_end_steps=plot_data['episode_end_steps']
            )
            
            # Additional metrics plot
            num_eps = len(plot_data['episode_history'])
            plot_additional_metrics(
                episode_history=plot_data['episode_history'],
                episode_lengths={'agent1': metrics_tracker.metrics.get('lengths', [0] * num_eps)},
                win_rate_history={'agent1': metrics_tracker.metrics.get('win_rate_100', [0.0] * num_eps)},
                avg_q_history={'agent1': metrics_tracker.metrics.get('avg_q_values_p1', [0.0] * num_eps)},
                entropy_history={'agent1': metrics_tracker.metrics.get('avg_entropy_p1', [0.0] * num_eps)},
                save_path=os.path.join(self.save_dir, f"additional_metrics_episode_{episode}.png")
            )
            
            # AAREN End-to-End Training Visualization
            try:
                from training_visualizer import plot_aaren_progress
                plot_aaren_progress(
                    episode_history=plot_data['episode_history'],
                    aaren_losses=metrics_tracker.metrics.get('aaren_loss', []),
                    aaren_accuracies=metrics_tracker.metrics.get('aaren_accuracy', []),
                    aaren_buffer_sizes=metrics_tracker.metrics.get('aaren_buffer_size', []),
                    aaren_grad_norms=metrics_tracker.metrics.get('aaren_grad_norm', []),
                    aaren_embedding_stds=metrics_tracker.metrics.get('aaren_embedding_std', []),
                    dqn_grad_norms=metrics_tracker.metrics.get('dqn_grad_norm', []),
                    save_path=os.path.join(self.save_dir, f"aaren_progress_episode_{episode}.png"),
                    total_episodes=episode
                )
            except Exception as aaren_plot_err:
                print(f"[WARN] AAREN progress plot failed: {aaren_plot_err}")
            
        except Exception as e:
            print(f"[WARN] Could not plot training progress: {e}")
