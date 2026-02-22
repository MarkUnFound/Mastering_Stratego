"""
Checkpointer: Model saving, loading, and plotting.

Extracted from train_dqn.py for better maintainability.
"""

import os
import glob
import tarfile
import torch
import shutil
import tempfile
from typing import Optional, Dict, Any, Tuple

from training_visualizer import plot_training_progress, plot_additional_metrics


class Checkpointer:
    """
    Handles model checkpointing, loading, and plot generation.
    Supports .tar.gz for continuous training and .pt for league.
    """
    
    def __init__(self, save_dir: str = "dqn_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    @staticmethod
    def extract_episode(filename: str) -> int:
        """Extract episode number from checkpoint filename."""
        try:
            # Handle .tar.gz and .pt
            base = os.path.basename(filename)
            if base.endswith('.tar.gz'):
                 return int(base.split('_episode_')[1].replace('.tar.gz', ''))
            elif base.endswith('.pt'):
                 return int(base.split('_episode_')[1].replace('.pt', ''))
            elif base.endswith('.pth'):
                 return int(base.split('_episode_')[1].replace('.pth', ''))
            return int(base.split('_')[-1].split('.')[0])
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
        Prioritizes .tar.gz (full state) over .pt (weights only).
        Returns start_episode (0 if no models found).
        """
        start_episode = 0
        
        # 1. Load Agent 1
        # Try .tar.gz first (Full state for seamless resumption)
        result = self.find_latest_checkpoint("agent1_rainbow_episode_*.tar.gz")
        if result:
            path, episode = result
            try:
                self._load_from_archive(agent1, path)
                start_episode = episode
                print(f"[OK] Loaded Agent 1 (Full State) from {path}")
            except Exception as e:
                print(f"[WARN] Failed to load Agent 1 archive {path}: {e}")
        
        # Fallback to .pt if no archive found or if archive was older than latest .pt
        pt_result = self.find_latest_checkpoint("agent1_rainbow_episode_*.pt")
        if pt_result:
            pt_path, pt_episode = pt_result
            if pt_episode > start_episode:
                try:
                    state = torch.load(pt_path, map_location=agent1.device, weights_only=True)
                    agent1.load_state_dict(state)
                    start_episode = pt_episode
                    print(f"[OK] Loaded Agent 1 (Weights Only) from {pt_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load Agent 1 weights {pt_path}: {e}")

        # 2. Load Agent 2
        # Try .tar.gz first
        result = self.find_latest_checkpoint("agent2_rainbow_episode_*.tar.gz")
        if result:
            path, pt_episode = result
            if pt_episode >= start_episode:
                try:
                    self._load_from_archive(agent2, path)
                    print(f"[OK] Loaded Agent 2 (Full State) from {path}")
                except Exception as e:
                    print(f"[WARN] Failed to load Agent 2 archive {path}: {e}")
        
        # Fallback to .pt
        pt_result = self.find_latest_checkpoint("agent2_rainbow_episode_*.pt")
        if pt_result:
            pt_path, pt_episode = pt_result
            # Check if this .pt is newer than whatever we loaded (if anything)
            if pt_episode >= start_episode: # Use >= for Agent 2 consistency
                try:
                    state = torch.load(pt_path, map_location=agent2.device, weights_only=True)
                    agent2.load_state_dict(state)
                    print(f"[OK] Loaded Agent 2 (Weights Only) from {pt_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load Agent 2 weights {pt_path}: {e}")
        
        return start_episode

    def _load_from_archive(self, agent, archive_path: str):
        """Extract and load agent state from a .tar.gz archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
            
            state_path = os.path.join(temp_dir, "agent_state.pt")
            if os.path.exists(state_path):
                # Using weights_only=False because we trust our own archives and buffers might have complex types
                full_state = torch.load(state_path, map_location=agent.device, weights_only=False)
                agent.load_state_dict(full_state)
    
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
        """Save periodic models (checkpoints) as .pt files."""
        # Save Agent 1 (Weights only for periodic checkpoints)
        agent1_path = os.path.join(self.save_dir, f"agent1_rainbow_episode_{episode}.pt")
        agent1_tmp = agent1_path + ".tmp"
        torch.save(agent1.state_dict(include_buffers=False), agent1_tmp)
        os.replace(agent1_tmp, agent1_path)

        # Save Agent 2 (if it's a trainable agent)
        if hasattr(agent2, 'state_dict'):
            agent2_path = os.path.join(self.save_dir, f"agent2_rainbow_episode_{episode}.pt")
            agent2_tmp = agent2_path + ".tmp"
            torch.save(agent2.state_dict(include_buffers=False), agent2_tmp)
            os.replace(agent2_tmp, agent2_path)
        
        # Export to league (.pt for inference only)
        if episode % league_interval == 0:
            league_path = os.path.join(self.save_dir, f"agent1_league_episode_{episode}.pt")
            league_tmp = league_path + ".tmp"
            # Reuse the periodic save if possible, or save explicitly
            league_state = agent1.state_dict(include_buffers=False)
            if 'optimizer_state_dict' in league_state:
                del league_state['optimizer_state_dict']
            torch.save(league_state, league_tmp)
            os.replace(league_tmp, league_path)
            league_manager.save_agent(league_path, episode)
        
        # Save curriculum state
        if curriculum:
            curriculum.save_state()
        
        # Save metrics
        metrics_tracker.save()

    def save_full_state(
        self,
        episode: int,
        agent1,
        agent2,
        curriculum,
        metrics_tracker
    ) -> str:
        """
        Save the complete training state (.tar.gz) for seamless resumption.
        Used primarily when training is interrupted.
        """
        # Save Agent 1 Full State (.tar.gz)
        agent1_archive_path = os.path.join(self.save_dir, f"agent1_rainbow_episode_{episode}.tar.gz")
        self._save_to_archive(agent1, agent1_archive_path)

        # Save Agent 2 Full State
        if hasattr(agent2, 'state_dict'):
            agent2_archive_path = os.path.join(self.save_dir, f"agent2_rainbow_episode_{episode}.tar.gz")
            self._save_to_archive(agent2, agent2_archive_path)

        # Ensure curriculum and metrics are also synced
        if curriculum:
            curriculum.save_state()
        metrics_tracker.save()
        
        return agent1_archive_path

    def _save_to_archive(self, agent, archive_path: str):
        """Save agent full state to a compressed .tar.gz archive."""
        archive_tmp = archive_path + ".tmp"
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "agent_state.pt")
            # Get full state with buffers
            full_state = agent.state_dict(include_buffers=True)
            torch.save(full_state, state_path)
            
            # Compress into tar.gz
            with tarfile.open(archive_tmp, "w:gz") as tar:
                tar.add(state_path, arcname="agent_state.pt")
                
        # Atomically replace to prevent corrupted loads if interrupted
        os.replace(archive_tmp, archive_path)
    
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
