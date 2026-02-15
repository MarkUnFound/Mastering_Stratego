"""
PBT Dashboard Visualization

Creates an aggregate visualization of all PBT workers' training progress,
showing performance comparisons and exploitation events.

Theory:
Population-Based Training benefits from visualizing the entire population's
progress, allowing researchers to:
1. Identify which hyperparameter configurations perform best
2. Track when exploitation events improve overall population fitness
3. Detect training instabilities or worker failures early

Usage:
    from pbt_visualizer import PBTDashboard
    dashboard = PBTDashboard(metrics_file='pbt_metrics.csv', output_dir='dqn_models')
    dashboard.update()  # Called periodically by supervisor
"""

import os
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Use Agg backend for non-interactive plotting (thread-safe)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class PBTDashboard:
    """
    Generates aggregate visualizations for PBT training.
    
    Reads metrics from the shared CSV file and creates plots comparing
    all workers' performance over time.
    """
    
    def __init__(self, metrics_file: str = 'pbt_metrics.csv', 
                 output_dir: str = 'dqn_models',
                 update_interval: int = 100):
        """
        Initialize the PBT dashboard.
        
        Args:
            metrics_file: Path to shared PBT metrics CSV
            output_dir: Directory to save dashboard images
            update_interval: Episodes between dashboard updates
        """
        self.metrics_file = metrics_file
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.last_update_episode = 0
        self.exploitation_events: List[Tuple[float, str]] = []  # (timestamp, description)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def record_exploitation(self, description: str = "Exploitation"):
        """Record an exploitation event for visualization."""
        self.exploitation_events.append((datetime.now().timestamp(), description))
    
    def read_metrics(self) -> Dict[int, Dict[str, List]]:
        """
        Read metrics from CSV file grouped by worker ID.
        
        Returns:
            Dictionary mapping worker_id to lists of metrics
        """
        workers = defaultdict(lambda: {
            'episodes': [],
            'rewards': [],
            'wins': [],
            'win_rates': [],
            'losses': [],
            'timestamps': []
        })
        
        if not os.path.exists(self.metrics_file):
            return dict(workers)
        
        try:
            with open(self.metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    worker_id = int(row.get('worker_id', -1))
                    if worker_id >= 0:
                        workers[worker_id]['episodes'].append(int(row.get('episode', 0)))
                        workers[worker_id]['rewards'].append(float(row.get('reward', 0)))
                        workers[worker_id]['wins'].append(int(row.get('win', 0)))
                        workers[worker_id]['win_rates'].append(float(row.get('win_rate', 0)))
                        workers[worker_id]['losses'].append(float(row.get('avg_loss', 0)))
                        workers[worker_id]['timestamps'].append(float(row.get('timestamp', 0)))
        except Exception as e:
            print(f"[PBT Dashboard] Error reading metrics: {e}")
        
        return dict(workers)
    
    def _smooth(self, data: List[float], window: int = 50) -> np.ndarray:
        """Apply rolling average smoothing to data."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def read_exploitation_events(self) -> List[Dict]:
        """
        Read exploitation events from JSON file.
        
        Returns:
            List of event dictionaries with timestamp, worker_id, etc.
        """
        events_file = os.path.join(self.output_dir, 'exploitation_events.json')
        
        if not os.path.exists(events_file):
            return []
        
        try:
            import json
            with open(events_file, 'r') as f:
                data = json.load(f)
                return data.get('events', [])
        except Exception as e:
            print(f"[PBT Dashboard] Error reading exploitation events: {e}")
            return []
    
    def _get_exploitation_episodes(self, events: List[Dict]) -> List[Tuple[int, str]]:
        """
        Convert exploitation events to episode-based markers.
        
        Returns:
            List of (episode, reason) tuples for vertical line markers
        """
        # Group by exploit_round to get unique exploitation times
        rounds = {}
        for event in events:
            round_id = event['exploit_round']
            if round_id not in rounds:
                rounds[round_id] = {
                    'episode': event['episode_count'],
                    'reason': event['reason']
                }
        
        return [(r['episode'], r['reason']) for r in rounds.values()]
    
    def update(self, force: bool = False):
        """
        Update the dashboard visualization.
        
        Args:
            force: If True, update regardless of interval
        """
        workers = self.read_metrics()
        
        if not workers:
            return
        
        # Check if we should update
        max_episode = max(
            max(w['episodes']) if w['episodes'] else 0 
            for w in workers.values()
        )
        
        if not force and max_episode - self.last_update_episode < self.update_interval:
            return
        
        self.last_update_episode = max_episode
        
        # Read exploitation events for markers
        events = self.read_exploitation_events()
        exploit_markers = self._get_exploitation_episodes(events)
        
        # Create dashboard
        self._create_dashboard(workers, exploit_markers)
        self._create_worker_comparison(workers)
    
    def _create_dashboard(self, workers: Dict[int, Dict[str, List]], 
                           exploit_markers: List[Tuple[int, str]] = None):
        """Create the main PBT dashboard with multiple subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PBT Training Dashboard', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(workers)))
        
        # Define colors for exploitation markers
        marker_colors = {
            'scheduled': '#e74c3c',  # Red for scheduled exploitation
            'adaptive': '#f39c12',   # Orange for adaptive culling
            'plateau': '#f39c12',    # Orange (same as adaptive)
            'leader_gap': '#9b59b6'  # Purple for leader gap
        }
        
        # --- Plot 1: Mean Reward Over Time ---
        ax1 = axes[0, 0]
        for i, (worker_id, metrics) in enumerate(sorted(workers.items())):
            if metrics['rewards']:
                smoothed = self._smooth(metrics['rewards'])
                episodes = np.arange(len(smoothed)) + 25  # Offset for smoothing
                ax1.plot(episodes, smoothed, label=f'Worker {worker_id}', 
                        color=colors[i], alpha=0.8)
        
        # Add exploitation markers as vertical lines
        if exploit_markers:
            for episode, reason in exploit_markers:
                color = marker_colors.get(reason, '#e74c3c')
                ax1.axvline(x=episode, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mean Reward (smoothed)')
        ax1.set_title('Reward Progress by Worker')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Win Rate Over Time ---
        ax2 = axes[0, 1]
        for i, (worker_id, metrics) in enumerate(sorted(workers.items())):
            if metrics['win_rates']:
                smoothed = self._smooth(metrics['win_rates'])
                episodes = np.arange(len(smoothed)) + 25
                ax2.plot(episodes, smoothed, label=f'Worker {worker_id}',
                        color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate (smoothed)')
        ax2.set_title('Win Rate Progress by Worker')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # --- Plot 3: Current Performance Comparison (Bar Chart) ---
        ax3 = axes[1, 0]
        worker_ids = sorted(workers.keys())
        current_rewards = []
        for wid in worker_ids:
            recent = workers[wid]['rewards'][-100:]
            current_rewards.append(np.mean(recent) if recent else 0)
        
        bars = ax3.bar(range(len(worker_ids)), current_rewards, color=colors[:len(worker_ids)])
        ax3.set_xticks(range(len(worker_ids)))
        ax3.set_xticklabels([f'W{wid}' for wid in worker_ids])
        ax3.set_xlabel('Worker')
        ax3.set_ylabel('Mean Reward (last 100 eps)')
        ax3.set_title('Current Worker Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Highlight best performer
        if current_rewards:
            best_idx = np.argmax(current_rewards)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        # --- Plot 4: Training Loss Comparison ---
        ax4 = axes[1, 1]
        for i, (worker_id, metrics) in enumerate(sorted(workers.items())):
            if metrics['losses']:
                smoothed = self._smooth(metrics['losses'])
                episodes = np.arange(len(smoothed)) + 25
                ax4.plot(episodes, smoothed, label=f'Worker {worker_id}',
                        color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Loss (smoothed)')
        ax4.set_title('Training Loss by Worker')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, 'pbt_dashboard.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[PBT Dashboard] Saved to {output_path}")
    
    def _create_worker_comparison(self, workers: Dict[int, Dict[str, List]]):
        """Create a detailed worker comparison summary."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for comparison
        worker_ids = sorted(workers.keys())
        num_workers = len(worker_ids)
        
        metrics_names = ['Mean Reward', 'Win Rate', 'Total Episodes']
        data = np.zeros((num_workers, 3))
        
        for i, wid in enumerate(worker_ids):
            m = workers[wid]
            data[i, 0] = np.mean(m['rewards'][-100:]) if m['rewards'] else 0
            data[i, 1] = np.mean(m['win_rates'][-100:]) if m['win_rates'] else 0
            data[i, 2] = len(m['episodes'])
        
        # Normalize for visualization
        data_norm = data.copy()
        for j in range(3):
            col_max = data[:, j].max()
            if col_max > 0:
                data_norm[:, j] = data[:, j] / col_max
        
        # Create grouped bar chart
        x = np.arange(num_workers)
        width = 0.25
        
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        
        for j, (metric, color) in enumerate(zip(metrics_names, colors)):
            offset = (j - 1) * width
            bars = ax.bar(x + offset, data_norm[:, j], width, label=metric, color=color, alpha=0.8)
            
            # Add value labels
            for i, bar in enumerate(bars):
                if j == 0:  # Mean Reward
                    label = f'{data[i, j]:.2f}'
                elif j == 1:  # Win Rate
                    label = f'{data[i, j]*100:.0f}%'
                else:  # Episodes
                    label = f'{int(data[i, j])}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       label, ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Worker')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Worker Performance Comparison (Normalized)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Worker {wid}' for wid in worker_ids])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'pbt_worker_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def update_dashboard(metrics_file: str = 'pbt_metrics.csv', 
                     output_dir: str = 'dqn_models'):
    """Convenience function to update the PBT dashboard."""
    dashboard = PBTDashboard(metrics_file=metrics_file, output_dir=output_dir)
    dashboard.update(force=True)


if __name__ == '__main__':
    # Test with existing metrics file
    update_dashboard()
