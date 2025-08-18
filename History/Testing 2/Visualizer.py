import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from typing import Dict, List, Optional

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Visualizer for training metrics and game analysis"""
    
    def __init__(self):
        self.figures = {}  # Store figures for saving later
        
    def plot_win_rates(self, data: Dict, window_size: int = 50, save_path: Optional[str] = None):
        """Plot win rates over episodes"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        episodes = data['episodes']
        
        ax.plot(episodes, data['player0_win_rate'], label='Agent 1 (Red)', linewidth=2, alpha=0.8)
        ax.plot(episodes, data['player1_win_rate'], label='Agent 2 (Blue)', linewidth=2, alpha=0.8)
        ax.plot(episodes, data['draw_rate'], label='Draws', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win Rate', fontsize=12)
        ax.set_title(f'Win Rates Over Training (Rolling Average, Window={window_size})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add horizontal line at 50%
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['win_rates'] = fig
        return fig
    
    def plot_rewards(self, data: Dict, save_path: Optional[str] = None):
        """Plot average rewards over episodes"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = data['episodes']
        
        # Average rewards
        ax1.plot(episodes, data['avg_rewards_p0'], label='Agent 1', linewidth=2, alpha=0.8)
        ax1.plot(episodes, data['avg_rewards_p1'], label='Agent 2', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Average Episode Rewards', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Reward difference
        reward_diff = np.array(data['avg_rewards_p0']) - np.array(data['avg_rewards_p1'])
        ax2.plot(episodes, reward_diff, color='purple', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Reward Difference (P1 - P2)', fontsize=12)
        ax2.set_title('Reward Advantage', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['rewards'] = fig
        return fig
    
    def plot_game_length(self, data: Dict, save_path: Optional[str] = None):
        """Plot game length statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        episodes = data['episodes']
        lengths = data['avg_game_length']
        
        # Time series
        ax1.plot(episodes, lengths, linewidth=2, alpha=0.8, color='green')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Game Length', fontsize=12)
        ax1.set_title('Game Length Over Training', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Distribution
        ax2.hist(lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Game Length', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Game Length Distribution', fontsize=14)
        ax2.axvline(np.mean(lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(lengths):.1f}')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['game_length'] = fig
        return fig
    
    def plot_training_loss(self, losses: List[float], agent_name: str = "Agent", 
                          save_path: Optional[str] = None):
        """Plot training loss"""
        if not losses:
            print(f"No loss data available for {agent_name}")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Smooth the losses
        window_size = min(100, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size + 1)
                smoothed_losses.append(np.mean(losses[start_idx:i+1]))
        else:
            smoothed_losses = losses
        
        ax.plot(losses, alpha=0.3, color='blue', label='Raw Loss')
        ax.plot(smoothed_losses, linewidth=2, color='red', 
               label=f'Smoothed Loss (window={window_size})')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{agent_name} Training Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures[f'loss_{agent_name.lower().replace(" ", "_")}'] = fig
        return fig
    
    def plot_epsilon_decay(self, epsilons: List[float], agent_name: str = "Agent",
                          save_path: Optional[str] = None):
        """Plot epsilon decay over training"""
        if not epsilons:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(epsilons, linewidth=2, color='orange')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Epsilon', fontsize=12)
        ax.set_title(f'{agent_name} Epsilon Decay', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures[f'epsilon_{agent_name.lower().replace(" ", "_")}'] = fig
        return fig
    
    def plot_performance_comparison(self, data1: Dict, data2: Dict, 
                                  labels: List[str] = ["Agent 1", "Agent 2"],
                                  save_path: Optional[str] = None):
        """Compare performance between two agents"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Win rates comparison
        ax1.plot(data1['episodes'], data1['player0_win_rate'], 
                label=f'{labels[0]} Win Rate', linewidth=2)
        ax1.plot(data2['episodes'], data2['player0_win_rate'], 
                label=f'{labels[1]} Win Rate', linewidth=2)
        ax1.set_ylabel('Win Rate', fontsize=11)
        ax1.set_title('Win Rate Comparison', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rewards comparison
        ax2.plot(data1['episodes'], data1['avg_rewards_p0'], 
                label=f'{labels[0]} Avg Reward', linewidth=2)
        ax2.plot(data2['episodes'], data2['avg_rewards_p0'], 
                label=f'{labels[1]} Avg Reward', linewidth=2)
        ax2.set_ylabel('Average Reward', fontsize=11)
        ax2.set_title('Average Reward Comparison', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Game length comparison
        ax3.plot(data1['episodes'], data1['avg_game_length'], 
                label=f'{labels[0]} Game Length', linewidth=2)
        ax3.plot(data2['episodes'], data2['avg_game_length'], 
                label=f'{labels[1]} Game Length', linewidth=2)
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_ylabel('Game Length', fontsize=11)
        ax3.set_title('Game Length Comparison', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance summary (box plots)
        final_win_rates = [
            data1['player0_win_rate'][-100:],  # Last 100 episodes
            data2['player0_win_rate'][-100:]
        ]
        ax4.boxplot(final_win_rates, labels=labels)
        ax4.set_xlabel('Agent', fontsize=11)
        ax4.set_ylabel('Win Rate', fontsize=11)
        ax4.set_title('Final Performance (Last 100 Episodes)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['comparison'] = fig
        return fig
    
    def plot_heatmap(self, win_matrix: np.ndarray, agent_names: List[str],
                    save_path: Optional[str] = None):
        """Plot win rate heatmap between different agents"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(win_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=agent_names, yticklabels=agent_names,
                   ax=ax, cbar_kws={'label': 'Win Rate'})
        
        ax.set_title('Agent vs Agent Win Rate Matrix', fontsize=14)
        ax.set_xlabel('Opponent', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['heatmap'] = fig
        return fig
    
    def create_summary_dashboard(self, data: Dict, agent_stats: List[Dict],
                               save_path: Optional[str] = None):
        """Create comprehensive training summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Win rates
        ax1 = fig.add_subplot(gs[0, :2])
        episodes = data['episodes']
        ax1.plot(episodes, data['player0_win_rate'], label='Agent 1', linewidth=2)
        ax1.plot(episodes, data['player1_win_rate'], label='Agent 2', linewidth=2)
        ax1.plot(episodes, data['draw_rate'], label='Draws', linewidth=2, alpha=0.7)
        ax1.set_title('Win Rates Over Training', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rewards
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(episodes, data['avg_rewards_p0'], label='Agent 1', linewidth=2)
        ax2.plot(episodes, data['avg_rewards_p1'], label='Agent 2', linewidth=2)
        ax2.set_title('Average Rewards', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Game length
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(episodes, data['avg_game_length'], linewidth=2, color='green')
        ax3.set_title('Game Length', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Moves')
        ax3.grid(True, alpha=0.3)
        
        # Final performance distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        final_episodes = 100
        recent_p0 = data['player0_win_rate'][-final_episodes:]
        recent_p1 = data['player1_win_rate'][-final_episodes:]
        ax4.hist([recent_p0, recent_p1], bins=15, alpha=0.7, 
                label=['Agent 1', 'Agent 2'], density=True)
        ax4.set_title(f'Win Rate Distribution (Last {final_episodes} Episodes)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Win Rate')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Training statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create statistics table
        stats_data = []
        for i, stats in enumerate(agent_stats):
            stats_data.append([
                f"Agent {i+1}",
                f"{stats.get('epsilon', 'N/A'):.3f}",
                f"{stats.get('avg_loss', 0):.4f}",
                f"{stats.get('memory_size', 0):,}",
                f"{stats.get('total_episodes', 0):,}",
                f"{np.mean(data[f'avg_rewards_p{i}'][-100:]):.2f}"
            ])
        
        table = ax5.table(cellText=stats_data,
                         colLabels=['Agent', 'Epsilon', 'Avg Loss', 'Memory Size', 
                                   'Episodes', 'Recent Avg Reward'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title('Training Statistics', fontsize=12, fontweight='bold', pad=20)
        
        # Main title
        fig.suptitle('Stratego DQN Training Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures['dashboard'] = fig
        return fig
    
    def save_all_plots(self, output_dir: str):
        """Save all generated plots to directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        # Also save as PDF for better quality
        pdf_dir = os.path.join(output_dir, 'pdf')
        os.makedirs(pdf_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = os.path.join(pdf_dir, f"{name}.pdf")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    def show_all(self):
        """Show all generated plots"""
        plt.show()
    
    def close_all(self):
        """Close all figures"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()