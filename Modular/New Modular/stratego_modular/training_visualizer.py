# stratego_modular/training_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict

def plot_training_progress(
    episode_history: List[int],
    rewards_history: Dict[str, List[float]],
    wins_history: Dict[str, List[int]],
    epsilon_history: Dict[str, List[float]],
    save_path: str
):
    """
    Plots and saves the training progress of DQN agents.

    Args:
        episode_history: List of episode numbers.
        rewards_history: Dict containing lists of average rewards for each agent.
        wins_history: Dict containing lists of win counts for each agent and draws.
        epsilon_history: Dict containing lists of epsilon values for each agent.
        save_path: Path to save the plot image.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('DQN Agent Training Progress', fontsize=16)

    # Plot 1: Average Rewards
    axs[0].plot(episode_history, rewards_history['agent1'], label='Agent 1 Avg Reward', color='blue')
    axs[0].plot(episode_history, rewards_history['agent2'], label='Agent 2 Avg Reward', color='red')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Average Reward (over interval)')
    axs[0].set_title('Average Rewards per Episode')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Win/Draw Counts
    axs[1].plot(episode_history, wins_history['agent1'], label='Agent 1 Wins', color='blue')
    axs[1].plot(episode_history, wins_history['agent2'], label='Agent 2 Wins', color='red')
    axs[1].plot(episode_history, wins_history['draws'], label='Draws', color='green')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Count (per interval)')
    axs[1].set_title('Wins and Draws per Interval')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Epsilon Decay
    axs[2].plot(episode_history, epsilon_history['agent1'], label='Agent 1 Epsilon', color='blue', linestyle='--')
    axs[2].plot(episode_history, epsilon_history['agent2'], label='Agent 2 Epsilon', color='red', linestyle='--')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Epsilon Value')
    axs[2].set_title('Epsilon Decay')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"📈 Training progress graph saved to {save_path}")
