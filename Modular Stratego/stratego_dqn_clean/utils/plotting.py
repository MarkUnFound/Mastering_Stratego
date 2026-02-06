
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from typing import List, Dict, Any

def plot_training_history(history_path: str, save_dir: str):
    """
    Plot training metrics from history file.
    
    Args:
        history_path: Path to training_history.json
        save_dir: Directory to save plots
    """
    if not os.path.exists(history_path):
        print(f"[Plotting] History file not found: {history_path}")
        return
        
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"[Plotting] Error loading history: {e}")
        return
        
    if not history:
        print("[Plotting] Empty history")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    steps = [entry['step'] for entry in history]
    rewards = [entry.get('avg_reward', 0) for entry in history]
    win_rates = [entry.get('win_rate', 0) for entry in history]
    losses = [entry.get('loss', 0) for entry in history]
    epsilons = [entry.get('epsilon', 0) for entry in history]
    
    # Plot 1: Win Rate & Epsilon
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Win Rate', color=color)
    ax1.plot(steps, win_rates, color=color, linewidth=2, label='Win Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(steps, epsilons, color=color, linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Progress: Win Rate & Exploration')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'win_rate_epsilon.png'))
    plt.close()
    
    # Plot 2: Average Reward
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, color='green', linewidth=2)
    plt.title('Average Reward per Episode')
    plt.xlabel('Steps')
    plt.ylabel('Avg Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_reward.png'))
    plt.close()
    
    # Plot 3: Loss (Log Scale)
    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(steps[-len(valid_losses):], valid_losses, color='red', alpha=0.7)
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.close()
        
    print(f"[Plotting] Training plots saved to {save_dir}")
