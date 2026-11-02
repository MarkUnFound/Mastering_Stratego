"""
Input Channel Visualization Script

Visualizes the 30 observation planes the Rainbow DQN agent receives (environment/observation.py).
Helps diagnose whether the agent can "see" important features like the enemy flag.

Planes 0-11:  Own piece types (FLAG, SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL, BOMB)
Planes 12-23: Enemy pieces of known type, same order
Plane 24:     Enemy pieces of unknown type
Plane 25:     Lake squares (obstacles)
Plane 26:     Own pieces the enemy knows
Planes 27-29: Observability flag, no-battle counter, game progress
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from piece import PieceType

# Channel names for visualization
CHANNEL_NAMES = [
    # Own pieces (0-11)
    "Own FLAG", "Own SPY", "Own SCOUT", "Own MINER", "Own SERGEANT", "Own LIEUTENANT",
    "Own CAPTAIN", "Own MAJOR", "Own COLONEL", "Own GENERAL", "Own MARSHAL", "Own BOMB",
    # Enemy pieces of known type (12-23)
    "Enemy FLAG", "Enemy SPY", "Enemy SCOUT", "Enemy MINER", "Enemy SERGEANT", "Enemy LIEUTENANT",
    "Enemy CAPTAIN", "Enemy MAJOR", "Enemy COLONEL", "Enemy GENERAL", "Enemy MARSHAL", "Enemy BOMB",
    # Other planes (24-29)
    "Enemy unknown", "Lake", "Own known to enemy", "Observability", "No-battle counter", "Game progress",
]
ENEMY_FLAG_PLANE = 12

def visualize_input_channels(save_path: str = "dqn_models/input_visualization.png"):
    """
    Create a visual representation of the 30 observation planes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment and get a game state
    print("Creating environment...")
    env = StrategoEnvironment(device=device, full_observability=True)
    game_state = env.reset()
    
    # Create agent to get state representation
    print("Creating agent...")
    agent = RainbowAgent(player_id=1, device=device, num_envs=1, use_pbs=False)
    
    # Get the observation planes (with full observability)
    print("Generating input tensor...")
    input_tensor = agent.get_state_representation(game_state, full_observability=True)
    
    print(f"Input tensor shape: {input_tensor.shape}")  # (30, 10, 10)
    
    # Convert to numpy for visualization
    input_np = input_tensor.cpu().numpy()
    
    # One subplot per plane (6 rows x 5 cols)
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    fig.suptitle(f"Agent Input Planes ({len(CHANNEL_NAMES)} total)", fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i in range(len(CHANNEL_NAMES)):
        ax = axes_flat[i]
        channel = input_np[i]
        
        # Highlight important channels
        if i == 0:  # Own FLAG
            cmap = 'Greens'
            title_color = 'green'
        elif i == ENEMY_FLAG_PLANE:  # Enemy FLAG (TARGET!)
            cmap = 'Reds'
            title_color = 'red'
        elif i == 24:  # Enemy of unknown type
            cmap = 'Oranges'
            title_color = 'orange'
        elif i == 25:  # Lake
            cmap = 'Blues'
            title_color = 'blue'
        else:
            cmap = 'viridis'
            title_color = 'black'
        
        im = ax.imshow(channel, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Ch {i}: {CHANNEL_NAMES[i]}", fontsize=9, color=title_color, fontweight='bold' if i in [0, ENEMY_FLAG_PLANE] else 'normal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Show values on cells where they are 1
        for r in range(10):
            for c in range(10):
                if channel[r, c] > 0:
                    ax.text(c, r, f"{channel[r, c]:.1f}", ha='center', va='center', 
                           fontsize=6, color='white' if channel[r, c] > 0.5 else 'black')
    
    # Hide extra subplots
    for i in range(len(CHANNEL_NAMES), len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='Own FLAG (Ch 0)'),
        mpatches.Patch(color='red', label=f'Enemy FLAG (Ch {ENEMY_FLAG_PLANE}) - TARGET!'),
        mpatches.Patch(color='orange', label='Enemy unknown (Ch 24)'),
        mpatches.Patch(color='blue', label='Lake (Ch 25)'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved to: {save_path}")
    
    # Also print key observations
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    
    # Check if enemy flag is visible
    enemy_flag_channel = input_np[ENEMY_FLAG_PLANE]
    flag_positions = np.argwhere(enemy_flag_channel > 0)
    if len(flag_positions) > 0:
        print(f" Enemy FLAG visible at: {flag_positions.tolist()}")
    else:
        print(f" Enemy FLAG NOT visible in channel {ENEMY_FLAG_PLANE}!")
    
    # Check own flag
    own_flag_channel = input_np[0]
    own_flag_positions = np.argwhere(own_flag_channel > 0)
    if len(own_flag_positions) > 0:
        print(f" Own FLAG at: {own_flag_positions.tolist()}")
    
    # Check enemy mask
    enemy_mask = input_np[12]
    enemy_count = np.sum(enemy_mask > 0)
    print(f" Enemy pieces visible: {int(enemy_count)}")
    
    # Check lake
    lake_channel = input_np[13]
    lake_count = np.sum(lake_channel > 0)
    print(f" Lake squares: {int(lake_count)}")
    
    # Check empty
    empty_channel = input_np[14]
    empty_count = np.sum(empty_channel > 0)
    print(f" Empty squares: {int(empty_count)}")
    
    print("="*60)
    
    plt.close()
    return input_tensor

if __name__ == "__main__":
    visualize_input_channels()
