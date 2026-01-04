"""
Input Channel Visualization Script

Visualizes the 27-channel input tensor that the Rainbow DQN agent receives.
Helps diagnose whether the agent can "see" important features like the enemy flag.

Channels 0-11:  Own piece types (FLAG, SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL, BOMB)
Channel 12:     All enemy pieces (binary mask)
Channel 13:     Lake squares (obstacles)
Channel 14:     Empty squares
Channels 15-26: Enemy piece type beliefs (PBS or ground truth)
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
    # Board features (12-14)
    "Enemy Mask", "Lake", "Empty",
    # Enemy piece beliefs (15-26)
    "Enemy FLAG", "Enemy SPY", "Enemy SCOUT", "Enemy MINER", "Enemy SERGEANT", "Enemy LIEUTENANT",
    "Enemy CAPTAIN", "Enemy MAJOR", "Enemy COLONEL", "Enemy GENERAL", "Enemy MARSHAL", "Enemy BOMB"
]

def visualize_input_channels(save_path: str = "dqn_models/input_visualization.png"):
    """
    Create a visual representation of the 27-channel input tensor.
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
    
    # Get the 27-channel input tensor (with full observability)
    print("Generating input tensor...")
    input_tensor = agent.get_state_representation(game_state, full_observability=True)
    
    print(f"Input tensor shape: {input_tensor.shape}")  # Should be (27, 10, 10)
    
    # Convert to numpy for visualization
    input_np = input_tensor.cpu().numpy()
    
    # Create figure with 27 subplots (6 rows x 5 cols, with 3 extra)
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    fig.suptitle("Agent Input Channels (27 channels total)", fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i in range(27):
        ax = axes_flat[i]
        channel = input_np[i]
        
        # Highlight important channels
        if i == 0:  # Own FLAG
            cmap = 'Greens'
            title_color = 'green'
        elif i == 15:  # Enemy FLAG (TARGET!)
            cmap = 'Reds'
            title_color = 'red'
        elif i == 12:  # Enemy mask
            cmap = 'Oranges'
            title_color = 'orange'
        elif i == 13:  # Lake
            cmap = 'Blues'
            title_color = 'blue'
        elif i == 14:  # Empty
            cmap = 'Greys'
            title_color = 'gray'
        else:
            cmap = 'viridis'
            title_color = 'black'
        
        im = ax.imshow(channel, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Ch {i}: {CHANNEL_NAMES[i]}", fontsize=9, color=title_color, fontweight='bold' if i in [0, 15] else 'normal')
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
    for i in range(27, 30):
        axes_flat[i].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='Own FLAG (Ch 0)'),
        mpatches.Patch(color='red', label='Enemy FLAG (Ch 15) - TARGET!'),
        mpatches.Patch(color='orange', label='Enemy Mask (Ch 12)'),
        mpatches.Patch(color='blue', label='Lake (Ch 13)'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # Also print key observations
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    
    # Check if enemy flag is visible
    enemy_flag_channel = input_np[15]
    flag_positions = np.argwhere(enemy_flag_channel > 0)
    if len(flag_positions) > 0:
        print(f"✅ Enemy FLAG visible at: {flag_positions.tolist()}")
    else:
        print("❌ Enemy FLAG NOT visible in channel 15!")
    
    # Check own flag
    own_flag_channel = input_np[0]
    own_flag_positions = np.argwhere(own_flag_channel > 0)
    if len(own_flag_positions) > 0:
        print(f"✅ Own FLAG at: {own_flag_positions.tolist()}")
    
    # Check enemy mask
    enemy_mask = input_np[12]
    enemy_count = np.sum(enemy_mask > 0)
    print(f"✅ Enemy pieces visible: {int(enemy_count)}")
    
    # Check lake
    lake_channel = input_np[13]
    lake_count = np.sum(lake_channel > 0)
    print(f"✅ Lake squares: {int(lake_count)}")
    
    # Check empty
    empty_channel = input_np[14]
    empty_count = np.sum(empty_channel > 0)
    print(f"✅ Empty squares: {int(empty_count)}")
    
    print("="*60)
    
    plt.close()
    return input_tensor

if __name__ == "__main__":
    visualize_input_channels()
