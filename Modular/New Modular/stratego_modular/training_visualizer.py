# stratego_modular/training_visualizer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image
import glob
from .piece import PieceType, PIECE_NAMES, PIECE_RANKS
from .board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE

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


def visualize_pbs_state(
    actual_board: torch.Tensor,
    agent1_pbs,
    agent2_pbs,
    episode: int,
    save_path: str
):
    """
    Visualize the Probabilistic Belief State (PBS) for both agents.
    
    Shows:
    - Actual board with pieces
    - Agent 1's PBS beliefs (inferred values and confidence)
    - Agent 2's PBS beliefs (inferred values and confidence)
    
    Args:
        actual_board: The actual game board (10x10 tensor)
        agent1_pbs: Agent 1's PBS object (or None if not available)
        agent2_pbs: Agent 2's PBS object (or None if not available)
        episode: Current episode number
        save_path: Path to save the visualization
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'PBS Visualization - Episode {episode}', fontsize=16, fontweight='bold')
    
    # Convert board to numpy
    board_np = actual_board.cpu().numpy() if isinstance(actual_board, torch.Tensor) else actual_board
    
    # Create subplots: Actual board, Agent 1 PBS, Agent 2 PBS
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot 1: Actual Board
    _plot_actual_board(ax1, board_np, "Actual Board State")
    
    # Plot 2: Agent 1 PBS
    if agent1_pbs:
        _plot_pbs_beliefs(ax2, board_np, agent1_pbs, player_id=1, title="Agent 1 PBS Beliefs")
    else:
        ax2.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Agent 1 PBS Beliefs")
    
    # Plot 3: Agent 2 PBS
    if agent2_pbs:
        _plot_pbs_beliefs(ax3, board_np, agent2_pbs, player_id=-1, title="Agent 2 PBS Beliefs")
    else:
        ax3.text(0.5, 0.5, 'PBS Not Available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title("Agent 2 PBS Beliefs")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🎯 PBS visualization saved to {save_path}")


def _plot_actual_board(ax, board_np: np.ndarray, title: str):
    """Plot the actual board with pieces."""
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.grid(True, color='gray', linewidth=0.5)
    
    # Draw board squares
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece_val = board_np[r, c]
            
            # Determine square color
            if piece_val == LAKE_SQUARE:
                color = 'lightblue'
            elif piece_val == EMPTY_SQUARE:
                color = 'white'
            elif piece_val > 0:
                color = 'lightcoral'  # Player 1
            else:
                color = 'lightgreen'  # Player 2
            
            # Draw square
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Draw piece label
            if piece_val != EMPTY_SQUARE and piece_val != LAKE_SQUARE:
                piece_type = PieceType(abs(int(piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                text_color = 'black' if abs(piece_val) <= 6 else 'white'
                ax.text(c, r, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=text_color)


def _plot_pbs_beliefs(ax, board_np: np.ndarray, pbs, player_id: int, title: str):
    """Plot PBS beliefs for a specific agent."""
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.grid(True, color='gray', linewidth=0.5)
    
    # Draw board squares with PBS information
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pos = (r, c)
            piece_val = board_np[r, c]
            
            # Determine square color
            if piece_val == LAKE_SQUARE:
                color = 'lightblue'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                continue
            elif piece_val == EMPTY_SQUARE:
                color = 'white'
            elif (player_id == 1 and piece_val > 0) or (player_id == -1 and piece_val < 0):
                # Own piece - show actual
                color = 'lightcoral' if player_id == 1 else 'lightgreen'
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                # Show actual piece
                piece_type = PieceType(abs(int(piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                ax.text(c, r, piece_name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
                continue
            else:
                # Enemy piece - show PBS beliefs
                color = 'lightyellow'  # Highlight unknown pieces
                # Get PBS beliefs
                if pos in pbs.belief_distributions:
                    beliefs = pbs.belief_distributions[pos]
                    # Find most likely piece type
                    most_likely = max(beliefs.items(), key=lambda x: x[1])
                    piece_type, confidence = most_likely
                    expected_val = pbs.get_expected_value(pos)
                    
                    # Color intensity based on confidence
                    # Use a color map: green (high confidence) to yellow (low confidence)
                    # plt.cm.RdYlGn returns RGBA tuple, convert to hex or use directly
                    rgba = plt.cm.RdYlGn(confidence)  # Green (high conf) to Red (low conf)
                    color = rgba[:3]  # Use RGB only (matplotlib patches can use this)
                else:
                    # No PBS data for this position
                    piece_type = None
                    confidence = 0.0
                    expected_val = 0.0
                    color = 'lightgray'
            
            # Draw square
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Draw PBS information
            if piece_val != EMPTY_SQUARE and piece_val != LAKE_SQUARE:
                if pos in pbs.belief_distributions and pos not in pbs.revealed_pieces:
                    # Show inferred piece and confidence
                    beliefs = pbs.belief_distributions[pos]
                    most_likely = max(beliefs.items(), key=lambda x: x[1])
                    piece_type, confidence = most_likely
                    piece_name = PIECE_NAMES.get(piece_type, '?')
                    
                    # Top: Inferred piece type
                    ax.text(c, r - 0.25, piece_name, ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='black')
                    # Bottom: Confidence score
                    ax.text(c, r + 0.25, f'{confidence:.2f}', ha='center', va='center', 
                           fontsize=8, color='darkblue', fontweight='bold')
                elif piece_val != EMPTY_SQUARE:
                    # Known piece (revealed)
                    piece_type = PieceType(abs(int(piece_val)))
                    piece_name = PIECE_NAMES.get(piece_type, '?')
                    ax.text(c, r, piece_name, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')


def create_training_gif(model_save_path: str, episode: int, gif_duration: int = 500):
    """
    Create a GIF from training progress and PBS visualization images.
    
    Args:
        model_save_path: Path to the model save directory
        episode: Current episode number (should be a multiple of 50)
        gif_duration: Duration of each frame in milliseconds
    """
    try:
        # Find all training progress images up to this episode
        progress_pattern = f"{model_save_path}/training_progress_episode_*.png"
        pbs_pattern = f"{model_save_path}/pbs_visualization_episode_*.png"
        
        progress_images = sorted(glob.glob(progress_pattern), 
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))
        pbs_images = sorted(glob.glob(pbs_pattern),
                           key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Filter to only include images up to current episode
        progress_images = [img for img in progress_images 
                          if int(img.split('_')[-1].split('.')[0]) <= episode]
        pbs_images = [img for img in pbs_images 
                     if int(img.split('_')[-1].split('.')[0]) <= episode]
        
        if not progress_images and not pbs_images:
            print(f"⚠️  No images found for GIF creation at episode {episode}")
            return
        
        # Combine and sort by episode number
        all_images = []
        image_dict = {}
        
        for img_path in progress_images:
            ep_num = int(img_path.split('_')[-1].split('.')[0])
            if ep_num not in image_dict:
                image_dict[ep_num] = {}
            image_dict[ep_num]['progress'] = img_path
        
        for img_path in pbs_images:
            ep_num = int(img_path.split('_')[-1].split('.')[0])
            if ep_num not in image_dict:
                image_dict[ep_num] = {}
            image_dict[ep_num]['pbs'] = img_path
        
        # Create frames: combine progress and PBS side by side for each episode
        frames = []
        for ep_num in sorted(image_dict.keys()):
            if 'progress' in image_dict[ep_num] and 'pbs' in image_dict[ep_num]:
                # Combine progress and PBS images side by side
                progress_img = Image.open(image_dict[ep_num]['progress'])
                pbs_img = Image.open(image_dict[ep_num]['pbs'])
                
                # Resize images to have same height
                max_height = max(progress_img.height, pbs_img.height)
                progress_img = progress_img.resize(
                    (int(progress_img.width * max_height / progress_img.height), max_height),
                    Image.Resampling.LANCZOS
                )
                pbs_img = pbs_img.resize(
                    (int(pbs_img.width * max_height / pbs_img.height), max_height),
                    Image.Resampling.LANCZOS
                )
                
                # Create combined image
                combined_width = progress_img.width + pbs_img.width
                combined_img = Image.new('RGB', (combined_width, max_height))
                combined_img.paste(progress_img, (0, 0))
                combined_img.paste(pbs_img, (progress_img.width, 0))
                
                frames.append(combined_img)
            elif 'progress' in image_dict[ep_num]:
                # Only progress image
                frames.append(Image.open(image_dict[ep_num]['progress']))
            elif 'pbs' in image_dict[ep_num]:
                # Only PBS image
                frames.append(Image.open(image_dict[ep_num]['pbs']))
        
        if not frames:
            print(f"⚠️  No frames to create GIF at episode {episode}")
            return
        
        # Save GIF
        gif_path = f"{model_save_path}/training_animation_episode_{episode}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration,
            loop=0,
            optimize=True
        )
        
        print(f"🎬 Training GIF saved to {gif_path} ({len(frames)} frames)")
        
    except Exception as e:
        print(f"⚠️  Error creating training GIF at episode {episode}: {e}")
        import traceback
        traceback.print_exc()


def visualize_game_state(board: torch.Tensor, move_num: int = 0, episode: int = 0, 
                        last_move: Optional[Tuple] = None) -> Image.Image:
    """
    Create a clean visualization of a single game state.
    
    Args:
        board: The game board tensor (10x10)
        move_num: Current move number
        episode: Episode number
        last_move: Last move made ((from_pos), (to_pos)) or None
        
    Returns:
        PIL Image of the board visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('white')
    
    # Convert board to numpy
    board_np = board.cpu().numpy() if isinstance(board, torch.Tensor) else board
    
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Episode {episode} - Move {move_num}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_xlabel('Column', fontsize=14, fontweight='bold')
    ax.set_ylabel('Row', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    
    # Highlight last move if provided
    if last_move:
        (from_r, from_c), (to_r, to_c) = last_move
        # Highlight source square
        rect_from = patches.Rectangle((from_c - 0.5, from_r - 0.5), 1, 1, 
                                     facecolor='yellow', edgecolor='orange', 
                                     linewidth=3, alpha=0.5)
        ax.add_patch(rect_from)
        # Highlight destination square
        rect_to = patches.Rectangle((to_c - 0.5, to_r - 0.5), 1, 1, 
                                   facecolor='yellow', edgecolor='orange', 
                                   linewidth=3, alpha=0.5)
        ax.add_patch(rect_to)
        # Draw arrow
        arrow = patches.FancyArrowPatch((from_c, from_r), (to_c, to_r),
                                       arrowstyle='->', mutation_scale=20,
                                       color='orange', linewidth=2,
                                       zorder=10)
        ax.add_patch(arrow)
    
    # Draw board squares and pieces
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece_val = board_np[r, c]
            
            # Determine square color
            if piece_val == LAKE_SQUARE:
                color = 'lightblue'
            elif piece_val == EMPTY_SQUARE:
                color = 'white'
            elif piece_val > 0:
                color = 'lightcoral'  # Player 1
            else:
                color = 'lightgreen'  # Player 2
            
            # Draw square
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Draw piece label with improved readability
            if piece_val != EMPTY_SQUARE and piece_val != LAKE_SQUARE:
                piece_type = PieceType(abs(int(piece_val)))
                piece_name = PIECE_NAMES.get(piece_type, '?')
                piece_rank = abs(int(piece_val))
                
                # Determine text and background colors for better contrast
                if piece_type == PieceType.BOMB:
                    # Bomb: black text on orange background
                    text_color = 'black'
                    bg_color = 'orange'
                    edge_color = 'darkorange'
                elif piece_type == PieceType.FLAG:
                    # Flag: white text on red background
                    text_color = 'white'
                    bg_color = 'red'
                    edge_color = 'darkred'
                elif piece_rank <= 6:
                    # Lower rank pieces: black text on white/light background
                    text_color = 'black'
                    bg_color = 'white'
                    edge_color = 'black'
                else:
                    # Higher rank pieces: white text on dark background
                    text_color = 'white'
                    bg_color = 'darkblue'
                    edge_color = 'navy'
                
                # Show only piece name (not rank) for clarity
                display_text = piece_name
                
                ax.text(c, r, display_text, ha='center', va='center', 
                       fontsize=16, fontweight='bold', color=text_color,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, 
                                edgecolor=edge_color, alpha=0.9, linewidth=2.5))
    
    plt.tight_layout()
    
    # Convert to PIL Image using BytesIO (more compatible across backends)
    # Use higher DPI for better quality and readability
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Ensure RGB format
    
    plt.close(fig)
    return img


def create_episode_gif(game_states: List[Dict], episode: int, save_path: str, 
                      frame_duration: int = 1000):
    """
    Create a GIF from a sequence of game states recorded during an episode.
    
    Args:
        game_states: List of dicts with keys: 'board', 'move_num', 'last_move'
        episode: Episode number
        save_path: Path to save the GIF
        frame_duration: Duration of each frame in milliseconds (default 1000ms = 1 second)
    """
    try:
        if not game_states:
            print(f"⚠️  No game states to create GIF for episode {episode}")
            return
        
        frames = []
        for i, state in enumerate(game_states):
            board = state['board']
            move_num = state.get('move_num', i)
            last_move = state.get('last_move', None)
            
            # Create visualization for this state
            img = visualize_game_state(board, move_num, episode, last_move)
            frames.append(img)
        
        if not frames:
            print(f"⚠️  No frames created for episode {episode}")
            return
        
        # Save GIF
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )
        
        print(f"🎬 Episode GIF saved to {save_path} ({len(frames)} frames, {frame_duration}ms per frame)")
        
    except Exception as e:
        print(f"⚠️  Error creating episode GIF for episode {episode}: {e}")
        import traceback
        traceback.print_exc()
