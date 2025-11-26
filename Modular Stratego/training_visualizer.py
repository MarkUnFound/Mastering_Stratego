# stratego_modular/training_visualizer.py

# Set matplotlib backend to non-interactive before importing pyplot
# This prevents Tkinter errors when used in multi-threaded environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image
import glob
from piece import PieceType, PIECE_NAMES, PIECE_RANKS
from board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE, HIDDEN_PIECE

def plot_training_progress(
    episode_history: List[int],
    rewards_history: Dict[str, List[float]],
    wins_history: Dict[str, List[int]],
    policy_loss_history: Dict[str, List[float]],
    save_path: str,
    total_episodes: Optional[int] = None,
    total_steps: Optional[int] = None
):
    """
    Plots and saves the training progress of DQN agents.

    Args:
        episode_history: List of episode numbers.
        rewards_history: Dict containing lists of average rewards for each agent.
        wins_history: Dict containing lists of win counts for each agent (draws removed).
        policy_loss_history: Dict containing lists of policy loss values for each agent.
        save_path: Path to save the plot image.
        total_episodes: Total episodes across all training runs (for display).
        total_steps: Total steps across all training runs (for display).
    """
    # Validate input data
    if not episode_history or len(episode_history) == 0:
        raise ValueError("episode_history is empty - cannot plot training progress")
    
    if not rewards_history or not rewards_history.get('agent1') or not rewards_history.get('agent2'):
        raise ValueError("rewards_history is empty or missing agent data")
    
    if len(episode_history) != len(rewards_history['agent1']) or len(episode_history) != len(rewards_history['agent2']):
        raise ValueError(f"Length mismatch: episode_history={len(episode_history)}, "
                        f"rewards_history agent1={len(rewards_history['agent1'])}, "
                        f"rewards_history agent2={len(rewards_history['agent2'])}")
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Create title with total episodes and steps if provided
    title = 'DQN Agent Training Progress'
    if total_episodes is not None or total_steps is not None:
        title_parts = [title]
        if total_episodes is not None:
            title_parts.append(f'Total Episodes: {total_episodes:,}')
        if total_steps is not None:
            title_parts.append(f'Total Steps: {total_steps:,}')
        title = ' | '.join(title_parts)
    
    fig.suptitle(title, fontsize=16)

    # Plot 1: Average Rewards (with discrete points and cumulative average line)
    if len(episode_history) > 0:
        # Plot discrete points for each episode
        axs[0].scatter(episode_history, rewards_history['agent1'], label='Agent 1 Reward', 
                      color='blue', marker='o', s=30, alpha=0.5, zorder=3)
        axs[0].scatter(episode_history, rewards_history['agent2'], label='Agent 2 Reward', 
                      color='red', marker='o', s=30, alpha=0.5, zorder=3)
        
        # Calculate and plot cumulative average from the start
        if len(episode_history) >= 1:
            # Cumulative average: average of all episodes from episode 1 to current
            agent1_cumulative_avg = np.cumsum(rewards_history['agent1']) / np.arange(1, len(rewards_history['agent1']) + 1)
            agent2_cumulative_avg = np.cumsum(rewards_history['agent2']) / np.arange(1, len(rewards_history['agent2']) + 1)
            
            axs[0].plot(episode_history, agent1_cumulative_avg, color='blue', linestyle='-', linewidth=2, 
                       label='Agent 1 Cumulative Avg', alpha=0.8, zorder=2)
            axs[0].plot(episode_history, agent2_cumulative_avg, color='red', linestyle='-', linewidth=2, 
                       label='Agent 2 Cumulative Avg', alpha=0.8, zorder=2)
    
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Rewards per Episode (with Cumulative Average)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Win Counts (cumulative wins with win rate)
    if len(episode_history) > 0:
        # Plot discrete points for cumulative wins
        scatter1 = axs[1].scatter(episode_history, wins_history['agent1'], label='Agent 1 Wins', 
                      color='blue', marker='o', s=30, alpha=0.5, zorder=3)
        scatter2 = axs[1].scatter(episode_history, wins_history['agent2'], label='Agent 2 Wins', 
                      color='red', marker='o', s=30, alpha=0.5, zorder=3)
        
        # Calculate and plot cumulative average win rate from the start (on secondary y-axis)
        if len(episode_history) >= 1:
            # Cumulative win rate: wins / total episodes
            episode_nums = np.arange(1, len(wins_history['agent1']) + 1)
            agent1_win_rate = np.array(wins_history['agent1'], dtype=float) / episode_nums
            agent2_win_rate = np.array(wins_history['agent2'], dtype=float) / episode_nums
            
            # Create secondary y-axis for win rate
            axs1_twin = axs[1].twinx()
            line1, = axs1_twin.plot(episode_history, agent1_win_rate, color='blue', linestyle='--', linewidth=2, 
                       label='Agent 1 Win Rate', alpha=0.8, zorder=2)
            line2, = axs1_twin.plot(episode_history, agent2_win_rate, color='red', linestyle='--', linewidth=2, 
                       label='Agent 2 Win Rate', alpha=0.8, zorder=2)
            axs1_twin.set_ylabel('Win Rate (0-1)', color='gray')
            axs1_twin.tick_params(axis='y', labelcolor='gray')
            axs1_twin.set_ylim(0, 1)
            
            # Create combined legend manually
            legend_elements = [
                scatter1, scatter2, line1, line2
            ]
            axs[1].legend(legend_elements, ['Agent 1 Wins', 'Agent 2 Wins', 'Agent 1 Win Rate', 'Agent 2 Win Rate'], 
                         loc='upper left')
    
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Cumulative Win Count', color='black')
    axs[1].tick_params(axis='y', labelcolor='black')
    axs[1].set_title('Cumulative Wins (with Win Rate)')
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Policy Loss (with cumulative average)
    if len(episode_history) > 0 and len(policy_loss_history.get('agent1', [])) > 0:
        # Plot discrete points for each episode
        axs[2].scatter(episode_history, policy_loss_history['agent1'], label='Agent 1 Policy Loss', 
                      color='blue', marker='o', s=30, alpha=0.5, zorder=3)
        axs[2].scatter(episode_history, policy_loss_history['agent2'], label='Agent 2 Policy Loss', 
                      color='red', marker='o', s=30, alpha=0.5, zorder=3)
        
        # Calculate and plot cumulative average from the start
        if len(episode_history) >= 1:
            # Cumulative average: average of all policy losses from episode 1 to current
            agent1_loss_avg = np.cumsum(policy_loss_history['agent1']) / np.arange(1, len(policy_loss_history['agent1']) + 1)
            agent2_loss_avg = np.cumsum(policy_loss_history['agent2']) / np.arange(1, len(policy_loss_history['agent2']) + 1)
            
            axs[2].plot(episode_history, agent1_loss_avg, color='blue', linestyle='-', linewidth=2, 
                       label='Agent 1 Cumulative Avg', alpha=0.8, zorder=2)
            axs[2].plot(episode_history, agent2_loss_avg, color='red', linestyle='-', linewidth=2, 
                       label='Agent 2 Cumulative Avg', alpha=0.8, zorder=2)
    
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Policy Loss')
    axs[2].set_title('Policy Loss per Episode (with Cumulative Average)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if path contains a directory component
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure with error handling
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # Verify file was created
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Failed to create file: {save_path}")
    except Exception as e:
        plt.close(fig)
        raise Exception(f"Error saving plot to {save_path}: {e}")
    
    plt.close(fig)


# visualize_pbs_state, _plot_actual_board, and _plot_pbs_beliefs have been moved to pbs_visualizer.py
# This file now only contains training progress visualization functions


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


def plot_pbs_evaluator_progress(
    episode_history: List[int],
    evaluator1_losses: List[float],
    evaluator2_losses: List[float],
    evaluator1_buffer_sizes: List[int],
    evaluator2_buffer_sizes: List[int],
    save_path: str,
    total_episodes: Optional[int] = None
):
    """
    Plot PBS evaluator improvement metrics.
    
    Args:
        episode_history: List of episode numbers
        evaluator1_losses: List of training losses for evaluator 1
        evaluator2_losses: List of training losses for evaluator 2
        evaluator1_buffer_sizes: List of experience buffer sizes for evaluator 1
        evaluator2_buffer_sizes: List of experience buffer sizes for evaluator 2
        save_path: Path to save the plot
        total_episodes: Total episodes across all training runs (for display)
    """
    # Validate input data
    if not episode_history:
        return  # Skip if no data
    
    # Ensure all lists have the same length
    min_len = min(len(episode_history), len(evaluator1_losses), len(evaluator2_losses),
                  len(evaluator1_buffer_sizes), len(evaluator2_buffer_sizes))
    if min_len == 0:
        return  # Skip if no data
    
    episode_history = episode_history[:min_len]
    evaluator1_losses = evaluator1_losses[:min_len]
    evaluator2_losses = evaluator2_losses[:min_len]
    evaluator1_buffer_sizes = evaluator1_buffer_sizes[:min_len]
    evaluator2_buffer_sizes = evaluator2_buffer_sizes[:min_len]
    
    # Create figure with subplots (2 rows, 1 column for cleaner vertical layout)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.patch.set_facecolor('white')
    
    # Title
    title = 'PBS Evaluator Metrics'
    if total_episodes is not None:
        title += f' | Total Episodes: {total_episodes:,}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Training Loss (Raw + Moving Average)
    ax1 = axes[0]
    
    # Filter out None values
    valid_episodes_1 = [ep for ep, loss in zip(episode_history, evaluator1_losses) if loss is not None]
    valid_losses_1 = [loss for loss in evaluator1_losses if loss is not None]
    valid_episodes_2 = [ep for ep, loss in zip(episode_history, evaluator2_losses) if loss is not None]
    valid_losses_2 = [loss for loss in evaluator2_losses if loss is not None]
    
    has_data = False
    
    # Plot Agent 1
    if valid_episodes_1:
        # Raw loss (faint)
        ax1.plot(valid_episodes_1, valid_losses_1, 'b-', alpha=0.2, linewidth=1, label='Evaluator 1 (Raw)')
        
        # Moving Average (bold)
        window = min(50, len(valid_episodes_1) // 2) if valid_episodes_1 else 10
        if window > 1:
            ma_losses_1 = []
            ma_episodes_1 = []
            for i in range(window, len(valid_losses_1)):
                ma_losses_1.append(np.mean(valid_losses_1[i-window:i]))
                ma_episodes_1.append(valid_episodes_1[i])
            if ma_episodes_1:
                ax1.plot(ma_episodes_1, ma_losses_1, 'b-', linewidth=2.5, label='Evaluator 1 (Moving Avg)')
        has_data = True

    # Plot Agent 2
    if valid_episodes_2:
        # Raw loss (faint)
        ax1.plot(valid_episodes_2, valid_losses_2, 'r-', alpha=0.2, linewidth=1, label='Evaluator 2 (Raw)')
        
        # Moving Average (bold)
        window = min(50, len(valid_episodes_2) // 2) if valid_episodes_2 else 10
        if window > 1:
            ma_losses_2 = []
            ma_episodes_2 = []
            for i in range(window, len(valid_losses_2)):
                ma_losses_2.append(np.mean(valid_losses_2[i-window:i]))
                ma_episodes_2.append(valid_episodes_2[i])
            if ma_episodes_2:
                ax1.plot(ma_episodes_2, ma_losses_2, 'r-', linewidth=2.5, label='Evaluator 2 (Moving Avg)')
        has_data = True
    
    if not has_data:
        ax1.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Loss')
    ax1.set_title('Evaluator Training Loss (Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Buffer Size
    ax2 = axes[1]
    
    # Filter out None values
    valid_episodes_buf_1 = [ep for ep, size in zip(episode_history, evaluator1_buffer_sizes) if size is not None]
    valid_sizes_1 = [size for size in evaluator1_buffer_sizes if size is not None]
    valid_episodes_buf_2 = [ep for ep, size in zip(episode_history, evaluator2_buffer_sizes) if size is not None]
    valid_sizes_2 = [size for size in evaluator2_buffer_sizes if size is not None]
    
    if valid_episodes_buf_1:
        ax2.plot(valid_episodes_buf_1, valid_sizes_1, 'b-', linewidth=2, label='Evaluator 1 Buffer')
    
    if valid_episodes_buf_2:
        ax2.plot(valid_episodes_buf_2, valid_sizes_2, 'r-', linewidth=2, label='Evaluator 2 Buffer')
        
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Buffer Size')
    ax2.set_title('Experience Buffer Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 PBS Evaluator progress plot saved to {save_path}")
    except Exception as e:
        print(f"⚠️  Error saving PBS evaluator plot: {e}")
    
    plt.close(fig)


def plot_setup_agent_progress(
    episode_history: List[int],
    setup_agent1_rewards: List[float],
    setup_agent2_rewards: List[float],
    setup_agent1_losses: List[float],
    setup_agent2_losses: List[float],
    save_path: str
):
    """
    Plots and saves the training progress of setup agents.

    Args:
        episode_history: List of episode numbers.
        setup_agent1_rewards: List of average rewards for setup agent 1.
        setup_agent2_rewards: List of average rewards for setup agent 2.
        setup_agent1_losses: List of loss values for setup agent 1.
        setup_agent2_losses: List of loss values for setup agent 2.
        save_path: Path to save the plot image.
    """
    # Validate input data
    if not episode_history or len(episode_history) == 0:
        raise ValueError("episode_history is empty - cannot plot setup agent progress")
    
    # Handle empty rewards gracefully - skip plotting if no data yet
    if not setup_agent1_rewards or not setup_agent2_rewards or len(setup_agent1_rewards) == 0 or len(setup_agent2_rewards) == 0:
        print("⚠️  Setup agent rewards are empty - skipping setup agent progress plot")
        # Create a simple placeholder plot with a message
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No setup agent data available yet', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Setup Agent Training Progress', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    if len(episode_history) != len(setup_agent1_rewards) or len(episode_history) != len(setup_agent2_rewards):
        raise ValueError(f"Length mismatch: episode_history={len(episode_history)}, "
                        f"setup_agent1_rewards={len(setup_agent1_rewards)}, "
                        f"setup_agent2_rewards={len(setup_agent2_rewards)}")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Setup Agent Training Progress', fontsize=16)

    # Plot 1: Average Rewards (with discrete points and cumulative average line)
    if len(episode_history) > 0:
        # Plot discrete points for each episode
        axs[0].scatter(episode_history, setup_agent1_rewards, label='Setup Agent 1 Reward', 
                      color='blue', marker='o', s=30, alpha=0.5, zorder=3)
        axs[0].scatter(episode_history, setup_agent2_rewards, label='Setup Agent 2 Reward', 
                      color='red', marker='o', s=30, alpha=0.5, zorder=3)
        
        # Calculate and plot cumulative average from the start
        if len(episode_history) >= 1:
            # Cumulative average: average of all episodes from episode 1 to current
            agent1_cumulative_avg = np.cumsum(setup_agent1_rewards) / np.arange(1, len(setup_agent1_rewards) + 1)
            agent2_cumulative_avg = np.cumsum(setup_agent2_rewards) / np.arange(1, len(setup_agent2_rewards) + 1)
            
            axs[0].plot(episode_history, agent1_cumulative_avg, color='blue', linestyle='-', linewidth=2, 
                       label='Setup Agent 1 Cumulative Avg', alpha=0.8, zorder=2)
            axs[0].plot(episode_history, agent2_cumulative_avg, color='red', linestyle='-', linewidth=2, 
                       label='Setup Agent 2 Cumulative Avg', alpha=0.8, zorder=2)
    
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].set_title('Rewards per Episode (with Cumulative Average)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Loss (with cumulative average)
    if len(episode_history) > 0 and len(setup_agent1_losses) > 0:
        # Plot discrete points for each episode
        axs[1].scatter(episode_history, setup_agent1_losses, label='Setup Agent 1 Loss', 
                      color='blue', marker='o', s=30, alpha=0.5, zorder=3)
        axs[1].scatter(episode_history, setup_agent2_losses, label='Setup Agent 2 Loss', 
                      color='red', marker='o', s=30, alpha=0.5, zorder=3)
        
        # Calculate and plot cumulative average from the start
        if len(episode_history) >= 1:
            # Cumulative average: average of all losses from episode 1 to current
            agent1_loss_avg = np.cumsum(setup_agent1_losses) / np.arange(1, len(setup_agent1_losses) + 1)
            agent2_loss_avg = np.cumsum(setup_agent2_losses) / np.arange(1, len(setup_agent2_losses) + 1)
            
            axs[1].plot(episode_history, agent1_loss_avg, color='blue', linestyle='-', linewidth=2, 
                       label='Setup Agent 1 Cumulative Avg', alpha=0.8, zorder=2)
            axs[1].plot(episode_history, agent2_loss_avg, color='red', linestyle='-', linewidth=2, 
                       label='Setup Agent 2 Cumulative Avg', alpha=0.8, zorder=2)
    
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss per Episode (with Cumulative Average)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if path contains a directory component
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure with error handling
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # Verify file was created
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Failed to create file: {save_path}")
    except Exception as e:
        plt.close(fig)
        raise Exception(f"Error saving plot to {save_path}: {e}")
    
    plt.close(fig)


def plot_additional_metrics(
    episode_history: List[int],
    epsilon_history: Dict[str, List[float]],
    pbs_buffer_sizes: Dict[str, List[int]],
    avg_q_history: Dict[str, List[float]],
    entropy_history: Dict[str, List[float]],
    save_path: str
):
    """
    Plots and saves additional training metrics.
    
    Args:
        episode_history: List of episode numbers.
        epsilon_history: Dict containing lists of epsilon values.
        pbs_buffer_sizes: Dict containing lists of PBS buffer sizes.
        avg_q_history: Dict containing lists of average Q-values.
        entropy_history: Dict containing lists of action entropy values.
        save_path: Path to save the plot image.
    """
    # Validate input data
    if not episode_history or len(episode_history) == 0:
        return
        
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Additional Training Metrics', fontsize=16)
    
    # Plot 1: Epsilon
    if epsilon_history and 'agent1' in epsilon_history:
        axs[0, 0].plot(episode_history, epsilon_history['agent1'], label='Agent 1', color='blue', alpha=0.7)
        axs[0, 0].plot(episode_history, epsilon_history['agent2'], label='Agent 2', color='red', alpha=0.7)
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Epsilon')
    axs[0, 0].set_title('Exploration Rate (Epsilon)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PBS Buffer Size
    if pbs_buffer_sizes and 'agent1' in pbs_buffer_sizes:
        axs[0, 1].plot(episode_history, pbs_buffer_sizes['agent1'], label='Agent 1', color='blue', alpha=0.7)
        axs[0, 1].plot(episode_history, pbs_buffer_sizes['agent2'], label='Agent 2', color='red', alpha=0.7)
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Buffer Size')
    axs[0, 1].set_title('PBS Experience Buffer Size')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Q-Value
    if avg_q_history and 'agent1' in avg_q_history:
        axs[1, 0].plot(episode_history, avg_q_history['agent1'], label='Agent 1', color='blue', alpha=0.7)
        axs[1, 0].plot(episode_history, avg_q_history['agent2'], label='Agent 2', color='red', alpha=0.7)
    axs[1, 0].set_xlabel('Episodes')
    axs[1, 0].set_ylabel('Avg Q-Value')
    axs[1, 0].set_title('Average Q-Value (Higher is Better) ↑')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Action Entropy
    if entropy_history and 'agent1' in entropy_history:
        axs[1, 1].plot(episode_history, entropy_history['agent1'], label='Agent 1', color='blue', alpha=0.7)
        axs[1, 1].plot(episode_history, entropy_history['agent2'], label='Agent 2', color='red', alpha=0.7)
    axs[1, 1].set_xlabel('Episodes')
    axs[1, 1].set_ylabel('Entropy')
    axs[1, 1].set_title('Action Entropy (Higher = More Exploration) ↑')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving additional metrics plot: {e}")
    
    plt.close(fig)