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
    total_steps: Optional[int] = None,
    num_envs: int = 1,
    phase_history: Optional[List[int]] = None,
    loss_steps: Optional[List[int]] = None,
    episode_end_steps: Optional[List[int]] = None
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
        num_envs: Number of parallel environments (for win rate normalization).
        phase_history: List of curriculum phase values per episode.
        loss_steps: List of global_step values when each loss was recorded (for uniform x-axis).
        episode_end_steps: List of global_step values when each episode ended (for phase mapping).
    """
    # Validate input data
    if not episode_history or len(episode_history) == 0:
        raise ValueError("episode_history is empty - cannot plot training progress")
    
    if not rewards_history or not rewards_history.get('agent1') or not rewards_history.get('agent2'):
        raise ValueError("rewards_history is empty or missing agent data")
    
    # Find minimum common length across all arrays for safe plotting
    min_len = min(
        len(episode_history),
        len(rewards_history['agent1']),
        len(rewards_history['agent2']),
        len(wins_history.get('agent1', [])) if wins_history else 0,
        len(wins_history.get('agent2', [])) if wins_history else 0,
    )
    
    if min_len == 0:
        raise ValueError("No data to plot - all history arrays are empty")
    
    # Truncate all arrays to minimum length to ensure alignment
    episode_history = episode_history[:min_len]
    rewards_history = {
        'agent1': rewards_history['agent1'][:min_len],
        'agent2': rewards_history['agent2'][:min_len]
    }
    wins_history = {
        'agent1': wins_history['agent1'][:min_len] if wins_history.get('agent1') else [],
        'agent2': wins_history['agent2'][:min_len] if wins_history.get('agent2') else []
    }
    if phase_history:
        phase_history = phase_history[:min_len]
    if episode_end_steps:
        episode_end_steps = episode_end_steps[:min_len]
    
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
            # Cumulative win rate: wins / (total episodes * num_envs)
            episode_nums = np.arange(1, len(wins_history['agent1']) + 1)
            # Normalize by number of environments per episode
            normalization_factor = episode_nums # num_envs not needed as episode_nums counts individual games
            agent1_win_rate = np.array(wins_history['agent1'], dtype=float) / normalization_factor
            agent2_win_rate = np.array(wins_history['agent2'], dtype=float) / normalization_factor
            
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

    # Plot 3: Policy Loss - Agent 1 Only (Agent 2 doesn't train)
    # Uses LINEAR SCALE with step-based x-axis for uniform spacing
    losses = policy_loss_history.get('agent1', [])
    
    if losses and len(losses) > 0:
        # Determine x-axis: Use SEQUENTIAL UPDATE COUNT to remove gaps
        x_values_scaled = list(range(1, len(losses) + 1))
        x_label = 'Training Updates (Sequential)'
        
        # Filter out zero/near-zero values (outliers that make graph unreadable)
        threshold = 1e-6
        
        valid_x = []
        valid_losses = []
        for x, loss in zip(x_values_scaled, losses):
            if loss > threshold:
                valid_x.append(x)
                valid_losses.append(loss)
        
        if valid_x:
            # Plot discrete points (scatter) for raw data - very faint
            axs[2].scatter(valid_x, valid_losses, label='Raw Loss', 
                          color='blue', marker='o', s=10, alpha=0.15, linewidths=0, zorder=1)
            
            # SMOOTHED CURVE 1: Rolling average (short window for local trends)
            window_short = min(50, max(5, len(valid_losses) // 20)) if len(valid_losses) > 5 else 1
            if len(valid_losses) >= window_short and window_short > 1:
                windowed_avg = np.convolve(valid_losses, np.ones(window_short)/window_short, mode='valid')
                windowed_x = valid_x[window_short-1:]
                axs[2].plot(windowed_x, windowed_avg, color='dodgerblue', linestyle='-', linewidth=1.5, 
                           label=f'Moving Avg ({window_short})', alpha=0.7, zorder=2)
            
            # SMOOTHED CURVE 2: Exponential Moving Average (for overall trend)
            if len(valid_losses) >= 20:
                alpha_ema = 0.01  # Smoothing factor (smaller = smoother)
                ema = [valid_losses[0]]
                for loss in valid_losses[1:]:
                    ema.append(alpha_ema * loss + (1 - alpha_ema) * ema[-1])
                axs[2].plot(valid_x, ema, color='darkblue', linestyle='-', linewidth=2.5, 
                           label='EMA Trend (α=0.01)', alpha=0.9, zorder=3)
        else:
            axs[2].text(0.5, 0.5, 'No loss data recorded yet', ha='center', va='center', 
                       transform=axs[2].transAxes, fontsize=12)
        
        axs[2].set_xlabel(x_label)
    else:
        axs[2].text(0.5, 0.5, 'No loss data recorded yet', ha='center', va='center', 
                   transform=axs[2].transAxes, fontsize=12)
        axs[2].set_xlabel('Training Steps')
    
    axs[2].set_ylabel('Policy Loss')
    axs[2].set_title('Agent 1 Policy Loss (Agent 2 does not train)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    

    # Draw curriculum phase boundaries on all subplots
    if phase_history is not None and len(phase_history) > 0:
        # Phase colors and names
        phase_colors = {
            1: '#4CAF50',  # Phase 1: Green (Physics of War)
            2: '#2196F3',  # Phase 2: Blue (Memory Gap)
            3: '#FF9800',  # Phase 3: Orange (Self-Play)
            4: '#9C27B0',  # Phase 4: Purple (League Training)
            5: '#F44336',  # Phase 5: Red (Scenario Drills)
        }
        phase_names = {
            1: 'P1: Physics',
            2: 'P2: Memory',
            3: 'P3: Self-Play',
            4: 'P4: League',
            5: 'P5: Drills',
        }
        
        # Find phase transition points and build phase regions
        transitions_ep = []
        regions_ep = []  # (start_ep, end_ep, phase)
        current_phase = phase_history[0] if phase_history else 1
        region_start = episode_history[0]
        
        for i, phase in enumerate(phase_history):
            if phase != current_phase:
                transitions_ep.append((episode_history[i], current_phase, phase))
                regions_ep.append((region_start, episode_history[i], current_phase))
                region_start = episode_history[i]
                current_phase = phase
        
        # Add final region (from last transition to end)
        regions_ep.append((region_start, episode_history[-1], current_phase))
        
        # --- Apply shading to plots ---
        
        # 1. Episode-based Plots (Rewards, Wins)
        for i in [0, 1]:  # axs[0] and axs[1] use Episode x-axis
             for start_ep, end_ep, phase in regions_ep:
                color = phase_colors.get(phase, 'gray')
                axs[i].axvspan(start_ep, end_ep, alpha=0.1, color=color, zorder=0)
             for ep, _, to_phase in transitions_ep:
                color = phase_colors.get(to_phase, 'gray')
                axs[i].axvline(x=ep, color=color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)

        # 2. Update-based Plot (Loss)
        # We need to map episode transitions to update indices
        if losses and len(losses) > 0 and loss_steps:
             # loss_steps contains the global_step for each update index
             # We need to find the update index corresponding to the episode transition steps
             
             # Convert episode regions to update index regions
             max_ep = episode_history[-1]
             max_step = total_steps if total_steps else (loss_steps[-1] if loss_steps else 1)
             
             import bisect
             
             for start_ep, end_ep, phase in regions_ep:
                 # Find approximate (or exact) global step bounds
                 start_global_step = 0
                 end_global_step = 0
                 
                 # Use exact tracking if available
                 if episode_end_steps and len(episode_end_steps) == len(episode_history):
                     # Find index of start_ep in episode_history
                     # (Assumes episode_history corresponds 1:1 with episode_end_steps)
                     try:
                        idx_start = episode_history.index(start_ep)
                        start_global_step = episode_end_steps[idx_start]
                     except ValueError:
                         start_global_step = (start_ep / max_ep) * max_step # Fallback
                     
                     try:
                        idx_end = episode_history.index(end_ep)
                        end_global_step = episode_end_steps[idx_end]
                     except ValueError:
                        end_global_step = (end_ep / max_ep) * max_step # Fallback
                 else:
                     # Linear interpolation fallback
                     start_global_step = (start_ep / max_ep) * max_step
                     end_global_step = (end_ep / max_ep) * max_step
                 
                 # Map global steps to update indices using bisect on loss_steps
                 # x_values_scaled is 1-based index
                 start_idx = bisect.bisect_left(loss_steps, start_global_step) + 1
                 end_idx = bisect.bisect_left(loss_steps, end_global_step) + 1
                 
                 # Clamp
                 start_idx = max(1, min(start_idx, len(losses)))
                 end_idx = max(1, min(end_idx, len(losses)))
                 
                 color = phase_colors.get(phase, 'gray')
                 axs[2].axvspan(start_idx, end_idx, alpha=0.1, color=color, zorder=0)
             
             # Vertical lines for loss plot (simpler loop over regions usually suffices, but being strict)
             for start_ep, end_ep, phase in regions_ep:
                 # We only draw the START line of a new phase (which is the previous phase's end, technically)
                 # Actually, transitions loop is better
                 pass

             for ep, _, to_phase in transitions_ep:
                 # Calculate transition step
                 transition_step = 0
                 if episode_end_steps and len(episode_end_steps) == len(episode_history):
                     try:
                        idx = episode_history.index(ep)
                        transition_step = episode_end_steps[idx]
                     except ValueError:
                        transition_step = (ep / max_ep) * max_step
                 else:
                     transition_step = (ep / max_ep) * max_step
                 
                 trans_idx = bisect.bisect_left(loss_steps, transition_step) + 1
                 color = phase_colors.get(to_phase, 'gray')
                 axs[2].axvline(x=trans_idx, color=color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)

        # Add phase labels at top of first subplot with colored borders
        for start_ep, end_ep, phase in regions_ep:
            mid_ep = (start_ep + end_ep) / 2
            color = phase_colors.get(phase, 'gray')
            # Get y position at top of plot
            y_max = axs[0].get_ylim()[1]
            y_min = axs[0].get_ylim()[0]
            y_pos = y_max - (y_max - y_min) * 0.05  # 5% from top
            
            axs[0].text(mid_ep, y_pos,
                       phase_names.get(phase, f'P{phase}'),
                       ha='center', va='top', fontsize=10, 
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                edgecolor='white', alpha=0.9, linewidth=2))

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
            print(f"[WARN] No images found for GIF creation at episode {episode}")
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
            print(f"[WARN] No frames to create GIF at episode {episode}")
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
        
        print(f"[INFO] Training GIF saved to {gif_path} ({len(frames)} frames)")
        
    except Exception as e:
        print(f"[WARN] Error creating training GIF at episode {episode}: {e}")
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
            print(f"[WARN] No game states to create GIF for episode {episode}")
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
            print(f"[WARN] No frames created for episode {episode}")
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
        
        print(f"[INFO] Episode GIF saved to {save_path} ({len(frames)} frames, {frame_duration}ms per frame)")
        
    except Exception as e:
        print(f"[WARN] Error creating episode GIF for episode {episode}: {e}")
        import traceback
        traceback.print_exc()


def plot_pbs_evaluator_progress(
    episode_history: List[int],
    evaluator1_losses: List[float],
    evaluator2_losses: List[float],
    evaluator1_buffer_sizes: List[int],
    evaluator2_buffer_sizes: List[int],
    save_path: str,
    total_episodes: Optional[int] = None,
    aaren_losses: Optional[List[float]] = None,
    aaren_accuracies: Optional[List[float]] = None,
    aaren_buffer_sizes: Optional[List[int]] = None,
    # New end-to-end AAREN metrics
    aaren_grad_norms: Optional[List[float]] = None,
    aaren_embedding_stds: Optional[List[float]] = None,
    aaren_active_positions: Optional[List[int]] = None,
    dqn_grad_norms: Optional[List[float]] = None  # DQN gradient norms for comparison
):
    """
    Plot PBS evaluator and AAREN improvement metrics.
    
    Args:
        episode_history: List of episode numbers
        evaluator1_losses: List of training losses for evaluator 1
        evaluator2_losses: List of training losses for evaluator 2
        evaluator1_buffer_sizes: List of experience buffer sizes for evaluator 1
        evaluator2_buffer_sizes: List of experience buffer sizes for evaluator 2
        save_path: Path to save the plot
        total_episodes: Total episodes across all training runs (for display)
        aaren_losses: Optional list of AAREN training losses (legacy separate training)
        aaren_accuracies: Optional list of AAREN prediction accuracies (legacy)
        aaren_buffer_sizes: Optional list of AAREN buffer sizes (legacy)
        aaren_grad_norms: Optional list of AAREN gradient norms (end-to-end training)
        aaren_embedding_stds: Optional list of embedding std dev (end-to-end training)
        aaren_active_positions: Optional list of active positions with history
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
    
    # Check if AAREN data is available (legacy OR end-to-end)
    has_legacy_aaren = (aaren_losses is not None and aaren_accuracies is not None and 
                 len(aaren_losses) > 0 and len(aaren_accuracies) > 0)
    has_e2e_aaren = (aaren_grad_norms is not None and aaren_embedding_stds is not None and
                     len(aaren_grad_norms) > 0 and len(aaren_embedding_stds) > 0)
    has_aaren = has_legacy_aaren or has_e2e_aaren
    
    # Create figure with 2x2 grid if AAREN data available, else 2x1
    if has_aaren:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.patch.set_facecolor('white')
    
    # Title
    title = 'AAREN End-to-End Training Metrics' if has_e2e_aaren else ('PBS Evaluator & AAREN Metrics' if has_aaren else 'PBS Evaluator Metrics')
    if total_episodes is not None:
        title += f' | Total Episodes: {total_episodes:,}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. AAREN vs DQN Gradient Norms (shows if AAREN is learning alongside DQN)
    ax1 = axes[0]
    
    if has_e2e_aaren and aaren_grad_norms and dqn_grad_norms and len(dqn_grad_norms) > 0:
        # Filter valid data
        valid_eps_aaren = []
        valid_aaren_grads = []
        valid_eps_dqn = []
        valid_dqn_grads = []
        
        for i, (ep, ag, dg) in enumerate(zip(episode_history[:len(aaren_grad_norms)], 
                                              aaren_grad_norms, 
                                              dqn_grad_norms[:len(aaren_grad_norms)])):
            if ag is not None and ag > 0:
                valid_eps_aaren.append(ep)
                valid_aaren_grads.append(ag)
            if dg is not None and dg > 0:
                valid_eps_dqn.append(ep)
                valid_dqn_grads.append(dg)
        
        if valid_aaren_grads:
            ax1.scatter(valid_eps_aaren, valid_aaren_grads, c='green', alpha=0.3, s=15, label='AAREN Grad')
            # Moving average
            window = min(50, len(valid_aaren_grads) // 2) if len(valid_aaren_grads) > 1 else 1
            if len(valid_aaren_grads) >= window and window > 1:
                ma_vals = []
                ma_eps = []
                for i in range(window, len(valid_aaren_grads) + 1):
                    ma_vals.append(np.mean(valid_aaren_grads[i-window:i]))
                    ma_eps.append(valid_eps_aaren[i-1])
                if ma_eps:
                    ax1.plot(ma_eps, ma_vals, 'g-', linewidth=2.5, label=f'AAREN MA ({window})')
        
        if valid_dqn_grads:
            ax1.scatter(valid_eps_dqn, valid_dqn_grads, c='blue', alpha=0.2, s=10, label='DQN Grad')
            # Moving average
            window = min(50, len(valid_dqn_grads) // 2) if len(valid_dqn_grads) > 1 else 1
            if len(valid_dqn_grads) >= window and window > 1:
                ma_vals = []
                ma_eps = []
                for i in range(window, len(valid_dqn_grads) + 1):
                    ma_vals.append(np.mean(valid_dqn_grads[i-window:i]))
                    ma_eps.append(valid_eps_dqn[i-1])
                if ma_eps:
                    ax1.plot(ma_eps, ma_vals, 'b-', linewidth=2, label=f'DQN MA ({window})')
        
        ax1.set_yscale('log')
        ax1.set_title('Gradient Norms: AAREN vs DQN (log scale)')
    else:
        ax1.text(0.5, 0.5, 'Waiting for gradient data...', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Gradient Norms: AAREN vs DQN')
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Gradient Norm (log)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient Ratio: AAREN / DQN (shows relative learning rate)
    ax2 = axes[1]
    
    if has_e2e_aaren and aaren_grad_norms and dqn_grad_norms and len(dqn_grad_norms) > 0:
        valid_eps_ratio = []
        valid_ratios = []
        
        for ep, ag, dg in zip(episode_history[:len(aaren_grad_norms)], 
                              aaren_grad_norms, 
                              dqn_grad_norms[:len(aaren_grad_norms)]):
            if ag is not None and dg is not None and ag > 0 and dg > 0:
                valid_eps_ratio.append(ep)
                valid_ratios.append(ag / dg)  # Ratio of gradients
        
        if valid_ratios:
            ax2.scatter(valid_eps_ratio, valid_ratios, c='purple', alpha=0.4, s=15, label='AAREN/DQN Ratio')
            # Moving average
            window = min(50, len(valid_ratios) // 2) if len(valid_ratios) > 1 else 1
            if len(valid_ratios) >= window and window > 1:
                ma_vals = []
                ma_eps = []
                for i in range(window, len(valid_ratios) + 1):
                    ma_vals.append(np.mean(valid_ratios[i-window:i]))
                    ma_eps.append(valid_eps_ratio[i-1])
                if ma_eps:
                    ax2.plot(ma_eps, ma_vals, 'purple', linewidth=2.5, label=f'MA ({window})')
            
            # Reference line at 1.0 (equal gradients)
            ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Equal (1.0)')
            ax2.set_title('AAREN/DQN Gradient Ratio')
        else:
            ax2.text(0.5, 0.5, 'Waiting for ratio data...', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('AAREN/DQN Gradient Ratio')
    else:
        ax2.text(0.5, 0.5, 'Waiting for gradient data...', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('AAREN/DQN Gradient Ratio')
        
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Gradient Ratio')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. AAREN Metrics (gradient norm for end-to-end OR loss for legacy)
    if has_aaren:
        ax3 = axes[2]
        
        if has_e2e_aaren and aaren_grad_norms:
            # End-to-end training: plot gradient norms
            valid_eps = []
            valid_grads = []
            for ep, grad in zip(episode_history[:len(aaren_grad_norms)], aaren_grad_norms):
                if grad is not None and grad > 0:
                    valid_eps.append(ep)
                    valid_grads.append(grad)
            
            if valid_grads:
                ax3.scatter(valid_eps, valid_grads, c='green', alpha=0.4, s=20, label='AAREN Grad Norm')
                
                # Moving average
                window = min(50, len(valid_grads) // 2) if len(valid_grads) > 1 else 1
                if len(valid_grads) >= window and window > 1:
                    ma_vals = []
                    ma_eps = []
                    for i in range(window, len(valid_grads) + 1):
                        ma_vals.append(np.mean(valid_grads[i-window:i]))
                        ma_eps.append(valid_eps[i-1])
                    if ma_eps:
                        ax3.plot(ma_eps, ma_vals, 'g-', linewidth=2.5, label=f'Moving Avg ({window} ep)')
                
                ax3.set_yscale('log')
            else:
                ax3.text(0.5, 0.5, 'Waiting for gradients...', ha='center', va='center', transform=ax3.transAxes)
            
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Gradient Norm (log)')
            ax3.set_title('AAREN Gradient Norm (Higher = Active Learning)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        elif has_legacy_aaren and aaren_losses:
            # Legacy separate training: plot loss
            valid_aaren_eps = []
            valid_aaren_losses = []
            for ep, loss in zip(episode_history[:len(aaren_losses)], aaren_losses):
                if loss is not None and loss > 0:
                    valid_aaren_eps.append(ep)
                    valid_aaren_losses.append(loss)
            
            if valid_aaren_losses:
                ax3.scatter(valid_aaren_eps, valid_aaren_losses, c='green', alpha=0.4, s=20, label='AAREN Loss')
                
                window = min(50, len(valid_aaren_losses) // 2) if len(valid_aaren_losses) > 1 else 1
                if len(valid_aaren_losses) >= window and window > 1:
                    ma_losses = []
                    ma_eps = []
                    for i in range(window, len(valid_aaren_losses) + 1):
                        ma_losses.append(np.mean(valid_aaren_losses[i-window:i]))
                        ma_eps.append(valid_aaren_eps[i-1])
                    if ma_eps:
                        ax3.plot(ma_eps, ma_losses, 'g-', linewidth=2.5, label=f'Moving Avg ({window} ep)')
                
                ax3.set_yscale('log')
            else:
                ax3.text(0.5, 0.5, 'No AAREN loss data yet', ha='center', va='center', transform=ax3.transAxes)
            
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('AAREN Loss (log)')
            ax3.set_title('AAREN Training Loss (Lower is Better)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. AAREN Embedding Std (end-to-end) OR Accuracy (legacy)
        ax4 = axes[3]
        
        if has_e2e_aaren and aaren_embedding_stds:
            # End-to-end training: plot embedding std
            valid_eps = []
            valid_stds = []
            for ep, std in zip(episode_history[:len(aaren_embedding_stds)], aaren_embedding_stds):
                if std is not None:
                    valid_eps.append(ep)
                    valid_stds.append(std)
            
            if valid_stds:
                ax4.scatter(valid_eps, valid_stds, c='purple', alpha=0.4, s=20, label='Embedding Std')
                
                window = min(50, len(valid_stds) // 2) if len(valid_stds) > 1 else 1
                if len(valid_stds) >= window and window > 1:
                    ma_vals = []
                    ma_eps = []
                    for i in range(window, len(valid_stds) + 1):
                        ma_vals.append(np.mean(valid_stds[i-window:i]))
                        ma_eps.append(valid_eps[i-1])
                    if ma_eps:
                        ax4.plot(ma_eps, ma_vals, 'purple', linewidth=2.5, label=f'Moving Avg ({window} ep)')
            else:
                ax4.text(0.5, 0.5, 'Waiting for embeddings...', ha='center', va='center', transform=ax4.transAxes)
            
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Embedding Std Dev')
            ax4.set_title('AAREN Embedding Diversity (Non-zero = Learning)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        elif has_legacy_aaren and aaren_accuracies:
            # Legacy separate training: plot accuracy
            valid_acc_eps = []
            valid_accs = []
            for ep, acc in zip(episode_history[:len(aaren_accuracies)], aaren_accuracies):
                if acc is not None:
                    valid_acc_eps.append(ep)
                    valid_accs.append(acc * 100)
            
            if valid_accs:
                ax4.scatter(valid_acc_eps, valid_accs, c='purple', alpha=0.4, s=20, label='AAREN Accuracy')
                
                window = min(50, len(valid_accs) // 2) if len(valid_accs) > 1 else 1
                if len(valid_accs) >= window and window > 1:
                    ma_accs = []
                    ma_eps = []
                    for i in range(window, len(valid_accs) + 1):
                        ma_accs.append(np.mean(valid_accs[i-window:i]))
                        ma_eps.append(valid_acc_eps[i-1])
                    if ma_eps:
                        ax4.plot(ma_eps, ma_accs, 'purple', linewidth=2.5, label=f'Moving Avg ({window} ep)')
            else:
                ax4.text(0.5, 0.5, 'No accuracy data yet', ha='center', va='center', transform=ax4.transAxes)
            
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('AAREN Prediction Accuracy (Higher is Better)')
            ax4.set_ylim(0, 100)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] PBS/AAREN progress plot saved to {save_path}")
    except Exception as e:
        print(f"[WARN] Error saving PBS evaluator plot: {e}")
    
    plt.close(fig)


def plot_aaren_progress(
    episode_history: List[int],
    aaren_losses: List[float],
    aaren_accuracies: List[float],
    aaren_buffer_sizes: List[int],
    aaren_grad_norms: Optional[List[float]] = None,
    aaren_embedding_stds: Optional[List[float]] = None,
    dqn_grad_norms: Optional[List[float]] = None,
    save_path: str = "aaren_progress.png",
    total_episodes: Optional[int] = None
):
    """
    Wrapper for plot_pbs_evaluator_progress using only AAREN-relevant parameters.
    Called from training/checkpointing.py.
    """
    plot_pbs_evaluator_progress(
        episode_history=episode_history,
        evaluator1_losses=[0.0] * len(episode_history),
        evaluator2_losses=[0.0] * len(episode_history),
        evaluator1_buffer_sizes=[0] * len(episode_history),
        evaluator2_buffer_sizes=[0] * len(episode_history),
        save_path=save_path,
        total_episodes=total_episodes,
        aaren_losses=aaren_losses,
        aaren_accuracies=aaren_accuracies,
        aaren_buffer_sizes=aaren_buffer_sizes,
        aaren_grad_norms=aaren_grad_norms,
        aaren_embedding_stds=aaren_embedding_stds,
        dqn_grad_norms=dqn_grad_norms
    )



def plot_additional_metrics(
    episode_history: List[int],
    episode_lengths: Dict[str, List[float]],
    win_rate_history: Dict[str, List[float]],
    avg_q_history: Dict[str, List[float]],
    entropy_history: Dict[str, List[float]],
    save_path: str
):
    """
    Plots and saves additional training metrics.
    
    Args:
        episode_history: List of episode numbers.
        episode_lengths: Dict containing lists of episode lengths.
        pbs_buffer_sizes: Dict containing lists of PBS buffer sizes.
        avg_q_history: Dict containing lists of average Q-values.
        entropy_history: Dict containing lists of action entropy values.
        save_path: Path to save the plot image.
    """
    # Validate input data
    if not episode_history or len(episode_history) == 0:
        return
    
    # Ensure all metric lists match episode_history length by PADDING with None at the front.
    # This handles cases where new metrics (like Q-values) were added mid-training
    # and don't have history for early episodes.
    target_len = len(episode_history)
    
    def pad_history(history_dict):
        if not history_dict:
            return {}
        padded_dict = {}
        for key, val_list in history_dict.items():
            if len(val_list) < target_len:
                # Pad with None at the BEGINNING (assuming data corresponds to latest episodes)
                padding = [None] * (target_len - len(val_list))
                padded_dict[key] = padding + val_list
            elif len(val_list) > target_len:
                # Truncate from the BEGINNING if too long (unlikely, but safe)
                # actually usually we trim from end if mismatch, but history grows...
                # Let's assume simplest case: trim to match if longer
                padded_dict[key] = val_list[:target_len]
            else:
                padded_dict[key] = val_list
        return padded_dict

    episode_lengths = pad_history(episode_lengths)
    win_rate_history = pad_history(win_rate_history)
    avg_q_history = pad_history(avg_q_history)
    entropy_history = pad_history(entropy_history)
        
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Additional Training Metrics', fontsize=16)
    
    # Plot 1: Episode Length
    if episode_lengths and 'agent1' in episode_lengths:
        # Use dot markers (o) and line (-)
        axs[0, 0].plot(episode_history, episode_lengths['agent1'], 'o-', label='Agent 1', color='green', alpha=0.7, markersize=3)
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Steps')
    axs[0, 0].set_title('Average Episode Length (Moving Avg)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Win Rate Moving Average (Replaces PBS Buffer)
    if win_rate_history and 'agent1' in win_rate_history:
        axs[0, 1].plot(episode_history, win_rate_history['agent1'], 'o-', label='Agent 1 (100-ep Avg)', color='purple', alpha=0.7, markersize=3)
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Win Rate (0-1)')
    axs[0, 1].set_title('Run Win Rate (Last 100 Episodes)')
    axs[0, 1].set_ylim(0, 1.0)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Q-Value
    if avg_q_history and 'agent1' in avg_q_history:
        axs[1, 0].plot(episode_history, avg_q_history['agent1'], 'o-', label='Agent 1', color='blue', alpha=0.7, markersize=3)
    axs[1, 0].set_xlabel('Episodes')
    axs[1, 0].set_ylabel('Avg Q-Value')
    axs[1, 0].set_title('Average Q-Value (Higher = Better Learning) ↑')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Action Entropy (Noisy Net Sigma)
    if entropy_history and 'agent1' in entropy_history:
        axs[1, 1].plot(episode_history, entropy_history['agent1'], 'o-', label='Agent 1', color='orange', alpha=0.7, markersize=3)
    axs[1, 1].set_xlabel('Episodes')
    axs[1, 1].set_ylabel('Entropy (Sigma)')
    axs[1, 1].set_title('Exploration Entropy (Noisy Net Sigma) ↑')
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