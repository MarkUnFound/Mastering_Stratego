"""
Training Script for DQN Agents in Stratego
"""

# Set matplotlib backend to non-interactive BEFORE any imports
# This prevents Tkinter errors when used in multi-threaded environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (thread-safe, no GUI)

import torch
import numpy as np
import random
import os
import sys
import glob
import queue
import threading
import time
import copy
import traceback
import json
from typing import List, Tuple, Optional
from tqdm import tqdm

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import StrategoEnvironment
from parallel_environment import ParallelStrategoEnvironment
from dqn_agent import DQNAgent
from setup_agent import SetupAgent
from game_state import GameState
from training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_setup_agent_progress, plot_pbs_evaluator_progress, plot_additional_metrics
from pbs_visualizer import visualize_pbs_state, create_pbs_gif
from piece import PieceType, PIECE_RANKS
from board import LAKE_SQUARE

# Import reset function (optional)
try:
    from reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False


# Hyperparameters
NUM_ENVS = 16  # Number of parallel environments (balanced for CPU/GPU)
BATCH_SIZE = 128  # Large batch size to maximize GPU usage and amortize PER sampling cost
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995  # Slower decay for longer training
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000000
LEARNING_RATE = 0.0001
NUM_EPISODES = 10000  # Total episodes to train
SAVE_INTERVAL = 50   # Save model every N episodes
EVAL_INTERVAL = 100  # Evaluate every N episodes
PREFETCH_QUEUE_SIZE = 4 # Size of the prefetch queue
REPLAY_UPDATE_INTERVAL = 2 # Train every N steps (train very frequently)
REPLAY_UPDATES_PER_STEP = 4 # Multiple gradient updates per training step (maximize GPU work)
TARGET_UPDATE_INTERVAL = 1000 # Update target network every N steps

# Visualization settings
GENERATE_GIFS = False # Whether to generate GIFs of games
GIF_INTERVAL = 100   # Generate GIF every N episodes (reduced frequency)

class ReplayPrefetcher:
    """Background prefetcher that samples replay batches asynchronously."""

    def __init__(self, agent, max_queue_size: int = 4):
        self.agent = agent
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            batch = self.agent.sample_replay_batch()
            if batch is None:
                time.sleep(0.01)
                continue
            try:
                self.queue.put(batch, timeout=0.1)
            except queue.Full:
                continue

    def get_batch(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def _save_counters(total_episodes_file, total_steps_file, total_episodes, total_steps):
    with open(total_episodes_file, 'w') as f:
        f.write(str(total_episodes))
    with open(total_steps_file, 'w') as f:
        f.write(str(total_steps))
    print(f"💾 Saved persistent counters: {total_episodes} episodes, {total_steps:,} steps")


def _save_training_history(model_save_path: str, 
                          episode_history: List[int],
                          rewards_history: dict,
                          wins_history: dict,
                          epsilon_history: dict,
                          policy_loss_history: dict,
                          setup_agent1_rewards: List[float],
                          setup_agent2_rewards: List[float],
                          setup_agent1_losses: List[float],
                          setup_agent2_losses: List[float],
                          pbs_evaluator1_losses: List[float],
                          pbs_evaluator2_losses: List[float],
                          pbs_evaluator1_buffer_sizes: List[int],
                          pbs_evaluator2_buffer_sizes: List[int],
                          avg_q_history: dict,
                          entropy_history: dict):
    """Save training history to JSON file for continuity across training sessions"""
    history_file = os.path.join(model_save_path, "training_history.json")
    try:
        history_data = {
            'episode_history': episode_history,
            'rewards_history': rewards_history,
            'wins_history': wins_history,
            'epsilon_history': epsilon_history,
            'policy_loss_history': policy_loss_history,
            'setup_agent1_rewards': setup_agent1_rewards,
            'setup_agent2_rewards': setup_agent2_rewards,
            'setup_agent1_losses': setup_agent1_losses,
            'setup_agent2_losses': setup_agent2_losses,
            'pbs_evaluator1_losses': pbs_evaluator1_losses,
            'pbs_evaluator2_losses': pbs_evaluator2_losses,
            'pbs_evaluator1_buffer_sizes': pbs_evaluator1_buffer_sizes,
            'pbs_evaluator1_buffer_sizes': pbs_evaluator1_buffer_sizes,
            'pbs_evaluator2_buffer_sizes': pbs_evaluator2_buffer_sizes,
            'avg_q_history': avg_q_history,
            'entropy_history': entropy_history
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save training history: {e}")


def calculate_setup_agent_reward(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                 player_id: int, winner: int, move_count: int) -> float:
    """
    Calculate reward for a setup agent based on game outcome and placement quality.
    
    Args:
        placement: List of (piece_type, position) tuples
        player_id: 1 or -1
        winner: 1, -1, or 0 (draw)
        move_count: Number of moves in the game
        
    Returns:
        Reward value (higher is better)
    """
    # Base reward from game outcome
    if winner == player_id:
        base_reward = 10.0
    elif winner == -player_id:
        base_reward = -10.0
    else:
        base_reward = 0.0
    
    # Bonus/penalty for game length
    # Winning quickly or losing slowly is better
    if winner == player_id:
        # Win bonus: faster wins are better (max 50 moves as reference)
        length_bonus = max(0, (50 - move_count) / 50.0) * 5.0
    elif winner == -player_id:
        # Loss penalty: longer games mean better defense (max 100 moves)
        length_bonus = max(0, move_count / 100.0) * 3.0
    else:
        length_bonus = 0.0
    
    # Evaluate placement quality
    quality_bonus = 0.0
    
    # Find flag position
    flag_pos = None
    bomb_positions = []
    for piece, pos in placement:
        if piece == PieceType.FLAG:
            flag_pos = pos
        elif piece == PieceType.BOMB:
            bomb_positions.append(pos)
    
    if flag_pos:
        # Reward flag protection (surrounded by bombs)
        adjacent_positions = [
            (flag_pos[0]-1, flag_pos[1]), (flag_pos[0]+1, flag_pos[1]),
            (flag_pos[0], flag_pos[1]-1), (flag_pos[0], flag_pos[1]+1)
        ]
        protected_count = sum(1 for pos in adjacent_positions if pos in bomb_positions)
        quality_bonus += protected_count * 0.5
        
        # Reward flag being in back row (safer)
        if player_id == 1 and flag_pos[0] == 9:  # Player 1 back row
            quality_bonus += 1.0
        elif player_id == -1 and flag_pos[0] == 0:  # Player 2 back row
            quality_bonus += 1.0
    
    total_reward = base_reward + length_bonus + quality_bonus
    return total_reward

def _load_training_history(model_save_path: str) -> dict:
    """Load training history from JSON file if it exists"""
    history_file = os.path.join(model_save_path, "training_history.json")
    
    if not os.path.exists(history_file):
        return None
    
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        print(f"📊 Loaded training history: {len(history_data.get('episode_history', []))} episodes")
        return history_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Could not load training history: {e}, starting fresh")
        return None


def evaluate_flag_protection(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int, board_size: int = 10) -> float:
    """
    Evaluate how well the flag is protected by bombs, lakes, or strong pieces.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        board_size: Size of the board (default 10)
        
    Returns:
        Protection score (0.0 to 1.0, higher is better)
    """
    # Find flag position
    flag_pos = None
    for piece, pos in placement:
        if piece == PieceType.FLAG:
            flag_pos = pos
            break
    
    if flag_pos is None:
        return 0.0  # No flag found
    
    flag_r, flag_c = flag_pos
    
    # Define adjacent positions (4 directions: up, down, left, right)
    adjacent_positions = [
        (flag_r - 1, flag_c),  # Up
        (flag_r + 1, flag_c),  # Down
        (flag_r, flag_c - 1),  # Left
        (flag_r, flag_c + 1),  # Right
    ]
    
    # Filter valid positions (within board bounds)
    valid_adjacent = [(r, c) for r, c in adjacent_positions 
                     if 0 <= r < board_size and 0 <= c < board_size]
    
    # Create a map of positions to pieces
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Check for protection
    protection_score = 0.0
    
    # IMPROVED: Add row-based vulnerability penalty
    # Flags closer to enemy (front rows) are more vulnerable
    if player_id == 1:
        # Player 1: row 9 is front (most vulnerable), row 6 is back (safest)
        # Vulnerability: 0.0 (back) to 1.0 (front)
        row_vulnerability = (flag_r - 6) / 3.0 if flag_r >= 6 else 0.0
    else:  # player_id == -1
        # Player 2: row 0 is front (most vulnerable), row 3 is back (safest)
        # Vulnerability: 0.0 (back) to 1.0 (front)
        row_vulnerability = (3 - flag_r) / 3.0 if flag_r <= 3 else 0.0
    
    # Reduce protection score based on vulnerability
    # Front-row flags need MUCH more protection to get same score
    vulnerability_penalty = row_vulnerability * 0.5  # Reduce protection score by up to 50% for front-row flags
    
    for adj_pos in valid_adjacent:
        if adj_pos in position_to_piece:
            piece = position_to_piece[adj_pos]
            
            # Check if it's a bomb (strong protection)
            if piece == PieceType.BOMB:
                protection_score += 0.4  # Bomb provides strong protection
            # Check if it's a strong piece (MARSHAL, GENERAL, COLONEL)
            elif piece in [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL]:
                protection_score += 0.3  # Strong pieces provide good protection
            # Check if it's a medium piece (MAJOR, CAPTAIN)
            elif piece in [PieceType.MAJOR, PieceType.CAPTAIN]:
                protection_score += 0.2  # Medium pieces provide some protection
            # Any other piece provides minimal protection
            else:
                protection_score += 0.1
        else:
            # Check if it's a lake (lakes provide protection by blocking movement)
            # Lakes are at fixed positions: (4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)
            lakes = [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]
            if adj_pos in lakes:
                protection_score += 0.3  # Lake provides protection
    
    # Apply vulnerability penalty (front-row flags need more protection)
    protection_score = max(0.0, protection_score - vulnerability_penalty)
    
    # Normalize score (max possible is 4 adjacent positions * 0.4 = 1.6, but we cap at 1.0)
    return min(1.0, protection_score)


def evaluate_piece_distribution(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                player_id: int) -> float:
    """
    Evaluate balanced piece distribution - reward spreading strong pieces across rows.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Distribution score (0.0 to 1.0, higher is better)
    """
    # Define strong pieces
    strong_pieces = [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL, 
                     PieceType.MAJOR, PieceType.CAPTAIN]
    
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]  # Bottom rows
    else:
        player_rows = [0, 1, 2, 3]  # Top rows
    
    # Count strong pieces per row
    row_counts = {row: 0 for row in player_rows}
    total_strong = 0
    
    for piece, (r, c) in placement:
        if piece in strong_pieces and r in player_rows:
            row_counts[r] += 1
            total_strong += 1
    
    if total_strong == 0:
        return 0.5  # Neutral if no strong pieces
    
    # Calculate distribution variance (lower variance = better distribution)
    counts = list(row_counts.values())
    mean_count = sum(counts) / len(counts)
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    
    # Normalize: perfect distribution (all rows equal) = 1.0, all in one row = 0.0
    max_variance = (total_strong ** 2) / len(player_rows)  # Worst case: all in one row
    if max_variance == 0:
        return 1.0
    
    distribution_score = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, distribution_score))


def evaluate_scout_placement(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int) -> float:
    """
    Reward scouts placed in forward positions for early scouting.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Scout placement score (0.0 to 1.0, higher is better)
    """
    # Forward rows for each player (closest to enemy)
    if player_id == 1:
        forward_rows = [6]  # Row 6 is closest to enemy (row 5 is lakes, row 4 is enemy territory)
    else:
        forward_rows = [3]  # Row 3 is closest to enemy (row 4 is lakes, row 5 is enemy territory)
    
    scouts = [(piece, pos) for piece, pos in placement if piece == PieceType.SCOUT]
    if len(scouts) == 0:
        return 0.0
    
    # Count scouts in forward positions
    forward_scouts = sum(1 for _, (r, c) in scouts if r in forward_rows)
    
    # Reward: 0.5 base + 0.5 for forward placement ratio
    forward_ratio = forward_scouts / len(scouts)
    return 0.5 + (0.5 * forward_ratio)


def evaluate_bomb_placement(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                           player_id: int) -> float:
    """
    Reward bombs protecting key pieces (not just flag).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Bomb placement score (0.0 to 1.0, higher is better)
    """
    # Create position to piece map
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Key pieces that should be protected (high value pieces)
    key_pieces = [PieceType.FLAG, PieceType.MARSHAL, PieceType.GENERAL]
    
    # Find key piece positions
    key_positions = [pos for piece, pos in placement if piece in key_pieces]
    
    if len(key_positions) == 0:
        return 0.0
    
    # Count how many key pieces have bombs adjacent
    protected_count = 0
    
    for key_pos in key_positions:
        key_r, key_c = key_pos
        adjacent = [
            (key_r - 1, key_c), (key_r + 1, key_c),
            (key_r, key_c - 1), (key_r, key_c + 1)
        ]
        
        # Check if any adjacent position has a bomb
        for adj_pos in adjacent:
            if adj_pos in position_to_piece:
                if position_to_piece[adj_pos] == PieceType.BOMB:
                    protected_count += 1
                    break  # Count each key piece only once
    
    # Score: ratio of protected key pieces
    return protected_count / len(key_positions)


def evaluate_defensive_formation(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                 player_id: int) -> float:
    """
    Reward pieces forming defensive lines (pieces in same row/column).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Formation score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Count pieces per row and column
    row_counts = {r: 0 for r in player_rows}
    col_counts = {c: 0 for c in range(10)}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            row_counts[r] += 1
            col_counts[c] += 1
    
    # Reward rows with multiple pieces (defensive lines)
    row_score = sum(1 for count in row_counts.values() if count >= 8) / len(player_rows)
    
    # Reward columns with multiple pieces (vertical defense)
    col_score = sum(1 for count in col_counts.values() if count >= 3) / 10.0
    
    # Average of row and column formation
    return (row_score + col_score) / 2.0


def evaluate_piece_coordination(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                player_id: int) -> float:
    """
    Reward pieces that can support each other (e.g., miners near bombs).
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Coordination score (0.0 to 1.0, higher is better)
    """
    # Create position to piece map
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find bomb positions
    bomb_positions = [pos for piece, pos in placement if piece == PieceType.BOMB]
    
    if len(bomb_positions) == 0:
        return 0.0
    
    # Count miners adjacent to bombs (miners can defuse bombs)
    coordinated_count = 0
    
    for bomb_pos in bomb_positions:
        bomb_r, bomb_c = bomb_pos
        adjacent = [
            (bomb_r - 1, bomb_c), (bomb_r + 1, bomb_c),
            (bomb_r, bomb_c - 1), (bomb_r, bomb_c + 1)
        ]
        
        # Check if any adjacent position has a miner
        for adj_pos in adjacent:
            if adj_pos in position_to_piece:
                if position_to_piece[adj_pos] == PieceType.MINER:
                    coordinated_count += 1
                    break  # Count each bomb only once
    
    # Score: ratio of bombs with adjacent miners
    return coordinated_count / len(bomb_positions)


def evaluate_piece_value_distribution(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                     player_id: int) -> float:
    """
    Reward for spreading high-value pieces across rows (not all in one row).
    Prevents clustering of strong pieces.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Distribution score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Calculate total piece value per row
    row_values = {r: 0.0 for r in player_rows}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            piece_value = PIECE_RANKS.get(piece, 0)
            row_values[r] += piece_value
    
    # Calculate variance (lower variance = better distribution)
    values = [v for v in row_values.values() if v > 0]
    if len(values) == 0:
        return 0.0
    
    mean_value = sum(values) / len(values)
    if mean_value == 0:
        return 0.0
    
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    max_variance = mean_value ** 2  # Worst case: all value in one row
    
    if max_variance == 0:
        return 1.0
    
    distribution_score = 1.0 - (variance / max_variance)
    return max(0.0, min(1.0, distribution_score))


def evaluate_strategic_positioning(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                                  player_id: int) -> float:
    """
    Reward for strategic piece positioning:
    - High-value pieces in center/back (protected)
    - Scouts in front (aggressive)
    - Bombs near flag (defensive)
    - Miners near bombs (tactical)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Strategic positioning score (0.0 to 1.0, higher is better)
    """
    score = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find flag position
    flag_pos = next((pos for piece, pos in placement if piece == PieceType.FLAG), None)
    
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        
        # High-value pieces (8+) should be in back rows
        if piece_value >= 8:
            if player_id == 1:
                if r >= 7:  # Back rows (7-9)
                    score += 0.1
            else:
                if r <= 2:  # Back rows (0-2)
                    score += 0.1
        
        # Scouts should be in front rows
        if piece == PieceType.SCOUT:
            if player_id == 1:
                if r >= 8:  # Front rows (8-9)
                    score += 0.05
            else:
                if r <= 1:  # Front rows (0-1)
                    score += 0.05
        
        # Bombs should be near flag
        if piece == PieceType.BOMB and flag_pos:
            flag_r, flag_c = flag_pos
            distance = abs(r - flag_r) + abs(c - flag_c)
            if distance <= 2:
                score += 0.1
        
        # Miners should be near bombs
        if piece == PieceType.MINER:
            for bomb_pos, bomb_piece in position_to_piece.items():
                if bomb_piece == PieceType.BOMB:
                    bomb_r, bomb_c = bomb_pos
                    distance = abs(r - bomb_r) + abs(c - bomb_c)
                    if distance <= 2:
                        score += 0.05
                        break
    
    return min(1.0, score)


def evaluate_defensive_depth(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                            player_id: int) -> float:
    """
    Reward for creating multiple defensive layers (not just one row).
    Strong pieces in multiple rows provide better defense.
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Defensive depth score (0.0 to 1.0, higher is better)
    """
    # Get rows for this player
    if player_id == 1:
        player_rows = [6, 7, 8, 9]
    else:
        player_rows = [0, 1, 2, 3]
    
    # Count strong pieces (value >= 7) per row
    row_strong_pieces = {r: 0 for r in player_rows}
    
    for piece, (r, c) in placement:
        if r in player_rows:
            piece_value = PIECE_RANKS.get(piece, 0)
            if piece_value >= 7:
                row_strong_pieces[r] += 1
    
    # Reward for having strong pieces in multiple rows
    rows_with_strong = sum(1 for count in row_strong_pieces.values() if count > 0)
    
    if rows_with_strong >= 3:
        return 1.0
    elif rows_with_strong == 2:
        return 0.6
    elif rows_with_strong == 1:
        return 0.3
    else:
        return 0.0


def evaluate_piece_synergy(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                           player_id: int) -> float:
    """
    Reward for placing pieces that synergize:
    - Marshal/General near each other (command structure)
    - Miners near bombs (defusing capability)
    - Strong pieces protecting weaker ones
    - Scouts in groups (coordination)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Synergy score (0.0 to 1.0, higher is better)
    """
    score = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    for piece, (r, c) in placement:
        # Check adjacent pieces for synergy
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = r + dr, c + dc
                adj_pos = (adj_r, adj_c)
                
                if adj_pos in position_to_piece:
                    adj_piece = position_to_piece[adj_pos]
                    
                    # Marshal/General synergy
                    if piece in [PieceType.MARSHAL, PieceType.GENERAL]:
                        if adj_piece in [PieceType.MARSHAL, PieceType.GENERAL, PieceType.COLONEL]:
                            score += 0.05
                    
                    # Miner-Bomb synergy
                    if piece == PieceType.MINER and adj_piece == PieceType.BOMB:
                        score += 0.1
                    
                    # Strong-weak protection
                    piece_value = PIECE_RANKS.get(piece, 0)
                    adj_value = PIECE_RANKS.get(adj_piece, 0)
                    if piece_value >= 8 and adj_value < 5:
                        score += 0.03
    
    return min(1.0, score)


def evaluate_vulnerability(placement: List[Tuple[PieceType, Tuple[int, int]]], 
                          player_id: int) -> float:
    """
    Penalty for vulnerable piece placements:
    - High-value pieces in front rows (exposed)
    - Flag with weak protection
    - Isolated pieces (no support)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        
    Returns:
        Vulnerability penalty score (0.0 to 1.0, higher = more vulnerable)
    """
    penalty = 0.0
    position_to_piece = {pos: piece for piece, pos in placement}
    
    # Find flag
    flag_pos = next((pos for piece, pos in placement if piece == PieceType.FLAG), None)
    
    for piece, (r, c) in placement:
        piece_value = PIECE_RANKS.get(piece, 0)
        
        # High-value pieces in front rows
        if piece_value >= 9:
            if player_id == 1:
                if r >= 8:  # Front row (8-9)
                    penalty += 0.2
            else:
                if r <= 1:  # Front row (0-1)
                    penalty += 0.2
        
        # Isolated pieces (no adjacent friendly pieces)
        adjacent_friendly = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = r + dr, c + dc
                if (adj_r, adj_c) in position_to_piece:
                    adjacent_friendly += 1
        
        if adjacent_friendly == 0 and piece_value >= 7:
            penalty += 0.1
    
    # Flag vulnerability
    if flag_pos:
        flag_r, flag_c = flag_pos
        protection_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_r, adj_c = flag_r + dr, flag_c + dc
                if (adj_r, adj_c) in position_to_piece:
                    protection_count += 1
        
        if protection_count < 2:
            penalty += 0.3
    
    return min(1.0, penalty)


def calculate_setup_agent_reward(placement: List[Tuple[PieceType, Tuple[int, int]]],
                                  player_id: int,
                                  winner: Optional[int],
                                  move_count: int,
                                  min_survival_moves: int = 100) -> float:
    """
    Calculate reward for setup agent based on:
    1. Flag protection (bombs, lakes, strong pieces)
    2. Game length (penalty for short games, reward for long games)
    3. Win/loss outcome
    4. Piece distribution (balanced placement)
    5. Scout placement (forward positions)
    6. Bomb placement (protecting key pieces)
    7. Defensive formation (pieces in lines)
    8. Piece coordination (miners near bombs)
    9. Early game survival (flag survives first 50 moves)
    
    Args:
        placement: List of (piece, position) tuples
        player_id: Player ID (1 or -1)
        winner: Winner of the game (1, -1, or None for draw)
        move_count: Number of moves in the game
        min_survival_moves: Minimum moves to avoid penalty (default 100)
        
    Returns:
        Total reward for the setup agent
    """
    reward = 0.0
    
    # Find flag position for row-based penalties
    flag_pos = None
    for piece, pos in placement:
        if piece == PieceType.FLAG:
            flag_pos = pos
            break
    
    # 0. CRITICAL: Penalty for flag in front row (very vulnerable)
    # ADJUSTED: Reduced penalty to allow learning while still discouraging bad placement
    if flag_pos is not None:
        flag_r, flag_c = flag_pos
        # Player 1's front row is row 9 (closest to enemy)
        # Player 2's front row is row 0 (closest to enemy)
        if (player_id == 1 and flag_r == 9) or (player_id == -1 and flag_r == 0):
            reward -= 0.5  # Scaled down by 10x (was -5.0, now -0.5)
        # Penalty for second row (still vulnerable)
        elif (player_id == 1 and flag_r == 8) or (player_id == -1 and flag_r == 1):
            reward -= 0.25  # Scaled down by 10x (was -2.5, now -0.25)
        # Bonus for back rows (safer)
        elif (player_id == 1 and flag_r <= 6) or (player_id == -1 and flag_r >= 3):
            reward += 0.3  # Scaled down by 10x (was 3.0, now 0.3)
    
    # 1. Flag protection reward (0.0 to 1.0, scaled to 0-0.5)
    protection_score = evaluate_flag_protection(placement, player_id)
    reward += protection_score * 0.5  # Scaled down by 10x (was 5.0, now 0.5)
    
    # 2. Game length reward/penalty (scaled down by 10x)
    if move_count < min_survival_moves:
        # Penalty for short games (games that end too quickly)
        # Linear penalty: -0.01 per move below threshold (was -0.1)
        penalty = -0.01 * (min_survival_moves - move_count)
        reward += penalty
    else:
        # Reward for surviving longer (games that last at least min_survival_moves)
        # Small reward for each move above threshold
        bonus = 0.001 * (move_count - min_survival_moves)  # Scaled down by 10x (was 0.01)
        reward += bonus
    
    # 3. Win/loss reward (scaled down by 10x)
    if winner == player_id:
        # Big reward for winning
        reward += 1.0  # Scaled down by 10x (was 10.0)
    elif winner is not None and winner != player_id:
        # Penalty for losing (but less severe than short game penalty)
        reward -= 0.2  # Scaled down by 10x (was -2.0)
    else:
        # Small reward for draw
        reward += 0.1  # Scaled down by 10x (was 1.0)
    
    # 4. Piece distribution bonus (0.0 to 1.0, scaled to 0-0.2)
    distribution_score = evaluate_piece_distribution(placement, player_id)
    reward += distribution_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 5. Scout placement reward (0.0 to 1.0, scaled to 0-0.15)
    scout_score = evaluate_scout_placement(placement, player_id)
    reward += scout_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 6. Bomb placement reward (0.0 to 1.0, scaled to 0-0.2)
    bomb_score = evaluate_bomb_placement(placement, player_id)
    reward += bomb_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 7. Defensive formation reward (0.0 to 1.0, scaled to 0-0.15)
    formation_score = evaluate_defensive_formation(placement, player_id)
    reward += formation_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 8. Piece coordination reward (0.0 to 1.0, scaled to 0-0.1)
    coordination_score = evaluate_piece_coordination(placement, player_id)
    reward += coordination_score * 0.1  # Scaled down by 10x (was 1.0)
    
    # 9. Early game survival bonus (extra reward if flag survives first 50 moves)
    if move_count >= 50:
        # Bonus for surviving early game
        reward += 0.2  # Scaled down by 10x (was 2.0)
    
    # 1. Piece Value Distribution Rewards (0.0 to 1.0, scaled to 0-0.2)
    value_distribution_score = evaluate_piece_value_distribution(placement, player_id)
    reward += value_distribution_score * 0.2  # Scaled down by 10x (was 2.0)
    
    # 2. Strategic Piece Positioning (0.0 to 1.0, scaled to 0-0.25)
    strategic_score = evaluate_strategic_positioning(placement, player_id)
    reward += strategic_score * 0.25  # Scaled down by 10x (was 2.5)
    
    # 3. Defensive Depth Rewards (0.0 to 1.0, scaled to 0-0.15)
    defensive_depth_score = evaluate_defensive_depth(placement, player_id)
    reward += defensive_depth_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 4. Piece Synergy Rewards (0.0 to 1.0, scaled to 0-0.15)
    synergy_score = evaluate_piece_synergy(placement, player_id)
    reward += synergy_score * 0.15  # Scaled down by 10x (was 1.5)
    
    # 5. Vulnerability Assessment Penalties (0.0 to 1.0, scaled to 0-0.3 penalty)
    vulnerability_penalty = evaluate_vulnerability(placement, player_id)
    reward -= vulnerability_penalty * 0.3  # Scaled down by 10x (was 3.0)
    
    return reward


def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     model_save_path: str = "dqn_models",
                     use_setup_agents: bool = True,
                     generate_gifs: bool = True):
    """
    Train two DQN agents through self-play
    
    Args:
        num_episodes: Number of training episodes
        save_interval: Interval for saving models and plots
        model_save_path: Path to save models and visualizations
        use_setup_agents: Whether to use setup agents for piece placement
        generate_gifs: Whether to generate GIFs (False to skip overhead)
    """
    REPLAY_UPDATE_INTERVAL = 4
    REPLAY_UPDATES_PER_STEP = 2
    PREFETCH_QUEUE_SIZE = 6
    TRAINING_BATCH_SIZE = 128
    TARGET_UPDATE_INTERVAL = 10000  # Update target network every 10000 steps (Increased for better stability)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimize GPU settings for better performance
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    if device.type == 'cuda':
        # Enable TensorFloat32 (TF32) for faster float32 matrix multiplication on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Enable optimizations for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("⚠️  Using CPU - training will be slower")
    
    # Load persistent total episode and step counters (survives across training runs)
    total_episodes_file = os.path.join(model_save_path, "total_episodes.txt")
    total_steps_file = os.path.join(model_save_path, "total_steps.txt")
    
    total_episodes = 0
    total_steps = 0
    
    if os.path.exists(total_episodes_file):
        try:
            with open(total_episodes_file, 'r') as f:
                total_episodes = int(f.read().strip())
            print(f"📊 Loaded total episodes counter: {total_episodes}")
        except (ValueError, IOError) as e:
            print(f"⚠️  Could not load total episodes counter: {e}, starting from 0")
            total_episodes = 0
    else:
        print(f"📊 Starting total episodes counter from 0")
    
    if os.path.exists(total_steps_file):
        try:
            with open(total_steps_file, 'r') as f:
                total_steps = int(f.read().strip())
            print(f"📊 Loaded total steps counter: {total_steps}")
        except (ValueError, IOError) as e:
            print(f"⚠️  Could not load total steps counter: {e}, starting from 0")
            total_steps = 0
    else:
        print(f"📊 Starting total steps counter from 0")
    
    # Create environment
    # Create environment
    # env = StrategoEnvironment(device=device)
    env = ParallelStrategoEnvironment(num_envs=NUM_ENVS)
    
    # Create model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Load existing training history if available (for continuity across training sessions)
    loaded_history = _load_training_history(model_save_path)
    
    # Create game-playing agents (with increased learning rate for CNN)
    agent1 = DQNAgent(player_id=1, device=device, lr=0.0001, batch_size=TRAINING_BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    agent2 = DQNAgent(player_id=-1, device=device, lr=0.0001, batch_size=TRAINING_BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    
    # Try to load the most recent saved models (separate files)
    try:
        # Find the most recent model checkpoint files (new separate format)
        agent1_dqn_files = glob.glob(os.path.join(model_save_path, "agent1_dqn_episode_*.pth"))
        agent2_dqn_files = glob.glob(os.path.join(model_save_path, "agent2_dqn_episode_*.pth"))
        
        # Also check for old combined format (backward compatibility)
        agent1_old_files = glob.glob(os.path.join(model_save_path, "agent1_episode_*.pth"))
        agent2_old_files = glob.glob(os.path.join(model_save_path, "agent2_episode_*.pth"))
        
        # Also check for final models
        final_agent1_dqn_path = os.path.join(model_save_path, "agent1_dqn_final.pth")
        final_agent2_dqn_path = os.path.join(model_save_path, "agent2_dqn_final.pth")
        final_agent1_old_path = os.path.join(model_save_path, "agent1_final.pth")
        final_agent2_old_path = os.path.join(model_save_path, "agent2_final.pth")
        
        def extract_episode(filepath):
            basename = os.path.basename(filepath)
            try:
                episode_num = int(basename.split('_episode_')[1].split('.')[0])
                return episode_num
            except:
                return 0
        
        # Determine which DQN models to load
        agent1_dqn_path = None
        agent2_dqn_path = None
        
        if agent1_dqn_files:
            agent1_dqn_files.sort(key=extract_episode, reverse=True)
            agent2_dqn_files.sort(key=extract_episode, reverse=True)
            agent1_dqn_path = agent1_dqn_files[0]
            if agent2_dqn_files:
                agent2_dqn_path = agent2_dqn_files[0]
        elif os.path.exists(final_agent1_dqn_path):
            agent1_dqn_path = final_agent1_dqn_path
            if os.path.exists(final_agent2_dqn_path):
                agent2_dqn_path = final_agent2_dqn_path
        elif agent1_old_files:
            # Fallback to old combined format
            agent1_old_files.sort(key=extract_episode, reverse=True)
            agent2_old_files.sort(key=extract_episode, reverse=True)
            agent1_dqn_path = agent1_old_files[0]
            if agent2_old_files:
                agent2_dqn_path = agent2_old_files[0]
        elif os.path.exists(final_agent1_old_path):
            agent1_dqn_path = final_agent1_old_path
            if os.path.exists(final_agent2_old_path):
                agent2_dqn_path = final_agent2_old_path
        
        # Load DQN models if found
        if agent1_dqn_path and os.path.exists(agent1_dqn_path):
            try:
                agent1.load_model(agent1_dqn_path)
                print(f"✅ Loaded Agent 1 DQN model from: {agent1_dqn_path}")
            except Exception as e:
                print(f"⚠️  Could not load Agent 1 DQN model from {agent1_dqn_path}: {e}")
                print(f"   File may be corrupted. Starting with fresh Agent 1.")
                agent1_dqn_path = None  # Mark as failed so we don't try to load PBS components
            
            # Check if old combined format has PBS components and load them (only if DQN loaded successfully)
            if agent1_dqn_path:
                try:
                    checkpoint = torch.load(agent1_dqn_path, map_location=device)
                    if 'pbs_aaren_state_dict' in checkpoint or 'pbs_lstm_state_dict' in checkpoint:
                        # Old combined format - try to load AAREN from it
                        if 'pbs_aaren_state_dict' in checkpoint:
                            agent1.pbs.aaren_model.load_state_dict(checkpoint['pbs_aaren_state_dict'])
                            if 'pbs_aaren_optimizer_state_dict' in checkpoint:
                                agent1.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_aaren_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 1 AAREN from combined checkpoint")
                        elif 'pbs_lstm_state_dict' in checkpoint:
                            agent1.pbs.aaren_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
                            if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                                agent1.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 1 AAREN (from LSTM) from combined checkpoint")
                    
                    if 'pbs_evaluator_state_dict' in checkpoint and agent1.pbs.evaluator is not None:
                        agent1.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
                        agent1.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
                        if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
                            agent1.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
                        agent1.pbs.evaluator.update_target_network()
                        print(f"✅ Loaded Agent 1 PBS Evaluator from combined checkpoint")
                except Exception as e:
                    # Not a combined checkpoint or error loading PBS components - that's okay
                    pass
        
        if agent2_dqn_path and os.path.exists(agent2_dqn_path):
            try:
                agent2.load_model(agent2_dqn_path)
                print(f"✅ Loaded Agent 2 DQN model from: {agent2_dqn_path}")
            except Exception as e:
                print(f"⚠️  Could not load Agent 2 DQN model from {agent2_dqn_path}: {e}")
                print(f"   File may be corrupted. Starting with fresh Agent 2.")
                agent2_dqn_path = None  # Mark as failed so we don't try to load PBS components
            
            # Check if old combined format has PBS components and load them (only if DQN loaded successfully)
            if agent2_dqn_path:
                try:
                    checkpoint = torch.load(agent2_dqn_path, map_location=device)
                    if 'pbs_aaren_state_dict' in checkpoint or 'pbs_lstm_state_dict' in checkpoint:
                        # Old combined format - try to load AAREN from it
                        if 'pbs_aaren_state_dict' in checkpoint:
                            agent2.pbs.aaren_model.load_state_dict(checkpoint['pbs_aaren_state_dict'])
                            if 'pbs_aaren_optimizer_state_dict' in checkpoint:
                                agent2.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_aaren_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 2 AAREN from combined checkpoint")
                        elif 'pbs_lstm_state_dict' in checkpoint:
                            agent2.pbs.aaren_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
                            if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                                agent2.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 2 AAREN (from LSTM) from combined checkpoint")
                    
                    if 'pbs_evaluator_state_dict' in checkpoint and agent2.pbs.evaluator is not None:
                        agent2.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
                        agent2.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
                        if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
                            agent2.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
                        agent2.pbs.evaluator.update_target_network()
                        print(f"✅ Loaded Agent 2 PBS Evaluator from combined checkpoint")
                except Exception as e:
                    # Not a combined checkpoint or error loading PBS components - that's okay
                    pass
        
        # Load AAREN models if found (separate files)
        agent1_aaren_files = glob.glob(os.path.join(model_save_path, "agent1_aaren_episode_*.pth"))
        agent2_aaren_files = glob.glob(os.path.join(model_save_path, "agent2_aaren_episode_*.pth"))
        final_agent1_aaren_path = os.path.join(model_save_path, "agent1_aaren_final.pth")
        final_agent2_aaren_path = os.path.join(model_save_path, "agent2_aaren_final.pth")
        
        agent1_aaren_path = None
        agent2_aaren_path = None
        
        if agent1_aaren_files:
            agent1_aaren_files.sort(key=extract_episode, reverse=True)
            agent2_aaren_files.sort(key=extract_episode, reverse=True)
            agent1_aaren_path = agent1_aaren_files[0]
            if agent2_aaren_files:
                agent2_aaren_path = agent2_aaren_files[0]
        elif os.path.exists(final_agent1_aaren_path):
            agent1_aaren_path = final_agent1_aaren_path
            if os.path.exists(final_agent2_aaren_path):
                agent2_aaren_path = final_agent2_aaren_path
        
        if agent1_aaren_path and os.path.exists(agent1_aaren_path):
            try:
                agent1.load_aaren_model(agent1_aaren_path)
            except Exception as e:
                print(f"⚠️  Could not load Agent 1 AAREN model: {e}")
        
        if agent2_aaren_path and os.path.exists(agent2_aaren_path):
            try:
                agent2.load_aaren_model(agent2_aaren_path)
            except Exception as e:
                print(f"⚠️  Could not load Agent 2 AAREN model: {e}")
        
        # Load PBS Evaluator models if found (separate files)
        agent1_evaluator_files = glob.glob(os.path.join(model_save_path, "agent1_pbs_evaluator_episode_*.pth"))
        agent2_evaluator_files = glob.glob(os.path.join(model_save_path, "agent2_pbs_evaluator_episode_*.pth"))
        final_agent1_evaluator_path = os.path.join(model_save_path, "agent1_pbs_evaluator_final.pth")
        final_agent2_evaluator_path = os.path.join(model_save_path, "agent2_pbs_evaluator_final.pth")
        
        agent1_evaluator_path = None
        agent2_evaluator_path = None
        
        if agent1_evaluator_files:
            agent1_evaluator_files.sort(key=extract_episode, reverse=True)
            agent2_evaluator_files.sort(key=extract_episode, reverse=True)
            agent1_evaluator_path = agent1_evaluator_files[0]
            if agent2_evaluator_files:
                agent2_evaluator_path = agent2_evaluator_files[0]
        elif os.path.exists(final_agent1_evaluator_path):
            agent1_evaluator_path = final_agent1_evaluator_path
            if os.path.exists(final_agent2_evaluator_path):
                agent2_evaluator_path = final_agent2_evaluator_path
        
        if agent1_evaluator_path and os.path.exists(agent1_evaluator_path):
            try:
                agent1.load_pbs_evaluator(agent1_evaluator_path)
            except Exception as e:
                print(f"⚠️  Could not load Agent 1 PBS Evaluator model: {e}")
        
        if agent2_evaluator_path and os.path.exists(agent2_evaluator_path):
            try:
                agent2.load_pbs_evaluator(agent2_evaluator_path)
            except Exception as e:
                print(f"⚠️  Could not load Agent 2 PBS Evaluator model: {e}")
                
    except Exception as e:
        print(f"⚠️  Could not load saved models: {e}")
        print("   Starting with fresh agents")
        traceback.print_exc()
    
    
    # Ensure learning rates are not too low (reset if decayed too much)
    # This prevents agents from being "stuck" if the LR was crushed in a previous run
    # CRITICAL FIX: Re-initialize optimizers completely to clear bad momentum/variance state
    print("🧹 Re-initializing optimizers to clear potential 'stuck' states...")
    for agent in [agent1, agent2]:
        # Keep the weights, but get a fresh optimizer
        agent.optimizer = torch.optim.AdamW(agent.q_network.parameters(), lr=0.0001, weight_decay=0.01)
        print(f"   Re-initialized optimizer for {agent.name}")

    # Create setup agents (for piece placement)
    setup_agent1 = SetupAgent(player_id=1, device=device) if use_setup_agents else None
    setup_agent2 = SetupAgent(player_id=-1, device=device) if use_setup_agents else None
    
    # Try to load setup agent models if available
    if use_setup_agents and setup_agent1 and setup_agent2:
        try:
            setup_agent1_files = glob.glob(os.path.join(model_save_path, "setup_agent1_episode_*.pth"))
            setup_agent2_files = glob.glob(os.path.join(model_save_path, "setup_agent2_episode_*.pth"))
            
            final_setup_agent1_path = os.path.join(model_save_path, "setup_agent1_final.pth")
            final_setup_agent2_path = os.path.join(model_save_path, "setup_agent2_final.pth")
            
            setup_agent1_path = None
            setup_agent2_path = None
            
            if setup_agent1_files:
                def extract_episode(filepath):
                    basename = os.path.basename(filepath)
                    try:
                        episode_num = int(basename.split('_episode_')[1].split('.')[0])
                        return episode_num
                    except:
                        return 0
                
                setup_agent1_files.sort(key=extract_episode, reverse=True)
                setup_agent2_files.sort(key=extract_episode, reverse=True)
                setup_agent1_path = setup_agent1_files[0]
                if setup_agent2_files:
                    setup_agent2_path = setup_agent2_files[0]
            elif os.path.exists(final_setup_agent1_path):
                setup_agent1_path = final_setup_agent1_path
                if os.path.exists(final_setup_agent2_path):
                    setup_agent2_path = final_setup_agent2_path
            
            if setup_agent1_path and os.path.exists(setup_agent1_path):
                setup_agent1.load_model(setup_agent1_path)
                print(f"✅ Loaded Setup Agent 1 model from: {setup_agent1_path}")
            
            if setup_agent2_path and os.path.exists(setup_agent2_path):
                setup_agent2.load_model(setup_agent2_path)
                print(f"✅ Loaded Setup Agent 2 model from: {setup_agent2_path}")
        except Exception as e:
            error_str = str(e)
            # Check if this is an architecture mismatch (expected when model architecture changes)
            is_architecture_mismatch = (
                "Missing key" in error_str or 
                "Unexpected key" in error_str or 
                "size mismatch" in error_str
            )
            
            if is_architecture_mismatch:
                print(f"ℹ️  Saved setup agent models use an older architecture and cannot be loaded.")
                print(f"   This is expected when the model architecture has been updated.")
                print(f"   Starting with fresh setup agents (training will continue from scratch).")
            else:
                print(f"⚠️  Could not load saved setup agent models: {e}")
                print("   Starting with fresh setup agents")
    
    # Track GIF creation flags
    winning_game_pbs_gif_count = 0  # Track number of winning game PBS GIFs created (max 5)
    winning_game_gif_created = False  # Track if winning game (non-PBS) GIF was created
    
    # History for plotting (load from previous session or start fresh)
    if loaded_history:
        episode_history = loaded_history.get('episode_history', [])
        rewards_history = loaded_history.get('rewards_history', {'agent1': [], 'agent2': []})
        wins_history = loaded_history.get('wins_history', {'agent1': [], 'agent2': [], 'draws': []})
        epsilon_history = loaded_history.get('epsilon_history', {'agent1': [], 'agent2': []})
        policy_loss_history = loaded_history.get('policy_loss_history', {'agent1': [], 'agent2': []})
        setup_agent1_rewards = loaded_history.get('setup_agent1_rewards', [])
        setup_agent2_rewards = loaded_history.get('setup_agent2_rewards', [])
        setup_agent1_losses = loaded_history.get('setup_agent1_losses', [])
        setup_agent2_losses = loaded_history.get('setup_agent2_losses', [])
        pbs_evaluator1_losses = loaded_history.get('pbs_evaluator1_losses', [])
        pbs_evaluator2_losses = loaded_history.get('pbs_evaluator2_losses', [])
        pbs_evaluator1_buffer_sizes = loaded_history.get('pbs_evaluator1_buffer_sizes', [])
        pbs_evaluator1_buffer_sizes = loaded_history.get('pbs_evaluator1_buffer_sizes', [])
        pbs_evaluator2_buffer_sizes = loaded_history.get('pbs_evaluator2_buffer_sizes', [])
        avg_q_history = loaded_history.get('avg_q_history', {'agent1': [], 'agent2': []})
        entropy_history = loaded_history.get('entropy_history', {'agent1': [], 'agent2': []})
        
        # Initialize win counters from loaded history (use last values for continuity)
        if wins_history and len(wins_history.get('agent1', [])) > 0:
            wins_agent1 = wins_history['agent1'][-1]
        else:
            wins_agent1 = 0
        
        if wins_history and len(wins_history.get('agent2', [])) > 0:
            wins_agent2 = wins_history['agent2'][-1]
        else:
            wins_agent2 = 0
        
        if wins_history and len(wins_history.get('draws', [])) > 0:
            draws = wins_history['draws'][-1]
        else:
            draws = 0
        
        # Initialize reward lists from loaded history
        total_rewards_agent1 = rewards_history.get('agent1', [])[-50:] if rewards_history.get('agent1') else []
        total_rewards_agent2 = rewards_history.get('agent2', [])[-50:] if rewards_history.get('agent2') else []
        
        # Align all history lists to the same length as episode_history
        # This fixes issues where some lists might be shorter due to incomplete saves or different tracking
        target_length = len(episode_history)
        if target_length > 0:
            # Pad shorter lists with their last value (or 0.0 if empty)
            def pad_list(lst, target_len, default=0.0):
                if len(lst) == target_len:
                    return lst
                elif len(lst) > target_len:
                    return lst[:target_len]  # Truncate if longer
                else:
                    # Pad with last value or default
                    pad_value = lst[-1] if len(lst) > 0 else default
                    return lst + [pad_value] * (target_len - len(lst))
            
            setup_agent1_rewards = pad_list(setup_agent1_rewards, target_length, 0.0)
            setup_agent2_rewards = pad_list(setup_agent2_rewards, target_length, 0.0)
            setup_agent1_losses = pad_list(setup_agent1_losses, target_length, 0.0)
            setup_agent2_losses = pad_list(setup_agent2_losses, target_length, 0.0)
            pbs_evaluator1_losses = pad_list(pbs_evaluator1_losses, target_length, 0.0)
            pbs_evaluator2_losses = pad_list(pbs_evaluator2_losses, target_length, 0.0)
            pbs_evaluator1_buffer_sizes = pad_list(pbs_evaluator1_buffer_sizes, target_length, 0)
            pbs_evaluator2_buffer_sizes = pad_list(pbs_evaluator2_buffer_sizes, target_length, 0)
            
            # Align dictionary-based histories
            for key in rewards_history:
                rewards_history[key] = pad_list(rewards_history[key], target_length, 0.0)
            for key in wins_history:
                # Explicitly use the last value as default for wins to prevent reset
                last_val = wins_history[key][-1] if wins_history[key] else 0
                wins_history[key] = pad_list(wins_history[key], target_length, last_val)
            for key in epsilon_history:
                epsilon_history[key] = pad_list(epsilon_history[key], target_length, 1.0)
            for key in policy_loss_history:
                policy_loss_history[key] = pad_list(policy_loss_history[key], target_length, 0.0)
            for key in avg_q_history:
                avg_q_history[key] = pad_list(avg_q_history[key], target_length, 0.0)
            for key in entropy_history:
                entropy_history[key] = pad_list(entropy_history[key], target_length, 0.0)
        
        print(f"📈 Continuing training history from previous session")
        print(f"   Last episode: {episode_history[-1] if episode_history else 0}")
        print(f"   Wins (Agent1/Agent2/Draws): {wins_agent1}/{wins_agent2}/{draws}")
        print(f"   Win Counts Initialized to: Agent1={wins_agent1}, Agent2={wins_agent2}")
        
        # Sync total_episodes with history if needed
        if episode_history and episode_history[-1] > total_episodes:
            print(f"⚠️  Total episodes counter ({total_episodes}) is behind history ({episode_history[-1]}). Syncing...")
            total_episodes = episode_history[-1]
    else:
        episode_history = []
        rewards_history = {'agent1': [], 'agent2': []}
        wins_history = {'agent1': [], 'agent2': [], 'draws': []}
        epsilon_history = {'agent1': [], 'agent2': []}
        policy_loss_history = {'agent1': [], 'agent2': []}
        setup_agent1_rewards = []
        setup_agent2_rewards = []
        setup_agent1_losses = []
        setup_agent2_losses = []
        pbs_evaluator1_losses = []
        pbs_evaluator2_losses = []
        pbs_evaluator1_buffer_sizes = []
        pbs_evaluator1_buffer_sizes = []
        pbs_evaluator2_buffer_sizes = []
        avg_q_history = {'agent1': [], 'agent2': []}
        entropy_history = {'agent1': [], 'agent2': []}
        
        # Initialize win counters to 0 for fresh start
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        total_rewards_agent1 = []
        total_rewards_agent2 = []
        
        print(f"📈 Starting fresh training history")
    
    # Initialize loss lists (transient for current run)
    agent1_losses = []
    agent2_losses = []
    
    # agent1_prefetcher = ReplayPrefetcher(agent1, max_queue_size=PREFETCH_QUEUE_SIZE)
    # agent2_prefetcher = ReplayPrefetcher(agent2, max_queue_size=PREFETCH_QUEUE_SIZE)
    # prefetchers = [agent1_prefetcher, agent2_prefetcher]
    prefetchers = []
    
    print(f"Starting DQN training for {num_episodes} episodes...")
    print(f"Total Episodes (all runs): {total_episodes}")
    print(f"Total Steps (all runs): {total_steps}")
    print("=" * 60)
    
    # Initialize parallel environment
    # env is already initialized as ParallelStrategoEnvironment
    
    # Reset all environments initially
    print("Resetting all environments...")
    
    p1_placements = []
    p2_placements = []
    
    if use_setup_agents and setup_agent1 and setup_agent2:
        # Generate placements for all envs
        # Note: This is sequential, could be parallelized but happens only at start/reset
        for i in range(NUM_ENVS):
            # ParallelEnv doesn't expose _generate_pieces easily. 
            # We can just use random placement if setup agent fails or just let env reset randomly first?
            # Actually, we can just pass None and let env random reset, then use setup agents for subsequent resets.
            p1_placements.append(None)
            p2_placements.append(None)
    
    states_tuple, rewards, dones, infos, valid_moves_tuple = env.reset()
    # Convert tuples to lists for mutability
    states = list(states_tuple)
    valid_moves = list(valid_moves_tuple)
    
    # Track episode stats
    episode_rewards_agent1 = [0.0] * NUM_ENVS
    episode_rewards_agent2 = [0.0] * NUM_ENVS
    episode_moves = [0] * NUM_ENVS
    
    # Track pending resets
    pending_resets = [False] * NUM_ENVS
    
    # Track setup agent placements for reward calculation
    placement_memory = {}  # env_index -> {'p1_placement': ..., 'p2_placement': ..., 'episode_start': ...}
    
    # Main training loop
    target_total_episodes = total_episodes + num_episodes
    pbar = tqdm(total=num_episodes, desc="Training Episodes")
    
    # Track completed episodes in this run
    completed_episodes = 0
    
    # Track last saved episode to prevent saving multiple times for the same episode
    last_saved_episode = total_episodes - (total_episodes % save_interval) if total_episodes > 0 else -save_interval
    
    # Track last plotted episode to prevent duplicate plot generation in parallel training
    last_plotted_episode = -50  # Initialize to -50 so first plot at episode 50 works
    
    def save_checkpoint(episode_num, is_final=False):
        """Helper function to save all models"""
        suffix = "final" if is_final else f"episode_{episode_num}"
        print(f"\n💾 Saving models (Episode {episode_num}, Final={is_final})...")
        
        os.makedirs(model_save_path, exist_ok=True)
        
        try:
            # Save DQN models
            agent1.save_model(f"{model_save_path}/agent1_dqn_{suffix}.pth")
            agent2.save_model(f"{model_save_path}/agent2_dqn_{suffix}.pth")
            
            # Save AAREN models
            try:
                agent1.save_aaren_model(f"{model_save_path}/agent1_aaren_{suffix}.pth")
                agent2.save_aaren_model(f"{model_save_path}/agent2_aaren_{suffix}.pth")
            except Exception as e:
                print(f"⚠️  Warning: Could not save AAREN models: {e}")
                
            # Save PBS Evaluator models
            try:
                agent1.save_pbs_evaluator(f"{model_save_path}/agent1_pbs_evaluator_{suffix}.pth")
                agent2.save_pbs_evaluator(f"{model_save_path}/agent2_pbs_evaluator_{suffix}.pth")
            except Exception as e:
                print(f"⚠️  Warning: Could not save PBS Evaluator models: {e}")
                
            # Save setup agents
            if use_setup_agents and setup_agent1 and setup_agent2:
                setup_agent1.save_model(f"{model_save_path}/setup_agent1_{suffix}.pth")
                setup_agent2.save_model(f"{model_save_path}/setup_agent2_{suffix}.pth")
                
            print(f"✅ Models saved successfully.")
            
        except Exception as e:
            print(f"⚠️  Error saving models: {e}")
            traceback.print_exc()

    try:
        while completed_episodes < num_episodes:
            # 1. Determine actions for active environments
            actions_list = [None] * NUM_ENVS
        
            # Identify which agent acts in which env
            agent1_indices = []
            agent2_indices = []
        
            for i in range(NUM_ENVS):
                if not pending_resets[i]:
                    if states[i].current_player == 1:
                        agent1_indices.append(i)
                    else:
                        agent2_indices.append(i)
        
            # Get actions for Agent 1
            if agent1_indices:
                batch_states = [states[i] for i in agent1_indices]
                batch_valid_moves = [valid_moves[i] for i in agent1_indices]
                # Pass game_states as states (since states are GameState objects)
                batch_actions = agent1.act_batch(batch_states, batch_valid_moves, game_states=batch_states)
                for idx, action in zip(agent1_indices, batch_actions):
                    actions_list[idx] = action
                
            # Get actions for Agent 2
            if agent2_indices:
                batch_states = [states[i] for i in agent2_indices]
                batch_valid_moves = [valid_moves[i] for i in agent2_indices]
                batch_actions = agent2.act_batch(batch_states, batch_valid_moves, game_states=batch_states)
                for idx, action in zip(agent2_indices, batch_actions):
                    actions_list[idx] = action
        
            # 2. Update PBS (Cross-update: Agent 1's action updates Agent 2's PBS)
            if agent1_indices:
                 update_actions = [actions_list[i] for i in agent1_indices]
                 update_states = [states[i] for i in agent1_indices]
                 agent2.update_pbs_batch(update_actions, update_states, acting_player=1)
             
            if agent2_indices:
                 update_actions = [actions_list[i] for i in agent2_indices]
                 update_states = [states[i] for i in agent2_indices]
                 agent1.update_pbs_batch(update_actions, update_states, acting_player=-1)
             
            # 3. Prepare commands (Actions or Resets)
            commands = []
            for i in range(NUM_ENVS):
                if pending_resets[i]:
                    # Generate placement for reset
                    p1_place = None
                    p2_place = None
                    if use_setup_agents and setup_agent1 and setup_agent2:
                        # Generate pieces list (standard Stratego composition)
                        pieces_p1 = [PieceType.FLAG, PieceType.SPY] + [PieceType.BOMB]*6 + [PieceType.MARSHAL] + \
                                   [PieceType.GENERAL] + [PieceType.COLONEL]*2 + [PieceType.MAJOR]*3 + \
                                   [PieceType.CAPTAIN]*4 + [PieceType.LIEUTENANT]*4 + [PieceType.SERGEANT]*4 + \
                                   [PieceType.MINER]*5 + [PieceType.SCOUT]*8
                        pieces_p2 = pieces_p1.copy()  # Same composition for both players
                    
                        # Get available positions for each player
                        # Player 1: rows 6-9, Player 2: rows 0-3 (excluding lakes)
                        lakes = [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]
                        available_p1 = [(r, c) for r in range(6, 10) for c in range(10) if (r, c) not in lakes]
                        available_p2 = [(r, c) for r in range(0, 4) for c in range(10) if (r, c) not in lakes]
                    
                        # Use setup agents to place pieces
                        p1_place = setup_agent1.place_pieces(pieces_p1, available_p1)
                        p2_place = setup_agent2.place_pieces(pieces_p2, available_p2)
                    
                        # Store placements for later reward calculation
                        placement_memory[i] = {
                            'p1_placement': p1_place,
                            'p2_placement': p2_place,
                            'episode_start': total_episodes
                        }
                    
                    commands.append(('reset', {'p1_placement': p1_place, 'p2_placement': p2_place}))
                else:
                    commands.append(actions_list[i])
                
            # 4. Step environment
            next_states_tuple, step_rewards, step_dones, step_infos, next_valid_moves_tuple = env.step(commands)
            # Convert tuples to lists for mutability
            next_states = list(next_states_tuple)
            next_valid_moves = list(next_valid_moves_tuple)
        
            # 5. Process results
            for i in range(NUM_ENVS):
                if pending_resets[i]:
                    # Just reset, update state and continue
                    states[i] = next_states[i] # New initial state
                    valid_moves[i] = next_valid_moves[i]
                    pending_resets[i] = False
                    episode_rewards_agent1[i] = 0.0
                    episode_rewards_agent2[i] = 0.0
                    episode_moves[i] = 0
                    continue
                
                # It was a step
                action = actions_list[i]
                if action is None: # Should not happen if logic is correct
                    continue
                
                reward = step_rewards[i].item()
                done = step_dones[i].item()
            
                # Determine agents
                player = states[i].current_player
                current_agent = agent1 if player == 1 else agent2
                opponent_agent = agent2 if player == 1 else agent1
            
                # Get next state representation for storage
                # We need to convert GameState to tensor
                # We can use the single-item method since we are iterating
                # Or we could have batched this before loop.
                # For simplicity/correctness, do it here.
                state_tensor = current_agent.get_state_representation(states[i])
                next_state_tensor = current_agent.get_state_representation(next_states[i])
            
                # Store experience
                current_agent.remember(state_tensor, 
                                     current_agent._move_to_action_index(action),
                                     reward,
                                     next_state_tensor,
                                     done)
            
                # Update stats
                if player == 1:
                    episode_rewards_agent1[i] += reward
                else:
                    episode_rewards_agent2[i] += reward
                episode_moves[i] += 1
            
                # Handle reveals (PBS)
                # We need to check if pieces were revealed.
                # ParallelEnv step info might contain reveals?
                # We need to ensure ParallelEnv passes 'revealed_pieces' in info.
                # Currently it returns info dict.
                # We should check info[i].
                if 'revealed_pieces_p1' in step_infos[i]:
                     for pos, val in step_infos[i]['revealed_pieces_p1'].items():
                         agent1.update_pbs_from_reveal_batch([[(pos, PieceType(abs(val)))]], game_phase='middle', turn_count=episode_moves[i])
                if 'revealed_pieces_p2' in step_infos[i]:
                     for pos, val in step_infos[i]['revealed_pieces_p2'].items():
                         agent2.update_pbs_from_reveal_batch([[(pos, PieceType(abs(val)))]], game_phase='middle', turn_count=episode_moves[i])

                # Handle Done
                if done:
                    pending_resets[i] = True
                    completed_episodes += 1
                    total_episodes += 1
                    pbar.update(1)
                
                    # Update global stats
                    total_rewards_agent1.append(episode_rewards_agent1[i])
                    total_rewards_agent2.append(episode_rewards_agent2[i])
                
                    # Determine winner
                    winner = step_infos[i].get('winner', 0)
                    if winner == 1:
                        wins_agent1 += 1
                    elif winner == -1:
                        wins_agent2 += 1
                    else:
                        draws += 1
                
                    # ============================================
                    # CONTINUOUS METRIC TRACKING (EVERY EPISODE)
                    # ============================================
                    # Track metrics for every completed episode, not just when plotting
                    episode_history.append(total_episodes)
                
                    # Record episode rewards
                    rewards_history['agent1'].append(episode_rewards_agent1[i])
                    rewards_history['agent2'].append(episode_rewards_agent2[i])
                
                    # Record cumulative win counts
                    wins_history['agent1'].append(wins_agent1)
                    wins_history['agent2'].append(wins_agent2)
                    wins_history['draws'].append(draws)
                
                    # Record epsilon values
                    epsilon_history['agent1'].append(agent1.epsilon)
                    epsilon_history['agent2'].append(agent2.epsilon)
                
                    # Record recent average policy losses
                    recent_window = 10  # Average over last 10 training steps
                    avg_loss_agent1 = np.mean(agent1_losses[-recent_window:]) if len(agent1_losses) >= recent_window else (np.mean(agent1_losses) if agent1_losses else 0.0)
                    avg_loss_agent2 = np.mean(agent2_losses[-recent_window:]) if len(agent2_losses) >= recent_window else (np.mean(agent2_losses) if agent2_losses else 0.0)
                    policy_loss_history['agent1'].append(avg_loss_agent1)
                    policy_loss_history['agent2'].append(avg_loss_agent2)
                
                    # Record PBS evaluator metrics
                    pbs_eval_loss_1 = agent1.pbs.evaluator.get_average_loss() if agent1.pbs.evaluator else 0.0
                    pbs_eval_loss_2 = agent2.pbs.evaluator.get_average_loss() if agent2.pbs.evaluator else 0.0
                    pbs_evaluator1_losses.append(pbs_eval_loss_1)
                    pbs_evaluator2_losses.append(pbs_eval_loss_2)
                    pbs_evaluator1_buffer_sizes.append(len(agent1.pbs.evaluator.memory) if agent1.pbs.evaluator else 0)
                    pbs_evaluator2_buffer_sizes.append(len(agent2.pbs.evaluator.memory) if agent2.pbs.evaluator else 0)
                
                    # Record Average Q-Value and Entropy
                    avg_q_history['agent1'].append(agent1.get_average_q_value())
                    avg_q_history['agent2'].append(agent2.get_average_q_value())
                    entropy_history['agent1'].append(agent1.get_average_entropy())
                    entropy_history['agent2'].append(agent2.get_average_entropy())
                
                    # ============================================
                    # SETUP AGENT TRAINING
                    # ============================================
                    # Train setup agents if placements were generated by them
                    if i in placement_memory and use_setup_agents and setup_agent1 and setup_agent2:
                        # Calculate setup rewards based on game outcome
                        setup_reward_1 = calculate_setup_agent_reward(
                            placement_memory[i]['p1_placement'],
                            player_id=1,
                            winner=winner,
                            move_count=episode_moves[i]
                        )
                        setup_reward_2 = calculate_setup_agent_reward(
                            placement_memory[i]['p2_placement'],
                            player_id=-1,
                            winner=winner,
                            move_count=episode_moves[i]
                        )
                    
                        # Apply rewards to setup agent episode memory and store in replay buffer
                        setup_agent1.finish_episode(setup_reward_1)
                        setup_agent2.finish_episode(setup_reward_2)
                    
                        # Train setup agents
                        setup_loss_1 = setup_agent1.replay()
                        setup_loss_2 = setup_agent2.replay()
                    
                        # Track setup agent performance for plotting
                        setup_agent1_rewards.append(setup_reward_1)
                        setup_agent2_rewards.append(setup_reward_2)
                        # Always append loss values (0 if training didn't happen)
                        setup_agent1_losses.append(setup_loss_1 if setup_loss_1 is not None else 0.0)
                        setup_agent2_losses.append(setup_loss_2 if setup_loss_2 is not None else 0.0)
                    
                        # Clean up placement memory after training
                        del placement_memory[i]
                    else:
                        # No setup agent data for this episode - append placeholders to maintain length match
                        if use_setup_agents:
                            setup_agent1_rewards.append(0.0)
                            setup_agent2_rewards.append(0.0)
                            setup_agent1_losses.append(0.0)
                            setup_agent2_losses.append(0.0)
                
                    # ============================================
                    # GENERATE PLOTS EVERY 50 EPISODES
                    # ============================================
                    # Prevent duplicate plot generation in parallel environment by checking last_plotted_episode
                    if total_episodes % 50 == 0 and total_episodes > last_plotted_episode:
                        # Generate training progress plots
                        try:
                            if len(episode_history) > 0:
                                # Plot DQN agent training progress
                                plot_path = f"{model_save_path}/training_progress_episode_{total_episodes}.png"
                                plot_training_progress(
                                    episode_history,
                                    rewards_history,
                                    wins_history,
                                    policy_loss_history,
                                    plot_path,
                                    total_episodes=total_episodes,
                                    total_steps=total_steps
                                )
                                print(f"📊 Training progress plot saved: {plot_path}")
                            
                                # Plot setup agent progress (if using setup agents and have data)
                                if use_setup_agents and len(setup_agent1_rewards) > 0:
                                    setup_plot_path = f"{model_save_path}/setup_agent_progress_episode_{total_episodes}.png"
                                    plot_setup_agent_progress(
                                        episode_history,
                                        setup_agent1_rewards,
                                        setup_agent2_rewards,
                                        setup_agent1_losses,
                                        setup_agent2_losses,
                                        setup_plot_path
                                    )
                                    print(f"📊 Setup agent progress plot saved: {setup_plot_path}")
                            
                                # Plot PBS evaluator progress
                                if len(pbs_evaluator1_losses) > 0:
                                    pbs_plot_path = f"{model_save_path}/pbs_evaluator_progress_episode_{total_episodes}.png"
                                    plot_pbs_evaluator_progress(
                                        episode_history,
                                        pbs_evaluator1_losses,
                                        pbs_evaluator2_losses,
                                        pbs_evaluator1_buffer_sizes,
                                        pbs_evaluator2_buffer_sizes,
                                        pbs_plot_path,
                                        total_episodes=total_episodes
                                    )
                                    print(f"📊 PBS evaluator progress plot saved: {pbs_plot_path}")
                                
                                # Plot additional metrics (Epsilon, Buffer Size, Q-Value, Entropy)
                                additional_metrics_path = f"{model_save_path}/additional_metrics_episode_{total_episodes}.png"
                                plot_additional_metrics(
                                    episode_history,
                                    epsilon_history,
                                    {'agent1': pbs_evaluator1_buffer_sizes, 'agent2': pbs_evaluator2_buffer_sizes},
                                    avg_q_history,
                                    entropy_history,
                                    additional_metrics_path
                                )
                                print(f"📊 Additional metrics plot saved: {additional_metrics_path}")
                                
                                # Update last_plotted_episode to prevent duplicate plotting
                                last_plotted_episode = total_episodes
                                
                                # Save training history JSON for continuity
                                try:
                                    _save_training_history(
                                        model_save_path,
                                        episode_history,
                                        rewards_history,
                                        wins_history,
                                        epsilon_history,
                                        policy_loss_history,
                                        setup_agent1_rewards,
                                        setup_agent2_rewards,
                                        setup_agent1_losses,
                                        setup_agent2_losses,
                                        pbs_evaluator1_losses,
                                        pbs_evaluator2_losses,
                                        pbs_evaluator1_buffer_sizes,
                                        pbs_evaluator2_buffer_sizes,
                                        avg_q_history,
                                        entropy_history
                                    )
                                    print(f"💾 Training history JSON saved at episode {total_episodes}")
                                except Exception as json_err:
                                    print(f"⚠️  Could not save training history JSON: {json_err}")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not generate plots at episode {total_episodes}: {e}")
                            traceback.print_exc()
                     
                else:
                    states[i] = next_states[i]
                    valid_moves[i] = next_valid_moves[i]
                
            # 6. Train Agents (Batched)
            # Train every REPLAY_UPDATE_INTERVAL steps
            # Since we process NUM_ENVS steps per iteration, we train more frequently naturally
            if total_steps % REPLAY_UPDATE_INTERVAL == 0:
                 for _ in range(REPLAY_UPDATES_PER_STEP):
                     loss1 = agent1.replay()
                     if loss1 is not None:
                         agent1_losses.append(loss1)
                 
                     loss2 = agent2.replay()
                     if loss2 is not None:
                         agent2_losses.append(loss2)
                     
                     # Train PBS Evaluator
                     eval_loss1 = agent1.train_pbs_evaluator()
                     if eval_loss1 is not None:
                         pbs_evaluator1_losses.append(eval_loss1)
                     
                     eval_loss2 = agent2.train_pbs_evaluator()
                     if eval_loss2 is not None:
                         pbs_evaluator2_losses.append(eval_loss2)
                 
                 # DEBUG: Print status every 100 training steps (approx 400 env steps)
                 if (total_steps // REPLAY_UPDATE_INTERVAL) % 100 == 0:
                     print(f"\n🔍 Diagnostics (Step {total_steps}):")
                     print(f"  Agent 1: Mem={len(agent1.memory)}, Loss={agent1_losses[-1] if agent1_losses else 'N/A'}, LR={agent1.optimizer.param_groups[0]['lr']:.2e}")
                     print(f"  Agent 2: Mem={len(agent2.memory)}, Loss={agent2_losses[-1] if agent2_losses else 'N/A'}, LR={agent2.optimizer.param_groups[0]['lr']:.2e}")
                     if len(agent2.memory) < agent2.batch_size:
                         print(f"  ⚠️ Agent 2 memory too low to train (< {agent2.batch_size})")
                     if agent2_losses and agent2_losses[-1] == 0.0:
                         print(f"  ⚠️ Agent 2 loss is exactly 0.0!")

                 
            # Update target networks
            # Update every TARGET_UPDATE_INTERVAL steps
            if total_steps % TARGET_UPDATE_INTERVAL == 0:
                 agent1.update_target_network()
                 agent2.update_target_network()
             
            total_steps += NUM_ENVS # Approx
            
            # Save models periodically (keep at save_interval)
            # Only save when we cross a new save interval threshold
            # Save models periodically (keep at save_interval)
            # Only save when we cross a new save interval threshold
            if total_episodes > 0 and total_episodes % save_interval == 0 and total_episodes > last_saved_episode:
                save_checkpoint(total_episodes)
                # Update last saved episode to prevent saving again
                last_saved_episode = total_episodes
            
        # Save final persistent counters
        try:
            with open(total_episodes_file, 'w') as f:
                f.write(str(total_episodes))
            with open(total_steps_file, 'w') as f:
                f.write(str(total_steps))
            print(f"💾 Saved persistent counters: {total_episodes} episodes, {total_steps:,} steps")
        except Exception as e:
            print(f"⚠️  Could not save persistent counters: {e}")
    
        # Save final training history for continuity
        try:
            _save_training_history(
                model_save_path,
                episode_history,
                rewards_history,
                wins_history,
                epsilon_history,
                policy_loss_history,
                setup_agent1_rewards,
                setup_agent2_rewards,
                setup_agent1_losses,
                setup_agent2_losses,
                pbs_evaluator1_losses,
                pbs_evaluator2_losses,
                pbs_evaluator1_buffer_sizes,
                pbs_evaluator2_buffer_sizes,
                avg_q_history,
                entropy_history
            )
            print(f"💾 Saved final training history for continuity")
        except Exception as e:
            print(f"⚠️  Could not save final training history: {e}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user! Saving current state...")
        save_checkpoint(total_episodes)
        raise

    finally:
        # Ensure history is saved even if interrupted
        try:
            _save_training_history(
                model_save_path,
                episode_history,
                rewards_history,
                wins_history,
                epsilon_history,
                policy_loss_history,
                setup_agent1_rewards,
                setup_agent2_rewards,
                setup_agent1_losses,
                setup_agent2_losses,
                pbs_evaluator1_losses,
                pbs_evaluator2_losses,
                pbs_evaluator1_buffer_sizes,
                pbs_evaluator2_buffer_sizes,
                avg_q_history,
                entropy_history
            )
            print(f"💾 Saved training history (finally block)")
        except Exception as e:
            print(f"⚠️  Could not save training history in finally block: {e}")
            
        print("Closing environment...")
        try:
            env.close()
        except:
            pass
            
        for prefetcher in prefetchers:
            try:
                prefetcher.stop()
            except:
                pass
    
    # Final training metrics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Episodes this run: {num_episodes}")
    print(f"Total Episodes (all runs): {total_episodes}")
    print(f"Total Steps (all runs): {total_steps:,}")
    print(f"Agent 1 Wins: {wins_agent1} ({wins_agent1/num_episodes*100:.1f}%)")
    print(f"Agent 2 Wins: {wins_agent2} ({wins_agent2/num_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.1f}%)")
    print(f"Average Reward Agent 1: {np.mean(total_rewards_agent1):.2f}")
    print(f"Average Reward Agent 2: {np.mean(total_rewards_agent2):.2f}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save final models (separate files)
    # Save final models (separate files)
    save_checkpoint(total_episodes, is_final=True)
    
    # Training completed
    return agent1, agent2


def main():
    """Main function to run DQN training"""
    print("🎮 DQN Agent Training for Stratego")
    print("=" * 50)
    
    # Training parameters
    model_save_path = "dqn_models"
    use_setup_agents = True  # Enable setup agents for piece placement
    
    try:
        # Train agents with parallel environment
        agent1, agent2 = train_dqn_agents(NUM_EPISODES, SAVE_INTERVAL, model_save_path,
                                          use_setup_agents=use_setup_agents,
                                          generate_gifs=GENERATE_GIFS)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
