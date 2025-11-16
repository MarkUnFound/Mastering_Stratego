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

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratego_modular.environment import StrategoEnvironment
from stratego_modular.dqn_agent import DQNAgent
from stratego_modular.setup_agent import SetupAgent
from stratego_modular.game_state import GameState
from stratego_modular.training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_setup_agent_progress, plot_pbs_evaluator_progress
from stratego_modular.pbs_visualizer import visualize_pbs_state, create_pbs_gif
from stratego_modular.piece import PieceType, PIECE_RANKS
from stratego_modular.board import LAKE_SQUARE

# Import reset function (optional)
try:
    from stratego_modular.reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False


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
                          pbs_evaluator2_buffer_sizes: List[int]):
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
            'pbs_evaluator2_buffer_sizes': pbs_evaluator2_buffer_sizes
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save training history: {e}")


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
    
    # NEW: Additional Setup Agent Rewards (#1-5) - Scaled down by 10x
    
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
    env = StrategoEnvironment(device=device)
    
    # Create model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Load existing training history if available (for continuity across training sessions)
    loaded_history = _load_training_history(model_save_path)
    
    # Create game-playing agents (with reduced learning rate)
    agent1 = DQNAgent(player_id=1, device=device, lr=0.00001, batch_size=TRAINING_BATCH_SIZE)
    agent2 = DQNAgent(player_id=-1, device=device, lr=0.00001, batch_size=TRAINING_BATCH_SIZE)
    
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
            agent1.load_model(agent1_dqn_path)
            print(f"✅ Loaded Agent 1 DQN model from: {agent1_dqn_path}")
            
            # Check if old combined format has PBS components and load them
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
            agent2.load_model(agent2_dqn_path)
            print(f"✅ Loaded Agent 2 DQN model from: {agent2_dqn_path}")
            
            # Check if old combined format has PBS components and load them
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
            print(f"⚠️  Could not load saved setup agent models: {e}")
            print("   Starting with fresh setup agents")
    
    # Track GIF creation flags
    episode1_pbs_gif_created = False  # Track if episode 1 PBS GIF was created
    winning_game_pbs_gif_created = False  # Track if winning game PBS GIF was created
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
        pbs_evaluator2_buffer_sizes = loaded_history.get('pbs_evaluator2_buffer_sizes', [])
        
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
        
        print(f"📈 Continuing training history from previous session")
        print(f"   Last episode: {episode_history[-1] if episode_history else 0}")
        print(f"   Wins (Agent1/Agent2/Draws): {wins_agent1}/{wins_agent2}/{draws}")
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
        pbs_evaluator2_buffer_sizes = []
        
        # Initialize win counters to 0 for fresh start
        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0
        total_rewards_agent1 = []
        total_rewards_agent2 = []
        
        print(f"📈 Starting fresh training history")
    
    agent1_prefetcher = ReplayPrefetcher(agent1, max_queue_size=PREFETCH_QUEUE_SIZE)
    agent2_prefetcher = ReplayPrefetcher(agent2, max_queue_size=PREFETCH_QUEUE_SIZE)
    prefetchers = [agent1_prefetcher, agent2_prefetcher]
    
    print(f"Starting DQN training for {num_episodes} episodes...")
    print(f"Total Episodes (all runs): {total_episodes}")
    print(f"Total Steps (all runs): {total_steps}")
    print("=" * 60)
    
    def reset_agents():
        """Reset both agents"""
        if RESET_AVAILABLE:
            reset_existing_agents(agent1, agent2)
        else:
            agent1.reset()
            agent2.reset()
            print("Agents reset successfully.")
    
    for episode in range(num_episodes):
        # Setup pieces using setup agents if enabled
        p1_placement = None
        p2_placement = None
        p1_pieces = None
        p2_pieces = None
        p1_positions = None
        p2_positions = None
        
        if use_setup_agents and setup_agent1 and setup_agent2:
            # Generate pieces
            p1_pieces = env._generate_pieces()
            p2_pieces = env._generate_pieces()
            p1_positions = env._get_p1_positions()
            p2_positions = env._get_p2_positions()
            
            # Use setup agents to place pieces
            p1_placement = setup_agent1.place_pieces(p1_pieces, p1_positions)
            p2_placement = setup_agent2.place_pieces(p2_pieces, p2_positions)
        
        # Reset environment with custom placements
        env.reset(p1_placement=p1_placement, p2_placement=p2_placement)
        game_state = env._get_game_state()
        done = False
        move_count = 0
        max_moves = 500  # Prevent infinite games
        
        # Track if this is a winning game (for GIF generation)
        is_winning_game = False
        
        # Track PBS states for GIF creation
        # Only track for episode 1 OR first winning game
        is_episode_1 = (episode == 0)
        should_track_pbs = is_episode_1 or (not winning_game_pbs_gif_created and not is_episode_1)
        episode_pbs_states = [] if should_track_pbs else None  # Only track if needed
        
        # Track actual game board states for GIF creation (non-PBS)
        # Track for first winning game (not episode 1, as episode 1 is already tracked for PBS)
        should_track_game = generate_gifs and (not winning_game_gif_created and not is_episode_1)
        episode_game_states = [] if should_track_game else None  # Track actual board states
        
        # Track PBS captures for episode 1 and checkpoint episodes (multiples of 50)
        # Capture PBS at setup, move 50, and end of game for episode 1 and episodes 50, 100, 150, etc.
        is_checkpoint_episode = (episode + 1) % 50 == 0
        captured_move_50_pbs = False  # Track if we've captured PBS at move 50
        
        # Create PBS visualization for episode 1 and checkpoint episodes (50, 100, 150, etc.)
        # Episode 1 gets all three snapshots: setup, move_50, and end (same as checkpoint episodes)
        if episode == 0 or is_checkpoint_episode:
            try:
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                
                if current_actual_board is not None:
                    # Use episode + 1 for numbering (starts from 1, not 0)
                    episode_number = episode + 1
                    pbs_setup_save_path = f"{model_save_path}/pbs_visualization_episode_{episode_number}_setup.png"
                    visualize_pbs_state(
                        current_actual_board,
                        agent1_pbs,
                        agent2_pbs,
                        episode_number,
                        pbs_setup_save_path,
                        visible_board_p1,
                        visible_board_p2
                    )
                    print(f"🎯 PBS visualization after setup for episode {episode_number}: {pbs_setup_save_path}")
            except Exception as e:
                print(f"⚠️  Error scheduling PBS visualization after setup for episode {episode + 1}: {e}")
                traceback.print_exc()
        
        # Episode rewards
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0
        
        # Get initial state representations (GPU tensors)
        state1 = agent1.get_state_representation(game_state)
        state2 = agent2.get_state_representation(game_state)
        
        # Capture initial PBS state at setup (for episode 1 or first winning game)
        if generate_gifs and should_track_pbs:
            try:
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                
                if current_actual_board is not None:
                    # Clone tensors to avoid reference issues
                    cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                    cloned_visible_p1 = visible_board_p1.clone() if visible_board_p1 is not None and hasattr(visible_board_p1, 'clone') else visible_board_p1
                    cloned_visible_p2 = visible_board_p2.clone() if visible_board_p2 is not None and hasattr(visible_board_p2, 'clone') else visible_board_p2
                    
                    episode_pbs_states.append({
                        'actual_board': cloned_actual,
                        'agent1_pbs': agent1_pbs,
                        'agent2_pbs': agent2_pbs,
                        'move_num': 0,  # Setup state
                        'visible_board_p1': cloned_visible_p1,
                        'visible_board_p2': cloned_visible_p2
                    })
            except Exception as e:
                # Silently skip if PBS capture fails (don't interrupt game)
                pass
        
        # Capture initial game board state at setup (for winning games, non-PBS)
        if should_track_game:
            try:
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                
                if current_actual_board is not None:
                    # Clone tensor to avoid reference issues
                    cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                    
                    episode_game_states.append({
                        'board': cloned_actual,
                        'move_num': 0,  # Setup state
                        'last_move': None
                    })
            except Exception as e:
                # Silently skip if game state capture fails (don't interrupt game)
                pass
        
        while not done and move_count < max_moves:
            # Determine current player and agent
            current_agent = agent1 if env.current_player == 1 else agent2
            current_state = state1 if env.current_player == 1 else state2
            
            # Get valid moves
            valid_moves = env.get_valid_moves()
            
            if not valid_moves:
                # No valid moves, game ends
                done = True
                break
                
            # Agent selects action (PBS-enhanced if enabled)
            # PBS first gets the value and creates possible values with confidence scores
            # Then DQN calculates Q-value
            action = current_agent.act(current_state, valid_moves, game_state=game_state)
            
            if action is None:
                # Invalid action, game ends
                done = True
                break
            
            # Update PBS from action (before executing, to track opponent's pieces)
            # Update both agents' PBS to track opponent actions
            if env.current_player == 1:
                agent2.update_pbs_from_action(action, game_state, acting_player=1)
            else:
                agent1.update_pbs_from_action(action, game_state, acting_player=-1)
                
            # Execute action (action is guaranteed to be valid from get_valid_moves)
            next_game_state, reward, done, info = env.step(action)
            
            # Track PBS states every move for episode 1 or first winning game (to create PBS GIF)
            # Capture every move to show complete game progression
            if generate_gifs and should_track_pbs:  # Capture every move
                try:
                    current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                    visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                    visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                    agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                    agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                    
                    if current_actual_board is not None:
                        # Clone tensors to avoid reference issues
                        cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                        cloned_visible_p1 = visible_board_p1.clone() if visible_board_p1 is not None and hasattr(visible_board_p1, 'clone') else visible_board_p1
                        cloned_visible_p2 = visible_board_p2.clone() if visible_board_p2 is not None and hasattr(visible_board_p2, 'clone') else visible_board_p2
                        
                        episode_pbs_states.append({
                            'actual_board': cloned_actual,
                            'agent1_pbs': agent1_pbs,
                            'agent2_pbs': agent2_pbs,
                            'move_num': move_count + 1,
                            'visible_board_p1': cloned_visible_p1,
                            'visible_board_p2': cloned_visible_p2
                        })
                except Exception as e:
                    # Silently skip if PBS capture fails (don't interrupt game)
                    pass
            
            # Track actual game board states every move for winning games (to create non-PBS game GIF)
            if should_track_game:  # Track for first winning game
                try:
                    current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                    
                    if current_actual_board is not None:
                        # Clone tensor to avoid reference issues
                        cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                        
                        # Get last move for visualization
                        last_move = None
                        if move_count > 0 and hasattr(env, 'last_action') and env.last_action:
                            last_move = env.last_action  # Should be ((from_r, from_c), (to_r, to_c))
                        
                        episode_game_states.append({
                            'board': cloned_actual,
                            'move_num': move_count + 1,
                            'last_move': last_move
                        })
                except Exception as e:
                    # Silently skip if game state capture fails (don't interrupt game)
                    pass
            
            # Determine game phase for PBS evaluator data collection
            # Early game: turns 0-50, Middle game: turns 51-200, End game: turns 201+
            game_phase = 'early'
            if move_count > 200:
                game_phase = 'end'
            elif move_count > 50:
                game_phase = 'middle'
            
            # OPTIMIZATION: Update PBS from revealed pieces more efficiently
            # Check if pieces were revealed in the battle
            if hasattr(env, 'revealed_pieces_p1') and hasattr(env, 'revealed_pieces_p2'):
                # Update agent1's PBS with revealed pieces (only if PBS exists)
                if hasattr(agent1, 'pbs') and agent1.pbs:
                    for pos, piece_value in env.revealed_pieces_p1.items():
                        # OPTIMIZATION: Check revealed_pieces set first (faster than dict lookup)
                        if pos not in agent1.pbs.revealed_pieces:
                            piece_type = PieceType(abs(piece_value))
                            agent1.update_pbs_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=move_count)
                # Update agent2's PBS with revealed pieces (only if PBS exists)
                if hasattr(agent2, 'pbs') and agent2.pbs:
                    for pos, piece_value in env.revealed_pieces_p2.items():
                        # OPTIMIZATION: Check revealed_pieces set first (faster than dict lookup)
                        if pos not in agent2.pbs.revealed_pieces:
                            piece_type = PieceType(abs(piece_value))
                            agent2.update_pbs_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=move_count)
            
            # Get next state representation (returns GPU tensor)
            next_state = current_agent.get_state_representation(next_game_state)
            
            # Store experience (only for valid moves) - tensors stay on GPU
            current_agent.remember(current_state, 
                                 current_agent._move_to_action_index(action),
                                 reward,
                                 next_state,
                                 done)
            
            # Accumulate rewards
            if env.current_player == 1:
                episode_reward_agent1 += reward
            else:
                episode_reward_agent2 += reward
                
            # Update states
            if env.current_player == 1:
                state1 = next_state
            else:
                state2 = next_state
                
            move_count += 1  # Only increment for valid moves
            
            # Capture PBS at move 50 for episode 1 and checkpoint episodes (50, 100, 150, etc.)
            if (episode == 0 or is_checkpoint_episode) and move_count == 50 and not done and not captured_move_50_pbs:
                try:
                    # Use EXACTLY the same logic as setup visualization for consistency
                    current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                    visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                    visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                    agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                    agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                    
                    if current_actual_board is not None:
                        # Use episode + 1 for numbering (starts from 1, not 0)
                        episode_number = episode + 1
                        pbs_save_path = f"{model_save_path}/pbs_visualization_episode_{episode_number}_move_50.png"
                        visualize_pbs_state(
                            current_actual_board,
                            agent1_pbs,
                            agent2_pbs,
                            episode_number,
                            pbs_save_path,
                            visible_board_p1,
                            visible_board_p2
                        )
                        print(f"🎯 PBS visualization at move 50 of episode {episode_number}: {pbs_save_path}")
                        captured_move_50_pbs = True
                except Exception as e:
                    print(f"⚠️  Error scheduling PBS visualization at move 50 of episode {episode + 1}: {e}")
                    traceback.print_exc()
            
            # Batched training updates to maximize GPU utilization
            if move_count % REPLAY_UPDATE_INTERVAL == 0:
                for _ in range(REPLAY_UPDATES_PER_STEP):
                    agent1.replay(batch=agent1_prefetcher.get_batch())
                    agent2.replay(batch=agent2_prefetcher.get_batch())
                
        # Post-episode training burst to use prefetched batches
        for _ in range(REPLAY_UPDATES_PER_STEP):
            agent1.replay(batch=agent1_prefetcher.get_batch())
            agent2.replay(batch=agent2_prefetcher.get_batch())
        
        # ADD THIS: Check for loss explosion after episode (every 10 episodes)
        if (episode + 1) % 10 == 0:
            avg_loss1 = agent1.get_average_policy_loss(10)
            avg_loss2 = agent2.get_average_policy_loss(10)
            
            # Reset if loss explodes
            if avg_loss1 > 1000 or avg_loss2 > 1000:
                print(f"⚠️ Loss explosion detected after episode {episode + 1}! Resetting agents...")
                print(f"   Agent 1 loss: {avg_loss1:.2f}, Agent 2 loss: {avg_loss2:.2f}")
                reset_agents()
                # Try to reload last good checkpoint if available
                try:
                    checkpoint_episode = max(0, total_episodes - 100)
                    agent1_checkpoint = f"{model_save_path}/agent1_episode_{checkpoint_episode}.pth"
                    agent2_checkpoint = f"{model_save_path}/agent2_episode_{checkpoint_episode}.pth"
                    if os.path.exists(agent1_checkpoint):
                        agent1.load_model(agent1_checkpoint)
                        print(f"   Reloaded Agent 1 from checkpoint {checkpoint_episode}")
                    if os.path.exists(agent2_checkpoint):
                        agent2.load_model(agent2_checkpoint)
                        print(f"   Reloaded Agent 2 from checkpoint {checkpoint_episode}")
                except Exception as e:
                    print(f"   Could not reload checkpoint: {e}")
        
        # Game finished - get final game state for PBS visualization
        # Use the last game state (next_game_state from the loop, or get it from env)
        final_game_state = next_game_state if 'next_game_state' in locals() else env._get_game_state()
        actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
        
        # OPTIMIZATION: Collect end-game PBS data more efficiently
        # Only collect for positions that actually have beliefs (avoid full board scan)
        if actual_board is not None and move_count > 50:  # Only collect if game progressed past early phase
            # OPTIMIZATION: Only iterate through positions that have beliefs, not entire board
            # OPTIMIZATION: Batch process positions to reduce .item() calls
            # Collect data for agent1's PBS (opponent pieces are player -1)
            if agent1.pbs and agent1.pbs.evaluator and agent1.pbs.belief_distributions:
                # Only check positions that have beliefs (much faster than scanning entire board)
                # OPTIMIZATION: Batch extract piece values to reduce .item() calls
                positions_to_check = [pos for pos in agent1.pbs.belief_distributions.keys() 
                                     if 0 <= pos[0] < 10 and 0 <= pos[1] < 10 and pos not in agent1.pbs.revealed_pieces]
                if positions_to_check:
                    # Batch extract values (faster than individual .item() calls)
                    for pos in positions_to_check:
                        r, c = pos
                        piece_value = int(actual_board[r, c].item())
                        # Check if this is an opponent piece (negative for player 1)
                        if piece_value < 0 and piece_value != LAKE_SQUARE:  # Not empty or lake
                            piece_type = PieceType(abs(piece_value))
                            agent1.pbs.update_from_reveal(pos, piece_type, game_phase='end', turn_count=move_count)
            
            # Collect data for agent2's PBS (opponent pieces are player 1)
            if agent2.pbs and agent2.pbs.evaluator and agent2.pbs.belief_distributions:
                # Only check positions that have beliefs (much faster than scanning entire board)
                # OPTIMIZATION: Batch process positions
                positions_to_check = [pos for pos in agent2.pbs.belief_distributions.keys() 
                                     if 0 <= pos[0] < 10 and 0 <= pos[1] < 10 and pos not in agent2.pbs.revealed_pieces]
                if positions_to_check:
                    for pos in positions_to_check:
                        r, c = pos
                        piece_value = int(actual_board[r, c].item())
                        # Check if this is an opponent piece (positive for player -1)
                        if piece_value > 0:
                            piece_type = PieceType(abs(piece_value))
                            agent2.pbs.update_from_reveal(pos, piece_type, game_phase='end', turn_count=move_count)
        
        # Get winner from the final state or environment
        winner = final_game_state.winner if hasattr(final_game_state, 'winner') else (env.winner if hasattr(env, 'winner') else None)
        
        # Capture final PBS state for episode 1 or first winning game (to include in GIF)
        if generate_gifs and should_track_pbs and (winner == 1 or winner == -1 or is_episode_1):
            try:
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                
                if current_actual_board is not None:
                    # Clone tensors to avoid reference issues
                    cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                    cloned_visible_p1 = visible_board_p1.clone() if visible_board_p1 is not None and hasattr(visible_board_p1, 'clone') else visible_board_p1
                    cloned_visible_p2 = visible_board_p2.clone() if visible_board_p2 is not None and hasattr(visible_board_p2, 'clone') else visible_board_p2
                    
                    episode_pbs_states.append({
                        'actual_board': cloned_actual,
                        'agent1_pbs': agent1_pbs,
                        'agent2_pbs': agent2_pbs,
                        'move_num': move_count,
                        'visible_board_p1': cloned_visible_p1,
                        'visible_board_p2': cloned_visible_p2
                    })
            except Exception as e:
                # Silently skip if PBS capture fails (don't interrupt game)
                pass
        
        # Capture final game board state for winning games (non-PBS)
        if should_track_game and (winner == 1 or winner == -1):
            try:
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                
                if current_actual_board is not None:
                    # Clone tensor to avoid reference issues
                    cloned_actual = current_actual_board.clone() if hasattr(current_actual_board, 'clone') else current_actual_board
                    
                    # Get last move for visualization
                    last_move = None
                    if move_count > 0 and hasattr(env, 'last_action') and env.last_action:
                        last_move = env.last_action
                    
                    episode_game_states.append({
                        'board': cloned_actual,
                        'move_num': move_count,
                        'last_move': last_move
                    })
            except Exception as e:
                # Silently skip if game state capture fails (don't interrupt game)
                pass
        
        if winner == 1:
            wins_agent1 += 1
            is_winning_game = True
            # Give positive reward to winner, negative to loser
            agent1.remember(state1, agent1._move_to_action_index(action), 10.0, next_state, True)
            agent2.remember(state2, agent2._move_to_action_index(action), -10.0, next_state, True)
            
            # Reward setup agent with enhanced reward calculation
            if use_setup_agents and setup_agent1 and p1_placement is not None:
                setup_reward = calculate_setup_agent_reward(
                    p1_placement, player_id=1, winner=winner, move_count=move_count
                )
                setup_agent1.remember(
                    setup_agent1.get_state_representation(p1_pieces, p1_positions),
                    0,  # Action index (simplified)
                    setup_reward,  # Enhanced reward
                    setup_agent1.get_state_representation([], []),  # Next state (empty after game)
                    True  # Done
                )
                # Train setup agent
                setup_agent1.replay()
        elif winner == -1:
            wins_agent2 += 1
            is_winning_game = True
            # Give positive reward to winner, negative to loser
            agent2.remember(state2, agent2._move_to_action_index(action), 10.0, next_state, True)
            agent1.remember(state1, agent1._move_to_action_index(action), -10.0, next_state, True)
            
            # Reward setup agent with enhanced reward calculation
            if use_setup_agents and setup_agent2 and p2_placement is not None:
                setup_reward = calculate_setup_agent_reward(
                    p2_placement, player_id=-1, winner=winner, move_count=move_count
                )
                setup_agent2.remember(
                    setup_agent2.get_state_representation(p2_pieces, p2_positions),
                    0,  # Action index (simplified)
                    setup_reward,  # Enhanced reward
                    setup_agent2.get_state_representation([], []),  # Next state (empty after game)
                    True  # Done
                )
                # Train setup agent
                setup_agent2.replay()
        else:
            draws += 1
            # CHANGED: Penalize draws to encourage decisive play
            agent1.remember(state1, agent1._move_to_action_index(action), -5.0, next_state, True)
            agent2.remember(state2, agent2._move_to_action_index(action), -5.0, next_state, True)
            
            # Reward setup agents for draw (with enhanced reward calculation)
            if use_setup_agents and setup_agent1 and p1_placement is not None:
                setup_reward = calculate_setup_agent_reward(
                    p1_placement, player_id=1, winner=None, move_count=move_count
                )
                setup_agent1.remember(
                    setup_agent1.get_state_representation(p1_pieces, p1_positions),
                    0,
                    setup_reward,
                    setup_agent1.get_state_representation([], []),
                    True
                )
                setup_agent1.replay()
            
            if use_setup_agents and setup_agent2 and p2_placement is not None:
                setup_reward = calculate_setup_agent_reward(
                    p2_placement, player_id=-1, winner=None, move_count=move_count
                )
                setup_agent2.remember(
                    setup_agent2.get_state_representation(p2_pieces, p2_positions),
                    0,
                    setup_reward,
                    setup_agent2.get_state_representation([], []),
                    True
                )
                setup_agent2.replay()
        
        # Create PBS visualization GIF for episode 1
        if generate_gifs and is_episode_1 and episode_pbs_states:
            pbs_gif_path = f"{model_save_path}/pbs_visualization_episode_1.gif"
            print(f"🎬 Creating PBS GIF for episode 1 with 750ms per frame...")
            create_pbs_gif(
                episode_pbs_states,
                1,
                pbs_gif_path,
                frame_duration=750  # 750ms per frame as requested
            )
            print(f"✅ PBS GIF created for episode 1")
            episode1_pbs_gif_created = True
        
        # Create PBS visualization GIF for first winning game (if episode 1 didn't win)
        # If episode 1 won, wait for next winning game
        if generate_gifs and is_winning_game and not winning_game_pbs_gif_created:
            if is_episode_1:
                # Episode 1 won - mark that we'll skip this and wait for next winning game
                # Don't create winning game GIF for episode 1, already created episode 1 GIF
                pass
            elif episode_pbs_states:
                # First winning game after episode 1 - create winning game GIF
                pbs_gif_path = f"{model_save_path}/pbs_visualization_win_{total_episodes}.gif"
                print(f"🎬 Creating PBS GIF for winning game at episode {episode + 1} with 750ms per frame...")
                create_pbs_gif(
                    episode_pbs_states,
                    episode + 1,
                    pbs_gif_path,
                    frame_duration=750  # 750ms per frame as requested
                )
                print(f"✅ PBS GIF created for winning game at episode {episode + 1}")
                winning_game_pbs_gif_created = True
        
        # Create actual game board GIF for first winning game (non-PBS)
        if generate_gifs and is_winning_game and not winning_game_gif_created:
            if is_episode_1:
                # Episode 1 won - mark that we'll skip this and wait for next winning game
                # Don't create winning game GIF for episode 1, already created episode 1 GIF
                pass
            elif episode_game_states:
                # First winning game after episode 1 - create actual game board GIF
                game_gif_path = f"{model_save_path}/game_visualization_win_{total_episodes}.gif"
                print(f"🎬 Creating game board GIF for winning game at episode {episode + 1} with 750ms per frame...")
                create_episode_gif(
                    episode_game_states,
                    episode + 1,
                    game_gif_path,
                    frame_duration=750  # 750ms per frame
                )
                print(f"✅ Game board GIF created for winning game at episode {episode + 1}")
                winning_game_gif_created = True
        
        # Clear PBS states to save memory
        if generate_gifs and episode_pbs_states is not None:
            episode_pbs_states = []
        
        # Clear game states to save memory
        if generate_gifs and episode_game_states is not None:
            episode_game_states = []
        
        # Capture PBS at end for episode 1 and checkpoint episodes (50, 100, 150, etc.)
        if episode == 0 or is_checkpoint_episode:
            try:
                # Use EXACTLY the same logic as setup visualization for consistency
                current_actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                visible_board_p1 = env.board.get_visible_board(1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                visible_board_p2 = env.board.get_visible_board(-1) if hasattr(env, 'board') and hasattr(env.board, 'get_visible_board') else None
                agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') else None
                agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') else None
                
                if current_actual_board is not None:
                    # Use episode + 1 for numbering (starts from 1, not 0)
                    episode_number = episode + 1
                    pbs_save_path = f"{model_save_path}/pbs_visualization_episode_{episode_number}_end.png"
                    visualize_pbs_state(
                        current_actual_board,
                        agent1_pbs,
                        agent2_pbs,
                        episode_number,
                        pbs_save_path,
                        visible_board_p1,
                        visible_board_p2
                    )
                    print(f"🎯 PBS visualization at end of episode {episode_number} (move {move_count}): {pbs_save_path}")
            except Exception as e:
                print(f"⚠️  Error scheduling PBS visualization at end of episode {episode + 1}: {e}")
                traceback.print_exc()
        
        # Note: PBS visualization for checkpoint episodes (multiples of 50) is now handled above
        # It captures PBS at both move 50 (if game hasn't ended) and at end of game
            
        # Update target networks periodically
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
        
        # Train PBS evaluators periodically (only if they have collected data)
        if episode % 20 == 0 and episode > 0:
            # Train evaluators on collected data
            evaluator1_loss = None
            evaluator2_loss = None
            if agent1.pbs and agent1.pbs.evaluator:
                evaluator1_loss = agent1.pbs.train_evaluator(epochs=2)
                if evaluator1_loss is not None and episode % 100 == 0:
                    print(f"  PBS Evaluator 1 Loss: {evaluator1_loss:.4f}")
            if agent2.pbs and agent2.pbs.evaluator:
                evaluator2_loss = agent2.pbs.train_evaluator(epochs=2)
                if evaluator2_loss is not None and episode % 100 == 0:
                    print(f"  PBS Evaluator 2 Loss: {evaluator2_loss:.4f}")
            
            # Update target networks for evaluators
            if agent1.pbs and agent1.pbs.evaluator:
                agent1.pbs.evaluator.update_target_network()
            if agent2.pbs and agent2.pbs.evaluator:
                agent2.pbs.evaluator.update_target_network()
            
            # Track PBS evaluator metrics
            pbs_evaluator1_losses.append(evaluator1_loss)
            pbs_evaluator2_losses.append(evaluator2_loss)
            pbs_evaluator1_buffer_sizes.append(
                len(agent1.pbs.evaluator.memory) if agent1.pbs and agent1.pbs.evaluator else 0
            )
            pbs_evaluator2_buffer_sizes.append(
                len(agent2.pbs.evaluator.memory) if agent2.pbs and agent2.pbs.evaluator else 0
            )
        else:
            # Track metrics even when not training (to keep lists aligned)
            pbs_evaluator1_losses.append(None)
            pbs_evaluator2_losses.append(None)
            pbs_evaluator1_buffer_sizes.append(
                len(agent1.pbs.evaluator.memory) if agent1.pbs and agent1.pbs.evaluator else 0
            )
            pbs_evaluator2_buffer_sizes.append(
                len(agent2.pbs.evaluator.memory) if agent2.pbs and agent2.pbs.evaluator else 0
            )
            
        # Update persistent counters (survive across agent resets)
        total_episodes += 1
        
        # Update total steps by adding moves from this episode
        # move_count tracks the number of moves in this episode
        # This is more reliable than using agent.step_count which resets
        total_steps += move_count
        
        # Update epsilon based on total_steps (not agent's step_count which resets)
        # This ensures epsilon decay continues even after agent resets
        epsilon_decay_interval = 500_000  # Same as agent's epsilon_decay_interval
        # Epsilon decay is now handled internally by the agent (with minimum epsilon for exploration)
        # No need to manually set epsilon here - the agent manages it with adaptive adjustments
        
        # Store episode rewards
        total_rewards_agent1.append(episode_reward_agent1)
        total_rewards_agent2.append(episode_reward_agent2)
        
        # Update adaptive epsilon based on episode performance
        if hasattr(agent1, 'update_episode_reward'):
            agent1.update_episode_reward(episode_reward_agent1)
        if hasattr(agent2, 'update_episode_reward'):
            agent2.update_episode_reward(episode_reward_agent2)
        
        # Update history for plotting every episode (discrete points)
        # Use total_episodes for continuity across training sessions
        episode_history.append(total_episodes)
        rewards_history['agent1'].append(episode_reward_agent1)  # Store individual episode reward
        rewards_history['agent2'].append(episode_reward_agent2)  # Store individual episode reward
        wins_history['agent1'].append(wins_agent1)
        wins_history['agent2'].append(wins_agent2)
        wins_history['draws'].append(draws)
        epsilon_history['agent1'].append(agent1.epsilon)
        epsilon_history['agent2'].append(agent2.epsilon)
        
        # Add policy loss to history (get current average)
        avg_loss1 = agent1.get_average_policy_loss(100) if hasattr(agent1, 'get_average_policy_loss') else 0.0
        avg_loss2 = agent2.get_average_policy_loss(100) if hasattr(agent2, 'get_average_policy_loss') else 0.0
        policy_loss_history['agent1'].append(avg_loss1)
        policy_loss_history['agent2'].append(avg_loss2)
        
        # Track setup agent rewards and losses (ensure both lists stay same length as episode_history)
        if use_setup_agents and setup_agent1:
            if len(setup_agent1.episode_rewards) > 0:
                setup_agent1_rewards.append(setup_agent1.episode_rewards[-1])
            else:
                # Append last known reward or 0.0 if no rewards yet
                last_reward = setup_agent1_rewards[-1] if len(setup_agent1_rewards) > 0 else 0.0
                setup_agent1_rewards.append(last_reward)
            avg_setup_loss1 = setup_agent1.get_average_policy_loss(100) if hasattr(setup_agent1, 'get_average_policy_loss') else 0.0
            setup_agent1_losses.append(avg_setup_loss1)
        elif use_setup_agents:
            # Setup agent 1 not available, append 0.0 to keep lists aligned
            setup_agent1_rewards.append(0.0)
            setup_agent1_losses.append(0.0)
        
        if use_setup_agents and setup_agent2:
            if len(setup_agent2.episode_rewards) > 0:
                setup_agent2_rewards.append(setup_agent2.episode_rewards[-1])
            else:
                # Append last known reward or 0.0 if no rewards yet
                last_reward = setup_agent2_rewards[-1] if len(setup_agent2_rewards) > 0 else 0.0
                setup_agent2_rewards.append(last_reward)
            avg_setup_loss2 = setup_agent2.get_average_policy_loss(100) if hasattr(setup_agent2, 'get_average_policy_loss') else 0.0
            setup_agent2_losses.append(avg_setup_loss2)
        elif use_setup_agents:
            # Setup agent 2 not available, append 0.0 to keep lists aligned
            setup_agent2_rewards.append(0.0)
            setup_agent2_losses.append(0.0)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward1 = np.mean(total_rewards_agent1[-50:]) if total_rewards_agent1 else 0
            avg_reward2 = np.mean(total_rewards_agent2[-50:]) if total_rewards_agent2 else 0
            print(f"Episode {episode + 1}/{num_episodes} (Total: {total_episodes})")
            print(f"  Total Steps: {total_steps:,}")
            print(f"  Agent 1 wins: {wins_agent1}, Agent 2 wins: {wins_agent2}, Draws: {draws}")
            print(f"  Avg Reward Agent 1 (last 50): {avg_reward1:.2f}")
            print(f"  Avg Reward Agent 2 (last 50): {avg_reward2:.2f}")
            print(f"  Avg Policy Loss Agent 1 (last 100): {avg_loss1:.4f}")
            print(f"  Avg Policy Loss Agent 2 (last 100): {avg_loss2:.4f}")
            # Show detailed loss statistics if available
            if hasattr(agent1, 'get_policy_loss_stats'):
                stats1 = agent1.get_policy_loss_stats(100)
                stats2 = agent2.get_policy_loss_stats(100)
                print(f"  Loss Stats Agent 1: min={stats1['min']:.2f}, max={stats1['max']:.2f}, median={stats1['median']:.2f}, std={stats1['std']:.2f}")
                print(f"  Loss Stats Agent 2: min={stats2['min']:.2f}, max={stats2['max']:.2f}, median={stats2['median']:.2f}, std={stats2['std']:.2f}")
            # Show smoothed loss if available
            if hasattr(agent1, 'get_smoothed_loss'):
                smoothed1 = agent1.get_smoothed_loss()
                smoothed2 = agent2.get_smoothed_loss()
                print(f"  Smoothed Loss Agent 1: {smoothed1:.4f}, Agent 2: {smoothed2:.4f}")
            print(f"  Epsilon Agent 1: {agent1.epsilon:.3f}, Epsilon Agent 2: {agent2.epsilon:.3f}")
            # Show learning rates if available (with more precision and change tracking)
            if hasattr(agent1, 'get_current_learning_rate'):
                lr1 = agent1.get_current_learning_rate()
                lr2 = agent2.get_current_learning_rate()
                # Show if LR has changed from initial
                lr1_change = ((lr1 - agent1.initial_lr) / agent1.initial_lr * 100) if hasattr(agent1, 'initial_lr') else 0
                lr2_change = ((lr2 - agent2.initial_lr) / agent2.initial_lr * 100) if hasattr(agent2, 'initial_lr') else 0
                print(f"  Learning Rate Agent 1: {lr1:.8f} ({lr1_change:+.1f}%), Agent 2: {lr2:.8f} ({lr2_change:+.1f}%)")
            print("-" * 60)
            
            # Reset agents if average reward is too large
            if abs(avg_reward1) > 100 or abs(avg_reward2) > 100:
                print("Average reward too large, resetting agents...")
                reset_agents()
                # Reset statistics
                wins_agent1 = 0
                wins_agent2 = 0
                draws = 0
                total_rewards_agent1 = []
                total_rewards_agent2 = []
                # Reset history for plotting (but keep previous session's history)
                # Only clear current session's data, not loaded history
                # This allows graphs to show the reset point clearly
                current_session_start = len(episode_history)
                episode_history = episode_history[:current_session_start]  # Keep previous history
                # Truncate other histories to match
                for key in rewards_history:
                    rewards_history[key] = rewards_history[key][:current_session_start]
                for key in wins_history:
                    wins_history[key] = wins_history[key][:current_session_start]
                for key in epsilon_history:
                    epsilon_history[key] = epsilon_history[key][:current_session_start]
                for key in policy_loss_history:
                    policy_loss_history[key] = policy_loss_history[key][:current_session_start]
                setup_agent1_rewards = setup_agent1_rewards[:current_session_start]
                setup_agent2_rewards = setup_agent2_rewards[:current_session_start]
                setup_agent1_losses = setup_agent1_losses[:current_session_start]
                setup_agent2_losses = setup_agent2_losses[:current_session_start]
                pbs_evaluator1_losses = pbs_evaluator1_losses[:current_session_start]
                pbs_evaluator2_losses = pbs_evaluator2_losses[:current_session_start]
                pbs_evaluator1_buffer_sizes = pbs_evaluator1_buffer_sizes[:current_session_start]
                pbs_evaluator2_buffer_sizes = pbs_evaluator2_buffer_sizes[:current_session_start]
                # Reset setup agent tracking lists
                if use_setup_agents:
                    setup_agent1_rewards = []
                    setup_agent2_rewards = []
                    setup_agent1_losses = []
                    setup_agent2_losses = []
                # Reset PBS evaluator tracking lists
                pbs_evaluator1_losses = []
                pbs_evaluator2_losses = []
                pbs_evaluator1_buffer_sizes = []
                pbs_evaluator2_buffer_sizes = []
            
            # OPTIMIZATION: Save persistent counters every 100 episodes (not just every 50)
            # This ensures counters are saved more frequently
            if (episode + 1) % 100 == 0:
                _save_counters(
                    total_episodes_file,
                    total_steps_file,
                    total_episodes,
                    total_steps
                )
                # Also save training history for continuity
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
                    pbs_evaluator2_buffer_sizes
                )
            
            # Save chart every 50 episodes (only if we have data)
            if len(episode_history) > 0:
                # Save main DQN training progress (separate from setup agent progress)
                plot_training_progress(
                    episode_history,
                    rewards_history,
                    wins_history,
                    policy_loss_history,
                    save_path=f"{model_save_path}/training_progress_episode_{total_episodes}.png",
                    total_episodes=total_episodes,
                    total_steps=total_steps
                )
                print(f"📈 Training progress graph saved: {model_save_path}/training_progress_episode_{total_episodes}.png")
                
                # Save setup agent progress chart (separate PNG file, independent from main DQN plot)
                if use_setup_agents and setup_agent1 and setup_agent2:
                    # Only plot if we have data (rewards lists are not empty)
                    if setup_agent1_rewards and setup_agent2_rewards and len(setup_agent1_rewards) > 0 and len(setup_agent2_rewards) > 0:
                        plot_setup_agent_progress(
                            episode_history,
                            setup_agent1_rewards,
                            setup_agent2_rewards,
                            setup_agent1_losses,
                            setup_agent2_losses,
                            save_path=f"{model_save_path}/setup_agent_progress_episode_{total_episodes}.png"
                        )
                        print(f"📊 Setup agent progress graph saved: {model_save_path}/setup_agent_progress_episode_{total_episodes}.png")
                    else:
                        # Create placeholder plot if no data yet
                        plot_setup_agent_progress(
                            episode_history,
                            setup_agent1_rewards,
                            setup_agent2_rewards,
                            setup_agent1_losses,
                            setup_agent2_losses,
                            save_path=f"{model_save_path}/setup_agent_progress_episode_{total_episodes}.png"
                        )
                
                # Save PBS evaluator progress chart (separate PNG file)
                plot_pbs_evaluator_progress(
                    episode_history,
                    pbs_evaluator1_losses,
                    pbs_evaluator2_losses,
                    pbs_evaluator1_buffer_sizes,
                    pbs_evaluator2_buffer_sizes,
                    save_path=f"{model_save_path}/pbs_evaluator_progress_episode_{total_episodes}.png",
                    total_episodes=total_episodes
                )
                print(f"📊 PBS evaluator progress graph saved: {model_save_path}/pbs_evaluator_progress_episode_{total_episodes}.png")
            else:
                print(f"⚠️  Skipping progress plots - no episode history yet (agents were reset)")
            
        # Save models periodically (keep at save_interval)
        if (episode + 1) % save_interval == 0:
            # Create model save directory if it doesn't exist
            os.makedirs(model_save_path, exist_ok=True)
            try:
                # Save DQN models separately
                agent1_dqn_path = f"{model_save_path}/agent1_dqn_episode_{total_episodes}.pth"
                agent2_dqn_path = f"{model_save_path}/agent2_dqn_episode_{total_episodes}.pth"
                agent1.save_model(agent1_dqn_path)
                agent2.save_model(agent2_dqn_path)
                
                # Save AAREN models separately
                agent1_aaren_path = f"{model_save_path}/agent1_aaren_episode_{total_episodes}.pth"
                agent2_aaren_path = f"{model_save_path}/agent2_aaren_episode_{total_episodes}.pth"
                try:
                    agent1.save_aaren_model(agent1_aaren_path)
                    agent2.save_aaren_model(agent2_aaren_path)
                except Exception as e:
                    print(f"⚠️  Warning: Could not save AAREN models: {e}")
                
                # Save PBS Evaluator models separately
                agent1_evaluator_path = f"{model_save_path}/agent1_pbs_evaluator_episode_{total_episodes}.pth"
                agent2_evaluator_path = f"{model_save_path}/agent2_pbs_evaluator_episode_{total_episodes}.pth"
                try:
                    agent1.save_pbs_evaluator(agent1_evaluator_path)
                    agent2.save_pbs_evaluator(agent2_evaluator_path)
                except Exception as e:
                    print(f"⚠️  Warning: Could not save PBS Evaluator models: {e}")
                
                # Save setup agents if they exist
                if use_setup_agents and setup_agent1 and setup_agent2:
                    setup_agent1_path = f"{model_save_path}/setup_agent1_episode_{total_episodes}.pth"
                    setup_agent2_path = f"{model_save_path}/setup_agent2_episode_{total_episodes}.pth"
                    setup_agent1.save_model(setup_agent1_path)
                    setup_agent2.save_model(setup_agent2_path)
                
                # Verify files were created
                if os.path.exists(agent1_dqn_path) and os.path.exists(agent2_dqn_path):
                    print(f"💾 Models saved at episode {episode + 1}:")
                    print(f"   - {agent1_dqn_path} (DQN only)")
                    print(f"   - {agent2_dqn_path} (DQN only)")
                    if os.path.exists(agent1_aaren_path):
                        print(f"   - {agent1_aaren_path} (AAREN-RNN)")
                    if os.path.exists(agent2_aaren_path):
                        print(f"   - {agent2_aaren_path} (AAREN-RNN)")
                    if os.path.exists(agent1_evaluator_path):
                        print(f"   - {agent1_evaluator_path} (PBS Evaluator)")
                    if os.path.exists(agent2_evaluator_path):
                        print(f"   - {agent2_evaluator_path} (PBS Evaluator)")
                    if use_setup_agents and setup_agent1 and setup_agent2:
                        print(f"   - {setup_agent1_path} (Setup Agent NN)")
                        print(f"   - {setup_agent2_path} (Setup Agent NN)")
                else:
                    print(f"⚠️  Warning: Model files may not have been created at episode {episode + 1}")
            except Exception as e:
                print(f"⚠️  Error saving models at episode {episode + 1}: {e}")
                traceback.print_exc()
            
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
            pbs_evaluator2_buffer_sizes
        )
        print(f"💾 Saved final training history for continuity")
    except Exception as e:
        print(f"⚠️  Could not save final training history: {e}")
    
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
    try:
        # Save final DQN models
        agent1_dqn_final_path = f"{model_save_path}/agent1_dqn_final.pth"
        agent2_dqn_final_path = f"{model_save_path}/agent2_dqn_final.pth"
        agent1.save_model(agent1_dqn_final_path)
        agent2.save_model(agent2_dqn_final_path)
        
        # Save final AAREN models
        try:
            agent1_aaren_final_path = f"{model_save_path}/agent1_aaren_final.pth"
            agent2_aaren_final_path = f"{model_save_path}/agent2_aaren_final.pth"
            agent1.save_aaren_model(agent1_aaren_final_path)
            agent2.save_aaren_model(agent2_aaren_final_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not save final AAREN models: {e}")
        
        # Save final PBS Evaluator models
        try:
            agent1_evaluator_final_path = f"{model_save_path}/agent1_pbs_evaluator_final.pth"
            agent2_evaluator_final_path = f"{model_save_path}/agent2_pbs_evaluator_final.pth"
            agent1.save_pbs_evaluator(agent1_evaluator_final_path)
            agent2.save_pbs_evaluator(agent2_evaluator_final_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not save final PBS Evaluator models: {e}")
        
        # Save final setup agents if they exist
        if use_setup_agents and setup_agent1 and setup_agent2:
            setup_agent1_final_path = f"{model_save_path}/setup_agent1_final.pth"
            setup_agent2_final_path = f"{model_save_path}/setup_agent2_final.pth"
            setup_agent1.save_model(setup_agent1_final_path)
            setup_agent2.save_model(setup_agent2_final_path)
        
        # Verify files were created
        if os.path.exists(agent1_dqn_final_path) and os.path.exists(agent2_dqn_final_path):
            print(f"\n💾 Final models saved:")
            print(f"   - {agent1_dqn_final_path} (DQN only)")
            print(f"   - {agent2_dqn_final_path} (DQN only)")
            if os.path.exists(agent1_aaren_final_path):
                print(f"   - {agent1_aaren_final_path} (AAREN-RNN)")
            if os.path.exists(agent2_aaren_final_path):
                print(f"   - {agent2_aaren_final_path} (AAREN-RNN)")
            if os.path.exists(agent1_evaluator_final_path):
                print(f"   - {agent1_evaluator_final_path} (PBS Evaluator)")
            if os.path.exists(agent2_evaluator_final_path):
                print(f"   - {agent2_evaluator_final_path} (PBS Evaluator)")
            if use_setup_agents and setup_agent1 and setup_agent2:
                print(f"   - {setup_agent1_final_path} (Setup Agent NN)")
                print(f"   - {setup_agent2_final_path} (Setup Agent NN)")
        else:
            print(f"\n⚠️  Warning: Final model files may not have been created")
    except Exception as e:
        print(f"\n⚠️  Error saving final models: {e}")
        traceback.print_exc()
    
    # Training completed
    print("Training environment closed")
    
    for prefetcher in prefetchers:
        prefetcher.stop()
    
    return agent1, agent2


def main():
    """Main function to run DQN training"""
    print("🎮 DQN Agent Training for Stratego")
    print("=" * 50)
    
    # Training parameters
    num_episodes = 4000  # More than 500 as requested
    save_interval = 100
    model_save_path = "dqn_models"
    use_setup_agents = True  # Enable setup agents for piece placement
    generate_gifs = False  # Set to True to enable GIF generation for episode 1 and winning games
    
    try:
        # Train agents with single environment
        agent1, agent2 = train_dqn_agents(num_episodes, save_interval, model_save_path,
                                          use_setup_agents=use_setup_agents,
                                          generate_gifs=generate_gifs)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
