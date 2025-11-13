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
from typing import List, Tuple, Optional

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratego_modular.environment import StrategoEnvironment
from stratego_modular.dqn_agent import DQNAgent
from stratego_modular.setup_agent import SetupAgent
from stratego_modular.game_state import GameState
from stratego_modular.training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_setup_agent_progress
from stratego_modular.pbs_visualizer import visualize_pbs_state
from stratego_modular.piece import PieceType, PIECE_RANKS

# Import reset function (optional)
try:
    from stratego_modular.reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False


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
    
    # 1. Flag protection reward (0.0 to 1.0, scaled to 0-5.0)
    protection_score = evaluate_flag_protection(placement, player_id)
    reward += protection_score * 5.0  # Scale to 0-5.0
    
    # 2. Game length reward/penalty
    if move_count < min_survival_moves:
        # Penalty for short games (games that end too quickly)
        # Linear penalty: -0.1 per move below threshold
        penalty = -0.1 * (min_survival_moves - move_count)
        reward += penalty
    else:
        # Reward for surviving longer (games that last at least min_survival_moves)
        # Small reward for each move above threshold
        bonus = 0.01 * (move_count - min_survival_moves)
        reward += bonus
    
    # 3. Win/loss reward
    if winner == player_id:
        # Big reward for winning
        reward += 10.0
    elif winner is not None and winner != player_id:
        # Penalty for losing (but less severe than short game penalty)
        reward -= 2.0
    else:
        # Small reward for draw
        reward += 1.0
    
    # 4. Piece distribution bonus (0.0 to 1.0, scaled to 0-2.0)
    distribution_score = evaluate_piece_distribution(placement, player_id)
    reward += distribution_score * 2.0
    
    # 5. Scout placement reward (0.0 to 1.0, scaled to 0-1.5)
    scout_score = evaluate_scout_placement(placement, player_id)
    reward += scout_score * 1.5
    
    # 6. Bomb placement reward (0.0 to 1.0, scaled to 0-2.0)
    bomb_score = evaluate_bomb_placement(placement, player_id)
    reward += bomb_score * 2.0
    
    # 7. Defensive formation reward (0.0 to 1.0, scaled to 0-1.5)
    formation_score = evaluate_defensive_formation(placement, player_id)
    reward += formation_score * 1.5
    
    # 8. Piece coordination reward (0.0 to 1.0, scaled to 0-1.0)
    coordination_score = evaluate_piece_coordination(placement, player_id)
    reward += coordination_score * 1.0
    
    # 9. Early game survival bonus (extra reward if flag survives first 50 moves)
    if move_count >= 50:
        # Bonus for surviving early game
        reward += 2.0
    
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
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimize GPU settings for better performance
    if device.type == 'cuda':
        # Enable TensorFloat32 (TF32) for faster float32 matrix multiplication on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        if hasattr(torch, 'compile'):
            print("✅ torch.compile available - networks will be optimized (if Triton is available)")
        else:
            print("⚠️  torch.compile not available - upgrade to PyTorch 2.0+ for better GPU utilization")
    
    # Create environment
    env = StrategoEnvironment(device=device)
    
    # Create game-playing agents
    agent1 = DQNAgent(player_id=1, device=device, lr=0.001)
    agent2 = DQNAgent(player_id=-1, device=device, lr=0.001)
    
    # Create setup agents (for piece placement)
    setup_agent1 = SetupAgent(player_id=1, device=device) if use_setup_agents else None
    setup_agent2 = SetupAgent(player_id=-1, device=device) if use_setup_agents else None
    
    # Memory buffer for winning games (for GIF generation)
    winning_games_buffer = []
    
    # Create model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Training metrics
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    total_rewards_agent1 = []
    total_rewards_agent2 = []

    # History for plotting
    episode_history = []
    rewards_history = {'agent1': [], 'agent2': []}
    wins_history = {'agent1': [], 'agent2': [], 'draws': []}
    epsilon_history = {'agent1': [], 'agent2': []}
    policy_loss_history = {'agent1': [], 'agent2': []}
    
    # Setup agent history for plotting
    setup_agent1_rewards = []
    setup_agent2_rewards = []
    setup_agent1_losses = []
    setup_agent2_losses = []
    
    print(f"Starting DQN training for {num_episodes} episodes...")
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
        episode_game_states = []
        
        # Track PBS captures for checkpoint episodes (multiples of 50)
        # Capture PBS at move 50 and end of game for episodes 50, 100, 150, etc.
        is_checkpoint_episode = (episode + 1) % 50 == 0
        captured_move_50_pbs = False  # Track if we've captured PBS at move 50
        
        # Episode rewards
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0
        
        # Get initial state representations (GPU tensors)
        state1 = agent1.get_state_representation(game_state)
        state2 = agent2.get_state_representation(game_state)
        
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
            
            # Record game state ONLY for potential winning games (we'll check at the end)
            # We record all moves but will only create GIF if it's a winning game
            if generate_gifs:
                current_board = env.board.actual_board.clone() if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                if current_board is not None:
                    episode_game_states.append({
                        'board': current_board,
                        'move_num': move_count + 1,
                        'last_move': action
                    })
            
            # Determine game phase for PBS evaluator data collection
            # Early game: turns 0-50, Middle game: turns 51-200, End game: turns 201+
            game_phase = 'early'
            if move_count > 200:
                game_phase = 'end'
            elif move_count > 50:
                game_phase = 'middle'
            
            # Update PBS from revealed pieces (after battle)
            # Check if pieces were revealed in the battle
            if hasattr(env, 'revealed_pieces_p1') and hasattr(env, 'revealed_pieces_p2'):
                # Update agent1's PBS with revealed pieces
                for pos, piece_value in env.revealed_pieces_p1.items():
                    if pos not in agent1.pbs.revealed_pieces if agent1.pbs else True:
                        from stratego_modular.piece import PieceType
                        piece_type = PieceType(abs(piece_value))
                        agent1.update_pbs_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=move_count)
                # Update agent2's PBS with revealed pieces
                for pos, piece_value in env.revealed_pieces_p2.items():
                    if pos not in agent2.pbs.revealed_pieces if agent2.pbs else True:
                        from stratego_modular.piece import PieceType
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
            
            # Capture PBS at move 50 for checkpoint episodes (multiples of 50) if game hasn't ended
            if is_checkpoint_episode and move_count == 50 and not done and not captured_move_50_pbs:
                try:
                    agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') and agent1.pbs else None
                    agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') and agent2.pbs else None
                    
                    # Get visible boards for each player
                    visible_board_p1 = None
                    visible_board_p2 = None
                    if hasattr(env, 'board') and hasattr(env.board, 'visible_board_p1'):
                        visible_board_p1 = env.board.visible_board_p1.clone()
                        visible_board_p2 = env.board.visible_board_p2.clone()
                    
                    # Get actual board
                    current_actual_board = env.board.actual_board.clone() if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
                    
                    if current_actual_board is not None:
                        pbs_save_path = f"{model_save_path}/pbs_visualization_episode_{episode + 1}_move_50.png"
                        visualize_pbs_state(
                            actual_board=current_actual_board,
                            agent1_pbs=agent1_pbs,
                            agent2_pbs=agent2_pbs,
                            episode=episode + 1,
                            save_path=pbs_save_path,
                            visible_board_p1=visible_board_p1,
                            visible_board_p2=visible_board_p2
                        )
                        print(f"🎯 PBS visualization saved at move 50 of episode {episode + 1}: {pbs_save_path}")
                        captured_move_50_pbs = True
                except Exception as e:
                    print(f"⚠️  Error creating PBS visualization at move 50 of episode {episode + 1}: {e}")
            
            # Train agents periodically
            if move_count % 4 == 0:  # Train every 4 moves
                agent1.replay()
                agent2.replay()
                
        # Game finished - get final game state for PBS visualization
        # Use the last game state (next_game_state from the loop, or get it from env)
        final_game_state = next_game_state if 'next_game_state' in locals() else env._get_game_state()
        actual_board = env.board.actual_board if hasattr(env, 'board') and hasattr(env.board, 'actual_board') else None
        
        # Collect end-game PBS data for all remaining pieces (ground truth from actual board)
        if actual_board is not None and move_count > 50:  # Only collect if game progressed past early phase
            from stratego_modular.piece import PieceType
            # Collect data for agent1's PBS (opponent pieces are player -1)
            if agent1.pbs and agent1.pbs.evaluator:
                for r in range(10):
                    for c in range(10):
                        pos = (r, c)
                        piece_value = actual_board[r, c].item()
                        # Check if this is an opponent piece (negative for player 1)
                        if piece_value < 0 and piece_value != -13:  # Not empty or lake
                            if pos not in agent1.pbs.revealed_pieces and pos in agent1.pbs.belief_distributions:
                                piece_type = PieceType(abs(piece_value))
                                agent1.pbs.update_from_reveal(pos, piece_type, game_phase='end', turn_count=move_count)
            
            # Collect data for agent2's PBS (opponent pieces are player 1)
            if agent2.pbs and agent2.pbs.evaluator:
                for r in range(10):
                    for c in range(10):
                        pos = (r, c)
                        piece_value = actual_board[r, c].item()
                        # Check if this is an opponent piece (positive for player -1)
                        if piece_value > 0:
                            if pos not in agent2.pbs.revealed_pieces and pos in agent2.pbs.belief_distributions:
                                piece_type = PieceType(abs(piece_value))
                                agent2.pbs.update_from_reveal(pos, piece_type, game_phase='end', turn_count=move_count)
        
        # Get winner from the final state or environment
        winner = final_game_state.winner if hasattr(final_game_state, 'winner') else (env.winner if hasattr(env, 'winner') else None)
        
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
            # Give small reward to both for draw
            agent1.remember(state1, agent1._move_to_action_index(action), 1.0, next_state, True)
            agent2.remember(state2, agent2._move_to_action_index(action), 1.0, next_state, True)
            
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
        
        # Store winning games and create GIFs ONLY for winning games
        if generate_gifs and is_winning_game and episode_game_states:
            winning_games_buffer.append({
                'episode': episode + 1,
                'winner': winner,
                'game_states': episode_game_states
            })
            
            # Create GIF for winning game
            try:
                episode_gif_path = f"{model_save_path}/episode_recording_win_{episode + 1}.gif"
                create_episode_gif(episode_game_states, episode + 1, episode_gif_path, frame_duration=750)
                print(f"✅ Created GIF for winning game at episode {episode + 1}")
            except Exception as e:
                print(f"⚠️  Error creating winning game GIF at episode {episode + 1}: {e}")
        elif generate_gifs and not is_winning_game:
            # Clear game states for non-winning games to save memory
            episode_game_states = []
        
        # Capture PBS at end of checkpoint episodes (multiples of 50)
        if is_checkpoint_episode and actual_board is not None:
            try:
                agent1_pbs = agent1.pbs if hasattr(agent1, 'pbs') and agent1.pbs else None
                agent2_pbs = agent2.pbs if hasattr(agent2, 'pbs') and agent2.pbs else None
                
                # Get visible boards for each player
                visible_board_p1 = None
                visible_board_p2 = None
                if hasattr(env, 'board') and hasattr(env.board, 'visible_board_p1'):
                    visible_board_p1 = env.board.visible_board_p1.clone()
                    visible_board_p2 = env.board.visible_board_p2.clone()
                
                # Create PBS visualization at end of checkpoint episode
                pbs_save_path = f"{model_save_path}/pbs_visualization_episode_{episode + 1}_end.png"
                visualize_pbs_state(
                    actual_board=actual_board,
                    agent1_pbs=agent1_pbs,
                    agent2_pbs=agent2_pbs,
                    episode=episode + 1,
                    save_path=pbs_save_path,
                    visible_board_p1=visible_board_p1,
                    visible_board_p2=visible_board_p2
                )
                print(f"🎯 PBS visualization saved at end of episode {episode + 1} (move {move_count}): {pbs_save_path}")
            except Exception as e:
                print(f"⚠️  Error creating PBS visualization at end of episode {episode + 1}: {e}")
        
        # Note: PBS visualization for checkpoint episodes (multiples of 50) is now handled above
        # It captures PBS at both move 50 (if game hasn't ended) and at end of game
            
        # Update target networks periodically
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
        
        # Train PBS evaluators periodically (only if they have collected data)
        if episode % 20 == 0 and episode > 0:
            # Train evaluators on collected data
            if agent1.pbs and agent1.pbs.evaluator:
                loss1 = agent1.pbs.train_evaluator(epochs=2)
                if loss1 is not None and episode % 100 == 0:
                    print(f"  PBS Evaluator 1 Loss: {loss1:.4f}")
            if agent2.pbs and agent2.pbs.evaluator:
                loss2 = agent2.pbs.train_evaluator(epochs=2)
                if loss2 is not None and episode % 100 == 0:
                    print(f"  PBS Evaluator 2 Loss: {loss2:.4f}")
            
            # Update target networks for evaluators
            if agent1.pbs and agent1.pbs.evaluator:
                agent1.pbs.evaluator.update_target_network()
            if agent2.pbs and agent2.pbs.evaluator:
                agent2.pbs.evaluator.update_target_network()
            
        # Store episode rewards
        total_rewards_agent1.append(episode_reward_agent1)
        total_rewards_agent2.append(episode_reward_agent2)
        
        # Update history for plotting every episode (discrete points)
        episode_history.append(episode + 1)
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
        
        # Track setup agent rewards and losses
        if use_setup_agents and setup_agent1:
            if len(setup_agent1.episode_rewards) > 0:
                setup_agent1_rewards.append(setup_agent1.episode_rewards[-1])
            avg_setup_loss1 = setup_agent1.get_average_policy_loss(100) if hasattr(setup_agent1, 'get_average_policy_loss') else 0.0
            setup_agent1_losses.append(avg_setup_loss1)
        
        if use_setup_agents and setup_agent2:
            if len(setup_agent2.episode_rewards) > 0:
                setup_agent2_rewards.append(setup_agent2.episode_rewards[-1])
            avg_setup_loss2 = setup_agent2.get_average_policy_loss(100) if hasattr(setup_agent2, 'get_average_policy_loss') else 0.0
            setup_agent2_losses.append(avg_setup_loss2)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward1 = np.mean(total_rewards_agent1[-50:]) if total_rewards_agent1 else 0
            avg_reward2 = np.mean(total_rewards_agent2[-50:]) if total_rewards_agent2 else 0
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Agent 1 wins: {wins_agent1}, Agent 2 wins: {wins_agent2}, Draws: {draws}")
            print(f"  Avg Reward Agent 1 (last 50): {avg_reward1:.2f}")
            print(f"  Avg Reward Agent 2 (last 50): {avg_reward2:.2f}")
            print(f"  Avg Policy Loss Agent 1 (last 100): {avg_loss1:.4f}")
            print(f"  Avg Policy Loss Agent 2 (last 100): {avg_loss2:.4f}")
            print(f"  Epsilon Agent 1: {agent1.epsilon:.3f}, Epsilon Agent 2: {agent2.epsilon:.3f}")
            print("-" * 60)
            
            # Reset agents if average reward is too large
            if abs(avg_reward1) > 50 or abs(avg_reward2) > 50:
                print("Average reward too large, resetting agents...")
                reset_agents()
                # Reset statistics
                wins_agent1 = 0
                wins_agent2 = 0
                draws = 0
                total_rewards_agent1 = []
                total_rewards_agent2 = []
                # Also reset history for plotting
                episode_history = []
                rewards_history = {'agent1': [], 'agent2': []}
                wins_history = {'agent1': [], 'agent2': [], 'draws': []}
                epsilon_history = {'agent1': [], 'agent2': []}
                policy_loss_history = {'agent1': [], 'agent2': []}
            
            # Save chart every 50 episodes
            # Save main DQN training progress (separate from setup agent progress)
            try:
                plot_training_progress(
                    episode_history,
                    rewards_history,
                    wins_history,
                    policy_loss_history,
                    save_path=f"{model_save_path}/training_progress_episode_{episode + 1}.png"
                )
                print(f"📈 Training progress graph saved to {model_save_path}/training_progress_episode_{episode + 1}.png")
            except Exception as e:
                print(f"⚠️  Error saving training progress graph at episode {episode + 1}: {e}")
                import traceback
                traceback.print_exc()
            
            # Save setup agent progress chart (separate PNG file, independent from main DQN plot)
            if use_setup_agents and setup_agent1 and setup_agent2:
                try:
                    plot_setup_agent_progress(
                        episode_history,
                        setup_agent1_rewards,
                        setup_agent2_rewards,
                        setup_agent1_losses,
                        setup_agent2_losses,
                        save_path=f"{model_save_path}/setup_agent_progress_episode_{episode + 1}.png"
                    )
                    print(f"📊 Setup agent progress graph saved to {model_save_path}/setup_agent_progress_episode_{episode + 1}.png")
                except Exception as e:
                    print(f"⚠️  Error saving setup agent progress graph: {e}")
                    import traceback
                    traceback.print_exc()
            
        # Save models periodically (keep at save_interval)
        if (episode + 1) % save_interval == 0:
            # Create model save directory if it doesn't exist
            os.makedirs(model_save_path, exist_ok=True)
            try:
                agent1_path = f"{model_save_path}/agent1_episode_{episode + 1}.pth"
                agent2_path = f"{model_save_path}/agent2_episode_{episode + 1}.pth"
                agent1.save_model(agent1_path)
                agent2.save_model(agent2_path)
                
                # Save setup agents if they exist
                if use_setup_agents and setup_agent1 and setup_agent2:
                    setup_agent1_path = f"{model_save_path}/setup_agent1_episode_{episode + 1}.pth"
                    setup_agent2_path = f"{model_save_path}/setup_agent2_episode_{episode + 1}.pth"
                    setup_agent1.save_model(setup_agent1_path)
                    setup_agent2.save_model(setup_agent2_path)
                
                # Verify files were created
                if os.path.exists(agent1_path) and os.path.exists(agent2_path):
                    print(f"💾 Models saved at episode {episode + 1}:")
                    print(f"   - {agent1_path} (includes DQN + PBS LSTM + PBS Evaluator NN)")
                    print(f"   - {agent2_path} (includes DQN + PBS LSTM + PBS Evaluator NN)")
                    if use_setup_agents and setup_agent1 and setup_agent2:
                        print(f"   - {setup_agent1_path} (Setup Agent NN)")
                        print(f"   - {setup_agent2_path} (Setup Agent NN)")
                else:
                    print(f"⚠️  Warning: Model files may not have been created at episode {episode + 1}")
            except Exception as e:
                print(f"⚠️  Error saving models at episode {episode + 1}: {e}")
                import traceback
                traceback.print_exc()
            
    # Final training metrics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total Episodes: {num_episodes}")
    print(f"Agent 1 Wins: {wins_agent1} ({wins_agent1/num_episodes*100:.1f}%)")
    print(f"Agent 2 Wins: {wins_agent2} ({wins_agent2/num_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.1f}%)")
    print(f"Average Reward Agent 1: {np.mean(total_rewards_agent1):.2f}")
    print(f"Average Reward Agent 2: {np.mean(total_rewards_agent2):.2f}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save final models
    try:
        agent1_final_path = f"{model_save_path}/agent1_final.pth"
        agent2_final_path = f"{model_save_path}/agent2_final.pth"
        agent1.save_model(agent1_final_path)
        agent2.save_model(agent2_final_path)
        
        # Save final setup agents if they exist
        if use_setup_agents and setup_agent1 and setup_agent2:
            setup_agent1_final_path = f"{model_save_path}/setup_agent1_final.pth"
            setup_agent2_final_path = f"{model_save_path}/setup_agent2_final.pth"
            setup_agent1.save_model(setup_agent1_final_path)
            setup_agent2.save_model(setup_agent2_final_path)
        
        # Verify files were created
        if os.path.exists(agent1_final_path) and os.path.exists(agent2_final_path):
            print(f"\n💾 Final models saved:")
            print(f"   - {agent1_final_path} (includes DQN + PBS LSTM + PBS Evaluator NN)")
            print(f"   - {agent2_final_path} (includes DQN + PBS LSTM + PBS Evaluator NN)")
            if use_setup_agents and setup_agent1 and setup_agent2:
                print(f"   - {setup_agent1_final_path} (Setup Agent NN)")
                print(f"   - {setup_agent2_final_path} (Setup Agent NN)")
        else:
            print(f"\n⚠️  Warning: Final model files may not have been created")
    except Exception as e:
        print(f"\n⚠️  Error saving final models: {e}")
        import traceback
        traceback.print_exc()
    
    # Training completed
    print("Training environment closed")
    
    return agent1, agent2


def main():
    """Main function to run DQN training"""
    print("🎮 DQN Agent Training for Stratego")
    print("=" * 50)
    
    # Training parameters
    num_episodes = 2000  # More than 500 as requested
    save_interval = 100
    model_save_path = "dqn_models"
    use_setup_agents = True  # Enable setup agents for piece placement
    generate_gifs = False  # Set to False to skip GIF generation overhead (GIFs created only for wins)
    
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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
