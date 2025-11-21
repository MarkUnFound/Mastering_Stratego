"""
Test script to run a single game and generate a GIF to verify legal moves.
This will help us check if the models are playing legally after code changes.
"""

import torch
import os
import sys
from typing import List, Dict, Tuple

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dqn_agent import DQNAgent
from game_state import GameState
from environment import StrategoEnvironment
from piece import PieceType, PIECE_NAMES
from board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from dqn_visualizer import DQNMoveVisualizer
from training_visualizer import create_episode_gif


def find_latest_model():
    """Find the latest trained model in the dqn_models directory."""
    model_dir = "dqn_models"
    if not os.path.exists(model_dir):
        return None, None
    
    # Look for episode-specific models
    agent1_models = []
    agent2_models = []
    
    for filename in os.listdir(model_dir):
        if filename.startswith("agent1_episode_") and filename.endswith(".pth"):
            episode_num = int(filename.split("_")[2].split(".")[0])
            agent1_models.append((episode_num, os.path.join(model_dir, filename)))
        elif filename.startswith("agent2_episode_") and filename.endswith(".pth"):
            episode_num = int(filename.split("_")[2].split(".")[0])
            agent2_models.append((episode_num, os.path.join(model_dir, filename)))
    
    if agent1_models and agent2_models:
        # Get the latest episode
        agent1_models.sort(reverse=True)
        agent2_models.sort(reverse=True)
        latest_episode = agent1_models[0][0]
        agent1_path = agent1_models[0][1]
        agent2_path = agent2_models[0][1]
        return latest_episode, (agent1_path, agent2_path)
    
    # Fallback to final models
    agent1_final = os.path.join(model_dir, "agent1_final.pth")
    agent2_final = os.path.join(model_dir, "agent2_final.pth")
    
    if os.path.exists(agent1_final) and os.path.exists(agent2_final):
        return "final", (agent1_final, agent2_final)
    
    return None, None


def check_move_legality(env: StrategoEnvironment, move: Tuple, game_state: GameState) -> Tuple[bool, str]:
    """
    Check if a move is legal and return detailed information.
    
    Returns:
        (is_legal, reason) - True if legal, False otherwise with reason
    """
    (r_from, c_from), (r_to, c_to) = move
    board = env.board.actual_board
    
    # Get piece at source
    piece_value = board[r_from, c_from].item()
    if piece_value == EMPTY_SQUARE:
        return False, f"No piece at source ({r_from}, {c_from})"
    
    piece_type = abs(piece_value)
    owner = 1 if piece_value > 0 else -1
    
    # Check if it's the correct player's turn
    if owner != env.current_player:
        return False, f"Wrong player - piece belongs to {owner}, current player is {env.current_player}"
    
    # Check if piece can move (bombs and flags cannot move)
    if piece_type == PieceType.BOMB.value:
        return False, "Bombs cannot move"
    if piece_type == PieceType.FLAG.value:
        return False, "Flags cannot move"
    
    # Check destination
    dest_value = board[r_to, c_to].item()
    
    # Cannot move to lake
    if dest_value == LAKE_SQUARE:
        return False, f"Cannot move to lake at ({r_to}, {c_to})"
    
    # Cannot capture own piece
    if dest_value != EMPTY_SQUARE:
        dest_owner = 1 if dest_value > 0 else -1
        if dest_owner == owner:
            return False, f"Cannot capture own piece at ({r_to}, {c_to})"
    
    # Check movement distance
    dr = abs(r_to - r_from)
    dc = abs(c_to - c_from)
    
    # Must be orthogonal
    if dr > 0 and dc > 0:
        return False, "Diagonal moves not allowed"
    
    # Scouts can move multiple squares, others only 1
    if piece_type == PieceType.SCOUT.value:
        # Check if path is clear for scout
        if dr > 0:  # Vertical movement
            step = 1 if r_to > r_from else -1
            for r in range(r_from + step, r_to, step):
                if board[r, c_from].item() != EMPTY_SQUARE:
                    return False, f"Scout path blocked at ({r}, {c_from})"
        else:  # Horizontal movement
            step = 1 if c_to > c_from else -1
            for c in range(c_from + step, c_to, step):
                if board[r_from, c].item() != EMPTY_SQUARE:
                    return False, f"Scout path blocked at ({r_from}, {c})"
    else:
        # Non-scouts can only move 1 square
        if dr + dc != 1:
            return False, f"Non-scout piece can only move 1 square, tried to move {dr + dc} squares"
    
    return True, "Legal move"


def run_test_game(generate_gif: bool = True, max_moves: int = 500):
    """
    Run a single test game and check for illegal moves.
    
    Args:
        generate_gif: Whether to generate a GIF of the game
        max_moves: Maximum number of moves before declaring a draw
    """
    print("=" * 80)
    print("🎮 STRATEGO GAME LEGALITY TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Find latest model
    episode, model_paths = find_latest_model()
    
    if model_paths is None:
        print("⚠️  No trained models found! Using untrained agents.")
        agent1 = DQNAgent(1, device)
        agent2 = DQNAgent(-1, device)
    else:
        print(f"✅ Loading models from episode {episode}")
        agent1 = DQNAgent(1, device)
        agent2 = DQNAgent(-1, device)
        
        try:
            agent1.load_model(model_paths[0])
            agent2.load_model(model_paths[1])
            print(f"   Agent 1: {model_paths[0]}")
            print(f"   Agent 2: {model_paths[1]}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("   Using untrained agents instead")
    
    # Set agents to exploitation mode (no exploration)
    agent1.epsilon = 0.0
    agent2.epsilon = 0.0
    
    print("\n🎯 Starting game...")
    print(f"   Max moves: {max_moves}")
    print(f"   Generate GIF: {generate_gif}")
    
    # Create environment
    env = StrategoEnvironment(device=device)
    game_state = env.reset()
    
    # Track game states for GIF
    game_states = []
    illegal_moves = []
    move_count = 0
    
    # Create visualizer
    visualizer = DQNMoveVisualizer()
    
    print("\n" + "=" * 80)
    print("🏁 GAME START")
    print("=" * 80)
    
    # Game loop
    while not game_state.game_over and move_count < max_moves:
        current_agent = agent1 if env.current_player == 1 else agent2
        valid_moves = env.get_valid_moves()
        
        if not valid_moves:
            print(f"\n❌ No valid moves for Player {env.current_player}")
            break
        
        # Get agent's action
        state_representation = current_agent.get_state_representation(game_state)
        action = current_agent.act(state_representation, valid_moves)
        
        if action is None:
            print(f"\n❌ Agent returned None action")
            break
        
        # Check if the move is legal
        is_legal, reason = check_move_legality(env, action, game_state)
        
        (r_from, c_from), (r_to, c_to) = action
        piece_value = env.board.actual_board[r_from, c_from].item()
        piece_type = abs(piece_value)
        
        try:
            piece_name = PIECE_NAMES[PieceType(piece_type)]
        except:
            piece_name = f"Unknown({piece_type})"
        
        # Print move info
        move_symbol = "✅" if is_legal else "❌"
        print(f"{move_symbol} Move {move_count + 1}: Player {env.current_player} - {piece_name} from ({r_from},{c_from}) to ({r_to},{c_to})")
        
        if not is_legal:
            print(f"   ⚠️  ILLEGAL MOVE: {reason}")
            illegal_moves.append({
                'move_num': move_count + 1,
                'player': env.current_player,
                'action': action,
                'piece': piece_name,
                'reason': reason
            })
        
        # Record state for GIF
        if generate_gif:
            game_states.append({
                'board': env.board.actual_board.clone(),
                'move_num': move_count,
                'last_move': action
            })
            visualizer.record_move(action, game_state, env.current_player)
        
        # Execute move
        try:
            game_state, reward, done, info = env.step(action)
            move_count += 1
        except Exception as e:
            print(f"   ❌ Error executing move: {e}")
            illegal_moves.append({
                'move_num': move_count + 1,
                'player': env.current_player,
                'action': action,
                'piece': piece_name,
                'reason': f"Exception: {e}"
            })
            break
        
        if done:
            break
    
    # Game finished
    print("\n" + "=" * 80)
    print("🏁 GAME FINISHED")
    print("=" * 80)
    
    # Print results
    if game_state.winner == 1:
        print(f"🏆 Winner: Player 1 (Agent 1)")
    elif game_state.winner == -1:
        print(f"🏆 Winner: Player 2 (Agent 2)")
    else:
        print(f"🤝 Game ended in a draw")
    
    print(f"📊 Total moves: {move_count}")
    print(f"📊 Illegal moves detected: {len(illegal_moves)}")
    
    # Print illegal moves summary
    if illegal_moves:
        print("\n" + "=" * 80)
        print("⚠️  ILLEGAL MOVES SUMMARY")
        print("=" * 80)
        for illegal in illegal_moves:
            print(f"Move {illegal['move_num']}: Player {illegal['player']} - {illegal['piece']}")
            print(f"   Action: {illegal['action']}")
            print(f"   Reason: {illegal['reason']}")
            print()
    else:
        print("\n✅ ALL MOVES WERE LEGAL!")
    
    # Generate GIF
    if generate_gif and game_states:
        print("\n" + "=" * 80)
        print("🎬 GENERATING GIF")
        print("=" * 80)
        
        gif_dir = "test_gifs"
        os.makedirs(gif_dir, exist_ok=True)
        
        gif_path = os.path.join(gif_dir, f"legality_test_episode_{episode}.gif")
        
        try:
            create_episode_gif(game_states, episode if isinstance(episode, int) else 0, gif_path, frame_duration=1000)
            print(f"✅ GIF saved to: {gif_path}")
        except Exception as e:
            print(f"❌ Error creating GIF: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    
    return len(illegal_moves) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test game legality and generate GIF")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--max-moves", type=int, default=500, help="Maximum moves before draw")
    
    args = parser.parse_args()
    
    success = run_test_game(generate_gif=not args.no_gif, max_moves=args.max_moves)
    
    sys.exit(0 if success else 1)
