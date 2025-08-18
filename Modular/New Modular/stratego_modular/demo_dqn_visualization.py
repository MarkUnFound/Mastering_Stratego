"""
Demo script for DQN Agent Move Visualization in Stratego
"""

import sys
import os
import torch

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .environment import StrategoEnvironment
from .dqn_visualizer import DQNMoveVisualizer


def demo_dqn_visualization():
    """Demonstrate the DQN move visualization feature."""
    print("=== DQN Agent Move Visualization Demo ===\n")
    
    # Initialize environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = StrategoEnvironment(device)
    
    # Reset environment to start a new game
    state = env.reset()
    print(f"Environment reset. Current player: {env.current_player}")
    print(f"Game over: {env.game_over}")
    
    # Get some valid moves to demonstrate
    valid_moves = env.get_valid_moves()
    print(f"\nFound {len(valid_moves)} valid moves for Player 1")
    
    if valid_moves:
        # Execute a few sample moves
        sample_moves = valid_moves[:3]  # Take first 3 moves
        
        # Add a repeated move to demonstrate penalty
        if len(sample_moves) >= 2:
            repeated_move = sample_moves[0]  # Use the first move again
            sample_moves.append(repeated_move)  # Add it again to create repetition
        
        for i, move in enumerate(sample_moves):
            print(f"\n--- Executing Move {i+1} ---")
            print(f"Move: {move[0]} to {move[1]}")
            
            # Execute move
            state, reward, done, info = env.step(move)
            print(f"Reward: {reward:.3f}")
            print(f"Game over: {done}")
            
        # Visualize move history with improved formatting
        print("\n--- Move History ---")
        env.visualize_moves()
        
        # Visualize specific move
        print("\n--- Visualizing Specific Move ---")
        env.visualize_moves(move_index=1, save_path="sample_move.png")
        
        # Create GIF of all moves
        print("\n--- Creating GIF Animation ---")
        env.dqn_visualizer.create_move_gif("game_demo.gif", duration=1.0)
        
        # Clear history
        print("\n--- Clearing Move History ---")
        env.clear_move_history()
        env.visualize_moves()  # Should show no moves
        
    else:
        print("No valid moves found.")
        
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_dqn_visualization()
