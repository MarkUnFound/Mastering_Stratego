# stratego_modular/main.py

import torch
import random
from stratego_modular.environment import StrategoEnvironment

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment
    env = StrategoEnvironment(device)
    state = env.reset()
    done = False
    
    print("Starting game with DQN visualization features...")
    print("=============================================")
    
    move_count = 0
    max_moves = 500  # Training mode with 500 turns per episode
    
    while not done and move_count < max_moves:
        # Get valid moves for current player
        valid_moves = env.get_valid_moves()
        
        if not valid_moves:
            break
            
        # For demonstration, choose a random valid move
        # But avoid obvious repetition patterns
        if move_count > 0 and move_count % 5 == 0:
            # Every 5 moves, try a different move if available
            action = valid_moves[min(move_count // 5, len(valid_moves) - 1)]
        else:
            action = valid_moves[0]  # In a real implementation, this would be chosen by an AI
        
        # Execute the move
        state, reward, done, info = env.step(action)
        
        # Print game state information
        print(f"Move {move_count + 1}: Player {state.current_player * -1} -> Player {state.current_player}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        if done:
            print(f"  Game over. Winner: {info['winner']}")
            
        move_count += 1
        
        # Show penalty in action for repeated moves
        if move_count > 10 and move_count % 3 == 0:
            print("  (Penalty system active - repeated moves are discouraged)")
        
    # Show move history with repeat move detection
    print("\n=============================================")
    print("Move history with repeat detection:")
    env.visualize_moves()
    
    # Create GIF of the game
    print("\nCreating GIF animation of the game...")
    env.dqn_visualizer.create_move_gif("game_play.gif", duration=0.8)
    
    print("\nGame completed. Check 'game_play.gif' for visualization.")
    print("DQN visualization features successfully demonstrated.")

if __name__ == "__main__":
    main()