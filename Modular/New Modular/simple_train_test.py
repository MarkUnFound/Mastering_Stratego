"""
Simple test script to verify DQN training works
"""

import torch
import sys
import os

# Add the stratego_modular directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stratego_modular'))

from stratego_modular.environment import StrategoEnvironment
from stratego_modular.dqn_agent import DQNAgent

def simple_training_test():
    """Simple test to verify training works"""
    print("Testing simple DQN training...")
    
    # Set up device
    device = torch.device('cpu')  # Use CPU for simplicity
    print(f"Using device: {device}")
    
    # Create environment
    env = StrategoEnvironment(device=device)
    env.reset()
    game_state = env._get_game_state()
    
    # Create agents
    agent1 = DQNAgent(player_id=1, device=device)
    agent2 = DQNAgent(player_id=-1, device=device)
    
    print(f"Agent 1 name: {agent1.name}")
    print(f"Agent 2 name: {agent2.name}")
    
    # Run a few moves to test training
    done = False
    move_count = 0
    max_moves = 10
    
    # Get initial state representations
    state1 = agent1.get_state_representation(game_state)
    state2 = agent2.get_state_representation(game_state)
    
    print(f"Initial states created successfully")
    
    while not done and move_count < max_moves:
        # Determine current player and agent
        current_agent = agent1 if env.current_player == 1 else agent2
        current_state = state1 if env.current_player == 1 else state2
        
        # Get valid moves
        valid_moves = env.get_valid_moves()
        
        if not valid_moves:
            print("No valid moves, ending test")
            break
        
        # Agent selects action
        action = current_agent.act(current_state, valid_moves)
        
        if action is None:
            print("Invalid action, ending test")
            break
        
        # Execute action
        next_game_state, reward, done, _ = env.step(action)
        
        # Get next state representation
        next_state = current_agent.get_state_representation(next_game_state)
        
        # Store experience
        current_agent.remember(current_state, 
                             current_agent._move_to_action_index(action),
                             reward,
                             next_state,
                             done)
        
        # Update states
        if env.current_player == 1:
            state1 = next_state
        else:
            state2 = next_state
        
        move_count += 1
        print(f"Move {move_count} completed: {action}")
        
        # Train agents periodically
        if move_count % 4 == 0:  # Train every 4 moves
            agent1.replay()
            agent2.replay()
            print(f"Agents trained at move {move_count}")
    
    print("✅ Simple training test completed successfully!")

if __name__ == "__main__":
    simple_training_test()
