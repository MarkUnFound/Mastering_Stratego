"""
Training Script for DQN Agents in Stratego
"""

import torch
import numpy as np
import random
import os
import sys
from typing import List, Tuple

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratego_modular.environment import StrategoEnvironment
from stratego_modular.dqn_agent import DQNAgent
from stratego_modular.game_state import GameState

# Import reset function (optional)
try:
    from stratego_modular.reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False


def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     model_save_path: str = "dqn_models"):
    """Train two DQN agents through self-play"""
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = StrategoEnvironment(device=device)
    
    # Create agents
    agent1 = DQNAgent(player_id=1, device=device, lr=0.001)
    agent2 = DQNAgent(player_id=-1, device=device, lr=0.001)
    
    # Create model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Training metrics
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    total_rewards_agent1 = []
    total_rewards_agent2 = []
    
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
        # Reset environment
        env.reset()
        game_state = env._get_game_state()
        done = False
        move_count = 0
        max_moves = 500  # Prevent infinite games
        
        # Episode rewards
        episode_reward_agent1 = 0
        episode_reward_agent2 = 0
        
        # Get initial state representations
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
                
            # Agent selects action
            action = current_agent.act(current_state, valid_moves)
            
            if action is None:
                # Invalid action, game ends
                done = True
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
                
            move_count += 1
            
            # Train agents periodically
            if move_count % 4 == 0:  # Train every 4 moves
                agent1.replay()
                agent2.replay()
                
        # Game finished
        if game_state.winner == 1:
            wins_agent1 += 1
            # Give positive reward to winner, negative to loser
            agent1.remember(state1, agent1._move_to_action_index(action), 10.0, next_state, True)
            agent2.remember(state2, agent2._move_to_action_index(action), -10.0, next_state, True)
        elif game_state.winner == -1:
            wins_agent2 += 1
            # Give positive reward to winner, negative to loser
            agent2.remember(state2, agent2._move_to_action_index(action), 10.0, next_state, True)
            agent1.remember(state1, agent1._move_to_action_index(action), -10.0, next_state, True)
        else:
            draws += 1
            # Give small reward to both for draw
            agent1.remember(state1, agent1._move_to_action_index(action), 1.0, next_state, True)
            agent2.remember(state2, agent2._move_to_action_index(action), 1.0, next_state, True)
            
        # Update target networks periodically
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            
        # Store episode rewards
        total_rewards_agent1.append(episode_reward_agent1)
        total_rewards_agent2.append(episode_reward_agent2)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward1 = np.mean(total_rewards_agent1[-50:]) if total_rewards_agent1 else 0
            avg_reward2 = np.mean(total_rewards_agent2[-50:]) if total_rewards_agent2 else 0
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Agent 1 wins: {wins_agent1}, Agent 2 wins: {wins_agent2}, Draws: {draws}")
            print(f"  Avg Reward Agent 1 (last 50): {avg_reward1:.2f}")
            print(f"  Avg Reward Agent 2 (last 50): {avg_reward2:.2f}")
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
            
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            # Create model save directory if it doesn't exist
            os.makedirs(model_save_path, exist_ok=True)
            try:
                agent1.save_model(f"{model_save_path}/agent1_episode_{episode + 1}.pth")
                agent2.save_model(f"{model_save_path}/agent2_episode_{episode + 1}.pth")
                print(f"Models saved at episode {episode + 1}")
            except Exception as e:
                print(f"⚠️  Error saving models at episode {episode + 1}: {e}")
            
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
        agent1.save_model(f"{model_save_path}/agent1_final.pth")
        agent2.save_model(f"{model_save_path}/agent2_final.pth")
        print(f"\nFinal models saved to {model_save_path}/")
    except Exception as e:
        print(f"\n⚠️  Error saving final models: {e}")
    
    # Training completed
    print("Training environment closed")
    
    return agent1, agent2


def main():
    """Main function to run DQN training"""
    print("🎮 DQN Agent Training for Stratego")
    print("=" * 50)
    
    # Training parameters
    num_episodes = 1000  # More than 500 as requested
    save_interval = 100
    model_save_path = "dqn_models"
    
    try:
        # Train agents
        agent1, agent2 = train_dqn_agents(num_episodes, save_interval, model_save_path)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
