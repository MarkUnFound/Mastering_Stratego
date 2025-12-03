"""
Script to reset DQN agents
"""

import torch
import os
import sys

# Ensure the current module directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from drqn_agent import DRQNAgent


def reset_dqn_agents(device=None, lr=0.001):
    """Reset DQN agents by reinitializing their networks and parameters
    
    Args:
        device: PyTorch device to use (if None, will auto-detect)
        lr: Learning rate for the new agents
        
    Returns:
        tuple: (agent1, agent2) New agent instances
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create new agents with default parameters
    agent1 = DRQNAgent(player_id=1, device=device, lr=lr)
    agent2 = DRQNAgent(player_id=-1, device=device, lr=lr)
    
    # Reset the agents
    agent1.reset()
    agent2.reset()
    
    print("✅ DQN agents have been reset successfully!")
    print("   - Neural networks reinitialized")
    print("   - Optimizer states cleared")
    print("   - Experience replay buffers cleared")
    print("   - Exploration parameters reset to defaults")
    
    return agent1, agent2


def reset_existing_agents(agent1, agent2):
    """Reset existing agent instances in place
    
    Args:
        agent1: First DQN agent to reset
        agent2: Second DQN agent to reset
    """
    agent1.reset()
    agent2.reset()
    
    print("✅ Existing DQN agents have been reset successfully!")
    print("   - Neural networks reinitialized")
    print("   - Optimizer states cleared")
    print("   - Experience replay buffers cleared")
    print("   - Exploration parameters reset to defaults")


def main():
    """Main function to reset DQN agents"""
    print("🔄 Resetting DQN Agents for Stratego")
    print("=" * 40)
    
    try:
        # Create and reset new agents
        agent1, agent2 = reset_dqn_agents()
        print("\n✅ Reset completed successfully!")
        print("\nTo use these agents in your training script, import this function and replace your existing agents:")
        print("   from reset_dqn import reset_dqn_agents")
        print("   new_agent1, new_agent2 = reset_dqn_agents(device, lr)")
        print("   agent1, agent2 = new_agent1, new_agent2")
        
    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
