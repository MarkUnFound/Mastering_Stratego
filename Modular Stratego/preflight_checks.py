import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from training_visualizer import plot_training_progress, plot_pbs_evaluator_progress, plot_additional_metrics
from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from training_config import NUM_ENVS

def run_preflight_checks(model_save_path: str):
    print("\n✈️  Running Pre-Flight Checks...")
    os.makedirs(model_save_path, exist_ok=True)

    # 1. Test Plotting Functions
    print("   Testing plotting functions...")
    try:
        _test_plotting(model_save_path)
        print("   ✅ Plotting functions verified.")
    except Exception as e:
        print(f"   ❌ Plotting check failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 2. Test Game Logic (Simulation)
    print("   Testing game logic (simulation)...")
    try:
        _test_game_simulation()
        print("   ✅ Game logic verified (no illegal moves in demo).")
    except Exception as e:
        print(f"   ❌ Game logic check failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("✅ All Pre-Flight Checks Passed!")
    return True

def _test_plotting(save_path):
    # Create dummy data
    episodes = list(range(1, 11))
    rewards = {'agent1': [0.1 * i for i in episodes], 'agent2': [0.05 * i for i in episodes]}
    wins = {'agent1': [i for i in episodes], 'agent2': [0 for i in episodes], 'draws': [0 for i in episodes]}
    losses = {'agent1': [1.0 / i for i in episodes], 'agent2': [0.8 / i for i in episodes]}
    
    # Test plot_training_progress
    plot_training_progress(episodes, rewards, wins, losses, f"{save_path}/test_training_progress.png")
    

    
    # Test PBS evaluator progress
    pbs_losses1 = [0.3 / i for i in episodes]
    pbs_losses2 = [0.25 / i for i in episodes]
    pbs_buffers1 = [100 * i for i in episodes]
    pbs_buffers2 = [90 * i for i in episodes]
    plot_pbs_evaluator_progress(episodes, pbs_losses1, pbs_losses2, pbs_buffers1, pbs_buffers2, f"{save_path}/test_pbs_progress.png")
    
    # Test additional metrics
    epsilon = {'agent1': [1.0 - 0.1 * i for i in episodes], 'agent2': [1.0 - 0.1 * i for i in episodes]}
    avg_q = {'agent1': [0.5 * i for i in episodes], 'agent2': [0.4 * i for i in episodes]}
    entropy = {'agent1': [2.0 - 0.2 * i for i in episodes], 'agent2': [1.8 - 0.15 * i for i in episodes]}
    pbs_buffers = {'agent1': pbs_buffers1, 'agent2': pbs_buffers2}
    plot_additional_metrics(episodes, epsilon, pbs_buffers, avg_q, entropy, f"{save_path}/test_metrics.png")

def _test_game_simulation():
    # Initialize environment and agents
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {device}")

    # Create dummy agents (minimal config)
    print("      Instantiating RainbowAgent to verify class...")
    try:
        agent = RainbowAgent(player_id=1, device=device, num_envs=1)
        print("      RainbowAgent instantiated successfully.")
    except Exception as e:
        print(f"      Failed to instantiate RainbowAgent: {e}")
        raise
    
    # Use single environment for testing
    env = StrategoEnvironment(device=device)
    
    print("      Resetting environment...")
    obs = env.reset()
    
    print("      Running simulation steps...")
    # Run a few steps
    for step in range(20):
        if env.game_over:
            print(f"      Game over at step {step}")
            break
            
        # Get valid moves to ensure we don't pass invalid ones
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print(f"      No valid moves at step {step}")
            break
            
        # Pick a random valid move
        action = valid_moves[np.random.randint(len(valid_moves))]
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Basic assertions
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert isinstance(done, bool), f"Done should be bool, got {type(done)}"
        
        if done:
            print(f"      Game finished at step {step}")
            break
            
    print("      Simulation completed.")

if __name__ == "__main__":
    run_preflight_checks("models/test_preflight")
