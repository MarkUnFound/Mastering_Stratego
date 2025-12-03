import torch
from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from drqn_agent import RainbowAgent

def test_env():
    print("Testing StrategoEnvironment...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        env = StrategoEnvironment(device=device)
        print("Environment initialized.")
        
        print("Resetting environment...")
        obs = env.reset()
        print("Environment reset successfully.")
        
        print("Running simulation loop...")
        import numpy as np
        for step in range(20):
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            action = valid_moves[np.random.randint(len(valid_moves))]
            next_obs, reward, done, info = env.step(action)
            if done:
                break
        print("Simulation loop completed.")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()
