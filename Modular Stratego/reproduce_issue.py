
import torch
from environment import StrategoEnvironment

def test_env_init():
    print("Testing StrategoEnvironment initialization...")
    try:
        device = torch.device('cpu')
        env = StrategoEnvironment(device=device)
        print("Successfully initialized StrategoEnvironment")
        env.reset()
        print("Successfully reset environment")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env_init()
