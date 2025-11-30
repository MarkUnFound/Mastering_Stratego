import torch
from environment import StrategoEnvironment

try:
    print("Initializing StrategoEnvironment...")
    env = StrategoEnvironment(device='cpu')
    print("StrategoEnvironment initialized successfully.")
    print("Resetting environment...")
    env.reset()
    print("Environment reset successfully.")
except Exception as e:
    print(f"FAILED to initialize/reset environment: {e}")
    import traceback
    traceback.print_exc()
