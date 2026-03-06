import torch
import numpy as np
import os
import sys

# Add repository root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drqn_agent import DQNAgent
from environment import StrategoEnvironment

def test_hooks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing hooks on {device}...")
    
    # Initialize agent
    agent = DQNAgent(player_id=1, device=device)
    # Use a dummy state for testing if model not found
    model_path = r"c:\Users\Mark Lawrence Quibot\repo\Research\History\12\agent1_rainbow_episode_8000.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded.")
    else:
        print("Using random model.")
    
    agent.q_network.eval()
    
    # Hook for attention
    attn_weights = None
    def hook_fn(module, input, output):
        nonlocal attn_weights
        _, attn_weights = output
        print("Hook triggered!")
    
    handle = agent.q_network.spatial_attention.attn.register_forward_hook(hook_fn)
    
    # Forward pass
    env = StrategoEnvironment(device)
    state = env.reset()
    state_tensor = agent.get_state_representation(state.board, pbs_instance=agent.history)
    if state_tensor.dim() == 3:
        state_tensor = state_tensor.unsqueeze(0)
    
    with torch.no_grad():
        logits = agent.q_network(state_tensor)
        probs = logits.exp()
    
    handle.remove()
    
    # Verify Attention Weights
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
        assert attn_weights.shape == (1, 100, 100)
        print("Attention Hook: SUCCESS")
    else:
        print("Attention Hook: FAILED")
        return False

    # Verify C51 Probabilities
    if probs is not None:
        print(f"Probabilities shape: {probs.shape}")
        assert probs.shape == (1, 400, 51)
        # Check sum to 1
        sums = probs.sum(dim=2)
        print(f"Probabilities sum (mean): {sums.mean().item()}")
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        print("C51 Extraction: SUCCESS")
    else:
        print("C51 Extraction: FAILED")
        return False

    # Verify AAREN Embeddings
    embeddings = agent.history.get_embedding_tensor()
    print(f"AAREN Embedding shape: {embeddings.shape}")
    assert embeddings.shape == (64, 10, 10)
    print("AAREN Embedding: SUCCESS")
    
    return True

if __name__ == "__main__":
    success = test_hooks()
    if success:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nTESTS FAILED")
        sys.exit(1)
