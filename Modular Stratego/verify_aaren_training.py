
import torch
import numpy as np
from drqn_agent import RainbowAgent
from game_state import GameState
from piece import PieceType

def verify_aaren_training():
    print("🧪 Verifying AAREN Training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. Create Agent
    print("   Creating Agent...")
    agent = RainbowAgent(player_id=1, device=device, num_envs=1)
    
    # 2. Setup Mock Data
    print("   Setting up mock training data...")
    # We need to simulate:
    # - A piece being revealed (added to revealed_pieces)
    # - Action history for that piece
    
    # Mock PBS instance
    pbs = agent.pbs
    
    # Position (6, 5)
    pos = (6, 5)
    
    # Add to revealed pieces (Scout)
    pbs.revealed_pieces[pos] = PieceType.SCOUT
    
    # Add to action history (some random moves)
    # History stores FEATURES (list of 24 floats), not raw actions
    # Mocking 3 steps of history
    history = []
    for _ in range(3):
        # Create a random feature vector of size 24
        feat = np.random.rand(24).tolist()
        history.append(feat)
        
    pbs.piece_action_history[pos] = history
    
    # 3. Check Weights Before Training
    print("   Checking weights before training...")
    if not pbs.aaren_model:
        print("   ❌ No AAREN model found!")
        return False
        
    weights_before = pbs.aaren_model.input_proj.weight.clone()
    weight_sum_before = weights_before.sum().item()
    print(f"   Weight sum before: {weight_sum_before:.6f}")
    
    # 4. Train PBS
    print("   Calling train_pbs()...")
    agent.train_pbs(epochs=5)
    
    # 5. Check Weights After Training
    print("   Checking weights after training...")
    weights_after = pbs.aaren_model.input_proj.weight
    weight_sum_after = weights_after.sum().item()
    print(f"   Weight sum after:  {weight_sum_after:.6f}")
    
    # Calculate difference
    diff = torch.abs(weights_after - weights_before).sum().item()
    print(f"   Total weight difference: {diff:.6f}")
    
    if diff > 1e-6:
        print("   ✅ SUCCESS: AAREN weights changed! Training is working.")
        return True
    else:
        print("   ❌ FAILURE: AAREN weights did not change.")
        return False

if __name__ == "__main__":
    try:
        if verify_aaren_training():
            print("\n✅ Verification Passed!")
            exit(0)
        else:
            print("\n❌ Verification Failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Verification Crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
