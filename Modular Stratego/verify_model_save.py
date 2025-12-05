
import torch
import os
import shutil
from drqn_agent import RainbowAgent
from pbs import ProbabilisticBeliefState

def verify_model_save():
    print("🧪 Verifying Model Save/Load with PBS...")
    
    # Setup paths
    test_dir = "models/test_save_verification"
    os.makedirs(test_dir, exist_ok=True)
    model_path = os.path.join(test_dir, "test_agent.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. Create Agent 1 and modify its AAREN model
    print("   Creating Agent 1...")
    agent1 = RainbowAgent(player_id=1, device=device, num_envs=1)
    
    # Verify PBS exists
    if not agent1.pbs or not agent1.pbs.aaren_model:
        print("   ❌ Agent 1 has no PBS or AAREN model!")
        return False
        
    # Modify AAREN weights manually to ensure they are distinct
    print("   Modifying Agent 1 AAREN weights...")
    with torch.no_grad():
        # Access the first layer's weight and add 1.0
        agent1.pbs.aaren_model.input_proj.weight.add_(1.0)
        
    # Store a reference value to check against later
    original_weight_sum = agent1.pbs.aaren_model.input_proj.weight.sum().item()
    print(f"   Agent 1 AAREN weight sum: {original_weight_sum:.4f}")
    
    # 2. Save Agent 1
    print(f"   Saving Agent 1 to {model_path}...")
    agent1.save_model(model_path)
    
    # 3. Create Agent 2 (fresh)
    print("   Creating Agent 2 (fresh)...")
    agent2 = RainbowAgent(player_id=1, device=device, num_envs=1)
    
    # Check initial weights (should be random and different)
    initial_weight_sum = agent2.pbs.aaren_model.input_proj.weight.sum().item()
    print(f"   Agent 2 Initial AAREN weight sum: {initial_weight_sum:.4f}")
    
    if abs(initial_weight_sum - original_weight_sum) < 1e-5:
        print("   ⚠️  Warning: Initial weights match randomly? Unlikely but possible.")
        
    # 4. Load Agent 1's model into Agent 2
    print("   Loading model into Agent 2...")
    agent2.load_model(model_path)
    
    # 5. Verify weights match
    loaded_weight_sum = agent2.pbs.aaren_model.input_proj.weight.sum().item()
    print(f"   Agent 2 Loaded AAREN weight sum: {loaded_weight_sum:.4f}")
    
    if abs(loaded_weight_sum - original_weight_sum) < 1e-5:
        print("   ✅ SUCCESS: AAREN weights match after load!")
        
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print("   Cleaned up test directory.")
        except Exception as e:
            print(f"   Warning: Could not clean up {test_dir}: {e}")
            
        return True
    else:
        print("   ❌ FAILURE: AAREN weights do not match!")
        print(f"   Expected: {original_weight_sum:.4f}")
        print(f"   Got:      {loaded_weight_sum:.4f}")
        return False

if __name__ == "__main__":
    try:
        success = verify_model_save()
        if success:
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
