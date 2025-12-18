
import torch
import numpy as np
from drqn_agent import RainbowAgent
import traceback
import sys

def verify_fix():
    print("🧪 Verifying CUDA Crash Fix in drqn_agent.py...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        agent = RainbowAgent(player_id=1, device=device, state_size=27, action_size=10, batch_size=4, buffer_size=100)
        
        # 1. Fill memory with dummy data
        print("📝 Filling memory with dummy data...")
        state = np.zeros((27, 10, 10), dtype=np.float32)
        next_state = np.zeros((27, 10, 10), dtype=np.float32)
        
        for i in range(10):
            agent.remember(state, 0, 1.0, next_state, False, {}, {})
            
        # 2. Test Normal Replay
        print("🔄 Testing Normal Replay...")
        loss = agent.replay(batch_size=4)
        if loss is not None:
             print(f"✅ Normal replay successful. Loss: {loss}")
        else:
             print("⚠️ Normal replay returned None (might be due to buffer size, but normally should run)")

        # 3. Test NaN Reward Handling (Simulated)
        print("Testing NaN Reward Handling...")
        # We need to manually inject a NaN into the batch or mock the memory sampling.
        # Since we can't easily force memory to return NaN without polluting it,
        # we will monkey-patch the memory.sample method for this test.
        
        original_sample = agent.memory.sample
        
        def mock_sample_nan(batch_size):
            res = original_sample(batch_size)
            if res is None: return None
            # Inject NaN into rewards
            # (states, actions, rewards, next_states, dones, indices, weights) or (states, actions, rewards, next_states, dones)
            if len(res) == 7:
                states, actions, rewards, next_states, dones, indices, weights = res
                rewards[0] = float('nan')
                return states, actions, rewards, next_states, dones, indices, weights
            else:
                states, actions, rewards, next_states, dones = res
                rewards[0] = float('nan')
                return states, actions, rewards, next_states, dones
                
        agent.memory.sample = mock_sample_nan
        
        loss_nan = agent.replay(batch_size=4)
        
        if loss_nan is None:
            print("✅ Replay correctly handled NaN reward by returning None (skipped update).")
        else:
            print(f"❌ Replay did NOT skip NaN reward! Loss: {loss_nan}")
            
        # Restore sample
        agent.memory.sample = original_sample

        # 4. Test Extreme Values (Potential Index Out of Bounds)
        print("Testing Extreme/Boundary Values...")
        
        def mock_sample_extreme(batch_size):
            res = original_sample(batch_size)
            if res is None: return None
            if len(res) == 7:
                states, actions, rewards, next_states, dones, indices, weights = res
                rewards.fill_(1000.0) # Very high reward
                return states, actions, rewards, next_states, dones, indices, weights
            else:
                states, actions, rewards, next_states, dones = res
                rewards.fill_(1000.0)
                return states, actions, rewards, next_states, dones
        
        agent.memory.sample = mock_sample_extreme
        
        try:
            loss_extreme = agent.replay(batch_size=4)
            print(f"✅ Extreme value replay successful. Loss: {loss_extreme}")
        except RuntimeError as e:
             if "device-side assert" in str(e):
                 print(f"❌ CAUGHT EXPECTED CRASH: {e}")
                 # This means our fix FAILED to prevent the crash if it still happens.
                 # But if we fixed it, it should NOT crash.
             else:
                 print(f"❌ Error during extreme replay: {e}")
                 
        print("\n🎉 Verification Complete!")

    except Exception as e:
        print(f"❌ Fatal Error during verification: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
