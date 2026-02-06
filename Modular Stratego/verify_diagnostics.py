import torch
import numpy as np
from drqn_agent import RainbowAgent
from diagnostics import DiagnosticTracker
from training_config import HISTORY_EMBEDDING_SIZE

def test_diagnostics():
    print("Initializing Agent...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agent
    # Buffer size 1000 is small.
    agent = RainbowAgent(player_id=1, device=device, num_envs=1, buffer_size=1000)
    
    # Configure diagnostics
    agent.diagnostic_enabled = True
    agent.diagnostics = DiagnosticTracker(log_interval=1) 
    
    print(f"Agent initialized. Network expects {agent.input_channels} channels.")
    
    print("Populating Memory...")
    # Add dummy experiences
    # Use 15 channels (Raw Board) as remember() processes it
    raw_channels = 15 
    
    # 1. Add normal experience
    state = torch.randn(raw_channels, 10, 10)
    next_state = torch.randn(raw_channels, 10, 10)
    action = ((0,0), (0,1)) 
    reward = 1.0 # Low reward
    done = False
    agent.remember(state, action, reward, next_state, done)
    
    # 2. Add HIGH REWARD experience (Flag Capture simulation)
    state = torch.randn(raw_channels, 10, 10)
    next_state = torch.randn(raw_channels, 10, 10)
    action = ((1,0), (1,1))
    reward = 25.0 # High reward!
    done = True
    agent.remember(state, action, reward, next_state, done)
    
    # Fill memory to ensure replay works
    for i in range(50):
         state = torch.randn(raw_channels, 10, 10)
         next_state = torch.randn(raw_channels, 10, 10)
         action = ((0,0), (0,1))
         reward = 1.0
         done = False
         agent.remember(state, action, reward, next_state, done)

    print("Running Replay...")
    try:
        # Replay might skip if internal WARMUP check fails.
        # But we can try. If it returns None, we know why.
        
        # We expect immediate logs from remember() for the high reward
        
        loss = agent.replay(batch_size=4, episode=100) 
        
        if loss is not None:
             print(f"Replay successful. Loss: {loss:.6f}")
        else:
             print("Replay skipped (likely warmup).")
             
        # Check Tracker Stats
        print("\nDiagnostic Summary:")
        print(f"Flag Captures detected: {agent.diagnostics.signal_tracker.flag_captures}")
        print(f"Max Reward seen: {agent.diagnostics.signal_tracker.max_reward_seen}")
        print(f"High TD Errors: {agent.diagnostics.signal_tracker.high_TD_errors}")
        
        if agent.diagnostics.signal_tracker.flag_captures > 0:
            print("\n[SUCCESS] Signal Tracking Verified!")
        else:
            print("\n[FAIL] Signal Tracking failed to detect flag capture.")
            
    except Exception as e:
        print(f"\n[ERROR] Replay failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_diagnostics()
