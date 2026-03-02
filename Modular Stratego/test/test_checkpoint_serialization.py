import os
import torch
import numpy as np
import sys
import tarfile
import tempfile
import shutil
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drqn_agent import RainbowAgent
from training import Checkpointer, MetricsTracker
from league import LeagueManager

def test_serialization():
    print("Starting serialization test...")
    device = torch.device('cpu')
    save_dir = "test_serialization_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Initialize Agent and add dummy data
    print("1. Initializing Agent 1...")
    agent1 = RainbowAgent(player_id=1, device=device, buffer_size=1000)
    
    # Simulate some training steps
    agent1.step_count = 1234
    
    # Add dummy experience
    state = torch.randn(79, 10, 10)
    next_state = torch.randn(79, 10, 10)
    agent1.memory.add(state, 42, 1.0, next_state, False)
    
    # Modify optimizer state
    agent1.optimizer.param_groups[0]['lr'] = 0.000123
    
    # 2. Save Checkpoint
    print("2. Saving Checkpoint...")
    checkpointer = Checkpointer(save_dir=save_dir)
    league_manager = LeagueManager(league_dir=os.path.join(save_dir, "league"))
    metrics_tracker = MetricsTracker(save_dir=save_dir)
    
    checkpointer.save_checkpoint(
        episode=100,
        agent1=agent1,
        agent2=None,
        league_manager=league_manager,
        curriculum=None,
        metrics_tracker=metrics_tracker,
        league_interval=100
    )
    
    # 3. Verify Files Exist
    archive_path = os.path.join(save_dir, "agent1_rainbow_episode_100.tar.gz")
    league_path = os.path.join(save_dir, "agent1_league_episode_100.pt")
    
    assert os.path.exists(archive_path), "Archive file missing"
    assert os.path.exists(league_path), "League file missing"
    print("[OK] Files created successfully.")
    
    # 4. Load into New Agent
    print("4. Loading into New Agent...")
    agent_new = RainbowAgent(player_id=1, device=device, buffer_size=1000)
    checkpointer.load_agent_models(agent_new, RainbowAgent(player_id=-1, device=device, use_pbs=False))
    
    # 5. Verify State
    print("5. Verifying State...")
    assert agent_new.step_count == 1234, f"Step count mismatch: {agent_new.step_count}"
    assert agent_new.optimizer.param_groups[0]['lr'] == 0.000123, "Optimizer LR mismatch"
    assert len(agent_new.memory) == 1, f"Memory size mismatch: {len(agent_new.memory)}"
    print("[OK] Agent state restored correctly.")
    
    # 6. Verify League Model (inference only)
    print("6. Verifying League Model...")
    league_state = torch.load(league_path, map_location='cpu', weights_only=True)
    assert 'q_network_state_dict' in league_state, "League model missing Q-network"
    assert 'optimizer_state_dict' not in league_state, "League model should NOT have optimizer"
    assert 'memory_state_dict' not in league_state, "League model should NOT have memory"
    print("[OK] League model (inference-only) verified.")
    
    # Cleanup
    # shutil.rmtree(save_dir)
    print("\nALL SERIALIZATION TESTS PASSED! [OK]")

if __name__ == "__main__":
    try:
        test_serialization()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
