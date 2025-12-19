import torch
import sys
import os

# Add parent directory to path to import Modular Stratego modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drqn_agent import RainbowAgent

def test_action_mapping():
    print("🧪 Testing Action Mapping (3600 Actions)...")
    agent = RainbowAgent(player_id=1, device=torch.device('cpu'), action_size=3600)
    
    test_moves = [
        ((6, 0), (5, 0)),  # Up 1
        ((6, 0), (3, 0)),  # Up 3 (Scout)
        ((0, 0), (0, 9)),  # Right 9 (Scout)
        ((9, 9), (0, 9)),  # Up 9 (Scout)
        ((3, 3), (3, 4)),  # Right 1
        ((3, 3), (4, 3)),  # Down 1
        ((3, 3), (3, 2)),  # Left 1
    ]
    
    passed = 0
    for move in test_moves:
        idx = agent._move_to_action_index(move)
        decoded_move = agent._action_index_to_move(idx)
        
        if move == decoded_move:
            print(f"✅ PASSED: {move} -> ID:{idx} -> {decoded_move}")
            passed += 1
        else:
            print(f"❌ FAILED: {move} -> ID:{idx} -> {decoded_move}")
            
    print(f"\n✨ Passed {passed}/{len(test_moves)} tests.")
    
    # Boundary checks
    print("\n📦 Checking Action Space Boundaries...")
    print(f"Min Index (0): {agent._action_index_to_move(0)}")
    print(f"Max Index (3599): {agent._action_index_to_move(3599)}")
    
    assert passed == len(test_moves), "Action mapping logic failed!"
    print("\n✅ All mapping tests passed!")

if __name__ == "__main__":
    test_action_mapping()
