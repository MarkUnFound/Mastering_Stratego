
import torch
import sys
import os

# Add parent directory to path to import Modular Stratego modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drqn_agent import DQNAgent

def test_action_mapping():
    print(" Testing Action Mapping (400 Actions)...")
    agent = DQNAgent(player_id=1, device=torch.device('cpu'), action_size=400, num_envs=1)
    
    # Scout moves: Distance > 1 should be mapped to Direction Index
    # BUT decoding back will only give Distance 1.
    # This asymmetry is expected in the new logic.
    test_moves = [
        ((6, 0), (5, 0)),  # Up 1
        ((6, 0), (3, 0)),  # Up 3 (Scout) -> Should map to Up 1 Index
        ((0, 0), (0, 9)),  # Right 9 (Scout) -> Should map to Right 1 Index
        ((3, 3), (3, 4)),  # Right 1
        ((3, 3), (4, 3)),  # Down 1
        ((3, 3), (3, 2)),  # Left 1
    ]
    
    passed = 0
    for move in test_moves:
        idx = agent._move_to_action_index(move)
        decoded_move = agent._action_index_to_move(idx)
        
        # Check Direction Match
        (r1, c1), (r2, c2) = move
        (dr1, dc1), (dr2, dc2) = decoded_move
        
        # Original move direction
        orig_dir = (r2 - r1, c2 - c1)
        orig_sign = (np.sign(orig_dir[0]), np.sign(orig_dir[1]))
        
        # Decoded move direction
        dec_dir = (dr2 - dr1, dc2 - dc1)
        dec_sign = (np.sign(dec_dir[0]), np.sign(dec_dir[1]))
        
        # Decoded dist must be 1
        dec_dist = abs(dr2 - dr1) + abs(dc2 - dc1)
        
        is_same_dir = (orig_sign == dec_sign)
        is_dist_1 = (dec_dist == 1)
        is_same_start = ((r1, c1) == (dr1, dc1))
        
        if is_same_dir and is_dist_1 and is_same_start:
             print(f" PASSED: {move} -> ID:{idx} -> {decoded_move} (Direction Preserved)")
             passed += 1
        else:
             print(f" FAILED: {move} -> ID:{idx} -> {decoded_move}")

    print(f"\n Passed {passed}/{len(test_moves)} tests.")
    
    # Boundary checks
    print("\n Checking Action Space Boundaries...")
    print(f"Min Index (0): {agent._action_index_to_move(0)}")
    print(f"Max Index (399): {agent._action_index_to_move(399)}")
    print(f"Out of Bounds (400): {agent._action_index_to_move(400)}")
    
    assert passed == len(test_moves), "Action mapping logic failed!"
    print("\n All mapping tests passed!")

import numpy as np
if __name__ == "__main__":
    test_action_mapping()
