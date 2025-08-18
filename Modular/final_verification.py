# final_verification.py

import sys
import os

# Add the stratego_modular directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stratego_modular'))

import torch
from stratego_modular.live_environment import LiveStrategoEnvironment
from stratego_modular.piece import PieceType

def final_verification():
    """Final verification that all fixes are working correctly."""
    print("🔍 Final verification of all fixes...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LiveStrategoEnvironment(device=device, show_live_view=False, show_agent_views=False)
    
    # Test 1: Environment initialization
    print("\n1. Testing environment initialization...")
    state = env.reset()
    print("✅ Environment initialization successful")
    
    # Test 2: Flag deployment
    print("\n2. Testing flag deployment...")
    p1_flags = 1 if env.p1_flag_position is not None else 0
    p2_flags = 1 if env.p2_flag_position is not None else 0
    print(f"✅ Flag tracking: Player 1: {p1_flags}, Player 2: {p2_flags}")
    
    if p1_flags == 1 and p2_flags == 1:
        print("✅ Flag deployment test PASSED")
    else:
        print("❌ Flag deployment test FAILED")
        return False
    
    # Test 3: Bomb movement prevention
    print("\n3. Testing bomb movement prevention...")
    valid_moves = env.get_valid_moves()
    
    # Check if any valid moves involve bombs
    bomb_moves = 0
    for move in valid_moves:
        (r_from, c_from), (r_to, c_to) = move
        piece_value = env.board.actual_board[r_from, c_from].item()
        piece_type = PieceType(abs(piece_value))
        if piece_type == PieceType.BOMB:
            bomb_moves += 1
    
    print(f"✅ Found {bomb_moves} bomb moves in valid moves")
    
    if bomb_moves == 0:
        print("✅ Bomb movement prevention test PASSED")
    else:
        print("❌ Bomb movement prevention test FAILED")
        return False
    
    # Test 4: Ownership tracking
    print("\n4. Testing ownership tracking...")
    ownership_count = len(env.piece_ownership)
    print(f"✅ Found {ownership_count} pieces with tracked ownership")
    
    if ownership_count > 0:
        print("✅ Ownership tracking test PASSED")
    else:
        print("❌ Ownership tracking test FAILED")
        return False
    
    print("\n🎉 All tests PASSED! All fixes are working correctly.")
    return True

if __name__ == "__main__":
    success = final_verification()
    if success:
        print("\n✅ FINAL VERIFICATION: ALL FIXES ARE WORKING CORRECTLY")
    else:
        print("\n❌ FINAL VERIFICATION: SOME ISSUES REMAIN")
