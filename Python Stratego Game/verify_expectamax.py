import os
import sys

# Add the bot directory to path
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

from dqn_bot_logic import ExpectamaxSearch

def test_utility():
    # Mock agent
    class MockAgent:
        def __init__(self):
            self.device = 'cpu'
            self.history = None
    
    agent = MockAgent()
    search = ExpectamaxSearch(agent)
    
    from piece import PieceType

    scenarios = [
        # GUI Ranks (stratego.py: Marshal=10, General=9, Miner=3, Scout=2, Spy=1, Bomb=0, Flag=-1)
        ("GUI: Marshal (10) vs Flag (-1)", 10, -1, 10.0),
        ("GUI: Marshal (10) vs General (9)", 10, 9, 0.5 + (10/12.0)*0.5),
        ("GUI: Scout (2) vs Marshal (10)", 2, 10, -0.5 - ((12-3)/12.0)*0.5),
        ("GUI: Miner (3) vs Bomb (0)", 3, 0, 1.0),
        ("GUI: Sergeant (4) vs Bomb (0)", 4, 0, -1.0),
        ("GUI: Spy (1) vs Marshal (10)", 1, 10, 1.0),
        ("GUI: General (9) vs Spy (1)", 9, 1, 0.8),
        ("GUI: Major (7) vs Major (7)", 7, 7, -0.1),
        
        # PieceType Enums (piece.py)
        ("Enum: Marshal vs Flag", PieceType.MARSHAL, PieceType.FLAG, 10.0),
        ("Enum: Miner vs Bomb", PieceType.MINER, PieceType.BOMB, 1.0),
        ("Enum: Sergeant vs Bomb", PieceType.SERGEANT, PieceType.BOMB, -1.0),
        ("Enum: Spy vs Marshal", PieceType.SPY, PieceType.MARSHAL, 1.0),
        ("Enum: Marshal vs General", PieceType.MARSHAL, PieceType.GENERAL, 0.5 + (10/12.0)*0.5),
        ("Enum: Major vs Major", PieceType.MAJOR, PieceType.MAJOR, -0.1),
    ]
    
    print("--- Expectamax Utility Verification ---")
    all_passed = True
    for desc, attacker, defender, expected in scenarios:
        actual = search._calculate_outcome_utility(attacker, defender)
        if abs(actual - expected) < 1e-6:
            status = "[PASS]"
        else:
            status = f"[FAIL] (Expected {expected:.3f}, got {actual:.3f})"
            all_passed = False
        print(f"{desc:<35} | Utility: {actual:>6.3f} | {status}")
    
    return all_passed

if __name__ == "__main__":
    if test_utility():
        print("\nAll utility scenarios passed! Expectamax logic is sound.")
        sys.exit(0)
    else:
        print("\nSome scenarios failed.")
        sys.exit(1)
