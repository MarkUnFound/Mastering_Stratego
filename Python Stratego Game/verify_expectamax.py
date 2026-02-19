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
    
    scenarios = [
        ("Marshal (1) vs Flag (10)", 1, 10, 10.0),
        ("Marshal (1) vs General (2)", 1, 2, 0.5 + (2/12.0)*0.5),
        ("Scout (9) vs Marshal (1)", 9, 1, -0.5 - ((12-9)/12.0)*0.5),
        ("Miner (8) vs Bomb (11)", 8, 11, 1.0),
        ("Sergeant (7) vs Bomb (11)", 7, 11, -1.0),
        ("Spy (12) vs Marshal (1)", 12, 1, 1.0),
        ("General (2) vs Spy (12)", 2, 12, 0.8),
        ("Major (4) vs Major (4)", 4, 4, -0.1),
    ]
    
    print("--- Expectamax Utility Verification ---")
    all_passed = True
    for desc, attacker, defender, expected in scenarios:
        actual = search._calculate_outcome_utility(attacker, defender)
        if abs(actual - expected) < 1e-6:
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (Expected {expected}, got {actual})"
            all_passed = False
        print(f"{desc:<30} | Utility: {actual:>6.3f} | {status}")
    
    return all_passed

if __name__ == "__main__":
    if test_utility():
        print("\nAll utility scenarios passed! Expectamax logic is sound.")
        sys.exit(0)
    else:
        print("\nSome scenarios failed.")
        sys.exit(1)
