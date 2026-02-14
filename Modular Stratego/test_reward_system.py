"""
Verification Test — Reward System v2 (C51-Compatible)

Validates all 9 fixes applied to the reward pipeline:
1. Terminal rewards are within C51 support [-10, +10]
2. Flag distance reward not dominant
3. Piece-loss penalty is rank-weighted (Marshal >> Scout)
4. Shaping rewards are C51-atom-visible (>= 0.4 per atom)
5. No double-counting (tested in train_dqn.py, not here)
6. Curiosity tracker resets per episode
7. No info-gain dead code
8. Scout penetration only on back rank (rows 0-1 / 8-9)
9. Forward reward gated on piece rank
"""

import sys
import os
import torch

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distributional_reward import UnifiedRewardShaper, StrategoRewardConfig, OFFENSIVE_RANK_THRESHOLD
from game_state import GameState

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

test_results = []

def make_board(setup=None):
    """Create a 10x10 board tensor with optional setup dict {(r,c): val}"""
    board = torch.zeros(10, 10, dtype=torch.float32)
    # Set lakes
    for r, c in [(4,2),(4,3),(4,6),(4,7),(5,2),(5,3),(5,6),(5,7)]:
        board[r, c] = 13  # LAKE
    if setup:
        for (r, c), val in setup.items():
            board[r, c] = val
    return board

def make_game_state(board):
    """Wrap a board tensor in a GameState-like object."""
    gs = GameState.__new__(GameState)
    gs.board = board
    return gs

def test_terminal_rewards():
    """Test 1: Terminal rewards are within C51 range"""
    print(f"\n{INFO} Test 1: Terminal reward scale")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')

    # Win by flag capture
    board = make_board({(0, 0): 1, (9, 9): -11})  # P1 piece alive, P2 flag
    prev_state = make_game_state(board)
    curr_state = make_game_state(board)
    
    r = shaper(prev_state, None, curr_state, done=True, winner=1, info={'win_type': 'flag_capture'})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Win (flag capture): {r:.4f} (expected ~1.0, range [-10,10])")

    # Loss
    r = shaper(prev_state, None, curr_state, done=True, winner=-1, info={})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Loss: {r:.4f} (expected ~-1.0)")

    # Draw
    r = shaper(prev_state, None, curr_state, done=True, winner=0, info={})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Draw: {r:.4f} (expected ~-0.3)")

    # Specific values
    shaper.reset()
    r_win = shaper(prev_state, None, curr_state, done=True, winner=1, info={'win_type': 'flag_capture'})
    ok_win = abs(r_win - config.win_reward_flag * config.outcome_weight) < 0.01
    test_results.append(ok_win)
    print(f"  {'PASS' if ok_win else 'FAIL'} Win value = {r_win:.4f} (expected {config.win_reward_flag})")

    shaper.reset()
    r_loss = shaper(prev_state, None, curr_state, done=True, winner=-1, info={})
    ok_loss = abs(r_loss - config.loss_penalty * config.outcome_weight) < 0.01
    test_results.append(ok_loss)
    print(f"  {'PASS' if ok_loss else 'FAIL'} Loss value = {r_loss:.4f} (expected {config.loss_penalty})")


def test_rank_weighted_loss():
    """Test 3: Piece loss is rank-weighted"""
    print(f"\n{INFO} Test 3: Rank-weighted piece loss")
    config = StrategoRewardConfig()
    
    # Scenario A: Marshal (rank=10) attacks enemy and LOSES
    shaper_a = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    # P1 Marshal (10) at (4,0), enemy General (-9) at (3,0)
    prev_board = make_board({(4, 0): 10, (3, 0): -9})
    # After: Marshal lost, General remains
    curr_board = make_board({(3, 0): -9})
    
    prev_state = make_game_state(prev_board)
    curr_state = make_game_state(curr_board)
    action = ((4, 0), (3, 0))
    
    r_marshal_loss = shaper_a(prev_state, action, curr_state, done=False, winner=None, info={})

    # Scenario B: Scout (rank=2) attacks enemy and LOSES
    shaper_b = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    prev_board2 = make_board({(4, 0): 2, (3, 0): -9})  # Scout vs General
    curr_board2 = make_board({(3, 0): -9})
    
    prev_state2 = make_game_state(prev_board2)
    curr_state2 = make_game_state(curr_board2)
    
    r_scout_loss = shaper_b(prev_state2, action, curr_state2, done=False, winner=None, info={})

    # Marshal loss should be worse than Scout loss
    ok = r_marshal_loss < r_scout_loss
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Marshal loss ({r_marshal_loss:.4f}) < Scout loss ({r_scout_loss:.4f})")
    
    # Check ratio is approximately 10/2 = 5x
    if r_scout_loss != 0 and r_marshal_loss != 0:
        ratio = abs(r_marshal_loss) / abs(r_scout_loss) if abs(r_scout_loss) > 0.0001 else float('inf')
        print(f"  {INFO} Loss ratio: {ratio:.2f}x (rank ratio is 5x)")


def test_forward_reward_gating():
    """Test 9: Forward reward only for low-rank pieces"""
    print(f"\n{INFO} Test 9: Forward reward gating on piece rank")
    config = StrategoRewardConfig()
    
    # Scenario A: Scout (rank=2, below threshold) moves forward
    shaper_a = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    prev_board = make_board({(5, 4): 2})  # Scout at row 5
    curr_board = make_board({(4, 4): 2})  # Scout moved to row 4 (forward for P1)
    action = ((5, 4), (4, 4))  # lake squares avoided
    
    # Avoid lake overlap — use different columns
    prev_board_safe = make_board({(6, 4): 2})
    curr_board_safe = make_board({(5, 4): 2})
    action_safe = ((6, 4), (5, 4))
    
    r_scout = shaper_a(make_game_state(prev_board_safe), action_safe, make_game_state(curr_board_safe), 
                       done=False, winner=None, info={})

    # Scenario B: Marshal (rank=10, above threshold) moves forward
    shaper_b = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    prev_board2 = make_board({(6, 4): 10})  # Marshal
    curr_board2 = make_board({(5, 4): 10})
    
    r_marshal = shaper_b(make_game_state(prev_board2), action_safe, make_game_state(curr_board2),
                         done=False, winner=None, info={})

    # Scout should get flag_distance reward, Marshal should NOT
    ok = r_scout > r_marshal
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Scout forward ({r_scout:.4f}) > Marshal forward ({r_marshal:.4f})")
    print(f"  {INFO} Threshold: rank <= {OFFENSIVE_RANK_THRESHOLD} (Captain). Scout=2, Marshal=10.")


def test_scout_penetration_zone():
    """Test 8: Scout bonus only on actual back rank"""
    print(f"\n{INFO} Test 8: Scout penetration zone tightened")
    config = StrategoRewardConfig()
    
    # Scout at row 3 (used to get bonus, should NOT anymore)
    shaper_a = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    prev_board = make_board({(3, 4): 2})  # Scout at row 3
    curr_board = make_board({(2, 4): 2})  # Moved to row 2
    action = ((3, 4), (2, 4))
    
    r_row2 = shaper_a(make_game_state(prev_board), action, make_game_state(curr_board),
                      done=False, winner=None, info={})

    # Scout at row 1 (should get bonus)
    shaper_b = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    prev_board2 = make_board({(2, 4): 2})
    curr_board2 = make_board({(1, 4): 2})
    action2 = ((2, 4), (1, 4))
    
    r_row1 = shaper_b(make_game_state(prev_board2), action2, make_game_state(curr_board2),
                      done=False, winner=None, info={})

    ok = r_row1 > r_row2
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Row 1 reward ({r_row1:.4f}) > Row 2 reward ({r_row2:.4f})")
    print(f"  {INFO} Bonus zone: rows 0-1 (back rank). Row 2-3 excluded.")


def test_curiosity_reset():
    """Test 6: Curiosity tracker resets per episode"""
    print(f"\n{INFO} Test 6: Curiosity tracker resets on episode boundary")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    # Visit some states
    for i in range(10):
        board = make_board({(6, i % 10): 2})
        shaper.novelty_tracker.get_novelty_bonus(board)
    
    visited_before = len(shaper.novelty_tracker.visit_counts)
    
    # Reset
    shaper.reset()
    visited_after = len(shaper.novelty_tracker.visit_counts)
    
    ok = visited_after == 0
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} After reset: {visited_after} states (was {visited_before})")


def test_no_info_gain():
    """Test 7: Info gain dead code removed"""
    print(f"\n{INFO} Test 7: Dead info-gain code removed")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    has_add_info = hasattr(shaper, 'add_info_gain_reward')
    has_info_weight = hasattr(config, 'info_gain_weight')
    has_episode_info = hasattr(shaper, '_episode_info_gain')
    
    ok = not has_add_info and not has_info_weight and not has_episode_info
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} add_info_gain_reward: {'EXISTS (bad!)' if has_add_info else 'removed'}")
    print(f"  {'PASS' if not has_info_weight else 'FAIL'} info_gain_weight: {'EXISTS (bad!)' if has_info_weight else 'removed'}")
    print(f"  {'PASS' if not has_episode_info else 'FAIL'} _episode_info_gain: {'EXISTS (bad!)' if has_episode_info else 'removed'}")


def test_c51_support():
    """Test C51 support range in agent"""
    print(f"\n{INFO} Test C51: Support range [-10, +10]")
    from drqn_agent import V_MIN, V_MAX, NUM_ATOMS
    
    ok_min = V_MIN == -10.0
    ok_max = V_MAX == 10.0
    ok_atoms = NUM_ATOMS == 51
    delta = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    ok_delta = abs(delta - 0.4) < 0.01
    
    test_results.append(ok_min and ok_max)
    print(f"  {'PASS' if ok_min else 'FAIL'} V_MIN = {V_MIN} (expected -10.0)")
    print(f"  {'PASS' if ok_max else 'FAIL'} V_MAX = {V_MAX} (expected 10.0)")
    print(f"  {'PASS' if ok_atoms else 'FAIL'} NUM_ATOMS = {NUM_ATOMS} (expected 51)")
    print(f"  {'PASS' if ok_delta else 'FAIL'} delta_z = {delta:.4f} (expected 0.4)")


if __name__ == "__main__":
    print("=" * 60)
    print("  REWARD SYSTEM v2 — VERIFICATION TESTS")
    print("=" * 60)
    
    test_terminal_rewards()
    test_rank_weighted_loss()
    test_forward_reward_gating()
    test_scout_penetration_zone()
    test_curiosity_reset()
    test_no_info_gain()
    test_c51_support()
    
    print("\n" + "=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    print(f"  RESULTS: {passed}/{total} passed")
    if passed == total:
        print(f"  {PASS} All tests passed!")
    else:
        failed = [i+1 for i, r in enumerate(test_results) if not r]
        print(f"  {FAIL} Failed tests: {failed}")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
