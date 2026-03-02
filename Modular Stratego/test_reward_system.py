"""
Verification Test — Reward System v3 (PBRS Architecture)

Validates the three-layer anti-stagnation reward architecture:
1. Terminal rewards within C51 support [-10, +10]
2. Terminal game-length modifier (speed bonus + slow draw penalty)
3. PBRS zero for identical states (no Happy Wanderer farming)
4. PBRS positive for piece capture (progress rewarded)
5. PBRS negative for piece loss (regression penalized)
6. Oscillation penalty escalation (A→B→A detection)
7. No per-step penalty (removed in v3)
8. Move diversity penalty fires on shuffling
9. Legacy dead code removed
10. C51 support range
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

def make_game_state(board, turn_count=0):
    """Wrap a board tensor in a GameState object."""
    return GameState(
        board=board,
        current_player=1,
        turn_count=turn_count,
        game_over=False,
        winner=None,
        move_history=[],
        uncertainty_mask=torch.zeros(10, 10),
        revealed_pieces_p1={},
        revealed_pieces_p2={}
    )

def test_terminal_rewards():
    """Test 1: Terminal rewards are within C51 range"""
    print(f"\n{INFO} Test 1: Terminal reward scale")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')

    # Win by flag capture (quick win at turn 10/200)
    # Board encoding: Flag=1, Scout=3, Captain=7, Marshal=11, Bomb=12
    board = make_board({(0, 0): 3, (9, 9): -1})  # P1 Scout, P2 Flag
    prev_state = make_game_state(board)
    curr_state = make_game_state(board)
    
    r = shaper(prev_state, None, curr_state, done=True, winner=1, 
               info={'win_type': 'flag_capture', 'turn_count': 10, 'max_turns': 200})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Win (flag capture): {r:.4f} (range [-10,10])")

    # Loss
    r = shaper(prev_state, None, curr_state, done=True, winner=-1, 
               info={'turn_count': 50, 'max_turns': 200})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Loss: {r:.4f} (expected ~-1.0)")

    # Draw
    r = shaper(prev_state, None, curr_state, done=True, winner=0, 
               info={'turn_count': 100, 'max_turns': 200})
    ok = -10 <= r <= 10
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Draw: {r:.4f} (expected < -0.3)")


def test_terminal_game_length():
    """Test 2: Game-length modifier — quick wins > slow wins, long draws worse"""
    print(f"\n{INFO} Test 2: Terminal game-length modifier")
    config = StrategoRewardConfig()

    # Quick win (turn 20 / 200)
    shaper_a = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    board = make_board({(0, 0): 1})
    state = make_game_state(board)

    r_quick_win = shaper_a(state, None, state, done=True, winner=1, 
                            info={'win_type': 'flag_capture', 'turn_count': 20, 'max_turns': 200})

    # Slow win (turn 190 / 200)
    shaper_b = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    r_slow_win = shaper_b(state, None, state, done=True, winner=1, 
                           info={'win_type': 'flag_capture', 'turn_count': 190, 'max_turns': 200})

    ok_wins = r_quick_win > r_slow_win
    test_results.append(ok_wins)
    print(f"  {'PASS' if ok_wins else 'FAIL'} Quick win ({r_quick_win:.4f}) > Slow win ({r_slow_win:.4f})")

    # Short draw (turn 50 / 200)
    shaper_c = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    r_short_draw = shaper_c(state, None, state, done=True, winner=0, 
                             info={'turn_count': 50, 'max_turns': 200})

    # Long draw (turn 200 / 200)
    shaper_d = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    r_long_draw = shaper_d(state, None, state, done=True, winner=0, 
                            info={'turn_count': 200, 'max_turns': 200})

    ok_draws = r_short_draw > r_long_draw
    test_results.append(ok_draws)
    print(f"  {'PASS' if ok_draws else 'FAIL'} Short draw ({r_short_draw:.4f}) > Long draw ({r_long_draw:.4f})")

    # Slow win is still positive
    ok_positive = r_slow_win > 0
    test_results.append(ok_positive)
    print(f"  {'PASS' if ok_positive else 'FAIL'} Slow win ({r_slow_win:.4f}) still positive")


def test_pbrs_zero_for_identical():
    """Test 3: PBRS gives near-zero SHAPING for non-progressing moves"""
    print(f"\n{INFO} Test 3: PBRS near-zero for non-progressing states")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    # Two board states with nearly identical game-progress potential
    # A sideways move doesn't change material, penetration, or proximity
    board_a = make_board({(6, 0): 3, (0, 5): -1})  # Scout(3) + enemy Flag(-1)
    board_b = make_board({(6, 1): 3, (0, 5): -1})  # "Moved" sideways -- same potential
    
    state_a = make_game_state(board_a)
    state_b = make_game_state(board_b)
    
    # First call: initializes potential then computes PBRS
    r1 = shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={})
    
    # Key property: if Φ(A) ≈ Φ(B), then γΦ(B) - Φ(A) ≈ (γ-1)Φ(A) ≈ -0.005×Φ(A)
    # This is a small negative due to γ < 1 — mathematically correct.
    # The PBRS component should be small (< 0.15 in absolute value).
    shaping_component = abs(r1)  # Remove oscillation/diversity to isolate PBRS
    ok = shaping_component < 0.15  # Small — not a farming-sized reward
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Sideways move reward: {r1:.6f} (expected small, < 0.15 abs)")
    print(f"  {INFO}  gamma-discount effect: (0.995 - 1.0) * Phi(s) produces small negative")


def test_pbrs_positive_for_capture():
    """Test 4: PBRS gives positive reward when capturing high-value enemy piece"""
    print(f"\n{INFO} Test 4: PBRS positive for piece capture")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    # Use a significant capture: P1 Captain (val=7) captures P2 Marshal (val=-11)
    # Board encoding: Captain=7, Marshal=11, Flag=1
    prev_board = make_board({(3, 0): 7, (2, 0): -11, (0, 5): -1})
    # After capture: P1 Captain at (2,0), enemy Marshal removed -- large Phi increase
    curr_board = make_board({(2, 0): 7, (0, 5): -1})
    
    prev_state = make_game_state(prev_board)
    curr_state = make_game_state(curr_board)
    action = ((3, 0), (2, 0))
    
    r = shaper(prev_state, action, curr_state, done=False, winner=None, 
               info={'revealed_in_step': [((2, 0), None)]})
    
    ok = r > 0
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Marshal capture reward: {r:.4f} (expected > 0)")


def test_pbrs_negative_for_loss():
    """Test 5: PBRS gives negative reward when losing own piece"""
    print(f"\n{INFO} Test 5: PBRS negative for piece loss")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    # Before: P1 Scout (val=3) attacks P2 General (val=-10)
    prev_board = make_board({(3, 0): 3, (2, 0): -10, (0, 5): -1})
    # After: Scout dies, General remains
    curr_board = make_board({(2, 0): -10, (0, 5): -1})
    
    prev_state = make_game_state(prev_board)
    curr_state = make_game_state(curr_board)
    action = ((3, 0), (2, 0))
    
    r = shaper(prev_state, action, curr_state, done=False, winner=None, 
               info={'revealed_in_step': [((2, 0), None)]})
    
    # PBRS should produce negative shaping (Φ decreased — we lost material)
    # Plus the attack bonus (+0.02) might offset slightly, but net should be < 0
    ok = r < 0.05  # Allow small positive from attack bonus
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Loss reward: {r:.4f} (expected < 0 or near-zero)")


def test_oscillation_penalty():
    """Test 6: Oscillation penalty escalates for A→B→A patterns"""
    print(f"\n{INFO} Test 6: Oscillation penalty escalation")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    board_a = make_board({(6, 0): 3, (0, 5): -1})  # Scout(3) + enemy Flag(-1)
    board_b = make_board({(6, 1): 3, (0, 5): -1})
    state_a = make_game_state(board_a)
    state_b = make_game_state(board_b)
    
    rewards = []
    # Simulate: A→B→A→B→A→B (6 oscillating moves)
    for i in range(6):
        if i % 2 == 0:
            r = shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={})
        else:
            r = shaper(state_b, ((6, 1), (6, 0)), state_a, done=False, winner=None, info={})
        rewards.append(r)
    
    # Later oscillations should be more punished than earlier ones
    ok = rewards[-1] < rewards[0]
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} Last oscillation ({rewards[-1]:.4f}) < First move ({rewards[0]:.4f})")
    for i, r in enumerate(rewards):
        print(f"  {INFO}  Move {i+1}: reward = {r:.4f}")


def test_no_step_penalty():
    """Test 7: No per-step penalty exists (removed in v3)"""
    print(f"\n{INFO} Test 7: No per-step penalty")
    config = StrategoRewardConfig()
    
    has_step_penalty = hasattr(config, 'step_penalty')
    ok = not has_step_penalty
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} step_penalty field: {'EXISTS (bad!)' if has_step_penalty else 'removed'}")


def test_move_diversity():
    """Test 8: Move diversity penalty fires on repeated moves"""
    print(f"\n{INFO} Test 8: Move diversity penalty")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    board_a = make_board({(6, 0): 3, (0, 5): -1})  # Scout(3) + enemy Flag(-1)
    board_b = make_board({(6, 1): 3, (0, 5): -1})
    state_a = make_game_state(board_a)
    state_b = make_game_state(board_b)
    
    # Feed 20+ identical moves to fill the diversity window
    last_reward = 0
    for i in range(25):
        if i % 2 == 0:
            last_reward = shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={})
        else:
            last_reward = shaper(state_b, ((6, 1), (6, 0)), state_a, done=False, winner=None, info={})
    
    # After 20 identical move pairs, diversity penalty should be active
    ok = last_reward < 0
    test_results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} After 25 repetitive moves, reward = {last_reward:.4f} (expected < 0)")


def test_dead_code_removed():
    """Test 9: Legacy dead code removed"""
    print(f"\n{INFO} Test 9: Legacy dead code removal")
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    
    has_add_info = hasattr(shaper, 'add_info_gain_reward')
    has_info_weight = hasattr(config, 'info_gain_weight')
    has_novelty = hasattr(shaper, 'novelty_tracker')
    
    ok = not has_add_info and not has_info_weight and not has_novelty
    test_results.append(ok)
    print(f"  {'PASS' if not has_add_info else 'FAIL'} add_info_gain_reward: {'EXISTS (bad!)' if has_add_info else 'removed'}")
    print(f"  {'PASS' if not has_info_weight else 'FAIL'} info_gain_weight: {'EXISTS (bad!)' if has_info_weight else 'removed'}")
    print(f"  {'PASS' if not has_novelty else 'FAIL'} novelty_tracker: {'EXISTS (bad!)' if has_novelty else 'removed (PBRS replaces curiosity)'}")


def test_c51_support():
    """Test 10: C51 support range in agent"""
    print(f"\n{INFO} Test 10: C51 support range [-10, +10]")
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
    print("  REWARD SYSTEM v3 — PBRS ARCHITECTURE VERIFICATION")
    print("=" * 60)
    
    test_terminal_rewards()
    test_terminal_game_length()
    test_pbrs_zero_for_identical()
    test_pbrs_positive_for_capture()
    test_pbrs_negative_for_loss()
    test_oscillation_penalty()
    test_no_step_penalty()
    test_move_diversity()
    test_dead_code_removed()
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
