"""
Reward shaping: PBRS from each side's own view (network/distributional_reward.py).

1. PBRS near zero for a non-progressing move, positive for a capture, negative for a loss
2. Φ reads only the side's own view; hidden enemy pieces count with the mean army rank
3. Terminal transitions get −Φ(s), so a game's shaping telescopes to −Φ(s_0)
4. Non-potential terms are off by default; when switched on they fire on the right events
5. C51 support range

(The thesis terminal branch — speed bonus, draw penalty, MARL scaling — is gone: the trainer's
terminal reward is outcome_reward: +1 / -1, and DRAW_REWARD for a draw.
Its tests went with it; the draw reward's are in test_turn_handling.)
"""

import random

import pytest
import torch

from distributional_reward import (UnifiedRewardShaper, StrategoRewardConfig, compute_potential,
                                   ARMY_MATERIAL, UNKNOWN_PIECE_RANK)
from game_state import GameState
from piece import PieceType
from training_config import GAMMA

HIDDEN = -20


def make_board(setup=None):
    """Create a 10x10 board tensor with optional setup dict {(r,c): val}"""
    board = torch.zeros(10, 10, dtype=torch.float32)
    # Set lakes
    for r, c in [(4,2),(4,3),(4,6),(4,7),(5,2),(5,3),(5,6),(5,7)]:
        board[r, c] = -13  # LAKE (environment encoding)
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


SCOUT, CAPTAIN, GENERAL, MARSHAL = (PieceType.SCOUT.value, PieceType.CAPTAIN.value,
                                    PieceType.GENERAL.value, PieceType.MARSHAL.value)


def test_pbrs_zero_for_identical():
    """PBRS gives near-zero shaping for a non-progressing move"""
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    state_a = make_game_state(make_board({(6, 0): SCOUT, (0, 5): -1}))
    state_b = make_game_state(make_board({(6, 1): SCOUT, (0, 5): -1}))  # sideways: same potential
    r = shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={})
    assert r == pytest.approx((GAMMA - 1) * compute_potential(state_a.board, 1))
    assert abs(r) < 0.01


def test_pbrs_positive_for_capture():
    """P1 Captain captures a P2 Marshal"""
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    prev_state = make_game_state(make_board({(3, 0): CAPTAIN, (2, 0): -MARSHAL, (0, 5): -1}))
    curr_state = make_game_state(make_board({(2, 0): CAPTAIN, (0, 5): -1}))
    assert shaper(prev_state, ((3, 0), (2, 0)), curr_state, done=False, winner=None, info={}) > 0


def test_pbrs_negative_for_loss():
    """P1 Scout attacks a P2 General and dies"""
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    prev_state = make_game_state(make_board({(3, 0): SCOUT, (2, 0): -GENERAL, (0, 5): -1}))
    curr_state = make_game_state(make_board({(2, 0): -GENERAL, (0, 5): -1}))
    assert shaper(prev_state, ((3, 0), (2, 0)), curr_state, done=False, winner=None, info={}) < 0


# ---------------------------------------------------------------- potential from the own view

def test_hidden_enemy_pieces_count_with_the_mean_army_rank():
    """The thesis potential skipped HIDDEN squares, so a fogged enemy army had no material."""
    with_piece = make_board({(6, 0): CAPTAIN, (2, 0): HIDDEN, (0, 5): HIDDEN})
    captured = make_board({(6, 0): CAPTAIN, (0, 5): HIDDEN})
    gain = compute_potential(captured, 1) - compute_potential(with_piece, 1)
    assert UNKNOWN_PIECE_RANK == pytest.approx(3.7) and ARMY_MATERIAL == 148
    assert gain == pytest.approx(0.4 * UNKNOWN_PIECE_RANK / ARMY_MATERIAL)


def test_hidden_markers_are_enemy_pieces_for_player_two_as_well():
    """Player 2's view also shows Player 1's unseen pieces as HIDDEN (a negative value)."""
    view = make_board({(3, 0): -CAPTAIN, (7, 0): HIDDEN, (9, 9): HIDDEN})
    without = make_board({(3, 0): -CAPTAIN, (9, 9): HIDDEN})
    assert compute_potential(without, -1) - compute_potential(view, -1) == pytest.approx(
        0.4 * UNKNOWN_PIECE_RANK / ARMY_MATERIAL)


def test_potential_does_not_see_unrevealed_enemy_ranks(cpu):
    """Φ reads the side's view only: changing the true ranks behind P2's hidden pieces (the
    GameState also carries the true board) changes nothing."""
    from environment import StrategoEnvironment
    env = StrategoEnvironment(cpu, full_observability=False)
    state = env.reset()
    assert (state.board == HIDDEN).sum() == 40
    other = make_game_state(state.board.clone())
    other.actual_board = state.actual_board.clone()
    other.actual_board[other.actual_board < 0] = -MARSHAL
    move = env.get_valid_moves()[0]
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    next_state, _, _, _ = env.step(move)
    assert shaper(state, move, next_state, False) == shaper(other, move, next_state, False)


# ---------------------------------------------------------------- terminal transitions

def test_terminal_transition_gets_minus_phi_of_the_last_state():
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    prev_state = make_game_state(make_board({(1, 5): CAPTAIN, (0, 5): -1}))
    curr_state = make_game_state(make_board({(0, 5): CAPTAIN}))
    r = shaper(prev_state, ((1, 5), (0, 5)), curr_state, done=True, winner=1, info={})
    assert r == pytest.approx(-compute_potential(prev_state.board, 1))


@pytest.mark.parametrize("full_observability", [True, False])
def test_a_games_shaping_telescopes_to_minus_phi_of_the_first_state(cpu, full_observability):
    """With Φ(terminal) = 0, sum_t gamma^t F_t = -Φ(s_0) for every game, so the shaping cannot
    change which policy is optimal. (The thesis trainer gave the last transition the outcome
    only, which added gamma^T Φ(s_T) to every game.)"""
    from environment import StrategoEnvironment
    from network import TurnTracker
    random.seed(3)
    env = StrategoEnvironment(cpu, full_observability=full_observability, max_turns=80)
    state = env.reset()
    shapers = {p: UnifiedRewardShaper(player_id=p, device='cpu') for p in (1, -1)}
    tracker = TurnTracker(1)
    totals, steps, first = {1: 0.0, -1: 0.0}, {1: 0, -1: 0}, {}
    done = False
    while not done:
        player = state.current_player
        move = random.choice(env.get_valid_moves())
        next_state, _, done, info = env.step(move)
        for t in tracker.record(0, player, state, move, next_state, done, info):
            p = t['player']
            first.setdefault(p, compute_potential(t['state'].board, p))
            f = shapers[p](t['state'], t['action'], t['next_state'], t['done'], t['winner'], t['info'])
            totals[p] += GAMMA ** steps[p] * f
            steps[p] += 1
        state = next_state
    for p in (1, -1):
        assert totals[p] == pytest.approx(-first[p], abs=1e-9)


# ---------------------------------------------------------------- non-potential terms

def test_non_potential_terms_are_off_by_default():
    config = StrategoRewardConfig()
    assert config.attack_bonus == config.spy_marsh_bonus == config.miner_bomb_bonus == 0
    assert config.oscillation_penalty == config.low_diversity_penalty == config.stalemate_penalty == 0
    shaper = UnifiedRewardShaper(player_id=1, device='cpu')
    prev_state = make_game_state(make_board({(3, 0): SCOUT, (2, 0): HIDDEN, (0, 5): HIDDEN}))
    curr_state = make_game_state(make_board({(2, 0): SCOUT, (0, 5): HIDDEN}))
    r = shaper(prev_state, ((3, 0), (2, 0)), curr_state, done=False, info={'num_valid_moves': 1})
    assert r == pytest.approx(GAMMA * compute_potential(curr_state.board, 1) - compute_potential(prev_state.board, 1))


def test_spy_marshal_and_miner_bomb_bonuses_fire_on_their_pieces():
    """The thesis code tested values 1/10 (Flag vs General) and 3/11 (Scout vs Marshal)."""
    config = StrategoRewardConfig(spy_marsh_bonus=0.3, miner_bomb_bonus=0.15)
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    shaper.config.pbrs_weight = 0.0

    def bonus(attacker, defender):
        prev_state = make_game_state(make_board({(6, 0): attacker, (5, 0): -defender}))
        return shaper(prev_state, ((6, 0), (5, 0)), make_game_state(make_board({(5, 0): attacker})), False)

    assert bonus(PieceType.SPY.value, MARSHAL) == pytest.approx(0.3)
    assert bonus(PieceType.MINER.value, PieceType.BOMB.value) == pytest.approx(0.15)
    assert bonus(SCOUT, MARSHAL) == 0 and bonus(PieceType.FLAG.value, GENERAL) == 0


def test_oscillation_penalty():
    """When switched on, the oscillation penalty escalates for A→B→A patterns"""
    config = StrategoRewardConfig(oscillation_penalty=-0.02)
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    state_a = make_game_state(make_board({(6, 0): SCOUT, (0, 5): -1}))
    state_b = make_game_state(make_board({(6, 1): SCOUT, (0, 5): -1}))
    rewards = []
    for i in range(6):
        if i % 2 == 0:
            rewards.append(shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={}))
        else:
            rewards.append(shaper(state_b, ((6, 1), (6, 0)), state_a, done=False, winner=None, info={}))
    assert rewards[-1] < rewards[0]


def test_move_diversity():
    """When switched on, the move diversity penalty fires on repeated moves"""
    config = StrategoRewardConfig(low_diversity_penalty=-0.05)
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    state_a = make_game_state(make_board({(6, 0): SCOUT, (0, 5): -1}))
    state_b = make_game_state(make_board({(6, 1): SCOUT, (0, 5): -1}))
    for i in range(25):
        if i % 2 == 0:
            last_reward = shaper(state_a, ((6, 0), (6, 1)), state_b, done=False, winner=None, info={})
        else:
            last_reward = shaper(state_b, ((6, 1), (6, 0)), state_a, done=False, winner=None, info={})
    assert last_reward < -0.04


def test_no_step_penalty():
    """No per-step penalty exists (removed in v3)"""
    assert not hasattr(StrategoRewardConfig(), 'step_penalty')


def test_dead_code_removed():
    """Legacy dead code removed"""
    config = StrategoRewardConfig()
    shaper = UnifiedRewardShaper(player_id=1, config=config, device='cpu')
    assert not hasattr(shaper, 'add_info_gain_reward')
    assert not hasattr(config, 'info_gain_weight')
    assert not hasattr(shaper, 'novelty_tracker')


def test_c51_support():
    """C51 support range in agent"""
    from drqn_agent import V_MIN, V_MAX, NUM_ATOMS
    assert V_MIN == -10.0 and V_MAX == 10.0 and NUM_ATOMS == 51
    assert abs((V_MAX - V_MIN) / (NUM_ATOMS - 1) - 0.4) < 0.01
