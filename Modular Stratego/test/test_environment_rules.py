"""Game rules and simulator speed: move generation, legal-move cache, no-battle draw."""

import random
import time

import pytest

from environment import StrategoEnvironment
from piece import PieceType as P


def position(cpu, p1, p2, player=1, **kwargs):
    """An environment holding only the given pieces ({(r, c): PieceType}) plus lakes; flags go in the
    corners unless placed explicitly."""
    env = StrategoEnvironment(cpu, **kwargs)
    env.board.reset()
    env._flag_positions = {1: None, -1: None}
    p1 = {(9, 0): P.FLAG, **p1} if P.FLAG not in p1.values() else p1
    p2 = {(0, 9): P.FLAG, **p2} if P.FLAG not in p2.values() else p2
    env.place_pieces(1, [(t, pos) for pos, t in p1.items()])
    env.place_pieces(-1, [(t, pos) for pos, t in p2.items()])
    env.current_player = player
    return env


def moves_from(env, square):
    return sorted(dst for src, dst in env.get_valid_moves() if src == square)


def test_ordinary_piece_moves_one_square_and_attacks(cpu):
    env = position(cpu, {(6, 4): P.MAJOR}, {(5, 4): P.MINER})
    assert moves_from(env, (6, 4)) == [(5, 4), (6, 3), (6, 5), (7, 4)]


def test_scout_slides_until_blocked_and_stops_on_attack(cpu):
    # Scout at (6,2): up is the lake at (5,2); right runs to the own Sergeant at (6,6);
    # left runs to the edge; down attacks the enemy Miner at (8,2) and stops there
    env = position(cpu, {(6, 2): P.SCOUT, (6, 6): P.SERGEANT}, {(8, 2): P.MINER})
    assert moves_from(env, (6, 2)) == [(6, 0), (6, 1), (6, 3), (6, 4), (6, 5), (7, 2), (8, 2)]


def test_scout_cannot_jump_over_pieces(cpu):
    env = position(cpu, {(9, 5): P.SCOUT}, {(7, 5): P.MINER, (3, 5): P.SPY})
    # up: (8,5) then the attack on (7,5), never reaching the Spy behind it; left stops at the own Flag
    assert moves_from(env, (9, 5)) == [(7, 5), (8, 5), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)]


def test_bombs_and_flags_never_move(cpu):
    env = position(cpu, {(6, 0): P.BOMB, (6, 1): P.SERGEANT}, {})
    sources = {src for src, _ in env.get_valid_moves()}
    assert sources == {(6, 1)}


def test_second_player_moves_mirror_the_first(cpu):
    env = position(cpu, {(6, 6): P.SERGEANT}, {(3, 2): P.SCOUT, (1, 2): P.BOMB}, player=-1)
    assert moves_from(env, (3, 2)) == [(2, 2), (3, 0), (3, 1), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]
    assert all(src != (6, 6) for src, _ in env.get_valid_moves())


def test_move_cache_follows_board_edits_and_is_not_aliased(cpu):
    env = position(cpu, {(6, 4): P.MAJOR}, {})
    first = env.get_valid_moves()
    first.clear()  # callers may mutate what they get
    assert moves_from(env, (6, 4)) == [(5, 4), (6, 3), (6, 5), (7, 4)]
    env.board.place_piece(6, 5, P.SERGEANT.value)  # edited from outside the environment
    assert moves_from(env, (6, 4)) == [(5, 4), (6, 3), (7, 4)]
    env.current_player = -1
    assert all(src != (6, 4) for src, _ in env.get_valid_moves())


def _quiet_move(env):
    """A legal move onto an empty square (no battle)."""
    board = env.board.actual_board
    return next(m for m in sorted(env.get_valid_moves()) if board[m[1]].item() == 0)


def test_no_battle_draw_after_k_plies(cpu):
    env = position(cpu, {(7, 0): P.SCOUT}, {(2, 9): P.SCOUT}, no_battle_draw_plies=10)
    for ply in range(1, 11):
        state, _, done, info = env.step(_quiet_move(env))
        assert info['plies_since_battle'] == ply == state.plies_since_battle
        assert done == (ply == 10)
    assert env.winner == 0 and env.win_type == 'no_battle_draw'


def test_battle_resets_the_no_battle_counter(cpu):
    env = position(cpu, {(6, 4): P.MARSHAL, (7, 0): P.SCOUT}, {(5, 4): P.SERGEANT, (2, 9): P.SCOUT},
                   no_battle_draw_plies=10)
    env.step(((7, 0), (7, 1)))
    env.step(((2, 9), (2, 8)))
    assert env.plies_since_battle == 2
    _, _, done, info = env.step(((6, 4), (5, 4)))  # Marshal takes Sergeant
    assert info['plies_since_battle'] == 0 and not done


def test_no_battle_rule_can_be_turned_off(cpu):
    env = position(cpu, {(7, 0): P.SCOUT}, {(2, 9): P.SCOUT}, no_battle_draw_plies=0, max_turns=10**6)
    for _ in range(300):
        _, _, done, _ = env.step(_quiet_move(env))
        assert not done


def test_training_and_eval_limits(cpu):
    from training_config import NO_BATTLE_DRAW_PLIES, EVAL_NO_BATTLE_DRAW_PLIES
    assert (NO_BATTLE_DRAW_PLIES, EVAL_NO_BATTLE_DRAW_PLIES) == (100, 200)
    assert StrategoEnvironment(cpu).no_battle_draw_plies == NO_BATTLE_DRAW_PLIES
    assert StrategoEnvironment(cpu, no_battle_draw_plies=EVAL_NO_BATTLE_DRAW_PLIES).no_battle_draw_plies == 200


def test_random_play_is_fast(cpu):
    """The thesis simulator ran about 220 plies/s; the list-based move generator runs about 6000."""
    env = StrategoEnvironment(cpu, max_turns=10**9)
    rng = random.Random(0)
    plies, start = 0, time.perf_counter()
    while plies < 3000:
        moves = env.get_valid_moves()
        _, _, done, _ = env.step(rng.choice(moves) if moves else None)
        plies += 1
        if done:
            env.reset()
    assert plies / (time.perf_counter() - start) > 1000
