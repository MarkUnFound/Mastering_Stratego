"""The agent's action index <-> move mapping must be lossless for every legal move.

(Replaces verify_action_mapping.py, which asserted the lossy 400-action mapping: long Scout
moves collapsed to a 1-square move in the same direction.)
"""

import random

import pytest

from drqn_agent import RainbowAgent
from environment import StrategoEnvironment


def _legal_moves_seen_in_random_games(device, games=3, plies=120, seed=0):
    """Legal moves for both players over a few random games (long Scout moves included)."""
    rng = random.Random(seed)
    seen = {1: set(), -1: set()}
    env = StrategoEnvironment(device)
    for _ in range(games):
        env.reset()
        for _ in range(plies):
            moves = env.get_valid_moves()
            if not moves:
                break
            seen[env.current_player].update(moves)
            _, _, done, _ = env.step(rng.choice(moves))
            if done:
                break
    return seen


def _is_long(move):
    (r1, c1), (r2, c2) = move
    return abs(r2 - r1) + abs(c2 - c1) > 1


def test_random_games_produce_long_scout_moves_for_both_players(cpu):
    seen = _legal_moves_seen_in_random_games(cpu)
    for player in (1, -1):
        assert any(_is_long(m) for m in seen[player]), f"no long moves seen for player {player}"


@pytest.mark.xfail(strict=True, reason="Step 4: 400-action encoding drops Scout distance")
def test_every_legal_move_round_trips(cpu, tiny_network):
    seen = _legal_moves_seen_in_random_games(cpu)
    for player in (1, -1):
        agent = RainbowAgent(player_id=player, device=cpu, num_envs=1)
        bad = [m for m in seen[player] if agent._action_index_to_move(agent._move_to_action_index(m)) != m]
        assert not bad, f"player {player}: {len(bad)} of {len(seen[player])} moves lost, e.g. {bad[:3]}"


@pytest.mark.xfail(strict=True, reason="Step 4: 400-action encoding drops Scout distance")
def test_distinct_legal_moves_get_distinct_indices(cpu, tiny_network):
    seen = _legal_moves_seen_in_random_games(cpu)
    for player in (1, -1):
        agent = RainbowAgent(player_id=player, device=cpu, num_envs=1)
        indices = {agent._move_to_action_index(m) for m in seen[player]}
        assert len(indices) == len(seen[player])
