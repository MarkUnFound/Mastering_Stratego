"""The agent's action index <-> move mapping must be lossless for every legal move.

(Replaces verify_action_mapping.py, which asserted the lossy 400-action mapping: long Scout
moves collapsed to a 1-square move in the same direction.)
"""

import random

import pytest
import torch

from drqn_agent import RainbowAgent
from environment import StrategoEnvironment

NUM_ACTIONS = 3600


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


def test_every_legal_move_round_trips(cpu, tiny_network):
    seen = _legal_moves_seen_in_random_games(cpu)
    for player in (1, -1):
        agent = RainbowAgent(player_id=player, device=cpu, num_envs=1)
        bad = [m for m in seen[player] if agent._action_index_to_move(agent._move_to_action_index(m)) != m]
        assert not bad, f"player {player}: {len(bad)} of {len(seen[player])} moves lost, e.g. {bad[:3]}"


def test_distinct_legal_moves_get_distinct_indices(cpu, tiny_network):
    seen = _legal_moves_seen_in_random_games(cpu)
    for player in (1, -1):
        agent = RainbowAgent(player_id=player, device=cpu, num_envs=1)
        indices = {agent._move_to_action_index(m) for m in seen[player]}
        assert len(indices) == len(seen[player])


def _rotate(move):
    (r1, c1), (r2, c2) = move
    return (9 - r1, 9 - c1), (9 - r2, 9 - c2)


def test_indices_are_in_range_and_both_sides_share_the_canonical_frame(cpu, tiny_network):
    """Player 2's moves are indexed on the board rotated 180 degrees: a move and its rotation by
    the other side get the same index, so one network serves both sides."""
    seen = _legal_moves_seen_in_random_games(cpu, games=1)
    p1 = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    p2 = RainbowAgent(player_id=-1, device=cpu, num_envs=1)
    assert p1.action_size == p2.action_size == NUM_ACTIONS
    for move in seen[1]:
        idx = p1._move_to_action_index(move)
        assert 0 <= idx < NUM_ACTIONS
        assert p2._move_to_action_index(_rotate(move)) == idx
        assert p2._action_index_to_move(idx) == _rotate(move)


def test_gui_player_two_uses_the_same_frame_as_trainer_player_two(cpu, tiny_network):
    """The GUI numbers the sides 1 and 2, the trainer 1 and -1."""
    gui = RainbowAgent(player_id=2, device=cpu, num_envs=1)
    trainer = RainbowAgent(player_id=-1, device=cpu, num_envs=1)
    move = ((3, 4), (4, 4))
    assert gui._move_to_action_index(move) == trainer._move_to_action_index(move)


class _FakeNetwork(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

    def set_noise(self, enabled):  # the agent acts with NoisyNet noise; this network has none
        pass

    def reset_noise(self):
        pass


def _agent_preferring(agent, preferred_index, monkeypatch):
    """Make the agent's network value `preferred_index` highest and everything else equal."""
    def fake_forward(x):
        logits = torch.zeros(x.size(0), agent.action_size, agent.num_atoms)
        logits[:, :, 0] = 5.0                       # mass on the lowest return ...
        logits[:, preferred_index, -1] = 10.0       # ... except the preferred action
        return torch.log_softmax(logits, dim=2)
    monkeypatch.setattr(agent, 'q_network', _FakeNetwork(fake_forward))
    return agent


def _position_with_a_scout_lane(cpu):
    env = StrategoEnvironment(cpu)
    for _ in range(50):
        state = env.reset()
        long_moves = [m for m in env.get_valid_moves() if abs(m[0][0] - m[1][0]) + abs(m[0][1] - m[1][1]) > 1]
        if long_moves:
            return env, state, long_moves[0]
    pytest.skip("no long Scout move in 50 random setups")


@pytest.mark.parametrize("use_filter", [True, False])
def test_agent_can_choose_a_long_scout_move(cpu, tiny_network, config_override, monkeypatch, use_filter):
    """The thesis agent played every long Scout move as the one-square step in its direction."""
    config_override(IMITATION_ENABLED=False)
    env, state, long_move = _position_with_a_scout_lane(cpu)
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    agent.epsilon_start = agent.epsilon_end = 0.0
    agent.use_heuristic_filter = use_filter
    _agent_preferring(agent, agent._move_to_action_index(long_move), monkeypatch)
    chosen = agent.act_batch([state.board], [env.get_valid_moves()], [state], env_indices=[0])[0]
    assert chosen == long_move
