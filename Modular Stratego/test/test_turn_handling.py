"""Turn handling: who moves first, whose view each transition holds, terminal rewards, per-lane buffers."""

from collections import Counter

import pytest
import torch

from environment import StrategoEnvironment
from test_trainer_smoke import run_tiny_training


# ---------------------------------------------------------------- environment

@pytest.mark.parametrize('first', [1, -1])
def test_reset_sets_the_starting_player(cpu, first):
    env = StrategoEnvironment(cpu)
    state = env.reset(starting_player=first)
    assert env.current_player == state.current_player == first
    sources = {src for src, _ in env.get_valid_moves()}
    assert all(env.board.actual_board[src].item() * first > 0 for src in sources)


# ---------------------------------------------------------------- trainer

def _record_moves(monkeypatch, record):
    """Wrap every agent's act_batch: record (side the agent plays, side to move, owner of the moved piece)."""
    from network import RainbowAgent
    from network.opponents import RandomAgent
    from curriculum import TrueRandomOpponent, HeuristicOpponent, SmartHeuristicOpponent

    for cls in (RainbowAgent, RandomAgent, TrueRandomOpponent, HeuristicOpponent, SmartHeuristicOpponent):
        original = cls.act_batch

        def wrapped(self, states, valid_moves_list, game_states=None, *args, _original=original, **kwargs):
            actions = _original(self, states, valid_moves_list, game_states, *args, **kwargs)
            side = getattr(self, 'player_id', -1)
            for gs, move in zip(game_states, actions):
                if move is not None:
                    owner = 1 if gs.board[move[0]].item() > 0 else -1
                    record.append((side, gs.current_player, owner))
            return actions

        monkeypatch.setattr(cls, 'act_batch', wrapped)


def test_every_move_is_made_by_the_side_to_move_with_its_own_pieces(tmp_path, config_override, monkeypatch):
    record = []
    _record_moves(monkeypatch, record)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=6)
    assert record
    bad = Counter(r for r in record if not (r[0] == r[1] == r[2]))
    assert not bad, f"{sum(bad.values())} of {len(record)} moves made out of turn: {dict(bad)}"


def test_both_sides_start_games(tmp_path, config_override, monkeypatch):
    """With a random starting player, P2 should open a fair share of the games."""
    from environment import parallel_environment
    starts = []
    original = parallel_environment.ParallelStrategoEnvironment.reset

    def recording_reset(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        starts.extend(s.current_player for s in out[0])
        return out

    monkeypatch.setattr(parallel_environment.ParallelStrategoEnvironment, 'reset', recording_reset)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2, NUM_LANES=8, NUM_ENVS=8)
    assert set(starts) == {1, -1}


def test_stored_transitions_hold_the_agents_own_view_and_one_terminal_per_game(tmp_path, config_override, monkeypatch):
    from network import RainbowAgent
    stored = []
    original = RainbowAgent.remember_batch

    def recording(self, states, actions, rewards, next_states, dones, active_mask, game_states=None,
                  next_game_states=None, infos=None, **kwargs):
        if self.player_id == 1:
            for gs, ngs, r, d, lane in zip(game_states, next_game_states, rewards, dones, kwargs['lane_ids']):
                stored.append((gs.current_player, ngs.current_player, ngs.game_over, r, d, lane))
        return original(self, states, actions, rewards, next_states, dones, active_mask, game_states,
                        next_game_states, infos, **kwargs)

    monkeypatch.setattr(RainbowAgent, 'remember_batch', recording)
    episodes = 6
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=episodes)
    assert stored
    assert all(cur == 1 for cur, *_ in stored), "state is not P1's own view"
    for cur, nxt, over, r, done, lane in stored:
        assert done == over
        if not done:
            assert nxt == 1, "next state must be P1's view after the opponent's reply"
        else:
            assert r in (-1.0, 0.0, 1.0)
    # every finished game gives P1 exactly one terminal transition (the run may finish a few extra games)
    assert sum(d for *_, d, _ in stored) >= episodes


# ---------------------------------------------------------------- TurnTracker

class S:
    """Stand-in for a GameState: whose view it is."""
    def __init__(self, view):
        self.current_player = view


def test_move_and_reply_make_one_transition_ending_in_the_movers_view():
    from network import TurnTracker
    tt = TurnTracker(2)
    s1, s2, s3 = S(1), S(-1), S(1)
    assert tt.record(0, 1, s1, 'a', s2, False, {'revealed_in_step': ['x']}) == []
    [t] = tt.record(0, -1, s2, 'b', s3, False, {'revealed_in_step': ['y']})
    assert (t['player'], t['state'], t['action'], t['next_state'], t['done']) == (1, s1, 'a', s3, False)
    assert t['info']['revealed_in_step'] == ['x', 'y']  # battles of the move and of the reply


def test_game_end_closes_both_sides_with_the_outcome():
    from network import TurnTracker, outcome_reward
    tt = TurnTracker(1)
    tt.record(0, 1, S(1), 'a', S(-1), False, {})
    end = {'winner': -1}
    ts = tt.record(0, -1, S(-1), 'b', S(1), True, end)  # P2's move wins
    assert [(t['player'], t['done'], outcome_reward(t['winner'], t['player'])) for t in ts] == [(1, True, -1.0), (-1, True, 1.0)]
    draw = TurnTracker(1)
    draw.record(0, 1, S(1), 'a', S(-1), False, {})
    assert [outcome_reward(t['winner'], t['player']) for t in draw.record(0, -1, S(-1), 'b', S(1), True, {'winner': 0})] == [0.0, 0.0]


def test_lanes_do_not_mix():
    from network import TurnTracker
    tt = TurnTracker(2)
    tt.record(0, 1, S(1), 'lane0', S(-1), False, {})
    assert tt.record(1, -1, S(-1), 'b', S(1), False, {}) == []
    [t] = tt.record(0, -1, S(-1), 'c', S(1), False, {})
    assert t['action'] == 'lane0' and t['lane'] == 0


# ---------------------------------------------------------------- n-step buffer and lane keying

def test_n_step_buffer_keeps_the_episode_tail():
    from prioritized_memory import NStepBuffer
    buf = NStepBuffer(n_steps=3, gamma=0.5)
    out = []
    for t in range(5):
        r = buf.add(t, t, 1.0, t + 1, t == 4)
        if r:
            out.append(r)
    out += buf.flush()
    assert [o[1] for o in out] == [0, 1, 2, 3, 4]
    assert [o[2] for o in out] == [1.75, 1.75, 1.75, 1.5, 1.0]
    assert [o[4] for o in out] == [False, False, True, True, True]
    assert [o[3] for o in out] == [3, 4, 5, 5, 5]


def test_remember_batch_uses_the_lanes_n_step_buffer(cpu, tiny_network):
    from drqn_agent import RainbowAgent
    from environment import StrategoEnvironment
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=3, buffer_size=100)
    gs = StrategoEnvironment(cpu).reset()
    move = StrategoEnvironment(cpu).get_valid_moves()[0]
    agent.remember_batch([gs.board], [move], [0.0], [gs.board], [False], [True], [gs], [gs], infos=[{}], lane_ids=[2])
    assert [len(b.buffer) for b in agent.n_step_buffers] == [0, 0, 1]


def test_opponent_moves_update_the_lanes_own_history(cpu, tiny_network):
    from drqn_agent import RainbowAgent
    from environment import StrategoEnvironment
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=3)
    state = StrategoEnvironment(cpu).reset()
    agent.update_history_batch([((3, 0), (4, 0))], [state], acting_player=-1, lane_ids=[2])
    sums = [h.get_embedding_tensor().abs().sum().item() for h in agent.history_instances]
    assert sums[0] == 0 and sums[1] == 0 and sums[2] > 0
    x = agent.get_batch_state_representation([state.board], [state], env_indices=[2])
    assert x[0, 15:].abs().sum() > 0


def test_self_play_stores_the_opponents_own_view(tmp_path, config_override, monkeypatch):
    """From Phase 4 agent2 learns too: its transitions must be P2's own view, after P1's reply."""
    from network import RainbowAgent
    stored = []
    original = RainbowAgent.remember_batch

    def recording(self, states, actions, rewards, next_states, dones, active_mask, game_states=None,
                  next_game_states=None, infos=None, **kwargs):
        if self.player_id == -1:
            stored.extend((gs.current_player, ngs.current_player, d) for gs, ngs, d in zip(game_states, next_game_states, dones))
        return original(self, states, actions, rewards, next_states, dones, active_mask, game_states,
                        next_game_states, infos, **kwargs)

    monkeypatch.setattr(RainbowAgent, 'remember_batch', recording)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=4, CURRICULUM_START_PHASE=4)
    assert stored
    assert all(cur == -1 and (done or nxt == -1) for cur, nxt, done in stored)
