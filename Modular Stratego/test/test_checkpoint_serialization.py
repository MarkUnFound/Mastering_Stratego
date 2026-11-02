"""Checkpoints: periodic .pt files restore weights; full-state .tar.gz archives restore training state.

(Converted from the old script, which expected save_checkpoint to write a .tar.gz.
save_checkpoint writes weights-only .pt files; the full state goes through save_full_state.)
"""

import os

import pytest
import torch

from drqn_agent import RainbowAgent
from training import Checkpointer, MetricsTracker
from league import LeagueManager


def _agent(cpu, player_id=1):
    return RainbowAgent(player_id=player_id, device=cpu, buffer_size=1000, num_envs=1)


def _same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.q_network.state_dict().values(),
                                                 b.q_network.state_dict().values()))


def test_periodic_checkpoint_restores_weights_and_episode(cpu, tiny_network, tmp_path):
    agent = _agent(cpu)
    save_dir = str(tmp_path / 'models')
    os.makedirs(save_dir)
    ckpt = Checkpointer(save_dir=save_dir)
    ckpt.save_checkpoint(episode=100, agent1=agent, agent2=None,
                         league_manager=LeagueManager(league_dir=os.path.join(save_dir, 'league')),
                         curriculum=None, metrics_tracker=MetricsTracker(save_dir=save_dir),
                         league_interval=100)

    assert os.path.exists(os.path.join(save_dir, 'agent1_rainbow_episode_100.pt'))
    league_path = os.path.join(save_dir, 'agent1_league_episode_100.pt')
    league_state = torch.load(league_path, map_location='cpu', weights_only=True)
    assert 'q_network_state_dict' in league_state
    assert 'optimizer_state_dict' not in league_state
    assert 'memory_state_dict' not in league_state

    fresh = _agent(cpu)
    assert not _same_weights(agent, fresh)
    assert ckpt.load_agent_models(fresh, _agent(cpu, -1)) == 100
    assert _same_weights(agent, fresh)


def _full_state_round_trip(cpu, tmp_path):
    agent = _agent(cpu)
    agent.step_count = 1234
    agent.optimizer.param_groups[0]['lr'] = 0.000123
    c = agent.input_channels
    agent.memory.add(torch.randn(c, 10, 10), 42, 1.0, torch.randn(c, 10, 10), False)
    save_dir = str(tmp_path / 'models')
    os.makedirs(save_dir)
    ckpt = Checkpointer(save_dir=save_dir)
    path = ckpt.save_full_state(episode=7, agent1=agent, agent2=None, curriculum=None,
                                metrics_tracker=MetricsTracker(save_dir=save_dir))
    assert path.endswith('.tar.gz') and os.path.exists(path)
    fresh = _agent(cpu)
    episode = ckpt.load_agent_models(fresh, _agent(cpu, -1))
    return agent, fresh, episode


def test_full_state_archive_restores_weights_and_counters(cpu, tiny_network, tmp_path):
    agent, fresh, episode = _full_state_round_trip(cpu, tmp_path)
    assert episode == 7
    assert _same_weights(agent, fresh)
    assert fresh.step_count == 1234
    assert fresh.optimizer.param_groups[0]['lr'] == pytest.approx(0.000123)


def test_full_state_archive_restores_replay_memory(cpu, tiny_network, tmp_path):
    _, fresh, _ = _full_state_round_trip(cpu, tmp_path)
    assert len(fresh.memory) == 1
