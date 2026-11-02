"""Regression tests for crashes fixed in Step 1."""

import io
import sys

import pytest
import torch

from test_trainer_smoke import run_tiny_training


def test_nan_reward_batch_is_skipped_without_crashing(cpu, tiny_network, config_override):
    """drqn_agent used tqdm.write without importing tqdm: a NaN reward batch raised NameError."""
    config_override(WARMUP_STEPS=0)
    from drqn_agent import RainbowAgent
    agent = RainbowAgent(player_id=1, device=cpu, buffer_size=200, num_envs=1)
    c = agent.input_channels
    for i in range(agent.batch_size * 2):
        agent.memory.add(torch.randn(c, 10, 10), i % agent.action_size, float('nan'), torch.randn(c, 10, 10), False)
    assert agent.replay() is None


def test_phase_transition_log_survives_a_cp1252_console(tmp_path, monkeypatch):
    """advance_phase printed a non-ASCII arrow, which raises UnicodeEncodeError on a Windows console."""
    from curriculum import CurriculumManager
    curriculum = CurriculumManager(start_phase=1, save_dir=str(tmp_path))
    console = io.TextIOWrapper(io.BytesIO(), encoding='cp1252', errors='strict')
    monkeypatch.setattr(sys, 'stdout', console)
    assert curriculum.advance_phase()
    console.flush()


def test_trainer_reraises_crashes(tmp_path, config_override, monkeypatch):
    """The training loop used to print the traceback and return normally, hiding crashes from callers."""
    from network import RainbowAgent

    def broken_replay(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(RainbowAgent, 'replay', broken_replay)
    with pytest.raises(RuntimeError, match="boom"):
        run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2)
