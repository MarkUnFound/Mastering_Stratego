"""EpisodicReplayBuffer (trajectory segment sampling).

(Converted from a module-level script with check() counters and sys.exit at import; same checks.)
"""

import torch

from prioritized_memory import EpisodicReplayBuffer


def make_transition(step_idx, reward=0.1):
    state = torch.randn(15, 10, 10)  # 15-channel board only (AAREN reconstructed at replay)
    next_state = torch.randn(15, 10, 10)
    action = ((step_idx % 10, step_idx % 10), ((step_idx + 1) % 10, step_idx % 10))
    return state, action, reward, next_state, False


def add_episode(buf, env_id, length, outcome, final_reward=1.0):
    buf.start_episode(env_id)
    for step in range(length):
        is_last = step == length - 1
        reward = final_reward if is_last else 0.01
        state, action, _, next_state, _ = make_transition(step, reward)
        buf.add(env_id, state, action, reward, next_state, is_last)
    buf.end_episode(env_id, outcome, final_reward)


def test_add_end_sample_cycle():
    buf = EpisodicReplayBuffer(max_episodes=10, segment_length=4, device='cpu', num_envs=2)
    assert buf.num_episodes == 0 and len(buf) == 0
    add_episode(buf, env_id=0, length=20, outcome=1.0)
    assert buf.num_episodes == 1
    assert len(buf) == 20

    result = buf.sample_segments(4)
    assert result is not None and len(result) == 7
    states, actions, rewards, next_states, dones, hist_snaps, next_hist_snaps = result
    assert states.shape == (4, 15, 10, 10)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    assert len(hist_snaps) == 4
    assert all(r != 0 for r in rewards.tolist())


def test_episode_shorter_than_segment_is_still_sampled():
    buf = EpisodicReplayBuffer(max_episodes=10, segment_length=16, device='cpu', num_envs=1)
    add_episode(buf, env_id=0, length=5, outcome=-1.0)
    assert buf.num_episodes == 1
    result = buf.sample_segments(2)
    assert result is not None
    assert result[0].shape[0] == 2


def test_single_step_episode_is_discarded():
    buf = EpisodicReplayBuffer(max_episodes=10, segment_length=16, device='cpu', num_envs=1)
    buf.start_episode(0)
    state, action, _, next_state, _ = make_transition(0, 1.0)
    buf.add(0, state, action, 1.0, next_state, True)
    buf.end_episode(0, 1.0, 1.0)
    assert buf.num_episodes == 0


def test_draws_are_filtered_out():
    buf = EpisodicReplayBuffer(max_episodes=100, segment_length=4, device='cpu', num_envs=1)
    for _ in range(50):
        add_episode(buf, 0, length=10, outcome=0.0)
    for _ in range(5):
        add_episode(buf, 0, length=10, outcome=1.0)
    assert buf.num_episodes == 5
    stats = buf.get_stats()
    assert stats['wins'] == 5
    assert stats['draws'] == 0
    assert abs(stats['avg_length'] - 10.0) < 0.1


def test_fifo_eviction_at_capacity():
    buf = EpisodicReplayBuffer(max_episodes=5, segment_length=4, device='cpu', num_envs=1)
    for ep in range(10):
        add_episode(buf, 0, length=8, outcome=1.0 if ep >= 5 else -1.0)
    assert buf.num_episodes == 5
    assert len(buf) == 5 * 8
    stats = buf.get_stats()
    assert stats['wins'] == 5 and stats['losses'] == 0


def test_interleaved_envs_are_tracked_separately():
    buf = EpisodicReplayBuffer(max_episodes=10, segment_length=4, device='cpu', num_envs=4)
    for env_id in range(4):
        buf.start_episode(env_id)
    for step in range(15):
        for env_id in range(4):
            state, action, _, next_state, _ = make_transition(step)
            is_last = step == 14
            buf.add(env_id, state, action, 1.0 if is_last else 0.1, next_state, is_last)
    for env_id in range(4):
        buf.end_episode(env_id, outcome=1.0 if env_id % 2 == 0 else -1.0, total_reward=1.0)
    assert buf.num_episodes == 4
    assert len(buf) == 60


def test_sampled_actions_are_long():
    buf = EpisodicReplayBuffer(max_episodes=10, segment_length=4, device='cpu', num_envs=1)
    add_episode(buf, 0, length=10, outcome=1.0)
    result = buf.sample_segments(2)
    assert result is not None and result[1].dtype == torch.long


def test_config_flags():
    from training_config import EPISODE_REPLAY_ENABLED, EPISODE_REPLAY_MAX_EPISODES
    assert isinstance(EPISODE_REPLAY_ENABLED, bool)
    assert EPISODE_REPLAY_MAX_EPISODES > 0
