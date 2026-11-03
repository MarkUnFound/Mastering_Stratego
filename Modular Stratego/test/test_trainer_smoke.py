"""The real training loop runs end to end: a few short episodes, 2 lanes, tiny network, learning on."""

import torch


def run_tiny_training(tmp_path, config_override, monkeypatch, episodes=4, **extra):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config_override(**{
        'NUM_LANES': 2, 'NUM_ENVS': 2,
        'NETWORK_CHANNELS': 16, 'RESNET_BLOCKS': 1, 'ATTENTION_HEADS': 2,
        'CURRICULUM_START_PHASE': 1,
        'PHASE_MAX_TURNS': {1: 40, 2: 40, 3: 40, 4: 40},
        'WARMUP_STEPS': 16, 'BATCH_SIZE': 8, 'MEMORY_SIZE': 2000,
        'REPLAY_UPDATE_INTERVAL': 2, 'TARGET_UPDATE_INTERVAL': 20,
        'SAVE_INTERVAL': 10**9, 'PLOT_INTERVAL': 10**9,
        **extra,
    })
    import train_dqn
    from network import RainbowAgent

    losses = []
    original_replay = RainbowAgent.replay

    def counting_replay(self, *args, **kwargs):
        loss = original_replay(self, *args, **kwargs)
        if loss is not None and self.player_id == 1:
            losses.append(loss)
        return loss

    monkeypatch.setattr(RainbowAgent, 'replay', counting_replay)
    train_dqn.train_dqn_agents(num_episodes=episodes, save_interval=10**9, plot_interval=10**9,
                               model_save_path=str(tmp_path / 'models'), generate_gifs=False)
    return losses


def test_trainer_runs_and_learns(tmp_path, config_override, monkeypatch):
    losses = run_tiny_training(tmp_path, config_override, monkeypatch)
    assert losses, "replay() never produced a loss"
    assert all(torch.isfinite(torch.tensor(losses)))


def test_the_trainer_records_how_each_game_ended(tmp_path, config_override, monkeypatch):
    import train_dqn
    MetricsTracker = train_dqn.MetricsTracker  # the trainer's copy (training.* is a second module)
    trackers = []
    original_init = MetricsTracker.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trackers.append(self)

    monkeypatch.setattr(MetricsTracker, '__init__', init)
    run_tiny_training(tmp_path, config_override, monkeypatch)
    m = trackers[0].metrics
    episodes = len(m['rewards_p1'])
    assert episodes >= 4 and len(m['end_types']) == len(m['opponents']) == episodes
    draws = [t for t in m['end_types'] if t.startswith('draw:')]
    assert len(draws) == m['draws']
    assert all(t.split(':')[0] in ('win', 'loss', 'draw') for t in m['end_types'])
    # draws_by_timeout counts the turn-limit draws only (it counted every draw)
    assert m.get('draws_by_timeout', 0) == sum(t == 'draw:timeout' for t in draws)
