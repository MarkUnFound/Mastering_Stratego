"""The training plots show what the run is doing: outcome shares over recent games (not a
cumulative rate from episode 1), the evaluation results against the phase gates, how games end,
and no figure of empty axes when AAREN is off."""
import os

import numpy as np

import training_visualizer as V
from training import Checkpointer, MetricsTracker


def test_moving_average_and_outcomes():
    assert np.allclose(V.moving_average([1, 0, 1, 1], window=2), [1, 0.5, 0.5, 1])
    # cumulative wins of both sides -> +1 / -1 / 0 per episode
    assert list(V.episode_outcomes([1, 1, 1, 2], [0, 1, 1, 1])) == [1, -1, 0, 1]


def fake_run(episodes=300, aaren=False):
    tracker = MetricsTracker(save_dir='unused')
    m = tracker.metrics
    rng = np.random.default_rng(0)
    w1 = w2 = 0
    for ep in range(episodes):
        outcome = rng.choice(['win', 'loss', 'draw'], p=[0.2, 0.5, 0.3])
        w1 += outcome == 'win'
        w2 += outcome == 'loss'
        m['wins_p1_history'].append(int(w1))
        m['wins_p2_history'].append(int(w2))
        m['rewards_p1'].append({'win': 0.6, 'loss': -1.3, 'draw': -0.9}[outcome])
        m['rewards_p2'].append({'win': -1.3, 'loss': 0.6, 'draw': -0.9}[outcome])
        m['lengths'].append(int(rng.integers(20, 1000)))
        m['phase_history'].append(1)
        m['episode_end_steps'].append(ep * 200)
        m['avg_q_values_p1'].append(0.1)
        m['avg_entropy_p1'].append(0.03)
        m.setdefault('end_types', []).append(
            {'win': 'win:flag_capture', 'loss': 'loss:no_moves', 'draw': 'draw:no_battle_draw'}[outcome])
        if aaren:
            m['aaren_grad_norm'].append(0.5)
            m['aaren_embedding_std'].append(0.1)
    m['losses_p1'] = list(rng.uniform(1, 4, 500))
    m['loss_steps_p1'] = list(range(0, 500 * 100, 100))
    m['eval_history'] = [
        {'episode': 100 * k, 'phase': 1, 'global_step': 0, 'gates': {'random': 0.7, 'heuristic': 0.5},
         'results': {'random': {'wins': 10 + k, 'losses': 20, 'draws': 10 - k, 'games': 40,
                                'win_rate': (10 + k) / 40},
                     'heuristic': {'wins': k, 'losses': 35, 'draws': 5 - k, 'games': 40, 'win_rate': k / 40}}}
        for k in (1, 2, 3)]
    m['global_step'] = episodes * 200
    return tracker


def capture_figures(monkeypatch):
    figures = {}
    original = V.plt.savefig

    def savefig(path, *args, **kwargs):
        figures[os.path.basename(path)] = V.plt.gcf()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(V.plt, 'savefig', savefig)
    monkeypatch.setattr(V.plt, 'close', lambda *a, **k: None)
    return figures


def test_the_plots_show_outcomes_evaluation_and_endings(tmp_path, monkeypatch):
    figures = capture_figures(monkeypatch)
    Checkpointer(save_dir=str(tmp_path)).plot_progress(episode=300, metrics_tracker=fake_run(),
                                                       global_step=60000, num_envs=2)
    # AAREN is off in this run: no figure of empty axes
    assert sorted(os.listdir(tmp_path)) == ['additional_metrics_episode_300.png',
                                            'training_progress_episode_300.png']
    progress = figures['training_progress_episode_300.png'].axes
    titles = [ax.get_title() for ax in progress]
    assert 'Training Games: Outcome Shares' in titles
    outcomes = progress[titles.index('Training Games: Outcome Shares')]
    shares = {line.get_label(): line.get_ydata() for line in outcomes.get_lines()}
    assert set(shares) == {'Agent wins', 'Opponent wins', 'Draws'}
    assert np.allclose(shares['Agent wins'] + shares['Opponent wins'] + shares['Draws'], 1.0)
    evaluation = next(ax for ax in progress if ax.get_title().startswith('Evaluation'))
    lines = {line.get_label(): line for line in evaluation.get_lines()}
    assert list(lines['vs random: wins'].get_xdata()) == [100, 200, 300]
    assert np.allclose(lines['vs heuristic: wins'].get_ydata(), [1 / 40, 2 / 40, 3 / 40])
    assert np.allclose(lines['random gate'].get_ydata(), 0.7)
    endings = next(ax for ax in figures['additional_metrics_episode_300.png'].axes
                   if ax.get_title() == 'How Training Games End')
    assert {line.get_label() for line in endings.get_lines()} == {
        'win: flag_capture', 'loss: no_moves', 'draw: no_battle_draw'}


def test_the_aaren_figure_is_drawn_when_aaren_has_data(tmp_path, monkeypatch):
    capture_figures(monkeypatch)
    Checkpointer(save_dir=str(tmp_path)).plot_progress(episode=300, metrics_tracker=fake_run(aaren=True),
                                                       global_step=60000, num_envs=2)
    assert 'aaren_progress_episode_300.png' in os.listdir(tmp_path)
