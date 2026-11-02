"""The docs/ analysis scripts import without side effects and run on files passed to them."""

import importlib
import json
import os
import sys

import pytest

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
SCRIPTS = ['architecture_comparison', 'full_comparison_analysis', 'generate_comparison',
           'debug_shape', 'parse_history', 'examine_checkpoint']


@pytest.fixture(autouse=True)
def _docs_on_path(monkeypatch):
    monkeypatch.syspath_prepend(DOCS)
    yield
    for name in SCRIPTS:
        sys.modules.pop(name, None)


@pytest.mark.parametrize('name', SCRIPTS)
def test_imports_without_side_effects(name, tmp_path, capsys):
    importlib.import_module(name)
    assert os.listdir(tmp_path) == []  # nothing written to the cwd
    assert capsys.readouterr().out == ''


def _history(tmp_path, name, episodes, wins_p1):
    """A training_history.json as MetricsTracker writes it: wins_p1_history holds the running win count."""
    data = {
        'rewards_p1': [1.0] * episodes,
        'wins_p1_history': [min(i + 1, wins_p1) for i in range(episodes)],
        'wins_p1': wins_p1, 'draws': 0, 'wins_by_flag': wins_p1, 'wins_by_depletion': 0,
        'global_step': episodes * 50,
        'avg_loss_p1_history': [0.5] * episodes, 'avg_q_values_p1': [0.1] * episodes,
    }
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def test_generate_comparison_writes_a_table(tmp_path):
    gc = importlib.import_module('generate_comparison')
    out = tmp_path / 'cmp.md'
    gc.main([_history(tmp_path, 'a.json', 10, 3), _history(tmp_path, 'b.json', 10, 7), '-o', str(out)])
    text = out.read_text()
    assert '| Total Episodes | 10 | 10 |' in text
    assert '| Overall Win Rate | 30.00% | 70.00% |' in text


@pytest.mark.xfail(strict=True, reason="Step 8: moving win rate sums the running win count (100 wins -> 5050%)")
def test_moving_win_rate_of_100_straight_wins_is_100_percent(tmp_path):
    gc = importlib.import_module('generate_comparison')
    result = gc.process_file(_history(tmp_path, 'h.json', 100, 100))
    assert result['Final Moving Win Rate (100-ep)'] == '100.00%'
    assert result['Max Moving Win Rate (100-ep)'] == '100.00%'
