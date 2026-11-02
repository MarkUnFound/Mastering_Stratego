"""Shared pytest setup: import paths, CPU device, config overrides, slow-test switch."""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same import layout as train_dqn.py (modules import each other by bare name)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
for _d in ['environment', 'network', 'settings', 'test', 'visualizers', 'utils']:
    _p = os.path.join(PROJECT_ROOT, _d)
    if _p not in sys.path:
        sys.path.append(_p)

import matplotlib
matplotlib.use('Agg')
import torch

torch.set_num_threads(min(4, torch.get_num_threads()))

_PROJECT_KEY = os.path.normcase(os.path.realpath(PROJECT_ROOT))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow learning tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cpu():
    return torch.device('cpu')


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Code that writes relative paths (logs, plots) writes into the test's tmp dir, not the repo."""
    monkeypatch.chdir(tmp_path)


def _project_modules():
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and os.path.normcase(os.path.realpath(f)).startswith(_PROJECT_KEY):
            yield mod


@pytest.fixture
def config_override(monkeypatch):
    """Override training_config constants everywhere they were copied by `from ... import`.

    Modules read config at import time (`from training_config import X`, `from settings import *`),
    so the value is patched on every loaded project module that has the name, plus on
    training_config and settings so later imports pick it up. Undone after the test.
    """
    import training_config
    import settings

    def apply(**values):
        for name, value in values.items():
            monkeypatch.setattr(training_config, name, value, raising=False)
            monkeypatch.setattr(settings, name, value, raising=False)
            for mod in _project_modules():
                if hasattr(mod, name):
                    monkeypatch.setattr(mod, name, value)
    return apply


@pytest.fixture
def tiny_network(config_override):
    """A RainbowDQN small enough for fast CPU tests."""
    config_override(NETWORK_CHANNELS=16, RESNET_BLOCKS=1, ATTENTION_HEADS=2)
