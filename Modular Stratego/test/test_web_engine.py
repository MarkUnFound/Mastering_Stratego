"""The browser engine (web/play.html) plays by the Python environment's rules and builds the same
observation planes and action indices: games recorded here are replayed there by node."""
import importlib.util
import json
import os
import random
import shutil
import subprocess

import pytest
import torch

from action_encoding import legal_action_indices
from environment import StrategoEnvironment, HeuristicSetupAgent
from observation import encode_observation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which('node')


def record_game(seed, max_turns, no_battle):
    random.seed(seed)
    torch.manual_seed(seed)
    env = StrategoEnvironment(torch.device('cpu'), full_observability=False, max_turns=max_turns,
                              no_battle_draw_plies=no_battle)
    setups = {p: HeuristicSetupAgent(player_id=p) for p in (1, -1)}
    place = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p)) for p in (1, -1)}
    start = 1 if seed % 2 == 0 else -1
    state = env.reset(place[1], place[-1], starting_player=start)
    plies, done, info = [], False, {}
    while not done:
        moves = env.get_valid_moves()
        obs = encode_observation(state, state.current_player)
        # prefer attacks now and then, so battles, reveals and flag captures happen
        attacks = [m for m in moves if env.board.actual_board[m[1]].item() not in (0, -13)]
        move = random.choice(attacks) if attacks and random.random() < 0.5 else random.choice(moves)
        plies.append({'player': state.current_player,
                      'legal': [[list(a), list(b)] for a, b in moves],
                      'indices': sorted(legal_action_indices(moves, state.current_player)),
                      'obs_nonzero': torch.nonzero(obs[:28].flatten()).flatten().tolist(),
                      'no_battle_plane': float(obs[28, 0, 0]), 'progress_plane': float(obs[29, 0, 0]),
                      'move': [list(move[0]), list(move[1])]})
        state, _, done, info = env.step(move)
    return {'p1': [[p.value, list(rc)] for p, rc in place[1]], 'p2': [[p.value, list(rc)] for p, rc in place[-1]],
            'start': start, 'max_turns': max_turns, 'no_battle': no_battle,
            'winner': info.get('winner', 0), 'win_type': info.get('win_type'), 'plies': plies}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_engine_matches_the_python_environment(tmp_path):
    games = [record_game(seed, max_turns=300, no_battle=40) for seed in range(6)]
    assert {g['win_type'] for g in games} - {None}, games
    path = tmp_path / 'games.json'
    path.write_text(json.dumps(games))
    out = subprocess.run([NODE, os.path.join(ROOT, 'web', 'test_engine.js'), str(path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr + out.stdout
    assert out.stdout.startswith('ok')


@pytest.mark.skipif(not (importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime')),
                    reason="onnx / onnxruntime are not installed")
def test_onnx_export_gives_the_agents_expected_q_values(tmp_path, tiny_network):
    """tools/export_onnx.py: the browser model returns sum_z p(z) z of the greedy network."""
    import sys
    import numpy as np
    import onnxruntime
    from network import RainbowAgent
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import export_onnx
    agent = RainbowAgent(player_id=1, device=torch.device('cpu'), use_pbs=False, buffer_size=100)
    agent.save_model(str(tmp_path / 'agent.pth'))
    out = export_onnx.export(str(tmp_path / 'agent.pth'), str(tmp_path / 'agent.onnx'))
    env = StrategoEnvironment(torch.device('cpu'))
    obs = torch.stack([encode_observation(env.reset(), 1), encode_observation(env.reset(starting_player=-1), -1)])
    net = export_onnx.load_agent(str(tmp_path / 'agent.pth')).q_network.eval()
    with torch.no_grad():
        expected = (net(obs).exp() * agent.support).sum(2).numpy()
    got = onnxruntime.InferenceSession(out).run(None, {'obs': obs.numpy()})[0]
    assert got.shape == (2, 3600)
    assert np.abs(got - expected).max() < 1e-4
    assert json.load(open(tmp_path / 'agent.json'))['input_planes'] == 30
