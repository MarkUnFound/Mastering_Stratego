"""Step 7: curriculum (start, evaluation gates, caps, PFSP) and the scripted opponents' rank tables."""

import random
from collections import Counter

import pytest
import torch

from piece import PieceType as P

HIDDEN = -20


def _curriculum(tmp_path, phase=1):
    from curriculum import CurriculumManager
    return CurriculumManager(start_phase=phase, save_dir=str(tmp_path))


# ---------------------------------------------------------------- start and advancement

def test_training_starts_at_phase_one():
    from training_config import CURRICULUM_START_PHASE
    assert CURRICULUM_START_PHASE == 1


def test_draws_do_not_count_as_wins(tmp_path):
    """The thesis gate used decisive games only: 30 wins and 70 draws in 100 games passed 70%."""
    c = _curriculum(tmp_path)
    for opp in ('random', 'heuristic'):
        for k in range(100):
            c.update_metrics({'winner': 1 if k < 30 else 0, 'opponent_type': opp})
    assert not c.check_phase_transition()
    c.record_evaluation({'random': 0.3, 'heuristic': 0.3})
    assert not c.check_phase_transition()


def test_training_games_alone_never_advance_a_phase(tmp_path):
    c = _curriculum(tmp_path)
    for k in range(500):
        c.update_metrics({'winner': 1, 'opponent_type': 'random' if k % 2 else 'heuristic'})
    assert not c.check_phase_transition()


def test_every_phase_is_reachable_by_evaluation(tmp_path):
    from curriculum import TrainingPhase
    c = _curriculum(tmp_path)
    for phase in (1, 2, 3):
        assert c.current_phase == phase
        gates = c.evaluation_gates()
        assert gates
        c.record_evaluation({opp: threshold - 0.01 for opp, threshold in gates.items()})
        assert not c.check_phase_transition()
        c.record_evaluation({opp: threshold for opp, threshold in gates.items()})
        assert c.check_phase_transition() and c.advance_phase()
    assert c.current_phase == TrainingPhase.LEAGUE_TRAINING
    assert c.evaluation_gates() == {} and not c.check_phase_transition()


def test_an_evaluation_from_the_previous_phase_does_not_carry_over(tmp_path):
    c = _curriculum(tmp_path)
    c.record_evaluation({opp: 1.0 for opp in c.evaluation_gates()})
    assert c.check_phase_transition() and c.advance_phase()
    assert not c.check_phase_transition()


def test_a_phase_ends_at_its_episode_cap(tmp_path, config_override):
    config_override(PHASE_1_MAX_EPISODES=10)
    c = _curriculum(tmp_path)
    for _ in range(9):
        c.update_metrics({'winner': -1, 'opponent_type': 'random'})
    assert not c.check_phase_transition()
    c.update_metrics({'winner': -1, 'opponent_type': 'random'})
    assert c.check_phase_transition()


# ---------------------------------------------------------------- PFSP

def test_opponents_are_sampled_by_their_pfsp_weights(tmp_path, monkeypatch):
    """The thesis trainer cycled through the sorted opponent types in fixed blocks."""
    from curriculum import CurriculumManager
    c = _curriculum(tmp_path)
    monkeypatch.setattr(CurriculumManager, 'get_opponent_distribution',
                        lambda self: {'random': 0.9, 'smart_heuristic': 0.1})
    rng = random.Random(0)
    counts = Counter(c.sample_opponent(rng) for _ in range(4000))
    assert counts['random'] / 4000 == pytest.approx(0.9, abs=0.02)


def test_pfsp_weights_follow_the_recent_win_rate(tmp_path):
    c = _curriculum(tmp_path, phase=2)
    for _ in range(200):  # an old losing streak ...
        c.update_metrics({'winner': -1, 'opponent_type': 'frozen_heuristic'})
    for _ in range(100):  # ... then mastery
        c.update_metrics({'winner': 1, 'opponent_type': 'frozen_heuristic'})
    for k in range(100):
        c.update_metrics({'winner': 1 if k < 40 else -1, 'opponent_type': 'smart_heuristic'})
    metrics = c.metrics[c.current_phase]
    assert metrics.win_rate_vs('frozen_heuristic') == 1.0
    assert metrics.win_rate_vs('smart_heuristic') == pytest.approx(0.4)
    weights = c.get_opponent_distribution()
    assert weights['smart_heuristic'] > weights['frozen_heuristic']


# ---------------------------------------------------------------- battle rules and rank tables

def test_battle_from_values_matches_the_environment_rules():
    from battle import BattleResolver, battle_from_values
    resolver = BattleResolver()
    for attacker in P:
        for defender in P:
            expected = resolver.resolve_battle(attacker, defender)
            assert battle_from_values(attacker.value, -defender.value) == expected
            assert battle_from_values(-attacker.value, defender.value) == expected
    assert battle_from_values(P.MARSHAL.value, HIDDEN) is None


def test_is_enemy_value():
    from battle import is_enemy_value
    assert is_enemy_value(-5, 1) and is_enemy_value(5, -1)
    assert is_enemy_value(HIDDEN, 1) and is_enemy_value(HIDDEN, -1)  # the marker on either side's view
    assert not is_enemy_value(-13, 1) and not is_enemy_value(0, 1) and not is_enemy_value(5, 1)


def _board(setup):
    b = [[0] * 10 for _ in range(10)]
    for (r, c), v in setup.items():
        b[r][c] = v
    return b


def _scores(opponent, attacker, defender, score):
    """Score of P2's `attacker` (at (3, 0)) attacking P1's `defender` (at (4, 0))."""
    board = _board({(3, 0): -attacker, (4, 0): defender})
    return score(opponent, ((3, 0), (4, 0)), board)


SCORERS = {
    'heuristic': (lambda: __import__('curriculum').HeuristicOpponent('cpu', -1),
                  lambda o, m, b: o._score_move(m, b, -1)),
    'smart_heuristic': (lambda: __import__('curriculum').SmartHeuristicOpponent('cpu', -1),
                        lambda o, m, b: o._score_move(m, b, [m], -1)),
    'greedy': (lambda: __import__('opponents').GreedyAgent(player_id=-1),
               lambda o, m, b: o._score_move(m, b, -1)),
}


@pytest.mark.parametrize('name', sorted(SCORERS))
def test_scripted_opponents_know_the_special_battles(name):
    """Board values are PieceType values (Flag 1, Spy 2, Scout 3, Miner 4 ... Marshal 11, Bomb 12);
    the thesis tables read them as ranks 0-11, so the Spy and Miner never took their targets."""
    make, score = SCORERS[name]
    opponent = make()
    spy_takes_marshal = _scores(opponent, P.SPY.value, P.MARSHAL.value, score)
    spy_attacks_general = _scores(opponent, P.SPY.value, P.GENERAL.value, score)
    miner_takes_bomb = _scores(opponent, P.MINER.value, P.BOMB.value, score)
    major_hits_bomb = _scores(opponent, P.MAJOR.value, P.BOMB.value, score)
    assert spy_takes_marshal > spy_attacks_general
    assert miner_takes_bomb > major_hits_bomb


@pytest.mark.parametrize('name', ['heuristic', 'smart_heuristic'])
def test_a_hidden_piece_is_not_scored_as_a_certain_loss(name):
    """HIDDEN (-20) read as rank 20, so an attack on an unseen piece always looked lost."""
    make, score = SCORERS[name]
    opponent = make()
    hidden = score(opponent, ((3, 0), (4, 0)), _board({(3, 0): -P.SERGEANT.value, (4, 0): HIDDEN}))
    certain_loss = _scores(opponent, P.SERGEANT.value, P.MARSHAL.value, score)
    assert hidden > certain_loss


def test_smart_heuristic_values_its_own_pieces_by_type():
    from curriculum import SmartHeuristicOpponent
    values = SmartHeuristicOpponent('cpu', -1).piece_values
    assert values[P.MARSHAL.value] == 100 and values[P.GENERAL.value] == 80 and values[P.FLAG.value] == 200
    assert values[P.BOMB.value] == 0


def test_smart_heuristic_does_not_take_lakes_for_enemies():
    from curriculum import SmartHeuristicOpponent
    board = _board({(3, 2): P.CAPTAIN.value, (4, 2): -13, (2, 2): -P.SCOUT.value})
    assert SmartHeuristicOpponent('cpu', 1)._check_adjacent_enemies(3, 2, board, 1) == 1


# ---------------------------------------------------------------- evaluation games

def test_evaluation_games_are_counted_and_greedy(cpu, tiny_network):
    from drqn_agent import RainbowAgent
    from curriculum import TrueRandomOpponent
    from network.training.evaluation import play_evaluation_games
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    seen = []
    original = agent._q_forward_for_acting

    def spy(x):
        seen.append(agent.noisy_acting)
        return original(x)

    agent._q_forward_for_acting = spy
    result = play_evaluation_games(agent, TrueRandomOpponent(cpu), num_games=2, device=cpu, max_turns=30)
    assert result['wins'] + result['losses'] + result['draws'] == result['games'] == 2
    assert result['win_rate'] == result['wins'] / 2
    assert seen and not any(seen)
    assert agent.noisy_acting  # restored afterwards


# ---------------------------------------------------------------- trainer wiring

def test_trainer_evaluates_and_moves_through_the_phases(tmp_path, config_override, monkeypatch):
    """With tiny caps the run must leave Phase 1 (the thesis config started at Phase 4)."""
    import json
    from test_trainer_smoke import run_tiny_training
    from environment import CurriculumManager  # the class the trainer uses
    evaluations = []
    original = CurriculumManager.record_evaluation

    def record(self, results):
        evaluations.append((self.current_phase.value, dict(results)))
        return original(self, results)

    monkeypatch.setattr(CurriculumManager, 'record_evaluation', record)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=12,
                      EVAL_INTERVAL=4, EVAL_GAMES=1, PHASE_1_MAX_EPISODES=4, PHASE_2_MAX_EPISODES=4)
    assert evaluations and set(evaluations[0][1]) == {'random', 'heuristic'}
    state = json.load(open(tmp_path / 'models' / 'curriculum_state.json'))
    assert state['current_phase'] >= 2


def test_self_play_does_not_overwrite_the_learning_agent2(tmp_path, config_override, monkeypatch):
    """Phase 4: the thesis trainer loaded agent1's weights (or a league checkpoint) into agent2,
    the second learner, every time it was picked as the opponent."""
    from test_trainer_smoke import run_tiny_training
    from network import RainbowAgent
    from environment import CurriculumManager  # the class the trainer uses
    monkeypatch.setattr(CurriculumManager, 'get_opponent_distribution', lambda self: {'self': 0.5, 'league': 0.5})
    agents = []
    original_init = RainbowAgent.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        agents.append(self)

    monkeypatch.setattr(RainbowAgent, '__init__', init)
    overwrites = []
    original_load = torch.nn.Module.load_state_dict

    def load(self, *args, **kwargs):
        if any(self is a.q_network and a.player_id == -1 and not a.inference_only for a in agents):
            overwrites.append(self)
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', load)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=8, CURRICULUM_START_PHASE=4)
    assert not overwrites
    assert any(a.player_id == -1 and a.inference_only for a in agents)  # the frozen snapshot opponent


def test_agent2_and_the_snapshot_read_aaren_history_from_phase_one(tmp_path, config_override, monkeypatch):
    """With AAREN on, agent2 got no history before Phase 4 although its network reads the AAREN
    planes (and agent1's weights, which expect them, were loaded into it)."""
    from test_trainer_smoke import run_tiny_training
    from network import RainbowAgent
    agents = []
    original_init = RainbowAgent.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        agents.append(self)

    monkeypatch.setattr(RainbowAgent, '__init__', init)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2, AAREN_ENABLED=True)
    p2_agents = [a for a in agents if a.player_id == -1]
    assert len(p2_agents) == 2  # the learner agent2 and the frozen snapshot opponent
    assert all(a.history is not None and len(a.history_instances) == 2 for a in p2_agents)


def test_greedy_play_never_imitates_the_heuristic():
    """Imitation moves are for training only: an agent acting greedily plays its own argmax."""
    import torch
    from network import RainbowAgent
    from environment import StrategoEnvironment
    env = StrategoEnvironment(torch.device('cpu'))
    state = env.reset()
    agent = RainbowAgent(player_id=1, device=torch.device('cpu'), use_pbs=False, inference_only=True)
    calls = []
    agent.move_filter.get_filtered_actions = lambda *a, **k: calls.append(1) or []
    for _ in range(200):
        agent.act_batch([state.board], [env.get_valid_moves()], [state], env_indices=[None])
    assert not calls
