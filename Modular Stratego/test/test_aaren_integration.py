"""AAREN history aggregator: wiring into the agent's input, and the bugs fixed in Step 4.

AAREN is off by default (AAREN_ENABLED); these tests switch it on.
(Converted from the check_aaren_integration.py print script.)
"""

import pytest
import torch

from drqn_agent import RainbowAgent
from history_aggregator import HistoryAggregator
from networks.rainbow_dqn import RainbowDQN
from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS, HISTORY_EMBEDDING_SIZE
from environment import StrategoEnvironment
from board import HIDDEN_PIECE, LAKE_SQUARE
from piece import PieceType

OBS_PLANES = 30
NUM_ACTIONS = 3600
ENEMY_MOVE = ((6, 0), (5, 0))  # treated as an opponent move by the aggregator
LAKES = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]


@pytest.fixture
def aaren_on(config_override, tiny_network):
    config_override(AAREN_ENABLED=True)


def _aggregator(cpu, player=1, hidden=AAREN_HIDDEN_SIZE):
    ha = HistoryAggregator(player_id=player, device=cpu, hidden_size=hidden, num_layers=AAREN_NUM_LAYERS)
    ha.aaren.eval()
    return ha


def test_aaren_is_off_by_default(cpu, tiny_network):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    assert agent.history is None
    assert agent.input_channels == OBS_PLANES
    agent.enable_history(1)  # the trainer calls this at Phase 4; must not change the input
    assert agent.history is None and agent.input_channels == OBS_PLANES


def test_embedding_size_matches_network_input(cpu, aaren_on):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    assert agent.history_embedding_size == HISTORY_EMBEDDING_SIZE
    assert agent.input_channels == OBS_PLANES + HISTORY_EMBEDDING_SIZE
    assert agent.q_network.conv_in.in_channels == agent.input_channels


def test_embedding_is_zero_until_an_enemy_move_then_nonzero_at_destination(cpu):
    ha = _aggregator(cpu)
    state = StrategoEnvironment(cpu).reset()
    assert ha.get_embedding_tensor().abs().sum() == 0
    ha.update(ENEMY_MOVE, state, acting_player=-1)
    emb = ha.get_embedding_tensor()
    assert emb.shape == (AAREN_HIDDEN_SIZE, 10, 10)
    assert emb[:, 5, 0].abs().sum() > 0
    assert emb[:, 6, 0].abs().sum() == 0  # history follows the piece


def test_own_moves_are_not_tracked(cpu):
    ha = _aggregator(cpu)
    ha.update(ENEMY_MOVE, StrategoEnvironment(cpu).reset(), acting_player=1)
    assert ha.get_embedding_tensor().abs().sum() == 0


def test_state_representation_concatenates_observation_and_embedding(cpu, aaren_on):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1, use_pbs=True)
    state = StrategoEnvironment(cpu).reset()
    agent.history.update(ENEMY_MOVE, state, acting_player=-1)
    x = agent.get_state_representation(state, pbs_instance=agent.history)
    assert x.shape == (OBS_PLANES + HISTORY_EMBEDDING_SIZE, 10, 10)
    assert x[:OBS_PLANES].sum() > 0
    assert x[OBS_PLANES:, 5, 0].abs().sum() > 0


def test_player_two_embedding_planes_are_in_its_canonical_frame(cpu, aaren_on):
    agent = RainbowAgent(player_id=-1, device=cpu, num_envs=1, use_pbs=True)
    state = StrategoEnvironment(cpu).reset()
    agent.history.update(((6, 0), (5, 0)), state, acting_player=1)
    x = agent.get_state_representation(state, pbs_instance=agent.history)
    assert x[OBS_PLANES:, 4, 9].abs().sum() > 0   # (5, 0) rotated 180 degrees
    assert x[OBS_PLANES:, 5, 0].abs().sum() == 0


def test_forward_pass_gives_normalised_return_distributions(cpu, tiny_network):
    channels = OBS_PLANES + HISTORY_EMBEDDING_SIZE
    net = RainbowDQN(input_shape=(channels, 10, 10), output_size=NUM_ACTIONS, num_atoms=51)
    log_probs = net(torch.randn(3, channels, 10, 10))
    assert log_probs.shape == (3, NUM_ACTIONS, 51)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(3, NUM_ACTIONS), atol=1e-5)


def test_many_enemy_moves_give_distinct_finite_embeddings(cpu):
    ha = _aggregator(cpu)
    state = StrategoEnvironment(cpu).reset()
    for move in [((6, 0), (5, 0)), ((6, 1), (5, 1)), ((5, 1), (5, 2)), ((7, 3), (6, 3))]:
        ha.update(move, state, acting_player=-1)
    emb = ha.get_embedding_tensor()
    active = emb.abs().sum(0) > 1e-6
    assert torch.isfinite(emb).all()
    assert active.sum() == 3  # (5,0), (5,2), (6,3)
    cols = emb[:, active]
    assert (cols[:, :1] - cols[:, 1:]).abs().sum() > 1e-6


# --- Step 4 bug fixes -------------------------------------------------------------------------

def test_embedding_is_the_networks_representation_of_the_piece_history(cpu):
    """The thesis aggregator wrote the last layer's raw attention numerator into the planes,
    not the layer-normalised output the classifier is trained on."""
    ha = _aggregator(cpu)
    state = StrategoEnvironment(cpu).reset()
    for move in [((3, 0), (4, 0)), ((4, 0), (5, 0)), ((5, 0), (5, 1))]:
        ha.update(move, state, acting_player=-1)
    sequence = torch.tensor([ha.position_histories[(5, 1)]])
    with torch.no_grad():
        expected = ha.aaren.forward_embedding(sequence)[0]
    assert torch.allclose(ha.get_embedding_tensor()[:, 5, 1], expected, atol=1e-5)


def test_padding_does_not_change_a_sequences_prediction(cpu):
    """Supervised training pads sequences at the end; the thesis model predicted from the last
    (padding) step of every shorter sequence."""
    ha = _aggregator(cpu, hidden=16)
    short = [[0.1 * (i + 1)] * ha.input_size for i in range(2)]
    long = [[0.05 * (i + 1)] * ha.input_size for i in range(6)]
    padded = short + [[0.0] * ha.input_size] * (len(long) - len(short))
    with torch.no_grad():
        alone = ha.aaren.forward_parallel(torch.tensor([short]))
        batched = ha.aaren.forward_parallel(torch.tensor([padded, long]), torch.tensor([2, 6]))[:1]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_supervised_training_uses_each_sequences_real_length(cpu, monkeypatch):
    ha = _aggregator(cpu, hidden=16)
    ha.training_buffer = [([[0.1] * ha.input_size] * (2 + i % 5), i % 12) for i in range(40)]
    seen = {}
    real = ha.aaren.forward_parallel

    def spy(x, lengths=None):
        seen['lengths'] = lengths
        return real(x, lengths)
    monkeypatch.setattr(ha.aaren, 'forward_parallel', spy)
    ha.train(epochs=1)
    assert seen['lengths'] is not None and int(seen['lengths'].min()) >= 2


def test_predictions_do_not_feed_a_fake_zero_move(cpu):
    ha = _aggregator(cpu)
    state = StrategoEnvironment(cpu).reset()
    ha.update(((3, 0), (4, 0)), state, acting_player=-1)
    probs = ha.get_piece_predictions((4, 0))
    with torch.no_grad():
        expected = torch.softmax(ha.aaren.head(ha.get_embedding_tensor()[:, 4, 0].unsqueeze(0)), dim=1)[0]
    assert torch.allclose(torch.tensor([probs[PieceType(i + 1)] for i in range(12)]), expected, atol=1e-5)


def _battle_env(cpu, enemy_type, own_type):
    """Enemy `enemy_type` at (3, 0) and our `own_type` at (5, 0), Player 2 (enemy) to move."""
    env = StrategoEnvironment(cpu)
    env.reset()
    b = env.board
    b.actual_board.zero_()
    for r, c in LAKES:
        b.actual_board[r, c] = LAKE_SQUARE
    b.actual_board[3, 0] = -enemy_type.value
    b.actual_board[5, 0] = own_type.value
    b.actual_board[0, 9] = -PieceType.FLAG.value
    b.actual_board[9, 9] = PieceType.FLAG.value
    b.visible_board_p1.copy_(b.actual_board)
    b.visible_board_p1[3, 0] = HIDDEN_PIECE
    b.visible_board_p1[0, 9] = HIDDEN_PIECE
    b.visible_board_p2.copy_(b.actual_board)
    b.visible_board_p2[5, 0] = HIDDEN_PIECE
    b.visible_board_p2[9, 9] = HIDDEN_PIECE
    env._flag_positions = {1: (9, 9), -1: (0, 9)}
    env.current_player = -1
    return env


def _enemy_moves_then_attacks(env, ha):
    """Trainer order: the history is updated with the move, then the battle is recorded."""
    for move in [((3, 0), (4, 0)), ((4, 0), (5, 0))]:
        env.current_player = -1
        state = env._get_game_state()
        ha.update(move, state, acting_player=-1)
        _, _, _, info = env.step(move)
    (_, attacker), (_, defender) = info['revealed_in_step']
    ha.record_battle(move, attacker, defender, acting_player=-1)
    return info


def test_enemy_attacker_is_labelled_with_its_own_type(cpu):
    """The thesis trainer labelled an attacking enemy's history with the type of our piece it hit."""
    env = _battle_env(cpu, PieceType.SCOUT, PieceType.SERGEANT)
    ha = _aggregator(cpu, hidden=16)
    _enemy_moves_then_attacks(env, ha)
    assert [PieceType(t + 1) for _, t in ha.training_buffer] == [PieceType.SCOUT]
    assert (5, 0) not in ha.position_histories  # the Scout died: its history is gone


def test_surviving_enemy_attacker_keeps_its_history(cpu):
    env = _battle_env(cpu, PieceType.MARSHAL, PieceType.SERGEANT)
    ha = _aggregator(cpu, hidden=16)
    _enemy_moves_then_attacks(env, ha)
    assert [PieceType(t + 1) for _, t in ha.training_buffer] == [PieceType.MARSHAL]
    assert len(ha.position_histories[(5, 0)]) == 2


def test_defending_enemy_is_labelled_with_its_own_type(cpu):
    ha = _aggregator(cpu, hidden=16)
    state = StrategoEnvironment(cpu).reset()
    ha.update(((3, 0), (4, 0)), state, acting_player=-1)
    ha.update(((4, 0), (4, 1)), state, acting_player=-1)
    ha.record_battle(((5, 1), (4, 1)), PieceType.MAJOR, PieceType.CAPTAIN, acting_player=1)
    assert [PieceType(t + 1) for _, t in ha.training_buffer] == [PieceType.CAPTAIN]
    assert (4, 1) not in ha.position_histories  # we won: the Captain died


def test_supervised_training_reaches_every_aaren_tensor(cpu):
    """The thesis relied on a 'gradient bridge' from the DQN loss, which reached 1 of AAREN's 21
    tensors. AAREN is trained by its supervised reveal loss; that must reach all of them."""
    ha = HistoryAggregator(player_id=1, device=cpu, hidden_size=16, num_layers=AAREN_NUM_LAYERS)
    ha.training_buffer = [([[0.01 * (i + j)] * ha.input_size for j in range(2 + i % 4)], i % 12) for i in range(64)]
    grads = {}
    real_step = ha.optimizer.step

    def capture(*a, **k):
        for n, p in ha.aaren.named_parameters():
            grads[n] = 0.0 if p.grad is None else float(p.grad.abs().sum())
        return real_step(*a, **k)
    ha.optimizer.step = capture
    assert ha.train(epochs=1) is not None
    assert len(grads) == len(list(ha.aaren.parameters()))
    assert all(g > 0 for g in grads.values()), [n for n, g in grads.items() if g == 0]


def test_dqn_optimizer_does_not_hold_aaren_parameters(cpu, aaren_on):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=2, use_pbs=True)
    dqn = {id(p) for group in agent.optimizer.param_groups for p in group['params']}
    assert not dqn & {id(p) for p in agent.shared_aaren.parameters()}
    assert dqn == {id(p) for p in agent.q_network.parameters()}


def test_all_lanes_reveal_data_reaches_the_shared_model(cpu, aaren_on):
    """train_history() trains on lane 0's buffer; the thesis code left the other lanes' labelled
    reveals in buffers nothing trained on."""
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=2, use_pbs=True)
    lane1 = agent.history_instances[1]
    lane1.training_buffer.extend(([[0.1] * lane1.input_size] * 3, 4) for _ in range(40))
    assert agent.train_history(epochs=1) is not None


def test_next_state_is_stored_with_the_history_it_will_be_acted_on_with(cpu, aaren_on, config_override):
    """The thesis stored the state's snapshot for the next state too and replaced it by zero
    planes at replay, so the target network read padding as 'no history'."""
    config_override(N_STEPS=1, ENABLE_DATA_AUGMENTATION=False, WARMUP_STEPS=0)
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1, use_pbs=True, batch_size=1, buffer_size=10)
    env = StrategoEnvironment(cpu)
    state = env.reset()
    agent.history.update(((3, 0), (4, 0)), state, acting_player=-1)
    before = agent.history.get_history_snapshot()
    agent.history.update(((3, 1), (4, 1)), state, acting_player=-1)  # the opponent's reply
    agent.remember_batch([state.board], [((6, 0), (5, 0))], [0.0], [state.board], [False], [True],
                         [state], [state], lane_ids=[0], history_snapshots=[before])
    exp = agent.memory.tree.data[0]
    assert torch.equal(exp.history_snapshot, before)
    assert exp.next_history_snapshot[:, 4, 1].abs().sum() > 0
    assert exp.history_snapshot[:, 4, 1].abs().sum() == 0

    seen = {}
    real = agent.q_network.forward

    def spy(x):
        seen.setdefault('inputs', []).append(x.detach().clone())
        return real(x)
    agent.q_network.forward = spy
    agent.amp_enabled = False
    agent.replay(batch_size=1)
    next_input = seen['inputs'][1]  # online pass on the next states
    assert next_input[0, OBS_PLANES:, 4, 1].abs().sum() > 0


def test_trainer_runs_with_aaren_on(tmp_path, config_override, monkeypatch):
    """Both agents track histories (agent2 from Phase 4) and battles are recorded per tracker."""
    from test_trainer_smoke import run_tiny_training
    battles = []
    real = HistoryAggregator.record_battle

    def spy(self, move, attacker_type, defender_type, acting_player, **kwargs):
        battles.append(self.player_id)
        return real(self, move, attacker_type, defender_type, acting_player, **kwargs)
    monkeypatch.setattr(HistoryAggregator, 'record_battle', spy)
    losses = run_tiny_training(tmp_path, config_override, monkeypatch, episodes=4,
                               AAREN_ENABLED=True, AAREN_HIDDEN_SIZE=16, HISTORY_EMBEDDING_SIZE=16,
                               CURRICULUM_START_PHASE=4)
    assert losses and all(torch.isfinite(torch.tensor(losses)))
    assert 1 in battles
