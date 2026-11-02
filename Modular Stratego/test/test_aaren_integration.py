"""AAREN history aggregator is wired into the agent's input: shapes, embeddings, forward pass.

(Converted from the check_aaren_integration.py print script.)
"""

import torch

from drqn_agent import RainbowAgent
from history_aggregator import HistoryAggregator
from networks.rainbow_dqn import RainbowDQN
from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS, HISTORY_EMBEDDING_SIZE
from environment import StrategoEnvironment

BOARD_CHANNELS = 15
ENEMY_MOVE = ((6, 0), (5, 0))  # treated as an opponent move by the aggregator


def test_embedding_size_matches_network_input(cpu, tiny_network):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    assert agent.history_embedding_size == HISTORY_EMBEDDING_SIZE
    assert agent.input_channels == BOARD_CHANNELS + HISTORY_EMBEDDING_SIZE
    assert agent.q_network.conv_in.in_channels == agent.input_channels


def test_embedding_is_zero_until_an_enemy_move_then_nonzero_at_destination(cpu):
    ha = HistoryAggregator(player_id=1, device=cpu, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
    state = StrategoEnvironment(cpu).reset()
    assert ha.get_embedding_tensor().abs().sum() == 0
    ha.update(ENEMY_MOVE, state, acting_player=-1)
    emb = ha.get_embedding_tensor()
    assert emb.shape == (AAREN_HIDDEN_SIZE, 10, 10)
    assert emb[:, 5, 0].abs().sum() > 0
    assert emb[:, 6, 0].abs().sum() == 0  # history follows the piece


def test_own_moves_are_not_tracked(cpu):
    ha = HistoryAggregator(player_id=1, device=cpu, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
    ha.update(ENEMY_MOVE, StrategoEnvironment(cpu).reset(), acting_player=1)
    assert ha.get_embedding_tensor().abs().sum() == 0


def test_state_representation_concatenates_board_and_embedding(cpu, tiny_network):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1, use_pbs=True)
    state = StrategoEnvironment(cpu).reset()
    agent.history.update(ENEMY_MOVE, state, acting_player=-1)
    x = agent.get_state_representation(state.board, pbs_instance=agent.history)
    assert x.shape == (BOARD_CHANNELS + HISTORY_EMBEDDING_SIZE, 10, 10)
    assert x[:BOARD_CHANNELS].sum() > 0
    assert x[BOARD_CHANNELS:].abs().sum() > 0


def test_forward_pass_gives_normalised_return_distributions(cpu, tiny_network):
    net = RainbowDQN(input_shape=(BOARD_CHANNELS + HISTORY_EMBEDDING_SIZE, 10, 10), output_size=400, num_atoms=51)
    log_probs = net(torch.randn(3, BOARD_CHANNELS + HISTORY_EMBEDDING_SIZE, 10, 10))
    assert log_probs.shape == (3, 400, 51)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(3, 400), atol=1e-5)


def test_many_enemy_moves_give_distinct_finite_embeddings(cpu):
    ha = HistoryAggregator(player_id=1, device=cpu, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
    state = StrategoEnvironment(cpu).reset()
    for move in [((6, 0), (5, 0)), ((6, 1), (5, 1)), ((5, 1), (5, 2)), ((7, 3), (6, 3))]:
        ha.update(move, state, acting_player=-1)
    emb = ha.get_embedding_tensor()
    active = emb.abs().sum(0) > 1e-6
    assert torch.isfinite(emb).all()
    assert active.sum() == 3  # (5,0), (5,2), (6,3)
    cols = emb[:, active]
    assert (cols[:, :1] - cols[:, 1:]).abs().sum() > 1e-6
