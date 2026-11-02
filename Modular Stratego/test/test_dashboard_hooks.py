"""The tensors the dashboard reads (attention map, C51 distributions, AAREN embedding) have the right shapes.

(Converted from the old script; it loaded a model from a hard-coded path on one machine.
A freshly initialised agent has the same shapes.)
"""

import torch

from drqn_agent import RainbowAgent
from environment import StrategoEnvironment
from training_config import HISTORY_EMBEDDING_SIZE


def test_dashboard_tensors(cpu, tiny_network):
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    agent.q_network.eval()

    captured = {}
    handle = agent.q_network.spatial_attention.attn.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('attn', output[1]))
    state = StrategoEnvironment(cpu).reset()
    x = agent.get_state_representation(state.board, pbs_instance=agent.history)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    with torch.no_grad():
        probs = agent.q_network(x).exp()
    handle.remove()

    assert captured['attn'].shape == (1, 100, 100)
    assert probs.shape == (1, agent.action_size, 51)
    assert torch.allclose(probs.sum(-1), torch.ones(1, agent.action_size), atol=1e-5)
    assert agent.history.get_embedding_tensor().shape == (HISTORY_EMBEDDING_SIZE, 10, 10)
