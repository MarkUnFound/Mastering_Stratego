"""
Export a trained agent's Q-network to ONNX for the browser player (web/play.html).

    python tools/export_onnx.py <checkpoint> [out.onnx]

<checkpoint> is a RainbowAgent save: a .pth from save_model, an agent1_league_episode_*.pt, or a
checkpoint .tar.gz from the Checkpointer. The ONNX model takes the observation planes
"obs" (B, C, 10, 10) float32 in the mover's canonical frame (environment/observation.py; C = 30,
plus the AAREN planes if the agent was trained with AAREN_ENABLED, which the browser fills with
zeros) and returns "q" (B, 3600): the expected value of each action's C51 distribution, indexed
as in environment/action_encoding.py. The browser masks the illegal actions and plays the argmax.
A JSON sidecar (<out>.json) records the input planes and the encoding version.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ('', 'environment', 'network', 'settings', 'test', 'visualizers', 'utils'):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from network import RainbowAgent  # noqa: E402
from network.training import Checkpointer  # noqa: E402


class ExpectedQ(torch.nn.Module):
    """Greedy-acting head: mean weights (NoisyNet noise off), eval mode, Q = sum_z p(z) z."""

    def __init__(self, net, support):
        super().__init__()
        self.net = net
        self.register_buffer('support', support.detach().clone().float())

    def forward(self, obs):
        return (self.net(obs).exp() * self.support).sum(dim=2)


def load_agent(path: str) -> RainbowAgent:
    agent = RainbowAgent(player_id=1, device=torch.device('cpu'), use_pbs=False, inference_only=True)
    before = [p.detach().clone() for p in agent.q_network.parameters()]
    if path.endswith('.tar.gz'):
        Checkpointer(save_dir=os.path.dirname(path) or '.')._load_from_archive(agent, path)
    else:
        agent.load_model(path)  # prints and returns on failure
    if all(torch.equal(a, b) for a, b in zip(before, agent.q_network.parameters())):
        raise SystemExit(f"could not load the network weights from {path}")
    return agent


def export(path: str, out: str) -> str:
    agent = load_agent(path)
    net = getattr(agent.q_network, '_orig_mod', agent.q_network)  # undo torch.compile
    net.eval()
    if hasattr(net, 'set_noise'):
        net.set_noise(False)
    model = ExpectedQ(net, agent.support).eval()
    channels = agent.input_channels
    dummy = torch.zeros(1, channels, 10, 10)
    # nn.MultiheadAttention's inference fast path is a fused op that ONNX cannot express
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(model, (dummy,), out, input_names=['obs'], output_names=['q'],
                          dynamic_axes={'obs': {0: 'batch'}, 'q': {0: 'batch'}}, opset_version=17,
                          dynamo=False)
    meta = {'input_planes': channels, 'observation_planes': 30,
            'aaren_planes': channels - 30, 'actions': int(agent.action_size),
            'encoding': 'canonical-frame obs (30 planes) / index = sq*36 + dir*9 + dist-1',
            'source': os.path.basename(path)}
    with open(os.path.splitext(out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src.replace('.tar.gz', ''))[0] + '.onnx'
    print(f"exported {export(src, dst)}")
