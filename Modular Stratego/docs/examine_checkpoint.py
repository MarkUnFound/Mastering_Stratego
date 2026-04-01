import tarfile
import torch
import os

tar_path = r'c:\Users\Mark Lawrence Quibot\repo\Research\History\dqn_models\agent1_dqn_episode_37213.tar.gz'
tmp_dir = r'c:\Users\Mark Lawrence Quibot\repo\Research\tmp_extract'
os.makedirs(tmp_dir, exist_ok=True)

with tarfile.open(tar_path, 'r:gz') as t:
    t.extractall(tmp_dir)

try:
    state_dict = torch.load(os.path.join(tmp_dir, 'agent_state.pt'), map_location='cpu', weights_only=False)
except:
    state_dict = torch.load(os.path.join(tmp_dir, 'agent_state.pt'), map_location='cpu')

print("Keys in checkpoint:")
print(list(state_dict.keys()))
for k in state_dict.keys():
    v = state_dict[k]
    if isinstance(v, (int, float, str)):
        print(f"{k}: {v}")
    elif isinstance(v, list):
        print(f"{k}: list of length {len(v)}")
    elif isinstance(v, torch.Tensor):
        print(f"{k}: Tensor of shape {tuple(v.shape)}")
    else:
        print(f"{k}: {type(v).__name__}")
