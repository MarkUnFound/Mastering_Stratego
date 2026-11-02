"""List the contents of a .tar.gz training checkpoint: python examine_checkpoint.py <archive>"""
import os
import sys
import tarfile
import tempfile

import torch


def examine(tar_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(tmp_dir)
        state_dict = torch.load(os.path.join(tmp_dir, 'agent_state.pt'), map_location='cpu', weights_only=False)

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


if __name__ == "__main__":
    examine(sys.argv[1])
