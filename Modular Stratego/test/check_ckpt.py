import torch
import json
import os

path = r'c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\agent1_rainbow_episode_1000.pt'
if os.path.exists(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print("--- Checkpoint Analysis ---")
    keys = list(ckpt.keys())
    aaren_keys = [k for k in keys if 'aaren' in k.lower() or 'history' in k.lower()]
    print(f"Total keys: {len(keys)}")
    print(f"AAREN/History keys count: {len(aaren_keys)}")
    if aaren_keys:
        print("Sample AAREN keys:", aaren_keys[:10])
    elif keys and isinstance(ckpt[keys[0]], dict):
         # maybe it's nested
         print("Looks nested. Top keys:", keys)
else:
    print(f"Checkpoint not found at {path}")

metrics_path = r'c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\metrics.json'
if os.path.exists(metrics_path):
    print("\n--- Metrics Analysis ---")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        print("Metrics keys:", metrics.keys())
        if 'aaren_grad_norm' in metrics:
            data = metrics['aaren_grad_norm']
            print(f"aaren_grad_norm len: {len(data)}")
            print(f"aaren_grad_norm samples: {data[-10:]}")
            non_null = [d for d in data if d is not None]
            print(f"aaren_grad_norm non-null count: {len(non_null)}")
        if 'aaren_accuracy' in metrics:
            data = metrics['aaren_accuracy']
            print(f"aaren_accuracy len: {len(data)}")
            non_null = [d for d in data if d is not None]
            print(f"aaren_accuracy non-null count: {len(non_null)}")
