"""
Root-level entry point for Stratego Rainbow DQN Training.
Dispatches execution to 'Modular Stratego/train_dqn.py'.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    modular_dir = os.path.join(repo_root, "Modular Stratego")
    target_script = os.path.join(modular_dir, "train_dqn.py")

    if not os.path.isfile(target_script):
        print(f"[ERROR] Could not find training script at: {target_script}")
        sys.exit(1)

    cmd = [sys.executable, target_script] + sys.argv[1:]
    try:
        sys.exit(subprocess.call(cmd, cwd=modular_dir))
    except KeyboardInterrupt:
        print("\n[INFO] Terminated by user.")
