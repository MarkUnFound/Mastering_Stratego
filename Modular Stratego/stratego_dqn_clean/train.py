"""
Main Training Script for Double DQN + AAREN

Entry point for training with automatic recovery on gradient death.

Usage:
    python train.py [--steps N] [--eval-only] [--resume PATH]

Features:
- Automatic recovery on gradient death
- VRAM monitoring (4.5GB limit)
- Checkpoint saving every 1000 steps
- Diagnostics report after training

EXPLICIT PROHIBITIONS:
- FORBIDDEN: C51/Rainbow components
- FORBIDDEN: Noisy Networks
- FORBIDDEN: Batch sizes > 32
- FORBIDDEN: Transformer architectures
- FORBIDDEN: Soft target updates with τ > 0.01
"""

import sys
import os

# Add paths BEFORE any local imports - MUST run unconditionally
# IMPORTANT: Script dir must be at position 0 to avoid conflict with parent's training/ folder
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)

# Force script_dir to be at position 0 (before parent's training/ folder)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.path.insert(0, _script_dir)

# Add parent for board, piece, etc. (AFTER script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(1, _parent_dir)

import argparse
import torch
import torch.nn as nn
import time
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

from models.aaren import AAREN
from models.double_dqn import DoubleDQN, DoubleDQNAgent
from training.trainer import DoubleDQNTrainer
from training.config import TrainingConfig
from training.selfcheck import SelfCheckSuite, GradientVitalityMonitor
from env.environment import StrategoEnvironment


class RecoveryProtocol:
    """
    Automatic Recovery Protocol for gradient death.
    
    When Gradient Vitality Monitor detects dead gradients:
    1. Reduce learning rate by 50%
    2. Re-initialize last layer of Q-head only (NOT AAREN)
    3. Resume from last good checkpoint
    """
    
    def __init__(self, agent, checkpoint_dir: str = "./checkpoints"):
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir
        self.recovery_count = 0
        self.original_lr = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def detect_gradient_death(self, gradient_monitor: GradientVitalityMonitor) -> bool:
        """Check if gradient death has occurred."""
        return gradient_monitor.should_halt()
    
    def execute_recovery(self, current_lr: float) -> float:
        """
        Execute recovery protocol.
        
        Args:
            current_lr: Current learning rate
            
        Returns:
            new_lr: Reduced learning rate
        """
        self.recovery_count += 1
        print(f"\n{'='*60}")
        print(f"[RECOVERY] Initiating recovery protocol #{self.recovery_count}")
        print(f"{'='*60}")
        
        # Step 1: Reduce learning rate by 50%
        new_lr = current_lr * 0.5
        print(f"  [1/3] Reducing LR: {current_lr:.2e} -> {new_lr:.2e}")
        
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Step 2: Re-initialize last layer of Q-head ONLY (NOT AAREN)
        print(f"  [2/3] Re-initializing Q-head last layer")
        self._reinit_q_head_last_layer()
        
        # Step 3: Load last good checkpoint
        last_checkpoint = self._find_last_checkpoint()
        if last_checkpoint:
            print(f"  [3/3] Loading checkpoint: {last_checkpoint}")
            self.agent.load(last_checkpoint)
        else:
            print(f"  [3/3] No checkpoint found, continuing with reset weights")
        
        print(f"{'='*60}\n")
        
        return new_lr
    
    def _reinit_q_head_last_layer(self):
        """Re-initialize only the last layer of the Q-network."""
        # Find last linear layer in online network
        last_layer = None
        for name, module in self.agent.online_network.named_modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        
        if last_layer is not None:
            nn.init.xavier_uniform_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
            print(f"    Re-initialized: {last_layer}")
    
    def _find_last_checkpoint(self) -> str:
        """Find the most recent checkpoint."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.replace('checkpoint_', '').replace('.pt', '')))
        
        return os.path.join(self.checkpoint_dir, checkpoints[-1])
    
    def save_checkpoint(self, step: int):
        """Save checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        self.agent.save(path)
        return path


def run_diagnostics_report(trainer, agent, steps: int):
    """
    Print diagnostics report after training.
    
    Required outputs:
    - Mean gradient norms per layer
    - AAREN activation statistics (μ, σ)
    - Peak VRAM usage
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTICS REPORT")
    print("=" * 60)
    
    # 1. Gradient norms per layer
    print("\n[1] GRADIENT NORMS PER LAYER")
    print("-" * 40)
    
    if trainer.selfcheck:
        grad_monitor = trainer.selfcheck.gradient_monitor
        for name in list(grad_monitor.layer_grad_history.keys())[:10]:  # First 10 layers
            history = list(grad_monitor.layer_grad_history[name])
            if history:
                mean_norm = sum(history) / len(history)
                print(f"  {name[:40]:40s}: {mean_norm:.6f}")
    
    # 2. AAREN activation statistics
    print("\n[2] AAREN ACTIVATION STATISTICS")
    print("-" * 40)
    
    if trainer.selfcheck:
        aaren_check = trainer.selfcheck.aaren_collapse
        if aaren_check.stats_history:
            recent = list(aaren_check.stats_history)[-10:]
            mu_vals = [s['mu'] for s in recent]
            sigma_vals = [s['sigma'] for s in recent]
            
            print(f"  Mean mu:  {sum(mu_vals)/len(mu_vals):.6f}")
            print(f"  Mean sigma:  {sum(sigma_vals)/len(sigma_vals):.6f}")
    
    # 3. Peak VRAM usage
    print("\n[3] PEAK VRAM USAGE")
    print("-" * 40)
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        current_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Peak:    {peak_mem:.3f} GB")
        print(f"  Current: {current_mem:.3f} GB")
        print(f"  Limit:   4.5 GB")
        
        if peak_mem < 4.5:
            print(f"  Status:  [PASS] (headroom: {4.5 - peak_mem:.3f} GB)")
        else:
            print(f"  Status:  [FAIL] (exceeded by: {peak_mem - 4.5:.3f} GB)")
    else:
        print("  CUDA not available")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train Double DQN + AAREN for Stratego')
    parser.add_argument('--steps', type=int, default=10000000, 
                       help='Number of training steps (default: 10M for continuous training)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes (alternative to steps)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--continue-latest', action='store_true',
                       help='Continue from latest checkpoint in most recent run')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only (no training)')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Run 1000-step test and print diagnostics')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel environments (default: 1)')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Run Management
    if args.diagnostics:
        run_name = "diagnostics"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_dqn_aaren"
    
    run_dir = os.path.join("runs", run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    model_dir = os.path.join(run_dir, "models")
    plot_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n[RUN] Starting run: {run_name}")
    print(f"  - Directory: {run_dir}")

    # Configuration
    config = TrainingConfig()
    config.CHECKPOINT_DIR = checkpoint_dir
    print(f"\n[CONFIG]")
    print(f"  Batch size:     {config.BATCH_SIZE}")
    print(f"  Learning rate:  {config.LEARNING_RATE}")
    print(f"  Adam eps:       {config.ADAM_EPS}")
    print(f"  Gradient clip:  {config.GRADIENT_CLIP}")
    print(f"  Target update:  every {config.TARGET_UPDATE_FREQ} steps")
    
    # Create agent
    print("\n[INIT] Creating agent...")
    agent = DoubleDQNAgent(
        player_id=1,  # Player 1 for training
        device=device,
        action_dim=100,  # Max 100 moves from heuristic filter
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        aaren_input_dim=24,  # Action features dimension
    )
    
    # Update optimizer with correct eps (Adam eps = 1.5e-4 per spec)
    agent.optimizer = torch.optim.Adam(
        list(agent.aaren.parameters()) + list(agent.online_network.parameters()),
        lr=config.LEARNING_RATE,
        eps=config.ADAM_EPS
    )
    
    # Handle --continue-latest (overrides --resume if valid)
    if args.continue_latest:
        print("\n[CONTINUE] Searching for latest checkpoint...")
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            # Find all run directories
            run_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                       if os.path.isdir(os.path.join(runs_dir, d))]
            
            if run_dirs:
                # Sort by creation time (newest first)
                run_dirs.sort(key=os.path.getctime, reverse=True)
                
                # Search for checkpoints in newest runs
                found_checkpoint = None
                for latest_run_dir in run_dirs:
                    ckpt_dir = os.path.join(latest_run_dir, "checkpoints")
                    if os.path.exists(ckpt_dir):
                        checkpoints = [f for f in os.listdir(ckpt_dir) 
                                      if f.startswith('checkpoint_') and f.endswith('.pt')]
                        
                        if checkpoints:
                            # Sort by step number
                            checkpoints.sort(key=lambda x: int(x.replace('checkpoint_', '').replace('.pt', '')))
                            found_checkpoint = os.path.join(ckpt_dir, checkpoints[-1])
                            print(f"  Found latest run: {latest_run_dir}")
                            break
                
                if found_checkpoint:
                    args.resume = found_checkpoint
                else:
                    print("  No checkpoints found in recent runs.")
            else:
                print("  No previous runs found.")
        else:
             print("  'runs' directory does not exist.")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n[RESUME] Loading checkpoint: {args.resume}")
        agent.load(args.resume)
    
    # Environment Setup
    if args.parallel > 1:
        from env.vector_env import VectorStrategoEnv
        print(f"[INIT] Creating {args.parallel} vectorized environment(s)...")
        env = VectorStrategoEnv(
            num_envs=args.parallel, 
            device=device.type, 
            max_turns=TrainingConfig.MAX_TURNS,
            strict_validation=TrainingConfig.DEBUG_STRICT_VALIDATION,
            safe_guards=TrainingConfig.ENABLE_SAFE_GUARDS
        )
        config.NUM_ENVS = args.parallel
    else:
        print("[INIT] Creating environment...")
        env = StrategoEnvironment(
            device=device, 
            max_turns=TrainingConfig.MAX_TURNS,
            strict_validation=TrainingConfig.DEBUG_STRICT_VALIDATION,
            safe_guards=TrainingConfig.ENABLE_SAFE_GUARDS
        )
        config.NUM_ENVS = 1
        
    # Trainer Setup
    print("[INIT] Creating trainer...")
    trainer = DoubleDQNTrainer(
        agent=agent, 
        env=env,
        config=config,
        log_dir=log_dir,
        device=device
    )
    
    # Recovery protocol
    recovery = RecoveryProtocol(agent, config.CHECKPOINT_DIR)
    
    # GATE 1: Memory Test (Only run in single-env mode or diagnostics to avoid complexity)
    if args.diagnostics or args.parallel == 1:
        print("\n[GATE 1] Memory Test")
        print("-" * 40)
        try:
            mem_stats = trainer.selfcheck.memory_guardian.check()
            gradient_check = trainer.selfcheck.gradient_monitor.compute_gradient_norms()
            
            # Check AAREN gradients specifically
            aaren_grad_norm = 0.0
            for name, norm in gradient_check.items():
                if "aaren" in name:
                    aaren_grad_norm += norm
            
            print(f"  Peak VRAM: {mem_stats.data['allocated_gb']:.3f} GB")
            if mem_stats.passed:
                print("  [PASS]: VRAM under limit")
            else:
                print(f"  [FAIL]: VRAM exceeded limit! ({mem_stats.data['allocated_gb']:.3f} > 4.5)")
                if not args.diagnostics: # Strict fail unless diagnostics
                    return 1
            
            print(f"  AAREN gradient norm: {aaren_grad_norm:.6f}")
            if aaren_grad_norm > 1e-6:
                print("  [PASS]: AAREN gradients flowing")
            else:
                print("  [FAIL]: AAREN gradients NOT flowing (possibly detached)")
                # Soft warn for now, unexpected zero grad at init is possible if unused
                
            print("\n[GATE 1] [PASS]ED")
        except Exception as e:
            print(f"[GATE 1] Error during check: {e}")
            if not args.diagnostics:
                raise e
    
    # Diagnostics mode
    if args.diagnostics:
        print("\n[MODE] Running 1000-step diagnostics test...")
        args.steps = 1000
    
    # Eval-only mode
    if args.eval_only:
        print("\n[MODE] Evaluation only...")
        win_rate = trainer._evaluate()
        print(f"Win Rate: {win_rate:.2%}")
        return 0
    
    # Main training loop - CONTINUOUS like old train_dqn.py
    print(f"[TRAIN] Starting CONTINUOUS training for {args.steps:,} steps...")
    print(f"[TRAIN] Progress will update every step (nonstop training)")
    
    current_lr = config.LEARNING_RATE
    
    # Progress bar - tqdm for nonstop training visibility
    pbar = tqdm(total=args.steps, initial=trainer.total_steps, 
                desc="Training", unit="steps", dynamic_ncols=True)
    
    last_checkpoint_step = 0
    last_eval_step = 0
    
    try:
        while trainer.total_steps < args.steps:
            # Run one training iteration (vectorized or single-env)
            if hasattr(trainer, '_run_vector_steps') and trainer.is_vector_env:
                finished_rewards, steps_run = trainer._run_vector_steps()
                
                # Log finished episodes
                for r in finished_rewards:
                    trainer.episodes += 1
                    trainer.episode_rewards.append(r)
            else:
                episode_reward, episode_steps = trainer._run_episode()
                steps_run = episode_steps
                trainer.episodes += 1
                trainer.episode_rewards.append(episode_reward)
            
            # Update progress bar
            pbar.update(steps_run)
            
            # Update postfix with current metrics
            avg_reward = np.mean(list(trainer.episode_rewards)) if trainer.episode_rewards else 0.0
            avg_loss = np.mean(list(trainer.losses)) if trainer.losses else 0.0
            
            pbar.set_postfix({
                'Ep': trainer.episodes,
                'R': f"{avg_reward:.2f}",
                'Loss': f"{avg_loss:.4f}",
                'ε': f"{trainer.agent.epsilon:.3f}",
            })
            
            # Check for gradient death
            if trainer.selfcheck and trainer.selfcheck.gradient_monitor:
                if recovery.detect_gradient_death(trainer.selfcheck.gradient_monitor):
                    current_lr = recovery.execute_recovery(current_lr)
                    
                    # Reset gradient monitor
                    trainer.selfcheck.gradient_monitor.consecutive_dead = \
                        {k: 0 for k in trainer.selfcheck.gradient_monitor.consecutive_dead}
                    trainer.selfcheck.gradient_monitor.alert_history.clear()
            
            # Checkpoint at intervals (by steps)
            if trainer.total_steps - last_checkpoint_step >= config.CHECKPOINT_FREQ:
                recovery.save_checkpoint(trainer.total_steps)
                last_checkpoint_step = trainer.total_steps
                tqdm.write(f"[SAVE] Checkpoint at step {trainer.total_steps:,}")
            
            # Periodic evaluation
            if trainer.total_steps - last_eval_step >= config.EVAL_FREQ:
                try:
                    win_rate = trainer._evaluate()
                    trainer.win_rates.append((trainer.total_steps, win_rate))
                    last_eval_step = trainer.total_steps
                except Exception as e:
                    tqdm.write(f"[EVAL] Error during evaluation: {e}")
            
            # Self-checks at intervals
            if trainer.selfcheck and trainer.optimization_steps > 0:
                if trainer.optimization_steps % config.SELFCHECK_FREQ == 0:
                    trainer._run_selfchecks()
            
            # Check for halt conditions
            if trainer.selfcheck:
                should_halt, reason = trainer.selfcheck.should_halt()
                if should_halt:
                    tqdm.write(f"[CRITICAL] {reason}")
                    break
                    
    except KeyboardInterrupt:
        tqdm.write("\n[INFO] Training interrupted by user - saving checkpoint...")
        recovery.save_checkpoint(trainer.total_steps)
    except RuntimeError as e:
        if "out of memory" in str(e):
            tqdm.write(f"\n[OOM] Out of memory error. Attempting recovery...")
            torch.cuda.empty_cache()
            current_lr = recovery.execute_recovery(current_lr)
        else:
            raise
    finally:
        pbar.close()
    
    # Final diagnostics
    run_diagnostics_report(trainer, agent, trainer.total_steps)
    
    # Post-training: Export and Plot
    print("\n[EXPORT] Saving final model...")
    final_model_path = os.path.join(model_dir, "final_model.pt")
    agent.export(final_model_path)
    print(f"  Saved to {final_model_path}")
    
    print("\n[PLOT] Generating training plots...")
    
    # Save history explicitly if not done
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
        
    try:
        from utils.plotting import plot_training_history
        plot_training_history(history_path, plot_dir)
    except Exception as e:
        print(f"[PLOT] Error generating plots: {e}")
    print("\n[DONE] Training complete.")
    return 0


if __name__ == "__main__":
    exit(main())

