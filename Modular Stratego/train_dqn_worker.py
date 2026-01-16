"""
DQN Training Worker for Population-Based Training (PBT)

This worker script wraps train_dqn.py to enable PBT integration:
- Accepts hyperparameters via command-line arguments
- Can initialize from pre-trained weights (for cloning)
- Reports metrics to a shared CSV file for supervisor monitoring

Launch via pbt_supervisor.py or directly for testing:
    python train_dqn_worker.py --worker_id 0 --seed 12345 --model_dir dqn_models/worker_0
"""

import os
import sys
import csv
import time
import argparse
import signal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random


def set_seeds(seed: int):
    """Set all random seeds for reproducibility and diversity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PBTMetricsReporter:
    """Reports training metrics to shared CSV for PBT supervisor monitoring."""
    
    def __init__(self, metrics_file: str, worker_id: int):
        self.metrics_file = metrics_file
        self.worker_id = worker_id
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create metrics file with header if it doesn't exist."""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'worker_id', 'episode', 'reward', 'win', 
                    'win_rate', 'avg_loss', 'timestamp'
                ])
    
    def report(self, episode: int, reward: float, win: int, 
               win_rate: float = 0.0, avg_loss: float = 0.0):
        """Append a metrics row to the shared CSV file."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.worker_id, episode, reward, win,
                win_rate, avg_loss, time.time()
            ])


def parse_args():
    """Parse command-line arguments for worker configuration."""
    parser = argparse.ArgumentParser(
        description='DQN Training Worker for PBT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Worker identity
    parser.add_argument('--worker_id', type=int, required=True,
                        help='Unique worker identifier')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    
    # Directories
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory to save model checkpoints')
    parser.add_argument('--metrics_file', type=str, default='pbt_metrics.csv',
                        help='Shared CSV file for metrics reporting')
    
    # Initialization
    parser.add_argument('--init_weights', type=str, default=None,
                        help='Path to initial weights (for cloning)')
    
    # Hyperparameters (can be perturbed by PBT supervisor)
    parser.add_argument('--learning_rate', type=float, default=0.00003,
                        help='Learning rate for optimizer')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995,
                        help='Epsilon decay rate per episode')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor')
    
    # Training control
    parser.add_argument('--num_episodes', type=int, default=1000000,
                        help='Maximum episodes to train')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Episodes between checkpoint saves')
    
    return parser.parse_args()


def run_worker(args):
    """
    Main worker training loop with PBT integration.
    
    This function imports and runs the training logic from train_dqn.py,
    but with custom hyperparameters and metrics reporting.
    """
    print(f"[Worker {args.worker_id}] Starting with seed {args.seed}")
    print(f"[Worker {args.worker_id}] Model directory: {args.model_dir}")
    print(f"[Worker {args.worker_id}] Learning rate: {args.learning_rate}")
    print(f"[Worker {args.worker_id}] Epsilon: {args.epsilon_start} -> {args.epsilon_end}")
    
    # Set random seeds for this worker
    set_seeds(args.seed)
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize metrics reporter
    reporter = PBTMetricsReporter(args.metrics_file, args.worker_id)
    
    # Override training_config values with worker-specific hyperparameters
    import training_config
    training_config.LEARNING_RATE = args.learning_rate
    training_config.BATCH_SIZE = args.batch_size
    training_config.GAMMA = args.gamma
    training_config.EXPLORATION_EPSILON_START = args.epsilon_start
    training_config.EXPLORATION_EPSILON_END = args.epsilon_end
    training_config.EXPLORATION_EPSILON_DECAY = args.epsilon_decay
    
    # Import training modules after config override
    from train_dqn import train_dqn_agents
    from preflight_checks import run_preflight_checks
    
    # Run preflight checks
    if not run_preflight_checks(model_save_path=args.model_dir):
        print(f"[Worker {args.worker_id}] Pre-flight checks failed!")
        return
    
    # Start training with metrics callback
    train_dqn_agents(
        num_episodes=args.num_episodes,
        save_interval=args.save_interval,
        model_save_path=args.model_dir,
        generate_gifs=False,  # Disable GIFs for workers
        # Pass PBT-specific parameters
        init_weights=args.init_weights,
        pbt_reporter=reporter,
    )


def main():
    """Entry point for the worker script."""
    args = parse_args()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[Worker {args.worker_id}] Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_worker(args)
    except Exception as e:
        print(f"[Worker {args.worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
