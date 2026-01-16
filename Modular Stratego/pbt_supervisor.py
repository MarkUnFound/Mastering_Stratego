"""
Population-Based Training (PBT) Supervisor for DQN Agents

This supervisor manages multiple independent training workers, periodically
evaluating their performance and applying exploitation/exploration strategies
to accelerate learning. Workers with poor performance are terminated and
replaced with clones of top performers, but with perturbed hyperparameters.

Theory:
PBT combines the advantages of parallel random search (diversity) with online
model selection (exploitation). Unlike fixed hyperparameter schedules, PBT
adapts hyperparameters during training by copying from successful workers.
This is particularly effective for RL where optimal hyperparameters change
as the policy improves (e.g., lower epsilon as Q-values stabilize).

Usage:
    python pbt_supervisor.py --num_workers 4 --exploit_interval 3600

Architecture:
    Supervisor (this script)
        └── Worker 0 (subprocess running train_dqn_worker.py --seed 0)
        └── Worker 1 (subprocess running train_dqn_worker.py --seed 1)
        └── Worker 2 (subprocess running train_dqn_worker.py --seed 2)
        └── Worker 3 (subprocess running train_dqn_worker.py --seed 3)
"""

import os
import sys
import json
import time
import glob
import shutil
import signal
import argparse
import subprocess
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PBT] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PBTConfig:
    """Configuration for PBT supervisor."""
    # Worker settings
    num_workers: int = 4            # Number of parallel workers (4-8 recommended)
    worker_script: str = "train_dqn_worker.py"  # Script each worker runs
    base_model_dir: str = "dqn_models"  # Base directory for models
    
    # Exploitation settings
    exploit_interval_seconds: int = 3600  # 1 hour between exploitation rounds
    exploit_fraction: float = 0.5       # Bottom 50% are replaced
    min_episodes_before_exploit: int = 100  # Minimum episodes before first exploit
    
    # Adaptive culling triggers
    plateau_episodes: int = 500         # Cull if no improvement for N episodes
    plateau_threshold: float = 0.01     # Minimum improvement to reset plateau counter
    leader_gap_threshold: float = 0.3   # Cull if worker falls X% behind leader (0.3 = 30%)
    adaptive_culling_enabled: bool = True  # Enable plateau/gap-based culling
    
    # Exploration (hyperparameter perturbation) settings
    perturb_factors: Tuple[float, float] = (0.8, 1.2)  # Multiply hyperparams
    reset_epsilon: float = 0.1          # Reset exploration rate for cloned workers
    epsilon_decay_perturbation: float = 0.2  # ±20% perturbation on epsilon decay
    learning_rate_perturbation: float = 0.2  # ±20% perturbation on LR
    
    # Monitoring settings
    metrics_file: str = "pbt_metrics.csv"  # Shared metrics file
    log_dir: str = "logs"               # Directory for worker logs
    metric_key: str = "mean_reward"     # Metric to optimize ("mean_reward" or "win_rate")
    metric_window: int = 100            # Episodes to average for metric calculation
    
    # Robustness
    worker_timeout_seconds: int = 7200  # Kill worker if no update in 2 hours
    max_restarts_per_worker: int = 3     # Max restarts before giving up on a worker


@dataclass
class WorkerState:
    """Tracks state of a single worker."""
    worker_id: int
    seed: int
    process: Optional[subprocess.Popen] = None
    start_time: float = 0.0
    last_update_time: float = 0.0
    total_episodes: int = 0
    current_metric: float = float('-inf')
    hyperparams: Dict[str, float] = None
    restart_count: int = 0
    model_dir: str = ""
    status: str = "idle"  # idle, running, terminated, failed
    
    # Adaptive culling tracking
    best_metric: float = float('-inf')  # Best metric ever achieved
    episodes_since_improvement: int = 0  # Episodes since last improvement
    metric_history: List[float] = None   # Recent metric values for trend detection
    
    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = {}
        if self.metric_history is None:
            self.metric_history = []


# =============================================================================
# WORKER MANAGEMENT
# =============================================================================

class PBTSupervisor:
    """
    Manages a population of training workers using Population-Based Training.
    
    The supervisor:
    1. Launches workers with unique random seeds
    2. Monitors their performance via shared metrics
    3. Periodically culls bottom performers
    4. Clones top performers with perturbed hyperparameters
    """
    
    def __init__(self, config: PBTConfig):
        self.config = config
        self.workers: Dict[int, WorkerState] = {}
        self.running = False
        self.exploit_count = 0
        self.start_time = time.time()
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.base_model_dir, exist_ok=True)
        
        # Initialize PBT dashboard for aggregate visualization
        try:
            from pbt_visualizer import PBTDashboard
            self.dashboard = PBTDashboard(
                metrics_file=config.metrics_file,
                output_dir=config.base_model_dir,
                update_interval=100
            )
            logger.info("PBT Dashboard initialized")
        except ImportError:
            self.dashboard = None
            logger.warning("PBT Dashboard not available (pbt_visualizer.py not found)")
        
        # Exploitation events tracking (for visualization markers)
        self.exploitation_events_file = os.path.join(config.base_model_dir, 'exploitation_events.json')
        self._init_exploitation_events_file()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"PBT Supervisor initialized with {config.num_workers} workers")
        logger.info(f"Exploitation interval: {config.exploit_interval_seconds}s")
        logger.info(f"Culling fraction: {config.exploit_fraction * 100}%")
    
    def _init_exploitation_events_file(self):
        """Initialize the exploitation events JSON file if it doesn't exist."""
        if not os.path.exists(self.exploitation_events_file):
            with open(self.exploitation_events_file, 'w') as f:
                json.dump({"events": []}, f, indent=2)
    
    def _load_exploitation_events(self) -> dict:
        """Load exploitation events from JSON file."""
        if os.path.exists(self.exploitation_events_file):
            try:
                with open(self.exploitation_events_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"events": []}
        return {"events": []}
    
    def _save_exploitation_events(self, data: dict):
        """Save exploitation events to JSON file."""
        with open(self.exploitation_events_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_exploitation_event(self, event_type: str, worker_id: int, 
                                source_worker: int = -1, reason: str = "scheduled"):
        """
        Log an exploitation event to JSON file for visualization.
        
        Args:
            event_type: "cull", "clone", or "restart"
            worker_id: Worker that was affected
            source_worker: Worker that was cloned from (-1 if N/A)
            reason: "scheduled", "adaptive", "plateau", or "leader_gap"
        """
        episode_count = self.workers[worker_id].total_episodes if worker_id in self.workers else 0
        
        event = {
            "timestamp": time.time(),
            "exploit_round": self.exploit_count,
            "event_type": event_type,
            "worker_id": worker_id,
            "source_worker": source_worker,
            "episode_count": episode_count,
            "reason": reason
        }
        
        data = self._load_exploitation_events()
        data["events"].append(event)
        self._save_exploitation_events(data)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, terminating workers...")
        self.stop()
        sys.exit(0)
    
    # =========================================================================
    # WORKER LIFECYCLE
    # =========================================================================
    
    def launch_worker(self, worker_id: int, seed: int, 
                      clone_from: Optional[str] = None,
                      hyperparams: Optional[Dict[str, float]] = None) -> WorkerState:
        """
        Launch a new training worker as a subprocess.
        
        Args:
            worker_id: Unique identifier for this worker
            seed: Random seed for reproducibility
            clone_from: Path to model weights to initialize from (optional)
            hyperparams: Hyperparameters to override (optional)
        
        Returns:
            WorkerState object tracking this worker
        """
        worker_dir = os.path.join(self.config.base_model_dir, f"worker_{worker_id}")
        log_file = os.path.join(self.config.log_dir, f"worker_{worker_id}.log")
        
        os.makedirs(worker_dir, exist_ok=True)
        
        # Build command line arguments
        cmd = [
            sys.executable,
            self.config.worker_script,
            "--seed", str(seed),
            "--worker_id", str(worker_id),
            "--model_dir", worker_dir,
            "--metrics_file", self.config.metrics_file,
        ]
        
        if clone_from:
            cmd.extend(["--init_weights", clone_from])
        
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                "learning_rate": 0.00003,
                "epsilon_start": 0.1 if clone_from else 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.9995,
            }
        
        # Add hyperparameters to command
        for key, value in hyperparams.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Launching worker {worker_id} with seed {seed}")
        if clone_from:
            logger.info(f"  Cloning weights from: {clone_from}")
        
        # Launch subprocess
        with open(log_file, 'a') as log_fh:
            log_fh.write(f"\n{'='*60}\n")
            log_fh.write(f"Worker {worker_id} started at {datetime.now()}\n")
            log_fh.write(f"Command: {' '.join(cmd)}\n")
            log_fh.write(f"{'='*60}\n\n")
            log_fh.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        
        state = WorkerState(
            worker_id=worker_id,
            seed=seed,
            process=process,
            start_time=time.time(),
            last_update_time=time.time(),
            hyperparams=hyperparams,
            model_dir=worker_dir,
            status="running",
        )
        
        self.workers[worker_id] = state
        return state
    
    def terminate_worker(self, worker_id: int, reason: str = "exploitation"):
        """Terminate a worker process."""
        if worker_id not in self.workers:
            return
        
        state = self.workers[worker_id]
        if state.process and state.process.poll() is None:
            logger.info(f"Terminating worker {worker_id} ({reason})")
            state.process.terminate()
            try:
                state.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state.process.kill()
        
        state.status = "terminated"
        state.process = None
    
    def check_worker_health(self, worker_id: int) -> bool:
        """
        Check if a worker is healthy (running and producing updates).
        
        Returns:
            True if worker is healthy, False if it needs restart
        """
        if worker_id not in self.workers:
            return False
        
        state = self.workers[worker_id]
        
        # Check if process is running
        if state.process is None or state.process.poll() is not None:
            logger.warning(f"Worker {worker_id} process is not running")
            state.status = "failed"
            return False
        
        # Check for timeout (no updates)
        time_since_update = time.time() - state.last_update_time
        if time_since_update > self.config.worker_timeout_seconds:
            logger.warning(f"Worker {worker_id} timed out (no update in {time_since_update:.0f}s)")
            return False
        
        return True
    
    def restart_failed_workers(self):
        """Restart any workers that have failed or timed out."""
        for worker_id, state in list(self.workers.items()):
            if not self.check_worker_health(worker_id):
                if state.restart_count < self.config.max_restarts_per_worker:
                    logger.info(f"Restarting worker {worker_id} (attempt {state.restart_count + 1})")
                    self.terminate_worker(worker_id, reason="restart")
                    
                    # Find latest checkpoint to resume from
                    checkpoint = self._find_latest_checkpoint(state.model_dir)
                    
                    state.restart_count += 1
                    self.launch_worker(
                        worker_id=worker_id,
                        seed=state.seed,
                        clone_from=checkpoint,
                        hyperparams=state.hyperparams,
                    )
                else:
                    logger.error(f"Worker {worker_id} exceeded max restarts, giving up")
                    state.status = "failed"
    
    def _find_latest_checkpoint(self, model_dir: str) -> Optional[str]:
        """Find the latest model checkpoint in a directory."""
        pattern = os.path.join(model_dir, "agent1_rainbow_episode_*.pth")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        
        # Sort by episode number
        def extract_episode(path):
            try:
                return int(os.path.basename(path).split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                return -1
        
        checkpoints.sort(key=extract_episode, reverse=True)
        return checkpoints[0]
    
    # =========================================================================
    # METRICS MONITORING
    # =========================================================================
    
    def read_worker_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Read performance metrics from all workers.
        
        Returns:
            Dictionary mapping worker_id to their latest metrics
        """
        metrics = {}
        metrics_path = self.config.metrics_file
        
        if not os.path.exists(metrics_path):
            return metrics
        
        try:
            with open(metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    worker_id = int(row.get('worker_id', -1))
                    if worker_id >= 0:
                        # Parse and aggregate metrics
                        if worker_id not in metrics:
                            metrics[worker_id] = {
                                'episodes': [],
                                'rewards': [],
                                'wins': [],
                                'timestamps': [],
                            }
                        
                        metrics[worker_id]['episodes'].append(int(row.get('episode', 0)))
                        metrics[worker_id]['rewards'].append(float(row.get('reward', 0)))
                        metrics[worker_id]['wins'].append(int(row.get('win', 0)))
                        metrics[worker_id]['timestamps'].append(float(row.get('timestamp', 0)))
        
        except Exception as e:
            logger.error(f"Error reading metrics file: {e}")
        
        return metrics
    
    def calculate_worker_performance(self) -> Dict[int, float]:
        """
        Calculate performance score for each worker based on configured metric.
        
        Returns:
            Dictionary mapping worker_id to performance score
        """
        raw_metrics = self.read_worker_metrics()
        performance = {}
        
        for worker_id, worker_metrics in raw_metrics.items():
            if worker_id not in self.workers:
                continue
            
            # Use recent window of episodes
            rewards = worker_metrics['rewards'][-self.config.metric_window:]
            wins = worker_metrics['wins'][-self.config.metric_window:]
            
            if not rewards:
                performance[worker_id] = float('-inf')
                continue
            
            if self.config.metric_key == "mean_reward":
                performance[worker_id] = sum(rewards) / len(rewards)
            elif self.config.metric_key == "win_rate":
                performance[worker_id] = sum(wins) / len(wins) if wins else 0.0
            else:
                performance[worker_id] = sum(rewards) / len(rewards)
            
            # Update worker state
            worker_state = self.workers[worker_id]
            prev_episodes = worker_state.total_episodes
            new_episodes = len(worker_metrics['episodes'])
            episodes_delta = new_episodes - prev_episodes
            
            worker_state.current_metric = performance[worker_id]
            worker_state.total_episodes = new_episodes
            if worker_metrics['timestamps']:
                worker_state.last_update_time = max(worker_metrics['timestamps'])
            
            # Track metric history for plateau detection
            worker_state.metric_history.append(performance[worker_id])
            if len(worker_state.metric_history) > 20:  # Keep last 20 measurements
                worker_state.metric_history = worker_state.metric_history[-20:]
            
            # Check for improvement (plateau detection)
            if performance[worker_id] > worker_state.best_metric + self.config.plateau_threshold:
                worker_state.best_metric = performance[worker_id]
                worker_state.episodes_since_improvement = 0
            else:
                worker_state.episodes_since_improvement += episodes_delta
        
        return performance
    
    def detect_workers_to_cull(self, performance: Dict[int, float]) -> List[int]:
        """
        Detect workers that should be culled based on adaptive triggers.
        
        Triggers:
        1. Plateau: No improvement for N episodes
        2. Leader gap: Falling X% behind the leader
        
        Returns:
            List of worker IDs that should be culled
        """
        if not self.config.adaptive_culling_enabled or not performance:
            return []
        
        workers_to_cull = []
        leader_metric = max(performance.values())
        
        for worker_id, metric in performance.items():
            if worker_id not in self.workers:
                continue
            
            worker_state = self.workers[worker_id]
            cull_reason = None
            
            # Check plateau: no improvement for too many episodes
            if worker_state.episodes_since_improvement >= self.config.plateau_episodes:
                cull_reason = f"plateau ({worker_state.episodes_since_improvement} episodes without improvement)"
            
            # Check leader gap: falling too far behind
            if leader_metric > 0:  # Only check if leader has positive metric
                gap = (leader_metric - metric) / abs(leader_metric)
                if gap >= self.config.leader_gap_threshold:
                    cull_reason = f"leader gap ({gap*100:.1f}% behind)"
            
            if cull_reason:
                logger.info(f"Worker {worker_id} flagged for adaptive culling: {cull_reason}")
                workers_to_cull.append(worker_id)
        
        return workers_to_cull
    
    def _cull_workers(self, workers_to_cull: List[int], performance: Dict[int, float]):
        """
        Cull specific workers and clone from top performers.
        
        This is the shared culling logic used by both scheduled exploitation
        and adaptive culling triggers.
        
        Args:
            workers_to_cull: List of worker IDs to terminate and replace
            performance: Current performance scores for all workers
        """
        if not workers_to_cull or not performance:
            return
        
        # Find top performers to clone from (exclude workers being culled)
        top_workers = sorted(
            [(wid, score) for wid, score in performance.items() if wid not in workers_to_cull],
            key=lambda x: x[1],
            reverse=True
        )
        
        if not top_workers:
            logger.warning("No healthy workers to clone from")
            return
        
        self.exploit_count += 1
        
        for i, bottom_wid in enumerate(workers_to_cull):
            # Select a top performer to clone from (round-robin)
            source_wid = top_workers[i % len(top_workers)][0]
            source_state = self.workers[source_wid]
            
            # Find source model checkpoint
            source_checkpoint = self._find_latest_checkpoint(source_state.model_dir)
            if not source_checkpoint:
                logger.warning(f"No checkpoint found for source worker {source_wid}")
                continue
            
            # Perturb hyperparameters
            new_hyperparams = self._perturb_hyperparams(source_state.hyperparams)
            
            logger.info(f"Cloning worker {source_wid} -> worker {bottom_wid}")
            
            # Log exploitation event for visualization
            self.log_exploitation_event(
                event_type="cull", 
                worker_id=bottom_wid,
                source_worker=source_wid,
                reason="adaptive"
            )
            
            # Terminate bottom worker
            self.terminate_worker(bottom_wid, reason="adaptive_culling")
            
            # Copy model weights to new worker's directory
            bottom_state = self.workers[bottom_wid]
            target_checkpoint = os.path.join(
                bottom_state.model_dir,
                f"agent1_rainbow_cloned.pth"
            )
            shutil.copy2(source_checkpoint, target_checkpoint)
            
            # Reset adaptive tracking for new worker
            bottom_state.best_metric = float('-inf')
            bottom_state.episodes_since_improvement = 0
            bottom_state.metric_history = []
            
            # Launch new worker
            self.launch_worker(
                worker_id=bottom_wid,
                seed=bottom_state.seed + self.exploit_count * 1000,
                clone_from=target_checkpoint,
                hyperparams=new_hyperparams,
            )
            
            # Log clone event
            self.log_exploitation_event(
                event_type="clone",
                worker_id=bottom_wid,
                source_worker=source_wid,
                reason="adaptive"
            )
    
    # =========================================================================
    # EXPLOITATION (CULLING + CLONING)
    # =========================================================================
    
    def exploit(self):
        """
        Perform PBT exploitation: cull bottom performers and clone top performers.
        
        This is the core PBT mechanism. Bottom workers are terminated and
        restarted with weights cloned from top performers, but with slightly
        perturbed hyperparameters to encourage exploration.
        """
        self.exploit_count += 1
        logger.info(f"{'='*60}")
        logger.info(f"EXPLOITATION ROUND {self.exploit_count}")
        logger.info(f"{'='*60}")
        
        # Calculate current performance
        performance = self.calculate_worker_performance()
        
        if len(performance) < 2:
            logger.warning("Not enough workers with metrics for exploitation")
            return
        
        # Check minimum episodes requirement
        min_episodes = min(
            self.workers[wid].total_episodes 
            for wid in performance.keys() 
            if wid in self.workers
        )
        
        if min_episodes < self.config.min_episodes_before_exploit:
            logger.info(f"Skipping exploitation: min episodes {min_episodes} < {self.config.min_episodes_before_exploit}")
            return
        
        # Rank workers by performance
        ranked = sorted(performance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Current worker rankings:")
        for rank, (worker_id, score) in enumerate(ranked, 1):
            status = "TOP" if rank <= len(ranked) // 2 else "BOTTOM"
            logger.info(f"  {rank}. Worker {worker_id}: {score:.4f} ({status})")
        
        # Determine cutoff
        num_to_cull = int(len(ranked) * self.config.exploit_fraction)
        if num_to_cull == 0:
            num_to_cull = 1  # Always cull at least one
        
        top_workers = [wid for wid, _ in ranked[:len(ranked) - num_to_cull]]
        bottom_workers = [wid for wid, _ in ranked[-num_to_cull:]]
        
        logger.info(f"Culling {len(bottom_workers)} workers: {bottom_workers}")
        logger.info(f"Top performers: {top_workers}")
        
        # Clone and restart
        for i, bottom_wid in enumerate(bottom_workers):
            # Select a top performer to clone from (round-robin)
            source_wid = top_workers[i % len(top_workers)]
            source_state = self.workers[source_wid]
            
            # Find source model checkpoint
            source_checkpoint = self._find_latest_checkpoint(source_state.model_dir)
            if not source_checkpoint:
                logger.warning(f"No checkpoint found for source worker {source_wid}")
                continue
            
            # Perturb hyperparameters
            new_hyperparams = self._perturb_hyperparams(source_state.hyperparams)
            
            logger.info(f"Cloning worker {source_wid} -> worker {bottom_wid}")
            logger.info(f"  New hyperparams: {new_hyperparams}")
            
            # Log exploitation event for visualization
            self.log_exploitation_event(
                event_type="cull",
                worker_id=bottom_wid,
                source_worker=source_wid,
                reason="scheduled"
            )
            
            # Terminate bottom worker
            self.terminate_worker(bottom_wid, reason="exploitation")
            
            # Copy model weights to new worker's directory
            bottom_state = self.workers[bottom_wid]
            target_checkpoint = os.path.join(
                bottom_state.model_dir,
                f"agent1_rainbow_cloned.pth"
            )
            shutil.copy2(source_checkpoint, target_checkpoint)
            
            # Launch new worker
            self.launch_worker(
                worker_id=bottom_wid,
                seed=bottom_state.seed + self.exploit_count * 1000,  # New seed
                clone_from=target_checkpoint,
                hyperparams=new_hyperparams,
            )
        
        logger.info(f"Exploitation round {self.exploit_count} complete")
    
    def _perturb_hyperparams(self, source_params: Dict[str, float]) -> Dict[str, float]:
        """
        Perturb hyperparameters for exploration.
        
        Applies random multiplicative perturbation to encourage diversity
        while maintaining reasonable parameter ranges.
        """
        import random
        
        perturbed = source_params.copy()
        low, high = self.config.perturb_factors
        
        # Perturb learning rate
        if 'learning_rate' in perturbed:
            factor = random.uniform(low, high)
            perturbed['learning_rate'] = max(1e-6, min(1e-2, perturbed['learning_rate'] * factor))
        
        # Reset epsilon for renewed exploration
        perturbed['epsilon_start'] = self.config.reset_epsilon
        
        # Perturb epsilon decay
        if 'epsilon_decay' in perturbed:
            factor = random.uniform(
                1 - self.config.epsilon_decay_perturbation,
                1 + self.config.epsilon_decay_perturbation
            )
            perturbed['epsilon_decay'] = max(0.99, min(0.9999, perturbed['epsilon_decay'] * factor))
        
        return perturbed
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def start(self):
        """Start the PBT supervisor and all workers."""
        logger.info("Starting PBT Supervisor")
        self.running = True
        
        # Launch initial workers with unique seeds
        for i in range(self.config.num_workers):
            self.launch_worker(
                worker_id=i,
                seed=i * 12345,  # Unique seeds for diversity
            )
        
        # Main supervision loop
        last_exploit_time = time.time()
        
        while self.running:
            try:
                # Check worker health and restart if needed
                self.restart_failed_workers()
                
                # Calculate and log current performance
                performance = self.calculate_worker_performance()
                
                if performance:
                    best_worker = max(performance.items(), key=lambda x: x[1])
                    worst_worker = min(performance.items(), key=lambda x: x[1])
                    logger.info(f"Current best: Worker {best_worker[0]} ({best_worker[1]:.4f})")
                    
                    # Check for adaptive culling triggers (plateau, leader gap)
                    workers_to_cull = self.detect_workers_to_cull(performance)
                    if workers_to_cull:
                        logger.info(f"Adaptive culling triggered for {len(workers_to_cull)} workers")
                        self._cull_workers(workers_to_cull, performance)
                
                # Check if it's time for scheduled exploitation
                time_since_exploit = time.time() - last_exploit_time
                if time_since_exploit >= self.config.exploit_interval_seconds:
                    self.exploit()
                    last_exploit_time = time.time()
                else:
                    remaining = self.config.exploit_interval_seconds - time_since_exploit
                    logger.debug(f"Next exploitation in {remaining:.0f}s")
                
                # Update PBT dashboard visualization
                if self.dashboard:
                    try:
                        self.dashboard.update()
                    except Exception as e:
                        logger.debug(f"Dashboard update error: {e}")
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(f"Error in supervision loop: {e}")
                time.sleep(60)
        
        self.stop()
    
    def stop(self):
        """Stop all workers and clean up."""
        logger.info("Stopping PBT Supervisor")
        self.running = False
        
        for worker_id in list(self.workers.keys()):
            self.terminate_worker(worker_id, reason="shutdown")
        
        logger.info("All workers terminated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all workers."""
        return {
            'running': self.running,
            'exploit_count': self.exploit_count,
            'uptime_seconds': time.time() - self.start_time,
            'workers': {
                wid: {
                    'status': state.status,
                    'episodes': state.total_episodes,
                    'metric': state.current_metric,
                    'restarts': state.restart_count,
                }
                for wid, state in self.workers.items()
            }
        }


# =============================================================================
# WORKER SCRIPT TEMPLATE
# =============================================================================

WORKER_SCRIPT_TEMPLATE = '''"""
DQN Training Worker for PBT

This is a standalone training worker that reports metrics back to the
PBT supervisor. It can be initialized with pre-trained weights and
custom hyperparameters.

This script is launched by pbt_supervisor.py and should not be run directly.
"""

import os
import sys
import csv
import time
import argparse
import torch
import numpy as np
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_dqn import train_dqn_agents
from training_config import *


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def report_metrics(metrics_file: str, worker_id: int, episode: int, 
                   reward: float, win: int, timestamp: float):
    """Append metrics to shared CSV file for supervisor monitoring."""
    file_exists = os.path.exists(metrics_file)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['worker_id', 'episode', 'reward', 'win', 'timestamp'])
        writer.writerow([worker_id, episode, reward, win, timestamp])


def main():
    parser = argparse.ArgumentParser(description='DQN Training Worker for PBT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker_id', type=int, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_file', type=str, required=True)
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.9995)
    
    args = parser.parse_args()
    
    print(f"[Worker {args.worker_id}] Starting with seed {args.seed}")
    set_seeds(args.seed)
    
    # TODO: Integrate with your train_dqn.py training loop
    # This is a template - you'll need to modify train_dqn.py to:
    # 1. Accept hyperparameters as arguments
    # 2. Load initial weights if provided
    # 3. Report metrics via report_metrics() after each episode
    
    # Example integration point:
    # train_dqn_agents(
    #     num_episodes=NUM_EPISODES,
    #     save_interval=SAVE_INTERVAL,
    #     model_save_path=args.model_dir,
    #     learning_rate=args.learning_rate,
    #     init_weights=args.init_weights,
    #     metrics_callback=lambda ep, rew, win: report_metrics(
    #         args.metrics_file, args.worker_id, ep, rew, win, time.time()
    #     ),
    # )


if __name__ == '__main__':
    main()
'''


def create_worker_script(output_path: str = "train_dqn_worker.py"):
    """Generate the worker script template."""
    with open(output_path, 'w') as f:
        f.write(WORKER_SCRIPT_TEMPLATE)
    logger.info(f"Created worker script template: {output_path}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Population-Based Training Supervisor for DQN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with 4 workers, exploit every hour
  python pbt_supervisor.py --num_workers 4 --exploit_interval 3600

  # Start with 8 workers, exploit every 30 minutes, cull top 30%
  python pbt_supervisor.py --num_workers 8 --exploit_interval 1800 --exploit_fraction 0.3

  # Generate worker script template only
  python pbt_supervisor.py --generate_worker_script
        """
    )
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel training workers (default: 4)')
    parser.add_argument('--exploit_interval', type=int, default=3600,
                        help='Seconds between exploitation rounds (default: 3600)')
    parser.add_argument('--exploit_fraction', type=float, default=0.5,
                        help='Fraction of workers to cull each round (default: 0.5)')
    parser.add_argument('--metric', type=str, default='mean_reward',
                        choices=['mean_reward', 'win_rate'],
                        help='Metric to optimize (default: mean_reward)')
    parser.add_argument('--model_dir', type=str, default='dqn_models',
                        help='Base directory for model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for worker logs')
    parser.add_argument('--generate_worker_script', action='store_true',
                        help='Generate worker script template and exit')
    parser.add_argument('--reset_epsilon', type=float, default=0.1,
                        help='Epsilon value for cloned workers (default: 0.1)')
    
    # Adaptive culling options
    parser.add_argument('--plateau_episodes', type=int, default=500,
                        help='Cull if no improvement for N episodes (default: 500)')
    parser.add_argument('--leader_gap', type=float, default=0.3,
                        help='Cull if worker falls X%% behind leader (default: 0.3 = 30%%)')
    parser.add_argument('--no_adaptive_culling', action='store_true',
                        help='Disable adaptive culling (plateau/gap detection)')
    
    args = parser.parse_args()
    
    if args.generate_worker_script:
        create_worker_script()
        return
    
    # Create and start supervisor
    config = PBTConfig(
        num_workers=args.num_workers,
        exploit_interval_seconds=args.exploit_interval,
        exploit_fraction=args.exploit_fraction,
        metric_key=args.metric,
        base_model_dir=args.model_dir,
        log_dir=args.log_dir,
        reset_epsilon=args.reset_epsilon,
        plateau_episodes=args.plateau_episodes,
        leader_gap_threshold=args.leader_gap,
        adaptive_culling_enabled=not args.no_adaptive_culling,
    )
    
    supervisor = PBTSupervisor(config)
    supervisor.start()


if __name__ == '__main__':
    main()
