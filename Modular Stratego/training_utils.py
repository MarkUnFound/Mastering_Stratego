"""
Utility functions and classes for training DQN agents.
"""
import queue
import threading
import time
import os
import json
from typing import List, Dict, Optional

class ReplayPrefetcher:
    """Background prefetcher that samples replay batches asynchronously."""

    def __init__(self, agent, max_queue_size: int = 4):
        self.agent = agent
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            batch = self.agent.sample_replay_batch()
            if batch is None:
                time.sleep(0.01)
                continue
            try:
                self.queue.put(batch, timeout=0.1)
            except queue.Full:
                continue

    def get_batch(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def save_counters(total_episodes_file, total_steps_file, total_episodes, total_steps):
    with open(total_episodes_file, 'w') as f:
        f.write(str(total_episodes))
    with open(total_steps_file, 'w') as f:
        f.write(str(total_steps))
    print(f"💾 Saved persistent counters: {total_episodes} episodes, {total_steps:,} steps")


def save_training_history(model_save_path: str, 
                          episode_history: List[int],
                          rewards_history: dict,
                          wins_history: dict,
                          epsilon_history: dict,
                          policy_loss_history: dict,
                          setup_agent1_rewards: List[float],
                          setup_agent2_rewards: List[float],
                          setup_agent1_losses: List[float],
                          setup_agent2_losses: List[float],
                          pbs_evaluator1_losses: List[float],
                          pbs_evaluator2_losses: List[float],
                          pbs_evaluator1_buffer_sizes: List[int],
                          pbs_evaluator2_buffer_sizes: List[int],
                          avg_q_history: dict,
                          entropy_history: dict):
    """Save training history to JSON file for continuity across training sessions"""
    history_file = os.path.join(model_save_path, "training_history.json")
    try:
        history_data = {
            'episode_history': episode_history,
            'rewards_history': rewards_history,
            'wins_history': wins_history,
            'epsilon_history': epsilon_history,
            'policy_loss_history': policy_loss_history,
            'setup_agent1_rewards': setup_agent1_rewards,
            'setup_agent2_rewards': setup_agent2_rewards,
            'setup_agent1_losses': setup_agent1_losses,
            'setup_agent2_losses': setup_agent2_losses,
            'pbs_evaluator1_losses': pbs_evaluator1_losses,
            'pbs_evaluator2_losses': pbs_evaluator2_losses,
            'pbs_evaluator1_buffer_sizes': pbs_evaluator1_buffer_sizes,
            'pbs_evaluator2_buffer_sizes': pbs_evaluator2_buffer_sizes,
            'avg_q_history': avg_q_history,
            'entropy_history': entropy_history
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save training history: {e}")


def load_training_history(model_save_path: str) -> Optional[dict]:
    """Load training history from JSON file if it exists"""
    history_file = os.path.join(model_save_path, "training_history.json")
    
    if not os.path.exists(history_file):
        return None
    
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        print(f"📊 Loaded training history: {len(history_data.get('episode_history', []))} episodes")
        return history_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Could not load training history: {e}, starting fresh")
        return None
