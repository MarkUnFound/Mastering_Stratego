"""
Utility functions and classes for training DQN agents.
"""
import queue
import threading
import time
import os
import json
import numpy as np
from typing import List, Dict, Optional

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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


def save_training_history(metrics: dict, model_save_path: str):
    """Save training history to JSON file for continuity across training sessions"""
    history_file = os.path.join(model_save_path, "training_history.json")
    try:
        # Just dump the metrics dict directly, it already contains all the lists
        with open(history_file, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
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
