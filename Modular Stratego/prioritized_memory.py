
import numpy as np
import random
from collections import namedtuple

# Define Experience namedtuple to match DQNAgent
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """
    SumTree data structure for efficient priority-based sampling.
    Leaf nodes store priorities, internal nodes store sum of children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Stores experiences with priorities and samples them based on priority.
    """
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent (0 = uniform, 1 = full priority)
        self.epsilon = 0.01  # Small constant to ensure non-zero priority

    def add(self, experience, error=None):
        """
        Add a new experience to the buffer.
        New experiences are given max priority to ensure they are seen at least once.
        """
        if error is None:
            # Find max priority
            # Efficiently find max in leaf nodes (last capacity nodes)
            # Or just use 1.0 if empty, or keep track of max_p
            # Simplified: check tree leaves if possible, or just use a default high value
            # For SumTree implementation, we can look at the tree array
            # But simpler: just use max possible or 1.0 if empty
            
            # Get max priority from the tree leaves (indices capacity-1 to end)
            # This might be slow if we scan all. 
            # Optimization: maintain max_priority variable
            max_p = np.max(self.tree.tree[-self.capacity:]) if self.tree.n_entries > 0 else 1.0
            if max_p == 0:
                max_p = 1.0
        else:
            max_p = (abs(error) + self.epsilon) ** self.alpha
            
        self.tree.add(max_p, experience)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priority.
        Returns batch, indices (for update), and importance sampling weights.
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            # Handle case where data might be 0/None if tree not full/initialized correctly (edge case)
            if data == 0 or data is None:
                # Fallback: resample or pick random valid
                # This shouldn't happen if logic is correct
                continue
                
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Calculate Importance Sampling Weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()  # Normalize

        return batch, idxs, is_weight

    def update(self, idxs, errors):
        """
        Update priorities of sampled experiences based on new TD errors.
        """
        for idx, error in zip(idxs, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            
    def __len__(self):
        return self.tree.n_entries
    
    def clear(self):
        """Clear the buffer"""
        self.tree = SumTree(self.capacity)
