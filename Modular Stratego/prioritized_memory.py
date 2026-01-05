
import numpy as np
import random
import torch
from collections import namedtuple, deque

# Define Experience namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class StandardReplayBuffer:
    """
    Standard Replay Buffer for Rainbow DQN (Feed-Forward).
    Stores individual transitions on GPU with Float16 for memory efficiency.
    """
    def __init__(self, capacity, device='cuda', use_float16=True):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.store_on_gpu = (device != 'cpu' and device != torch.device('cpu'))
        self.use_float16 = use_float16 and self.store_on_gpu  # Float16 only on GPU
        self.storage_dtype = torch.float16 if self.use_float16 else torch.float32
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer with Float16 storage."""
        if self.store_on_gpu:
            # Convert to storage dtype (float16 for memory efficiency)
            state_t = state.to(self.device, dtype=self.storage_dtype) if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device, dtype=self.storage_dtype)
            next_state_t = next_state.to(self.device, dtype=self.storage_dtype) if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device, dtype=self.storage_dtype)
        else:
            # CPU fallback (always float32)
            state_t = state.cpu().float() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
            next_state_t = next_state.cpu().float() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float32)
        
        # Action/Reward/Done are usually scalars
        action_t = action.cpu() if isinstance(action, torch.Tensor) else action
        reward_t = reward.cpu() if isinstance(reward, torch.Tensor) else reward
        done_t = done.cpu() if isinstance(done, torch.Tensor) else done
        
        exp = Experience(state_t, action_t, reward_t, next_state_t, done_t)
        self.buffer.append(exp)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            states, actions, rewards, next_states, dones (all Tensors on device, float32)
        """
        batch = random.sample(self.buffer, batch_size)
        
        if self.store_on_gpu:
            # Already on GPU, stack and convert to float32 for computation
            states = torch.stack([e.state for e in batch]).float()
        else:
            # Move from CPU to GPU during sampling
            states = torch.stack([e.state for e in batch]).to(self.device).float()
        
        # Handle actions (might be int or tensor)
        actions_list = [e.action for e in batch]
        if isinstance(actions_list[0], torch.Tensor):
             actions = torch.stack(actions_list).to(self.device).long()
        else:
             actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
             
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        
        if self.store_on_gpu:
            next_states = torch.stack([e.next_state for e in batch]).float()
        else:
            next_states = torch.stack([e.next_state for e in batch]).to(self.device).float()
            
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class SumTree:
    """
    Sum Tree data structure for O(log n) prioritized sampling.
    Each leaf stores a priority, and each parent stores the sum of children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Leaf data
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find leaf index for a given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority sum."""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new data with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority at tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get data and tree index for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using Sum Tree.
    Samples transitions with probability proportional to TD-error.
    Uses Float16 storage for memory efficiency.
    
    Reference: Schaul et al. (2015) "Prioritized Experience Replay"
    """
    def __init__(self, capacity, device='cuda', alpha=0.6, beta_start=0.4, beta_end=1.0, beta_anneal_episodes=10000, use_float16=True):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha  # Priority exponent
        self.beta = beta_start  # Importance sampling weight
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_episodes = beta_anneal_episodes
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        self.max_priority = 1.0
        
        self.tree = SumTree(capacity)
        self.store_on_gpu = (device != 'cpu' and device != torch.device('cpu'))
        self.use_float16 = use_float16 and self.store_on_gpu
        self.storage_dtype = torch.float16 if self.use_float16 else torch.float32
        
    def add(self, state, action, reward, next_state, done, priority=None, is_winning_experience=False):
        """Add experience with Float16 storage and priority boost for winning experiences.
        
        Args:
            state, action, reward, next_state, done: Standard experience tuple
            priority: Optional explicit priority (if None, uses max_priority)
            is_winning_experience: If True, boost priority 10x for self-imitation learning
        """
        if self.store_on_gpu:
            # Convert to Float16 for memory efficiency
            state_t = state.to(self.device, dtype=self.storage_dtype) if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device, dtype=self.storage_dtype)
            next_state_t = next_state.to(self.device, dtype=self.storage_dtype) if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device, dtype=self.storage_dtype)
        else:
            state_t = state.cpu().float() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
            next_state_t = next_state.cpu().float() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float32)
        
        action_t = action.cpu() if isinstance(action, torch.Tensor) else action
        reward_t = reward.cpu() if isinstance(reward, torch.Tensor) else reward
        done_t = done.cpu() if isinstance(done, torch.Tensor) else done
        
        exp = Experience(state_t, action_t, reward_t, next_state_t, done_t)
        
        # Use max priority for new experiences
        if priority is None:
            priority = self.max_priority ** self.alpha
        
        # SELF-IMITATION LEARNING: Boost priority for winning experiences
        # This makes the agent learn more from its successes
        if is_winning_experience:
            priority *= 10.0  # 10x priority boost for wins
        
        self.tree.add(priority, exp)
    
    def sample(self, batch_size):
        """
        Sample batch with priorities.
        Returns: states, actions, rewards, next_states, dones, indices, weights
        """
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            if data is not None and data != 0:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        if len(batch) == 0:
            return None
        
        # Compute importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Stack tensors and convert to float32 for computation
        if self.store_on_gpu:
            states = torch.stack([e.state for e in batch]).float()
            next_states = torch.stack([e.next_state for e in batch]).float()
        else:
            states = torch.stack([e.state for e in batch]).to(self.device).float()
            next_states = torch.stack([e.next_state for e in batch]).to(self.device).float()
        
        actions_list = [e.action for e in batch]
        if isinstance(actions_list[0], torch.Tensor):
            actions = torch.stack(actions_list).to(self.device).long()
        else:
            actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def anneal_beta(self, episode):
        """Anneal beta from start to end over specified episodes."""
        fraction = min(episode / self.beta_anneal_episodes, 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def __len__(self):
        return self.tree.n_entries
    
    def clear(self):
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0


class NStepBuffer:
    """
    N-Step experience accumulator for multi-step returns.
    Collects n consecutive transitions and computes n-step return.
    
    Reference: Sutton & Barto (2018), Hessel et al. (2018) Rainbow
    """
    def __init__(self, n_steps=3, gamma=0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
        self.gamma_powers = [gamma ** i for i in range(n_steps)]
    
    def add(self, state, action, reward, next_state, done):
        """
        Add transition and return n-step transition if ready.
        Returns: (n_step_state, action, n_step_reward, n_step_next_state, done) or None
        """
        self.buffer.append((state, action, reward, next_state, done))
        
        # Check if episode ended
        if done:
            # Flush all remaining transitions with truncated n-step returns
            result = self._compute_nstep()
            self.buffer.clear()
            return result
        
        # Only return when we have n steps
        if len(self.buffer) == self.n_steps:
            return self._compute_nstep()
        
        return None
    
    def _compute_nstep(self):
        """Compute n-step return from buffer contents."""
        if len(self.buffer) == 0:
            return None
        
        # Get first transition's state and action
        first_state, first_action, _, _, _ = self.buffer[0]
        
        # Compute n-step reward: r_1 + γr_2 + γ²r_3 + ...
        n_step_reward = 0.0
        for i, (_, _, reward, _, _) in enumerate(self.buffer):
            n_step_reward += self.gamma_powers[i] * reward
        
        # Get last transition's next_state and done
        _, _, _, last_next_state, last_done = self.buffer[-1]
        
        return first_state, first_action, n_step_reward, last_next_state, last_done
    
    def flush(self):
        """Flush remaining partial trajectories (call at episode end)."""
        results = []
        while len(self.buffer) > 0:
            result = self._compute_nstep()
            if result:
                results.append(result)
            if len(self.buffer) > 0:
                self.buffer.popleft()
        return results
    
    def reset(self):
        """Clear buffer (call at episode start)."""
        self.buffer.clear()

