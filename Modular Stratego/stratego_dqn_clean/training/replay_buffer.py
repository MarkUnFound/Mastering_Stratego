"""
Simple Uniform Circular Replay Buffer for Double DQN
NO prioritized replay, NO sum-tree complexity.

This is a clean implementation for memory-constrained training.
"""

import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional, List, NamedTuple
import random


class Experience(NamedTuple):
    """Single experience tuple."""
    state: torch.Tensor        # (79, 10, 10)
    action: int                # Action index
    reward: float              # Scalar reward
    next_state: torch.Tensor   # (79, 10, 10)
    done: bool                 # Terminal flag


class UniformReplayBuffer:
    """
    Simple circular buffer for Double DQN.
    
    Features:
    - Uniform random sampling (no prioritization)
    - GPU storage option for faster sampling
    - Float16 storage for memory efficiency
    
    Memory constraints:
    - Designed for 6GB VRAM
    - Uses deque for O(1) append/pop
    """
    
    def __init__(
        self,
        capacity: int,
        device: torch.device = None,
        use_gpu_storage: bool = True,
        use_float16: bool = True
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: PyTorch device (default: cuda if available)
            use_gpu_storage: If True, store tensors on GPU
            use_float16: If True, use float16 for memory efficiency
        """
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu_storage = use_gpu_storage
        self.use_float16 = use_float16
        
        # Main storage
        self.buffer: deque = deque(maxlen=capacity)
        
        # Storage dtype
        self.dtype = torch.float16 if use_float16 else torch.float32
        
        print(f"[UniformReplayBuffer] Initialized with capacity={capacity:,}")
        print(f"  - Storage: {'GPU' if use_gpu_storage else 'CPU'}")
        print(f"  - Dtype: {self.dtype}")
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """
        Add experience to buffer.
        
        Args:
            state: (79, 10, 10) or (15, 10, 10) current state
            action: Action index
            reward: Scalar reward
            next_state: (79, 10, 10) or (15, 10, 10) next state
            done: Terminal flag
        """
        # Convert to appropriate format
        if self.use_gpu_storage:
            state = state.to(self.device, dtype=self.dtype)
            next_state = next_state.to(self.device, dtype=self.dtype)
        else:
            state = state.cpu().to(dtype=self.dtype)
            next_state = next_state.cpu().to(dtype=self.dtype)
        
        # Store as tuple (more memory efficient than NamedTuple for deque)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences uniformly.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            states: (B, C, H, W)
            actions: (B,)
            rewards: (B,)
            next_states: (B, C, H, W)
            dones: (B,)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences: {len(self.buffer)} < {batch_size}")
        
        # Random indices
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Gather experiences
        batch = [self.buffer[i] for i in indices]
        
        # Unpack and stack
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.float32, device=self.device)
        
        # Convert to float32 for training
        states = states.to(dtype=torch.float32)
        next_states = next_states.to(dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()
    
    def get_memory_usage(self) -> float:
        """Estimate memory usage in GB."""
        if len(self.buffer) == 0:
            return 0.0
        
        # Estimate based on first experience
        sample = self.buffer[0]
        state_size = sample[0].numel() * sample[0].element_size()
        next_state_size = sample[3].numel() * sample[3].element_size()
        other_size = 8 + 4 + 1  # action (int64), reward (float32), done (bool)
        
        exp_size = state_size + next_state_size + other_size
        total_bytes = exp_size * len(self.buffer)
        
        return total_bytes / 1e9


class NStepBuffer:
    """
    N-step return accumulator for multi-step learning.
    
    Collects n consecutive transitions and computes n-step return:
    G_t = R_t + γR_{t+1} + γ²R_{t+2} + ... + γ^{n-1}R_{t+n-1} + γ^n V(S_{t+n})
    
    Note: This is optional for Double DQN but can improve credit assignment.
    """
    
    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        """
        Initialize n-step buffer.
        
        Args:
            n_steps: Number of steps for multi-step returns
            gamma: Discount factor
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n_steps)
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> Optional[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        """
        Add transition and return n-step transition if ready.
        
        Returns:
            n-step transition tuple or None if buffer not full
        """
        self.buffer.append((state, action, reward, next_state, done))
        
        if len(self.buffer) < self.n_steps:
            return None
        
        return self._compute_nstep()
    
    def _compute_nstep(self) -> Tuple[torch.Tensor, int, float, torch.Tensor, bool]:
        """Compute n-step return from buffer."""
        # Get initial state and action
        state = self.buffer[0][0]
        action = self.buffer[0][1]
        
        # Compute n-step return
        n_step_return = 0.0
        gamma_power = 1.0
        
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_return += gamma_power * r
            gamma_power *= self.gamma
            
            if d:  # Terminal state
                return (state, action, n_step_return, self.buffer[i][3], True)
        
        # Non-terminal: return last next_state
        return (state, action, n_step_return, self.buffer[-1][3], False)
    
    def flush(self) -> List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        """Flush remaining partial trajectories at episode end."""
        results = []
        
        while len(self.buffer) > 0:
            if len(self.buffer) >= 1:
                results.append(self._compute_partial_nstep())
            self.buffer.popleft()
        
        return results
    
    def _compute_partial_nstep(self) -> Tuple[torch.Tensor, int, float, torch.Tensor, bool]:
        """Compute partial n-step return for remaining buffer."""
        state = self.buffer[0][0]
        action = self.buffer[0][1]
        
        n_step_return = 0.0
        gamma_power = 1.0
        
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_return += gamma_power * r
            gamma_power *= self.gamma
        
        # Use last transition's info
        last = self.buffer[-1]
        return (state, action, n_step_return, last[3], last[4])
    
    def reset(self):
        """Clear buffer for new episode."""
        self.buffer.clear()
