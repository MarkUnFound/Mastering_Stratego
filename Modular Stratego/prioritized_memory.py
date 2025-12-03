
import numpy as np
import random
import torch
from collections import namedtuple, deque

# Define Experience namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class StandardReplayBuffer:
    """
    Standard Replay Buffer for Rainbow DQN (Feed-Forward).
    Stores individual transitions.
    """
    def __init__(self, capacity, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        # Ensure tensors are on CPU for storage to save GPU memory, or keep on GPU if preferred
        # Here we keep them as is, assuming they are already processed/moved in agent
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            states, actions, rewards, next_states, dones (all Tensors)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device) # float for math
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
