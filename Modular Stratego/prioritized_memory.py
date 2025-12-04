
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
        # Store on CPU to save VRAM
        state_cpu = state.cpu() if isinstance(state, torch.Tensor) else torch.tensor(state)
        next_state_cpu = next_state.cpu() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state)
        
        # Action/Reward/Done are usually scalars or small tensors, but move to CPU for consistency
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
        reward_cpu = reward.cpu() if isinstance(reward, torch.Tensor) else reward
        done_cpu = done.cpu() if isinstance(done, torch.Tensor) else done
        
        exp = Experience(state_cpu, action_cpu, reward_cpu, next_state_cpu, done_cpu)
        self.buffer.append(exp)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            states, actions, rewards, next_states, dones (all Tensors on device)
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Move back to device (GPU) during sampling
        states = torch.stack([e.state for e in batch]).to(self.device)
        
        # Handle actions (might be int or tensor)
        actions_list = [e.action for e in batch]
        if isinstance(actions_list[0], torch.Tensor):
             actions = torch.stack(actions_list).to(self.device).long()
        else:
             actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
             
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([e.next_state for e in batch]).to(self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
