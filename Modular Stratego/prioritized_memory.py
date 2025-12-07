
import numpy as np
import random
import torch
from collections import namedtuple, deque

# Define Experience namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class StandardReplayBuffer:
    """
    Standard Replay Buffer for Rainbow DQN (Feed-Forward).
    Stores individual transitions on GPU for maximum performance.
    """
    def __init__(self, capacity, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.store_on_gpu = (device != 'cpu' and device != torch.device('cpu'))
        if self.store_on_gpu:
            print("Replay buffer storing on GPU")
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        if self.store_on_gpu:
            # Store on GPU for faster sampling
            state_t = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device)
            next_state_t = next_state.to(self.device) if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device)
        else:
            # CPU fallback
            state_t = state.cpu() if isinstance(state, torch.Tensor) else torch.tensor(state)
            next_state_t = next_state.cpu() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state)
        
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
            states, actions, rewards, next_states, dones (all Tensors on device)
        """
        batch = random.sample(self.buffer, batch_size)
        
        if self.store_on_gpu:
            # Already on GPU, just stack (fast path)
            states = torch.stack([e.state for e in batch])
        else:
            # Move from CPU to GPU during sampling
            states = torch.stack([e.state for e in batch]).to(self.device)
        
        # Handle actions (might be int or tensor)
        actions_list = [e.action for e in batch]
        if isinstance(actions_list[0], torch.Tensor):
             actions = torch.stack(actions_list).to(self.device).long()
        else:
             actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
             
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        
        if self.store_on_gpu:
            next_states = torch.stack([e.next_state for e in batch])
        else:
            next_states = torch.stack([e.next_state for e in batch]).to(self.device)
            
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
