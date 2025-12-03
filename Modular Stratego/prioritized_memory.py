
import numpy as np
import random
import torch
from collections import namedtuple, deque

# Define Experience namedtuple to match DRQNAgent
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SequentialReplayBuffer:
    """
    Replay Buffer for DRQN that stores entire episodes and samples sequences (traces).
    Handles zero-padding for sequences that exceed episode boundaries or for short episodes.
    """
    def __init__(self, capacity, trace_length=8, device='cuda'):
        self.capacity = capacity
        self.trace_length = trace_length
        self.device = device
        
        # Storage: List of episodes, where each episode is a list of Experience tuples
        self.buffer = deque(maxlen=capacity)
        
    def add(self, episode):
        """
        Add a full episode to the buffer.
        Args:
            episode: List of Experience tuples representing one complete game episode.
        """
        # Only add non-empty episodes
        if len(episode) > 0:
            self.buffer.append(episode)
    
    def sample(self, batch_size):
        """
        Sample a batch of sequences (traces) from stored episodes.
        
        Returns:
            batch_traces: List of list of Experience tuples (the sequences)
            mask: Tensor of shape (batch_size, trace_length) indicating valid steps (1=valid, 0=padded)
        """
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        batch_traces = []
        mask = torch.zeros((batch_size, self.trace_length), dtype=torch.float32, device=self.device)
        
        for i, episode in enumerate(sampled_episodes):
            # Pick a random start point for the trace
            # We want to ensure we can get at least some valid data
            if len(episode) <= self.trace_length:
                # If episode is shorter than trace, take the whole thing and pad
                start_idx = 0
                trace = episode
            else:
                # If episode is longer, pick a random start
                # We allow picking near the end, which will result in padding
                start_idx = random.randint(0, len(episode) - 1)
                trace = episode[start_idx : start_idx + self.trace_length]
            
            # Create the padded trace
            padded_trace = []
            
            # 1. Add valid experiences
            for j, exp in enumerate(trace):
                padded_trace.append(exp)
                mask[i, j] = 1.0
                
            # 2. Add zero-padding if necessary
            while len(padded_trace) < self.trace_length:
                # Create a zero-filled experience
                # We need to match the shape of the state tensors
                # Assuming state is (C, H, W)
                zero_state = torch.zeros_like(episode[0].state)
                zero_action = 0 # Dummy action
                zero_reward = 0.0
                zero_next_state = torch.zeros_like(episode[0].next_state)
                zero_done = True # Treat padding as done
                
                padding_exp = Experience(zero_state, zero_action, zero_reward, zero_next_state, zero_done)
                padded_trace.append(padding_exp)
                # Mask remains 0 for these steps
            
            batch_traces.append(padded_trace)
            
        return batch_traces, mask
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
