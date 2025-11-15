"""
Setup Agent for Stratego - Places pieces on the board before the game starts
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
from .piece import PieceType

# Define a named tuple for setup experiences
SetupExperience = namedtuple('SetupExperience', ['state', 'action', 'reward', 'next_state', 'done'])


class SetupNetwork(nn.Module):
    """Neural network for piece placement decisions"""
    
    def __init__(self, input_size: int = 400, hidden_size: int = 512, output_size: int = 400):
        """
        Initialize the setup network
        
        Args:
            input_size: Size of the input (40 pieces * 10 features)
            hidden_size: Size of hidden layers
            output_size: Number of possible positions (40 positions)
        """
        super(SetupNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class SetupAgent:
    """Agent that learns to place pieces on the board"""
    
    def __init__(self, player_id: int, device, 
                 lr: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.001,
                 buffer_size: int = 10000, batch_size: int = 32):
        """
        Initialize the setup agent
        
        Args:
            player_id: Player ID (1 or -1)
            device: PyTorch device
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Size of replay buffer
            batch_size: Size of training batches
        """
        self.player_id = player_id
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks (keep on GPU, no compilation for Windows compatibility)
        input_size = 400  # 40 pieces * 10 features (piece_type, row, col, etc.)
        output_size = 400  # 40 possible positions
        self.q_network = SetupNetwork(input_size, 512, output_size).to(device)
        self.target_network = SetupNetwork(input_size, 512, output_size).to(device)
        
        # Enable cuDNN benchmarking for faster convolutions (if using conv layers)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, non-deterministic
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Experience replay - store tensors on GPU
        self.memory = deque(maxlen=buffer_size)
        
        # Track policy losses
        self.policy_losses = []
        
        # Track rewards for visualization
        self.reward_history = []
        self.episode_rewards = []  # Track rewards per episode
        
        # Update target network
        self.update_target_network()
        
    def reset(self):
        """Reset the setup agent"""
        self.q_network = SetupNetwork(400, 512, 400).to(self.device)
        self.target_network = SetupNetwork(400, 512, 400).to(self.device)
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.memory.clear()
        self.policy_losses = []
        self.reward_history = []
        self.episode_rewards = []
        self.epsilon = 1.0
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_state_representation(self, pieces: List[PieceType], available_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Convert pieces and positions to state representation
        
        Args:
            pieces: List of pieces to place
            available_positions: List of available positions
            
        Returns:
            State tensor
        """
        # Create feature vector: [piece_type_1, row_1, col_1, piece_type_2, row_2, col_2, ...]
        features = []
        for i, piece in enumerate(pieces):
            features.extend([
                piece.value,  # Piece type
                0.0,  # Row (will be set when placed)
                0.0,  # Col (will be set when placed)
                float(i) / 40.0,  # Piece index
            ])
        
        # Pad to 400 features (40 pieces * 10 features)
        while len(features) < 400:
            features.append(0.0)
        features = features[:400]
        
        return torch.FloatTensor(features).to(self.device)
        
    def place_pieces(self, pieces: List[PieceType], available_positions: List[Tuple[int, int]]) -> List[Tuple[PieceType, Tuple[int, int]]]:
        """
        Place pieces on the board using the agent's policy
        
        Args:
            pieces: List of pieces to place (must be exactly 40)
            available_positions: List of available positions (must be exactly 40)
            
        Returns:
            List of (piece, position) tuples
        """
        if len(pieces) != 40 or len(available_positions) != 40:
            # Fallback to random placement
            random.shuffle(available_positions)
            return list(zip(pieces, available_positions))
        
        # Get state representation
        state = self.get_state_representation(pieces, available_positions)
        
        # Place pieces one by one
        placement = []
        remaining_positions = available_positions.copy()
        remaining_pieces = pieces.copy()
        
        for i in range(40):
            if not remaining_pieces or not remaining_positions:
                break
                
            # Select piece (in order)
            piece = remaining_pieces[0]
            remaining_pieces = remaining_pieces[1:]
            
            # Select position using epsilon-greedy
            if random.random() < self.epsilon:
                # Random exploration
                position = random.choice(remaining_positions)
            else:
                # Greedy action - select best available position
                # For simplicity, use a heuristic: prefer positions closer to center/back
                # In a full implementation, this would use the Q-network
                if self.player_id == 1:
                    # Player 1 prefers rows 6-7 (back rows)
                    position = max(remaining_positions, key=lambda p: 7 - p[0] if p[0] >= 6 else -10)
                else:
                    # Player 2 prefers rows 2-3 (back rows)
                    position = max(remaining_positions, key=lambda p: p[0] if p[0] <= 3 else -10)
            
            remaining_positions.remove(position)
            placement.append((piece, position))
        
        return placement
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(SetupExperience(state, action, reward, next_state, done))
        # Track reward for visualization
        self.reward_history.append(reward)
        if done:
            # If episode is done, store the final reward (which is the total episode reward)
            # The reward passed here is already the total calculated reward for the episode
            self.episode_rewards.append(reward)
        
    def replay(self) -> Optional[float]:
        """
        Train the model on a batch of experiences
        
        Returns:
            Policy loss value or None if not enough experiences
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e.state for e in batch])
        next_states = torch.stack([e.next_state for e in batch])
        
        # Create tensors directly on GPU (avoid CPU intermediate)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track policy loss
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
            
        return loss_value
        
    def get_average_policy_loss(self, window: int = 100) -> float:
        """Get average policy loss over the last N training steps"""
        if not self.policy_losses:
            return 0.0
        recent_losses = self.policy_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
        
    def save_model(self, path: str):
        """Save the model to a file"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        
    def load_model(self, path: str):
        """Load the model from a file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.update_target_network()

