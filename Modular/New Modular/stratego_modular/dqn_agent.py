"""
DQN Agent for Stratego Game
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
from .probabilistic_belief_state import ProbabilisticBeliefState

# Define a named tuple for experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """Deep Q-Network for Stratego"""
    
    def __init__(self, input_size: int = 200, hidden_size: int = 512, output_size: int = 1000):
        """
        Initialize the DQN network
        
        Args:
            input_size: Size of the input state representation
            hidden_size: Size of hidden layers
            output_size: Size of output (number of possible actions)
        """
        super(DQN, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation on output layer
        return x


class DQNAgent:
    """DQN Agent for Stratego with experience replay and target network"""
    
    def __init__(self, player_id: int, device, 
                 state_size: int = 200, action_size: int = 1000,
                 lr: float = 0.0001, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_min: float = 0.1, 
                 epsilon_decay: float = 0.001, 
                 buffer_size: int = 10000, batch_size: int = 32,
                 use_pbs: bool = True):
        """
        Initialize the DQN agent
        
        Args:
            player_id: Player ID (1 or -1)
            device: PyTorch device
            state_size: Size of state representation
            action_size: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Size of replay buffer
            batch_size: Size of training batches
            use_pbs: Whether to use Probabilistic Belief State
        """
        self.player_id = player_id
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.name = f"DQN Agent {player_id}"
        self.use_pbs = use_pbs
        
        # Probabilistic Belief State
        if self.use_pbs:
            self.pbs = ProbabilisticBeliefState(player_id, device)
        else:
            self.pbs = None
        
        # Neural networks
        self.q_network = DQN(state_size, 512, action_size).to(device)
        self.target_network = DQN(state_size, 512, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        
        # Update target network
        self.update_target_network()
        
    def reset(self):
        """Reset the DQN agent by reinitializing networks and optimizer"""
        # Reinitialize Q-network and target network
        self.q_network = DQN(self.state_size, 512, self.action_size).to(self.device)
        self.target_network = DQN(self.state_size, 512, self.action_size).to(self.device)
        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        # Reset epsilon to initial value
        self.epsilon = 1.0
        # Clear memory
        self.memory.clear()
        # Reset PBS
        if self.pbs:
            self.pbs.reset()
        # Update target network with new weights
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
            
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def act(self, state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
            game_state=None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Choose action using epsilon-greedy policy.
        
        Workflow:
        1. PBS first gets the value and creates possible values with confidence scores
        2. DQN then calculates Q-value using PBS-enhanced state
        
        Args:
            state: Current state representation (can be numpy array or None)
            valid_moves: List of valid moves
            game_state: Full game state object (for PBS)
        """
        if not valid_moves:
            return None
        
        # Step 1: PBS gets the value and creates possible values with confidence scores
        if self.pbs and game_state is not None:
            # Get PBS-enhanced state
            enhanced_state = self.pbs.get_belief_enhanced_state(game_state)
            if enhanced_state is not None:
                # Use enhanced state for DQN
                state = enhanced_state.cpu().numpy().flatten()
                # Pad or truncate to fixed size
                if len(state) < self.state_size:
                    state = np.pad(state, (0, self.state_size - len(state)), 'constant')
                elif len(state) > self.state_size:
                    state = state[:self.state_size]
        elif state is None and game_state is not None:
            # Fallback: get state representation if state is None
            state = self.get_state_representation(game_state)
        elif state is None:
            # No state available, return random action
            return random.choice(valid_moves)
            
        # Exploration: choose random action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
            
        # Step 2: DQN calculates Q-value using PBS-enhanced state
        # Exploitation: choose best action according to Q-network
        if not isinstance(state, torch.Tensor):
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            else:
                # Convert to numpy first
                state = np.array(state, dtype=np.float32)
                state = torch.FloatTensor(state).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
        self.q_network.train()
        
        # Convert valid moves to action indices and find best
        best_action = None
        best_q_value = float('-inf')
        
        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            q_value = q_values[0, action_idx].item()
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = move
                
        return best_action if best_action is not None else random.choice(valid_moves)
        
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
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
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _move_to_action_index(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        """Convert a move to an action index (0-999 for 10x10 board)"""
        (r_from, c_from), (r_to, c_to) = move
        # Encoding that fits within 1000 actions: from_position * 10 + to_position
        from_idx = r_from * 10 + c_from
        to_idx = r_to * 10 + c_to
        action_idx = from_idx * 10 + to_idx
        # Ensure action index is within bounds
        return action_idx % self.action_size
        
    def _action_index_to_move(self, action_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert an action index back to a move"""
        # Decode from the new encoding: action_idx = from_idx * 10 + to_idx
        from_idx = action_idx // 10
        to_idx = action_idx % 10
        r_from, c_from = from_idx // 10, from_idx % 10
        r_to, c_to = to_idx // 10, to_idx % 10
        return ((r_from, c_from), (r_to, c_to))
        
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
    def get_state_representation(self, game_state) -> np.ndarray:
        """
        Convert game state to neural network input.
        
        This method ensures that agents only use visible information.
        The game_state.board already contains only the visible board for the current player.
        
        If PBS is enabled, the state is enhanced with belief probabilities.
        """
        # Step 1: PBS gets the value and creates possible values with confidence scores
        if self.pbs and hasattr(game_state, 'board'):
            enhanced_state = self.pbs.get_belief_enhanced_state(game_state)
            if enhanced_state is not None:
                visible_board = enhanced_state.cpu().numpy()
            else:
                # Fallback to regular state
                visible_board = game_state.board
                if isinstance(visible_board, torch.Tensor):
                    visible_board = visible_board.cpu().numpy()
        else:
            # Ensure we're only using the visible board information
            if hasattr(game_state, 'board'):
                # It's a game state object with visible board for current player
                visible_board = game_state.board
                if isinstance(visible_board, torch.Tensor):
                    visible_board = visible_board.cpu().numpy()
            else:
                # It's already a board/array
                visible_board = game_state
                if isinstance(visible_board, torch.Tensor):
                    visible_board = visible_board.cpu().numpy()
        
        # Flatten the visible board (10x10 = 100 values)
        state = visible_board.flatten() if len(visible_board.shape) == 2 else visible_board
        
        # Pad or truncate to fixed size if needed
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant')
        elif len(state) > self.state_size:
            state = state[:self.state_size]
            
        return state
    
    def update_pbs_from_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                              game_state, acting_player: int):
        """
        Update PBS from an action taken.
        
        Args:
            action: The action taken
            game_state: Current game state
            acting_player: Player who took the action
        """
        if self.pbs:
            self.pbs.update_from_action(action, game_state, acting_player)
    
    def update_pbs_from_reveal(self, pos: Tuple[int, int], piece_type: PieceType):
        """
        Update PBS when a piece is revealed.
        
        Args:
            pos: Position of the revealed piece
            piece_type: Type of the revealed piece
        """
        if self.pbs:
            self.pbs.update_from_reveal(pos, piece_type)
