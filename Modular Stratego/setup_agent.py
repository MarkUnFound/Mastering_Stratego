"""
Setup Agent for Stratego - Places pieces on the board before the game starts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import List, Tuple, Optional, NamedTuple
from collections import deque
from piece import PieceType
from critic import SetupExploitabilityCritic

class SetupExperience(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

class QVectorSetupDQN(nn.Module):
    """
    Q-Vector Deep Q-Network for Setup Agent
    
    Architecture:
    - Input: (3, 10, 10) - Board, Mask, Piece
    - Output: Two heads (Q-Vector):
        - Win Head: Q_win(s, a) - Probability of winning with this setup
        - Leak Head: Q_leak(s, a) - Probability of information leakage (predictability)
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (3, 10, 10), output_size: int = 100):
        """
        Args:
            input_shape: Shape of input (channels, height, width)
            output_size: 100 (10x10 board positions)
        """
        super(QVectorSetupDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate flattened size: 64 * 10 * 10 = 6400
        self.flatten_size = 64 * 10 * 10
        
        self.fc_shared = nn.Linear(self.flatten_size, 512)
        
        # Head 1: Win Probability
        self.fc_win = nn.Linear(512, output_size)
        
        # Head 2: Information Leakage (Predictability)
        self.fc_leak = nn.Linear(512, output_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        
        q_win = self.fc_win(x)
        q_leak = self.fc_leak(x)
        
        # Stack to form Q-Vector: (batch, output_size, 2)
        return torch.stack([q_win, q_leak], dim=2)


class SetupAgent:
    """Agent that learns to place pieces on the board"""
    
    def __init__(self, player_id: int, device, 
                 lr: float = 0.0001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.001,
                 buffer_size: int = 10000, batch_size: int = 32):
        """
        Initialize the setup agent
        """
        self.player_id = player_id
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks (CNN) - matches main DQN architecture
        # Channel 0: Board state (piece values, normalized)
        # Channel 1: Valid positions mask
        # Channel 2: Piece type being placed (constant plane)
        input_shape = (3, 10, 10)
        output_size = 100  # 100 positions (10x10 board)
        
        self.q_network = QVectorSetupDQN(input_shape, output_size).to(device)
        self.target_network = QVectorSetupDQN(input_shape, output_size).to(device)
        
        # Exploitability Critic
        self.critic = SetupExploitabilityCritic(input_shape=input_shape, output_size=output_size).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01)
        self.critic_loss_fn = nn.CrossEntropyLoss()
        self.critic_weight = 0.1
        
        # Enable cuDNN benchmarking
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        self.memory = deque(maxlen=buffer_size)
        self.current_episode_memory = [] # Temporary storage for current episode steps
        self.policy_losses = []
        self.reward_history = []
        self.episode_rewards = []
        
        self.update_target_network()
        
    def reset(self):
        """Reset the setup agent"""
        input_shape = (3, 10, 10)
        output_size = 100
        self.q_network = QVectorSetupDQN(input_shape, output_size).to(self.device)
        self.target_network = QVectorSetupDQN(input_shape, output_size).to(self.device)
        
        self.critic = SetupExploitabilityCritic(input_shape=input_shape, output_size=output_size).to(self.device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=0.01)
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.memory.clear()
        self.current_episode_memory = []
        self.policy_losses = []
        self.reward_history = []
        self.episode_rewards = []
        self.epsilon = 1.0
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_state_representation(self, current_board: torch.Tensor, piece_to_place: PieceType, available_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Convert board and piece to CNN state representation (3, 10, 10)
        """
        # Channel 0: Board state (normalized)
        board_channel = current_board.clone().float() / 12.0 # Max piece value is roughly 12 (Bomb/Flag/Marshall)
        
        # Channel 1: Valid positions mask
        mask_channel = torch.zeros((10, 10), device=self.device)
        for r, c in available_positions:
            mask_channel[r, c] = 1.0
            
        # Channel 2: Piece type being placed
        piece_channel = torch.full((10, 10), float(piece_to_place.value) / 12.0, device=self.device)
        
        state = torch.stack([board_channel, mask_channel, piece_channel])
        return state.unsqueeze(0) # Add batch dimension: (1, 3, 10, 10)
        
    def place_pieces(self, pieces: List[PieceType], available_positions: List[Tuple[int, int]]) -> List[Tuple[PieceType, Tuple[int, int]]]:
        """
        Place pieces on the board using the agent's policy
        """
        self.current_episode_memory = [] # Clear previous episode memory
        
        if len(pieces) != 40:
             # Fallback for safety
             random.shuffle(available_positions)
             return list(zip(pieces, available_positions))
        
        # Initialize board state (empty)
        current_board = torch.zeros((10, 10), device=self.device)
        
        placement = []
        remaining_positions = available_positions.copy()
        remaining_pieces = pieces.copy() 
        
        for i in range(40):
            piece = remaining_pieces[0]
            remaining_pieces = remaining_pieces[1:]
            
            # Get state
            state = self.get_state_representation(current_board, piece, remaining_positions)
            
            # Select position
            if random.random() < self.epsilon:
                position = random.choice(remaining_positions)
            else:
                with torch.no_grad():
                    # Output: (1, 100, 2) -> [Q_win, Q_leak]
                    q_vectors = self.q_network(state)
                    q_vectors = q_vectors.view(100, 2)
                    
                    q_win = q_vectors[:, 0].view(10, 10)
                    q_leak = q_vectors[:, 1].view(10, 10)
                    
                    # Nash Strategy: Maximize Win - Leak
                    # We want a setup that is robust (high win prob) and unpredictable (low leak prob)
                    # Assuming Q_leak predicts predictability/exploitability
                    nash_score = q_win - q_leak
                    
                    # Mask invalid positions
                    mask = torch.full((10, 10), -float('inf'), device=self.device)
                    for r, c in remaining_positions:
                        mask[r, c] = 0
                    
                    masked_scores = nash_score + mask
                    
                    # Select best position
                    best_idx = torch.argmax(masked_scores).item()
                    r, c = best_idx // 10, best_idx % 10
                    position = (r, c)
            
            # Calculate action index (0-99)
            action_idx = position[0] * 10 + position[1]
            
            # Update board
            current_board[position[0], position[1]] = piece.value
            remaining_positions.remove(position)
            placement.append((piece, position))
            
            # Get next state (if not done)
            if i < 39:
                next_piece = remaining_pieces[0]
                next_state = self.get_state_representation(current_board, next_piece, remaining_positions)
                done = False
            else:
                # Final state
                next_state = torch.zeros_like(state) # Placeholder
                done = True
                
            # Store step in temporary memory (reward will be filled later)
            self.current_episode_memory.append({
                'state': state,
                'action': action_idx,
                'next_state': next_state,
                'done': done
            })
            
        return placement
        
    def finish_episode(self, total_reward: float):
        """
        Apply the final game reward to all steps in the episode and store in replay buffer.
        This assumes the reward is sparse and given only at the end.
        """
        for step in self.current_episode_memory:
            # We can use reward shaping here if we want, e.g. decay reward for earlier steps?
            # For now, just give the same reward to all placement decisions.
            self.remember(
                step['state'],
                step['action'],
                total_reward,
                step['next_state'],
                step['done']
            )
        self.current_episode_memory = [] # Clear after storing
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(SetupExperience(state, action, reward, next_state, done))
        self.reward_history.append(reward)
        if done:
            self.episode_rewards.append(reward)
        
    def replay(self) -> Optional[float]:
        """Train the model"""
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([e.state for e in batch]) # (B, 3, 10, 10)
        next_states = torch.cat([e.next_state for e in batch])
        
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        
        # --- Train Critic ---
        critic_logits = self.critic(states)
        critic_loss = self.critic_loss_fn(critic_logits, actions)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # --- Calculate Penalty ---
        with torch.no_grad():
            probs = F.softmax(self.critic(states), dim=1)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            penalty = self.critic_weight * action_probs
            
        penalized_rewards = rewards - penalty
        
        # --- Train Agent (Dual-Phase) ---
        # Current Q-Vectors
        current_q_vectors = self.q_network(states) # (B, 100, 2)
        
        # Gather Q-values for taken actions
        current_q_values = current_q_vectors.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1) # (B, 2)
        current_q_win = current_q_values[:, 0]
        current_q_leak = current_q_values[:, 1]
        
        # Next Q-Vectors
        next_q_vectors = self.target_network(next_states).detach() # (B, 100, 2)
        
        # Nash selection for next state
        next_scores = next_q_vectors[:, :, 0] - next_q_vectors[:, :, 1]
        best_next_actions = next_scores.argmax(dim=1)
        
        next_q_values = next_q_vectors.gather(1, best_next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
        next_q_win = next_q_values[:, 0]
        next_q_leak = next_q_values[:, 1]
        
        # Targets
        # Win Head: Predicts reward (winning)
        target_q_win = penalized_rewards + (self.gamma * next_q_win * ~dones)
        
        # Leak Head: Predicts penalty (exploitability)
        # We want Q_leak to estimate the penalty
        target_q_leak = penalty + (self.gamma * next_q_leak * ~dones)
        
        # Losses
        loss_win = F.mse_loss(current_q_win, target_q_win)
        loss_leak = F.mse_loss(current_q_leak, target_q_leak)
        
        loss = loss_win + loss_leak
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
            
        return loss_value
        
    def get_average_policy_loss(self, window: int = 100) -> float:
        if not self.policy_losses:
            return 0.0
        recent_losses = self.policy_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
        
    def save_model(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'critic': self.critic.state_dict(),
        }, path)
        
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
        self.update_target_network()
