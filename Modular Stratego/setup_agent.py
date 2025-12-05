"""
Setup Agent for Stratego - Places pieces on the board before the game starts
Rainbow DQN Architecture: Noisy Nets, Dueling Heads, C51 Distributional RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from typing import List, Tuple, Optional, NamedTuple
from collections import deque
from piece import PieceType
from critic import SetupExploitabilityCritic

# C51 Hyperparameters for Setup Agent
# Setup rewards typically range from -2 to +10, so we use a tighter range than the main agent
V_MIN = -10.0
V_MAX = 20.0
NUM_ATOMS = 51


class SetupExperience(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.
    Factorized Gaussian Noise.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized noise: outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class RainbowSetupNetwork(nn.Module):
    """
    Rainbow DQN Network for Setup Agent
    - CNN Feature Extractor
    - Dueling Architecture (Value + Advantage streams)
    - Noisy Nets for exploration
    - C51 Distributional Output (51 atoms)
    
    Input: (batch, 3, 10, 10)
        - Channel 0: Board state (normalized piece values)
        - Channel 1: Valid positions mask
        - Channel 2: Piece type being placed (constant plane)
    Output: log_probs (batch, 100, 51) - distributional Q over 100 positions
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (3, 10, 10), 
                 output_size: int = 100, num_atoms: int = 51):
        super(RainbowSetupNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size  # 100 board positions
        self.num_atoms = num_atoms
        
        # CNN Layers (Feature Extractor) - same backbone as original
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size: 64 * 10 * 10 = 6400
        self.flatten_size = 64 * 10 * 10
        
        # Dueling Architecture with Noisy Nets
        # Value stream: State -> Value Distribution (batch, 1, num_atoms)
        self.value_fc = NoisyLinear(self.flatten_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        # Advantage stream: State -> Advantage Distribution (batch, 100, num_atoms)
        self.advantage_fc = NoisyLinear(self.flatten_size, 512)
        self.advantage_out = NoisyLinear(512, output_size * num_atoms)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (batch, 3, 10, 10)
        Returns:
            log_probs: Log probabilities of shape (batch, 100, num_atoms)
        """
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Flatten to (batch, 6400)
        
        # Dueling Heads
        # Value stream
        val_hidden = F.relu(self.value_fc(x))
        val_out = self.value_out(val_hidden)  # (batch, num_atoms)
        val_out = val_out.view(batch_size, 1, self.num_atoms)  # (batch, 1, num_atoms)
        
        # Advantage stream
        adv_hidden = F.relu(self.advantage_fc(x))
        adv_out = self.advantage_out(adv_hidden)  # (batch, output_size * num_atoms)
        adv_out = adv_out.view(batch_size, self.output_size, self.num_atoms)  # (batch, 100, num_atoms)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # In Distributional RL, we combine the logits
        adv_mean = adv_out.mean(dim=1, keepdim=True)  # Mean over actions
        
        # Unnormalized logits
        q_logits = val_out + (adv_out - adv_mean)
        
        # Softmax to get probabilities (Log Softmax for stability with cross-entropy loss)
        log_probs = F.log_softmax(q_logits, dim=2)  # Softmax over atoms dimension
        
        return log_probs
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()


class SetupAgent:
    """
    Agent that learns to place pieces on the board using Rainbow DQN.
    Features:
    - Noisy Nets for exploration (no epsilon-greedy)
    - Dueling Architecture
    - C51 Distributional RL
    """
    
    def __init__(self, player_id: int, device, 
                 lr: float = 0.0001, gamma: float = 0.95,
                 buffer_size: int = 10000, batch_size: int = 32,
                 v_min: float = V_MIN, v_max: float = V_MAX, 
                 num_atoms: int = NUM_ATOMS):
        """
        Initialize the Rainbow DQN setup agent
        """
        self.player_id = player_id
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        # C51 Distributional RL parameters
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms, device=device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Neural networks (Rainbow DQN)
        input_shape = (3, 10, 10)
        output_size = 100  # 100 positions (10x10 board)
        
        self.q_network = RainbowSetupNetwork(input_shape, output_size, num_atoms).to(device)
        self.target_network = RainbowSetupNetwork(input_shape, output_size, num_atoms).to(device)
        
        # Exploitability Critic (unchanged)
        self.critic = SetupExploitabilityCritic(input_shape=input_shape, output_size=output_size).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01)
        self.critic_loss_fn = nn.CrossEntropyLoss()
        self.critic_weight = 0.1
        
        # Enable cuDNN benchmarking for CUDA
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        self.memory = deque(maxlen=buffer_size)
        self.current_episode_memory = []  # Temporary storage for current episode steps
        self.policy_losses = []
        self.reward_history = []
        self.episode_rewards = []
        
        self.update_target_network()
        
    def reset(self):
        """Reset the setup agent"""
        input_shape = (3, 10, 10)
        output_size = 100
        
        self.q_network = RainbowSetupNetwork(input_shape, output_size, self.num_atoms).to(self.device)
        self.target_network = RainbowSetupNetwork(input_shape, output_size, self.num_atoms).to(self.device)
        
        self.critic = SetupExploitabilityCritic(input_shape=input_shape, output_size=output_size).to(self.device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=0.01)
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.memory.clear()
        self.current_episode_memory = []
        self.policy_losses = []
        self.reward_history = []
        self.episode_rewards = []
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def reset_noise(self):
        """Reset noise in Noisy layers for exploration"""
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
    def get_state_representation(self, current_board: torch.Tensor, piece_to_place: PieceType, 
                                  available_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Convert board and piece to CNN state representation (3, 10, 10)
        """
        # Channel 0: Board state (normalized)
        board_channel = current_board.clone().float() / 12.0  # Max piece value is roughly 12
        
        # Channel 1: Valid positions mask
        mask_channel = torch.zeros((10, 10), device=self.device)
        for r, c in available_positions:
            mask_channel[r, c] = 1.0
            
        # Channel 2: Piece type being placed
        piece_channel = torch.full((10, 10), float(piece_to_place.value) / 12.0, device=self.device)
        
        state = torch.stack([board_channel, mask_channel, piece_channel])
        return state.unsqueeze(0)  # Add batch dimension: (1, 3, 10, 10)
        
    def place_pieces(self, pieces: List[PieceType], 
                     available_positions: List[Tuple[int, int]]) -> List[Tuple[PieceType, Tuple[int, int]]]:
        """
        Place pieces on the board using the agent's policy.
        Uses Noisy Nets for exploration instead of epsilon-greedy.
        """
        self.current_episode_memory = []  # Clear previous episode memory
        
        if len(pieces) != 40:
            # Fallback for safety
            random.shuffle(available_positions)
            return list(zip(pieces, available_positions))
        
        # Initialize board state (empty)
        current_board = torch.zeros((10, 10), device=self.device)
        
        placement = []
        remaining_positions = available_positions.copy()
        remaining_pieces = pieces.copy()
        
        # Reset noise at the start of each placement episode for fresh exploration
        self.reset_noise()
        
        for i in range(40):
            piece = remaining_pieces[0]
            remaining_pieces = remaining_pieces[1:]
            
            # Get state
            state = self.get_state_representation(current_board, piece, remaining_positions)
            
            # Action selection using distributional Q-values
            with torch.no_grad():
                # Get distributional output: (1, 100, 51)
                log_probs = self.q_network(state)
                probs = log_probs.exp()
                
                # Calculate expected Q-values: sum over atoms (probs * support)
                expected_q = (probs * self.support).sum(dim=2)  # (1, 100)
                expected_q = expected_q.squeeze(0)  # (100,)
                
                # Create mask for valid positions (set invalid to -inf)
                mask = torch.full((100,), -float('inf'), device=self.device)
                for r, c in remaining_positions:
                    mask[r * 10 + c] = 0
                
                # Apply mask and select best action
                masked_q = expected_q + mask
                best_idx = torch.argmax(masked_q).item()
                position = (best_idx // 10, best_idx % 10)
            
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
                next_state = torch.zeros_like(state)  # Placeholder
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
            self.remember(
                step['state'],
                step['action'],
                total_reward,
                step['next_state'],
                step['done']
            )
        self.current_episode_memory = []  # Clear after storing
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(SetupExperience(state, action, reward, next_state, done))
        self.reward_history.append(reward)
        if done:
            self.episode_rewards.append(reward)
        
    def replay(self) -> Optional[float]:
        """
        Train the model using C51 Distributional Loss with Exploitability Critic.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([e.state for e in batch])  # (B, 3, 10, 10)
        next_states = torch.cat([e.next_state for e in batch])
        
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        
        # --- Train Critic (unchanged) ---
        critic_logits = self.critic(states)
        critic_loss = self.critic_loss_fn(critic_logits, actions)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # --- Calculate Penalty (unchanged) ---
        with torch.no_grad():
            probs = F.softmax(self.critic(states), dim=1)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            penalty = self.critic_weight * action_probs
            
        penalized_rewards = rewards - penalty
        
        # --- C51 Distributional RL Target Calculation ---
        with torch.no_grad():
            # Reset noise for consistent evaluation
            self.target_network.reset_noise()
            
            # 1. Select best action in next state using Online Network (Double DQN)
            next_log_probs_online = self.q_network(next_states)
            next_probs_online = next_log_probs_online.exp()
            next_q_values_online = (next_probs_online * self.support).sum(dim=2)  # (batch, 100)
            next_actions = next_q_values_online.argmax(dim=1)  # (batch,)
            
            # 2. Get distribution of best action from Target Network
            next_log_probs_target = self.target_network(next_states)
            next_probs_target = next_log_probs_target.exp()
            
            # Gather distribution for the selected actions: (batch, 51)
            next_action_probs = next_probs_target.gather(
                1, next_actions.view(-1, 1, 1).expand(-1, -1, self.num_atoms)
            ).squeeze(1)
            
            # 3. Project Distribution (Categorical Algorithm)
            # T_z = r + gamma * z (if not done)
            not_dones = (~dones).to(torch.float32)
            T_z = penalized_rewards.unsqueeze(1) + (
                not_dones.unsqueeze(1) * self.gamma * self.support.unsqueeze(0)
            )
            T_z = T_z.clamp(min=self.v_min, max=self.v_max)
            
            # Compute L2 projection of T_z onto support
            b = (T_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge case where l == u
            l = l.clamp(0, self.num_atoms - 1)
            u = u.clamp(0, self.num_atoms - 1)
            
            # Distribute probability mass
            batch_size = len(batch)
            m = torch.zeros(batch_size, self.num_atoms, device=self.device)
            
            # Offset for scatter_add
            offset = torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, 
                device=self.device
            ).long().unsqueeze(1).expand(batch_size, self.num_atoms)
            
            # m_l = m_l + p(s', a') * (u - b)
            # m_u = m_u + p(s', a') * (b - l)
            m.view(-1).scatter_add_(
                0, (l + offset).view(-1), 
                (next_action_probs * (u.float() - b)).view(-1)
            )
            m.view(-1).scatter_add_(
                0, (u + offset).view(-1), 
                (next_action_probs * (b - l.float())).view(-1)
            )
        
        # --- Calculate Cross-Entropy Loss ---
        # Get current log probabilities
        current_log_probs = self.q_network(states)  # (batch, 100, 51)
        
        # Gather log probs for the actions taken
        action_log_probs = current_log_probs.gather(
            1, actions.view(-1, 1, 1).expand(-1, -1, self.num_atoms)
        ).squeeze(1)  # (batch, 51)
        
        # Cross Entropy Loss: - Sum(m * log_p)
        loss = -(m * action_log_probs).sum(dim=1).mean()
        
        # Safety check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Warning: NaN/Inf detected in setup agent loss. Skipping update.")
            self.optimizer.zero_grad()
            return None
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        
        return loss_value
        
    def get_average_policy_loss(self, window: int = 100) -> float:
        if not self.policy_losses:
            return 0.0
        recent_losses = self.policy_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
        
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'v_min': self.v_min,
            'v_max': self.v_max,
            'num_atoms': self.num_atoms,
        }, path)
        
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle both old (ConvSetupDQN) and new (RainbowSetupNetwork) checkpoints
        try:
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
        except RuntimeError as e:
            # Old checkpoint format - reinitialize networks instead of failing
            print(f"⚠️  Old checkpoint format detected, initializing fresh networks: {e}")
            return
            
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        self.update_target_network()
