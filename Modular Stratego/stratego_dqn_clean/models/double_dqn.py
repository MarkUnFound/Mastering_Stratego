"""
Double DQN Implementation for Stratego
Strict implementation per Van Hasselt et al. (2015) arXiv:1509.06461

Loss: L = (y - Q(s,a;θ))² where y = r + γ Q(s', argmax_a' Q(s',a';θ); θ⁻)

Key constraints:
- NO distributional atoms (NOT C51)
- NO dueling streams
- NO noisy networks
- Single Q-value head only
- ε-greedy exploration
- Target network hard update every C steps OR soft update τ ≤ 0.005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from collections import deque
import random

from .resnet_backbone import ResNetBackbone, ResNetBackboneWithCheckpointing
from .aaren import AAREN


class DoubleDQN(nn.Module):
    """
    Double DQN Network with AAREN integration.
    
    Architecture:
    - AAREN: 64-dim history embedding
    - Input: 15 board channels + 64 AAREN = 79 channels
    - ResNet Backbone: 4-layer (64→128→128→64)
    - Q-Head: Flatten → FC(512) → FC(action_dim)
    
    Output: Q-values for 100 actions (heuristically filtered)
    """
    
    def __init__(
        self, 
        action_dim: int = 100,
        board_channels: int = 15,
        aaren_dim: int = 64,
        use_checkpointing: bool = False
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.board_channels = board_channels
        self.aaren_dim = aaren_dim
        self.total_channels = board_channels + aaren_dim  # 79
        
        # ResNet backbone (with optional gradient checkpointing)
        if use_checkpointing:
            self.backbone = ResNetBackboneWithCheckpointing(input_channels=self.total_channels)
        else:
            self.backbone = ResNetBackbone(input_channels=self.total_channels)
        
        # Q-value head (standard, NOT dueling)
        # Backbone output: (B, 64, 10, 10) → Flatten → 6400
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
        # Initialize Q-head
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 79, 10, 10) - board + AAREN embedding
            
        Returns:
            q_values: (B, action_dim) Q-values
        """
        # ResNet backbone
        features = self.backbone(x)  # (B, 64, 10, 10)
        
        # Flatten
        x = features.flatten(1)  # (B, 6400)
        
        # Q-value head
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)  # (B, action_dim)
        
        return q_values
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing in backbone."""
        if hasattr(self.backbone, 'enable_checkpointing'):
            self.backbone.enable_checkpointing()
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing in backbone."""
        if hasattr(self.backbone, 'disable_checkpointing'):
            self.backbone.disable_checkpointing()
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DoubleDQNAgent:
    """
    Double DQN Agent with AAREN integration.
    
    Features:
    - Online + Target network (Double DQN)
    - AAREN for history embedding
    - ε-greedy exploration
    - Hard target update every C steps
    - MSE loss (NOT distributional)
    
    Memory constraints:
    - Batch size ≤ 32
    - 6GB VRAM ceiling
    """
    
    # Hyperparameters
    BATCH_SIZE_MAX = 32
    TARGET_UPDATE_FREQ = 1000  # Hard update every C steps
    SOFT_UPDATE_TAU = 0.005    # Alternative: soft update
    GAMMA = 0.99
    LR = 1e-4
    
    # Exploration (ε-greedy)
    EPSILON_START = 0.5
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 100000
    
    def __init__(
        self,
        player_id: int,
        device: torch.device,
        action_dim: int = 100,
        batch_size: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        use_soft_update: bool = False,  # False = hard update
        aaren_input_dim: int = 24,  # Action feature dimension
    ):
        self.player_id = player_id
        self.device = device
        self.action_dim = action_dim
        self.batch_size = min(batch_size, self.BATCH_SIZE_MAX)
        self.lr = lr
        self.gamma = gamma
        self.use_soft_update = use_soft_update
        
        # Networks
        self.aaren = AAREN(input_dim=aaren_input_dim, hidden_dim=64, device=device).to(device)
        self.online_network = DoubleDQN(action_dim=action_dim, use_checkpointing=False).to(device)
        self.target_network = DoubleDQN(action_dim=action_dim, use_checkpointing=False).to(device)
        
        # Initialize target network with online network weights
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Target never trains
        
        # Optimizer (fresh Adam, no excluded parameters)
        all_params = list(self.aaren.parameters()) + list(self.online_network.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
        # Training state
        self.step_count = 0
        self.epsilon = self.EPSILON_START
        self.training_losses = []
        
        # AAREN state for recurrent inference
        self.aaren_state = None
        
        print(f"[DoubleDQNAgent] Initialized for Player {player_id}")
        print(f"  - AAREN params: {self.aaren.count_parameters() if hasattr(self.aaren, 'count_parameters') else sum(p.numel() for p in self.aaren.parameters()):,}")
        print(f"  - DQN params: {self.online_network.count_parameters():,}")
        print(f"  - Total params: {sum(p.numel() for p in all_params):,}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Target update: {'Soft τ=' + str(self.SOFT_UPDATE_TAU) if use_soft_update else 'Hard every ' + str(self.TARGET_UPDATE_FREQ)}")
    
    def reset(self):
        """Reset agent state for new episode."""
        self.aaren_state = None
    
    def update_epsilon(self):
        """Decay epsilon linearly."""
        progress = min(1.0, self.step_count / self.EPSILON_DECAY_STEPS)
        self.epsilon = self.EPSILON_START + progress * (self.EPSILON_END - self.EPSILON_START)
    
    def get_state_with_aaren(
        self, 
        board_tensor: torch.Tensor,
        action_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine board tensor with AAREN embedding.
        
        Args:
            board_tensor: (B, 15, 10, 10) board state
            action_features: (B, seq_len, input_dim) action history
            
        Returns:
            combined: (B, 79, 10, 10) board + AAREN
        """
        batch_size = board_tensor.size(0)
        
        # Check if already processed (79 channels)
        if board_tensor.size(1) >= 79:
            return board_tensor
        
        if action_features is None:
            # No history yet - use AAREN's learnable default embedding (not zeros)
            # This prevents sparse death detection from halting training
            aaren_embedding = self.aaren.get_default_embedding(batch_size)
        else:
            # Get AAREN embedding
            with torch.cuda.amp.autocast(enabled=True):  # Mixed precision
                aaren_embedding = self.aaren(action_features)  # (B, 64)
        
        # Expand AAREN embedding to spatial dimensions
        # (B, 64) → (B, 64, 10, 10)
        aaren_spatial = aaren_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 10, 10)
        
        # Concatenate: (B, 15, 10, 10) + (B, 64, 10, 10) → (B, 79, 10, 10)
        combined = torch.cat([board_tensor, aaren_spatial], dim=1)
        
        return combined
    
    @torch.no_grad()
    def act(
        self,
        board_tensor: torch.Tensor,
        valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        action_features: Optional[torch.Tensor] = None,
        greedy: bool = False
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]:
        """
        Select action using ε-greedy policy.
        
        Args:
            board_tensor: (15, 10, 10) board state
            valid_moves: List of valid (from, to) tuples
            action_features: Optional action history for AAREN
            greedy: If True, always pick best action (evaluation mode)
            
        Returns:
            action: Selected move tuple
            action_idx: Index in valid_moves
        """
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # ε-greedy exploration
        if not greedy and random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(len(valid_moves))
            return valid_moves[action_idx], action_idx
        
        # Greedy action
        self.online_network.eval()
        
        # Prepare input
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        board_tensor = board_tensor.to(self.device)
        
        # Get state with AAREN embedding
        state = self.get_state_with_aaren(board_tensor, action_features)
        
        # Get Q-values
        q_values = self.online_network(state)  # (1, action_dim)
        q_values = q_values.squeeze(0)  # (action_dim,)
        
        # Mask invalid actions
        num_valid = min(len(valid_moves), self.action_dim)
        valid_q = q_values[:num_valid]
        
        # Select best valid action
        action_idx = valid_q.argmax().item()
        
        self.online_network.train()
        
        return valid_moves[action_idx], action_idx

    @torch.no_grad()
    def act_batch(
        self,
        board_tensor: torch.Tensor,
        valid_moves: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
        action_features: Optional[torch.Tensor] = None,
        greedy: bool = False
    ) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[int]]:
        """
        Select actions for a batch of environments.
        
        Args:
            board_tensor: (B, 15, 10, 10) board states
            valid_moves: List of valid move lists for each env
            action_features: Optional (B, seq, dim)
            greedy: If True, disable exploration
            
        Returns:
            actions: List of selected move tuples
            action_indices: List of selected indices
        """
        batch_size = len(valid_moves)
        actions = []
        action_indices = []
        
        # Determine exploration per env
        if greedy:
            exploring = [False] * batch_size
        else:
            exploring = [random.random() < self.epsilon for _ in range(batch_size)]
            
        # Optimization: if all exploring, skip network
        if all(exploring):
             for i in range(batch_size):
                 if not valid_moves[i]:
                     # Should not happen in active env
                     actions.append(None) 
                     action_indices.append(-1)
                     continue
                 idx = random.randrange(len(valid_moves[i]))
                 actions.append(valid_moves[i][idx])
                 action_indices.append(idx)
             return actions, action_indices

        # Forward pass for greedy items
        self.online_network.eval()
        
        # Ensure tensor on device
        board_tensor = board_tensor.to(self.device)
        
        state = self.get_state_with_aaren(board_tensor, action_features)
        q_values = self.online_network(state) # (B, action_dim)
        
        for i in range(batch_size):
            vm = valid_moves[i]
            if not vm:
                 actions.append(None)
                 action_indices.append(-1)
                 continue

            if exploring[i]:
                 idx = random.randrange(len(vm))
                 actions.append(vm[idx])
                 action_indices.append(idx)
            else:
                 # Mask invalid actions
                 num_valid = min(len(vm), self.action_dim)
                 valid_q = q_values[i, :num_valid]
                 idx = valid_q.argmax().item()
                 actions.append(vm[idx])
                 action_indices.append(idx)
                 
        self.online_network.train()
        return actions, action_indices
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Double DQN loss.
        
        L = (y - Q(s,a;θ))² where y = r + γ Q(s', argmax_a' Q(s',a';θ); θ⁻)
        
        Args:
            states: (B, 79, 10, 10)
            actions: (B,) action indices
            rewards: (B,)
            next_states: (B, 79, 10, 10)
            dones: (B,) done flags
            
        Returns:
            loss: Scalar MSE loss
        """
        batch_size = states.size(0)
        
        # Current Q-values: Q(s, a; θ)
        current_q = self.online_network(states)  # (B, action_dim)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Double DQN target computation
        with torch.no_grad():
            # Step 1: argmax_a' Q(s', a'; θ) using ONLINE network
            next_q_online = self.online_network(next_states)  # (B, action_dim)
            best_actions = next_q_online.argmax(dim=1)  # (B,)
            
            # Step 2: Q(s', best_action; θ⁻) using TARGET network
            next_q_target = self.target_network(next_states)  # (B, action_dim)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # (B,)
            
            # Target: y = r + γ Q(s', argmax Q; θ⁻) * (1 - done)
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # MSE Loss
        loss = F.mse_loss(current_q, target)
        
        return loss
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            states: (B, 15, 10, 10) or (B, 79, 10, 10) board states
            actions: (B,)
            rewards: (B,)
            next_states: (B, 15, 10, 10) or (B, 79, 10, 10)
            dones: (B,)
            
        Returns:
            loss: Training loss value
        """
        self.online_network.train()
        
        # Add AAREN embeddings if states are 15-channel
        if states.size(1) == 15:
            states = self.get_state_with_aaren(states, action_features=None)
            next_states = self.get_state_with_aaren(next_states, action_features=None)
        
        # Compute loss
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(
            list(self.aaren.parameters()) + list(self.online_network.parameters()),
            max_norm=10.0
        )
        
        self.optimizer.step()
        
        # Update step count
        self.step_count += 1
        
        # Update epsilon
        self.update_epsilon()
        
        # Update target network
        if self.use_soft_update:
            self._soft_update_target()
        else:
            if self.step_count % self.TARGET_UPDATE_FREQ == 0:
                self._hard_update_target()
        
        # Track loss
        loss_val = loss.item()
        self.training_losses.append(loss_val)
        
        return loss_val
    
    def _hard_update_target(self):
        """Hard update: copy online weights to target."""
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def _soft_update_target(self):
        """Soft update: θ⁻ ← τθ + (1-τ)θ⁻"""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.SOFT_UPDATE_TAU * online_param.data + 
                (1 - self.SOFT_UPDATE_TAU) * target_param.data
            )
    
    def get_online_target_distance(self) -> float:
        """Compute MSE between online and target network parameters."""
        total_mse = 0.0
        count = 0
        for online_param, target_param in zip(
            self.online_network.parameters(),
            self.target_network.parameters()
        ):
            total_mse += F.mse_loss(online_param, target_param).item()
            count += 1
        return total_mse / count if count > 0 else 0.0
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'aaren_state_dict': self.aaren.state_dict(),
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.aaren.load_state_dict(checkpoint['aaren_state_dict'])
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        
        # Load full training state if available
        if 'target_network_state_dict' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'step_count' in checkpoint:
            self.step_count = checkpoint['step_count']
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']

    def export(self, path: str):
        """
        Export inference-ready model (no optimizer/target net).
        
        Args:
            path: Export path
        """
        torch.save({
            'aaren_state_dict': self.aaren.state_dict(),
            'online_network_state_dict': self.online_network.state_dict(),
            'config': {
                'action_dim': self.action_dim,
                'aaren_dim': 64,  # Hardcoded per Phase 1
                'board_channels': 15
            },
            'timestamp': str(np.datetime64('now'))
        }, path)
