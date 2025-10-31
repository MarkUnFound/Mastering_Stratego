# stratego_modular/dcfr.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# Try both absolute and relative imports to support different import scenarios
try:
    # When imported as a module
    from .game_state import GameState
except ImportError:
    # When run directly
    from game_state import GameState

class DeepRegretNetwork(nn.Module):
    """Neural network to approximate regrets for DeepCFR."""
    
    def __init__(self, input_size, hidden_size, output_size, device):
        super(DeepRegretNetwork, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)
        
    def forward(self, x):
        return self.network(x)

class DeepStrategyNetwork(nn.Module):
    """Neural network to approximate average strategy for DeepCFR."""
    
    def __init__(self, input_size, hidden_size, output_size, device):
        super(DeepStrategyNetwork, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        ).to(device)
        
    def forward(self, x):
        return self.network(x)

class InfoSetEncoder:
    """Encodes information sets for neural network input."""
    
    def __init__(self, board_size=10, device='cuda'):
        self.board_size = board_size
        self.device = device
        # Stratego has 12 piece types (including empty and lakes)
        self.piece_type_dim = 13
        # Board features, piece types, turn count, move history features
        self.input_dim = board_size * board_size * self.piece_type_dim + 20
        
    def encode_infoset(self, game_state: GameState) -> torch.Tensor:
        """Encode the information set from a game state."""
        board = game_state.board
        current_player = game_state.current_player
        
        # One-hot encode board state (piece types)
        board_features = torch.zeros(self.board_size, self.board_size, 
                                    self.piece_type_dim, device=self.device)
        
        # For each position, encode the piece type
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece_value = board[r, c].item()
                if piece_value != 0:  # Not empty
                    piece_type = abs(piece_value)  # Get piece type regardless of player
                    player_owned = (piece_value * current_player) > 0  # Is it current player's piece?
                    
                    # Set appropriate feature
                    if player_owned:
                        board_features[r, c, piece_type] = 1.0
                    else:
                        # For opponent pieces with unknown type
                        # Check if piece is revealed
                        revealed_pieces = game_state.get_revealed_pieces(current_player)
                        if (r, c) in revealed_pieces:
                            board_features[r, c, revealed_pieces[(r, c)]] = 1.0
                        else:
                            # Unknown opponent piece
                            board_features[r, c, self.piece_type_dim - 1] = 1.0
                else:
                    # Empty square
                    board_features[r, c, 0] = 1.0
        
        # Flatten board features
        flat_board_features = board_features.flatten()
        
        # Add turn count and other game state features
        extra_features = torch.tensor(
            [game_state.turn_count / 500.0],  # Normalize turn count
            device=self.device
        )
        
        # Concatenate all features
        return torch.cat([flat_board_features, extra_features])
        
    def get_input_dim(self) -> int:
        return self.input_dim

class DeepCFR:
    """Deep Counterfactual Regret Minimization with external sampling MCCFR."""
    
    def __init__(self, env, device='cuda', hidden_size=256):
        self.env = env
        self.device = device
        
        # Information set encoder
        self.encoder = InfoSetEncoder(device=device)
        input_dim = self.encoder.get_input_dim()
        
        # Determine max output size (max number of actions)
        self.max_actions = 400  # Stratego has at most ~400 possible moves
        
        # Create neural networks
        self.regret_networks = [
            DeepRegretNetwork(input_dim, hidden_size, self.max_actions, device),
            DeepRegretNetwork(input_dim, hidden_size, self.max_actions, device)
        ]
        self.strategy_networks = [
            DeepStrategyNetwork(input_dim, hidden_size, self.max_actions, device),
            DeepStrategyNetwork(input_dim, hidden_size, self.max_actions, device)
        ]
        
        # Optimizers
        self.regret_optimizers = [
            optim.Adam(self.regret_networks[0].parameters(), lr=0.001),
            optim.Adam(self.regret_networks[1].parameters(), lr=0.001)
        ]
        self.strategy_optimizers = [
            optim.Adam(self.strategy_networks[0].parameters(), lr=0.001),
            optim.Adam(self.strategy_networks[1].parameters(), lr=0.001)
        ]
        
        # Training data
        self.regret_memory = [[], []]
        self.strategy_memory = [[], []]
        
        # CFR parameters
        self.iteration = 0
        self.regret_epsilon = 0.01  # Exploration constant
        
    def _get_strategy(self, game_state: GameState, valid_actions: List[Tuple]):
        """Get strategy for the current player at this information set."""
        player_idx = 0 if game_state.current_player == 1 else 1
        info_set_features = self.encoder.encode_infoset(game_state).unsqueeze(0)
        
        # Get regrets from network
        regrets = self.regret_networks[player_idx](info_set_features).squeeze()
        
        # Create a mask for valid actions
        action_mask = torch.zeros(self.max_actions, device=self.device)
        for i, action in enumerate(valid_actions):
            if i < self.max_actions:
                action_mask[i] = 1.0
        
        # Apply valid action mask
        masked_regrets = regrets * action_mask
        
        # Compute positive regrets
        positive_regrets = torch.maximum(masked_regrets, torch.zeros_like(masked_regrets))
        
        # Add epsilon exploration
        regret_sum = positive_regrets.sum().item()
        if regret_sum <= 0:
            # If all regrets are zero, use uniform strategy
            strategy = action_mask / action_mask.sum() if action_mask.sum() > 0 else action_mask
        else:
            # Normalize positive regrets to get strategy
            strategy = positive_regrets / regret_sum
            
            # Add epsilon exploration
            strategy = (1 - self.regret_epsilon) * strategy + self.regret_epsilon * action_mask / action_mask.sum()
        
        return strategy
        
    def _mccfr_external_sampling(self, game_state: GameState, player: int, depth: int = 0):
        """External sampling MCCFR for a single player traversal."""
        if game_state.game_over or depth > 100:  # Limit depth to avoid infinite recursion
            # Return terminal utilities
            if game_state.winner == 0:  # Draw
                return 0.0
            elif game_state.winner == player:  # Win
                return 1.0
            else:  # Loss
                return -1.0
        
        # Get valid actions
        self.env.current_player = game_state.current_player
        valid_actions = self.env.get_valid_moves()
        
        if len(valid_actions) == 0:
            # No valid moves, player loses
            return -1.0
        
        current_player = game_state.current_player
        is_traverser = (current_player == player)
        
        if is_traverser:
            # This is the traverser's turn - use regret matching to sample an action
            strategy = self._get_strategy(game_state, valid_actions)
            
            # Initialize action values
            action_values = torch.zeros(len(valid_actions), device=self.device)
            action_regrets = torch.zeros(len(valid_actions), device=self.device)
            
            # For each action, compute counterfactual value
            for i, action in enumerate(valid_actions):
                if i >= self.max_actions:
                    break
                    
                # Clone environment to simulate action
                next_state, _, done, _ = self.env.step(action)
                
                # Recursive call to get action value
                action_values[i] = self._mccfr_external_sampling(next_state, player, depth + 1)
            
            # Calculate expected value
            expected_value = 0.0
            for i in range(len(valid_actions)):
                if i < self.max_actions:
                    expected_value += strategy[i].item() * action_values[i].item()
            
            # Calculate regrets
            for i in range(len(valid_actions)):
                if i < self.max_actions:
                    action_regrets[i] = action_values[i].item() - expected_value
            
            # Store samples for network training
            info_set_features = self.encoder.encode_infoset(game_state)
            player_idx = 0 if player == 1 else 1
            
            # Add to regret memory
            self.regret_memory[player_idx].append({
                'features': info_set_features,
                'actions': valid_actions[:self.max_actions],
                'regrets': action_regrets[:self.max_actions],
                'legal_mask': torch.ones(len(valid_actions), device=self.device)[:self.max_actions]
            })
            
            # Add to strategy memory
            self.strategy_memory[player_idx].append({
                'features': info_set_features,
                'actions': valid_actions[:self.max_actions],
                'strategy': strategy[:len(valid_actions)],
                'legal_mask': torch.ones(len(valid_actions), device=self.device)[:self.max_actions]
            })
            
            return expected_value
        else:
            # Opponent's turn - sample a single action randomly
            action_idx = random.randint(0, len(valid_actions) - 1)
            action = valid_actions[action_idx]
            
            # Simulate action
            next_state, _, done, _ = self.env.step(action)
            
            # Recursive call
            return self._mccfr_external_sampling(next_state, player, depth + 1)
    
    def _train_networks(self):
        """Train neural networks on collected samples."""
        for player_idx in range(2):
            # Train regret network
            if self.regret_memory[player_idx]:
                self.regret_networks[player_idx].train()
                
                # Create batch
                batch_size = min(256, len(self.regret_memory[player_idx]))
                batch_indices = random.sample(range(len(self.regret_memory[player_idx])), batch_size)
                
                features = torch.stack([self.regret_memory[player_idx][i]['features'] for i in batch_indices])
                targets = torch.zeros(batch_size, self.max_actions, device=self.device)
                masks = torch.zeros(batch_size, self.max_actions, device=self.device)
                
                for batch_idx, idx in enumerate(batch_indices):
                    sample = self.regret_memory[player_idx][idx]
                    for action_idx, action in enumerate(sample['actions']):
                        if action_idx < self.max_actions:
                            targets[batch_idx, action_idx] = sample['regrets'][action_idx]
                            masks[batch_idx, action_idx] = 1.0
                
                # Forward pass
                self.regret_optimizers[player_idx].zero_grad()
                outputs = self.regret_networks[player_idx](features)
                
                # MSE loss on masked outputs
                loss = torch.sum(((outputs - targets) ** 2) * masks) / torch.sum(masks)
                loss.backward()
                self.regret_optimizers[player_idx].step()
                
            # Train strategy network
            if self.strategy_memory[player_idx]:
                self.strategy_networks[player_idx].train()
                
                # Create batch
                batch_size = min(256, len(self.strategy_memory[player_idx]))
                batch_indices = random.sample(range(len(self.strategy_memory[player_idx])), batch_size)
                
                features = torch.stack([self.strategy_memory[player_idx][i]['features'] for i in batch_indices])
                targets = torch.zeros(batch_size, self.max_actions, device=self.device)
                masks = torch.zeros(batch_size, self.max_actions, device=self.device)
                
                for batch_idx, idx in enumerate(batch_indices):
                    sample = self.strategy_memory[player_idx][idx]
                    for action_idx, action in enumerate(sample['actions']):
                        if action_idx < self.max_actions:
                            targets[batch_idx, action_idx] = sample['strategy'][action_idx]
                            masks[batch_idx, action_idx] = 1.0
                
                # Forward pass
                self.strategy_optimizers[player_idx].zero_grad()
                outputs = self.strategy_networks[player_idx](features)
                
                # Cross entropy loss on masked outputs
                masked_outputs = outputs * masks
                masked_targets = targets * masks
                loss = -torch.sum(masked_targets * torch.log(masked_outputs + 1e-8)) / torch.sum(masks)
                loss.backward()
                self.strategy_optimizers[player_idx].step()
    
    def train(self, iterations=10):
        """Train DeepCFR for a specified number of iterations."""
        for i in range(iterations):
            self.iteration += 1
            
            # Clear memories
            self.regret_memory = [[], []]
            self.strategy_memory = [[], []]
            
            # External sampling for player 1
            initial_state = self.env.reset()
            self._mccfr_external_sampling(initial_state, 1)
            
            # External sampling for player 2
            initial_state = self.env.reset()
            self._mccfr_external_sampling(initial_state, -1)
            
            # Train networks
            self._train_networks()
            
            print(f"Completed iteration {i+1}/{iterations}")
    
    def get_action(self, game_state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the best action according to the current strategy."""
        # Get valid actions
        self.env.current_player = game_state.current_player
        valid_actions = self.env.get_valid_moves()
        
        if not valid_actions:
            return None  # No valid actions
        
        # Get strategy
        strategy = self._get_strategy(game_state, valid_actions)
        
        # Sample action from strategy
        probs = strategy[:len(valid_actions)].cpu().numpy()
        action_idx = np.random.choice(len(valid_actions), p=probs)
        
        return valid_actions[action_idx]
    
    def get_nash_equilibrium(self, game_state: GameState) -> Dict[Tuple, float]:
        """Return a Nash equilibrium strategy for the current state."""
        # Get valid actions
        self.env.current_player = game_state.current_player
        valid_actions = self.env.get_valid_moves()
        
        if not valid_actions:
            return {}  # No valid actions
        
        # Get strategy
        strategy = self._get_strategy(game_state, valid_actions)
        
        # Convert to dictionary
        nash_strategy = {}
        for i, action in enumerate(valid_actions):
            if i < self.max_actions:
                nash_strategy[action] = strategy[i].item()
        
        return nash_strategy
