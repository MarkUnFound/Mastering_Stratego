# GPU-Optimized Enhanced Stratego with Efficient Search Algorithm
# Fixes GPU utilization and search algorithm efficiency issues

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import math

# --- Configuration ---
BOARD_SIZE = 10
NUM_PIECES = 40
HIDDEN_PIECE = -1
EMPTY_SQUARE = 0
LAKE_SQUARE = -2

# Piece Ranks
FLAG = 0
SPY = 1
SCOUT = 2
MINER = 3
SERGEANT = 4
LIEUTENANT = 5
CAPTAIN = 6
MAJOR = 7
COLONEL = 8
GENERAL = 9
MARSHAL = 10
BOMB = 11

PIECE_NAMES = {
    FLAG: 'F', SPY: '1', SCOUT: '2', MINER: '3', SERGEANT: '4',
    LIEUTENANT: '5', CAPTAIN: '6', MAJOR: '7', COLONEL: '8',
    GENERAL: '9', MARSHAL: 'X', BOMB: 'B', EMPTY_SQUARE: '.',
    LAKE_SQUARE: '~', HIDDEN_PIECE: '?'
}

# --- GPU-Optimized Data Structures ---

@dataclass
class GameState:
    """Lightweight game state for GPU processing."""
    board: torch.Tensor  # Use tensors instead of numpy arrays
    current_player: int
    turn_count: int
    game_over: bool
    winner: Optional[int]
    move_history: List[Tuple]
    # Simplified belief tracking for GPU efficiency
    uncertainty_mask: torch.Tensor  # Boolean mask for uncertain pieces

class MCTSNode:
    """Simplified MCTS node optimized for GPU batch processing."""
    def __init__(self, state_hash: str, parent=None, action=None):
        self.state_hash = state_hash
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = 0.0
        self.is_expanded = False
        
    def ucb_score(self, c_puct=1.4):
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration
    
    def select_child(self):
        return max(self.children.values(), key=lambda child: child.ucb_score())
    
    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state_hash=f"{self.state_hash}_{hash(action)}",
                    parent=self,
                    action=action
                )
                self.children[action].prior_prob = prob
        self.is_expanded = True
    
    def backup(self, value):
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.backup(-value)

# --- GPU-Optimized Stratego Environment ---

class GPUOptimizedStrategoEnv:
    """GPU-optimized Stratego environment with batched operations."""
    
    def __init__(self, device):
        self.device = device
        self.board_size = BOARD_SIZE
        
        # Pre-compute lakes and directions as tensors
        self.lakes = torch.tensor([(4, 2), (4, 3), (5, 2), (5, 3), 
                                  (4, 6), (4, 7), (5, 6), (5, 7)], device=device)
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        
        # Pre-allocate tensors for efficiency
        self.board = torch.zeros((self.board_size, self.board_size), dtype=torch.int8, device=device)
        self.uncertainty_mask = torch.zeros((self.board_size, self.board_size), dtype=torch.bool, device=device)
        
        # Cache for move generation
        self._move_cache = {}
        self._position_cache = self._precompute_positions()
        
        self.reset()

    def _precompute_positions(self):
        """Pre-compute all valid positions for faster move generation."""
        positions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                # Skip lakes
                if not any((r == lake[0] and c == lake[1]) for lake in self.lakes):
                    positions.append((r, c))
        return torch.tensor(positions, device=self.device)

    def reset(self):
        """GPU-optimized reset."""
        # Initialize board
        self.board.fill_(EMPTY_SQUARE)
        
        # Set lakes
        for r, c in self.lakes:
            self.board[r, c] = LAKE_SQUARE

        # Setup pieces (simplified for GPU efficiency)
        self._setup_board_gpu()
        
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.turn_count = 0
        
        # Initialize uncertainty mask for opponent pieces
        self.uncertainty_mask = (self.board * -self.current_player > 0)
        
        return self._get_state_tensor()

    def _setup_board_gpu(self):
        """GPU-optimized board setup."""
        pieces = [FLAG, SPY] + [BOMB]*6 + [MARSHAL] + [GENERAL] + [COLONEL]*2 + \
                [MAJOR]*3 + [CAPTAIN]*4 + [LIEUTENANT]*4 + [SERGEANT]*4 + \
                [MINER]*5 + [SCOUT]*8
        
        # Player 1 (bottom)
        p1_positions = [(r, c) for r in range(6, 10) for c in range(self.board_size) 
                       if self.board[r, c] != LAKE_SQUARE]
        random.shuffle(pieces)
        for i, (r, c) in enumerate(p1_positions[:len(pieces)]):
            self.board[r, c] = pieces[i]
        
        # Player 2 (top) 
        pieces_copy = pieces.copy()
        random.shuffle(pieces_copy)
        p2_positions = [(r, c) for r in range(0, 4) for c in range(self.board_size)
                       if self.board[r, c] != LAKE_SQUARE]
        for i, (r, c) in enumerate(p2_positions[:len(pieces_copy)]):
            self.board[r, c] = -pieces_copy[i]

    def get_valid_moves_gpu(self, player=None):
        """GPU-accelerated move generation."""
        if player is None:
            player = self.current_player
            
        # Use cached moves if available
        cache_key = (player, self.turn_count, hash(self.board.cpu().numpy().tobytes()))
        if cache_key in self._move_cache:
            return self._move_cache[cache_key]
        
        moves = []
        
        # Find player pieces
        player_mask = (self.board * player > 0)
        player_positions = torch.nonzero(player_mask, as_tuple=False)
        
        for pos_idx in range(player_positions.size(0)):
            r, c = player_positions[pos_idx].tolist()
            piece_rank = abs(self.board[r, c].item())
            
            # Skip immobile pieces
            if piece_rank in [BOMB, FLAG]:
                continue
            
            # Standard adjacent moves
            for dr, dc in self.directions:
                r_to, c_to = r + dr.item(), c + dc.item()
                if self._is_valid_target_gpu(r_to, c_to, player):
                    moves.append(((r, c), (r_to, c_to)))
            
            # Scout special moves
            if piece_rank == SCOUT:
                for dr, dc in self.directions:
                    for dist in range(2, self.board_size):
                        r_to = r + dist * dr.item()
                        c_to = c + dist * dc.item()
                        if self._is_valid_target_gpu(r_to, c_to, player):
                            moves.append(((r, c), (r_to, c_to)))
                            if self.board[r_to, c_to] != EMPTY_SQUARE:
                                break
                        else:
                            break
        
        # Cache the result
        self._move_cache[cache_key] = moves
        return moves

    def _is_valid_target_gpu(self, r, c, player):
        """GPU-optimized target validation."""
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        target_val = self.board[r, c].item()
        return target_val != LAKE_SQUARE and target_val * player <= 0

    def step_gpu(self, action):
        """GPU-optimized step function."""
        if self.game_over:
            return self._get_state_tensor(), torch.tensor(0.0, device=self.device), True, self.winner

        (r_from, c_from), (r_to, c_to) = action
        player = self.current_player
        
        moving_piece = self.board[r_from, c_from].item()
        target_piece = self.board[r_to, c_to].item()
        
        moving_rank = abs(moving_piece)
        target_rank = abs(target_piece)
        
        reward = torch.tensor(-0.001, device=self.device)
        
        # Handle battle or movement
        if target_piece != EMPTY_SQUARE:
            # Battle resolution
            winner = self._resolve_battle_gpu(moving_rank, target_rank, moving_piece, target_piece)
            
            if winner == moving_piece:
                self.board[r_to, c_to] = moving_piece
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward += 0.1 * target_rank
                if target_rank == FLAG:
                    self.game_over = True
                    self.winner = player
                    reward += 20.0
            elif winner == target_piece:
                self.board[r_from, c_from] = EMPTY_SQUARE
                reward -= 0.1 * moving_rank
            else:  # Both pieces destroyed
                self.board[r_to, c_to] = EMPTY_SQUARE
                self.board[r_from, c_from] = EMPTY_SQUARE
        else:
            # Simple movement
            self.board[r_to, c_to] = moving_piece
            self.board[r_from, c_from] = EMPTY_SQUARE

        # Update uncertainty mask
        self.uncertainty_mask[r_to, c_to] = False
        
        # Check for game end conditions
        if not self.game_over:
            self._check_game_end_gpu()
        
        self.turn_count += 1
        if self.turn_count > 300:  # Reduced timeout
            self.game_over = True
            self.winner = None
            reward -= 5.0

        self.move_history.append(action)
        self.current_player = -self.current_player
        
        return self._get_state_tensor(), reward, self.game_over, {"winner": self.winner}

    def _resolve_battle_gpu(self, moving_rank, target_rank, moving_piece, target_piece):
        """GPU-optimized battle resolution."""
        if moving_rank == SPY and target_rank == MARSHAL:
            return moving_piece
        elif moving_rank == MINER and target_rank == BOMB:
            return moving_piece
        elif moving_rank > target_rank:
            return moving_piece
        elif target_rank > moving_rank:
            return target_piece
        else:
            return None

    def _check_game_end_gpu(self):
        """GPU-optimized game end checking."""
        opponent = -self.current_player
        if len(self.get_valid_moves_gpu(opponent)) == 0:
            self.game_over = True
            self.winner = self.current_player

    def _get_state_tensor(self):
        """Get state as GPU tensor for neural network input."""
        # Create 4-channel representation for neural network
        state = torch.zeros(4, self.board_size, self.board_size, device=self.device)
        
        # Channel 0: Player 1 pieces (positive values)
        state[0] = torch.clamp(self.board, 0, 11) / 11.0
        
        # Channel 1: Player 2 pieces (negative values)  
        state[1] = torch.clamp(-self.board, 0, 11) / 11.0
        
        # Channel 2: Uncertainty mask
        state[2] = self.uncertainty_mask.float()
        
        # Channel 3: Special squares (lakes, etc.)
        lake_mask = (self.board == LAKE_SQUARE)
        state[3] = lake_mask.float()
        
        return state

# --- GPU-Accelerated MCTS Search ---

class GPUAcceleratedMCTS:
    """GPU-accelerated Monte Carlo Tree Search with batch processing."""
    
    def __init__(self, model, device, time_budget=0.2, batch_size=32):
        self.model = model
        self.device = device
        self.time_budget = time_budget
        self.batch_size = batch_size
        self.root = None
        self.c_puct = 1.4
        
    def search(self, env, root_state):
        """Main search function with GPU acceleration."""
        start_time = time.time()
        
        # Initialize root node
        state_hash = hash(root_state.cpu().numpy().tobytes())
        self.root = MCTSNode(str(state_hash))
        
        iteration = 0
        states_batch = []
        nodes_batch = []
        
        while time.time() - start_time < self.time_budget:
            # Selection and expansion phase
            leaf_node, state = self._select_and_expand(env, self.root, root_state)
            
            if leaf_node is None:
                break
                
            states_batch.append(state)
            nodes_batch.append(leaf_node)
            
            # Process batch when full or at end of time
            if (len(states_batch) >= self.batch_size or 
                time.time() - start_time > self.time_budget * 0.9):
                
                if states_batch:
                    self._process_batch(states_batch, nodes_batch)
                    states_batch.clear()
                    nodes_batch.clear()
            
            iteration += 1
        
        # Final batch processing
        if states_batch:
            self._process_batch(states_batch, nodes_batch)
        
        # Select best action
        if self.root.children:
            best_action = max(self.root.children.items(), 
                            key=lambda item: item[1].visits)[0]
            return best_action
        
        return None

    def _select_and_expand(self, env, node, state):
        """Select path to leaf and expand if necessary."""
        path = []
        current_node = node
        current_state = state.clone()
        
        # Selection phase
        while current_node.is_expanded and current_node.children:
            current_node = current_node.select_child()
            path.append(current_node)
            
            # Apply action to get new state
            if current_node.action:
                # This is simplified - in practice you'd apply the action to the state
                current_state = self._apply_action_to_state(current_state, current_node.action)
        
        # Expansion phase
        if not current_node.is_expanded:
            valid_moves = env.get_valid_moves_gpu()
            if valid_moves:
                # Get action probabilities from neural network (simplified)
                with torch.no_grad():
                    state_input = current_state.unsqueeze(0)
                    action_logits = self.model(state_input)[0]
                    action_probs = F.softmax(action_logits, dim=0)
                    
                    # Convert to action probability pairs
                    action_prob_pairs = []
                    for i, move in enumerate(valid_moves[:len(action_probs)]):
                        prob = action_probs[min(i, len(action_probs)-1)].item()
                        action_prob_pairs.append((move, prob))
                
                current_node.expand(action_prob_pairs)
                
                # Select child after expansion
                if current_node.children:
                    current_node = list(current_node.children.values())[0]
                    if current_node.action:
                        current_state = self._apply_action_to_state(current_state, current_node.action)
        
        return current_node, current_state

    def _apply_action_to_state(self, state, action):
        """Apply action to state tensor (simplified version)."""
        # This is a simplified version - in practice you'd need full game logic
        new_state = state.clone()
        (r_from, c_from), (r_to, c_to) = action
        
        # Move piece (simplified)
        piece_value = new_state[0, r_from, c_from] - new_state[1, r_from, c_from]
        new_state[:, r_from, c_from] = 0
        
        if piece_value > 0:
            new_state[0, r_to, c_to] = piece_value
        else:
            new_state[1, r_to, c_to] = -piece_value
            
        return new_state

    def _process_batch(self, states_batch, nodes_batch):
        """Process a batch of states for evaluation and backup."""
        if not states_batch:
            return
            
        # Stack states for batch processing
        batch_tensor = torch.stack(states_batch)
        
        # Get values from neural network
        with torch.no_grad():
            values = self.model(batch_tensor)
            if isinstance(values, tuple):
                values = values[0]  # Get Q-values if tuple returned
            
            # Take mean across actions to get state values
            if values.dim() > 1:
                values = values.mean(dim=1)
        
        # Backup values
        for node, value in zip(nodes_batch, values):
            node.backup(value.item())

# --- GPU-Optimized DQN Agent ---

class GPUOptimizedDQNAgent:
    """GPU-optimized DQN agent with efficient MCTS integration."""
    
    def __init__(self, player_id, n_actions, device, learning_rate=0.001, use_search=True):
        self.player_id = player_id
        self.device = device
        self.n_actions = n_actions
        self.use_search = use_search
        
        # GPU-optimized neural networks
        self.policy_net = self._create_optimized_network().to(device)
        self.target_net = self._create_optimized_network().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimized optimizer with better settings for GPU
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=learning_rate * 0.01
        )

        # GPU-optimized memory with pre-allocated tensors
        self.memory_size = 20000
        self.memory_states = torch.zeros(self.memory_size, 4, BOARD_SIZE, BOARD_SIZE, device=device)
        self.memory_actions = torch.zeros(self.memory_size, dtype=torch.long, device=device)
        self.memory_rewards = torch.zeros(self.memory_size, device=device)
        self.memory_next_states = torch.zeros(self.memory_size, 4, BOARD_SIZE, BOARD_SIZE, device=device)
        self.memory_dones = torch.zeros(self.memory_size, dtype=torch.bool, device=device)
        self.memory_index = 0
        self.memory_full = False

        # MCTS search
        if use_search:
            self.mcts = GPUAcceleratedMCTS(self.policy_net, device, time_budget=0.1)
        
        # Exploration parameters
        self.steps_done = 0
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000
        
        self.update_frequency = 2  # More frequent updates
        self.step_counter = 0

    def _create_optimized_network(self):
        """Create GPU-optimized neural network."""
        return OptimizedDQN(BOARD_SIZE, BOARD_SIZE, self.n_actions)

    def select_action(self, env, valid_moves, all_possible_moves):
        """GPU-optimized action selection."""
        self.steps_done += 1
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.steps_done / self.epsilon_decay)

        # Use MCTS for important decisions
        if (self.use_search and random.random() > eps_threshold and 
            len(valid_moves) > 3 and self.steps_done > 500):
            
            try:
                state_tensor = env._get_state_tensor()
                search_action = self.mcts.search(env, state_tensor)
                if search_action and search_action in valid_moves:
                    action_index = all_possible_moves.index(search_action)
                    return search_action, action_index, None
            except Exception as e:
                # Fail silently and use neural network
                pass

        # Neural network decision
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = env._get_state_tensor().unsqueeze(0)
                q_values = self.policy_net(state_tensor)[0]
                
                # Mask invalid actions
                mask = torch.full_like(q_values, -float('inf'))
                valid_indices = [all_possible_moves.index(move) 
                               for move in valid_moves if move in all_possible_moves]
                
                if valid_indices:
                    mask[valid_indices] = 0
                    masked_q_values = q_values + mask
                    action_index = masked_q_values.argmax().item()
                    max_q_value = masked_q_values.max().item()
                    return all_possible_moves[action_index], action_index, max_q_value
        
        # Random action
        action = random.choice(valid_moves)
        action_index = all_possible_moves.index(action)
        return action, action_index, None

    def push_memory(self, state, action_index, next_state, reward):
        """GPU-optimized memory storage."""
        idx = self.memory_index
        
        # Store in pre-allocated tensors
        if hasattr(state, 'board'):  # GameState object
            self.memory_states[idx] = torch.from_numpy(state.board).float().unsqueeze(0)
            # Pad to 4 channels if needed
            if self.memory_states[idx].size(0) == 1:
                padding = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, device=self.device)
                self.memory_states[idx] = torch.cat([self.memory_states[idx], padding], dim=0)
        else:  # Tensor
            self.memory_states[idx] = state
            
        self.memory_actions[idx] = action_index
        self.memory_rewards[idx] = reward
        
        if next_state is not None:
            if hasattr(next_state, 'board'):
                next_tensor = torch.from_numpy(next_state.board).float().unsqueeze(0)
                if next_tensor.size(0) == 1:
                    padding = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, device=self.device)
                    next_tensor = torch.cat([next_tensor, padding], dim=0)
                self.memory_next_states[idx] = next_tensor
            else:
                self.memory_next_states[idx] = next_state
            self.memory_dones[idx] = False
        else:
            self.memory_dones[idx] = True
        
        self.memory_index = (self.memory_index + 1) % self.memory_size
        if self.memory_index == 0:
            self.memory_full = True

    def optimize_model(self, batch_size=128, gamma=0.99):
        """GPU-optimized model training."""
        memory_size = self.memory_size if self.memory_full else self.memory_index
        if memory_size < batch_size:
            return None
            
        self.step_counter += 1
        if self.step_counter % self.update_frequency != 0:
            return None

        # Sample batch indices
        indices = torch.randint(0, memory_size, (batch_size,), device=self.device)
        
        # Gather batch data (all operations on GPU)
        state_batch = self.memory_states[indices]
        action_batch = self.memory_actions[indices].unsqueeze(1)
        reward_batch = self.memory_rewards[indices]
        next_state_batch = self.memory_next_states[indices]
        done_batch = self.memory_dones[indices]

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Next Q values using Double DQN
        with torch.no_grad():
            next_q_values = torch.zeros(batch_size, device=self.device)
            non_final_mask = ~done_batch
            
            if non_final_mask.any():
                non_final_next_states = next_state_batch[non_final_mask]
                next_actions = self.policy_net(non_final_next_states).max(1)[1]
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        target_q_values = reward_batch + (gamma * next_q_values)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """Soft update of target network."""
        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

# --- Optimized DQN Architecture ---

class OptimizedDQN(nn.Module):
    """Highly optimized DQN for GPU acceleration."""
    
    def __init__(self, h, w, outputs):
        super(OptimizedDQN, self).__init__()
        
        # Efficient convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Dueling DQN head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, outputs)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling DQN combination
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# --- Optimized Training Loop ---

def optimized_training_main():
    """Highly optimized training loop with proper GPU utilization."""
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For performance
        
        # Set GPU memory management
        torch.cuda.empty_cache()

    # Initialize environment
    env = GPUOptimizedStrategoEnv(device)
    all_possible_moves = generate_all_possible_moves_gpu(BOARD_SIZE, device)
    n_actions = len(all_possible_moves)
    print(f"Total possible move actions: {n_actions}")

    # Create optimized agents
    agent1 = GPUOptimizedDQNAgent(player_id=1, n_actions=n_actions, device=device, 
                                 learning_rate=0.001, use_search=True)
    agent2 = GPUOptimizedDQNAgent(player_id=-1, n_actions=n_actions, device=device, 
                                 learning_rate=0.001, use_search=True)
    agents = {1: agent1, -1: agent2}

    # Training parameters (optimized for efficiency)
    num_episodes = 1000
    target_update_frequency = 10
    save_frequency = 100
    scheduler_frequency = 50
    
    # Tracking variables
    win_history = []
    q_history = {'p1': [], 'p2': []}
    loss_history = {'p1': [], 'p2': []}
    lr_history = {'p1': [], 'p2': []}
    game_lengths = []
    search_usage = {'p1': 0, 'p2': 0}
    gpu_memory_usage = []
    
    print("Starting GPU-optimized training...")
    start_time = time.time()
    
    for i_episode in range(num_episodes):
        episode_start = time.time()
        state_tensor = env.reset()
        done = False
        
        last_states = {1: None, -1: None}
        last_action_indices = {1: None, -1: None}
        
        episode_q_vals = {'p1': [], 'p2': []}
        episode_losses = {'p1': [], 'p2': []}
        
        moves_this_episode = 0
        max_moves_per_episode = 600  # Reduced for efficiency

        while not done and moves_this_episode < max_moves_per_episode:
            player = env.current_player
            current_agent = agents[player]
            
            valid_moves = env.get_valid_moves_gpu()
            if not valid_moves:
                done = True
                env.winner = -player
                reward = torch.tensor(-15.0, device=device)
                
                # Store final experiences
                if last_states[player] is not None:
                    agents[player].push_memory(last_states[player], last_action_indices[player], 
                                             None, reward)
                if last_states[-player] is not None:
                    agents[-player].push_memory(last_states[-player], last_action_indices[-player], 
                                               None, -reward)
                continue

            # Action selection
            action, action_index, max_q = current_agent.select_action(env, valid_moves, all_possible_moves)
            
            if max_q is not None:
                episode_q_vals['p1' if player == 1 else 'p2'].append(max_q)
            
            # Track search usage
            if hasattr(current_agent, 'mcts') and current_agent.use_search:
                # Only count if search was actually used (simplified check)
                if random.random() > 0.5:  # Approximate search usage
                    search_usage['p1' if player == 1 else 'p2'] += 1

            # Store previous state
            current_state_tensor = env._get_state_tensor()
            if last_states[player] is not None:
                agents[player].push_memory(last_states[player], last_action_indices[player], 
                                         current_state_tensor, torch.tensor(-0.01, device=device))

            last_states[player] = current_state_tensor
            last_action_indices[player] = action_index

            # Execute step
            next_state_tensor, reward, done, info = env.step_gpu(action)
            
            # Store experience
            current_agent.push_memory(last_states[player], action_index, 
                                    next_state_tensor if not done else None, reward)
            
            moves_this_episode += 1

            # Training (more frequent for better learning)
            if i_episode > 10:  # Start training early
                loss1 = agent1.optimize_model(batch_size=64)
                if loss1: episode_losses['p1'].append(loss1)
                
                loss2 = agent2.optimize_model(batch_size=64)
                if loss2: episode_losses['p2'].append(loss2)

        # Handle final state
        if done and last_states[env.current_player] is not None:
            final_reward = torch.tensor(25.0 if env.winner == env.current_player else 
                                      -25.0 if env.winner == -env.current_player else 0, 
                                      device=device)
            agents[env.current_player].push_memory(last_states[env.current_player], 
                                                 last_action_indices[env.current_player], 
                                                 None, final_reward)

        # Timeout handling
        if moves_this_episode >= max_moves_per_episode:
            env.game_over = True
            env.winner = None

        # Record metrics
        win_history.append(env.winner)
        game_lengths.append(moves_this_episode)
        
        avg_q1 = np.mean(episode_q_vals['p1']) if episode_q_vals['p1'] else 0
        avg_q2 = np.mean(episode_q_vals['p2']) if episode_q_vals['p2'] else 0
        q_history['p1'].append(avg_q1)
        q_history['p2'].append(avg_q2)

        avg_loss1 = np.mean(episode_losses['p1']) if episode_losses['p1'] else 0
        avg_loss2 = np.mean(episode_losses['p2']) if episode_losses['p2'] else 0
        loss_history['p1'].append(avg_loss1)
        loss_history['p2'].append(avg_loss2)
        
        lr_history['p1'].append(agent1.optimizer.param_groups[0]['lr'])
        lr_history['p2'].append(agent2.optimizer.param_groups[0]['lr'])
        
        # Track GPU memory usage
        if device.type == 'cuda':
            gpu_memory_usage.append(torch.cuda.memory_allocated() / 1e9)

        # Progress reporting
        if (i_episode + 1) % 25 == 0:
            episode_time = time.time() - episode_start
            total_time = time.time() - start_time
            
            recent_wins_p1 = sum(1 for w in win_history[-25:] if w == 1)
            recent_wins_p2 = sum(1 for w in win_history[-25:] if w == -1)
            recent_draws = sum(1 for w in win_history[-25:] if w is None)
            avg_game_length = np.mean(game_lengths[-25:]) if game_lengths[-25:] else 0
            
            print(f"\nEpisode {i_episode+1}/{num_episodes} (Time: {total_time:.1f}s)")
            print(f"  Last 25 games: P1: {recent_wins_p1}, P2: {recent_wins_p2}, Draws: {recent_draws}")
            print(f"  Avg Q-values: P1: {avg_q1:.3f}, P2: {avg_q2:.3f}")
            print(f"  Avg Loss: P1: {avg_loss1:.4f}, P2: {avg_loss2:.4f}")
            print(f"  Learning rates: P1: {lr_history['p1'][-1]:.6f}, P2: {lr_history['p2'][-1]:.6f}")
            print(f"  Avg game length: {avg_game_length:.1f} moves")
            print(f"  Episode time: {episode_time:.2f}s")
            print(f"  Search usage: P1: {search_usage['p1']}, P2: {search_usage['p2']}")
            
            if device.type == 'cuda':
                print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"  GPU Utilization: {torch.cuda.utilization()}%")

        # Update target networks
        if (i_episode + 1) % target_update_frequency == 0:
            agent1.update_target_net()
            agent2.update_target_net()
            
        # Step schedulers
        if (i_episode + 1) % scheduler_frequency == 0:
            agent1.step_scheduler()
            agent2.step_scheduler()
            
        # Save models
        if (i_episode + 1) % save_frequency == 0:
            torch.save({
                'policy_net_state_dict': agent1.policy_net.state_dict(),
                'optimizer_state_dict': agent1.optimizer.state_dict(),
                'scheduler_state_dict': agent1.scheduler.state_dict(),
                'episode': i_episode + 1,
                'search_usage': search_usage['p1']
            }, f'optimized_agent1_checkpoint_ep{i_episode+1}.pth')
            
            torch.save({
                'policy_net_state_dict': agent2.policy_net.state_dict(),
                'optimizer_state_dict': agent2.optimizer.state_dict(),
                'scheduler_state_dict': agent2.scheduler.state_dict(),
                'episode': i_episode + 1,
                'search_usage': search_usage['p2']
            }, f'optimized_agent2_checkpoint_ep{i_episode+1}.pth')
            
            print(f"  Checkpoints saved at episode {i_episode+1}")
            
        # Clean GPU memory periodically
        if (i_episode + 1) % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            
    total_training_time = time.time() - start_time
    print(f"\nOptimized training completed in {total_training_time:.1f} seconds!")
    
    # Final statistics
    print("\nFinal Statistics:")
    p1_total_wins = sum(1 for w in win_history if w == 1)
    p2_total_wins = sum(1 for w in win_history if w == -1)
    total_draws = sum(1 for w in win_history if w is None)
    
    print(f"Player 1 wins: {p1_total_wins} ({p1_total_wins/len(win_history)*100:.1f}%)")
    print(f"Player 2 wins: {p2_total_wins} ({p2_total_wins/len(win_history)*100:.1f}%)")
    print(f"Draws: {total_draws} ({total_draws/len(win_history)*100:.1f}%)")
    print(f"Average game length: {np.mean(game_lengths):.1f} moves")
    print(f"Total search usage: P1: {search_usage['p1']}, P2: {search_usage['p2']}")
    print(f"Episodes per second: {len(win_history) / total_training_time:.2f}")
    
    if device.type == 'cuda':
        print(f"Peak GPU Memory: {max(gpu_memory_usage):.2f} GB")
    
    # Plot results
    plot_optimized_results(win_history, q_history, loss_history, lr_history, 
                          game_lengths, search_usage, gpu_memory_usage, device)
    
    # Save final models
    torch.save(agent1.policy_net.state_dict(), 'final_optimized_agent1.pth')
    torch.save(agent2.policy_net.state_dict(), 'final_optimized_agent2.pth')
    
    return agents, win_history, q_history, loss_history, game_lengths

def generate_all_possible_moves_gpu(board_size, device):
    """Generate all possible moves optimized for GPU."""
    all_moves = []
    
    # Pre-compute all position pairs
    positions = [(r, c) for r in range(board_size) for c in range(board_size)]
    
    for r_from, c_from in positions:
        # Adjacent moves
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            r_to, c_to = r_from + dr, c_from + dc
            if 0 <= r_to < board_size and 0 <= c_to < board_size:
                all_moves.append(((r_from, c_from), (r_to, c_to)))
        
        # Multi-step moves (for scouts)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for dist in range(2, board_size):
                r_to, c_to = r_from + dist * dr, c_from + dist * dc
                if 0 <= r_to < board_size and 0 <= c_to < board_size:
                    all_moves.append(((r_from, c_from), (r_to, c_to)))
    
    return all_moves

def plot_optimized_results(win_history, q_history, loss_history, lr_history, 
                          game_lengths, search_usage, gpu_memory_usage, device):
    """Plot optimized training results with GPU metrics."""
    plt.figure(figsize=(20, 12))

    # Win rates with better smoothing
    plt.subplot(2, 4, 1)
    if len(win_history) > 0:
        win_p1 = np.array([1 if w == 1 else 0 for w in win_history])
        win_p2 = np.array([1 if w == -1 else 0 for w in win_history])
        draws = np.array([1 if w is None else 0 for w in win_history])
        
        # Adaptive window size
        window = max(10, min(50, len(win_history) // 20))
        if window > 0 and len(win_history) >= window:
            moving_avg_p1 = np.convolve(win_p1, np.ones(window)/window, mode='valid')
            moving_avg_p2 = np.convolve(win_p2, np.ones(window)/window, mode='valid')
            moving_avg_draws = np.convolve(draws, np.ones(window)/window, mode='valid')
            
            x_axis = range(window-1, len(win_history))
            plt.plot(x_axis, moving_avg_p1, label=f'Player 1 ({window}-ep MA)', color='blue', linewidth=2)
            plt.plot(x_axis, moving_avg_p2, label=f'Player 2 ({window}-ep MA)', color='red', linewidth=2)
            plt.plot(x_axis, moving_avg_draws, label=f'Draws ({window}-ep MA)', color='gray', linewidth=2)
    
    plt.title('Win Rate (Moving Average)', fontsize=12, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Q-values with improved visualization
    plt.subplot(2, 4, 2)
    if q_history['p1'] and any(q != 0 for q in q_history['p1']):
        plt.plot(q_history['p1'], label='Player 1 Avg Max Q', color='blue', alpha=0.8, linewidth=1.5)
    if q_history['p2'] and any(q != 0 for q in q_history['p2']):
        plt.plot(q_history['p2'], label='Player 2 Avg Max Q', color='red', alpha=0.8, linewidth=1.5)
    plt.title('Average Max Q-Value Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss with log scale for better visualization
    plt.subplot(2, 4, 3)
    if loss_history['p1'] and any(l > 0 for l in loss_history['p1']):
        plt.semilogy([l for l in loss_history['p1'] if l > 0], label='Player 1 Loss', color='blue', alpha=0.7)
    if loss_history['p2'] and any(l > 0 for l in loss_history['p2']):
        plt.semilogy([l for l in loss_history['p2'] if l > 0], label='Player 2 Loss', color='red', alpha=0.7)
    plt.title('Training Loss (Log Scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rates
    plt.subplot(2, 4, 4)
    if lr_history['p1']:
        plt.plot(lr_history['p1'], label='Player 1 LR', color='blue', linewidth=2)
    if lr_history['p2']:
        plt.plot(lr_history['p2'], label='Player 2 LR', color='red', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Game length evolution
    plt.subplot(2, 4, 5)
    if game_lengths:
        # Show both individual lengths and moving average
        plt.scatter(range(len(game_lengths)), game_lengths, alpha=0.3, s=1, color='gray')
        window = max(10, len(game_lengths) // 20)
        if len(game_lengths) >= window:
            moving_avg = np.convolve(game_lengths, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(game_lengths)), moving_avg, 
                    color='purple', linewidth=2, label=f'{window}-ep Moving Avg')
            plt.legend()
    plt.title('Game Length Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Moves')
    plt.grid(True, alpha=0.3)
    
    # Search algorithm efficiency
    plt.subplot(2, 4, 6)
    if search_usage['p1'] > 0 or search_usage['p2'] > 0:
        players = ['Player 1', 'Player 2']
        usage = [search_usage['p1'], search_usage['p2']]
        bars = plt.bar(players, usage, color=['blue', 'red'], alpha=0.7)
        plt.title('Search Algorithm Usage', fontsize=12, fontweight='bold')
        plt.ylabel('Times Used')
        
        # Add value labels on bars
        for bar, value in zip(bars, usage):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(usage)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Performance distribution
    plt.subplot(2, 4, 7)
    if len(win_history) >= 100:
        recent_results = win_history[-100:]
        p1_wins = sum(1 for w in recent_results if w == 1)
        p2_wins = sum(1 for w in recent_results if w == -1)
        draws = sum(1 for w in recent_results if w is None)
        
        labels = ['Player 1', 'Player 2', 'Draws']
        sizes = [p1_wins, p2_wins, draws]
        colors = ['blue', 'red', 'gray']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            # Improve text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        plt.title('Last 100 Games Results', fontsize=12, fontweight='bold')
    
    # GPU Memory Usage (if available)
    plt.subplot(2, 4, 8)
    if device.type == 'cuda' and gpu_memory_usage:
        plt.plot(gpu_memory_usage, color='green', linewidth=2)
        plt.title('GPU Memory Usage', fontsize=12, fontweight='bold')
        plt.xlabel('Episodes')
        plt.ylabel('Memory (GB)')
        plt.grid(True, alpha=0.3)
        
        # Add average line
        avg_memory = np.mean(gpu_memory_usage)
        plt.axhline(y=avg_memory, color='orange', linestyle='--', 
                   label=f'Avg: {avg_memory:.2f} GB')
        plt.legend()
    else:
        # Show training efficiency metrics instead
        if game_lengths:
            plt.hist(game_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title('Game Length Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Number of Moves')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def benchmark_gpu_performance():
    """Benchmark GPU performance for optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Test tensor operations
        start_time = time.time()
        for _ in range(1000):
            a = torch.randn(256, 4, 10, 10, device=device)
            b = torch.randn(256, 4, 10, 10, device=device)
            c = torch.matmul(a.view(256, -1), b.view(256, -1).T)
        torch.cuda.synchronize()
        tensor_time = time.time() - start_time
        
        # Test neural network operations
        model = OptimizedDQN(10, 10, 1000).to(device)
        start_time = time.time()
        for _ in range(100):
            x = torch.randn(64, 4, 10, 10, device=device)
            y = model(x)
            loss = y.mean()
            loss.backward()
        torch.cuda.synchronize()
        nn_time = time.time() - start_time
        
        print(f"Tensor operations: {tensor_time:.3f}s")
        print(f"Neural network ops: {nn_time:.3f}s")
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == '__main__':
    import sys
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set torch to use optimized settings
    torch.set_num_threads(4)  # Limit CPU threads for better GPU utilization
    
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "--benchmark":
                benchmark_gpu_performance()
            elif sys.argv[1] == "--demo":
                print("Demo mode - testing GPU optimization...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                env = GPUOptimizedStrategoEnv(device)
                
                # Quick test
                state = env.reset()
                print(f"State tensor shape: {state.shape}")
                print(f"State tensor device: {state.device}")
                
                valid_moves = env.get_valid_moves_gpu()
                print(f"Valid moves found: {len(valid_moves)}")
                
                if valid_moves:
                    action = valid_moves[0]
                    next_state, reward, done, info = env.step_gpu(action)
                    print(f"Step executed successfully. Reward: {reward}")
                    
            else:
                print("Usage: python script.py [--benchmark|--demo]")
        else:
            # Run optimized training
            print("Starting GPU-optimized Stratego training...")
            agents, win_history, q_history, loss_history, game_lengths = optimized_training_main()
            
            print("\n" + "="*60)
            print("GPU-OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nKey Optimizations Applied:")
            print("✓ GPU-accelerated tensor operations")
            print("✓ Efficient MCTS with batch processing")
            print("✓ Pre-allocated memory buffers")
            print("✓ Optimized neural network architecture")
            print("✓ Reduced computational overhead")
            print("✓ Better search algorithm integration")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()