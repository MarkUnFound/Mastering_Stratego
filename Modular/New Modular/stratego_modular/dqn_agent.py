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
        
        # Compile networks with PyTorch 2.0+ for better GPU utilization
        # Try compilation, but fallback gracefully if Triton is not available
        self._compiled = False
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                # Test if compilation works by trying to compile a simple test
                # This helps catch Triton issues early
                test_model = nn.Linear(10, 10).to(device)
                try:
                    compiled_test = torch.compile(test_model, mode='default')
                    # Test that it actually works (catches TritonMissing at runtime)
                    test_input = torch.randn(1, 10, device=device)
                    _ = compiled_test(test_input)
                    del compiled_test, test_model, test_input
                    
                    # If test passed, compile the actual networks
                    self.q_network = torch.compile(self.q_network, mode='default')
                    self.target_network = torch.compile(self.target_network, mode='default')
                    self._compiled = True
                    print(f"✅ Compiled {self.name} networks with torch.compile")
                except Exception as compile_error:
                    # Compilation or execution failed (likely Triton missing)
                    error_msg = str(compile_error)
                    # Extract just the first line if it's a long error message
                    if '\n' in error_msg:
                        error_msg = error_msg.split('\n')[0]
                    # Suppress verbose Triton errors - it's optional
                    if 'triton' in error_msg.lower():
                        print(f"⚠️  Could not compile {self.name} networks (Triton not available - running without compilation)")
                    else:
                        print(f"⚠️  Could not compile {self.name} networks: {error_msg[:100]}")
                    self._compiled = False
            except Exception as e:
                # If compilation setup fails, continue without compilation
                error_msg = str(e)
                if '\n' in error_msg:
                    error_msg = error_msg.split('\n')[0]
                if 'triton' in error_msg.lower():
                    print(f"⚠️  Could not compile {self.name} networks (Triton not available - running without compilation)")
                else:
                    print(f"⚠️  Could not compile {self.name} networks: {error_msg[:100]}")
                self._compiled = False
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Experience replay - store tensors directly on GPU
        self.memory = deque(maxlen=buffer_size)
        
        # Track policy losses
        self.policy_losses = []
        
        # Step counter for epsilon decay (become deterministic at 500,000 steps)
        self.step_count = 0
        self.epsilon_decay_interval = 500_000  # Become deterministic (epsilon = 0) at 500,000 steps
        
        # Pre-allocate tensors on GPU for batch operations
        self._batch_actions = None
        self._batch_rewards = None
        self._batch_dones = None
        
        # Update target network
        self.update_target_network()
        
    def reset(self):
        """Reset the DQN agent by reinitializing networks and optimizer"""
        # Reinitialize Q-network and target network
        self.q_network = DQN(self.state_size, 512, self.action_size).to(self.device)
        self.target_network = DQN(self.state_size, 512, self.action_size).to(self.device)
        
        # Recompile networks if available
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and self._compiled:
            try:
                self.q_network = torch.compile(self.q_network, mode='default')
                self.target_network = torch.compile(self.target_network, mode='default')
            except Exception:
                self._compiled = False
        
        # Reinitialize optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        # Reset epsilon to initial value
        self.epsilon = 1.0
        # Clear memory
        self.memory.clear()
        # Clear policy losses
        self.policy_losses = []
        # Reset step counter
        self.step_count = 0
        # Reset PBS
        if self.pbs:
            self.pbs.reset()
        # Update target network with new weights
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer - keep tensors on GPU"""
        # Convert to tensors if needed, directly on GPU
        if not isinstance(state, torch.Tensor):
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            else:
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif state.device != self.device:
            state = state.to(self.device)
            
        if not isinstance(next_state, torch.Tensor):
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float().to(self.device)
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        elif next_state.device != self.device:
            next_state = next_state.to(self.device)
            
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # Increment step counter for epsilon decay
        self.step_count += 1
        
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
            # Get PBS-enhanced state (keep on GPU)
            enhanced_state = self.pbs.get_belief_enhanced_state(game_state)
            if enhanced_state is not None:
                # Keep on GPU, flatten and pad/truncate using torch operations
                state = enhanced_state.flatten()
                if len(state) < self.state_size:
                    state = torch.nn.functional.pad(state, (0, self.state_size - len(state)))
                elif len(state) > self.state_size:
                    state = state[:self.state_size]
                # Ensure it's on the correct device
                if state.device != self.device:
                    state = state.to(self.device)
        elif state is None and game_state is not None:
            # Fallback: get state representation if state is None (returns GPU tensor)
            state = self.get_state_representation(game_state)
        elif state is None:
            # No state available, return random action
            return random.choice(valid_moves)
            
        # Exploration: choose random action (use torch for GPU-friendly random)
        if torch.rand(1, device=self.device).item() <= self.epsilon:
            return random.choice(valid_moves)
            
        # Step 2: DQN calculates Q-value using PBS-enhanced state
        # Exploitation: choose best action according to Q-network
        # Ensure state is a GPU tensor
        if not isinstance(state, torch.Tensor):
            if isinstance(state, np.ndarray):
                # Convert numpy to tensor directly on GPU (single transfer)
                state = torch.from_numpy(state).float().to(self.device)
            else:
                # Convert to numpy first, then to tensor on GPU
                state = np.array(state, dtype=np.float32)
                state = torch.from_numpy(state).float().to(self.device)
        elif state.device != self.device:
            # Move to GPU if not already there
            state = state.to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
        self.q_network.train()
        
        # Convert valid moves to action indices and find best (all on GPU)
        best_action = None
        best_q_value = torch.tensor(float('-inf'), device=self.device)
        
        # Pre-compute action indices for all valid moves on GPU
        action_indices = torch.tensor(
            [self._move_to_action_index(move) for move in valid_moves],
            device=self.device,
            dtype=torch.long
        )
        
        # Get Q-values for all valid moves at once (vectorized)
        valid_q_values = q_values[0, action_indices]
        best_idx = torch.argmax(valid_q_values).item()
        best_action = valid_moves[best_idx]
                
        return best_action if best_action is not None else random.choice(valid_moves)
        
    def replay(self) -> Optional[float]:
        """
        Train the model on a batch of experiences - optimized for GPU
        
        Returns:
            Policy loss value or None if not enough experiences
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Stack states and next_states (already on GPU from remember())
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
        
        # Optimize with gradient clipping to prevent excessive loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion (max norm of 10.0)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Track policy loss (only move to CPU for tracking)
        # Clip loss value for reporting to prevent extremely large values
        loss_value = loss.item()
        # Cap reported loss at 100 to prevent misleading statistics
        loss_value_clipped = min(loss_value, 100.0)
        self.policy_losses.append(loss_value_clipped)
        
        # Gradual epsilon decay: linearly decrease from 1.0 to 0.0 over 500,000 steps
        if self.step_count < self.epsilon_decay_interval:
            # Linear decay: epsilon decreases from 1.0 to 0.0 over 500,000 steps
            # Formula: epsilon = 1.0 - (step_count / 500000)
            self.epsilon = 1.0 - (self.step_count / self.epsilon_decay_interval)
            # Clamp to valid range [0.0, 1.0]
            self.epsilon = max(0.0, min(1.0, self.epsilon))
        else:
            # After 500,000 steps: fully deterministic (epsilon = 0, no exploration)
            self.epsilon = 0.0
            
        return loss_value
        
    def get_average_policy_loss(self, window: int = 100) -> float:
        """Get average policy loss over the last N training steps"""
        if not self.policy_losses:
            return 0.0
        recent_losses = self.policy_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
            
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
        """Save the trained model including PBS models"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        
        # Save PBS models if they exist
        if self.pbs:
            # Save LSTM model
            if hasattr(self.pbs, 'lstm_model') and self.pbs.lstm_model is not None:
                checkpoint['pbs_lstm_state_dict'] = self.pbs.lstm_model.state_dict()
                checkpoint['pbs_lstm_optimizer_state_dict'] = self.pbs.lstm_optimizer.state_dict()
            
            # Save PBS evaluator if it exists
            if self.pbs.evaluator is not None:
                checkpoint['pbs_evaluator_state_dict'] = self.pbs.evaluator.evaluator_network.state_dict()
                checkpoint['pbs_evaluator_target_state_dict'] = self.pbs.evaluator.target_network.state_dict()
                checkpoint['pbs_evaluator_optimizer_state_dict'] = self.pbs.evaluator.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        
    def load_model(self, filepath: str):
        """Load a trained model including PBS models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # Load step_count if available (for backward compatibility)
        self.step_count = checkpoint.get('step_count', 0)
        
        # Load PBS models if they exist in checkpoint
        if self.pbs:
            # Load LSTM model if available
            if 'pbs_lstm_state_dict' in checkpoint and hasattr(self.pbs, 'lstm_model'):
                try:
                    self.pbs.lstm_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
                    if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                        self.pbs.lstm_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
                    print(f"✅ Loaded PBS LSTM model for {self.name}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load PBS LSTM model for {self.name}: {e}")
            
            # Load PBS evaluator if available
            if 'pbs_evaluator_state_dict' in checkpoint and self.pbs.evaluator is not None:
                try:
                    self.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
                    self.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
                    if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
                        self.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
                    self.pbs.evaluator.update_target_network()
                    print(f"✅ Loaded PBS evaluator model for {self.name}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load PBS evaluator model for {self.name}: {e}")
        
    def get_state_representation(self, game_state) -> torch.Tensor:
        """
        Convert game state to neural network input - returns GPU tensor.
        
        This method ensures that agents only use visible information.
        The game_state.board already contains only the visible board for the current player.
        
        If PBS is enabled, the state is enhanced with belief probabilities.
        Returns a torch.Tensor on the GPU to avoid CPU-GPU transfers.
        """
        # Step 1: PBS gets the value and creates possible values with confidence scores
        if self.pbs and hasattr(game_state, 'board'):
            enhanced_state = self.pbs.get_belief_enhanced_state(game_state)
            if enhanced_state is not None:
                # Keep on GPU
                visible_board = enhanced_state
                if visible_board.device != self.device:
                    visible_board = visible_board.to(self.device)
            else:
                # Fallback to regular state
                visible_board = game_state.board
                if isinstance(visible_board, torch.Tensor):
                    if visible_board.device != self.device:
                        visible_board = visible_board.to(self.device)
                else:
                    visible_board = torch.tensor(visible_board, dtype=torch.float32, device=self.device)
        else:
            # Ensure we're only using the visible board information
            if hasattr(game_state, 'board'):
                # It's a game state object with visible board for current player
                visible_board = game_state.board
                if isinstance(visible_board, torch.Tensor):
                    if visible_board.device != self.device:
                        visible_board = visible_board.to(self.device)
                else:
                    visible_board = torch.tensor(visible_board, dtype=torch.float32, device=self.device)
            else:
                # It's already a board/array
                visible_board = game_state
                if isinstance(visible_board, torch.Tensor):
                    if visible_board.device != self.device:
                        visible_board = visible_board.to(self.device)
                else:
                    visible_board = torch.tensor(visible_board, dtype=torch.float32, device=self.device)
        
        # Flatten the visible board (10x10 = 100 values) - keep on GPU
        if len(visible_board.shape) == 2:
            state = visible_board.flatten()
        else:
            state = visible_board
        
        # Pad or truncate to fixed size if needed - use torch operations
        if len(state) < self.state_size:
            state = torch.nn.functional.pad(state, (0, self.state_size - len(state)))
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
    
    def update_pbs_from_reveal(self, pos: Tuple[int, int], piece_type: PieceType,
                              game_phase: str = 'middle', turn_count: int = 0):
        """
        Update PBS when a piece is revealed.
        
        Args:
            pos: Position of the revealed piece
            piece_type: Type of the revealed piece
            game_phase: Game phase ('early', 'middle', or 'end') for evaluator data collection
            turn_count: Current turn number for evaluator data collection
        """
        if self.pbs:
            self.pbs.update_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=turn_count)
