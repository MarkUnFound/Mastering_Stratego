"""
DQN Agent for Stratego Game
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict
from piece import PieceType
from probabilistic_belief_state import ProbabilisticBeliefState, PieceActionAaren, PBS_EVALUATOR_AVAILABLE
if PBS_EVALUATOR_AVAILABLE:
    from pbs_evaluator import PBSEvaluator
from critic import ExploitabilityCritic

# Define a named tuple for experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QVectorDQN(nn.Module):
    """
    Q-Vector Deep Q-Network for Stratego
    
    Architecture:
    - Input: Belief State (13 channels: 1 visible board + 12 belief maps)
    - Encoder: CNN layers to extract spatial features
    - Output: Two heads (Q-Vector):
        - Agent Head: Q_agent(s, a) - Expected return for the agent
        - Opponent Head: Q_opp(s, a) - Expected return for the opponent
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (13, 10, 10), output_size: int = 1000):
        """
        Initialize the QVectorDQN network
        
        Args:
            input_shape: Shape of input (channels, height, width)
            output_size: Size of output (number of possible actions)
        """
        super(QVectorDQN, self).__init__()
        
        # Deeper CNN for richer belief state
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate flattened size: 128 * 10 * 10 = 12800
        self.flatten_size = 128 * 10 * 10
        
        # Shared dense layer
        self.fc_shared = nn.Linear(self.flatten_size, 512)
        
        # Head 1: Agent's Q-values
        self.fc_agent = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
        # Head 2: Opponent's Q-values (Modeled explicitly)
        self.fc_opp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Returns:
            q_vector: Tensor of shape (batch, output_size, 2)
                      [..., 0] = Q_agent
                      [..., 1] = Q_opp
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        
        q_agent = self.fc_agent(x)
        q_opp = self.fc_opp(x)
        
        # Stack to form Q-Vector: (batch, actions, 2)
        return torch.stack([q_agent, q_opp], dim=2)


class DQNAgent:
    """DQN Agent for Stratego with experience replay and target network"""
    
    def __init__(self, player_id: int, device, 
                 state_size: int = 200, action_size: int = 1000,
                 lr: float = 0.00001, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_min: float = 0.1, 
                 epsilon_decay: float = 0.001, 
                 buffer_size: int = 10000, batch_size: int = 32,
                 use_pbs: bool = True, num_envs: int = 1):
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
        self.num_envs = num_envs
        
        # Probabilistic Belief State
        self.pbs = None
        self.pbs_instances = []
        
        if self.use_pbs:
            if num_envs > 1:
                # Create shared models for parallel environments
                # AAREN model
                self.shared_aaren = PieceActionAaren(
                    input_size=24,
                    hidden_size=64,
                    num_layers=3,
                    output_size=12, # NUM_PIECE_TYPES
                    device=device
                ).to(device)
                self.shared_aaren_optimizer = optim.AdamW(self.shared_aaren.parameters(), lr=0.001, weight_decay=0.01)
                
                # PBS Evaluator
                self.shared_evaluator = None
                if PBS_EVALUATOR_AVAILABLE:
                    self.shared_evaluator = PBSEvaluator(device=device)
                
                # Create PBS instances sharing the models
                for _ in range(num_envs):
                    pbs_instance = ProbabilisticBeliefState(
                        player_id, device, 
                        shared_aaren_model=self.shared_aaren,
                        shared_evaluator=self.shared_evaluator
                    )
                    # Ensure shared optimizer is accessible for saving/training
                    pbs_instance.aaren_optimizer = self.shared_aaren_optimizer
                    self.pbs_instances.append(pbs_instance)
                
                # Set self.pbs to the first instance for backward compatibility
                self.pbs = self.pbs_instances[0]
            else:
                # Single environment - standard initialization
                self.pbs = ProbabilisticBeliefState(player_id, device)
                self.pbs_instances = [self.pbs]
        
        # Uncertainty-driven exploration parameters
        self.uncertainty_exploration_multiplier = 0.5  # How much uncertainty affects exploration
        self.uncertainty_penalty_scale = 0.3  # Penalty for uncertain actions
        
        # Action-PBS buffer for tracking Q-values with PBS states
        # Keyed by env_idx to ensure parallel safety
        self.action_pbs_buffer = {i: deque(maxlen=50) for i in range(num_envs)}
        
        # Performance monitoring
        self.pbs_dqn_alignment_history = deque(maxlen=100)  # Track alignment between PBS and DQN
        self.performance_metrics = {
            'pbs_accuracy_trend': deque(maxlen=100),
            'dqn_loss_trend': deque(maxlen=100),
            'action_prediction_alignment': deque(maxlen=100)
        }
        
        # Neural networks (keep on GPU, no compilation for Windows compatibility)
        # Neural networks (keep on GPU, no compilation for Windows compatibility)
        # Input shape: (13, 10, 10) - 1 visible board + 12 belief maps
        self.q_network = QVectorDQN(input_shape=(13, 10, 10), output_size=action_size).to(device)
        self.target_network = QVectorDQN(input_shape=(13, 10, 10), output_size=action_size).to(device)
        
        # Enable cuDNN benchmarking for faster convolutions (if using conv layers)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, non-deterministic
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Exploitability Critic
        self.critic = ExploitabilityCritic(input_shape=(13, 10, 10), output_size=action_size).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01)
        self.critic_loss_fn = nn.CrossEntropyLoss()
        self.critic_weight = 0.1  # Weight for predictability penalty
        
        # Experience replay - store tensors directly on GPU
        self.memory = deque(maxlen=buffer_size)
        
        # Track policy losses
        self.policy_losses = []
        self.q_values_history = []
        self.entropy_history = []
        
        # Loss smoothing (exponential moving average)
        self.smoothed_loss = None
        self.loss_smoothing_factor = 0.95  # EMA factor (higher = more smoothing)
        
        # Step counter for epsilon decay
        self.step_count = 0
        self.epsilon_decay_interval = 500_000  # Decay epsilon over 500,000 steps
        self.epsilon_min = max(epsilon_min, 0.01)  # Ensure minimum epsilon for continued exploration
        
        # Learning rate scheduling (simple exponential decay)
        self.initial_lr = lr
        self.lr_decay_factor = 0.5  # Reduce LR by half every 500k steps
        self.lr_decay_interval = 500_000
        self.min_lr = lr * 0.01  # Minimum learning rate (1% of initial)
        
        # Adaptive epsilon: increase if performance stagnates
        self.reward_history = deque(maxlen=1000)  # Track recent rewards
        self.stagnation_threshold = 50  # Episodes without improvement
        self.stagnation_episodes = 0
        self.best_avg_reward = float('-inf')
        
        # Pre-allocate tensors on GPU for batch operations
        self._batch_actions = None
        self._batch_rewards = None
        self._batch_dones = None
        
        # Update target network
        self.update_target_network()
        
    def reset(self):
        """Reset the DQN agent by reinitializing networks and optimizer"""
        # Reinitialize Q-network and target network
        # Reinitialize Q-network and target network
        self.q_network = QVectorDQN(input_shape=(13, 10, 10), output_size=self.action_size).to(self.device)
        self.target_network = QVectorDQN(input_shape=(13, 10, 10), output_size=self.action_size).to(self.device)
        
        # Reinitialize optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        
        # Reinitialize Critic
        self.critic = ExploitabilityCritic(input_shape=(13, 10, 10), output_size=self.action_size).to(self.device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=0.01)
        # Reset epsilon to initial value
        self.epsilon = 1.0
        # Clear memory
        self.memory.clear()
        # Clear policy losses
        self.policy_losses = []
        self.agent_losses = []
        self.opp_losses = []
        self.q_values_history = []
        self.entropy_history = []
        # Reset smoothed loss
        self.smoothed_loss = None
        # Reset step counter
        self.step_count = 0
        # Reset adaptive epsilon tracking
        self.reward_history = deque(maxlen=1000)
        self.stagnation_episodes = 0
        self.best_avg_reward = float('-inf')
        # Reset PBS
        if self.pbs:
            if self.num_envs > 1:
                for pbs in self.pbs_instances:
                    pbs.reset()
            else:
                self.pbs.reset()
        # Clear action-PBS buffer
        for i in range(self.num_envs):
            self.action_pbs_buffer[i].clear()
        if hasattr(self, '_action_q_values'):
            self._action_q_values.clear()
        # Clear alignment history
        self.pbs_dqn_alignment_history.clear()
        # Clear performance metrics
        for key in self.performance_metrics:
            self.performance_metrics[key].clear()
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
        
        if abs(reward) > 5.0:  # Win/loss experiences
            for _ in range(3):  # Triple-store important experiences
                self.memory.append(experience)
        else:
            self.memory.append(experience)
        
        # Increment step counter for epsilon decay
        self.step_count += 1

    def sample_replay_batch(self) -> Optional[List[Experience]]:
        """Sample a batch from the replay buffer without performing an update."""
        if len(self.memory) < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)
        
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
                # Keep on GPU, ensure it has channel dimension
                if enhanced_state.dim() == 2:
                    state = enhanced_state.unsqueeze(0)
                else:
                    state = enhanced_state
                # Ensure it's on the correct device
                if state.device != self.device:
                    state = state.to(self.device)
        elif state is None and game_state is not None:
            # Fallback: get state representation if state is None (returns GPU tensor)
            state = self.get_state_representation(game_state)
        elif state is None:
            # No state available, return random action
            return random.choice(valid_moves)
            
        # Step 2: Get uncertainty map for uncertainty-driven exploration
        uncertainty_map = {}
        if self.pbs and game_state is not None:
            uncertainty_map = self.pbs.get_uncertainty_map(game_state)
        
        # Step 3: DQN calculates Q-Vector using PBS-enhanced state
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
            
        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension
            
        self.q_network.eval()
        with torch.no_grad():
            # Output: (batch, actions, 2) -> [Q_agent, Q_opp]
            q_vectors = self.q_network(state)
            
            # Extract Agent and Opponent Q-values
            q_agent = q_vectors[:, :, 0]
            q_opp = q_vectors[:, :, 1]
            
            # Maximin Strategy: Maximize (Q_agent - Q_opp)
            # We want to maximize our gain relative to the opponent's gain
            # Or "Maximize worst case": If we assume opponent plays optimally to minimize our gain,
            # but here we have explicit predictions.
            # Let's use the relative advantage: Score = Q_agent - Q_opp
            # This effectively treats the game as zero-sum for selection, but learns non-zero-sum dynamics.
            base_q_values = q_agent - q_opp
            
        self.q_network.train()
        
        # Step 4: Apply uncertainty-aware Q-value calculation
        # We use the combined score as the "Q-value" for exploration logic
        q_values = self.calculate_uncertainty_aware_q_values(
            base_q_values, valid_moves, uncertainty_map
        )
        
        # Step 5: Uncertainty-driven exploration (modify epsilon based on uncertainty)
        # Higher uncertainty in available moves -> more exploration
        avg_uncertainty = self.get_average_uncertainty(valid_moves, uncertainty_map)
        adjusted_epsilon = self.epsilon + (avg_uncertainty * self.uncertainty_exploration_multiplier)
        adjusted_epsilon = min(1.0, adjusted_epsilon)  # Cap at 1.0
        
        # Exploration: choose random action with adjusted epsilon
        if torch.rand(1, device=self.device, dtype=torch.float32).item() <= adjusted_epsilon:
            return random.choice(valid_moves)
        
        # Step 6: Exploitation: choose best action with uncertainty weighting
        # Pre-compute action indices for all valid moves on GPU
        action_indices = torch.tensor(
            [self._move_to_action_index(move) for move in valid_moves],
            device=self.device,
            dtype=torch.long
        )
        
        # Get Q-values for all valid moves at once (vectorized)
        valid_q_values = q_values[0, action_indices]
        
        # Add exploration bonus for uncertain positions
        exploration_bonuses = torch.zeros(len(valid_moves), device=self.device)
        for i, move in enumerate(valid_moves):
            uncertainty = self.get_move_uncertainty(move, uncertainty_map)
            exploration_bonuses[i] = uncertainty * self.uncertainty_exploration_multiplier
        
        # Modified Q-values: Q + exploration_bonus
        modified_q_values = valid_q_values + exploration_bonuses
        
        # Choose action with highest modified Q-value
        best_idx = torch.argmax(modified_q_values).item()
        best_action = valid_moves[best_idx]
        
        # Store action-PBS state for feedback
        if self.pbs and game_state is not None:
            # Default to env_idx 0 for single action
            self.store_action_pbs_state(best_action, base_q_values, uncertainty_map, game_state, env_idx=0)
                
        return best_action if best_action is not None else random.choice(valid_moves)
        
    def replay(self, batch: Optional[List[Experience]] = None) -> Optional[float]:
        """
        Train the model on a batch of experiences - optimized for GPU
        
        Returns:
            Policy loss value or None if not enough experiences
        """
        if batch is None:
            if len(self.memory) < self.batch_size:
                return None
            batch = random.sample(self.memory, self.batch_size)
        
        # Stack states and next_states (already on GPU from remember())
        states = torch.stack([e.state for e in batch])
        next_states = torch.stack([e.next_state for e in batch])
        
        # Create tensors directly on GPU (avoid CPU intermediate)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        
        # --- 1. Train Critic First ---
        # Predict action from state
        critic_logits = self.critic(states)
        critic_loss = self.critic_loss_fn(critic_logits, actions)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # --- 2. Calculate Predictability Penalty ---
        # We use the updated critic to calculate the penalty
        with torch.no_grad():
            probs = F.softmax(self.critic(states), dim=1)
            # Get probability assigned to the taken action
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            # Penalty = critic_weight * probability (higher prob = higher penalty)
            # Reduced weight to 0.05 to avoid overwhelming game rewards
            penalty = 0.05 * action_probs
            avg_penalty = penalty.mean().item()
            
            # Calculate entropy for metrics
            # Entropy = -sum(p * log(p))
            entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            
        # --- 3. Train Agent with Penalized Rewards (Dual-Phase) ---
        # Current Q-Vectors
        # Shape: (batch, actions, 2)
        current_q_vectors = self.q_network(states)
        
        # Gather Q-values for taken actions
        # actions shape: (batch) -> unsqueeze -> (batch, 1) -> expand -> (batch, 1, 2)
        # gather result: (batch, 1, 2) -> squeeze -> (batch, 2)
        current_q_values = current_q_vectors.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
        
        current_q_agent = current_q_values[:, 0]
        current_q_opp = current_q_values[:, 1]
        
        # Calculate average Q-value for metrics (Agent's perspective)
        avg_q_value = current_q_agent.mean().item()
        
        # Next Q-Vectors from target network
        with torch.no_grad():
            next_q_vectors = self.target_network(next_states)
            # Maximin selection for next state:
            # Select action that maximizes (Q_agent - Q_opp)
            next_scores = next_q_vectors[:, :, 0] - next_q_vectors[:, :, 1]
            best_next_actions = next_scores.argmax(dim=1)
            
            # Gather target Q-values for best actions
            next_q_values = next_q_vectors.gather(1, best_next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
            
            next_q_agent = next_q_values[:, 0]
            next_q_opp = next_q_values[:, 1]
            
        # Penalize rewards: Reward - Penalty
        # This encourages the agent to choose actions that are less predictable
        penalized_rewards = rewards - penalty
        
        # Calculate target Q-values for both heads
        # Agent Head: Predicts my return
        target_q_agent = penalized_rewards + (self.gamma * next_q_agent * ~dones)
        
        # Opponent Head: Predicts opponent's return (which is usually -reward in zero-sum)
        # But we model it explicitly. If zero-sum, r_opp = -r_agent.
        # Let's assume zero-sum rewards for now: r_opp = -penalized_rewards
        target_q_opp = (-penalized_rewards) + (self.gamma * next_q_opp * ~dones)
        
        # Compute losses for both heads
        loss_agent = F.smooth_l1_loss(current_q_agent, target_q_agent)
        loss_opp = F.smooth_l1_loss(current_q_opp, target_q_opp)
        
        # Total loss
        loss = loss_agent + loss_opp
        
        # Optimize Agent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # --- 4. Logging & Updates ---
        
        # Track policy loss (minimize CPU transfer - only get item() once)
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        self.agent_losses.append(loss_agent.item())
        self.opp_losses.append(loss_opp.item())
        
        # Track critic metrics
        if not hasattr(self, 'critic_losses'):
            self.critic_losses = []
        self.critic_losses.append(critic_loss.item())
        
        if not hasattr(self, 'penalty_history'):
            self.penalty_history = []
        self.penalty_history.append(avg_penalty)
        
        # Track new metrics
        if not hasattr(self, 'q_values_history'):
            self.q_values_history = []
        self.q_values_history.append(avg_q_value)
        
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        self.entropy_history.append(entropy)
        
        # Update smoothed loss (exponential moving average)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss_value
        else:
            self.smoothed_loss = self.loss_smoothing_factor * self.smoothed_loss + (1 - self.loss_smoothing_factor) * loss_value
        
        # Simple exponential learning rate decay
        lr_decay_steps = self.step_count // self.lr_decay_interval
        current_lr = self.initial_lr * (self.lr_decay_factor ** lr_decay_steps)
        current_lr = max(current_lr, self.min_lr)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Gradual epsilon decay
        if self.step_count < self.epsilon_decay_interval:
            progress = self.step_count / self.epsilon_decay_interval
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * (1.0 - progress)
            self.epsilon = max(self.epsilon_min, min(1.0, self.epsilon))
        else:
            self.epsilon = self.epsilon_min
            
        return loss_value
        
    def get_average_policy_loss(self, window: int = 100) -> float:
        """Get average policy loss over the last N training steps"""
        if not self.policy_losses:
            return 0.0
        recent_losses = self.policy_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
    
    def get_smoothed_loss(self) -> float:
        """Get the exponentially smoothed loss value"""
        return self.smoothed_loss if self.smoothed_loss is not None else 0.0
    
    def get_policy_loss_stats(self, window: int = 100) -> dict:
        """Get detailed statistics about policy loss"""
        if not self.policy_losses:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'std': 0.0}
        
        recent_losses = self.policy_losses[-window:]
        if not recent_losses:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'std': 0.0}
        
        sorted_losses = sorted(recent_losses)
        n = len(sorted_losses)
        
        stats = {
            'mean': sum(recent_losses) / n,
            'min': sorted_losses[0],
            'max': sorted_losses[-1],
            'median': sorted_losses[n // 2] if n > 0 else 0.0,
            'std': (sum((x - sum(recent_losses) / n) ** 2 for x in recent_losses) / n) ** 0.5 if n > 1 else 0.0
        }
        
        # Add separate stats for agent and opponent losses if available
        if hasattr(self, 'agent_losses') and self.agent_losses:
            recent_agent = self.agent_losses[-window:]
            if recent_agent:
                stats['avg_agent_loss'] = sum(recent_agent) / len(recent_agent)
                
        if hasattr(self, 'opp_losses') and self.opp_losses:
            recent_opp = self.opp_losses[-window:]
            if recent_opp:
                stats['avg_opp_loss'] = sum(recent_opp) / len(recent_opp)
                
        return stats

    def get_average_critic_loss(self, window: int = 100) -> float:
        """Get average critic loss over the last N training steps"""
        if not hasattr(self, 'critic_losses') or not self.critic_losses:
            return 0.0
        recent_losses = self.critic_losses[-window:]
        return sum(recent_losses) / len(recent_losses)
        
    def get_average_penalty(self, window: int = 100) -> float:
        """Get average predictability penalty over the last N training steps"""
        if not hasattr(self, 'penalty_history') or not self.penalty_history:
            return 0.0
        recent_penalties = self.penalty_history[-window:]
        return sum(recent_penalties) / len(recent_penalties)
    
    def get_average_q_value(self, window: int = 100) -> float:
        """Get average Q-value over the last N training steps"""
        if not hasattr(self, 'q_values_history') or not self.q_values_history:
            return 0.0
        recent_q = self.q_values_history[-window:]
        return sum(recent_q) / len(recent_q)
        
    def get_average_entropy(self, window: int = 100) -> float:
        """Get average action entropy over the last N training steps"""
        if not hasattr(self, 'entropy_history') or not self.entropy_history:
            return 0.0
        recent_entropy = self.entropy_history[-window:]
        return sum(recent_entropy) / len(recent_entropy)
    
    def get_current_learning_rate(self) -> float:
        """Get the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def update_episode_reward(self, episode_reward: float):
        """
        Track episode reward and adaptively adjust epsilon if performance stagnates.
        This helps the agent continue exploring when it gets stuck in a local optimum.
        """
        self.reward_history.append(episode_reward)
        
        # Calculate average reward over last 100 episodes
        if len(self.reward_history) >= 100:
            recent_avg = sum(list(self.reward_history)[-100:]) / 100
            
            # Check if we've improved
            if recent_avg > self.best_avg_reward:
                self.best_avg_reward = recent_avg
                self.stagnation_episodes = 0
            else:
                self.stagnation_episodes += 1
            
            # If performance has stagnated, increase epsilon to encourage exploration
            if self.stagnation_episodes >= self.stagnation_threshold:
                # Increase epsilon (but cap at 0.2 to avoid too much randomness)
                self.epsilon = min(0.2, self.epsilon * 1.5)
                self.stagnation_episodes = 0  # Reset counter after adjustment
                # Optionally reset best_avg_reward to allow new baseline
                self.best_avg_reward = recent_avg
            
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
        """Save the DQN model only (without PBS components)"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        torch.save(checkpoint, filepath)
    
    def save_aaren_model(self, filepath: str):
        """Save the AAREN-RNN model separately"""
        if not self.pbs or not hasattr(self.pbs, 'aaren_model') or self.pbs.aaren_model is None:
            raise ValueError(f"AAREN model not available for {self.name}")
        
        checkpoint = {
            'pbs_aaren_state_dict': self.pbs.aaren_model.state_dict(),
            'pbs_aaren_optimizer_state_dict': self.pbs.aaren_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def save_pbs_evaluator(self, filepath: str):
        """Save the PBS Evaluator model separately"""
        if not self.pbs or self.pbs.evaluator is None:
            raise ValueError(f"PBS Evaluator not available for {self.name}")
        
        checkpoint = {
            'pbs_evaluator_state_dict': self.pbs.evaluator.evaluator_network.state_dict(),
            'pbs_evaluator_target_state_dict': self.pbs.evaluator.target_network.state_dict(),
            'pbs_evaluator_optimizer_state_dict': self.pbs.evaluator.optimizer.state_dict(),
            'pbs_evaluator_training_losses': self.pbs.evaluator.training_losses,
        }
        
        # Save experience buffer (convert to CPU for serialization)
        memory_data = []
        for exp in self.pbs.evaluator.memory:
            pbs_pred_cpu = exp.pbs_prediction.cpu().detach() if isinstance(exp.pbs_prediction, torch.Tensor) else exp.pbs_prediction
            memory_data.append({
                'pbs_prediction': pbs_pred_cpu,
                'ground_truth': exp.ground_truth.value,  # Save enum value
                'position': exp.position,
                'game_phase': exp.game_phase,
                'turn_count': exp.turn_count
            })
        checkpoint['pbs_evaluator_memory'] = memory_data
        
        torch.save(checkpoint, filepath)
        
    def load_model(self, filepath: str):
        """Load the DQN model only (without PBS components)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        # Load step_count if available (for backward compatibility)
        self.step_count = checkpoint.get('step_count', 0)
    
    def load_aaren_model(self, filepath: str):
        """Load the AAREN-RNN model separately"""
        if not self.pbs or not hasattr(self.pbs, 'aaren_model'):
            raise ValueError(f"AAREN model not available for {self.name}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Try AAREN keys first, then fall back to LSTM keys for backward compatibility
        if 'pbs_aaren_state_dict' in checkpoint:
            self.pbs.aaren_model.load_state_dict(checkpoint['pbs_aaren_state_dict'])
            if 'pbs_aaren_optimizer_state_dict' in checkpoint:
                self.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_aaren_optimizer_state_dict'])
            print(f"✅ Loaded PBS AAREN model for {self.name}")
        elif 'pbs_lstm_state_dict' in checkpoint:
            # Backward compatibility: Load old LSTM checkpoints as AAREN
            self.pbs.aaren_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
            if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                self.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
            print(f"✅ Loaded PBS AAREN model (migrated from old LSTM checkpoint) for {self.name}")
        else:
            raise KeyError(f"No AAREN/LSTM state dict found in {filepath}")
    
    def load_pbs_evaluator(self, filepath: str):
        """Load the PBS Evaluator model separately"""
        if not self.pbs or self.pbs.evaluator is None:
            raise ValueError(f"PBS Evaluator not available for {self.name}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
        self.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
        if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
            self.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
        self.pbs.evaluator.update_target_network()
        
        # Load experience buffer if available
        if 'pbs_evaluator_memory' in checkpoint:
            from pbs_evaluator import PBSEvaluationExperience
            from piece import PieceType
            
            self.pbs.evaluator.memory.clear()
            for mem_data in checkpoint['pbs_evaluator_memory']:
                # Convert back to PBSEvaluationExperience
                pbs_pred = mem_data['pbs_prediction']
                if isinstance(pbs_pred, torch.Tensor):
                    pbs_pred = pbs_pred.to(self.pbs.evaluator.device)
                else:
                    pbs_pred = torch.tensor(pbs_pred, device=self.pbs.evaluator.device)
                
                # Convert ground truth value back to PieceType
                ground_truth = PieceType(mem_data['ground_truth'])
                
                experience = PBSEvaluationExperience(
                    pbs_prediction=pbs_pred,
                    ground_truth=ground_truth,
                    position=tuple(mem_data['position']),
                    game_phase=mem_data['game_phase'],
                    turn_count=mem_data['turn_count']
                )
                self.pbs.evaluator.memory.append(experience)
            
            print(f"✅ Loaded PBS evaluator model for {self.name} with {len(self.pbs.evaluator.memory)} experiences")
        else:
            print(f"✅ Loaded PBS evaluator model for {self.name} (no experience buffer found)")
        
        # Load training losses if available
        if 'pbs_evaluator_training_losses' in checkpoint:
            self.pbs.evaluator.training_losses = checkpoint['pbs_evaluator_training_losses']
        
    def get_state_representation(self, game_state, pbs_instance=None) -> torch.Tensor:
        """
        Convert GameState to tensor for network input.
        Uses PBS if available to get belief-enhanced state.
        Returns: (C, H, W) tensor on GPU.
        """
        pbs = pbs_instance if pbs_instance else self.pbs
        if pbs and hasattr(game_state, 'board'):
            enhanced_state = pbs.get_belief_enhanced_state(game_state)
            if enhanced_state is not None:
                # Ensure it's on the correct device
                if enhanced_state.device != self.device:
                    enhanced_state = enhanced_state.to(self.device)
                
                # Ensure it's (13, 10, 10) - remove batch dim if present
                if enhanced_state.dim() == 4:
                    enhanced_state = enhanced_state.squeeze(0)
                
                return enhanced_state
        
        # Fallback to regular state (1 channel)
        if hasattr(game_state, 'board'):
            board = game_state.board
            if not isinstance(board, torch.Tensor):
                if isinstance(board, np.ndarray):
                    board = torch.from_numpy(board).float()
                else:
                    # Create empty board if invalid
                    board = torch.zeros((10, 10), dtype=torch.float32)
            
            board = board.to(self.device).float()
            
            # Normalize
            board = board / 12.0
            
            # Add channel dim: (1, 10, 10)
            if board.dim() == 2:
                return board.unsqueeze(0)
            return board
            
        return torch.zeros((1, 10, 10), device=self.device)

    def get_batch_state_representation(self, states, game_states=None) -> torch.Tensor:
        """
        Convert a batch of game states to a batch tensor.
        """
        tensor_list = []
        for i, state in enumerate(states):
            # Use game_states[i] if available for PBS, otherwise use state
            gs = game_states[i] if game_states else state
            # Use the corresponding PBS instance
            pbs_inst = self.pbs_instances[i] if self.pbs_instances and i < len(self.pbs_instances) else self.pbs
            
            tensor = self.get_state_representation(gs, pbs_instance=pbs_inst)
            tensor_list.append(tensor)
        
        return torch.stack(tensor_list)

    def act_batch(self, states, valid_moves_list, game_states=None) -> List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Choose actions for a batch of states.
        """
        batch_size = len(states)
        actions = [None] * batch_size
        
        # 1. Get batch state representation
        state_tensor = self.get_batch_state_representation(states, game_states)
        
        # 2. Get uncertainty maps (if PBS)
        uncertainty_maps = []
        if self.pbs_instances and game_states:
            for i, gs in enumerate(game_states):
                if gs:
                    uncertainty_maps.append(self.pbs_instances[i].get_uncertainty_map(gs))
                else:
                    uncertainty_maps.append({})
        else:
            uncertainty_maps = [{}] * batch_size
            
        # 3. Network forward pass
        self.q_network.eval()
        with torch.no_grad():
            base_q_values_batch = self.q_network(state_tensor)
        self.q_network.train()
        
        # 4. Process each env
        for i in range(batch_size):
            valid_moves = valid_moves_list[i]
            if not valid_moves:
                continue
                
            base_q_values = base_q_values_batch[i:i+1] # Keep batch dim (1, action_size)
            uncertainty_map = uncertainty_maps[i]
            
            # Calculate uncertainty aware Q-values
            q_values = self.calculate_uncertainty_aware_q_values(
                base_q_values, valid_moves, uncertainty_map
            )
            
            # Epsilon-greedy
            avg_uncertainty = self.get_average_uncertainty(valid_moves, uncertainty_map)
            adjusted_epsilon = self.epsilon + (avg_uncertainty * self.uncertainty_exploration_multiplier)
            adjusted_epsilon = min(1.0, adjusted_epsilon)
            
            if torch.rand(1, device=self.device, dtype=torch.float32).item() <= adjusted_epsilon:
                actions[i] = random.choice(valid_moves)
            else:
                # Exploitation
                action_indices = torch.tensor(
                    [self._move_to_action_index(move) for move in valid_moves],
                    device=self.device,
                    dtype=torch.long
                )
                valid_q_values = q_values[0, action_indices]
                
                exploration_bonuses = torch.zeros(len(valid_moves), device=self.device)
                for j, move in enumerate(valid_moves):
                    uncertainty = self.get_move_uncertainty(move, uncertainty_map)
                    exploration_bonuses[j] = uncertainty * self.uncertainty_exploration_multiplier
                
                modified_q_values = valid_q_values + exploration_bonuses
                best_idx = torch.argmax(modified_q_values).item()
                actions[i] = valid_moves[best_idx]
                
                # Store action-PBS state for feedback (only if using PBS)
                if self.pbs_instances and game_states and game_states[i]:
                    self.store_action_pbs_state(actions[i], base_q_values, uncertainty_map, game_states[i], env_idx=i)
        
        return actions

    def update_pbs_batch(self, actions, game_states, acting_player):
        """Update PBS for a batch of actions."""
        if not self.pbs_instances:
            return
            
        for i, (action, gs) in enumerate(zip(actions, game_states)):
            if action is not None and gs is not None:
                # Use the corresponding PBS instance
                if i < len(self.pbs_instances):
                    # Retrieve Q-value feedback if available
                    action_q_value = None
                    if i in self.action_pbs_buffer:
                        for stored in self.action_pbs_buffer[i]:
                            if stored['action'] == action:
                                action_q_value = stored['q_value']
                                break
                    
                    # Update PBS with action and Q-value
                    self.pbs_instances[i].update_from_action(action, gs, acting_player, q_value=action_q_value)

    def update_pbs_from_reveal_batch(self, reveals_list, game_phase='middle', turn_count=0):
        """Update PBS for a batch of reveals."""
        if not self.pbs_instances:
            return
            
        for i, reveals in enumerate(reveals_list):
            if reveals and i < len(self.pbs_instances):
                for pos, piece_type in reveals:
                    self.pbs_instances[i].update_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=turn_count)
    
    def calculate_uncertainty_aware_q_values(self, base_q_values: torch.Tensor,
                                             valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                             uncertainty_map: Dict[Tuple[int, int], float]) -> torch.Tensor:
        """
        Integrate PBS uncertainty into Q-value calculations.
        Reduces Q-values for actions in uncertain areas.
        
        Args:
            base_q_values: Base Q-values from network
            valid_moves: List of valid moves
            uncertainty_map: Dictionary mapping positions to uncertainty values
            
        Returns:
            Uncertainty-adjusted Q-values
        """
        q_values = base_q_values.clone()
        
        # Apply uncertainty penalty to each valid action
        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            uncertainty_penalty = self.get_uncertainty_penalty(move, uncertainty_map)
            q_values[0, action_idx] -= uncertainty_penalty * self.uncertainty_penalty_scale
        
        return q_values
    
    def get_uncertainty_penalty(self, move: Tuple[Tuple[int, int], Tuple[int, int]],
                                uncertainty_map: Dict[Tuple[int, int], float]) -> float:
        """
        Get uncertainty penalty for a move.
        Considers uncertainty at both source and destination.
        
        Args:
            move: Action tuple
            uncertainty_map: Dictionary mapping positions to uncertainty values
            
        Returns:
            Uncertainty penalty value
        """
        (r_from, c_from), (r_to, c_to) = move
        from_pos = (r_from, c_from)
        to_pos = (r_to, c_to)
        
        # Average uncertainty at source and destination
        from_uncertainty = uncertainty_map.get(from_pos, 0.0)
        to_uncertainty = uncertainty_map.get(to_pos, 0.0)
        avg_uncertainty = (from_uncertainty + to_uncertainty) / 2.0
        
        return float(avg_uncertainty)
    
    def get_move_uncertainty(self, move: Tuple[Tuple[int, int], Tuple[int, int]],
                            uncertainty_map: Dict[Tuple[int, int], float]) -> float:
        """
        Get uncertainty for a move (for exploration bonus).
        
        Args:
            move: Action tuple
            uncertainty_map: Dictionary mapping positions to uncertainty values
            
        Returns:
            Uncertainty value
        """
        (r_from, c_from), (r_to, c_to) = move
        from_pos = (r_from, c_from)
        to_pos = (r_to, c_to)
        
        # Average uncertainty at source and destination
        from_uncertainty = uncertainty_map.get(from_pos, 0.0)
        to_uncertainty = uncertainty_map.get(to_pos, 0.0)
        return (from_uncertainty + to_uncertainty) / 2.0
    
    def get_average_uncertainty(self, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                               uncertainty_map: Dict[Tuple[int, int], float]) -> float:
        """
        Get average uncertainty across all valid moves.
        
        Args:
            valid_moves: List of valid moves
            uncertainty_map: Dictionary mapping positions to uncertainty values
            
        Returns:
            Average uncertainty value
        """
        if not valid_moves:
            return 0.0
        
        uncertainties = [self.get_move_uncertainty(move, uncertainty_map) for move in valid_moves]
        return sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    
    def store_action_pbs_state(self, action: Tuple[Tuple[int, int], Tuple[int, int]],
                               q_values: torch.Tensor, uncertainty_map: Dict[Tuple[int, int], float],
                               game_state, env_idx: int = 0):
        """
        Store action with PBS state and Q-values for feedback learning.
        
        Args:
            action: Action taken
            q_values: Q-values from network
            uncertainty_map: Uncertainty map
            game_state: Current game state
            env_idx: Environment index
        """
        action_idx = self._move_to_action_index(action)
        action_q_value = q_values[0, action_idx].item()
        
        # Get uncertainty for this action
        action_uncertainty = self.get_move_uncertainty(action, uncertainty_map)
        
        if env_idx not in self.action_pbs_buffer:
            self.action_pbs_buffer[env_idx] = deque(maxlen=50)
            
        self.action_pbs_buffer[env_idx].append({
            'action': action,
            'q_value': action_q_value,
            'uncertainty': action_uncertainty,
            'game_state': game_state
        })
    
    def compute_combined_reward(self, game_reward: float, pbs_quality_reward: Optional[float] = None,
                               dqn_q_value: Optional[float] = None) -> Dict[str, float]:
        """
        Combine rewards from multiple sources:
        - game_reward: Standard game reward (win/loss/draw)
        - pbs_quality_reward: PBS prediction quality reward
        - dqn_q_value: Q-value confidence reward
        
        Args:
            game_reward: Standard game reward
            pbs_quality_reward: PBS quality reward (optional)
            dqn_q_value: Q-value for confidence (optional)
            
        Returns:
            Dictionary with combined reward components
        """
        combined = {
            'game_reward': game_reward * 1.0,
            'pbs_quality': 0.0,
            'q_confidence': 0.0,
            'total': game_reward
        }
        
        # Add PBS quality reward (weighted contribution)
        if pbs_quality_reward is not None:
            combined['pbs_quality'] = pbs_quality_reward * 0.3
            combined['total'] += combined['pbs_quality']
        
        # Add Q-value confidence reward (normalized)
        if dqn_q_value is not None:
            normalized_q = self.normalize_q_value(dqn_q_value)
            combined['q_confidence'] = normalized_q * 0.2
            combined['total'] += combined['q_confidence']
        
        return combined
    
    def normalize_q_value(self, q_value: float) -> float:
        """
        Normalize Q-value to [0, 1] range for reward shaping.
        
        Args:
            q_value: Raw Q-value
            
        Returns:
            Normalized Q-value in [0, 1]
        """
        # Simple normalization: use sigmoid to map to [0, 1]
        # Adjust scale based on typical Q-value range
        return 1.0 / (1.0 + math.exp(-q_value / 10.0))
    
    def train_pbs_evaluator(self, epochs: int = 1) -> Optional[float]:
        """
        Train the PBS Evaluator if available.
        Handles both single and parallel environments.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Average loss value or None
        """
        if self.pbs is None:
            return None
        
        # Train separate PBS instances if using parallel envs
        if self.num_envs > 1 and self.pbs_instances:
            total_loss = 0.0
            count = 0
            for pbs in self.pbs_instances:
                loss = pbs.train_evaluator(epochs=epochs)
                if loss is not None:
                    total_loss += loss
                    count += 1
            return total_loss / count if count > 0 else None
        else:
            return self.pbs.train_evaluator(epochs=epochs)
    
    def update_performance_metrics(self, pbs_accuracy: Optional[float] = None,
                                  dqn_loss: Optional[float] = None,
                                  action_alignment: Optional[float] = None):
        """
        Update cross-system performance metrics.
        
        Args:
            pbs_accuracy: PBS prediction accuracy (0-1)
            dqn_loss: DQN training loss
            action_alignment: Alignment between PBS predictions and DQN actions (0-1)
        """
        if pbs_accuracy is not None:
            self.performance_metrics['pbs_accuracy_trend'].append(pbs_accuracy)
        if dqn_loss is not None:
            self.performance_metrics['dqn_loss_trend'].append(dqn_loss)
        if action_alignment is not None:
            self.performance_metrics['action_prediction_alignment'].append(action_alignment)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get summary of performance metrics.
        
        Returns:
            Dictionary with average metrics
        """
        summary = {}
        
        if self.performance_metrics['pbs_accuracy_trend']:
            summary['avg_pbs_accuracy'] = sum(self.performance_metrics['pbs_accuracy_trend']) / len(self.performance_metrics['pbs_accuracy_trend'])
        else:
            summary['avg_pbs_accuracy'] = 0.0
        
        if self.performance_metrics['dqn_loss_trend']:
            summary['avg_dqn_loss'] = sum(self.performance_metrics['dqn_loss_trend']) / len(self.performance_metrics['dqn_loss_trend'])
        else:
            summary['avg_dqn_loss'] = 0.0
        
        if self.performance_metrics['action_prediction_alignment']:
            summary['avg_action_alignment'] = sum(self.performance_metrics['action_prediction_alignment']) / len(self.performance_metrics['action_prediction_alignment'])
        else:
            summary['avg_action_alignment'] = 0.0
        
        return summary
    
    def detect_pbs_dqn_misalignment(self) -> bool:
        """
        Detect if PBS and DQN are misaligned.
        
        Returns:
            True if misalignment detected
        """
        if len(self.performance_metrics['pbs_accuracy_trend']) < 10:
            return False
        
        # Check if PBS accuracy is low while DQN loss is also high
        recent_pbs = list(self.performance_metrics['pbs_accuracy_trend'])[-10:]
        recent_dqn = list(self.performance_metrics['dqn_loss_trend'])[-10:]
        
        avg_pbs = sum(recent_pbs) / len(recent_pbs)
        avg_dqn = sum(recent_dqn) / len(recent_dqn)
        
        # Misalignment: low PBS accuracy (< 0.5) and high DQN loss (> 0.5)
        return avg_pbs < 0.5 and avg_dqn > 0.5
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """
        Analyze trends and recommend system adjustments.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        if self.detect_pbs_dqn_misalignment():
            recommendations.append({
                'type': 'balance_adjustment',
                'suggestion': 'Increase PBS weight in DQN decision making',
                'priority': 'high'
            })
        
        # Check if PBS accuracy is consistently low
        if self.performance_metrics['pbs_accuracy_trend']:
            recent_pbs = list(self.performance_metrics['pbs_accuracy_trend'])[-20:]
            if len(recent_pbs) >= 10:
                avg_pbs = sum(recent_pbs) / len(recent_pbs)
                if avg_pbs < 0.4:
                    recommendations.append({
                        'type': 'pbs_improvement',
                        'suggestion': 'PBS accuracy is low - consider increasing training frequency',
                        'priority': 'medium'
                    })
        
        # Check if DQN loss is consistently high
        if self.performance_metrics['dqn_loss_trend']:
            recent_dqn = list(self.performance_metrics['dqn_loss_trend'])[-20:]
            if len(recent_dqn) >= 10:
                avg_dqn = sum(recent_dqn) / len(recent_dqn)
                if avg_dqn > 1.0:
                    recommendations.append({
                        'type': 'dqn_improvement',
                        'suggestion': 'DQN loss is high - consider adjusting learning rate',
                        'priority': 'medium'
                    })
        
        return recommendations
    
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
