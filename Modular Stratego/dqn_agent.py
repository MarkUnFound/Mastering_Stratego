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
from prioritized_memory import PrioritizedReplayBuffer

# Define a named tuple for experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ConvDQN(nn.Module):
    """Convolutional Deep Q-Network for Stratego"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (15, 10, 10), output_size: int = 1000):
        """
        Initialize the ConvDQN network
        
        Args:
            input_shape: Shape of input (channels, height, width)
            output_size: Size of output (number of possible actions)
        """
        super(ConvDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size: 64 * 10 * 10 = 6400
        self.flatten_size = 64 * 10 * 10
        
        # Dueling Architecture
        # Value stream: State -> Value V(s)
        self.value_fc = nn.Linear(self.flatten_size, 512)
        self.value_out = nn.Linear(512, 1)
        
        # Advantage stream: State -> Advantage A(s, a)
        self.advantage_fc = nn.Linear(self.flatten_size, 512)
        self.advantage_out = nn.Linear(512, output_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Value stream
        val = F.relu(self.value_fc(x))
        val = self.value_out(val)
        
        # Advantage stream
        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage_out(adv)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))


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
        
        # Exploration cycling and stagnation recovery
        self.epsilon_cycle_len = 1000
        self.stagnation_threshold = 50
        self.loss_history = deque(maxlen=100)
        self.base_lr = lr
        
        self.use_pbs = use_pbs
        
        # Probabilistic Belief State
        self.num_envs = num_envs
        
        # Probabilistic Belief State
        self.pbs = None
        self.pbs_instances = []
        self.action_pbs_buffer = {} # Buffer for storing action-PBS states for feedback
        
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
        self.uncertainty_exploration_multiplier = 0.05  # Reduced from 0.5 to prevent excessive randomness
        self.uncertainty_penalty_scale = 0.5  # Increased from 0.3 to encourage risk aversion (safety)
        
        # Neural networks (keep on GPU, no compilation for Windows compatibility)
        # Neural networks (keep on GPU, no compilation for Windows compatibility)
        self.q_network = ConvDQN(input_shape=(15, 10, 10), output_size=action_size).to(device)
        self.target_network = ConvDQN(input_shape=(15, 10, 10), output_size=action_size).to(device)
        
        # Enable cuDNN benchmarking for faster convolutions (if using conv layers)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, non-deterministic
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Exploitability Critic
        self.critic = ExploitabilityCritic(input_shape=(15, 10, 10), output_size=action_size).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01)
        self.critic_loss_fn = nn.CrossEntropyLoss()
        self.critic_weight = 0.1  # Relative weight (10% of reward magnitude) - Normalized to reward scale
        self.entropy_bonus = 0.02  # Bonus for action diversity
        
        # Experience replay - Prioritized (GPU-accelerated)
        self.memory = PrioritizedReplayBuffer(buffer_size, device=device)
        self.beta = 0.4  # Initial importance sampling weight
        self.beta_increment = 0.00001  # Increment per step
        
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
        
        # Learning rate scheduling with automatic adjustment
        self.initial_lr = lr
        self.lr_decay_factor = 0.5  # Reduce LR by half every 500k steps
        self.lr_decay_interval = 500_000
        self.min_lr = lr * 0.01  # Minimum learning rate (1% of initial)
        
        # Automatic learning rate adjustment based on loss trends
        self.loss_history_for_lr = deque(maxlen=200)  # Track losses for LR adjustment
        self.lr_adjustment_interval = 50  # Check every 50 training steps
        self.lr_adjustment_threshold = 1.5  # Changed from 1.2 (less sensitive to spikes)
        self.lr_reduction_factor = 0.5  # Reduce LR by half if needed
        self.lr_increase_factor = 1.1  # Increase LR by 10% when loss is very low
        self.lr_increase_threshold = 0.1  # If loss < 0.1, consider increasing LR
        self.high_loss_threshold = 100.0  # Threshold for aggressive reduction (was 20.0)
        self.critical_loss_threshold = 200.0  # Threshold for critical reduction (was 50.0)
        
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
        
        # Mixed Precision Training
        self.scaler = torch.amp.GradScaler('cuda')
        
    def reset(self):
        """Reset the DQN agent by reinitializing networks and optimizer"""
        # Reinitialize Q-network and target network
        # Reinitialize Q-network and target network
        self.q_network = ConvDQN(input_shape=(15, 10, 10), output_size=self.action_size).to(self.device)
        self.target_network = ConvDQN(input_shape=(15, 10, 10), output_size=self.action_size).to(self.device)
        
        # Reinitialize optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        # Reset epsilon to initial value
        self.epsilon = 1.0
        # Clear memory
        self.memory.clear()
        # Clear policy losses
        self.policy_losses = []
        self.q_values_history = []
        self.entropy_history = []
        # Reset smoothed loss
        self.smoothed_loss = None
        # Reset loss history for LR adjustment
        self.loss_history_for_lr = deque(maxlen=200)
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
            # With PER, we don't strictly need triple storage as priority handles importance
            # But we can give it a high initial error if we want to force priority
            # For now, just add normally, PER will prioritize if TD error is high
            self.memory.add(experience)
        else:
            self.memory.add(experience)
        
        # Increment step counter for epsilon decay
        self.step_count += 1

    def sample_replay_batch(self) -> Optional[List[Experience]]:
        """Sample a batch from the replay buffer without performing an update."""
        if len(self.memory) < self.batch_size:
            return None
        batch, _, _ = self.memory.sample(self.batch_size, self.beta)
        return batch
        
    def enable_search(self, num_simulations: int = 50, endgame_threshold: int = 15):
        """Enable hybrid search for endgame using ISMCTS."""
        from ismcts_agent import ISMCTSAgent
        self.search_agent = ISMCTSAgent(self, num_simulations=num_simulations)
        self.endgame_threshold = endgame_threshold
        print(f"🔍 Hybrid Search (ISMCTS) enabled for {self.name} (Sims={num_simulations}, Threshold={endgame_threshold})")

    def is_endgame(self, game_state) -> bool:
        """Check if the game is in the endgame phase."""
        if not hasattr(self, 'endgame_threshold'):
            return False
            
        # Count total pieces on board
        total_pieces = 0
        if hasattr(game_state, 'board'):
            board = game_state.board
            if isinstance(board, torch.Tensor):
                total_pieces = (board != 0).sum().item()
            else:
                total_pieces = np.count_nonzero(board)
        
        return total_pieces <= self.endgame_threshold

    def act(self, state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], game_state=None):
        """
        Choose action using epsilon-greedy policy.
        
        Workflow:
        1. PBS first gets the value and creates possible values with confidence scores
        2. DQN then calculates Q-value using PBS-enhanced state
        
        Args:
            state: Current state (board)
            valid_moves: List of valid moves
            game_state: Full game state object (for PBS)
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
            
        # Check for Hybrid Search (ISMCTS) in endgame
        if hasattr(self, 'search_agent') and self.search_agent and game_state:
            if self.is_endgame(game_state):
                # Use ISMCTS
                best_move = self.search_agent.act(game_state, valid_moves)
                if best_move:
                    return best_move
        
        # Exploitation
        # Get state representation (handles PBS internally)
        state_tensor = self.get_state_representation(state, pbs_instance=self.pbs)
        
        # Get uncertainty map if PBS is enabled
        uncertainty_map = {}
        if self.pbs and game_state:
            uncertainty_map = self.pbs.get_uncertainty_map(game_state)
            
        # Ensure state_tensor has batch dimension (B, C, H, W)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.q_network.eval()
        with torch.no_grad():
            base_q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        # Calculate uncertainty aware Q-values
        q_values = self.calculate_uncertainty_aware_q_values(
            base_q_values, valid_moves, uncertainty_map
        )
        
        # Filter valid moves
        valid_q_values = []
        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            
            # Add exploration bonus based on uncertainty
            uncertainty = self.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
            
            valid_q_values.append(q_values[0, action_idx].item() + exploration_bonus)
            
        best_move_idx = np.argmax(valid_q_values)
        
        # Store action-PBS state for feedback (only if using PBS)
        if self.pbs and game_state:
            best_move = valid_moves[best_move_idx]
            self.store_action_pbs_state(best_move, base_q_values, uncertainty_map, game_state)
            
        return valid_moves[best_move_idx]
        
    def replay(self, batch: Optional[List[Experience]] = None) -> Optional[float]:
        """
        Train the model on a batch of experiences - optimized for GPU
        
        Returns:
            Policy loss value or None if not enough experiences
        """
        if batch is None:
            if len(self.memory) < self.batch_size:
                return None
            # Sample with priorities
            batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
            
            # Anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # If batch provided externally (e.g. prefetcher), we assume it's just the batch list
            # This breaks PER update logic if prefetcher doesn't return idxs/weights
            # For now, assume standard internal sampling if batch is None
            # If using prefetcher, it needs to be updated to return (batch, idxs, weights)
            # Fallback for external batch: uniform weights, no update
            idxs = []
            is_weights = np.ones(len(batch))
        
        # Stack states and next_states (already on GPU from remember())
        states = torch.stack([e.state for e in batch])
        next_states = torch.stack([e.next_state for e in batch])
        
        # Create tensors directly on GPU (avoid CPU intermediate)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=self.device)
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        
        # --- 1. Train Critic First ---
        # Predict action from state
        with torch.amp.autocast('cuda'):
            critic_logits = self.critic(states)
            critic_loss = self.critic_loss_fn(critic_logits, actions)
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        
        # --- 2. Calculate Predictability Penalty ---
        # We use the updated critic to calculate the penalty
        with torch.no_grad():
            probs = F.softmax(self.critic(states), dim=1)
            # Get probability assigned to the taken action
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            # Penalty = critic_weight * probability * reward_scale
            # Normalize penalty to the scale of rewards in the batch
            reward_scale = torch.mean(torch.abs(rewards)).detach() + 1e-6
            penalty = self.critic_weight * action_probs * reward_scale
            avg_penalty = penalty.mean().item()
            
            # Calculate entropy for metrics
            # Entropy = -sum(p * log(p))
            entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            
        # --- 3. Train Agent with Penalized Rewards ---
        with torch.amp.autocast('cuda'):
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate average Q-value for metrics
        avg_q_value = current_q_values.mean().item()
        
        # Next Q values from target network
        # Double DQN: Use online network to select action, target network to evaluate it
        with torch.no_grad():
            # Select best action using online network
            next_state_actions = self.q_network(next_states).max(1)[1]
            # Evaluate that action using target network
            next_q_values = self.target_network(next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze()
            
        # Penalize rewards: Reward - Penalty
        # This encourages the agent to choose actions that are less predictable
        penalized_rewards = rewards - penalty
        
        # Calculate target Q-values
        target_q_values = penalized_rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with importance sampling weights
        # Smooth L1 loss per element (reduction='none')
        loss_elementwise = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        loss = (loss_elementwise * weights).mean()
        
        # Update priorities
        if idxs:
            td_errors = torch.abs(target_q_values - current_q_values.squeeze()).detach().cpu().numpy()
            self.memory.update(idxs, td_errors)
        
        # Optimize Agent
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # --- 4. Logging & Updates ---
        
        # Track policy loss (minimize CPU transfer - only get item() once)
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        
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
        
        # Track loss for automatic LR adjustment
        self.loss_history_for_lr.append(loss_value)
        
        # Learning rate scheduling: reduce LR over time for fine-tuning
        lr_decay_steps = self.step_count // self.lr_decay_interval
        base_lr = self.initial_lr * (self.lr_decay_factor ** lr_decay_steps)
        base_lr = max(base_lr, self.min_lr)
        
        # Automatic learning rate adjustment based on loss trends
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # DEBUG: Detailed logging for low loss investigation
        if self.step_count % 100 == 0:
            print(f"\n🔍 DEBUG [{self.name} Step {self.step_count}]:")
            print(f"  Loss: {loss_value:.6f} (Smoothed: {self.smoothed_loss:.6f})")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Avg Q-Value: {avg_q_value:.4f}")
            print(f"  Avg Penalty: {avg_penalty:.4f}")
            print(f"  Critic Loss: {critic_loss.item():.4f}")
            print(f"  Rewards (Batch): Min={rewards.min().item():.2f}, Max={rewards.max().item():.2f}, Mean={rewards.mean().item():.2f}")
            print(f"  Target Q (Batch): Min={target_q_values.min().item():.2f}, Max={target_q_values.max().item():.2f}, Mean={target_q_values.mean().item():.2f}")
            print(f"  Current Q (Batch): Min={current_q_values.min().item():.2f}, Max={current_q_values.max().item():.2f}, Mean={current_q_values.mean().item():.2f}")
            
            if loss_value < 1e-6:
                print("  ⚠️  EXTREMELY LOW LOSS DETECTED!")
        
        if len(self.loss_history_for_lr) >= self.lr_adjustment_interval:
            recent_losses = list(self.loss_history_for_lr)[-self.lr_adjustment_interval:]
            older_losses = list(self.loss_history_for_lr)[-self.lr_adjustment_interval * 2:-self.lr_adjustment_interval]
            
            if len(older_losses) > 0:
                recent_avg = sum(recent_losses) / len(recent_losses)
                older_avg = sum(older_losses) / len(older_losses)
                
                # If average loss is consistently very high, reduce LR
                if recent_avg > self.high_loss_threshold:
                    reduction_factor = 0.5 if recent_avg > self.critical_loss_threshold else 0.8
                    new_lr = current_lr * reduction_factor
                    new_lr = max(new_lr, self.min_lr)
                    if new_lr < current_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"📉 LR reduced to {new_lr:.2e} due to high loss ({recent_avg:.2f})")
                
                # If loss increased significantly, reduce learning rate
                elif older_avg > 0 and recent_avg > older_avg * self.lr_adjustment_threshold:
                    new_lr = current_lr * self.lr_reduction_factor
                    new_lr = max(new_lr, self.min_lr)
                    if new_lr < current_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                
                # If loss is very low (near zero), it might mean we're stuck or overfitting
                # We should try to INCREASE LR to kickstart learning
                elif recent_avg < 1e-5:
                     # KICKSTART: Boost LR significantly if loss is dead zero
                     new_lr = min(current_lr * 2.0, self.initial_lr * 5.0) # Cap at 5x initial
                     if new_lr > current_lr:
                         for param_group in self.optimizer.param_groups:
                             param_group['lr'] = new_lr
                         print(f"🚀 LR boosted to {new_lr:.2e} to kickstart learning (Loss ~ 0)")
                
                # If loss is low and stable (but not dead zero), consider increasing LR slightly
                elif recent_avg < self.lr_increase_threshold and recent_avg < older_avg * 0.9:
                    new_lr = min(current_lr * self.lr_increase_factor, self.initial_lr)
                    if new_lr > current_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                else:
                    # Use base LR from time-based decay
                    if abs(base_lr - current_lr) > 1e-8:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = base_lr
            else:
                if abs(base_lr - current_lr) > 1e-8:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = base_lr
        else:
            if abs(base_lr - current_lr) > 1e-8:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
        
        # Epsilon Cycling and Decay
        if self.epsilon_cycle_len > 0:
            # Cyclic epsilon: restarts every cycle_len steps
            cycle_progress = (self.step_count % self.epsilon_cycle_len) / self.epsilon_cycle_len
            # Decay from 1.0 to epsilon_min within the cycle
            # Use a cosine decay for smoother transitions or linear? Linear is fine.
            # Let's use a restart strategy: 
            # Epsilon starts high (0.5) at start of cycle, decays to min.
            cycle_epsilon_start = 0.5
            current_cycle_epsilon = self.epsilon_min + (cycle_epsilon_start - self.epsilon_min) * (1.0 - cycle_progress)
            
            # Combine with standard decay (which might be global)
            # Actually, let's override the standard decay if cycling is enabled
            self.epsilon = max(self.epsilon_min, current_cycle_epsilon)
        else:
            # Standard Gradual epsilon decay
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
                # Aggressive Stagnation Recovery
                # Increase epsilon significantly (cap at 0.5) to force exploration
                print(f"⚠️  Stagnation detected ({self.stagnation_episodes} episodes). Boosting epsilon!")
                self.epsilon = min(0.5, self.epsilon * 2.0 + 0.1) 
                self.stagnation_episodes = 0  # Reset counter after adjustment
                # Reset best_avg_reward slightly to allow climbing back up
                self.best_avg_reward = recent_avg * 0.95
            
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
        
    def reset_pbs(self, env_idx: int = None):
        """
        Reset PBS state for a specific environment (or all if env_idx is None).
        
        Args:
            env_idx: Index of the environment to reset. If None, resets all.
        """
        if self.num_envs > 1:
            if env_idx is not None:
                # Reset specific instance
                if 0 <= env_idx < len(self.pbs_instances):
                    self.pbs_instances[env_idx].reset()
                    # Also clear action buffer for this env
                    if env_idx in self.action_pbs_buffer:
                        self.action_pbs_buffer[env_idx].clear()
            else:
                # Reset all
                for pbs in self.pbs_instances:
                    pbs.reset()
                for i in range(self.num_envs):
                    if i in self.action_pbs_buffer:
                        self.action_pbs_buffer[i].clear()
        elif self.pbs:
            # Single environment
            self.pbs.reset()
            if 0 in self.action_pbs_buffer:
                self.action_pbs_buffer[0].clear()

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
        Convert game state to neural network input - returns GPU tensor.
        """
        # Use provided PBS instance or default self.pbs
        pbs = pbs_instance if pbs_instance else self.pbs
        
        # Step 1: PBS gets the multi-channel state
        if pbs and hasattr(game_state, 'board'):
            state = pbs.get_multi_channel_state(game_state)
            # Ensure it's on the correct device
            if state.device != self.device:
                state = state.to(self.device)
        else:
            # Fallback for no PBS (shouldn't happen in this config)
            # Create a basic 15-channel tensor with just board info
            state = torch.zeros((15, 10, 10), device=self.device, dtype=torch.float32)
            if hasattr(game_state, 'board'):
                board = game_state.board
                if isinstance(board, torch.Tensor):
                    board = board.to(self.device)
                else:
                    board = torch.tensor(board, device=self.device)
                
                # Channel 0: Own pieces (positive)
                if self.player_id == 1:
                    mask = (board > 0)
                    state[0][mask] = board[mask].float()
                else:
                    mask = (board < 0) & (board != -13) & (board != -20)
                    state[0][mask] = board[mask].abs().float()
                
                # Channel 1: Lakes
                state[1] = (board == -13).float()
        
        # Ensure float type
        if state.dtype != torch.float32:
            state = state.float()
            
        return state

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
                
            # Epsilon-greedy
            if np.random.rand() <= self.epsilon:
                actions[i] = random.choice(valid_moves)
            else:
                # Exploitation
                # Calculate uncertainty aware Q-values for this batch item
                q_values = self.calculate_uncertainty_aware_q_values(
                    base_q_values_batch[i].unsqueeze(0), 
                    valid_moves, 
                    uncertainty_maps[i]
                )
                
                # Filter valid moves and add exploration bonus
                valid_q_values = []
                for move in valid_moves:
                    action_idx = self._move_to_action_index(move)
                    uncertainty = self.get_move_uncertainty(move, uncertainty_maps[i])
                    exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
                    valid_q_values.append(q_values[0, action_idx].item() + exploration_bonus)
                
                best_move_idx = np.argmax(valid_q_values)
                actions[i] = valid_moves[best_move_idx]
                
            # Store action-PBS state for feedback (only if using PBS)
            if self.pbs_instances and game_states and game_states[i]:
                self.store_action_pbs_state(actions[i], base_q_values_batch[i].unsqueeze(0), uncertainty_maps[i], game_states[i], env_idx=i)
        
        return actions

    def get_state_value(self, game_state) -> float:
        """
        Get the Value V(s) of a state for Minimax heuristic.
        
        Args:
            game_state: Current game state
            
        Returns:
            Value of the state (scalar float)
        """
        state_tensor = self.get_state_representation(game_state)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        self.q_network.eval()
        with torch.no_grad():
            # ConvDQN returns Q(s,a), but internally computes V(s).
            # We can either modify ConvDQN to return V(s) or approximate V(s) = max Q(s,a)
            # Since we have Dueling DQN, V(s) is explicitly computed but combined.
            # max Q(s,a) is a good approximation of V(s) for the greedy policy.
            q_values = self.q_network(state_tensor)
            value = q_values.max().item()
            
            
        self.q_network.train()
        return value

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
    
    def update_pbs_from_reveal(self, revealed_pieces: List[Tuple[Tuple[int, int], PieceType]], env_idx: int = 0, game_phase: str = 'middle', turn_count: int = 0):
        """
        Update PBS with ground truth from revealed pieces.
        
        Args:
            revealed_pieces: List of ((row, col), piece_type) tuples
            env_idx: Environment index (for parallel envs)
            game_phase: Current game phase
            turn_count: Current turn count
        """
        if self.pbs is None or not revealed_pieces:
            return
            
        target_pbs = self.pbs
        if self.num_envs > 1 and self.pbs_instances:
                target_pbs = self.pbs_instances[env_idx]
        
        for pos, piece_type in revealed_pieces:
            target_pbs.update_from_reveal(pos, piece_type, game_phase=game_phase, turn_count=turn_count)

    def train_pbs_evaluator(self, epochs: int = 1) -> Optional[float]:
        """
        Train the PBS Evaluator if available.
        Handles both single and parallel environments efficiently.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Average loss value or None
        """
        if self.pbs is None:
            return None
        
        total_loss = 0.0
        loss_count = 0
        
        # 1. Train Shared Evaluator (ONCE per step)
        if self.shared_evaluator is not None:
            # Shared evaluator has its own memory buffer that is populated by all envs
            # So we only need to call train() once
            eval_loss = self.shared_evaluator.train(epochs=epochs)
            if eval_loss is not None:
                total_loss += eval_loss
                loss_count += 1
        
        # 2. Train Shared AAREN (ONCE per step)
        if self.shared_aaren is not None and self.num_envs > 1 and self.pbs_instances:
            # Aggregate training data from all PBS instances
            all_action_sequences = []
            all_true_piece_types = []
            all_evaluator_weights = []
            all_positions = []
            
            for pbs in self.pbs_instances:
                # Extract data from each instance
                if hasattr(pbs, 'get_aaren_training_data'):
                    sequences, types, weights, positions = pbs.get_aaren_training_data()
                    all_action_sequences.extend(sequences)
                    all_true_piece_types.extend(types)
                    all_evaluator_weights.extend(weights)
                    all_positions.extend(positions)
            
            # Train shared AAREN once with aggregated data
            if all_action_sequences:
                # Use the first PBS instance to drive the training
                self.pbs.train_aaren(
                    action_sequences=all_action_sequences,
                    true_piece_types=all_true_piece_types,
                    epochs=epochs,
                    evaluator_weights=all_evaluator_weights,
                    positions=all_positions
                )
        
        # 3. Fallback: Independent Training (if not shared)
        elif self.num_envs > 1 and self.pbs_instances:
            # If models are NOT shared, we must train each one
            for pbs in self.pbs_instances:
                loss = pbs.train_evaluator(epochs=epochs)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
        else:
            # Single environment case
            return self.pbs.train_evaluator(epochs=epochs)
            
        
        if loss_count > 0:
            return total_loss / loss_count
        return None

    
    def detect_pbs_dqn_misalignment(self) -> bool:
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
