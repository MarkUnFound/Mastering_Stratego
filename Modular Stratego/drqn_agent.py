"""
DQN Agent for Stratego Game (DRQN Version)
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
from prioritized_memory import SequentialReplayBuffer, Experience
from training_config import TRACE_LENGTH

class DRQN(nn.Module):
    """Deep Recurrent Q-Network for Stratego"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (15, 10, 10), output_size: int = 1000):
        """
        Initialize the DRQN network
        
        Args:
            input_shape: Shape of input (channels, height, width)
            output_size: Size of output (number of possible actions)
        """
        super(DRQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        
        # CNN Layers (Feature Extractor)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size: 64 * 10 * 10 = 6400
        self.flatten_size = 64 * 10 * 10
        
        # LSTM Layer for memory retention
        # Input: (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=self.flatten_size, hidden_size=512, batch_first=True)
        
        # Dueling Architecture Heads (from LSTM output)
        # Value stream: State -> Value V(s)
        self.value_fc = nn.Linear(512, 512)
        self.value_out = nn.Linear(512, 1)
        
        # Advantage stream: State -> Advantage A(s, a)
        self.advantage_fc = nn.Linear(512, 512)
        self.advantage_out = nn.Linear(512, output_size)
        
    def forward(self, x, hidden_state=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels, height, width)
            hidden_state: Tuple (h_0, c_0) for LSTM
            
        Returns:
            q_values: Tensor of shape (batch, seq_len, output_size)
            new_hidden_state: Tuple (h_n, c_n)
        """
        # Handle input shape
        if x.dim() == 4: # (batch, C, H, W) - Single step inference
            x = x.unsqueeze(1) # Add seq_len=1 -> (batch, 1, C, H, W)
            
        batch_size, seq_len, C, H, W = x.size()
        
        # Merge batch and seq_len for CNN processing
        x = x.view(batch_size * seq_len, C, H, W)
        
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, seq_len, -1)  # Flatten to (batch, seq_len, features)
        
        # LSTM Processing
        # self.lstm.flatten_parameters() # Optimization for GPU
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # Dueling Heads
        # Value stream
        val = F.relu(self.value_fc(lstm_out))
        val = self.value_out(val)
        
        # Advantage stream
        adv = F.relu(self.advantage_fc(lstm_out))
        adv = self.advantage_out(adv)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = val + (adv - adv.mean(dim=2, keepdim=True))
        
        return q_values, new_hidden_state


class DRQNAgent:
    """DRQN Agent for Stratego with sequential experience replay"""
    
    def __init__(self, player_id: int, device, 
                 state_size: int = 200, action_size: int = 1000,
                 lr: float = 0.00001, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_min: float = 0.1, 
                 epsilon_decay: float = 0.001, 
                 buffer_size: int = 1000, batch_size: int = 32,
                 use_pbs: bool = True, num_envs: int = 1):
        """
        Initialize the DRQN agent
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
        self.name = f"DRQN Agent {player_id}"
        self.num_envs = num_envs
        
        # Exploration cycling and stagnation recovery
        self.epsilon_cycle_len = 1000
        self.stagnation_threshold = 50
        self.loss_history = deque(maxlen=100)
        self.base_lr = lr
        
        self.use_pbs = use_pbs
        
        # Probabilistic Belief State
        self.pbs = None
        self.pbs_instances = []
        self.action_pbs_buffer = {} 
        
        if self.use_pbs:
            if num_envs > 1:
                # Create shared models for parallel environments
                self.shared_aaren = PieceActionAaren(
                    input_size=24, hidden_size=64, num_layers=3, output_size=12, device=device
                ).to(device)
                self.shared_aaren_optimizer = optim.AdamW(self.shared_aaren.parameters(), lr=0.001, weight_decay=0.01)
                
                self.shared_evaluator = None
                if PBS_EVALUATOR_AVAILABLE:
                    self.shared_evaluator = PBSEvaluator(device=device)
                
                for _ in range(num_envs):
                    pbs_instance = ProbabilisticBeliefState(
                        player_id, device, 
                        shared_aaren_model=self.shared_aaren,
                        shared_evaluator=self.shared_evaluator
                    )
                    pbs_instance.aaren_optimizer = self.shared_aaren_optimizer
                    self.pbs_instances.append(pbs_instance)
                self.pbs = self.pbs_instances[0]
            else:
                self.pbs = ProbabilisticBeliefState(player_id, device)
                self.pbs_instances = [self.pbs]
        
        # Uncertainty parameters
        self.uncertainty_exploration_multiplier = 0.05
        self.uncertainty_penalty_scale = 0.5
        
        # DRQN Networks
        self.q_network = DRQN(input_shape=(15, 10, 10), output_size=action_size).to(device)
        self.target_network = DRQN(input_shape=(15, 10, 10), output_size=action_size).to(device)
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Exploitability Critic (Standard CNN is fine for critic, or upgrade to DRQN later)
        # Keeping standard CNN for critic for now to save memory/complexity
        self.critic = ExploitabilityCritic(input_shape=(15, 10, 10), output_size=action_size).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01)
        self.critic_loss_fn = nn.CrossEntropyLoss()
        self.critic_weight = 0.1
        
        # Sequential Replay Buffer
        self.memory = SequentialReplayBuffer(buffer_size, trace_length=TRACE_LENGTH, device=device)
        
        # Episode Buffers (one per environment)
        self.current_episodes = [[] for _ in range(num_envs)]
        
        # Hidden States for Inference (1, batch, hidden_size)
        # We maintain a separate hidden state for each environment
        self.hidden_states = self._init_hidden_states(num_envs)
        
        # Metrics
        self.policy_losses = []
        self.q_values_history = []
        self.entropy_history = []
        self.smoothed_loss = None
        self.loss_smoothing_factor = 0.95
        self.step_count = 0
        self.epsilon_decay_interval = 500_000
        self.epsilon_min = max(epsilon_min, 0.01)
        
        # LR Scheduling
        self.initial_lr = lr
        self.lr_decay_factor = 0.5
        self.lr_decay_interval = 500_000
        self.min_lr = lr * 0.01
        self.loss_history_for_lr = deque(maxlen=200)
        self.lr_adjustment_interval = 50
        self.lr_adjustment_threshold = 1.5
        self.lr_reduction_factor = 0.5
        self.lr_increase_factor = 1.1
        self.lr_increase_threshold = 0.1
        self.high_loss_threshold = 100.0
        self.critical_loss_threshold = 200.0
        
        self.reward_history = deque(maxlen=1000)
        self.stagnation_episodes = 0
        self.best_avg_reward = float('-inf')
        
        self.update_target_network()
        
        # Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda')
        self.amp_enabled = True
        self.inf_gradient_count = 0
        self.total_optim_steps = 0
        
        self.performance_metrics = {
             'pbs_accuracy_trend': deque(maxlen=100),
             'dqn_loss_trend': deque(maxlen=100)
        }

    def _init_hidden_states(self, batch_size):
        """Initialize hidden states (h, c) with zeros"""
        h = torch.zeros(1, batch_size, 512, device=self.device)
        c = torch.zeros(1, batch_size, 512, device=self.device)
        return (h, c)
        
    def reset(self):
        """Reset the agent"""
        self.q_network = DRQN(input_shape=(15, 10, 10), output_size=self.action_size).to(self.device)
        self.target_network = DRQN(input_shape=(15, 10, 10), output_size=self.action_size).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.epsilon = 1.0
        self.memory.clear()
        self.current_episodes = [[] for _ in range(self.num_envs)]
        self.hidden_states = self._init_hidden_states(self.num_envs)
        
        self.policy_losses = []
        self.q_values_history = []
        self.entropy_history = []
        self.smoothed_loss = None
        self.loss_history_for_lr = deque(maxlen=200)
        self.step_count = 0
        self.reward_history = deque(maxlen=1000)
        self.stagnation_episodes = 0
        self.best_avg_reward = float('-inf')
        
        if self.pbs:
            if self.num_envs > 1:
                for pbs in self.pbs_instances:
                    pbs.reset()
            else:
                self.pbs.reset()
        
        for i in range(self.num_envs):
            if i in self.action_pbs_buffer:
                self.action_pbs_buffer[i].clear()
                
        self.update_target_network()
        
    def reset_episode(self, env_idx: int):
        """Reset hidden state and episode buffer for a specific environment"""
        # Reset hidden state for this env (set to zero)
        # hidden_states is tuple (h, c), each is (1, num_envs, 512)
        h, c = self.hidden_states
        h[:, env_idx, :].zero_()
        c[:, env_idx, :].zero_()
        self.hidden_states = (h, c)
        
        # Clear episode buffer
        self.current_episodes[env_idx] = []
        
        # Reset PBS
        self.reset_pbs(env_idx)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done, env_idx=0):
        """
        Store experience in the current episode buffer.
        If done, push the full episode to the replay buffer.
        """
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif state.device != self.device:
            state = state.to(self.device)
            
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        elif next_state.device != self.device:
            next_state = next_state.to(self.device)
            
        # Clip reward
        if abs(reward) <= 5.0:
            reward = max(-100.0, min(100.0, reward))
            
        experience = Experience(state, action, reward, next_state, done)
        
        # Add to current episode
        if env_idx < len(self.current_episodes):
            self.current_episodes[env_idx].append(experience)
            
            if done:
                # Push full episode to memory
                self.memory.add(list(self.current_episodes[env_idx]))
                # Clear buffer
                self.current_episodes[env_idx] = []
        
        self.step_count += 1

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
        Choose action using epsilon-greedy policy (Single Environment).
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
            
        if hasattr(self, 'search_agent') and self.search_agent and game_state:
            if self.is_endgame(game_state):
                best_move = self.search_agent.act(game_state, valid_moves)
                if best_move:
                    return best_move
        
        state_tensor = self.get_state_representation(state, pbs_instance=self.pbs)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.q_network.eval()
        with torch.no_grad():
            # Use hidden state for env 0 (assuming single env usage for act)
            h, c = self.hidden_states
            h_in = h[:, 0:1, :]
            c_in = c[:, 0:1, :]
            
            q_values, (h_out, c_out) = self.q_network(state_tensor, (h_in, c_in))
            
            # Update hidden state for env 0
            h[:, 0:1, :] = h_out
            c[:, 0:1, :] = c_out
            self.hidden_states = (h, c)
            
            base_q_values = q_values.squeeze(1)
            
        self.q_network.train()
        
        uncertainty_map = {}
        if self.pbs and game_state:
            uncertainty_map = self.pbs.get_uncertainty_map(game_state)
            
        q_values = self.calculate_uncertainty_aware_q_values(
            base_q_values, valid_moves, uncertainty_map
        )
        
        valid_q_values = []
        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            uncertainty = self.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
            valid_q_values.append(q_values[0, action_idx].item() + exploration_bonus)
            
        best_move_idx = np.argmax(valid_q_values)
        
        if self.pbs and game_state:
            best_move = valid_moves[best_move_idx]
            self.store_action_pbs_state(best_move, base_q_values, uncertainty_map, game_state)
            
        return valid_moves[best_move_idx]

    def get_state_value(self, game_state) -> float:
        """Get the Value V(s) of a state for Minimax heuristic."""
        state_tensor = self.get_state_representation(game_state)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        self.q_network.eval()
        with torch.no_grad():
            # Use zero hidden state for heuristic evaluation
            h = torch.zeros(1, 1, 512, device=self.device)
            c = torch.zeros(1, 1, 512, device=self.device)
            q_values, _ = self.q_network(state_tensor, (h, c))
            value = q_values.max().item()
            
        self.q_network.train()
        return value

    def act_batch(self, states, valid_moves_list, game_states=None, env_indices=None) -> List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Choose actions for a batch of states using DRQN.
        """
        batch_size = len(states)
        actions = [None] * batch_size
        
        if env_indices is None:
            env_indices = list(range(batch_size))
        
        # 1. Get batch state representation
        state_tensor = self.get_batch_state_representation(states, game_states)
        # state_tensor shape: (batch, C, H, W)
        
        # 2. Get uncertainty maps
        uncertainty_maps = []
        if self.pbs_instances and game_states:
            for i, gs in enumerate(game_states):
                if gs:
                    uncertainty_maps.append(self.pbs_instances[i].get_uncertainty_map(gs))
                else:
                    uncertainty_maps.append({})
        else:
            uncertainty_maps = [{}] * batch_size
            
        # 3. Network forward pass (with hidden states)
        self.q_network.eval()
        with torch.no_grad():
            # Select hidden states for these envs
            h, c = self.hidden_states
            indices_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
            
            h_in = h.index_select(1, indices_tensor)
            c_in = c.index_select(1, indices_tensor)
            
            # Pass hidden states. Input needs to be (batch, 1, C, H, W)
            q_values_batch, (h_out, c_out) = self.q_network(state_tensor, (h_in, c_in))
            
            # Update hidden states
            h.index_copy_(1, indices_tensor, h_out)
            c.index_copy_(1, indices_tensor, c_out)
            
            # q_values_batch is (batch, 1, output_size) -> squeeze to (batch, output_size)
            base_q_values_batch = q_values_batch.squeeze(1)
            
        self.q_network.train()
        
        # 4. Process each env
        for i in range(batch_size):
            valid_moves = valid_moves_list[i]
            if not valid_moves:
                continue
                
            if np.random.rand() <= self.epsilon:
                actions[i] = random.choice(valid_moves)
            else:
                # Exploitation
                q_values = self.calculate_uncertainty_aware_q_values(
                    base_q_values_batch[i].unsqueeze(0), 
                    valid_moves, 
                    uncertainty_maps[i]
                )
                
                valid_q_values = []
                for move in valid_moves:
                    action_idx = self._move_to_action_index(move)
                    uncertainty = self.get_move_uncertainty(move, uncertainty_maps[i])
                    exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
                    valid_q_values.append(q_values[0, action_idx].item() + exploration_bonus)
                
                best_move_idx = np.argmax(valid_q_values)
                actions[i] = valid_moves[best_move_idx]
                
            if self.pbs_instances and game_states and game_states[i]:
                # Use actual env index for storing PBS state
                actual_env_idx = env_indices[i]
                self.store_action_pbs_state(actions[i], base_q_values_batch[i].unsqueeze(0), uncertainty_maps[i], game_states[i], env_idx=actual_env_idx)
        
        return actions

    def replay(self, batch=None) -> Optional[float]:
        """
        Train the DRQN model on a batch of sequences.
        """
        # Check if we have enough episodes
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample sequences
        # batch_traces: List[List[Experience]]
        # mask: Tensor (batch, seq_len)
        batch_traces, mask = self.memory.sample(self.batch_size)
        
        if not batch_traces:
            return None
            
        # Prepare tensors
        # We need to stack experiences into (batch, seq_len, ...)
        # batch_traces is a list of lists.
        
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        
        for trace in batch_traces:
            # trace is a list of Experience objects
            s = torch.stack([e.state for e in trace]) # (seq_len, C, H, W)
            a = torch.tensor([e.action for e in trace], dtype=torch.long, device=self.device)
            r = torch.tensor([e.reward for e in trace], dtype=torch.float32, device=self.device)
            ns = torch.stack([e.next_state for e in trace])
            d = torch.tensor([e.done for e in trace], dtype=torch.bool, device=self.device)
            
            states_list.append(s)
            actions_list.append(a)
            rewards_list.append(r)
            next_states_list.append(ns)
            dones_list.append(d)
            
        # Stack into batch
        states = torch.stack(states_list) # (batch, seq_len, C, H, W)
        actions = torch.stack(actions_list) # (batch, seq_len)
        rewards = torch.stack(rewards_list) # (batch, seq_len)
        next_states = torch.stack(next_states_list) # (batch, seq_len, C, H, W)
        dones = torch.stack(dones_list) # (batch, seq_len)
        
        # --- 1. Train Critic (Optional/Simplified) ---
        # Critic currently takes (batch, C, H, W). We can flatten the sequence for critic training
        # or just skip critic for now to simplify. Let's keep it but flatten.
        flat_states = states.view(-1, *states.shape[2:]) # (batch*seq, C, H, W)
        flat_actions = actions.view(-1)
        
        # Only train on valid steps (mask=1)
        flat_mask = mask.view(-1).bool()
        valid_states = flat_states[flat_mask]
        valid_actions = flat_actions[flat_mask]
        
        if len(valid_states) > 0:
            critic_logits = self.critic(valid_states)
            critic_loss = self.critic_loss_fn(critic_logits, valid_actions)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
        else:
            critic_loss = torch.tensor(0.0)

        # --- 2. Train DRQN ---
        
        # Initialize hidden states for training (zero)
        # We start each sequence with zero hidden state
        train_hidden = self._init_hidden_states(self.batch_size)
        target_hidden = self._init_hidden_states(self.batch_size)
        
        # Forward pass (Online Network)
        # q_values: (batch, seq_len, output_size)
        q_values, _ = self.q_network(states, train_hidden)
        
        # Gather Q-values for taken actions
        # actions: (batch, seq_len) -> unsqueeze to (batch, seq_len, 1)
        current_q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2) # (batch, seq_len)
        
        # Target Network
        with torch.no_grad():
            # Double DQN: Select action with online, evaluate with target
            # We need next_states forward pass
            next_q_online, _ = self.q_network(next_states, train_hidden) # Use same hidden init? Yes.
            next_actions = next_q_online.max(2)[1] # (batch, seq_len)
            
            next_q_target, _ = self.target_network(next_states, target_hidden)
            next_q_values = next_q_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)
            
            # Calculate Target
            target_q = rewards + (self.gamma * next_q_values * ~dones)
            target_q = torch.clamp(target_q, -500.0, 500.0)
            
        # Calculate Loss
        # Apply mask to ignore padded steps
        loss_elementwise = F.smooth_l1_loss(current_q_values, target_q, reduction='none')
        loss = (loss_elementwise * mask).sum() / mask.sum()
        
        if torch.isnan(loss) or torch.isinf(loss):
             print(f"⚠️  Warning: NaN/Inf detected in loss. Skipping update.")
             self.optimizer.zero_grad()
             return None
             
        self.optimizer.zero_grad()
        
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        # Logging
        loss_value = loss.item()
        self.policy_losses.append(loss_value)
        
        # Update smoothed loss
        if self.smoothed_loss is None:
            self.smoothed_loss = loss_value
        else:
            self.smoothed_loss = self.loss_smoothing_factor * self.smoothed_loss + (1 - self.loss_smoothing_factor) * loss_value
            
        return loss_value

    # --- Helper Methods (Copied from original) ---
    def get_state_representation(self, game_state, pbs_instance=None) -> torch.Tensor:
        """Convert game state to neural network input - returns GPU tensor."""
        pbs = pbs_instance if pbs_instance else self.pbs
        if pbs and hasattr(game_state, 'board'):
            state = pbs.get_multi_channel_state(game_state)
            if state.device != self.device:
                state = state.to(self.device)
        else:
            state = torch.zeros((15, 10, 10), device=self.device, dtype=torch.float32)
            if hasattr(game_state, 'board'):
                board = game_state.board
                if isinstance(board, torch.Tensor):
                    board = board.to(self.device)
                else:
                    board = torch.tensor(board, device=self.device)
                if self.player_id == 1:
                    mask = (board > 0)
                    state[0][mask] = board[mask].float()
                else:
                    mask = (board < 0) & (board != -13) & (board != -20)
                    state[0][mask] = board[mask].abs().float()
                state[1] = (board == -13).float()
        if state.dtype != torch.float32:
            state = state.float()
        return state

    def get_batch_state_representation(self, states, game_states=None) -> torch.Tensor:
        tensor_list = []
        for i, state in enumerate(states):
            gs = game_states[i] if game_states else state
            pbs_inst = self.pbs_instances[i] if self.pbs_instances and i < len(self.pbs_instances) else self.pbs
            tensor = self.get_state_representation(gs, pbs_instance=pbs_inst)
            tensor_list.append(tensor)
        return torch.stack(tensor_list)

    def _move_to_action_index(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
        (r_from, c_from), (r_to, c_to) = move
        from_idx = r_from * 10 + c_from
        to_idx = r_to * 10 + c_to
        action_idx = from_idx * 10 + to_idx
        return action_idx % self.action_size
        
    def _action_index_to_move(self, action_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        from_idx = action_idx // 10
        to_idx = action_idx % 10
        r_from, c_from = from_idx // 10, from_idx % 10
        r_to, c_to = to_idx // 10, to_idx % 10
        return ((r_from, c_from), (r_to, c_to))

    def reset_pbs(self, env_idx: int = None):
        if self.num_envs > 1:
            if env_idx is not None:
                if 0 <= env_idx < len(self.pbs_instances):
                    self.pbs_instances[env_idx].reset()
                    if env_idx in self.action_pbs_buffer:
                        self.action_pbs_buffer[env_idx].clear()
            else:
                for pbs in self.pbs_instances:
                    pbs.reset()
                for i in range(self.num_envs):
                    if i in self.action_pbs_buffer:
                        self.action_pbs_buffer[i].clear()
        elif self.pbs:
            self.pbs.reset()
            if 0 in self.action_pbs_buffer:
                self.action_pbs_buffer[0].clear()

    def save_model(self, filepath: str):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load Q-Network with strict=False to ignore mismatches
            try:
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            except RuntimeError as e:
                print(f"⚠️  Shape mismatch loading Q-Network: {e}")
                print("   Attempting to load matching keys only...")
                model_dict = self.q_network.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['q_network_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.q_network.load_state_dict(model_dict)
                print(f"   ✅ Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")

            # Load Target Network
            try:
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            except RuntimeError as e:
                print(f"⚠️  Shape mismatch loading Target Network: {e}")
                print("   Attempting to load matching keys only...")
                model_dict = self.target_network.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['target_network_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.target_network.load_state_dict(model_dict)
                print(f"   ✅ Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")

            # Optimizer (skip if shapes changed significantly)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️  Could not load optimizer state (likely due to architecture change): {e}")
                print("   Resetting optimizer.")
                self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)

            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            print(f"✅ Model loaded from {filepath} (Ep: {self.epsilon:.4f}, Step: {self.step_count})")
            
        except Exception as e:
            print(f"❌ Failed to load model from {filepath}: {e}")

    # --- PBS Methods (Keep as is) ---
    def update_pbs_batch(self, actions, game_states, acting_player):
        if not self.pbs_instances: return
        all_aaren_inputs = []
        update_metadata = []
        for i, (action, gs) in enumerate(zip(actions, game_states)):
            if action is None or gs is None: continue
            if i >= len(self.pbs_instances): continue
            pbs = self.pbs_instances[i]
            q_val = None
            if i in self.action_pbs_buffer:
                for stored in self.action_pbs_buffer[i]:
                    if stored['action'] == action:
                        q_val = stored['q_value']
                        break
            result = pbs.prepare_aaren_update(action, gs, acting_player, q_value=q_val)
            if result:
                pos, sequence = result
                has_sequence = sequence is not None
                update_metadata.append((i, pos, action, gs, has_sequence))
                if has_sequence: all_aaren_inputs.append(sequence)
        
        aaren_probs_batch = []
        if all_aaren_inputs and self.shared_aaren is not None:
            max_len = max(len(seq) for seq in all_aaren_inputs)
            batch_size = len(all_aaren_inputs)
            input_size = len(all_aaren_inputs[0][0])
            x_batch = torch.zeros(batch_size, max_len, input_size, device=self.device)
            for j, seq in enumerate(all_aaren_inputs):
                seq_len = len(seq)
                x_batch[j, :seq_len, :] = torch.tensor(np.array(seq), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.shared_aaren(x_batch)
                probs = torch.softmax(logits, dim=1)
                aaren_probs_batch = [probs[j] for j in range(batch_size)]
        
        aaren_idx = 0
        for metadata in update_metadata:
            instance_idx, pos, action, gs, has_sequence = metadata
            pbs = self.pbs_instances[instance_idx]
            probs = None
            if has_sequence and aaren_idx < len(aaren_probs_batch):
                probs = aaren_probs_batch[aaren_idx]
                aaren_idx += 1
            pbs.apply_aaren_update(pos, probs, action, gs)

    def calculate_uncertainty_aware_q_values(self, base_q_values, valid_moves, uncertainty_map):
        q_values = base_q_values.clone()
        for move in valid_moves:
            action_idx = self._move_to_action_index(move)
            uncertainty_penalty = self.get_uncertainty_penalty(move, uncertainty_map)
            q_values[0, action_idx] -= uncertainty_penalty * self.uncertainty_penalty_scale
        return q_values

    def get_uncertainty_penalty(self, move, uncertainty_map):
        (r_from, c_from), (r_to, c_to) = move
        from_pos = (r_from, c_from)
        to_pos = (r_to, c_to)
        from_uncertainty = uncertainty_map.get(from_pos, 0.0)
        to_uncertainty = uncertainty_map.get(to_pos, 0.0)
        return float((from_uncertainty + to_uncertainty) / 2.0)

    def get_move_uncertainty(self, move, uncertainty_map):
        return self.get_uncertainty_penalty(move, uncertainty_map)

    def store_action_pbs_state(self, action, q_values, uncertainty_map, game_state, env_idx=0):
        action_idx = self._move_to_action_index(action)
        action_q_value = q_values[0, action_idx].item()
        action_uncertainty = self.get_move_uncertainty(action, uncertainty_map)
        if env_idx not in self.action_pbs_buffer:
            self.action_pbs_buffer[env_idx] = deque(maxlen=50)
        self.action_pbs_buffer[env_idx].append({
            'action': action,
            'q_value': action_q_value,
            'uncertainty': action_uncertainty,
            'game_state': game_state
        })

    def train_pbs_evaluator(self, epochs=1):
        if self.pbs is None: return None
        total_loss = 0.0
        loss_count = 0
        if self.shared_evaluator is not None:
            eval_loss = self.shared_evaluator.train(epochs=epochs)
            if eval_loss is not None:
                total_loss += eval_loss
                loss_count += 1
        if self.shared_aaren is not None and self.num_envs > 1 and self.pbs_instances:
            all_action_sequences = []
            all_true_piece_types = []
            all_evaluator_weights = []
            all_positions = []
            for pbs in self.pbs_instances:
                if hasattr(pbs, 'get_aaren_training_data'):
                    sequences, types, weights, positions = pbs.get_aaren_training_data()
                    all_action_sequences.extend(sequences)
                    all_true_piece_types.extend(types)
                    all_evaluator_weights.extend(weights)
                    all_positions.extend(positions)
            if all_action_sequences:
                self.pbs.train_aaren(all_action_sequences, all_true_piece_types, epochs, all_evaluator_weights, all_positions)
        elif self.num_envs > 1 and self.pbs_instances:
            for pbs in self.pbs_instances:
                loss = pbs.train_evaluator(epochs=epochs)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
        else:
            return self.pbs.train_evaluator(epochs=epochs)
        if loss_count > 0: return total_loss / loss_count
        return None
    
    # --- Metrics ---
    def get_average_policy_loss(self, window=100):
        if not self.policy_losses: return 0.0
        return sum(self.policy_losses[-window:]) / len(self.policy_losses[-window:])
    
    def get_average_q_value(self, window=100):
        if not self.q_values_history: return 0.0
        return sum(self.q_values_history[-window:]) / len(self.q_values_history[-window:])
        
    def get_average_entropy(self, window=100):
        if not self.entropy_history: return 0.0
        return sum(self.entropy_history[-window:]) / len(self.entropy_history[-window:])
