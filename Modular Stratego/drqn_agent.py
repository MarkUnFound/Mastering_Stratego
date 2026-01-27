"""
Rainbow DQN Agent for Stratego Game
Features:
- Feed-Forward Architecture (relies on AAREN PBS for memory)
- Noisy Nets for Exploration (No epsilon-greedy)
- C51 Distributional RL (Categorical DQN)
- Dueling Architecture
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
from board import LAKE_SQUARE

# Import from new modular structure
from networks import NoisyLinear, RainbowDQN
from history_aggregator import HistoryAggregator
from aaren import PieceActionAaren

from prioritized_memory import StandardReplayBuffer, PrioritizedReplayBuffer, NStepBuffer, Experience

# Heuristic Action Filter for Top-100 move selection
from heuristic_filter import HeuristicMoveFilter

# C51 Hyperparameters - CRITICAL FOR DISTRIBUTIONAL RL
# The support range must match the expected return scale!
# With heavily boosted rewards (win_reward_flag=25.0):
#   - Max expected return: ~+30 (win + captures + bonuses + N-step returns)
#   - Min expected return: ~-30 (loss + losses + penalties + N-step returns)
# Using [-30.0, +30.0] covers the full boosted range with margin
V_MIN = -30.0   # Expanded to match heavily boosted reward scale
V_MAX = 30.0    # Expanded to match heavily boosted reward scale
NUM_ATOMS = 51




# NoisyLinear and RainbowDQN are now imported from networks module

import sys





class RainbowAgent:
    """Rainbow DQN Agent for Stratego"""
    
    def __init__(self, player_id: int, device, 
                 state_size: int = 200, action_size: int = 400,
                 lr: float = 0.0001, gamma: float = 0.99, 
                 buffer_size: int = 10000, batch_size: int = 32,
                 use_pbs: bool = True, num_envs: int = 1):
        
        self.player_id = player_id
        self.device = device
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.name = f"Rainbow Agent {player_id}"
        self.num_envs = num_envs
        self.use_pbs = use_pbs
        
        # Epsilon-greedy exploration (hybrid with Noisy Nets)
        try:
            from training_config import EXPLORATION_EPSILON_START, EXPLORATION_EPSILON_END, EXPLORATION_EPSILON_DECAY
            self.epsilon_start = EXPLORATION_EPSILON_START
            self.epsilon_end = EXPLORATION_EPSILON_END
            self.epsilon_decay = EXPLORATION_EPSILON_DECAY
        except ImportError:
            self.epsilon_start = 0.3
            self.epsilon_end = 0.01
            self.epsilon_decay = 20000
        self.current_episode = 0  # Updated externally for epsilon decay
        
        # C51 Support (Atoms)
        self.num_atoms = NUM_ATOMS
        self.v_min = V_MIN
        self.v_max = V_MAX
        # Create support vector: [-100, -96, ..., 96, 100]
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Probabilistic Belief State
        self.pbs = None
        self.pbs_instances = []
        self.action_pbs_buffer = {} 
        
        # Import AAREN optimization settings
        try:
            from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS, HISTORY_EMBEDDING_SIZE
        except ImportError:
            AAREN_HIDDEN_SIZE = 64
            AAREN_NUM_LAYERS = 2
            HISTORY_EMBEDDING_SIZE = 64
        
        # History Aggregator (replaces PBS)
        # Uses AAREN embeddings instead of belief distributions
        self.history = None
        self.history_instances = []
        self.history_embedding_size = HISTORY_EMBEDDING_SIZE
        
        if self.use_pbs:  # use_pbs now means use_history
            if num_envs > 1:
                # Create shared AAREN for parallel environments
                self.shared_aaren = PieceActionAaren(
                    input_size=24, hidden_size=AAREN_HIDDEN_SIZE, 
                    num_layers=AAREN_NUM_LAYERS, output_size=12, device=device
                ).to(device)
                
                # Note: AAREN uses main optimizer (combined with DQN) for end-to-end training
                
                for _ in range(num_envs):
                    history_instance = HistoryAggregator(
                        player_id, device, 
                        hidden_size=AAREN_HIDDEN_SIZE,
                        num_layers=AAREN_NUM_LAYERS,
                        shared_aaren_model=self.shared_aaren
                    )
                    history_instance.optimizer = None  # Uses main optimizer
                    self.history_instances.append(history_instance)
                self.history = self.history_instances[0]
            else:
                self.history = HistoryAggregator(player_id, device, hidden_size=AAREN_HIDDEN_SIZE)
                self.history_instances = [self.history]
        
        # Legacy PBS references for backward compatibility (point to history)
        self.pbs = self.history
        self.pbs_instances = self.history_instances
        
        # Uncertainty parameters
        self.uncertainty_exploration_multiplier = 0.05
        self.uncertainty_penalty_scale = 0.5
        
        # Cached gradient norms (updated during replay for plotting)
        self._cached_aaren_grad_norm = 0.0
        self._cached_dqn_grad_norm = 0.0
        
        # Soft Update Param
        self.tau = 0.001
        
        # Rainbow Networks
        # 15 (Board) + HISTORY_EMBEDDING_SIZE (AAREN embeddings) = 79 Channels (with 64)
        self.input_channels = 15 + self.history_embedding_size
        self.q_network = RainbowDQN(input_shape=(self.input_channels, 10, 10), output_size=action_size, num_atoms=self.num_atoms).to(device)
        self.target_network = RainbowDQN(input_shape=(self.input_channels, 10, 10), output_size=action_size, num_atoms=self.num_atoms).to(device)
        self.update_target_network()
        
        # PyTorch 2.0+ Compilation (optional speedup)
        try:
            from training_config import USE_TORCH_COMPILE, TORCH_COMPILE_MODE
            if USE_TORCH_COMPILE:
                self.q_network = torch.compile(self.q_network, mode=TORCH_COMPILE_MODE)
                self.target_network = torch.compile(self.target_network, mode=TORCH_COMPILE_MODE)
                print(f"[OK] Rainbow DQN compiled with mode='{TORCH_COMPILE_MODE}'")
        except (ImportError, Exception) as e:
            pass  # Silently skip if unavailable
        
        # Reference Network for KL-Regularization (anti-cycling)
        # This network updates MUCH slower than target to anchor policy
        self.reference_network = RainbowDQN(
            input_shape=(self.input_channels, 10, 10), 
            output_size=action_size, 
            num_atoms=self.num_atoms
        ).to(device)
        self.reference_network.load_state_dict(self.q_network.state_dict())
        self.reference_network.eval()  # Never train directly
        self.reference_update_counter = 0
        
        # KL and Entropy Regularization Settings
        try:
            from training_config import (
                KL_REG_ENABLED, KL_REG_WEIGHT, REF_POLICY_UPDATE_INTERVAL,
                ENTROPY_REG_ENABLED, ENTROPY_COEFF_START, ENTROPY_COEFF_END, ENTROPY_ANNEAL_EPISODES
            )
            self.kl_reg_enabled = KL_REG_ENABLED
            self.kl_reg_weight = KL_REG_WEIGHT
            self.ref_update_interval = REF_POLICY_UPDATE_INTERVAL
            self.entropy_reg_enabled = ENTROPY_REG_ENABLED
            self.entropy_coeff_start = ENTROPY_COEFF_START
            self.entropy_coeff_end = ENTROPY_COEFF_END
            self.entropy_anneal_episodes = ENTROPY_ANNEAL_EPISODES
        except ImportError:
            self.kl_reg_enabled = False
            self.kl_reg_weight = 0.01
            self.ref_update_interval = 50000
            self.entropy_reg_enabled = False
            self.entropy_coeff_start = 0.1
            self.entropy_coeff_end = 0.01
            self.entropy_anneal_episodes = 50000
        
        # Ataraxos Techniques Configuration
        try:
            from training_config import (
                ADVANTAGE_FILTERING_ENABLED, ADVANTAGE_OVERSAMPLE_FACTOR, ADVANTAGE_MIN_BATCH,
                DYNAMIC_DAMPING_ENABLED, TOTAL_TRAINING_EPISODES,
                MAGNET_COEFF_START, MAGNET_COEFF_END, MAGNET_POWER,
                TARGET_KL_COEFF_START, TARGET_KL_COEFF_END, TARGET_KL_POWER
            )
            self.advantage_filtering = ADVANTAGE_FILTERING_ENABLED
            self.oversample_factor = ADVANTAGE_OVERSAMPLE_FACTOR
            self.min_batch_after_filter = ADVANTAGE_MIN_BATCH
            self.dynamic_damping = DYNAMIC_DAMPING_ENABLED
            self.total_episodes = TOTAL_TRAINING_EPISODES
            self.magnet_start = MAGNET_COEFF_START
            self.magnet_end = MAGNET_COEFF_END
            self.magnet_power = MAGNET_POWER
            self.target_kl_start = TARGET_KL_COEFF_START
            self.target_kl_end = TARGET_KL_COEFF_END
            self.target_kl_power = TARGET_KL_POWER
        except ImportError:
            self.advantage_filtering = False
            self.oversample_factor = 4
            self.min_batch_after_filter = 64
            self.dynamic_damping = False
            self.total_episodes = 100000
            self.magnet_start = 0.1
            self.magnet_end = 0.001
            self.magnet_power = 2.0
            self.target_kl_start = 0.001
            self.target_kl_end = 0.1
            self.target_kl_power = 2.0
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Combined Optimizer: DQN + AAREN (end-to-end training)
        # AAREN learns jointly with Rainbow DQN via shared gradient flow
        all_parameters = list(self.q_network.parameters())
        if self.use_pbs and hasattr(self, 'shared_aaren') and self.shared_aaren:
            all_parameters.extend(self.shared_aaren.parameters())
            print(f"[OK] AAREN params added to optimizer (end-to-end training)")
        elif self.use_pbs and self.history and self.history.owns_aaren:
            all_parameters.extend(self.history.aaren.parameters())
            print(f"[OK] AAREN params added to optimizer (end-to-end training)")
        
        self.optimizer = optim.AdamW(all_parameters, lr=lr, weight_decay=0.01)
        
        # Import PER and N-Step settings
        try:
            from training_config import (
                PER_ENABLED, PER_ALPHA, PER_BETA_START, PER_BETA_END, PER_BETA_ANNEAL_EPISODES,
                N_STEPS, GAMMA_N, LR_SCHEDULER_ENABLED, LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA
            )
        except ImportError:
            PER_ENABLED = False
            N_STEPS = 1
            GAMMA_N = gamma
            LR_SCHEDULER_ENABLED = False
        
        # Replay Buffer (Prioritized or Standard)
        self.per_enabled = PER_ENABLED
        if PER_ENABLED:
            self.memory = PrioritizedReplayBuffer(
                buffer_size, device=device, alpha=PER_ALPHA,
                beta_start=PER_BETA_START, beta_end=PER_BETA_END,
                beta_anneal_episodes=PER_BETA_ANNEAL_EPISODES
            )
            print(f"[OK] [P{self.player_id}] Prioritized Experience Replay enabled (alpha={PER_ALPHA})")
        else:
            self.memory = StandardReplayBuffer(buffer_size, device=device)
            print(f"[OK] [P{self.player_id}] Standard Replay Buffer enabled")
        
        # Data Augmentation Settings
        try:
            from training_config import ENABLE_DATA_AUGMENTATION, AUGMENTATION_TYPES
            self.aug_enabled = ENABLE_DATA_AUGMENTATION
            self.aug_types = AUGMENTATION_TYPES
        except ImportError:
            self.aug_enabled = False
            self.aug_types = []
        
        # N-Step Buffer for multi-step returns
        self.n_steps = N_STEPS
        self.gamma_n = GAMMA_N
        if N_STEPS > 1:
            self.n_step_buffers = [NStepBuffer(n_steps=N_STEPS, gamma=gamma) for _ in range(max(num_envs, 1))]
            print(f"[OK] [P{self.player_id}] N-Step returns enabled (n={N_STEPS})")
        else:
            self.n_step_buffers = None
        
        # Learning Rate Scheduler
        self.scheduler = None
        if LR_SCHEDULER_ENABLED:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA
            )
            print(f"[OK] [P{self.player_id}] LR Scheduler enabled (step={LR_SCHEDULER_STEP_SIZE}, gamma={LR_SCHEDULER_GAMMA})")
        
        # Metrics
        self.step_count = 0
        self.reward_history = deque(maxlen=1000)
        
        # Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda')
        self.amp_enabled = True
        
        # Heuristic Action Filter for Top-100 move selection
        self.move_filter = HeuristicMoveFilter()
        self.use_heuristic_filter = True  # Can be disabled for debugging
        self.max_filtered_moves = 100
        
        # Cache for state tensors (to avoid redundant computation in remember_batch)
        self._cached_state_tensor = None  # Cached from act_batch


    def reset(self):
        """Reset the agent"""
        self.q_network = RainbowDQN(input_shape=(self.input_channels, 10, 10), output_size=self.action_size, num_atoms=self.num_atoms).to(self.device)
        self.target_network = RainbowDQN(input_shape=(self.input_channels, 10, 10), output_size=self.action_size, num_atoms=self.num_atoms).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.memory.clear()
        self.step_count = 0
        self.update_target_network()
        
        self.reset_pbs()
            
    def reset_pbs(self):
        """Reset the PBS module state."""
        if self.pbs:
            if self.num_envs > 1:
                for pbs in self.pbs_instances:
                    pbs.reset()
            else:
                self.pbs.reset()

    def enable_pbs(self, num_envs: int = None):
        """
        Enable PBS for an agent that was created with use_pbs=False.
        This is used during curriculum phase transitions to dynamically enable PBS.
        
        Args:
            num_envs: Number of parallel environments (uses self.num_envs if not provided)
        """
        if self.use_pbs:
            return  # Already enabled
            
        if num_envs is None:
            num_envs = self.num_envs
            
        self.use_pbs = True
        self.num_envs = num_envs
        
        # Import AAREN optimization settings
        try:
            from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS, AAREN_USE_TORCHSCRIPT
        except ImportError:
            AAREN_HIDDEN_SIZE = 32
            AAREN_NUM_LAYERS = 2
            AAREN_USE_TORCHSCRIPT = True
        
        if num_envs > 1:
            # Create shared models for parallel environments with optimized settings
            self.shared_aaren = PieceActionAaren(
                input_size=24, hidden_size=AAREN_HIDDEN_SIZE, 
                num_layers=AAREN_NUM_LAYERS, output_size=12, device=self.device
            ).to(self.device)
            
            # TorchScript compilation for faster inference
            if AAREN_USE_TORCHSCRIPT:
                try:
                    self.shared_aaren = torch.jit.script(self.shared_aaren)
                    print(f"[OK] AAREN TorchScript compilation successful (enable_pbs)")
                except Exception as e:
                    print(f"[WARN] AAREN TorchScript failed, using eager mode: {e}")
            
            self.shared_aaren_optimizer = optim.AdamW(self.shared_aaren.parameters(), lr=0.001, weight_decay=0.01)
            
            self.shared_evaluator = None
            if PBS_EVALUATOR_AVAILABLE:
                self.shared_evaluator = PBSEvaluator(device=self.device)
            
            self.pbs_instances = []
            for _ in range(num_envs):
                pbs_instance = ProbabilisticBeliefState(
                    self.player_id, self.device, 
                    shared_aaren_model=self.shared_aaren,
                    shared_evaluator=self.shared_evaluator
                )
                pbs_instance.aaren_optimizer = self.shared_aaren_optimizer
                self.pbs_instances.append(pbs_instance)
            self.pbs = self.pbs_instances[0]
        else:
            self.pbs = ProbabilisticBeliefState(self.player_id, self.device)
            self.pbs_instances = [self.pbs]
        
        print(f"[OK] PBS enabled for {self.name} ({num_envs} environments)")

    def update_target_network(self):
        """Soft update target network."""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
    def get_state_representation(self, board, pbs_instance=None, full_observability=False):
        """
        Convert raw board to feature tensor.
        Now includes AAREN history embeddings instead of PBS beliefs.
        Total channels: 15 (Board) + HISTORY_EMBEDDING_SIZE (AAREN) = 79 (default)
        
        Args:
            board: Game board tensor or GameState object
            pbs_instance: History aggregator instance (legacy name for compatibility)
            full_observability: If True, use ground truth enemy types instead of embeddings
        """
        # Handle GameState object - extract the board tensor
        from game_state import GameState
        if isinstance(board, GameState):
            board = board.board
            
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).to(self.device)
            
        # 1. Board Features (15 channels)
        features = torch.zeros((15, 10, 10), device=self.device)
        
        if self.player_id == 1:
            # Player 1: Own pieces are positive (1-12)
            for i in range(1, 13):
                features[i-1] = (board == i).float()
            # Enemy pieces: Negative values > -13
            features[12] = ((board < 0) & (board > LAKE_SQUARE)).float()
        else:
            # Player 2: Own pieces are negative (-1 to -12)
            for i in range(1, 13):
                features[i-1] = (board == -i).float()
            # Enemy pieces: Positive values
            features[12] = (board > 0).float()
            
        # Obstacles (Channel 13)
        features[13] = (board == LAKE_SQUARE).float()
        
        # Channel 14: Empty squares (0)
        features[14] = (board == 0).float()
        
        state_tensor = features
        
        # --- AAREN EMBEDDING CHANNELS ---
        if full_observability:
            # FULL OBSERVABILITY: Create representation from ground truth
            # Expand piece type info to fill embedding channels
            embedding = torch.zeros((self.history_embedding_size, 10, 10), device=self.device)
            
            if self.player_id == 1:
                enemy_mask = (board < 0) & (board > LAKE_SQUARE)
                enemy_positions = torch.nonzero(enemy_mask)
                for pos in enemy_positions:
                    r, c = pos[0].item(), pos[1].item()
                    piece_type_idx = abs(int(board[r, c].item())) - 1
                    if 0 <= piece_type_idx < 12:
                        # Create a distinctive pattern for this piece type
                        # Use first 12 channels for one-hot, rest as zeros
                        embedding[piece_type_idx, r, c] = 1.0
            else:
                enemy_mask = board > 0
                enemy_positions = torch.nonzero(enemy_mask)
                for pos in enemy_positions:
                    r, c = pos[0].item(), pos[1].item()
                    piece_type_idx = int(board[r, c].item()) - 1
                    if 0 <= piece_type_idx < 12:
                        embedding[piece_type_idx, r, c] = 1.0
            
            full_state = torch.cat([state_tensor, embedding], dim=0)
            return full_state
            
        elif pbs_instance is not None:
            # PARTIAL OBSERVABILITY: Use AAREN embeddings from HistoryAggregator
            embedding = pbs_instance.get_embedding_tensor()
            
            # Ensure correct device
            if embedding.device != self.device:
                embedding = embedding.to(self.device)
                
            # Concatenate: (15 + history_embedding_size, 10, 10)
            full_state = torch.cat([state_tensor, embedding], dim=0)
            return full_state
        
        # Fallback: pad with zeros
        else:
            padding = torch.zeros((self.history_embedding_size, 10, 10), device=self.device)
            full_state = torch.cat([state_tensor, padding], dim=0)
            return full_state
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer with automated augmentation"""
        # Process states
        state_processed = self.get_state_representation(state, pbs_instance=self.pbs)
        next_state_processed = self.get_state_representation(next_state, pbs_instance=self.pbs)
        
        self._add_experience(state_processed, action, reward, next_state_processed, done)

    def _add_experience(self, state, action, reward, next_state, done, is_battle=False):
        """
        Internal helper to add experience to memory with potential augmentation.
        Always expects 'state' and 'next_state' to be processed tensors.
        'action' can be a move tuple or index.
        
        Args:
            is_battle: If True, indicates this transition involved a capture event
        """
        # 1. Convert action to index if it's a tuple
        action_idx = action
        move_tuple = None
        if isinstance(action, (tuple, list)):
            move_tuple = action
            action_idx = self._move_to_action_index(action)
        else:
            # If we already have an index, we need to decode it for augmentation
            move_tuple = self._action_index_to_move(action_idx)
            
        # 2. Add original experience
        # Clip reward to C51 support range to avoid instability
        reward_clipped = max(self.v_min, min(self.v_max, reward))
        
        # SELF-IMITATION LEARNING: Detect winning experiences for priority boost
        # Win rewards are 10-15, so use 10.0 as threshold
        is_winning_experience = (reward_clipped >= 10.0 and done)
        
        # Pass priority boost flags to PER (ignored by StandardBuffer)
        if hasattr(self.memory, 'add') and 'is_winning_experience' in self.memory.add.__code__.co_varnames:
            self.memory.add(state, action_idx, reward_clipped, next_state, done, 
                          is_winning_experience=is_winning_experience, is_battle=is_battle)
        else:
            self.memory.add(state, action_idx, reward_clipped, next_state, done)
        self.step_count += 1
        
        # 3. Add Augmented Experiences
        if self.aug_enabled and move_tuple:
            (r1, c1), (r2, c2) = move_tuple
            
            # --- Horizontal Flip (Mirror) ---
            if "flip" in self.aug_types:
                # Flip columns of the tensor: (C, H, W) -> [:, :, ::-1]
                state_flip = torch.flip(state, [2])
                next_state_flip = torch.flip(next_state, [2])
                
                # Flip action coordinates: c -> (9-c)
                move_flip = ((r1, 9-c1), (r2, 9-c2))
                action_idx_flip = self._move_to_action_index(move_flip)
                
                self.memory.add(state_flip, action_idx_flip, reward_clipped, next_state_flip, done)
                self.step_count += 1
                
            # --- Rotation 180 ---
            if "rotate" in self.aug_types:
                # Flip rows and columns: (C, H, W) -> [:, ::-1, ::-1]
                state_rot = torch.flip(state, [1, 2])
                next_state_rot = torch.flip(next_state, [1, 2])
                
                # Flip both coordinates: r -> (9-r), c -> (9-c)
                move_rot = ((9-r1, 9-c1), (9-r2, 9-c2))
                action_idx_rot = self._move_to_action_index(move_rot)
                
                self.memory.add(state_rot, action_idx_rot, reward_clipped, next_state_rot, done)
                self.step_count += 1

    def remember_batch(self, states, actions, rewards, next_states, dones, active_mask, game_states=None, next_game_states=None, infos=None):
        """
        Store multiple experiences efficiently with batched state processing.
        Uses N-step returns if enabled.
        
        Args:
            states: List of game states (boards)
            actions: List of actions (tuples)
            rewards: List of rewards
            next_states: List of next game states (boards)
            dones: List of done flags
            active_mask: Boolean array indicating which envs are active
            game_states: List of GameState objects (for PBS lookup)
            next_game_states: List of next GameState objects
            infos: List of info dicts from environment (contains revealed_in_step for battle detection)
        """
        # Get batch state representation once (much more efficient)
        # Get batch state representation once (much more efficient)
        # NOTE: Cannot use _cached_state_tensor because batch_states includes PENDING transitions
        # from previous turns, which do not match the current states used in act_batch.
        state_tensors = self.get_batch_state_representation(states, game_states)
            
        next_state_tensors = self.get_batch_state_representation(next_states, next_game_states)
        
        # Add to memory for active environments only
        for i in range(len(states)):
            if not active_mask[i]:
                continue
                
            action = actions[i]
            if action is None:
                continue
                
            # Reward is in a list here
            reward = rewards[i]
            
            # BATTLE DETECTION: Check if this transition involved a capture
            # Battles are detected via revealed_in_step in info dict
            is_battle = False
            if infos and i < len(infos) and infos[i]:
                revealed = infos[i].get('revealed_in_step', [])
                is_battle = len(revealed) > 0  # Any piece revealed = battle occurred
            
            # N-Step returns processing
            if self.n_step_buffers is not None and i < len(self.n_step_buffers):
                # Buffers expect the action as it was taken (tuple or index)
                # If N-step gives us a result, it returns the FIRST action in the sequence
                n_step_result = self.n_step_buffers[i].add(
                    state_tensors[i], action, reward, next_state_tensors[i], dones[i]
                )
                
                if n_step_result is not None:
                    # Got n-step experience - add to replay via central helper
                    # Note: For n-step, we boost priority if ANY step in the sequence had a battle
                    n_state, n_action, n_reward, n_next_state, n_done = n_step_result
                    self._add_experience(n_state, n_action, n_reward, n_next_state, n_done, is_battle=is_battle)
                
                # Flush remaining if episode done
                if dones[i]:
                    remaining = self.n_step_buffers[i].flush()
                    for result in remaining:
                        n_state, n_action, n_reward, n_next_state, n_done = result
                        self._add_experience(n_state, n_action, n_reward, n_next_state, n_done, is_battle=is_battle)
                    self.n_step_buffers[i].reset()
            else:
                # Standard 1-step: add directly using central helper
                self._add_experience(state_tensors[i], action, reward, next_state_tensors[i], dones[i], is_battle=is_battle)

    def act(self, state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], game_state=None):
        """
        Choose action using Noisy Nets (Exploration is implicit).
        Uses HeuristicMoveFilter to reduce action space for faster decisions.
        """
        if not valid_moves:
            return None
            
        # Pre-filter moves using heuristic filter for faster decisions
        # This reduces action space from potentially 200+ to top 100 moves
        if len(valid_moves) > 100:
            filtered_scored = self.move_filter.get_filtered_actions(
                state.actual_board if hasattr(state, 'actual_board') else state.board,
                valid_moves, 
                self.player_id,
                max_moves=100
            )
            valid_moves = [move for move, score in filtered_scored]
        
        state_tensor = self.get_state_representation(state, pbs_instance=self.pbs)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        self.q_network.eval()
        with torch.no_grad():
            # Mixed-precision inference for ~15-30% speedup
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                # Get Log Probabilities: (batch, action_size, num_atoms)
                log_probs = self.q_network(state_tensor)
                probs = log_probs.exp()
                
                # Calculate Expected Value: Sum(p_i * z_i)
                # probs: (1, actions, atoms)
                # support: (atoms)
                expected_q_values = (probs * self.support).sum(dim=2) # (1, actions)
                
            base_q_values = expected_q_values.squeeze(0) # (actions)
            
        self.q_network.train()
        
        # Uncertainty handling (same as before)
        uncertainty_map = {}
        if self.pbs and game_state:
            uncertainty_map = self.pbs.get_uncertainty_map(game_state)
            
        # Filter valid moves - now includes Scout multi-step moves
        valid_q_values = []
        valid_moves_filtered = []
        for move in valid_moves:
            (r1, c1), (r2, c2) = move
            dist = abs(r2 - r1) + abs(c2 - c1)
            
            # For Scout moves (dist > 1), map to 1-step direction action
            # This allows Scout to use its special ability
            if dist > 1:
                # Calculate direction unit vector
                if r2 != r1:
                    dr = 1 if r2 > r1 else -1
                    dc = 0
                else:
                    dr = 0
                    dc = 1 if c2 > c1 else -1
                # Create a "virtual" 1-step move for action indexing
                virtual_move = ((r1, c1), (r1 + dr, c1 + dc))
                action_idx = self._move_to_action_index(virtual_move)
            else:
                action_idx = self._move_to_action_index(move)
            
            if action_idx is None:
                continue
                
            q_val = base_q_values[action_idx].item()
            
            # Add uncertainty bonus
            uncertainty = self.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
            
            valid_q_values.append(q_val + exploration_bonus)
            valid_moves_filtered.append(move)
                     
        if not valid_moves_filtered:
            # Fallback
            return valid_moves[0] if valid_moves else None

        best_move_idx = np.argmax(valid_q_values)
        best_move = valid_moves_filtered[best_move_idx]
        
        if self.pbs and game_state:
            self.store_action_pbs_state(best_move, base_q_values.unsqueeze(0), uncertainty_map, game_state)
            
        return best_move

    def get_batch_state_representation(self, states, game_states=None, full_observability=False):
        """
        Convert batch of states to tensor, integrating PBS beliefs if available.
        
        Args:
            states: List of game board states
            game_states: List of GameState objects (for PBS lookup)
            full_observability: If True, use ground truth enemy types (Phase 1)
        """
        # Convert states to tensors
        state_tensors = []
        for i, state in enumerate(states):
            # Get PBS instance for this environment index if available
            pbs_instance = None
            if not full_observability:  # Only use PBS in partial observability mode
                if self.pbs_instances and game_states and i < len(game_states) and game_states[i]:
                    pbs_instance = self.pbs_instances[i]
                elif self.pbs and game_states and i < len(game_states) and game_states[i]:
                    # Fallback to single PBS instance if pbs_instances list not used
                    pbs_instance = self.pbs
                
            # Use get_state_representation for each state
            # This handles the 27-channel concatenation
            state_tensor = self.get_state_representation(state, pbs_instance=pbs_instance, full_observability=full_observability)
            state_tensors.append(state_tensor)
            
        # Stack into batch
        return torch.stack(state_tensors)

    def act_batch(self, states, valid_moves_list, game_states=None, env_indices=None, full_observability=False) -> List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Batch action selection with epsilon-greedy exploration
        
        Args:
            states: List of game board states
            valid_moves_list: List of valid moves for each environment
            game_states: List of GameState objects
            env_indices: Environment indices for PBS lookup
            full_observability: If True, use ground truth enemy types (Phase 1)
        """
        import random
        batch_size = len(states)
        actions = [None] * batch_size
        
        if env_indices is None:
            env_indices = list(range(batch_size))
        
        # EPSILON-GREEDY EXPLORATION: Random action with probability epsilon
        # Epsilon decays over training from epsilon_start to epsilon_end
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  max(0, 1 - self.current_episode / self.epsilon_decay)
        
        # Check which environments will take random actions
        random_action_mask = [random.random() < epsilon for _ in range(batch_size)]
        
        # IMITATION LEARNING: Use heuristic expert for portion of actions during early training
        try:
            from training_config import IMITATION_ENABLED, IMITATION_RATIO, IMITATION_EPISODES
        except ImportError:
            IMITATION_ENABLED = False
            IMITATION_RATIO = 0.0
            IMITATION_EPISODES = 0
        
        imitation_active = IMITATION_ENABLED and self.current_episode < IMITATION_EPISODES
        imitation_mask = [random.random() < IMITATION_RATIO if imitation_active else False 
                         for _ in range(batch_size)]
        
        # 1. Get batch state representation and cache it for remember_batch
        state_tensor = self.get_batch_state_representation(states, game_states, full_observability=full_observability)
        self._cached_state_tensor = state_tensor  # Cache for reuse in remember_batch
        
        # 2. Uncertainty maps removed - HistoryAggregator uses AAREN embeddings directly
        # The agent learns to interpret uncertainty implicitly from AAREN patterns
        uncertainty_maps = [{}] * batch_size
            
        # 3. Network forward pass with mixed-precision inference
        self.q_network.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                try:
                    log_probs = self.q_network(state_tensor)
                except RuntimeError as e:
                    print(f"DEBUG: act_batch CRASH. state_tensor shape: {state_tensor.shape}")
                    raise e
                probs = log_probs.exp()
                expected_q_values = (probs * self.support).sum(dim=2) # (batch, actions)
            
        self.q_network.train()
        
        # 4. Process each env with HEURISTIC ACTION MASKING
        for i in range(batch_size):
            valid_moves = valid_moves_list[i]
            if not valid_moves:
                continue
            
            # EPSILON-GREEDY: Random action with probability epsilon
            if random_action_mask[i]:
                actions[i] = random.choice(valid_moves)
                continue
            
            # IMITATION LEARNING: Use heuristic expert for early training
            if imitation_mask[i]:
                # Get board for heuristic scoring
                board = states[i]
                if hasattr(board, 'board'):
                    board = board.board
                # Get top heuristic-scored move as expert demonstration
                scored_moves = self.move_filter.get_filtered_actions(
                    board, valid_moves, self.player_id, max_moves=1
                )
                if scored_moves:
                    actions[i] = scored_moves[0][0]  # Best move from expert
                else:
                    actions[i] = valid_moves[0]
                continue
            
            # Get board for heuristic scoring
            board = states[i]
            if hasattr(board, 'board'):
                board = board.board
            
            if self.use_heuristic_filter:
                # ============================================================
                # HEURISTIC ACTION MASKING
                # 1. Score all moves using strategic heuristics
                # 2. Create mask that only allows Top-100 moves
                # 3. Apply mask to Q-values before argmax
                # ============================================================
                action_mask, filtered_moves = self.move_filter.get_action_mask(
                    board=board,
                    legal_moves=valid_moves,
                    player_id=self.player_id,
                    action_size=self.action_size,
                    max_moves=self.max_filtered_moves,
                    device=self.device
                )
                
                if not filtered_moves:
                    # Fallback if filter returned nothing (shouldn't happen)
                    if valid_moves:
                        actions[i] = valid_moves[0]
                    continue
                
                # Apply mask to Q-values: masked actions get -inf
                masked_q = expected_q_values[i] + action_mask
                
                # Add uncertainty bonus (only for unmasked actions)
                if uncertainty_maps[i]:
                    for move in filtered_moves:
                        action_idx = self._move_to_action_index(move)
                        if action_idx is not None and action_mask[action_idx] == 0.0:
                            uncertainty = self.get_move_uncertainty(move, uncertainty_maps[i])
                            masked_q[action_idx] += uncertainty * self.uncertainty_exploration_multiplier
                
                # Select best action from masked Q-values
                best_action_idx = masked_q.argmax().item()
                best_move = self._action_index_to_move(best_action_idx)
                
                if best_move is not None:
                    actions[i] = best_move
                elif filtered_moves:
                    actions[i] = filtered_moves[0]  # Fallback
            else:
                # LEGACY: Original filtering (now includes Scout moves)
                valid_q_values = []
                valid_moves_filtered = []
                for move in valid_moves:
                    (r1, c1), (r2, c2) = move
                    dist = abs(r2 - r1) + abs(c2 - c1)
                    
                    # For Scout moves (dist > 1), map to 1-step direction action
                    if dist > 1:
                        if r2 != r1:
                            dr = 1 if r2 > r1 else -1
                            dc = 0
                        else:
                            dr = 0
                            dc = 1 if c2 > c1 else -1
                        virtual_move = ((r1, c1), (r1 + dr, c1 + dc))
                        action_idx = self._move_to_action_index(virtual_move)
                    else:
                        action_idx = self._move_to_action_index(move)
                    
                    if action_idx is None:
                        continue
                        
                    q_val = expected_q_values[i, action_idx].item()
                    
                    uncertainty = self.get_move_uncertainty(move, uncertainty_maps[i])
                    exploration_bonus = uncertainty * self.uncertainty_exploration_multiplier
                    valid_q_values.append(q_val + exploration_bonus)
                    valid_moves_filtered.append(move)
                
                if not valid_moves_filtered:
                    if valid_moves:
                        actions[i] = valid_moves[0]
                    continue
                
                best_move_idx = np.argmax(valid_q_values)
                actions[i] = valid_moves_filtered[best_move_idx]
            
            if self.pbs_instances and game_states and game_states[i]:
                actual_env_idx = env_indices[i]
                self.store_action_pbs_state(actions[i], expected_q_values[i].unsqueeze(0), uncertainty_maps[i], game_states[i], env_idx=actual_env_idx)
        
        return actions

    def update_pbs_batch(self, actions: List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]], 
                        game_states: List, acting_player: int):
        """
        Update history aggregator for a batch of actions from the opponent.
        Legacy method name kept for backward compatibility.
        
        Uses HistoryAggregator's update() method to track action history per position.
        """
        if not self.history:
            return
        
        # Process each action
        for i, action in enumerate(actions):
            if action is None:
                continue
            
            # Get history instance for this environment
            history_instance = None
            if self.history_instances and i < len(self.history_instances):
                history_instance = self.history_instances[i]
            else:
                history_instance = self.history
                
            if not history_instance:
                continue
            
            # Update history with this action
            game_state = game_states[i] if i < len(game_states) else None
            if game_state:
                history_instance.update(action, game_state, acting_player)

    def train_pbs(self, epochs: int = 5):
        """
        Train the history aggregator (AAREN) model using collected reveal data.
        Legacy method name kept for backward compatibility.
        """
        if not self.history:
            return
            
        # If using shared AAREN model (Parallel Env)
        if hasattr(self, 'shared_aaren') and self.shared_aaren and self.history_instances:
            # Aggregate training buffers from all instances
            total_buffer_size = sum(h.get_buffer_size() for h in self.history_instances)
            
            if total_buffer_size > 0:
                # Train using first instance (has shared optimizer)
                loss = self.history_instances[0].train(epochs=epochs)
                if loss is not None:
                    return loss
        else:
            # Single Env - train directly
            if self.history:
                return self.history.train(epochs=epochs)
        
        return None

    def replay(self, batch_size=None, episode=None) -> Optional[float]:
        """
        Train the Rainbow model using C51 Distributional Loss.
        Supports PER and N-step returns. Enforces warmup period.
        
        Args:
            batch_size: Override batch size
            episode: Current episode number (for PER beta annealing)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Warmup check - don't train until we have enough experiences
        try:
            from training_config import WARMUP_STEPS
        except ImportError:
            WARMUP_STEPS = 0
        
        if len(self.memory) < max(batch_size, WARMUP_STEPS):
            return None
        
        # --- Sample Batch (PER or Standard) with ATARAXOS OVERSAMPLING ---
        indices = None
        weights = None
        
        # Determine sample size (oversample if advantage filtering enabled)
        sample_size = batch_size
        if self.advantage_filtering:
            sample_size = batch_size * self.oversample_factor
        
        if self.per_enabled:
            # Anneal beta
            if episode is not None:
                self.memory.anneal_beta(episode)
            
            sample_result = self.memory.sample(sample_size)
            if sample_result is None:
                return None
            states, actions, rewards, next_states, dones, indices, weights = sample_result
        else:
            sample_result = self.memory.sample(sample_size)
            if sample_result is None:
                return None
            states, actions, rewards, next_states, dones = sample_result
        
        # Use n-step gamma if available
        gamma_to_use = self.gamma_n if hasattr(self, 'gamma_n') and self.n_steps > 1 else self.gamma
        
        # --- Check for NaNs in Inputs ---
        if torch.isnan(rewards).any():
             tqdm.write("[WARN] Warning: NaN detected in rewards batch. Skipping update.")
             return None
        
        # --- ATARAXOS ADVANTAGE FILTERING ---
        # Filter to top 25% by N-step TD error magnitude
        if self.advantage_filtering and states.size(0) > batch_size:
            with torch.no_grad():
                # V(s) from online network (max Q-value)
                current_probs = self.q_network(states).exp()
                current_v = (current_probs * self.support).sum(dim=2).max(dim=1)[0]
                
                # V(s') from target network
                next_probs = self.target_network(next_states).exp()
                next_v = (next_probs * self.support).sum(dim=2).max(dim=1)[0]
                
                # N-step TD error: |r + γ^n V(s') - V(s)|
                gamma_n = self.gamma_n if hasattr(self, 'gamma_n') else self.gamma
                td_errors = torch.abs(rewards + gamma_n * (1 - dones) * next_v - current_v)
            
            # Keep top-k transitions by TD error magnitude
            k = max(batch_size, self.min_batch_after_filter)
            if states.size(0) > k:
                top_k_indices = torch.topk(td_errors, k).indices
                states = states[top_k_indices]
                actions = actions[top_k_indices]
                rewards = rewards[top_k_indices]
                next_states = next_states[top_k_indices]
                dones = dones[top_k_indices]
                if weights is not None:
                    weights = weights[top_k_indices]
                if indices is not None:
                    indices = [indices[i] for i in top_k_indices.cpu().tolist()]

        # --- Distributional RL Target Calculation ---
        with torch.no_grad():
            # 1. Select best action in next state (Double DQN)
            # Use Online Network to select action
            next_log_probs_online = self.q_network(next_states)
            next_probs_online = next_log_probs_online.exp()
            next_q_values_online = (next_probs_online * self.support).sum(dim=2)
            next_actions = next_q_values_online.argmax(dim=1) # (batch)
            
            # 2. Get distribution of best action from Target Network
            next_log_probs_target = self.target_network(next_states)
            next_probs_target = next_log_probs_target.exp()
            
            # Gather distribution for the selected actions
            # next_actions: (batch) -> (batch, 1, atoms)
            next_action_probs = next_probs_target.gather(1, next_actions.view(-1, 1, 1).expand(-1, -1, self.num_atoms)).squeeze(1)
            
            # 3. Project Distribution (Categorical Algorithm)
            # T_z = r + gamma^n * z (if not done) - uses n-step gamma
            T_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_to_use * self.support.unsqueeze(0)
            T_z = T_z.clamp(min=self.v_min, max=self.v_max)
            
            # Compute L2 projection of T_z onto support
            b = (T_z - self.v_min) / self.delta_z
            
            # Safe Clamp to prevent Index Out of Bounds (floating point errors)
            b = b.clamp(min=0.0, max=float(self.num_atoms - 1))
            
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle batch size from actual sample (may differ due to PER)
            actual_batch_size = states.size(0)
            
            # Distribute probability mass
            # m is the projected distribution
            m = torch.zeros(actual_batch_size, self.num_atoms, device=self.device)
            
            # m_l = m_l + p(s', a') * (u - b)
            # m_u = m_u + p(s', a') * (b - l)
            
            # We need to use scatter_add because multiple atoms might project to same index
            offset = torch.linspace(0, (actual_batch_size - 1) * self.num_atoms, actual_batch_size, device=self.device).long().unsqueeze(1).expand(actual_batch_size, self.num_atoms)
            
            m.view(-1).scatter_add_(0, (l + offset).view(-1), (next_action_probs * (u.float() - b)).view(-1))
            m.view(-1).scatter_add_(0, (u + offset).view(-1), (next_action_probs * (b - l.float())).view(-1))
            
        # --- Calculate Loss ---
        # Get current log probabilities
        current_log_probs = self.q_network(states)
        
        # Gather log probs for the actions taken
        # actions: (batch) -> (batch, 1, atoms)
        action_log_probs = current_log_probs.gather(1, actions.view(-1, 1, 1).expand(-1, -1, self.num_atoms)).squeeze(1)
        
        # Calculate element-wise loss (for PER priority updates)
        elementwise_loss = -(m * action_log_probs).sum(dim=1)
        
        # Apply importance sampling weights if PER
        if weights is not None:
            c51_loss = (weights * elementwise_loss).mean()
        else:
            c51_loss = elementwise_loss.mean()
        
        # --- KL-REGULARIZATION (Anti-Cycling) ---
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.kl_reg_enabled:
            with torch.no_grad():
                ref_log_probs = self.reference_network(states)
            # KL(ref || current) for each action's distribution
            # Using log-space: KL = sum(p * (log_p - log_q))
            ref_probs = ref_log_probs.exp()
            kl_per_action = (ref_probs * (ref_log_probs - current_log_probs)).sum(dim=2)  # (batch, actions)
            # Average over actions and batch
            kl_loss = kl_per_action.mean()
        
        # --- ENTROPY REGULARIZATION (Bluffing/Mixed Strategies) ---
        entropy_bonus = torch.tensor(0.0, device=self.device)
        if self.entropy_reg_enabled:
            # Anneal entropy coefficient
            entropy_coeff = self.entropy_coeff_end + (self.entropy_coeff_start - self.entropy_coeff_end) * \
                           max(0, 1 - (episode or 0) / self.entropy_anneal_episodes)
            # Entropy = -sum(p * log(p))
            current_probs = current_log_probs.exp()
            action_entropy = -(current_probs * current_log_probs).sum(dim=2)  # (batch, actions)
            # Higher entropy = more exploration, so we SUBTRACT entropy loss (add entropy bonus)
            entropy_bonus = action_entropy.mean() * entropy_coeff
        
        # --- ATARAXOS DYNAMIC DAMPING (Magnetic Regularization) ---
        magnet_loss = torch.tensor(0.0, device=self.device)
        target_kl_loss = torch.tensor(0.0, device=self.device)
        if self.dynamic_damping:
            # Normalized training progress [0, 1]
            t = min(1.0, (episode or 0) / self.total_episodes)
            
            # Magnet Policy: KL toward uniform (quadratic decay)
            # alpha(t) = alpha_0 * (1 - t)^p
            alpha_t = self.magnet_start * ((1 - t) ** self.magnet_power)
            # KL(π || uniform) = log(num_actions) - H(π)
            current_probs = current_log_probs.exp()
            action_entropy_for_magnet = -(current_probs * current_log_probs).sum(dim=2).mean()
            magnet_loss = alpha_t * (math.log(self.action_size) - action_entropy_for_magnet)
            
            # Target KL: KL toward target network (quadratic growth)
            # beta(t) = beta_0 + (beta_T - beta_0) * t^p
            beta_t = self.target_kl_start + (self.target_kl_end - self.target_kl_start) * (t ** self.target_kl_power)
            with torch.no_grad():
                target_log_probs = self.target_network(states)
            target_probs = target_log_probs.exp()
            target_kl_loss = beta_t * (target_probs * (target_log_probs - current_log_probs)).sum(dim=2).mean()
        
        # --- TOTAL LOSS ---
        loss = c51_loss + self.kl_reg_weight * kl_loss - entropy_bonus + magnet_loss + target_kl_loss
        
        # Update reference network periodically (much slower than target)
        if self.kl_reg_enabled:
            self.reference_update_counter += 1
            if self.reference_update_counter >= self.ref_update_interval:
                self.reference_network.load_state_dict(self.q_network.state_dict())
                self.reference_update_counter = 0
        
        # Check for NaN/Inf (handle tensor properly)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
             print(f"[WARN] Warning: NaN/Inf detected in loss. Skipping update.")
             self.optimizer.zero_grad()
             return None
             self.optimizer.zero_grad()
             return None
        
        # Update PER priorities with TD-errors
        if self.per_enabled and indices is not None:
            td_errors = elementwise_loss.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
             
        self.optimizer.zero_grad()
        
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Capture gradient norms AFTER backward, BEFORE optimizer step
            self._capture_gradient_norms()
            
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Capture gradient norms AFTER backward, BEFORE optimizer step
            self._capture_gradient_norms()
            
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()
            
        return loss.item()

    def save_model(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }
        
        # Save history aggregator (AAREN) state
        if self.history:
            checkpoint['history_state_dict'] = self.history.state_dict()
            
        torch.save(checkpoint, filepath)
        
    def load_model(self, filepath):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step_count = checkpoint.get('step_count', 0)
            
            # Load history aggregator (AAREN) state
            if self.history and 'history_state_dict' in checkpoint:
                self.history.load_state_dict(checkpoint['history_state_dict'])
                print(f"[OK] AAREN history state loaded from {filepath}")
            elif self.history and 'pbs_state_dict' in checkpoint:
                # Backward compatibility with old PBS checkpoints - skip
                print(f"[WARN] Old PBS checkpoint found, skipping (incompatible with HistoryAggregator)")
                
            print(f"[OK] Model loaded from {filepath} (Step: {self.step_count})")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {filepath}: {e}")

    def get_average_q(self) -> float:
        """
        Get average Q-value from recent replay buffer samples.
        Returns 0.0 if no experiences are available.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        try:
            # Sample a small batch to estimate average Q
            sample_data = self.memory.sample(min(64, len(self.memory)))
            if len(sample_data) == 7:
                 states, actions, rewards, next_states, dones, _, _ = sample_data
            else:
                 states, actions, rewards, next_states, dones = sample_data
            
            self.q_network.eval()
            with torch.no_grad():
                log_probs = self.q_network(states)
                probs = log_probs.exp()
                expected_q = (probs * self.support).sum(dim=2)  # (batch, actions)
                # Get max Q for each state
                max_q = expected_q.max(dim=1)[0]
                avg_q = max_q.mean().item()
            self.q_network.train()
            
            return avg_q
        except Exception:
            return 0.0

    def get_exploration_entropy(self) -> float:
        """
        Get exploration level proxy using noisy network sigma values.
        Higher values indicate more exploration (larger noise).
        Returns 0.0 if noisy networks are not available.
        """
        try:
            total_sigma = 0.0
            num_params = 0
            
            for name, module in self.q_network.named_modules():
                if isinstance(module, NoisyLinear):
                    # Get average sigma magnitude
                    sigma_w = module.weight_sigma.abs().mean().item()
                    sigma_b = module.bias_sigma.abs().mean().item()
                    total_sigma += (sigma_w + sigma_b) / 2
                    num_params += 1
                    
            if num_params > 0:
                return total_sigma / num_params
            return 0.0
        except Exception:
            return 0.0
    
    def _capture_gradient_norms(self):
        """
        Capture gradient norms for AAREN and DQN after backward pass.
        Called internally during replay() to store values for later retrieval.
        """
        try:
            # Capture AAREN gradient norm
            aaren_model = None
            if hasattr(self, 'shared_aaren') and self.shared_aaren:
                aaren_model = self.shared_aaren
            elif self.history and self.history.owns_aaren:
                aaren_model = self.history.aaren
            
            if aaren_model is not None:
                total_norm = 0.0
                num_params = 0
                for param in aaren_model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.norm().item()
                        num_params += 1
                self._cached_aaren_grad_norm = total_norm / max(num_params, 1)
            
            # Capture DQN gradient norm (for comparison)
            total_dqn_norm = 0.0
            num_dqn_params = 0
            for param in self.q_network.parameters():
                if param.grad is not None:
                    total_dqn_norm += param.grad.norm().item()
                    num_dqn_params += 1
            self._cached_dqn_grad_norm = total_dqn_norm / max(num_dqn_params, 1)
            
        except Exception:
            pass  # Silently fail, keep previous values
    
    def get_aaren_grad_norm(self) -> float:
        """
        Get the cached AAREN gradient norm (computed during replay).
        
        This measures how much the DQN loss is affecting AAREN (end-to-end training).
        Higher values = AAREN is learning more actively from DQN feedback.
        """
        return getattr(self, '_cached_aaren_grad_norm', 0.0)
    
    def get_dqn_grad_norm(self) -> float:
        """
        Get the cached DQN network gradient norm (computed during replay).
        Useful for comparison with AAREN gradient norm.
        """
        return getattr(self, '_cached_dqn_grad_norm', 0.0)
    
    def get_aaren_embedding_stats(self) -> dict:
        """
        Get statistics about AAREN embeddings (mean, std, max).
        
        This helps track if embeddings are meaningful and changing over training.
        - Mean near 0 with reasonable std = good
        - All zeros = AAREN not producing useful features
        - Very high values = potential instability
        """
        stats = {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'active_positions': 0}
        
        try:
            if self.history is None:
                return stats
            
            # Get current embeddings (without recomputing)
            with torch.no_grad():
                embedding = self.history.get_embedding_tensor()
                
                # Count non-zero positions (active history)
                active = (embedding.abs().sum(dim=0) > 1e-6).sum().item()
                stats['active_positions'] = int(active)
                
                if active > 0:
                    stats['mean'] = embedding.mean().item()
                    stats['std'] = embedding.std().item()
                    stats['max'] = embedding.abs().max().item()
            
            return stats
        except Exception:
            return stats


    def _move_to_action_index(self, move):
        """
        Convert move ((r1, c1), (r2, c2)) to action index.
        Total Actions: 400
        Index = (StartPos[100]) * (Direction[4])
        
        Directions: 0=Right, 1=Left, 2=Down, 3=Up
        """
        (r1, c1), (r2, c2) = move
        
        dr = r2 - r1
        dc = c2 - c1
        
        # Determine direction
        if dr == 0 and dc > 0: 
            dir_idx = 0 # Right (0, 1)
        elif dr == 0 and dc < 0: 
            dir_idx = 1 # Left (0, -1)
        elif dr > 0 and dc == 0: 
            dir_idx = 2 # Down (1, 0)
        elif dr < 0 and dc == 0: 
            dir_idx = 3 # Up (-1, 0)
        else:
            # Fallback
            dir_idx = 0
        
        start_pos_idx = r1 * 10 + c1
        idx = (start_pos_idx * 4) + dir_idx
        return idx

    def _action_index_to_move(self, index):
        """Decode action index (0-399) back to move coordinates (length 1)."""
        if not (0 <= index < 400):
            return None
            
        start_pos_idx = index // 4
        dir_idx = index % 4
        
        r1 = start_pos_idx // 10
        c1 = start_pos_idx % 10
        
        # Directions: 0=Right, 1=Left, 2=Down, 3=Up
        # Matching environment.py: 0:(0,1), 1:(0,-1), 2:(1,0), 3:(-1,0) - wait, check filtering logic
        # My filter logic above used: Right(0), Left(1), Down(2), Up(3)
        dr_unit, dc_unit = [(0, 1), (0, -1), (1, 0), (-1, 0)][dir_idx]
        
        r2, c2 = r1 + dr_unit, c1 + dc_unit
        
        # Bounds check
        if 0 <= r2 < 10 and 0 <= c2 < 10:
            return (r1, c1), (r2, c2)
        return None

    def get_move_uncertainty(self, move, uncertainty_map):
        """Get uncertainty for a move based on target square"""
        if not uncertainty_map:
            return 0.0
            
        (r1, c1), (r2, c2) = move
        # Uncertainty is about the TARGET square (what's there?)
        # If target is empty, uncertainty is 0.
        # If target has enemy, uncertainty is high if we don't know it.
        
        # We need to look up (r2, c2) in uncertainty_map
        # uncertainty_map keys are (r, c) tuples?
        # Let's assume uncertainty_map is a dict or 2D array.
        
        if isinstance(uncertainty_map, dict):
            return uncertainty_map.get((r2, c2), 0.0)
        elif isinstance(uncertainty_map, (np.ndarray, torch.Tensor)):
            return uncertainty_map[r2, c2]
        return 0.0

    def store_action_pbs_state(self, action, q_values, uncertainty_map, game_state, env_idx=0):
        """
        Store PBS state for visualization/debugging.
        """
        # This is just for visualization, not core logic.
        pass
