"""
Vanilla DQN Agent for Stratego Game
Features:
- Feed-Forward Architecture (AAREN HistoryAggregator for memory)
- Epsilon-Greedy Exploration
- Standard Q-Learning (MSE/Huber Loss)
- Double DQN Target Calculation
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
from networks import VanillaDQN
from history_aggregator import HistoryAggregator
from aaren import PieceActionAaren

from prioritized_memory import StandardReplayBuffer, PrioritizedReplayBuffer, Experience

# Heuristic Action Filter for Top-100 move selection
from heuristic_filter import HeuristicMoveFilter

# Legacy C51 constants removed — Vanilla DQN uses raw Q-values directly.




# VanillaDQN is now imported from networks module

import sys





class DQNAgent:
    """Vanilla DQN Agent for Stratego with AAREN"""
    
    def __init__(self, player_id: int, device, 
                 state_size: int = 200, action_size: int = 400,
                 lr: float = 0.0001, gamma: float = 0.99, 
                 buffer_size: int = 10000, batch_size: int = 32,
                 use_pbs: bool = True, num_envs: int = 1,
                 inference_only: bool = False):
        
        self.player_id = player_id
        self.device = device
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.name = f"Rainbow Agent {player_id}"
        self.num_envs = num_envs
        self.use_pbs = use_pbs
        self.inference_only = inference_only
        
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
        
        # Support Variables Removed for Vanilla DQN
        
        # History tracking (AAREN)
        self.pbs = None  # Legacy alias for self.history
        self.pbs_instances = []  # Legacy alias for self.history_instances
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
                # However, for supervised train_history(), we give the first instance its own optimizer
                for i in range(num_envs):
                    history_instance = HistoryAggregator(
                        player_id, device, 
                        hidden_size=AAREN_HIDDEN_SIZE,
                        num_layers=AAREN_NUM_LAYERS,
                        shared_aaren_model=self.shared_aaren
                    )
                    if i == 0:
                        history_instance.optimizer = optim.AdamW(self.shared_aaren.parameters(), lr=0.001, weight_decay=0.01)
                    else:
                        history_instance.optimizer = None
                    self.history_instances.append(history_instance)
                self.history = self.history_instances[0]
            else:
                self.history = HistoryAggregator(player_id, device, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
                self.history_instances = [self.history]
        
        # Legacy aliases (point to history)
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
        
        # Vanilla Networks
        # 15 (Board) + HISTORY_EMBEDDING_SIZE (AAREN embeddings) = 79 Channels (with 64)
        self.input_channels = 15 + self.history_embedding_size
        self.q_network = VanillaDQN(input_shape=(self.input_channels, 10, 10), output_size=action_size).to(device)
        self.target_network = VanillaDQN(input_shape=(self.input_channels, 10, 10), output_size=action_size).to(device)
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
        self.reference_network = VanillaDQN(
            input_shape=(self.input_channels, 10, 10), 
            output_size=action_size
        ).to(device)
        
        # PyTorch 2.0+ Compilation (optional speedup)
        try:
            from training_config import USE_TORCH_COMPILE, TORCH_COMPILE_MODE
            if USE_TORCH_COMPILE:
                self.reference_network = torch.compile(self.reference_network, mode=TORCH_COMPILE_MODE)
        except (ImportError, Exception):
            pass  # Silently skip if unavailable
            
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
        self.optimizer = None
        if not self.inference_only:
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
                LR_SCHEDULER_ENABLED, LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA
            )
        except ImportError:
            PER_ENABLED = False
            LR_SCHEDULER_ENABLED = False
        
        # Standard Replay Buffer
        self.memory = None
        if not self.inference_only:
            self.memory = StandardReplayBuffer(buffer_size, device=device)
            print(f"[OK] [P{self.player_id}] Standard Replay Buffer enabled")
        
        # Episode-Level Replay Buffer (shadow storage alongside PER)
        try:
            from training_config import (
                EPISODE_REPLAY_ENABLED, EPISODE_REPLAY_MAX_EPISODES,
                EPISODE_REPLAY_SEGMENT_LENGTH, EPISODE_REPLAY_MIX_RATIO
            )
            self.episode_replay_enabled = EPISODE_REPLAY_ENABLED
            self.episode_replay_mix_ratio = EPISODE_REPLAY_MIX_RATIO
        except ImportError:
            self.episode_replay_enabled = False
            EPISODE_REPLAY_MAX_EPISODES = 500
            EPISODE_REPLAY_SEGMENT_LENGTH = 16
            self.episode_replay_mix_ratio = 0.25
        
        if self.episode_replay_enabled and not self.inference_only:
            from prioritized_memory import EpisodicReplayBuffer
            self.episode_memory = EpisodicReplayBuffer(
                max_episodes=EPISODE_REPLAY_MAX_EPISODES,
                segment_length=EPISODE_REPLAY_SEGMENT_LENGTH,
                device=device,
                num_envs=num_envs if hasattr(self, 'num_envs') else 8
            )
            print(f"[OK] [P{self.player_id}] Episode Replay enabled (max={EPISODE_REPLAY_MAX_EPISODES}, seg={EPISODE_REPLAY_SEGMENT_LENGTH}, mix={self.episode_replay_mix_ratio})")
        else:
            self.episode_memory = None
        
        # Data Augmentation Settings
        try:
            from training_config import ENABLE_DATA_AUGMENTATION, AUGMENTATION_TYPES
            self.aug_enabled = ENABLE_DATA_AUGMENTATION
            self.aug_types = AUGMENTATION_TYPES
        except ImportError:
            self.aug_enabled = False
            self.aug_types = []
        
        # Learning Rate Scheduler
        self.scheduler = None
        if LR_SCHEDULER_ENABLED and not self.inference_only:
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
        self.q_network = VanillaDQN(input_shape=(self.input_channels, 10, 10), output_size=self.action_size).to(self.device)
        self.target_network = VanillaDQN(input_shape=(self.input_channels, 10, 10), output_size=self.action_size).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=0.01)
        self.memory.clear()
        self.step_count = 0
        self.update_target_network()
        
        self.reset_history()
            
    def reset_history(self):
        """Reset the history aggregator state."""
        if self.history:
            if self.num_envs > 1:
                for h in self.history_instances:
                    h.reset()
            else:
                self.history.reset()

    # Backward compatibility alias
    reset_pbs = reset_history

    def enable_history(self, num_envs: int = None):
        """
        Enable AAREN history aggregation for an agent created with use_pbs=False.
        Used during curriculum phase transitions to dynamically enable AAREN.
        
        Args:
            num_envs: Number of parallel environments (uses self.num_envs if not provided)
        """
        if self.use_pbs:
            return  # Already enabled
            
        if num_envs is None:
            num_envs = self.num_envs
            
        self.use_pbs = True
        self.num_envs = num_envs
        
        # Import AAREN settings
        try:
            from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS
        except ImportError:
            AAREN_HIDDEN_SIZE = 64
            AAREN_NUM_LAYERS = 2
        
        if num_envs > 1:
            # Create shared AAREN for parallel environments
            self.shared_aaren = PieceActionAaren(
                input_size=24, hidden_size=AAREN_HIDDEN_SIZE, 
                num_layers=AAREN_NUM_LAYERS, output_size=12, device=self.device
            ).to(self.device)
            
            for i in range(num_envs):
                history_instance = HistoryAggregator(
                    self.player_id, self.device, 
                    hidden_size=AAREN_HIDDEN_SIZE,
                    num_layers=AAREN_NUM_LAYERS,
                    shared_aaren_model=self.shared_aaren
                )
                if i == 0:
                    history_instance.optimizer = optim.AdamW(self.shared_aaren.parameters(), lr=0.001, weight_decay=0.01)
                else:
                    history_instance.optimizer = None
                self.history_instances.append(history_instance)
            self.history = self.history_instances[0]
        else:
            self.history = HistoryAggregator(self.player_id, self.device, hidden_size=AAREN_HIDDEN_SIZE)
            self.history_instances = [self.history]
        
        # Update legacy aliases
        self.pbs = self.history
        self.pbs_instances = self.history_instances
        
        print(f"[OK] AAREN history enabled for {self.name} ({num_envs} environments)")

    # Backward compatibility alias
    enable_pbs = enable_history

    def update_target_network(self):
        """Soft update target network."""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
    def get_state_representation(self, board, pbs_instance=None, full_observability=False):
        """
        Convert raw board to feature tensor.
        Includes AAREN history embeddings for enemy piece prediction.
        Total channels: 15 (Board) + HISTORY_EMBEDDING_SIZE (AAREN) = 79 (default)
        
        AAREN is never bypassed: the DQN always sees AAREN embeddings (or zeros
        if AAREN is not yet active). This ensures co-evolution of the DQN backbone
        and AAREN representations across all curriculum phases.
        
        Args:
            board: Game board tensor or GameState object
            pbs_instance: History aggregator instance (AAREN)
            full_observability: Controls board visibility only (does NOT affect AAREN channels)
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
            # Enemy pieces: Negative values (known pieces -1 to -12, hidden pieces -20)
            from board import HIDDEN_PIECE
            features[12] = (((board < 0) & (board > LAKE_SQUARE)) | (board == HIDDEN_PIECE)).float()
        else:
            # Player 2: Own pieces are negative (-1 to -12)
            for i in range(1, 13):
                features[i-1] = (board == -i).float()
            # Enemy pieces: Positive values (known pieces 1 to 12) AND hidden pieces (-20)
            # Note: For P2, the visible board represents hidden enemies as HIDDEN_PIECE (-20) too.
            from board import HIDDEN_PIECE
            features[12] = ((board > 0) | (board == HIDDEN_PIECE)).float()
            
        # Obstacles (Channel 13)
        features[13] = (board == LAKE_SQUARE).float()
        
        # Channel 14: Empty squares (0)
        features[14] = (board == 0).float()
        
        state_tensor = features
        
        # --- AAREN EMBEDDING CHANNELS ---
        # AAREN is never bypassed with ground-truth one-hot encoding.
        # The DQN must learn to rely on AAREN as the memory and piece prediction
        # module by classifying enemy actions, not by cheating with direct board info.
        if pbs_instance is not None:
            # Use AAREN embeddings from HistoryAggregator
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
    
    def get_board_only_representation(self, board):
        """
        Extract 15-channel board features only (no AAREN channels).
        
        Used for replay buffer storage. AAREN embeddings are reconstructed
        at replay time using stored history snapshots.
        
        Args:
            board: Game board tensor or GameState object
        Returns:
            Tensor of shape (15, 10, 10)
        """
        from game_state import GameState
        if isinstance(board, GameState):
            board = board.board
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).to(self.device)
        
        features = torch.zeros((15, 10, 10), device=self.device)
        
        if self.player_id == 1:
            for i in range(1, 13):
                features[i-1] = (board == i).float()
            from board import HIDDEN_PIECE
            features[12] = (((board < 0) & (board > LAKE_SQUARE)) | (board == HIDDEN_PIECE)).float()
        else:
            for i in range(1, 13):
                features[i-1] = (board == -i).float()
            from board import HIDDEN_PIECE
            features[12] = ((board > 0) | (board == HIDDEN_PIECE)).float()
        
        features[13] = (board == LAKE_SQUARE).float()
        features[14] = (board == 0).float()
        
        return features
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer with automated augmentation"""
        # Process states
        state_processed = self.get_state_representation(state, pbs_instance=self.pbs)
        next_state_processed = self.get_state_representation(next_state, pbs_instance=self.pbs)
        
        self._add_experience(state_processed, action, reward, next_state_processed, done)

    def _add_experience(self, state, action, reward, next_state, done, is_battle=False,
                         history_snapshot=None, next_history_snapshot=None):
        """
        Internal helper to add experience to memory with potential augmentation.
        Always expects 'state' and 'next_state' to be processed tensors.
        'action' can be a move tuple or index.
        
        Args:
            is_battle: If True, indicates this transition involved a capture event
            history_snapshot: AAREN history snapshot for state (for replay-time reconstruction)
            next_history_snapshot: AAREN history snapshot for next_state
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
        # Clip reward to default range to avoid instability
        reward_clipped = max(-10.0, min(10.0, reward))
        
        # SELF-IMITATION LEARNING: Detect winning experiences for priority boost
        # Win rewards are now ±1.0, so use 0.8 as threshold
        is_winning_experience = (reward_clipped >= 0.8 and done)
        
        # Pass priority boost flags and history snapshots to PER
        if hasattr(self.memory, 'add') and 'is_winning_experience' in self.memory.add.__code__.co_varnames:
            self.memory.add(state, action_idx, reward_clipped, next_state, done, 
                          is_winning_experience=is_winning_experience, is_battle=is_battle,
                          history_snapshot=history_snapshot, next_history_snapshot=next_history_snapshot)
        else:
            self.memory.add(state, action_idx, reward_clipped, next_state, done)
        self.step_count += 1
        
        # 3. Add Augmented Experiences (without history snapshots — augmented states
        # don't have meaningful AAREN histories since positions are transformed)
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
        
        Stores 15-channel board-only tensors + AAREN history snapshots.
        AAREN embeddings are reconstructed at replay() time using the current model.
        
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
        # Compute 15-channel board-only tensors (no AAREN — reconstructed at replay time)
        state_tensors = []
        next_state_tensors = []
        history_snapshots = []
        next_history_snapshots = []
        
        for i in range(len(states)):
            state_tensors.append(self.get_board_only_representation(states[i]))
            next_state_tensors.append(self.get_board_only_representation(next_states[i]))
            
            # Only snapshot active envs (snapshots are now just tensor copies — fast)
            if active_mask[i]:
                if self.history_instances and i < len(self.history_instances):
                    snap = self.history_instances[i].get_history_snapshot()
                elif self.history:
                    snap = self.history.get_history_snapshot()
                else:
                    snap = None
            else:
                snap = None
            
            history_snapshots.append(snap)
            next_history_snapshots.append(snap)  # Same snapshot (update happens after remember)
        
        state_tensors = torch.stack(state_tensors)
        next_state_tensors = torch.stack(next_state_tensors)
        
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
            
            # Episode buffer: shadow-store every raw transition
            if self.episode_replay_enabled and self.episode_memory is not None:
                self.episode_memory.add(
                    i, state_tensors[i], action, reward, next_state_tensors[i], dones[i],
                    history_snapshot=history_snapshots[i],
                    next_history_snapshot=next_history_snapshots[i]
                )
            
            # Standard 1-step: add directly using central helper
            self._add_experience(state_tensors[i], action, reward, next_state_tensors[i], dones[i],
                                 is_battle=is_battle, history_snapshot=history_snapshots[i],
                                 next_history_snapshot=next_history_snapshots[i])
            
            # Episode buffer: finalize episode on done
            if dones[i] and self.episode_replay_enabled and self.episode_memory is not None:
                # Outcome: sign of terminal reward (+1 win, -1 loss, 0 draw)
                outcome = 1.0 if reward > 0.5 else (-1.0 if reward < -0.5 else 0.0)
                total_reward = reward  # terminal reward as proxy
                self.episode_memory.end_episode(i, outcome, total_reward)

    def act(self, state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], game_state=None, temperature: float = 0.0):
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
                expected_q_values = self.q_network(state_tensor) # (1, actions)
                
            base_q_values = expected_q_values.squeeze(0) # (actions)
            
        self.q_network.train()
        
        # Uncertainty is now implicitly handled via AAREN embeddings
        uncertainty_map = {}
            
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

        if temperature > 0.0:
            # Boltzmann exploration
            q_tensor = torch.tensor(valid_q_values, dtype=torch.float32)
            probs = torch.softmax(q_tensor / temperature, dim=0).numpy()
            best_move_idx = np.random.choice(len(valid_q_values), p=probs)
        else:
            best_move_idx = np.argmax(valid_q_values)
        best_move = valid_moves_filtered[best_move_idx]
        
        if self.history and game_state:
            self.store_action_state(best_move, base_q_values.unsqueeze(0), uncertainty_map, game_state)
            
        return best_move

    def get_batch_state_representation(self, states, game_states=None, full_observability=False):
        """
        Convert batch of states to tensor, integrating PBS beliefs if available.
        
        Args:
            states: List of game board states
            game_states: List of GameState objects (for PBS lookup)
            full_observability: Controls board visibility (does NOT bypass AAREN)
        """
        # Convert states to tensors
        state_tensors = []
        for i, state in enumerate(states):
            # Always use AAREN embeddings regardless of observability mode
            # AAREN is never bypassed — it co-evolves with the DQN from Phase 1
            pbs_instance = None
            if self.pbs_instances and game_states and i < len(game_states) and game_states[i]:
                pbs_instance = self.pbs_instances[i]
            elif self.pbs and game_states and i < len(game_states) and game_states[i]:
                # Fallback to single PBS instance if pbs_instances list not used
                pbs_instance = self.pbs
                
            # Use get_state_representation for each state
            state_tensor = self.get_state_representation(state, pbs_instance=pbs_instance, full_observability=full_observability)
            state_tensors.append(state_tensor)
            
        # Stack into batch
        return torch.stack(state_tensors)

    def act_batch(self, states, valid_moves_list, game_states=None, env_indices=None, full_observability=False, temperature: float = 0.0) -> List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Batch action selection with epsilon-greedy exploration
        
        Args:
            states: List of game board states
            valid_moves_list: List of valid moves for each environment
            game_states: List of GameState objects
            env_indices: Environment indices for PBS lookup
            full_observability: Controls board visibility (does NOT bypass AAREN)
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
                         
        # Apply random and imitation actions
        for i in range(batch_size):
            if not valid_moves_list[i]:
                continue
                
            if imitation_mask[i] and self.use_heuristic_filter:
                # Ask heuristic expert for top move
                try:
                    expert_move = self.move_filter.expert_action(
                        board=states[i].board if hasattr(states[i], 'board') else states[i],
                        legal_moves=valid_moves_list[i],
                        player_id=self.player_id
                    )
                    actions[i] = expert_move
                except Exception:
                    pass
            
            # If not imitating, check epsilon greedy
            if actions[i] is None and random_action_mask[i]:
                actions[i] = random.choice(valid_moves_list[i])
        
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
                    expected_q_values = self.q_network(state_tensor) # (batch, actions)
                except RuntimeError as e:
                    print(f"DEBUG: act_batch CRASH. state_tensor shape: {state_tensor.shape}")
                    raise e
            
        self.q_network.train()
        
        # 4. VECTORIZED HEURISTIC ACTION MASKING
        batched_masks = []
        for i in range(batch_size):
            valid_moves = valid_moves_list[i]
            
            if not valid_moves or actions[i] is not None:
                # Mask out everything if no moves or already acted (e.g. epsilon greedy)
                batched_masks.append(torch.full((self.action_size,), float('-inf'), device=self.device))
                continue
                
            board = states[i]
            if hasattr(board, 'board'):
                board = board.board
                
            if self.use_heuristic_filter:
                action_mask, _ = self.move_filter.get_action_mask(
                    board=board,
                    legal_moves=valid_moves,
                    player_id=self.player_id,
                    action_size=self.action_size,
                    max_moves=self.max_filtered_moves,
                    device=self.device
                )
                batched_masks.append(action_mask)
            else:
                # Legacy direct masking without heuristic scoring
                mask = torch.full((self.action_size,), float('-inf'), device=self.device)
                for move in valid_moves:
                    idx = self._move_to_action_index(move)
                    if idx is not None:
                        mask[idx] = 0.0
                batched_masks.append(mask)

        # Stack into a batch tensor: (Batch, 400)
        batch_masks_tensor = torch.stack(batched_masks)
        
        # Apply masks to all Q-values efficiently on the GPU
        masked_q = expected_q_values + batch_masks_tensor
        
        # FIX: Ensure at least one finite value in each row to prevent NaNs in softmax
        # This happens for inactive environments or epsilon-greedy actions already taken.
        invalid_rows = (batch_masks_tensor == float('-inf')).all(dim=1)
        if invalid_rows.any():
            # For invalid rows, set first action to 0.0 to prevent NaN.
            # These rows will either be overwritten by epsilon-greedy actions or ignored later.
            masked_q[invalid_rows, 0] = 0.0
        
        # Determine best actions using batched PyTorch operations
        if temperature > 0.0:
            probs = torch.softmax(masked_q / temperature, dim=1)
            try:
                best_action_indices = torch.multinomial(probs, 1).squeeze(1).tolist()
            except RuntimeError as e:
                # Fallback and debug info
                print(f"DEBUG: Multinomial CRASH. Invalid rows: {invalid_rows.sum().item()}")
                print(f"DEBUG: Probs contains NaN: {torch.isnan(probs).any().item()}")
                # Last resort fallback to greedy if multinomial still fails
                best_action_indices = masked_q.argmax(dim=1).tolist()
        else:
            best_action_indices = masked_q.argmax(dim=1).tolist()
            
        # Map indices back to the exact Stratego move
        for i in range(batch_size):
            # Skip if already acted (Epsilon-greedy, imitation, or no valid moves)
            if actions[i] is not None or not valid_moves_list[i]:
                continue
                
            best_idx = best_action_indices[i]
            
            # Find the true move from the valid moves list that generated this index
            chosen_move = None
            for move in valid_moves_list[i]:
                if self._move_to_action_index(move) == best_idx:
                    chosen_move = move
                    break
                    
            if chosen_move is not None:
                actions[i] = chosen_move
            else:
                # Fallback in extreme edge cases
                actions[i] = valid_moves_list[i][0]
                
            if self.history_instances and game_states and game_states[i]:
                actual_env_idx = env_indices[i]
                self.store_action_state(actions[i], expected_q_values[i].unsqueeze(0), uncertainty_maps[i], game_states[i], env_idx=actual_env_idx)
        
        return actions

    def update_history_batch(self, actions: List[Optional[Tuple[Tuple[int, int], Tuple[int, int]]]], 
                        game_states: List, acting_player: int):
        """
        Update history aggregator for a batch of actions from the opponent.
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
            if game_state is not None:
                history_instance.update(action, game_state, acting_player)

    # Backward compatibility alias
    update_pbs_batch = update_history_batch

    def train_history(self, epochs: int = 5):
        """
        Train the history aggregator (AAREN) model using collected reveal data.
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

    # Backward compatibility alias
    train_pbs = train_history

    def replay(self, batch_size=None, episode=None) -> Optional[float]:
        """
        Train the Vanilla DQN model using Standard Q-Learning (MSE Loss).
        Enforces warmup period.
        
        Args:
            batch_size: Override batch size
            episode: Current episode number
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
        
        # --- Sample Batch ---
        sample_result = self.memory.sample(batch_size)
        if sample_result is None:
            return None
        states, actions, rewards, next_states, dones = sample_result
        hist_snapshots = [None] * states.size(0)
        
        # --- RECONSTRUCT AAREN EMBEDDINGS FROM STORED HISTORIES ---
        if self.history and states.size(1) == 15:  # 15ch = board-only format
            aaren_embeddings = self.history.reconstruct_embeddings_batch(
                hist_snapshots, device=self.device
            )
            
            aaren_tensor = torch.stack(aaren_embeddings)  # (batch, 64, 10, 10)
            states = torch.cat([states, aaren_tensor], dim=1)  # (batch, 79, 10, 10)
            
            next_padding = torch.zeros(
                next_states.size(0), self.history_embedding_size, 10, 10,
                device=self.device
            )
            next_states = torch.cat([next_states, next_padding], dim=1)
        
        # --- Standard Q-Learning Target Calculation ---
        with torch.no_grad():
            # 1. Double DQN Target Calculation
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                next_q_values_online = self.q_network(next_states)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
            
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                next_q_values_target = self.target_network(next_states)
                
            # Gather Q-values for selected actions
            next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
            
            # Compute target Q-values Q(s', a') = r + gamma * max(Q(s', a'))
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # --- Calculate Loss ---
        # Get current Q-values
        with torch.amp.autocast('cuda', enabled=self.amp_enabled):
            current_q_values = self.q_network(states)
        
        # Gather Q-values for the actions taken
        current_q = current_q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        
        # Smooth L1 Loss (Huber Loss)
        criterion = nn.SmoothL1Loss()
        base_loss = criterion(current_q, target_q)
        loss = base_loss
        
        # Check for NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
             print(f"[WARN] Warning: NaN/Inf detected in loss. Skipping update.")
             self.optimizer.zero_grad(set_to_none=True)
             return None
             
        self.optimizer.zero_grad(set_to_none=True)
        
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
            # PyTorch 2.6+ defaults weights_only=True; allowlist deque for our checkpoints
            import collections
            try:
                torch.serialization.add_safe_globals([collections.deque])
            except AttributeError:
                pass  # Older PyTorch versions don't have this
            
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback for complex checkpoints (we trust our own saves)
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Robust state dict unwrapping for backward compatibility
            # Checkpoints saved without torch.compile lack '_orig_mod.', while loaded models may expect it (or vice versa)
            def load_robustly(module, state_dict):
                module_keys = set(module.state_dict().keys())
                dict_keys = set(state_dict.keys())
                
                # If module expects _orig_mod. but dict doesn't have it (older checkpoint into compiled model)
                if any(k.startswith('_orig_mod.') for k in module_keys) and not any(k.startswith('_orig_mod.') for k in dict_keys):
                    state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
                # If dict has _orig_mod. but module doesn't expect it (newer checkpoint into uncompiled model)
                elif any(k.startswith('_orig_mod.') for k in dict_keys) and not any(k.startswith('_orig_mod.') for k in module_keys):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                    
                module.load_state_dict(state_dict)

            load_robustly(self.q_network, checkpoint['q_network_state_dict'])
            load_robustly(self.target_network, checkpoint['target_network_state_dict'])
            
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as opt_err:
                # Common when switching between single-env and multi-env (optimizer param count changes)
                print(f"[WARN] Failed to load optimizer state (architecture/env mismatch): {opt_err}")
                
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

    def state_dict(self, include_buffers=True) -> dict:
        """Get full agent state dict."""
        state = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'step_count': self.step_count
        }
        if getattr(self, 'optimizer', None) is not None:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.history:
            state['history_state_dict'] = self.history.state_dict(include_buffers=include_buffers)
        
        if include_buffers:
            if self.memory is not None:
                state['memory_state_dict'] = self.memory.state_dict()
            if hasattr(self, 'episode_memory') and self.episode_memory:
                 state['episode_memory_state_dict'] = self.episode_memory.state_dict()
        
        return state

    def load_state_dict(self, state_dict):
        """Load full agent state from dict."""
        def load_robustly(module, sd):
            module_keys = set(module.state_dict().keys())
            dict_keys = set(sd.keys())
            if any(k.startswith('_orig_mod.') for k in module_keys) and not any(k.startswith('_orig_mod.') for k in dict_keys):
                sd = {f'_orig_mod.{k}': v for k, v in sd.items()}
            elif any(k.startswith('_orig_mod.') for k in dict_keys) and not any(k.startswith('_orig_mod.') for k in module_keys):
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            module.load_state_dict(sd)
            
        load_robustly(self.q_network, state_dict['q_network_state_dict'])
        load_robustly(self.target_network, state_dict['target_network_state_dict'])
        
        if 'optimizer_state_dict' in state_dict and getattr(self, 'optimizer', None) is not None:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            
        self.step_count = state_dict.get('step_count', 0)
        
        if self.history and 'history_state_dict' in state_dict:
            self.history.load_state_dict(state_dict['history_state_dict'])
            
        if 'memory_state_dict' in state_dict and self.memory is not None:
            self.memory.load_state_dict(state_dict['memory_state_dict'])
            
        if 'episode_memory_state_dict' in state_dict and hasattr(self, 'episode_memory') and self.episode_memory:
            self.episode_memory.load_state_dict(state_dict['episode_memory_state_dict'])

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
            
            # Handle all sample formats: 5 (standard), 7 (old PER), 9 (PER + history snapshots)
            if len(sample_data) == 9:
                states, actions, rewards, next_states, dones, _, _, _, _ = sample_data
            elif len(sample_data) == 7:
                states, actions, rewards, next_states, dones, _, _ = sample_data
            else:
                states, actions, rewards, next_states, dones = sample_data
            
            # PERF: Zero-pad instead of reconstructing AAREN — this is a diagnostic
            # metric, exact embeddings aren't needed for a Q-value estimate.
            if states.size(1) == 15:
                padding = torch.zeros(states.size(0), self.history_embedding_size, 10, 10, 
                                      device=self.device)
                states = torch.cat([states, padding], dim=1)
            
            self.q_network.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                    q_values = self.q_network(states)  # (batch, actions) — raw Q-values
                # Get max Q for each state
                max_q = q_values.max(dim=1)[0]
                avg_q = max_q.mean().item()
            self.q_network.train()
            
            return avg_q
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        Get the cached AAREN gradient norm.
        
        Since AAREN is currently trained via supervised learning (train_history)
        rather than end-to-end through the DQN, this returns the gradient norm
        captured during the last supervised training pass.
        """
        if self.history and hasattr(self.history, '_last_grad_norm'):
            return self.history._last_grad_norm
        
        # Fallback to the value captured during end-to-end replay (if applicable)
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

    def store_action_state(self, action, q_values, uncertainty_map, game_state, env_idx=0):
        """
        Store action state for visualization/debugging.
        """
        # This is just for visualization, not core logic.
        pass

    # Backward compatibility alias
    store_action_pbs_state = store_action_state
