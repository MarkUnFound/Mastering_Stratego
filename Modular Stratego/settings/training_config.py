"""
Configuration settings for DQN training.
"""

# =============================================================================
# HARDWARE & TRAINING SCALE CONFIGURATION (Fixed)
# =============================================================================
NUM_LANES = 8             # Reduced to 8 (Safe for 6GB VRAM)
NUM_ENVS = NUM_LANES      # LEGACY alias - always equals NUM_LANES (kept for compatibility)
BATCH_SIZE = 128         # Reduced to 256 (Safe for 6GB VRAM)
GAMMA = 0.999           # Discount factor - EXTREMELY HIGH for 1500-step long-term flag capture planning in Phase 4
MEMORY_SIZE = 75000      # 200k on GPU (~2.2GB) to leave room for model and batches
LEARNING_RATE = 0.00003    # Reduced from 0.0001 for stable distributional RL
NUM_EPISODES = 1000000    # Effectively infinite - train until manually stopped
SAVE_INTERVAL = 1000       # Save model/export agent every N episodes
PLOT_INTERVAL = 500       # Save metrics plots every N episodes
EVAL_INTERVAL = 500       # Evaluate agent every N episodes
REPLAY_UPDATE_INTERVAL = 8    # Train every 8 steps (balances data diversity vs update frequency)
TARGET_UPDATE_INTERVAL = 2000   # Faster target updates for Phase 1 learning (was 5000)
MARL_TARGET_UPDATE_INTERVAL = 5000   # Smoothing for Phase 4 MARL
REWARD_SCALE = 1.0        # Scaling factor for reward calculations
MARL_REWARD_SCALE = 1.0   # Full terminal reward scale — dampening was killing signal contrast

# =============================================================================
# EXPLORATION SETTINGS (Noisy Networks Only - True Rainbow DQN)
# =============================================================================
# Epsilon-greedy DISABLED: Noisy Networks handle exploration (state-dependent)
# This follows the original Rainbow DQN paper which replaces ε-greedy with NoisyNets
EXPLORATION_EPSILON_START = 0.0   # Disabled - Noisy Networks handle exploration
EXPLORATION_EPSILON_END = 0.0     # Disabled
EXPLORATION_EPSILON_DECAY = 1     # Not used

# Entropy Regularization (Soft Q-Learning)
ENTROPY_REG_ENABLED = True
ENTROPY_COEFF_START = 0.1
ENTROPY_COEFF_END = 0.01
ENTROPY_ANNEAL_EPISODES = 50000

# =============================================================================
# CURRICULUM TURN LIMITS (Per-phase game length)
# =============================================================================
# Longer episodes let the agent experience full-game dynamics at each phase
PHASE_1_MAX_TURNS = 200   # (Accelerated) Enough turns to learn material + flag capture (100 per agent)
PHASE_2_MAX_TURNS = 400   # (Accelerated) Partial obs — memory-intensive games with restricted length
PHASE_3_MAX_TURNS = 1000  # Self-play — full strategic depth
PHASE_4_MAX_TURNS = 1500  # League — unrestricted competitive games
DEFAULT_MAX_TURNS = 1500  # Fallback (matches Phase 4)

# Single source of truth: phase number -> max turns
PHASE_MAX_TURNS = {
    1: PHASE_1_MAX_TURNS,
    2: PHASE_2_MAX_TURNS,
    3: PHASE_3_MAX_TURNS,
    4: PHASE_4_MAX_TURNS,
}

# Warmup Period (must collect this many experiences before training starts)
WARMUP_STEPS = 1000       # Reduced further for faster Phase 1 start

# =============================================================================
# IMITATION LEARNING SETTINGS (Learn from Heuristic Expert)
# =============================================================================
IMITATION_ENABLED = True      # Use heuristic policy as expert demonstration
IMITATION_RATIO = 0.10        # 10% of actions use heuristic expert (reduced for Phase 1)
IMITATION_EPISODES = 2000     # Only use imitation for first N episodes (then pure RL)
IMITATION_REWARD_BOOST = 1.5  # Multiply reward for imitation actions (encourages learning good moves)

# Multi-Step Returns (N-Step DQN)
N_STEPS = 5               # 5-step returns for deeper credit assignment in long-horizon games (200-500 moves)
GAMMA_N = GAMMA ** N_STEPS  # Pre-computed gamma^n

# Episode-Level Replay Settings (Option B: Trajectory Segment Sampling)
EPISODE_REPLAY_ENABLED = False        # Disable dual-buffer episode replay for speed
EPISODE_REPLAY_MAX_EPISODES = 500     # Max stored complete episodes (FIFO eviction)
EPISODE_REPLAY_SEGMENT_LENGTH = 16    # Contiguous steps per segment sample
EPISODE_REPLAY_MIX_RATIO = 0.25      # 25% of batch from episode segments, 75% from PER


# Learning Rate Scheduler
LR_SCHEDULER_ENABLED = False
LR_SCHEDULER_STEP_SIZE = 5000   # Reduce LR every N episodes
LR_SCHEDULER_GAMMA = 0.9        # Multiply LR by this factor

# Prioritized Experience Replay
PER_ENABLED = True              # Prioritized Experience Replay enabled
PER_ALPHA = 0.6                 # Priority exponent (0 = uniform, 1 = full prioritization)
PER_BETA_START = 0.4            # Initial importance sampling weight
PER_BETA_END = 1.0              # Final importance sampling weight
PER_BETA_ANNEAL_EPISODES = 10000  # Episodes to anneal beta

# Data Augmentation (State/Action Symmetry)
ENABLE_DATA_AUGMENTATION = True # Double/Triple transitions by flipping/rotating board
AUGMENTATION_TYPES = ["flip"] # Reduced to flip only to minimize data pressure and noise

# League Settings (Setup League)
LEAGUE_INTERVAL = 500 # Run setup league every N episodes
LEAGUE_GENERATIONS = 3 # Number of generations to evolve per interval

# Visualization settings
GENERATE_GIFS = False # Whether to generate GIFs of games
GIF_INTERVAL = 100   # Generate GIF every N episodes (reduced frequency)

# PBS Optimization Settings
PBS_UPDATE_INTERVAL = 4  # Update PBS every 4 steps (faster inference for MARL)
PBS_SKIP_SIMPLE_MOVES = True  # Skip AAREN for obvious 1-square non-attack moves
PBS_CACHE_UNCERTAINTY = True  # Cache uncertainty maps until beliefs change

# AAREN Performance Optimizations
AAREN_USE_FP16 = True           # Use half-precision inference (~30% faster)
AAREN_USE_TORCHSCRIPT = False   # Disabled - AAREN uses dynamic ops that don't compile
AAREN_HIDDEN_SIZE = 64          # Keep at 64 for checkpoint compatibility
AAREN_NUM_LAYERS = 3            # Keep at 3 for checkpoint compatibility
AAREN_USE_REVEAL_DATA = True    # When False, AAREN supervised training skips reveal data
                                # (debug flag to test if PieceActionAaren overfits on open info)

# History Aggregator Settings (Simplified PBS replacement)
# Uses AAREN embeddings instead of explicit belief distributions
HISTORY_EMBEDDING_SIZE = 64     # Size of AAREN embedding per position (matches hidden_size)

# Rainbow DQN Performance Optimizations  
USE_TORCH_COMPILE = False     # Disabled on Windows due to missing Triton dependency
TORCH_COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"

# League Training Settings (Opponent Diversity)
LEAGUE_TRAINING_ENABLED = True
LEAGUE_SAVE_INTERVAL = 500      # Save agent to league every N episodes
LEAGUE_MAX_AGENTS = 50          # Max historical agents to keep

# Opponent Selection Probabilities (ADJUSTED for faster learning)
# Phase 1: Mix of easy and medium opponents to learn basics AND winning strategy
OPPONENT_LEAGUE_PROB = 0.0      # 0% historical opponents (disabled for now)
OPPONENT_RANDOM_PROB = 0.85     # 85% random agent (easy to beat)
OPPONENT_GREEDY_PROB = 0.15     # 15% heuristic agent (learning pressure)
OPPONENT_SELF_PROB = 0.0        # 0% self-play (disabled for now)

# Epoch-Based Opponent Cycling (Variance Reduction)
# Instead of randomly sampling opponent type every episode, lock all lanes
# to the same opponent type for OPPONENT_CYCLE_INTERVAL episodes, then rotate.
# This reduces reward variance and gives stable learning signal per block.
OPPONENT_CYCLE_INTERVAL = 50    # Episodes per opponent type block

# =============================================================================
# CURRICULUM LEARNING SETTINGS
# =============================================================================
CURRICULUM_ENABLED = True       # Enable 5-phase curriculum learning
CURRICULUM_START_PHASE = 4      # Start directly at League Training for MARL

# Phase Transition Thresholds (LOWERED for faster progression)
PHASE_1_WIN_THRESHOLD_RANDOM = 0.70     # 70% win rate vs random (was 90%)
PHASE_1_WIN_THRESHOLD_HEURISTIC = 0.50  # 50% win rate vs heuristic (was 60%)
PHASE_2_PBS_ACCURACY_THRESHOLD = 0.70   # 70% PBS prediction accuracy
PHASE_2_WIN_THRESHOLD = 0.55            # 55% overall win rate

# Phase Episode Limits (min, max)
PHASE_1_MIN_EPISODES = 2000
PHASE_1_MAX_EPISODES = 5000
PHASE_2_MIN_EPISODES = 3000
PHASE_2_MAX_EPISODES = 7000
PHASE_3_MIN_EPISODES = 5000
PHASE_3_MAX_EPISODES = 15000

# Scenario Drill Settings (Phase 5)
SCENARIO_DRILL_INTERVAL = 1000  # Run scenario drills every N episodes during Phase 4

# =============================================================================
# TEST-TIME SEARCH SETTINGS (Policy Refinement)
# =============================================================================
# Search amplifies network quality during gameplay without extra training
SEARCH_ENABLED = False           # Disabled during training, enable for evaluation
SEARCH_DEPTH = 2                 # How many moves ahead to look (1-3 recommended)
SEARCH_BUDGET = 50               # Max simulations per decision
SEARCH_TOP_K = 5                 # Only expand top-K moves from Q-values
SEARCH_MIN_MOVES = 10            # Skip search if fewer legal moves available

# =============================================================================
# KL-REGULARIZATION SETTINGS (Anti-Cycling)
# =============================================================================
# Prevents strategy oscillation by anchoring to a reference policy
KL_REG_ENABLED = True            # Enable KL divergence regularization
KL_REG_WEIGHT = 0.01             # Weight of KL term in total loss
REF_POLICY_UPDATE_INTERVAL = 50000  # Update reference network every N steps

# =============================================================================
# ENTROPY REGULARIZATION SETTINGS (Bluffing/Mixed Strategies)
# =============================================================================
# Encourages stochastic policies to avoid being exploited
ENTROPY_REG_ENABLED = True       # Re-enabled for Phase 4 to prevent policy collapse
ENTROPY_COEFF_START = 0.01       # Small coefficient (keep exploration without chaos)
ENTROPY_COEFF_END = 0.001        # Final entropy coefficient (decays over training)
ENTROPY_ANNEAL_EPISODES = 50000  # Episodes to anneal entropy coefficient

# =============================================================================
# ATARAXOS ADVANTAGE FILTERING (Training Efficiency)
# =============================================================================
# Filter training to high-impact transitions based on TD error magnitude
ADVANTAGE_FILTERING_ENABLED = False  # DISABLED: causes 4× slowdown (2 extra forward passes on 4× batch every 2 steps). Re-enable at Phase 3+ when Q-values are meaningful.
ADVANTAGE_OVERSAMPLE_FACTOR = 4   # Sample 4× batch, keep top 25% by TD error
ADVANTAGE_MIN_BATCH = 64          # Minimum batch size after filtering

# =============================================================================
# ATARAXOS DYNAMIC DAMPING (Magnetic Regularization)
# =============================================================================
# Dual KL regularization with power-law annealing schedule
DYNAMIC_DAMPING_ENABLED = True
TOTAL_TRAINING_EPISODES = 100000  # For schedule normalization

# Magnet Policy (toward uniform): Starts high, decays to prevent overconfidence
MAGNET_COEFF_START = 0.1          # α_0: Initial coefficient
MAGNET_COEFF_END = 0.001          # α_T: Final coefficient  
MAGNET_POWER = 2.0                # Quadratic decay: α(t) = α_0 * (1-t)^p

# Target KL (toward target network): Starts low, grows for stability
TARGET_KL_COEFF_START = 0.001     # β_0: Initial coefficient
TARGET_KL_COEFF_END = 0.1         # β_T: Final coefficient
TARGET_KL_POWER = 2.0             # Quadratic growth: β(t) = β_0 + (β_T-β_0) * t^p

# =============================================================================
# ATARAXOS UPDATE-EQUIVALENCE SEARCH (Test-Time)
# =============================================================================
# Vectorized search with Magnetic Mirror Descent
UE_SEARCH_ENABLED = False         # Enable for evaluation only (slow)
UE_NUM_WORLDS = 1000              # Number of opponent configs to sample
UE_ROLLOUT_DEPTH = 5              # Ply depth (reduced from 40 for speed)
UE_MMD_STEP_SIZE = 0.1            # η for Magnetic Mirror Descent
UE_TEMPERATURE = 0.5              # Softmax temperature for prior

# REWARDS ARE NOW CONSOLIDATED IN distributional_reward.py
