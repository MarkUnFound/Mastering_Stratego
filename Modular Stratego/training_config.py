"""
Configuration settings for DQN training.
"""

# =============================================================================
# HARDWARE & TRAINING SCALE CONFIGURATION (Fixed)
# =============================================================================
NUM_LANES = 16            # Increased to 16 for higher data throughput (VRAM allows)
NUM_ENVS = NUM_LANES      # LEGACY alias - always equals NUM_LANES (kept for compatibility)
BATCH_SIZE = 1024         # Increased to 1024 for GPU saturation and stable gradients
GAMMA = 0.99           # Discount factor (User requested 0.99 for long-term depth)
MEMORY_SIZE = 500000      # 250k on GPU (~5GB VRAM) to minimize CPU usage
LEARNING_RATE = 0.0001    # Learning rate (increased for ±100 reward scale)
NUM_EPISODES = 35000      # Total training episodes (individual games, not batches)
SAVE_INTERVAL = 500       # Save model/export agent every N episodes
PLOT_INTERVAL = 100       # Save metrics plots every N episodes
EVAL_INTERVAL = 100       # Evaluate agent every N episodes
PREFETCH_QUEUE_SIZE = 4   # Prefetch queue for data loading
REPLAY_UPDATE_INTERVAL = 1    # Train every step (maximized for data augmentation)
TARGET_UPDATE_INTERVAL = 5000  # Soft update target network every N steps
REWARD_SCALE = 1.0        # Scaling factor for reward calculations

# Multi-Step Returns (N-Step DQN)
N_STEPS = 5               # 5-step returns for better credit assignment in sparse rewards
GAMMA_N = GAMMA ** N_STEPS  # Pre-computed gamma^n

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
PBS_UPDATE_INTERVAL = 2  # Update PBS every 2 steps (faster inference)
PBS_SKIP_SIMPLE_MOVES = True  # Skip AAREN for obvious 1-square non-attack moves
PBS_CACHE_UNCERTAINTY = True  # Cache uncertainty maps until beliefs change

# AAREN Performance Optimizations
AAREN_USE_FP16 = True           # Use half-precision inference (~30% faster)
AAREN_USE_TORCHSCRIPT = False   # Disabled - AAREN uses dynamic ops that don't compile
AAREN_HIDDEN_SIZE = 64          # Keep at 64 for checkpoint compatibility
AAREN_NUM_LAYERS = 3            # Keep at 3 for checkpoint compatibility

# League Training Settings (Opponent Diversity)
LEAGUE_TRAINING_ENABLED = True
LEAGUE_SAVE_INTERVAL = 500      # Save agent to league every N episodes
LEAGUE_MAX_AGENTS = 50          # Max historical agents to keep

# Opponent Selection Probabilities (adjusted for early training)
# More random opponents = easier learning; scale up difficulty as agent improves
OPPONENT_LEAGUE_PROB = 0.1      # 10% historical opponents (reduced from 50%)
OPPONENT_RANDOM_PROB = 0.6      # 60% random agent (increased from 20%)
OPPONENT_GREEDY_PROB = 0.2      # 20% greedy agent
OPPONENT_SELF_PROB = 0.1        # 10% self-play

# =============================================================================
# CURRICULUM LEARNING SETTINGS
# =============================================================================
CURRICULUM_ENABLED = True       # Enable 5-phase curriculum learning
CURRICULUM_START_PHASE = 1      # Start from Phase 1 (or resume from saved state)

# Phase Transition Thresholds
PHASE_1_WIN_THRESHOLD_RANDOM = 0.90     # 90% win rate vs random
PHASE_1_WIN_THRESHOLD_HEURISTIC = 0.60  # 60% win rate vs heuristic
PHASE_2_PBS_ACCURACY_THRESHOLD = 0.70   # 70% PBS prediction accuracy
PHASE_2_WIN_THRESHOLD = 0.55            # 55% overall win rate

# Phase Episode Limits (min, max)
PHASE_1_MIN_EPISODES = 5000
PHASE_1_MAX_EPISODES = 10000
PHASE_2_MIN_EPISODES = 5000
PHASE_2_MAX_EPISODES = 10000
PHASE_3_MIN_EPISODES = 5000
PHASE_3_MAX_EPISODES = 15000

# Scenario Drill Settings (Phase 5)
SCENARIO_DRILL_INTERVAL = 1000  # Run scenario drills every N episodes during Phase 4

# REWARDS ARE NOW CONSOLIDATED IN distributional_reward.py
