"""
Configuration settings for DQN training.
"""

# =============================================================================
# HARDWARE & TRAINING SCALE CONFIGURATION (Fixed)
# =============================================================================
NUM_LANES = 8             # Reduced to 8 (Safe for 6GB VRAM)
NUM_ENVS = NUM_LANES      # LEGACY alias - always equals NUM_LANES (kept for compatibility)
BATCH_SIZE = 256          # Reduced to 256 (Safe for 6GB VRAM)
GAMMA = 0.995           # Discount factor (User requested 0.99 for long-term depth)
MEMORY_SIZE = 150000      # 200k on GPU (~2.2GB) to leave room for model and batches
LEARNING_RATE = 0.00003    # Reduced from 0.0001 for stable distributional RL
NUM_EPISODES = 35000      # Total training episodes (individual games, not batches)
SAVE_INTERVAL = 250       # Save model/export agent every N episodes
PLOT_INTERVAL = 100       # Save metrics plots every N episodes
EVAL_INTERVAL = 100       # Evaluate agent every N episodes
REPLAY_UPDATE_INTERVAL = 1    # Train every step (maximized for data augmentation)
TARGET_UPDATE_INTERVAL = 10000  # Soft update target network every N steps
REWARD_SCALE = 1.0        # Scaling factor for reward calculations

# Warmup Period (must collect this many experiences before training starts)
WARMUP_STEPS = 10000      # 10k steps warmup for stable learning

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

# Opponent Selection Probabilities (SIMPLIFIED for early training)
# Phase 1: Easy opponents to learn basics, will be adjusted as training progresses
OPPONENT_LEAGUE_PROB = 0.0      # 0% historical opponents (disabled for now)
OPPONENT_RANDOM_PROB = 1.0      # 100% random agent (easy to beat)
OPPONENT_GREEDY_PROB = 0.0      # 0% heuristic agent (too hard for now)
OPPONENT_SELF_PROB = 0.0        # 0% self-play (disabled for now)

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
