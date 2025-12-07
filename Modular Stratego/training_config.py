"""
Configuration settings for DQN training.
"""

# Hyperparameters
# Hyperparameters
NUM_ENVS = 4 # Reduced to 4 for stability and speed
BATCH_SIZE = 128
TRACE_LENGTH = 8 # Unused for Rainbow
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000 # Increased buffer size for transitions (auto-scales by VRAM)
LEARNING_RATE = 0.0001
NUM_EPISODES = 35000
SAVE_INTERVAL = 250
EVAL_INTERVAL = 100
PREFETCH_QUEUE_SIZE = 4
REPLAY_UPDATE_INTERVAL = 32 # Train every 32 steps (optimized for speed, replay ratio ~1.0)
REPLAY_UPDATES_PER_STEP = 1 # Unused in current script
TARGET_UPDATE_INTERVAL = 5000
REWARD_SCALE = 1.0  # Scaling factor for reward calculations

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

# Opponent Selection Probabilities
OPPONENT_LEAGUE_PROB = 0.5      # 50% historical opponents
OPPONENT_RANDOM_PROB = 0.2      # 20% random agent
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
PHASE_1_MIN_EPISODES = 500
PHASE_1_MAX_EPISODES = 2000
PHASE_2_MIN_EPISODES = 1000
PHASE_2_MAX_EPISODES = 3000
PHASE_3_MIN_EPISODES = 2000
PHASE_3_MAX_EPISODES = 5000

# Scenario Drill Settings (Phase 5)
SCENARIO_DRILL_INTERVAL = 1000  # Run scenario drills every N episodes during Phase 4

# =============================================================================
# REWARD SHAPING WEIGHTS
# =============================================================================
REWARD_WEIGHT_OUTCOME = 1.0     # Win/Loss terminal reward weight
REWARD_WEIGHT_MATERIAL = 0.5    # Combat/capture reward weight
REWARD_WEIGHT_EPISTEMIC = 0.3   # Information gain reward weight
REWARD_WEIGHT_POSITIONAL = 0.2  # Strategic positioning reward weight

# =============================================================================
# DISTRIBUTIONAL RL REWARD SHAPING (C51-Compatible Anti-Stall)
# =============================================================================
# These settings are NORMALIZED for C51 Distributional RL with V_MIN=-3, V_MAX=+3
# All rewards are small so cumulative returns stay within the distribution support

DISTRIBUTIONAL_REWARD_ENABLED = True  # Use distributional-compatible rewards
DISTRIBUTIONAL_WEIGHT = 1.0           # Full weight on distributional rewards
ENV_REWARD_WEIGHT = 0.0               # Disable environment rewards (use only shaped)

# Anti-stall penalties (tiny to stay within bounds)
DIST_STEP_PENALTY = -0.005            # -0.005 per step (~100 steps = -0.5)
DIST_DRAW_PENALTY = -0.8              # Draw = WORSE than loss to force aggression

# Terminal rewards (normalized, NOT at V_MIN/V_MAX bounds)
DIST_WIN_REWARD = 1.0                 # Win (flag capture or elimination)
DIST_LOSS_PENALTY = -1.0              # Loss

# Combat rewards (material signal)
DIST_CAPTURE_SCALE = 0.1              # +0.1 * (rank/10) = max +0.1 per capture
DIST_LOSS_SCALE = -0.05               # -0.05 per piece lost

# Information gain (crucial for variance learning in C51)
DIST_REVEAL_BONUS = 0.02              # Bonus for revealing enemy rank
DIST_FIRST_REVEAL_BONUS = 0.03        # Extra bonus for first reveal of a type

# Strategic bonuses
DIST_SPY_KILLS_MARSHAL = 0.15         # Spy kills Marshal (rare, valuable)
DIST_MINER_DEFUSES_BOMB = 0.08        # Miner removes Bomb (strategic)

# Legacy aliases for backwards compatibility
AGGRESSION_ENABLED = DISTRIBUTIONAL_REWARD_ENABLED
AGGRESSION_WEIGHT = DISTRIBUTIONAL_WEIGHT
AGGRESSION_STEP_PENALTY = DIST_STEP_PENALTY
AGGRESSION_DRAW_PENALTY = DIST_DRAW_PENALTY
AGGRESSION_WIN_REWARD = DIST_WIN_REWARD
AGGRESSION_LOSS_PENALTY = DIST_LOSS_PENALTY
AGGRESSION_ATTACK_WIN_BASE = DIST_CAPTURE_SCALE
AGGRESSION_ATTACK_LOSE_PENALTY = DIST_LOSS_SCALE
AGGRESSION_INFO_BONUS = DIST_REVEAL_BONUS
AGGRESSION_RANK_SCALE = 0.01  # Reduced for normalization
AGGRESSION_TERRITORY_ADVANCE = 0.02
AGGRESSION_RETREAT_PENALTY = -0.01
AGGRESSION_SPY_CAPTURE = DIST_SPY_KILLS_MARSHAL
AGGRESSION_MINER_CAPTURE = DIST_MINER_DEFUSES_BOMB
