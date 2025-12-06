"""
Configuration settings for DQN training.
"""

# Hyperparameters
# Hyperparameters
NUM_ENVS = 4 # Reduced to 4 for stability and speed
BATCH_SIZE = 32
TRACE_LENGTH = 8 # Unused for Rainbow
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000 # Increased buffer size for transitions
LEARNING_RATE = 0.0001
NUM_EPISODES = 35000
SAVE_INTERVAL = 250
EVAL_INTERVAL = 100
PREFETCH_QUEUE_SIZE = 4
REPLAY_UPDATE_INTERVAL = 4 # Train every 4 steps
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

