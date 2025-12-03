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

# League Settings
LEAGUE_INTERVAL = 500 # Run setup league every N episodes
LEAGUE_GENERATIONS = 3 # Number of generations to evolve per interval

# Visualization settings
GENERATE_GIFS = False # Whether to generate GIFs of games
GIF_INTERVAL = 100   # Generate GIF every N episodes (reduced frequency)
