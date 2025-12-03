"""
Configuration settings for DQN training.
"""

# Hyperparameters
NUM_ENVS = 4 # Reduced to 4 to strictly limit memory usage for 6GB GPU (running 2 processes)
BATCH_SIZE = 128  # Large batch size to maximize GPU usage and amortize PER sampling cost
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995  # Slower decay for longer training
TARGET_UPDATE = 1000
MEMORY_SIZE = 50000 # Reduced to 50k to save VRAM
LEARNING_RATE = 0.0001
NUM_EPISODES = 35000  # Total episodes to train
SAVE_INTERVAL = 500   # Save model every N episodes
EVAL_INTERVAL = 100  # Evaluate every N episodes
PREFETCH_QUEUE_SIZE = 4 # Size of the prefetch queue
REPLAY_UPDATE_INTERVAL = 2 # Train every N steps (train very frequently)
REPLAY_UPDATES_PER_STEP = 4 # Multiple gradient updates per training step (maximize GPU work)
TARGET_UPDATE_INTERVAL = 1000 # Update target network every N steps

# League Settings
LEAGUE_INTERVAL = 500 # Run setup league every N episodes
LEAGUE_GENERATIONS = 3 # Number of generations to evolve per interval

# Visualization settings
GENERATE_GIFS = False # Whether to generate GIFs of games
GIF_INTERVAL = 100   # Generate GIF every N episodes (reduced frequency)
