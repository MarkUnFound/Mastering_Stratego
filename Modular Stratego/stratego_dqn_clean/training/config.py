"""
Centralized Configuration for Double DQN + AAREN Training.
"""

class TrainingConfig:
    """Configuration for Double DQN training."""
    
    # =============================================================================
    # HARDWARE & TRAINING SCALE
    # =============================================================================
    BATCH_SIZE = 32          # Hard ceiling for 6GB VRAM
    LEARNING_RATE = 1e-4
    ADAM_EPS = 1.5e-4        # Explicit eps per spec
    GAMMA = 0.99
    
    # =============================================================================
    # TARGET NETWORK
    # =============================================================================
    TARGET_UPDATE_FREQ = 1000  # Hard update every N steps
    USE_SOFT_UPDATE = False
    SOFT_UPDATE_TAU = 0.005    # FORBIDDEN: tau > 0.01
    
    # =============================================================================
    # EXPLORATION (ε-greedy)
    # =============================================================================
    EPSILON_START = 0.5
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 100000
    
    # =============================================================================
    # MEMORY
    # =============================================================================
    BUFFER_SIZE = 50000
    MIN_BUFFER_SIZE = 1000  # Minimum before training starts
    
    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    TRAIN_FREQ = 4           # Train every N steps
    GRADIENT_CLIP = 1.0      # Hardcoded per spec
    MAX_STEPS = 10000000     # Default: 10M for continuous training
    
    # =============================================================================
    # ENVIRONMENT & CURRICULUM
    # =============================================================================
    MAX_TURNS = 1000         # Maximum turns per episode
    
    # =============================================================================
    # CHECKPOINTS
    # =============================================================================
    CHECKPOINT_FREQ = 1000   # Save every 1000 steps per spec
    CHECKPOINT_EPISODE_FREQ = 1000 # Save every 1000 episodes
    CHECKPOINT_DIR = "./checkpoints"
    
    # =============================================================================
    # EVALUATION
    # =============================================================================
    EVAL_FREQ = 10000        # Steps between evaluations
    EVAL_GAMES = 100
    
    # =============================================================================
    # SELF-CHECKS & DIAGNOSTICS
    # =============================================================================
    SELFCHECK_ENABLED = True
    SELFCHECK_FREQ = 100     # Check every N optimization steps
    
    # =============================================================================
    # LOGGING
    # =============================================================================
    LOG_FREQ = 1000
    SAVE_FREQ = 50000
    
    # =============================================================================
    # RECOVERY PROTOCOL
    # =============================================================================
    RECOVERY_LR_FACTOR = 0.5  # Reduce LR by 50% on gradient death
    
    # =============================================================================
    # SAFETY & DEBUGGING
    # =============================================================================
    DEBUG_STRICT_VALIDATION = False  # SLOW: Check entire valid move list
    ENABLE_SAFE_GUARDS = True        # FAST: Check basic rules (lakes, friendly fire)

    # =============================================================================
    # VECTORIZATION
    # =============================================================================
    NUM_ENVS = 4              # Number of parallel environments
