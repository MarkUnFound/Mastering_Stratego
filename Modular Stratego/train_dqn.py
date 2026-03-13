"""
Training Script for Rainbow DQN Agents in Stratego

Features:
- Single-agent focus (Agent1 trains, Agent2 for opponents)
- League training: Auto-switches from Agent2 to historical opponents
- Diverse opponents: League (50%), Random (20%), Greedy (20%), Self (10%)
- AAREN inference for fair opponent play
"""

# Set matplotlib backend to non-interactive BEFORE any imports
# This prevents Tkinter errors when used in multi-threaded environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (thread-safe, no GUI)

import torch
import numpy as np
import random
import os
import sys
import glob
import queue
import threading
import time
import gc  # Garbage collection for memory cleanup
import copy
import traceback
import json
from typing import List, Tuple, Optional
from tqdm import tqdm

# Add the project root and subdirectories to sys.path to enable legacy imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
for d in ['environment', 'network', 'settings', 'test', 'visualizers', 'utils']:
    sys.path.append(os.path.join(project_root, d))
# Also add the parent directory for Research-level imports
sys.path.append(os.path.dirname(project_root))

# Consolidated imports from reorganized packages
from environment import (
    StrategoEnvironment, ParallelStrategoEnvironment, LAKE_SQUARE,
    PieceType, PIECE_RANKS, GameState,
    CurriculumManager, TrainingPhase, HeuristicOpponent, SmartHeuristicOpponent, TrueRandomOpponent,
    LeagueManager, HeuristicSetupAgent
)
from network import (
    DQNAgent, create_unified_reward_shaper, StrategoRewardConfig,
    RandomAgent, GreedyAgent, OpponentPool, RandomSetupAgent,
    get_random_exploiter, RusherAgent, TurtleAgent, FlankingAgent,
    LaneManager, MetricsTracker, Checkpointer, get_random_starting_player
)
from settings import *
from utils import save_training_history, load_training_history
from test.preflight_checks import run_preflight_checks
from test.scenario_drills import get_scenario_drill, get_random_scenario
from visualizers import plot_training_progress, create_training_gif, create_episode_gif, plot_additional_metrics



def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     plot_interval: int = PLOT_INTERVAL,
                     model_save_path: str = "dqn_models",
                     generate_gifs: bool = True,
                     init_weights: str = None,
                     pbt_reporter = None):
    """
    Train Rainbow DQN agent with league-based diverse opponents.
    Early training uses self-play, then transitions to historical opponents.
    
    Args:
        num_episodes: Maximum number of episodes to train
        save_interval: Episodes between checkpoint saves
        plot_interval: Episodes between progress plots
        model_save_path: Directory to save models and logs
        generate_gifs: Whether to generate visualization GIFs
        init_weights: Path to initial model weights (for PBT cloning)
        pbt_reporter: Optional PBTMetricsReporter for supervisor communication
    """
    device = torch.device('cpu')  # Default to CPU
    if torch.cuda.is_available():
        try:
            # Actually try to use CUDA to verify it works
            _ = torch.tensor([1.0], device='cuda')
            device = torch.device('cuda')
        except (AssertionError, RuntimeError) as e:
            print(f"[WARN] CUDA detected but not usable: {e}")
            print("   Falling back to CPU. Install PyTorch with CUDA support for GPU acceleration.")
    print(f"Using device: {device}")
    
    # Resolve model_save_path to absolute path (relative to script location)
    # This ensures consistent save location regardless of current working directory
    if not os.path.isabs(model_save_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = os.path.join(script_dir, model_save_path)
    model_save_path = os.path.abspath(model_save_path)
    print(f"Model save path: {model_save_path}")
    
    # Use fixed configuration from training_config.py
    num_envs = NUM_LANES
    batch_size = BATCH_SIZE
    memory_size = MEMORY_SIZE
    
    # Optimize GPU settings for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create model directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize Parallel Environment (provides worker processes for each lane)
    # The ParallelStrategoEnvironment creates subprocess workers for true parallelism
    print(f"Initializing {num_envs} parallel lanes...")
    parallel_env = ParallelStrategoEnvironment(num_envs=num_envs, device=device)

    
    # Initialize Agents
    print("Initializing DQN Agents...")
    agent1 = DQNAgent(player_id=1, device=device, lr=LEARNING_RATE, batch_size=batch_size, num_envs=num_envs, buffer_size=memory_size)
    # Agent2: Active learning MARL agent. Own buffer and optimizer.
    # AAREN history will be enabled for Agent 2 when reaching Phase 4
    agent2 = DQNAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=batch_size, num_envs=num_envs, buffer_size=memory_size, use_pbs=False, inference_only=False)
    print("[INFO] Agent 2 set to active learning (MARL). AAREN history disabled for early phases (will enable at Phase 4)")
    
    # Initialize Setup Agents (using fast heuristic instead of neural network)
    # This saves ~2 seconds per episode while maintaining strategic setups
    setup_agent1 = HeuristicSetupAgent(player_id=1, device=device)
    setup_agent2 = HeuristicSetupAgent(player_id=-1, device=device)
    # Using HeuristicSetupAgent (fast strategic placement)
    
    # Master reward configuration (shared by all agents and shapers)
    master_reward_config = StrategoRewardConfig.from_training_config()
    
    # Initialize League Manager and Opponent Pool
    league_dir = os.path.join(model_save_path, "league")
    league_manager = LeagueManager(league_dir=league_dir, max_agents=LEAGUE_MAX_AGENTS)
    
    opponent_pool = OpponentPool(
        league_manager=league_manager,
        device=device,
        league_prob=OPPONENT_LEAGUE_PROB,
        random_prob=OPPONENT_RANDOM_PROB,
        greedy_prob=OPPONENT_GREEDY_PROB,
        self_prob=OPPONENT_SELF_PROB,
        config=master_reward_config
    )
    
    # Specialized opponents (for non-league matches)
    random_agent = RandomAgent()
    random_setup_agent = RandomSetupAgent(player_id=-1)
    greedy_agent = GreedyAgent(device=device, player_id=-1, config=master_reward_config)
    heuristic_agent = HeuristicOpponent(device=device, player_id=-1)  # Frozen heuristic for Phase 2
    smart_heuristic_agent = SmartHeuristicOpponent(device=device, player_id=-1)  # Strong heuristic opponent
    true_random_agent = TrueRandomOpponent(device=device, player_id=-1)  # Truly random opponent (easiest)
    
    # Initialize Curriculum Manager
    curriculum = None
    if CURRICULUM_ENABLED:
        curriculum = CurriculumManager(start_phase=CURRICULUM_START_PHASE, save_dir=model_save_path)
        print(f"[INFO] Curriculum enabled: Phase {curriculum.current_phase.value} ({curriculum.get_phase_config().name})")
        
        # Set initial observability based on phase
        if curriculum.should_use_full_observability():
            parallel_env.set_full_observability(True)
            print("   Full observability mode: ENABLED (Phase 1)")
        
        # Set max turns based on curriculum phase
        from training_config import PHASE_MAX_TURNS, DEFAULT_MAX_TURNS
        max_turns = PHASE_MAX_TURNS.get(curriculum.current_phase.value, DEFAULT_MAX_TURNS)
        parallel_env.set_max_turns(max_turns)
        print(f"   Max turns per game: {max_turns} (Phase {curriculum.current_phase.value})")

        # Ensure Agent 2 has history enabled if starting in Phase 4+
        if curriculum.current_phase.value >= 4 and not agent2.use_pbs:
            agent2.enable_history(num_envs)
            print("[INFO] Agent 2 AAREN history ENABLED immediately (starting in Phase 4+)")
    
    # MARL Optimization: Disable heuristic move filtering for Agent 2
    # During League Training, we want Agent 2 to learn pure RL policies quickly
    # without the massive overhead of CPU-bound heuristic masking on every step.
    if curriculum and curriculum.current_phase.value >= 4:
        agent2.use_heuristic_filter = False
        print("[INFO] Agent 2 heuristic filtering DISABLED for MARL speed")
    
    # Reward shaper is initialized per lane later
    
    # Note: PIECE_VALUE_TRACKING removed in favor of simpler metrics
    piece_tracker = None
    # if PIECE_VALUE_TRACKING:
    #     piece_tracker = PieceValueTracker(save_path=os.path.join(model_save_path, "piece_value_tracking.json"))
    #     print(f"[INFO] Piece value tracking enabled ({piece_tracker.games_tracked} games loaded)")
    
    # Enable mixed precision for faster training
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # --- Load Existing Models ---
    checkpointer = Checkpointer(save_dir=model_save_path)
    metrics_tracker = MetricsTracker(save_dir=model_save_path)
    start_episode = 0

    # PBT Cloning: If init_weights is provided, load those weights (priority over checkpoints)
    if init_weights and os.path.exists(init_weights):
        try:
            # PBT init_weights might be raw .pt or .pth or .tar.gz
            if init_weights.endswith('.tar.gz'):
                checkpointer._load_from_archive(agent1, init_weights)
            else:
                agent1.load_model(init_weights)
            print(f"[PBT] Loaded cloned weights from {init_weights}")
            start_episode = 0  # Start fresh episode count for cloned worker
        except Exception as e:
            print(f"[WARN] Failed to load init_weights: {e}")
            init_weights = None  # Fall back to checkpoint loading
    
    # Look for Rainbow models (only if not using PBT init_weights)
    if not init_weights:
        start_episode = checkpointer.load_agent_models(agent1, agent2)
    
    # Load existing training history if resuming
    if start_episode > 0:
        if metrics_tracker.load():
            print(f"Loaded training history.")
    
    # Metrics dict reference for backward compatibility (to be gradually removed)
    metrics = metrics_tracker.metrics

    # Load global_step from metrics if resuming, otherwise start at 0
    global_step = metrics.get('global_step', 0)
    completed_episodes = start_episode  # Count of completed individual games
    if start_episode > 0 and global_step > 0:
        print(f"[INFO] Resuming from global step {global_step:,}, episode {completed_episodes}")
    
    # ==========================================================================
    # MULTI-LANE STATE TRACKING
    # Each lane (environment) is independent and tracks its own state
    # ==========================================================================
    
    # Initialize per-lane tracking
    lane_game_states = [None] * num_envs      # Current game state per lane
    lane_valid_moves = [None] * num_envs      # Valid moves per lane
    lane_current_player = [1] * num_envs      # Whose turn (1 or -1) per lane
    lane_episode_rewards_p1 = [0.0] * num_envs   # Cumulative reward per lane (P1)
    lane_episode_rewards_p2 = [0.0] * num_envs   # Cumulative reward per lane (P2 for visualization)
    lane_step_counts = [0] * num_envs         # Steps in current game per lane
    lane_opponent_types = ["self"] * num_envs # Opponent type per lane
    lane_opponent_uses_history = [True] * num_envs  # Whether opponent uses AAREN history per lane
    lane_current_opponents = [agent2] * num_envs  # Current opponent per lane
    
    # Track losses per episode for proper plotting
    episode_losses = []  # Accumulates losses between completed episodes
    
    # P1 Pending Transitions (for training on opponent moves)
    lane_pending_transitions = [None] * num_envs
    
    # P2 Pending Transitions (for training on opponent moves)
    lane_pending_transitions_p2 = [None] * num_envs
    
    # Unified reward shapers per lane
    lane_dist_rewards = [create_unified_reward_shaper(player_id=1, config=master_reward_config) for _ in range(num_envs)]
    lane_p2_shapers = [create_unified_reward_shaper(player_id=-1, config=master_reward_config) for _ in range(num_envs)]
    for dr, p2r in zip(lane_dist_rewards, lane_p2_shapers):
        dr.reset()
        p2r.reset()
        if curriculum and CURRICULUM_ENABLED:
            dr.set_phase(curriculum.current_phase.value)
            p2r.set_phase(curriculum.current_phase.value)
    
    # Observability Rate for curriculum (Transitions/Fog)
    obs_rate = 1.0
    if curriculum and CURRICULUM_ENABLED:
        obs_rate = curriculum.get_observability_rate()
    
    # Set Environment Observability (handles float for mixed/fog mode)
    parallel_env.set_full_observability(obs_rate)
    
    # Agent Logic: ALWAYS use AAREN embeddings (partial observability mode)
    # Agent learns with AAREN memory from day one, not just later phases
    # This lets the agent get accustomed to interpreting AAREN embeddings early
    use_full_obs = False  # Always use AAREN embeddings
    
    # Print AAREN status
    from training_config import HISTORY_EMBEDDING_SIZE
    print(f"[INFO] AAREN History Aggregator: ENABLED ({HISTORY_EMBEDDING_SIZE}-dim embeddings per position)")
    print(f"   Input channels: 15 (board) + {HISTORY_EMBEDDING_SIZE} (AAREN) = {15 + HISTORY_EMBEDDING_SIZE}")
    
    # AAREN optimization: track steps for interval-based updates
    from training_config import PBS_UPDATE_INTERVAL as HISTORY_UPDATE_INTERVAL
    
    # Helper function to generate placements and reset a single lane

    def reset_lane(lane_idx):
        """
        Generate new placements and prepare reset command for a lane.
        Now selects opponent FIRST to determine correct setup type.
        """
        # 1. Select Opponent for this episode
        opp_type, opp_uses_history, opp = select_opponent_for_lane(lane_idx)
        
        # 2. Setup Agent 1 (Always Heuristic/Smart)
        p1_pieces = parallel_env.call_method('_generate_pieces')
        p1_pos = parallel_env.call_method('get_valid_placement_positions', 1)
        p1_place = setup_agent1.place_pieces(p1_pieces, p1_pos)
        
        # 3. Setup Agent 2 (Evaluated based on opponent type)
        p2_pieces = parallel_env.call_method('_generate_pieces')
        p2_pos = parallel_env.call_method('get_valid_placement_positions', -1)
        
        # Always use Heuristic Setup for P2, even if opponent is RandomAgent
        p2_place = setup_agent2.place_pieces(p2_pieces, p2_pos)
        
        # Random starting player (50% chance for each player to move first)
        # This balances first-mover advantage during training
        starting_player = get_random_starting_player()
        
        return {
            'p1_placement': p1_place, 
            'p2_placement': p2_place,
            'opp_type': opp_type,
            'opp_uses_history': opp_uses_history,
            'opp': opp,
            'starting_player': starting_player
        }
    
    # Cache to avoid redundant disk loads when league selects the same opponent
    _last_league_path = [None]  # Mutable container for nonlocal access
    
    # Epoch-Based Opponent Cycling state
    from training_config import OPPONENT_CYCLE_INTERVAL
    _opponent_cycle_types = [None]  # Sorted list of opponent types for deterministic cycling
    _opponent_cycle_epoch = [-1]    # Current epoch index (tracks when to log transitions)
    
    # Lagged self-play cache: stores path of last checkpoint used as "self" opponent.
    # Updated at epoch boundaries to periodically refresh the lag target.
    _self_lag_path = [None]         # Mutable container for nonlocal access
    
    def _get_epoch_opponent_type(episode_count):
        """
        Determine the opponent type for the current epoch block.
        All lanes use the same type for OPPONENT_CYCLE_INTERVAL episodes.
        """
        if not (curriculum and CURRICULUM_ENABLED):
            return None  # Fall back to legacy random selection
        
        opponent_dist = curriculum.get_opponent_distribution()
        types = sorted(opponent_dist.keys())  # Deterministic order
        
        if not types:
            return "self"
        
        # Cache the type list for logging
        _opponent_cycle_types[0] = types
        
        # Determine epoch index and cycle through types
        epoch_idx = episode_count // OPPONENT_CYCLE_INTERVAL
        type_idx = epoch_idx % len(types)
        selected_type = types[type_idx]
        
        # Log on epoch transitions
        if epoch_idx != _opponent_cycle_epoch[0]:
            _opponent_cycle_epoch[0] = epoch_idx
            remaining = OPPONENT_CYCLE_INTERVAL - (episode_count % OPPONENT_CYCLE_INTERVAL)
            print(f"[CYCLE] Opponent epoch {epoch_idx}: '{selected_type}' for next {remaining} episodes (types: {types})")
            # Invalidate lagged self-play path so a fresh checkpoint is selected each epoch
            _self_lag_path[0] = None
        
        return selected_type
    
    def select_opponent_for_lane(lane_idx):
        """Select opponent type for a lane based on epoch-based cycling."""
        
        opponent_type = "self"
        opponent_uses_history = True
        current_opponent = agent2
        
        if curriculum and CURRICULUM_ENABLED:
            # Epoch-based cycling: deterministic opponent type per block
            opponent_type = _get_epoch_opponent_type(completed_episodes)
            if opponent_type is None:
                opponent_type = "self"  # Fallback
            
            # Configure opponent based on type
            if opponent_type == "true_random":
                current_opponent = true_random_agent
                opponent_uses_history = False
            elif opponent_type == "random":
                current_opponent = random_agent
                opponent_uses_history = False
            elif opponent_type in ["heuristic", "frozen_heuristic"]:
                current_opponent = heuristic_agent
                opponent_uses_history = False
            elif opponent_type == "smart_heuristic":
                current_opponent = smart_heuristic_agent
                opponent_uses_history = False
            elif opponent_type == "greedy":
                current_opponent = greedy_agent
                opponent_uses_history = False
            elif opponent_type == "league":
                path = league_manager.get_opponent()
                if path:
                    # Only reload if a different checkpoint was selected
                    if path != _last_league_path[0]:
                        agent2.load_model(path, load_optimizer=False)
                        _last_league_path[0] = path
                    current_opponent = agent2
                    opponent_uses_history = True
                else:
                    opponent_type = "self"
                    agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                    agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                    _last_league_path[0] = None  # Invalidate cache
                    current_opponent = agent2
                    opponent_uses_history = True
            elif opponent_type == "exploiters":
                current_opponent = get_random_exploiter(device, player_id=-1)
                opponent_uses_history = False
            else:  # self_500, self — use LAGGED league checkpoint to break Nash equilibrium
                # FIX: Instead of copying live agent1 weights (which causes symmetric
                # co-adaptation where both agents improve at the same rate and cancel
                # each other out), load a randomly selected historical checkpoint.
                # This gives agent1 an asymmetric, non-co-adapting target to exploit.
                lag_path = league_manager.get_opponent()  # Random historical checkpoint
                if lag_path:
                    # Only reload if epoch changed (avoid redundant disk I/O mid-epoch)
                    if lag_path != _self_lag_path[0]:
                        agent2.load_model(lag_path, load_optimizer=False)
                        _self_lag_path[0] = lag_path
                        _last_league_path[0] = None  # Invalidate league cache (different file)
                        print(f"[SELF-LAG] Loaded lagged opponent: {os.path.basename(lag_path)}")
                else:
                    # No league checkpoints yet — fall back to live copy (early training only)
                    agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                    agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                    _self_lag_path[0] = None
                    _last_league_path[0] = None  # Invalidate cache
                current_opponent = agent2
                opponent_uses_history = True
        else:
            # Legacy mode: use opponent pool
            opponent_type, opponent_data = opponent_pool.select_opponent()
            if opponent_type == "league":
                if opponent_data != _last_league_path[0]:
                    agent2.load_model(opponent_data, load_optimizer=False)
                    _last_league_path[0] = opponent_data
                current_opponent = agent2
                opponent_uses_history = True
            elif opponent_type == "random":
                current_opponent = random_agent
                opponent_uses_history = False
            elif opponent_type == "greedy":
                current_opponent = greedy_agent
                opponent_uses_history = False
            else:
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                _last_league_path[0] = None  # Invalidate cache
                current_opponent = agent2
                opponent_uses_history = True
        
        return opponent_type, opponent_uses_history, current_opponent
    
    # Initial reset for all lanes
    # print(f"Initializing {num_envs} lanes...") # Duplicate print
    initial_placements_p1 = []
    initial_placements_p2 = []
    for i in range(num_envs):
        reset_data = reset_lane(i)
        initial_placements_p1.append(reset_data['p1_placement'])
        initial_placements_p2.append(reset_data['p2_placement'])
        
        # Select opponent for each lane (Now comes from reset_lane)
        lane_opponent_types[i] = reset_data['opp_type']
        lane_opponent_uses_history[i] = reset_data['opp_uses_history']
        lane_current_opponents[i] = reset_data['opp']
    
    # Reset all environments
    game_states, _, _, _, valid_moves = parallel_env.reset(initial_placements_p1, initial_placements_p2)
    # Track starting players for initial setup
    initial_starting_players = []
    for i in range(num_envs):
        reset_data = reset_lane(i)
        initial_starting_players.append(reset_data.get('starting_player', 1))
        initial_placements_p1[i] = reset_data['p1_placement']
        initial_placements_p2[i] = reset_data['p2_placement']
        lane_opponent_types[i] = reset_data['opp_type']
        lane_opponent_uses_history[i] = reset_data['opp_uses_history']
        lane_current_opponents[i] = reset_data['opp']
    
    # Reset environments with placements
    game_states, _, _, _, valid_moves = parallel_env.reset(initial_placements_p1, initial_placements_p2)
    for i in range(num_envs):
        lane_game_states[i] = game_states[i]
        lane_valid_moves[i] = valid_moves[i]
        lane_current_player[i] = initial_starting_players[i]  # Random starting player
    
    # Loss tracking arrays
    lane_episode_loss_sum_p1 = np.zeros(num_envs)
    lane_episode_loss_count_p1 = np.zeros(num_envs, dtype=int)
    
    # Reset AAREN history for all lanes
    agent1.reset_history()
    agent2.reset_history()
    
    # Initialize episode tracking for episode replay buffer
    if hasattr(agent1, 'episode_replay_enabled') and agent1.episode_replay_enabled and agent1.episode_memory is not None:
        for i in range(num_envs):
            agent1.episode_memory.start_episode(i)
    if hasattr(agent2, 'episode_replay_enabled') and agent2.episode_replay_enabled and agent2.episode_memory is not None:
        for i in range(num_envs):
            agent2.episode_memory.start_episode(i)
    
    # State tracking for intervals
    last_replay_step = 0
    last_target_update_step = 0
    
    # Progress bar - updates based on completed episodes
    pbar = tqdm(total=num_episodes, initial=completed_episodes, desc="Training Episodes", dynamic_ncols=True)
    
    # ==========================================================================
    # MAIN TRAINING LOOP
    # All lanes run in parallel, each at their own pace
    # ==========================================================================
    
    # Import MARL target update interval
    from training_config import MARL_TARGET_UPDATE_INTERVAL
    
    try:
        while completed_episodes < num_episodes:
            # Prepare batch actions based on whose turn it is in each lane
            actions = [None] * num_envs
            p1_acting_mask = np.zeros(num_envs, dtype=bool)  # Track which lanes have P1 acting
            
            # --- DETERMINE TEMPERATURE FOR OPPONENT EXPLORATION ---
            # Introduce Boltzmann exploration for Agent 2 during Phase 3 and 4
            # to prevent Agent 1 from memorizing a deterministic opponent
            current_temp = 0.0
            if curriculum and CURRICULUM_ENABLED:
                if curriculum.current_phase.value == 3:
                    current_temp = 0.1
                elif curriculum.current_phase.value >= 4:
                    current_temp = 0.2
            
            # --- BATCHED ACTION SELECTION ---
            # Collect states for P1 and P2 separately for batched inference
            p1_indices = [i for i in range(num_envs) if lane_current_player[i] == 1]
            p2_indices = [i for i in range(num_envs) if lane_current_player[i] == -1]
            
            import time
            profiling = {"p1_act": 0.0, "p2_act": 0.0, "env_step": 0.0, "replay": 0.0}
            
            # P1 batch action
            t0 = time.perf_counter()
            if p1_indices:
                p1_states = [lane_game_states[i].board for i in p1_indices]
                p1_valid_moves = [lane_valid_moves[i] for i in p1_indices]
                p1_game_states = [lane_game_states[i] for i in p1_indices]
                
                p1_actions = agent1.act_batch(
                    p1_states, p1_valid_moves, p1_game_states,
                    env_indices=p1_indices, full_observability=use_full_obs
                )
                
                for idx, lane_i in enumerate(p1_indices):
                    actions[lane_i] = p1_actions[idx]
                    p1_acting_mask[lane_i] = True
            profiling["p1_act"] = time.perf_counter() - t0
            
            # P2 batch action
            t1 = time.perf_counter()
            if p2_indices:
                # Group P2 by opponent for batched inference
                opponent_groups = {}
                for i in p2_indices:
                    opp = lane_current_opponents[i]
                    opp_id = id(opp)
                    if opp_id not in opponent_groups:
                        opponent_groups[opp_id] = {'opponent': opp, 'indices': [], 'states': [], 'moves': [], 'game_states': []}
                    opponent_groups[opp_id]['indices'].append(i)
                    opponent_groups[opp_id]['states'].append(lane_game_states[i].board)
                    opponent_groups[opp_id]['moves'].append(lane_valid_moves[i])
                    opponent_groups[opp_id]['game_states'].append(lane_game_states[i])
                
                for opp_id, group in opponent_groups.items():
                    opp = group['opponent']
                    # Use temperature when opponents act_batch supports it.
                    # drqn_agent.py act_batch supports temperature. Other opponents ignore kwargs.
                    p2_actions = opp.act_batch(
                        group['states'], group['moves'], group['game_states'],
                        env_indices=group['indices'], full_observability=use_full_obs,
                        temperature=current_temp
                    )
                    for idx, lane_i in enumerate(group['indices']):
                        actions[lane_i] = p2_actions[idx]
            profiling["p2_act"] = time.perf_counter() - t1
            
            # --- STEP ALL ENVIRONMENTS ---
            t2 = time.perf_counter()
            next_states, rewards, dones, infos, next_valid_moves = parallel_env.step(actions)
            profiling["env_step"] = time.perf_counter() - t2
            
            # --- COUNT STEPS (Agent 1 only) ---
            p1_step_count = len(p1_indices)
            global_step += p1_step_count
            
            # --- PROCESS RESULTS PER LANE ---
            reset_commands = {}  # Lane index -> reset placements (for lanes that finished)
            
            # Batch lists for Agent 1 memory
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_game_states = []
            batch_next_game_states = []
            batch_infos = []  # Track infos for battle detection in PER
            batch_active_mask = [] # Fix missing initialization for Agent 1

            # Batch lists for Agent 2 memory
            batch_states_p2 = []
            batch_actions_p2 = []
            batch_rewards_p2 = []
            batch_next_states_p2 = []
            batch_dones_p2 = []
            batch_active_mask_p2 = []
            batch_game_states_p2 = []
            batch_next_game_states_p2 = []
            batch_infos_p2 = []

            for i in range(num_envs):
                lane_step_counts[i] += 1
                current_player = lane_current_player[i]
                
                # --- AGENT 1 (P1) LOGIC ---
                if current_player == 1:
                    # Agent 1 Just Acted - Unified Reward Shaper handles all components
                    done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                    reward = lane_dist_rewards[i](
                        previous_state=lane_game_states[i],
                        action=actions[i],
                        current_state=next_states[i],
                        done=done_bool,
                        winner=infos[i].get('winner'),
                        info=infos[i]
                    )
                    
                    lane_episode_rewards_p1[i] += reward
                    
                    if done_bool:
                        # Game ended on P1's turn
                        batch_states.append(lane_game_states[i].board)
                        batch_actions.append(actions[i])
                        batch_rewards.append(reward)
                        batch_next_states.append(next_states[i].board)
                        batch_dones.append(True)
                        batch_active_mask.append(True)
                        batch_game_states.append(lane_game_states[i])
                        batch_next_game_states.append(next_states[i])
                        batch_infos.append(infos[i])  # For PER battle detection
                        
                        lane_pending_transitions[i] = None # Clear
                    else:
                        # Game continues -> P2's turn
                        # Store as PENDING. Wait for P2's response.
                        lane_pending_transitions[i] = {
                            'state': lane_game_states[i].board,
                            'action': actions[i],
                            'reward': reward,
                            'game_state': lane_game_states[i]
                        }
                
                # --- AGENT 2 (P2) LOGIC ---
                else:
                    # Agent 2 Just Acted
                    done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                    
                    # Track reward for P2 visualization (using the same Unified Shaper)
                    p2_reward = lane_p2_shapers[i](
                        previous_state=lane_game_states[i],
                        action=actions[i],
                        current_state=next_states[i],
                        done=done_bool,
                        winner=infos[i].get('winner'),
                        info=infos[i]
                    )
                    lane_episode_rewards_p2[i] += p2_reward 
                    
                    # Check for P1 Pending Transition
                    if lane_pending_transitions[i]:
                        pending = lane_pending_transitions[i]
                        
                        # FIX #5: P1's reward is ONLY from P1's own action.
                        # No double-counting — we do NOT re-evaluate the shaper on P2's action.
                        # The TD target will naturally capture P2's impact via the next_state
                        # (which is the board state AFTER P2 acted).
                        
                        # Complete the transition
                        done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                        
                        batch_states.append(pending['state'])
                        batch_actions.append(pending['action'])
                        batch_rewards.append(pending['reward'])
                        batch_next_states.append(next_states[i].board)
                        batch_dones.append(done_bool)
                        batch_active_mask.append(True)
                        batch_game_states.append(pending['game_state'])
                        batch_next_game_states.append(next_states[i])
                        batch_infos.append(infos[i])  # For PER battle detection
                        
                        lane_pending_transitions[i] = None # Clear pending
                    
                    if done_bool:
                        # Game ended on P2's turn
                        if lane_current_opponents[i] is agent2:
                            batch_states_p2.append(lane_game_states[i].board)
                            batch_actions_p2.append(actions[i])
                            batch_rewards_p2.append(p2_reward)
                            batch_next_states_p2.append(next_states[i].board)
                            batch_dones_p2.append(True)
                            batch_active_mask_p2.append(True)
                            batch_game_states_p2.append(lane_game_states[i])
                            batch_next_game_states_p2.append(next_states[i])
                            batch_infos_p2.append(infos[i])
                        
                        lane_pending_transitions_p2[i] = None # Clear
                    else:
                        # Game continues -> P1's turn
                        # Store as PENDING. Wait for P1's response.
                        if lane_current_opponents[i] is agent2:
                            lane_pending_transitions_p2[i] = {
                                'state': lane_game_states[i].board,
                                'action': actions[i],
                                'reward': p2_reward,
                                'game_state': lane_game_states[i]
                            }
                
                # Check for P2 Pending Transition if P1 just acted
                if current_player == 1 and lane_pending_transitions_p2[i]:
                    pending_p2 = lane_pending_transitions_p2[i]
                    done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                    
                    if lane_current_opponents[i] is agent2:
                        batch_states_p2.append(pending_p2['state'])
                        batch_actions_p2.append(pending_p2['action'])
                        batch_rewards_p2.append(pending_p2['reward'])
                        batch_next_states_p2.append(next_states[i].board)
                        batch_dones_p2.append(done_bool)
                        batch_active_mask_p2.append(True)
                        batch_game_states_p2.append(pending_p2['game_state'])
                        batch_next_game_states_p2.append(next_states[i])
                        batch_infos_p2.append(infos[i])
                    
                    lane_pending_transitions_p2[i] = None # Clear pending
                        
                # Update AAREN history
                done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                if not done_bool:
                    if current_player == 1 and lane_opponent_uses_history[i]:
                        agent2.update_history_batch([actions[i]], [lane_game_states[i]], acting_player=1)
                    elif current_player == -1 and not use_full_obs:
                        agent1.update_history_batch([actions[i]], [lane_game_states[i]], acting_player=-1)
                
                # Feed reveal data to AAREN for supervised training
                # When battles occur, the environment returns revealed piece types
                if infos[i].get('revealed_in_step'):
                    from training_config import AAREN_USE_REVEAL_DATA
                    for pos, piece_type in infos[i]['revealed_in_step']:
                        if agent1.history_instances and i < len(agent1.history_instances):
                            game_phase = "early" if lane_step_counts[i] < 50 else ("mid" if lane_step_counts[i] < 200 else "end")
                            agent1.history_instances[i].update_from_reveal(
                                pos, piece_type, game_phase=game_phase, turn_count=lane_step_counts[i],
                                collect_training_data=AAREN_USE_REVEAL_DATA
                            )
                        if agent2.history_instances and i < len(agent2.history_instances) and lane_opponent_uses_history[i]:
                            game_phase = "early" if lane_step_counts[i] < 50 else ("mid" if lane_step_counts[i] < 200 else "end")
                            agent2.history_instances[i].update_from_reveal(
                                pos, piece_type, game_phase=game_phase, turn_count=lane_step_counts[i],
                                collect_training_data=AAREN_USE_REVEAL_DATA
                            )
                
                # Check for Game End
                if done_bool:
                    winner = infos[i].get('winner', 0)
                    
                    # Calculate average loss for this specific episode
                    if lane_episode_loss_count_p1[i] > 0:
                        avg_loss = lane_episode_loss_sum_p1[i] / lane_episode_loss_count_p1[i]
                    else:
                        avg_loss = 0.0 # No training happened during this episode (e.g. very short)
                    
                    metrics['avg_loss_p1_history'].append(avg_loss)
                    
                    # Metrics
                    if winner == 1:
                        metrics['wins_p1'] += 1
                        # Track win type
                        win_type = infos[i].get('win_type', 'unknown')
                        if win_type == 'flag_capture':
                            metrics['wins_by_flag'] = metrics.get('wins_by_flag', 0) + 1
                        elif win_type == 'no_moves':
                            metrics['wins_by_depletion'] = metrics.get('wins_by_depletion', 0) + 1
                    elif winner == -1:
                        metrics['wins_p2'] += 1
                        # Track loss type (how P1 lost = how P2 won)
                        win_type = infos[i].get('win_type', 'unknown')
                        if win_type == 'flag_capture':
                            metrics['losses_by_flag'] = metrics.get('losses_by_flag', 0) + 1
                        elif win_type == 'no_moves':
                            metrics['losses_by_depletion'] = metrics.get('losses_by_depletion', 0) + 1
                    else:
                        metrics['draws'] += 1
                        # Track draw type
                        win_type = infos[i].get('win_type', 'timeout')
                        metrics['draws_by_timeout'] = metrics.get('draws_by_timeout', 0) + 1
                    
                    metrics['rewards_p1'].append(lane_episode_rewards_p1[i])
                    metrics['rewards_p2'].append(lane_episode_rewards_p2[i])
                    metrics['lengths'].append(lane_step_counts[i])
                    
                    # Removed old sliding window loss code
                    
                    metrics['wins_p1_history'].append(metrics['wins_p1'])
                    metrics['wins_p2_history'].append(metrics['wins_p2'])
                    metrics['wins_by_flag_history'].append(metrics.get('wins_by_flag', 0))
                    metrics['wins_by_depletion_history'].append(metrics.get('wins_by_depletion', 0))
                    metrics['losses_by_flag_history'].append(metrics.get('losses_by_flag', 0))
                    metrics['losses_by_depletion_history'].append(metrics.get('losses_by_depletion', 0))
                    
                    if curriculum and CURRICULUM_ENABLED:
                        metrics['phase_history'].append(curriculum.current_phase.value)
                    else:
                        metrics['phase_history'].append(1)
                    
                    # Piece tracker
                    if piece_tracker is not None:
                        try:
                            board = lane_game_states[i].board if hasattr(lane_game_states[i], 'board') else lane_game_states[i]
                            surviving_p1 = {}
                            surviving_p2 = {}
                            for r in range(10):
                                for c in range(10):
                                    val = board[r, c].item() if hasattr(board[r, c], 'item') else board[r, c]
                                    if val > 0 and val <= 11:
                                        surviving_p1[val] = surviving_p1.get(val, 0) + 1
                                    elif val < 0 and val >= -11:
                                        surviving_p2[abs(val)] = surviving_p2.get(abs(val), 0) + 1
                            piece_tracker.record_game_end(winner, surviving_p1, surviving_p2)
                        except Exception:
                            pass
                    
                    # Update Curriculum
                    if curriculum and CURRICULUM_ENABLED:
                        curriculum.update_metrics({
                            'winner': winner,
                            'opponent_type': lane_opponent_types[i],
                            'pbs_accuracy': agent1.history.avg_accuracy if agent1.history and hasattr(agent1.history, 'avg_accuracy') else 0.0
                        })
                    
                    # --- GRANULAR METRICS (Per Episode) ---
                    # PBS evaluator metrics removed — AAREN replaced PBS evaluator
                    metrics['pbs_eval1_losses'].append(0.0)
                    metrics['pbs_eval1_buffer_sizes'].append(0)
                    metrics['pbs_eval1_accuracy'].append(0.0)
                    
                    # AAREN metrics (legacy + end-to-end)
                    if agent1.history:
                        # Legacy metric keys kept for backward compatibility
                        metrics['aaren_loss'].append(agent1.history.get_avg_loss() if hasattr(agent1.history, 'get_avg_loss') else 0.0)
                        metrics['aaren_accuracy'].append(agent1.history.get_accuracy() if hasattr(agent1.history, 'get_accuracy') else 0.0)
                        metrics['aaren_buffer_size'].append(agent1.history.get_buffer_size() if hasattr(agent1.history, 'get_buffer_size') else 0)
                        
                        # End-to-end training metrics (gradient norms captured during replay)
                        aaren_grad = agent1.get_aaren_grad_norm() if hasattr(agent1, 'get_aaren_grad_norm') else 0.0
                        dqn_grad = agent1.get_dqn_grad_norm() if hasattr(agent1, 'get_dqn_grad_norm') else 0.0
                        embedding_stats = agent1.get_aaren_embedding_stats() if hasattr(agent1, 'get_aaren_embedding_stats') else {'std': 0.0}
                        
                        metrics['aaren_grad_norm'] = metrics.get('aaren_grad_norm', [])
                        metrics['aaren_grad_norm'].append(aaren_grad)
                        metrics['dqn_grad_norm'] = metrics.get('dqn_grad_norm', [])
                        metrics['dqn_grad_norm'].append(dqn_grad)
                        metrics['aaren_embedding_std'] = metrics.get('aaren_embedding_std', [])
                        metrics['aaren_embedding_std'].append(embedding_stats.get('std', 0.0))
                    
                    # Q-value and entropy metrics
                    avg_q = agent1.get_average_q() if hasattr(agent1, 'get_average_q') else 0.0
                    if avg_q == 0.0 and hasattr(agent1, 'memory') and len(agent1.memory) > agent1.batch_size:
                         tqdm.write(f"[WARN] Avg Q-Value is 0.0. Memory: {len(agent1.memory)}")
                    metrics['avg_q_values_p1'].append(avg_q)
                    
                    
                    metrics['avg_entropy_p1'].append(agent1.get_exploration_entropy() if hasattr(agent1, 'get_exploration_entropy') else 0.0)
                    
                    # Track exact step when episode ended (for plotting alignment)
                    metrics.get('episode_end_steps', []).append(global_step)
                    
                    completed_episodes += 1
                    pbar.update(1)
                    
                    # --- PBT METRICS REPORTING ---
                    # Report metrics to supervisor for PBT exploitation decisions
                    if pbt_reporter is not None:
                        # Calculate recent win rate
                        recent_wins = sum(1 for r in metrics['rewards_p1'][-100:] if r > 0)
                        recent_win_rate = recent_wins / min(100, len(metrics['rewards_p1'])) if metrics['rewards_p1'] else 0.0
                        
                        # Get recent average loss
                        recent_loss = np.mean(metrics['losses_p1'][-100:]) if metrics['losses_p1'] else 0.0
                        
                        pbt_reporter.report(
                            episode=completed_episodes,
                            reward=lane_episode_rewards_p1[i],
                            win=1 if winner == 1 else 0,
                            win_rate=recent_win_rate,
                            avg_loss=recent_loss
                        )
                    
                    reset_data = reset_lane(i)
                    reset_commands[i] = {'p1_placement': reset_data['p1_placement'], 
                                       'p2_placement': reset_data['p2_placement']}
                    
                    # Update Opponent
                    lane_opponent_types[i] = reset_data['opp_type']
                    lane_opponent_uses_history[i] = reset_data['opp_uses_history']
                    lane_current_opponents[i] = reset_data['opp']
                    
                    # Reset Lane State
                    lane_episode_rewards_p1[i] = 0.0
                    lane_episode_rewards_p2[i] = 0.0
                    lane_episode_loss_sum_p1[i] = 0.0 # Reset loss sum
                    lane_episode_loss_count_p1[i] = 0 # Reset loss count
                    lane_step_counts[i] = 0
                    lane_current_player[i] = reset_data.get('starting_player', 1)  # Random starting player
                    lane_dist_rewards[i].reset()
                    lane_p2_shapers[i].reset()
                    lane_pending_transitions[i] = None
                    lane_pending_transitions_p2[i] = None
                    
                    if agent1.history_instances and i < len(agent1.history_instances):
                        agent1.history_instances[i].reset()
                    if lane_opponent_uses_history[i] and agent2.history_instances and i < len(agent2.history_instances):
                        agent2.history_instances[i].reset()
                    
                    # Start new episode tracking for episode replay buffer
                    if hasattr(agent1, 'episode_replay_enabled') and agent1.episode_replay_enabled and agent1.episode_memory is not None:
                        agent1.episode_memory.start_episode(i)
                    if hasattr(agent2, 'episode_replay_enabled') and agent2.episode_replay_enabled and agent2.episode_memory is not None:
                        agent2.episode_memory.start_episode(i)
                else:
                    # Continue Game
                    lane_current_player[i] *= -1
                    lane_game_states[i] = next_states[i]
                    lane_valid_moves[i] = next_valid_moves[i]
                    lane_game_states[i] = next_states[i]
                    lane_valid_moves[i] = next_valid_moves[i]
            
            # --- GRANULAR METRIC COLLECTION (Every Episode) ---
            # Record these if Agent 1 just finished a game (passed via done_bool check above)
            # Note: We collect these as global averages from the agent *at this moment*.
            # Since we have multiple lanes, we can record this whenever *any* lane finishes, 
            # or just once per completed_episodes increment.
            
            # To match "every episode point plot", we record whenever `completed_episodes` increments.
            # This logic is best placed inside the "if done_bool" block above, but that block runs for EACH lane.
            # We need to be careful not to record 8 times for 8 lanes finishing simultaneously.
            # Ideally, we append to `metrics` lists exactly when we increment `completed_episodes`.
            if batch_states:
                 agent1.remember_batch(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                    batch_active_mask,
                    batch_game_states,
                    batch_next_game_states,
                    infos=batch_infos  # Pass infos for PER battle detection
                )
            
            if batch_states_p2:
                agent2.remember_batch(
                    batch_states_p2,
                    batch_actions_p2,
                    batch_rewards_p2,
                    batch_next_states_p2,
                    batch_dones_p2,
                    batch_active_mask_p2,
                    batch_game_states_p2,
                    batch_next_game_states_p2,
                    infos=batch_infos_p2
                )
            
            # --- APPLY RESETS ---
            if reset_commands:
                for lane_i, placements in reset_commands.items():
                    parallel_env.remotes[lane_i].send(('reset', placements))
                
                for lane_i in reset_commands.keys():
                    result = parallel_env.remotes[lane_i].recv()
                    new_state, _, _, _, new_valid_moves = result
                    lane_game_states[lane_i] = new_state
                    lane_valid_moves[lane_i] = new_valid_moves
                    lane_current_player[lane_i] = 1
            
            # --- TRAINING STEP ---
            t3 = time.perf_counter()
            # Use threshold check instead of modulo to avoid skipping updates when step_count > 1
            if global_step - last_replay_step >= REPLAY_UPDATE_INTERVAL:
                loss1 = agent1.replay(episode=completed_episodes)
                last_replay_step = global_step
                
                # Active Agent 2 MARL learning only in Phase 4+ (League Training) 
                # Optimization: We offset Agent 2's training step by half the interval
                # so that both networks aren't doing heavy backprop in the same exact loop iteration.
                
                if loss1:
                    metrics['losses_p1'].append(loss1)
                    metrics['loss_steps_p1'].append(global_step)  # Track step for proper plotting
                    
            if curriculum and curriculum.current_phase.value >= 4:
                # Agent 2 updates slightly offset from Agent 1 to smooth CPU/GPU load spikes
                if global_step - getattr(agent2, 'last_replay_step_offset', 0) >= REPLAY_UPDATE_INTERVAL:
                    # Initialize offset tracker on first run
                    if not hasattr(agent2, 'last_replay_step_offset') or agent2.last_replay_step_offset == 0:
                        agent2.last_replay_step_offset = global_step + (REPLAY_UPDATE_INTERVAL // 2)
                        loss2 = None
                    else:
                        loss2 = agent2.replay(episode=completed_episodes)
                        agent2.last_replay_step_offset = global_step
                        
                        if loss2 and 'losses_p2' not in metrics:
                            metrics['losses_p2'] = []
                            metrics['loss_steps_p2'] = []
                        if loss2:
                            metrics['losses_p2'].append(loss2)
                            metrics['loss_steps_p2'].append(global_step)
            
            if global_step - last_replay_step == 0 and 'loss1' in locals() and loss1:
                # Accumulate loss for all active lanes (since replay updates are global but apply to active policy)
                # We interpret "episode loss" as the average loss *during the lifetime of that episode*.
                # Since training happens globally every N steps, we add this loss to all currently active episodes.
                for i in range(num_envs):
                    if not dones[i]: # Only if episode is active
                        lane_episode_loss_sum_p1[i] += loss1
                        lane_episode_loss_count_p1[i] += 1
            
            # --- PERIODIC CUDA CLEANUP ---
            profiling["replay"] = time.perf_counter() - t3
            # Removed aggressive empty_cache as it causes allocator sync and slowdown.
            
            # --- TARGET NETWORK UPDATE ---
            # Use threshold check to prevent double updates or skips
            # Apply MARL target update smoothing in Phase 4 (League Training) 
            current_target_interval = TARGET_UPDATE_INTERVAL
            if curriculum and curriculum.current_phase.value >= 4:
                current_target_interval = MARL_TARGET_UPDATE_INTERVAL

            if global_step - last_target_update_step >= current_target_interval:
                agent1.update_target_network()
                if curriculum and curriculum.current_phase.value >= 4:
                    agent2.update_target_network()
                    tqdm.write(f"[INFO] Target Networks (P1 & P2) Updated (Step {global_step} - {current_target_interval} interval)")
                else:
                    tqdm.write(f"[INFO] Target Network (P1 Only) Updated (Step {global_step})")
                last_target_update_step = global_step
            
            # --- CHECK CURRICULUM PHASE TRANSITION ---
            if curriculum and CURRICULUM_ENABLED and completed_episodes % 50 == 0:
                if curriculum.check_phase_transition():
                    old_phase = curriculum.current_phase
                    if curriculum.advance_phase():
                        tqdm.write(f"\n[PHASE] TRANSITION: {old_phase.name} -> {curriculum.current_phase.name}")
                        
                        # Update observability
                        use_full_obs = curriculum.should_use_full_observability()
                        parallel_env.set_full_observability(use_full_obs)
                        
                        # Update max turns for new phase
                        from training_config import PHASE_MAX_TURNS, DEFAULT_MAX_TURNS
                        new_max_turns = PHASE_MAX_TURNS.get(curriculum.current_phase.value, DEFAULT_MAX_TURNS)
                        parallel_env.set_max_turns(new_max_turns)
                        tqdm.write(f"[INFO] Max turns updated to {new_max_turns}")
                        
                        # Update reward shaping phases
                        for dr, p2r in zip(lane_dist_rewards, lane_p2_shapers):
                            dr.set_phase(curriculum.current_phase.value)
                            p2r.set_phase(curriculum.current_phase.value)
                        
                        # Enable Agent 2 AAREN history at Phase 4
                        if curriculum.current_phase.value >= 4 and not agent2.use_pbs:
                            agent2.enable_history(num_envs)
                            tqdm.write("[INFO] Agent 2 AAREN history ENABLED for Phase 4+")
            
            # --- LR SCHEDULER STEP (per fixed step interval) ---
            if agent1.scheduler and global_step % 1000 == 0:
                agent1.scheduler.step()
            if hasattr(agent2, 'scheduler') and agent2.scheduler and global_step % 1000 == 0:
                if curriculum and curriculum.current_phase.value >= 4:
                    agent2.scheduler.step()
            
            # Train AAREN supervised model periodically (lightweight cross-entropy on reveal data)
            if global_step % (REPLAY_UPDATE_INTERVAL * 10) == 0:
                aaren_loss_1 = agent1.train_history(epochs=1)
                # Optimization: Skip supervised AAREN training for Agent 2 in League Training. 
                # MARL agent 2 learns end-to-end; supervised reveal training is too computationally expensive here.
                # if agent2.use_pbs:
                #     aaren_loss_2 = agent2.train_history(epochs=1)
                
                # if aaren_loss_1 is not None:
                #     tqdm.write(f"[AAREN] Supervised loss: {aaren_loss_1:.4f}")
            
            # --- PERIODIC PLOTTING (every plot_interval episodes) ---
            if metrics_tracker.should_plot(completed_episodes, plot_interval):
                plot_episode = (completed_episodes // plot_interval) * plot_interval
                metrics_tracker.mark_plotted(plot_episode)
                
                # Update win rate
                metrics_tracker.update_win_rate()
                
                # Use Checkpointer for plotting
                try:
                    checkpointer.plot_progress(
                        episode=plot_episode,
                        metrics_tracker=metrics_tracker,
                        global_step=global_step,
                        num_envs=num_envs
                    )
                    tqdm.write(f"[INFO] Saved plots for episode {plot_episode}")
                except Exception as e:
                    tqdm.write(f"[WARN] Could not plot training progress: {e}")
            

            # --- PERIODIC MODEL SAVES (every SAVE_INTERVAL episodes) ---
            # Robust check: trigger if we passed a new multiple of save_interval
            if metrics_tracker.should_save(completed_episodes, save_interval, start_episode):
                save_episode = (completed_episodes // save_interval) * save_interval
                metrics_tracker.mark_saved(save_episode)
                
                metrics_tracker.set_global_step(global_step)

                # Use Checkpointer for save operations
                checkpointer.save_checkpoint(
                    episode=save_episode,
                    agent1=agent1,
                    agent2=agent2,
                    league_manager=league_manager,
                    curriculum=curriculum if CURRICULUM_ENABLED else None,
                    metrics_tracker=metrics_tracker,
                    league_interval=LEAGUE_SAVE_INTERVAL
                )
                
                tqdm.write(f"[SAVE] Models saved for episode {save_episode}")
                
                # Piece value tracking
                if piece_tracker is not None and save_episode % 500 == 0:
                    piece_tracker.log_comparison(save_episode)
                    piece_tracker.save()
                
                # --- MEMORY CLEANUP (Prevent OOM from fragmentation) ---
                # Run garbage collection and clear CUDA cache every save interval
                if device.type == 'cuda':
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Log memory stats periodically
                    if save_episode % 1000 == 0:
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        tqdm.write(f"[MEM] CUDA cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    
            # --- UPDATE PROGRESS BAR (only when episodes complete) ---
            if reset_commands:
                recent_reward = np.mean(metrics['rewards_p1'][-10:]) if metrics['rewards_p1'] else 0.0
                phase_val = curriculum.current_phase.value if curriculum and CURRICULUM_ENABLED else 1
                
                # Use fixed-width formatting to prevent flickering and ensure all metrics fit
                # W2 and Step are prioritized with padding to prevent terminal wrapping issues
                
                # Format a compact, unified postfix string with extra padding spaces
                # to overwrite any residual characters and prevent tqdm truncation
                postfix_str = (
                    f"R1={recent_reward:5.2f} "
                    f"W1={metrics['wins_p1']:<4d} "
                    f"W2={metrics['wins_p2']:<4d} "
                    f"P={phase_val:<1d} "
                    f"S={global_step/1000:<5.1f}k "
                    f"[P1:{profiling['p1_act']:.2f} P2:{profiling['p2_act']:.2f} Env:{profiling['env_step']:.2f} Rep:{profiling['replay']:.2f}]"
                )
                pbar.set_postfix_str(postfix_str, refresh=False)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user. Saving full state for seamless resumption...")
        stop_episode = completed_episodes
        metrics_tracker.set_global_step(global_step)
        archive_path = checkpointer.save_full_state(
            episode=stop_episode,
            agent1=agent1,
            agent2=agent2,
            curriculum=curriculum if CURRICULUM_ENABLED else None,
            metrics_tracker=metrics_tracker
        )
        print(f"[SUCCESS] Full training state (including buffers and AAREN) saved to: {archive_path}")
        print("[INFO] You can resume training by running the script again.")
    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}")
        traceback.print_exc()
    finally:
        # Crucial fix for multiprocessing connection crashes
        print("\n[INFO] Shutting down parallel environments...")
        parallel_env.close()
        pbar.close()

    print(f"\n[DONE] Training complete! Completed {completed_episodes} episodes, {global_step:,} steps")


if __name__ == "__main__":
    print("DQN Agent Training for Stratego")
    print("==================================================")
    print()
    
    # Resolve model path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "dqn_models")
    
    if run_preflight_checks(model_save_path=default_model_path):
        train_dqn_agents(
            num_episodes=NUM_EPISODES,
            save_interval=SAVE_INTERVAL,
            generate_gifs=GENERATE_GIFS
        )
    else:
        print("[ERROR] Pre-flight checks failed. Aborting training.")

