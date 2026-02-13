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

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import StrategoEnvironment
from parallel_environment import ParallelStrategoEnvironment
from drqn_agent import RainbowAgent
from heuristic_setup import HeuristicSetupAgent
from game_state import GameState
from training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_additional_metrics
from piece import PieceType, PIECE_RANKS
from board import LAKE_SQUARE

# Import reset function (optional)
try:
    from reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False

# Random starting player removed - was only swapping positions, not turn order
from league import LeagueManager
from opponents import RandomAgent, GreedyAgent, OpponentPool, RandomSetupAgent
from training_config import *
from training_utils import save_training_history, load_training_history
from preflight_checks import run_preflight_checks

# Curriculum and Reward Shaping
from curriculum import CurriculumManager, TrainingPhase, HeuristicOpponent, SmartHeuristicOpponent, TrueRandomOpponent
from exploiter_agents import get_random_exploiter, RusherAgent, TurtleAgent, FlankingAgent
from scenario_drills import get_scenario_drill, get_random_scenario

# Distributional RL-Compatible Reward Shaping (C51 Normalized Anti-Stall)
from distributional_reward import create_unified_reward_shaper, StrategoRewardConfig

# Extracted training modules (for future gradual migration)
from training import LaneManager, MetricsTracker, Checkpointer, get_random_starting_player

# Piece Value Tracking (for convergence analysis)
try:
    from piece_value_tracker import PieceValueTracker, ANALYTICAL_VALUES
    PIECE_VALUE_TRACKING = True
except ImportError:
    PIECE_VALUE_TRACKING = False
    print("[WARN] Piece value tracking disabled (module not found)")

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
    print("Initializing Rainbow Agents...")
    agent1 = RainbowAgent(player_id=1, device=device, lr=LEARNING_RATE, batch_size=batch_size, num_envs=num_envs, buffer_size=memory_size)
    # Agent2: minimal buffer, no AAREN history (saves ~18% training time in early phases)
    # AAREN history will be enabled for Agent 2 when reaching Phase 4
    agent2 = RainbowAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=batch_size, num_envs=num_envs, buffer_size=10000, use_pbs=False)
    print("[INFO] Agent 2 AAREN history disabled for early phases (will enable at Phase 4)")
    
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
        from training_config import PHASE_1_MAX_TURNS, PHASE_2_MAX_TURNS, PHASE_3_MAX_TURNS, PHASE_4_MAX_TURNS, DEFAULT_MAX_TURNS
        phase_max_turns = {1: PHASE_1_MAX_TURNS, 2: PHASE_2_MAX_TURNS, 3: PHASE_3_MAX_TURNS, 4: PHASE_4_MAX_TURNS}
        max_turns = phase_max_turns.get(curriculum.current_phase.value, DEFAULT_MAX_TURNS)
        parallel_env.set_max_turns(max_turns)
        print(f"   Max turns per game: {max_turns} (Phase {curriculum.current_phase.value})")
    
    # Reward shaper is initialized per lane later
    
    # Initialize Piece Value Tracker (for convergence analysis)
    piece_tracker = None
    if PIECE_VALUE_TRACKING:
        piece_tracker = PieceValueTracker(save_path=os.path.join(model_save_path, "piece_value_tracking.json"))
        print(f"[INFO] Piece value tracking enabled ({piece_tracker.games_tracked} games loaded)")
    
    
    # --- Load Existing Models ---
    start_episode = 0
    
    def extract_episode(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1

    # PBT Cloning: If init_weights is provided, load those weights (priority over checkpoints)
    if init_weights and os.path.exists(init_weights):
        try:
            agent1.load_model(init_weights)
            print(f"[PBT] Loaded cloned weights from {init_weights}")
            start_episode = 0  # Start fresh episode count for cloned worker
        except Exception as e:
            print(f"[WARN] Failed to load init_weights: {e}")
            init_weights = None  # Fall back to checkpoint loading
    
    # Look for Rainbow models (only if not using PBT init_weights)
    if not init_weights:
        agent1_files = glob.glob(os.path.join(model_save_path, "agent1_rainbow_episode_*.pth"))
        agent2_files = glob.glob(os.path.join(model_save_path, "agent2_rainbow_episode_*.pth"))
        
        if agent1_files:
            agent1_files.sort(key=extract_episode, reverse=True)
            latest_file = agent1_files[0]
            try:
                agent1.load_model(latest_file)
                start_episode = extract_episode(latest_file)
                print(f"[OK] Loaded Agent 1 Rainbow model from {latest_file}")
            except Exception as e:
                print(f"[WARN] Failed to load Agent 1 model: {e}")
    
    # Load Agent 2 model (always from checkpoints, not affected by PBT cloning)
    agent2_files = glob.glob(os.path.join(model_save_path, "agent2_rainbow_episode_*.pth"))
    if agent2_files:
        agent2_files.sort(key=extract_episode, reverse=True)
        latest_file = agent2_files[0]
        try:
            agent2.load_model(latest_file)
            print(f"[OK] Loaded Agent 2 Rainbow model from {latest_file}")
        except Exception as e:
            print(f"[WARN] Failed to load Agent 2 model: {e}")


    # (HeuristicSetupAgent doesn't need model loading)

    # PBS evaluator loading removed — AAREN replaced PBS

    # Initialize Checkpointer and MetricsTracker (extracted modules)
    checkpointer = Checkpointer(save_dir=model_save_path)
    metrics_tracker = MetricsTracker(save_dir=model_save_path)
    
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

    
    # Unified reward shapers per lane
    lane_dist_rewards = [create_unified_reward_shaper(player_id=1, config=master_reward_config) for _ in range(num_envs)]
    lane_p2_shapers = [create_unified_reward_shaper(player_id=-1, config=master_reward_config) for _ in range(num_envs)]
    for dr, p2r in zip(lane_dist_rewards, lane_p2_shapers):
        dr.reset()
        p2r.reset()
    
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
    
    def select_opponent_for_lane(lane_idx):
        """Select opponent type for a lane based on curriculum or opponent pool."""
        # nonlocal use_full_obs, agent2 # No longer needed as we return
        # Logic remains the same, just returning selections

        
        opponent_type = "self"
        opponent_uses_history = True
        current_opponent = agent2
        
        if curriculum and CURRICULUM_ENABLED:
            opponent_dist = curriculum.get_opponent_distribution()
            r = random.random()
            cumulative = 0.0
            for op_type, prob in opponent_dist.items():
                cumulative += prob
                if r < cumulative:
                    opponent_type = op_type
                    break
            
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
                    # Load league agent (shared agent2 weights)
                    agent2.load_model(path)
                    current_opponent = agent2
                    opponent_uses_history = True
                else:
                    opponent_type = "self"
                    agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                    agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                    current_opponent = agent2
                    opponent_uses_history = True
            elif opponent_type == "exploiters":
                current_opponent = get_random_exploiter(device, player_id=-1)
                opponent_uses_history = False
            else:  # self_500, self
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                current_opponent = agent2
                opponent_uses_history = True
        else:
            # Legacy mode: use opponent pool
            opponent_type, opponent_data = opponent_pool.select_opponent()
            if opponent_type == "league":
                agent2.load_model(opponent_data)
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
    
    # State tracking for intervals
    last_replay_step = 0
    last_target_update_step = 0
    
    # Progress bar - updates based on completed episodes
    pbar = tqdm(total=num_episodes, initial=completed_episodes, desc="Training Episodes")
    
    # ==========================================================================
    # MAIN TRAINING LOOP
    # All lanes run in parallel, each at their own pace
    # ==========================================================================
    
    while completed_episodes < num_episodes:
        # Prepare batch actions based on whose turn it is in each lane
        actions = [None] * num_envs
        p1_acting_mask = np.zeros(num_envs, dtype=bool)  # Track which lanes have P1 acting
        
        # --- BATCHED ACTION SELECTION ---
        # Collect states for P1 and P2 separately for batched inference
        p1_indices = [i for i in range(num_envs) if lane_current_player[i] == 1]
        p2_indices = [i for i in range(num_envs) if lane_current_player[i] == -1]
        
        # P1 batch action
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
        
        # P2 batch action
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
                p2_actions = opp.act_batch(
                    group['states'], group['moves'], group['game_states'],
                    env_indices=group['indices'], full_observability=use_full_obs
                )
                for idx, lane_i in enumerate(group['indices']):
                    actions[lane_i] = p2_actions[idx]
        
        # --- STEP ALL ENVIRONMENTS ---
        next_states, rewards, dones, infos, next_valid_moves = parallel_env.step(actions)
        
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
        batch_active_mask = []
        batch_game_states = []
        batch_next_game_states = []
        batch_infos = []  # Track infos for battle detection in PER

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
                    
                    # Calculate P1 reward impact from P2's move
                    p1_additional_reward = lane_dist_rewards[i](
                        previous_state=lane_game_states[i],
                        action=actions[i],
                        current_state=next_states[i],
                        done=done_bool,
                        winner=infos[i].get('winner'),
                        info=infos[i]
                    )
                    
                    total_p1_reward = pending['reward'] + p1_additional_reward
                    lane_episode_rewards_p1[i] += p1_additional_reward
                    
                    # Complete the transition
                    done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
                    
                    batch_states.append(pending['state'])
                    batch_actions.append(pending['action'])
                    batch_rewards.append(total_p1_reward)
                    batch_next_states.append(next_states[i].board)
                    batch_dones.append(done_bool)
                    batch_active_mask.append(True)
                    batch_game_states.append(pending['game_state'])
                    batch_next_game_states.append(next_states[i])
                    batch_infos.append(infos[i])  # For PER battle detection
                    
                    lane_pending_transitions[i] = None # Clear pending
            
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
                for pos, piece_type in infos[i]['revealed_in_step']:
                    if agent1.history_instances and i < len(agent1.history_instances):
                        game_phase = "early" if lane_step_counts[i] < 50 else ("mid" if lane_step_counts[i] < 200 else "end")
                        agent1.history_instances[i].update_from_reveal(
                            pos, piece_type, game_phase=game_phase, turn_count=lane_step_counts[i]
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
                
                if agent1.history_instances and i < len(agent1.history_instances):
                    agent1.history_instances[i].reset()
                if lane_opponent_uses_history[i] and agent2.history_instances and i < len(agent2.history_instances):
                    agent2.history_instances[i].reset()
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
        # Use threshold check instead of modulo to avoid skipping updates when step_count > 1
        if global_step - last_replay_step >= REPLAY_UPDATE_INTERVAL:
            loss = agent1.replay(episode=completed_episodes)
            last_replay_step = global_step
            
            if loss:
                metrics['losses_p1'].append(loss)
                metrics['loss_steps_p1'].append(global_step)  # Track step for proper plotting
                
                # Accumulate loss for all active lanes (since replay updates are global but apply to active policy)
                # We interpret "episode loss" as the average loss *during the lifetime of that episode*.
                # Since training happens globally every N steps, we add this loss to all currently active episodes.
                for i in range(num_envs):
                    if not dones[i]: # Only if episode is active
                        lane_episode_loss_sum_p1[i] += loss
                        lane_episode_loss_count_p1[i] += 1
        

        # --- TARGET NETWORK UPDATE ---
        # --- TARGET NETWORK UPDATE ---
        # Use threshold check to prevent double updates or skips
        if global_step - last_target_update_step >= TARGET_UPDATE_INTERVAL:
            agent1.update_target_network()
            tqdm.write(f"[INFO] Target Network Updated (Step {global_step})")
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
                    from training_config import PHASE_1_MAX_TURNS, PHASE_2_MAX_TURNS, PHASE_3_MAX_TURNS, PHASE_4_MAX_TURNS, DEFAULT_MAX_TURNS
                    phase_max_turns = {1: PHASE_1_MAX_TURNS, 2: PHASE_2_MAX_TURNS, 3: PHASE_3_MAX_TURNS, 4: PHASE_4_MAX_TURNS}
                    new_max_turns = phase_max_turns.get(curriculum.current_phase.value, DEFAULT_MAX_TURNS)
                    parallel_env.set_max_turns(new_max_turns)
                    tqdm.write(f"[INFO] Max turns updated to {new_max_turns}")
                    
                    # Enable Agent 2 AAREN history at Phase 4
                    if curriculum.current_phase.value >= 4 and not agent2.use_pbs:
                        agent2.enable_history(num_envs)
                        tqdm.write("[INFO] Agent 2 AAREN history ENABLED for Phase 4+")
        
        # --- LR SCHEDULER STEP (per fixed step interval) ---
        if agent1.scheduler and global_step % 1000 == 0:
            agent1.scheduler.step()
        
        # Train AAREN supervised model periodically (lightweight cross-entropy on reveal data)
        if global_step % (REPLAY_UPDATE_INTERVAL * 10) == 0:
            aaren_loss = agent1.train_history(epochs=1)
            if aaren_loss is not None:
                tqdm.write(f"[AAREN] Supervised loss: {aaren_loss:.4f}")
        
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
            metrics_tracker.set_global_step(global_step)
            
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
            pbar.set_postfix({
                'R1': f"{recent_reward:.2f}",
                'W1': metrics['wins_p1'],
                'W2': metrics['wins_p2'],
                'Ph': phase_val,
                'Steps': f"{global_step//1000}k"
            })
        
    
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

