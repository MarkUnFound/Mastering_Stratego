"""
Training Script for Rainbow DQN Agents in Stratego

Features:
- Single-agent focus (Agent1 trains, Agent2 for opponents)
- League training: Auto-switches from Agent2 to historical opponents
- Diverse opponents: League (50%), Random (20%), Greedy (20%), Self (10%)
- PBS/AAREN inference for fair opponent play
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
from training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_pbs_evaluator_progress, plot_additional_metrics
from pbs_visualizer import visualize_pbs_state, create_pbs_gif
from piece import PieceType, PIECE_RANKS
from board import LAKE_SQUARE

# Import reset function (optional)
try:
    from reset_dqn import reset_existing_agents
    RESET_AVAILABLE = True
except ImportError:
    RESET_AVAILABLE = False

# Import random starting player utilities
# This helps balance training by randomizing which agent goes first
from random_starting_player import should_swap_players, swap_placements, get_batch_swap_decisions
from league import LeagueManager
from opponents import RandomAgent, GreedyAgent, OpponentPool
from training_config import *
from training_utils import save_training_history, load_training_history
from preflight_checks import run_preflight_checks

# Curriculum and Reward Shaping
from curriculum import CurriculumManager, TrainingPhase, HeuristicOpponent, SmartHeuristicOpponent
from exploiter_agents import get_random_exploiter, RusherAgent, TurtleAgent, FlankingAgent
from scenario_drills import get_scenario_drill, get_random_scenario

# Distributional RL-Compatible Reward Shaping (C51 Normalized Anti-Stall)
from distributional_reward import create_unified_reward_shaper, StrategoRewardConfig

# Piece Value Tracking (for convergence analysis)
try:
    from piece_value_tracker import PieceValueTracker, ANALYTICAL_VALUES
    PIECE_VALUE_TRACKING = True
except ImportError:
    PIECE_VALUE_TRACKING = False
    print("⚠️ Piece value tracking disabled (module not found)")

def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     plot_interval: int = PLOT_INTERVAL,
                     model_save_path: str = "dqn_models",
                     generate_gifs: bool = True):
    """
    Train Rainbow DQN agent with league-based diverse opponents.
    Early training uses self-play, then transitions to historical opponents.
    """
    device = torch.device('cpu')  # Default to CPU
    if torch.cuda.is_available():
        try:
            # Actually try to use CUDA to verify it works
            _ = torch.tensor([1.0], device='cuda')
            device = torch.device('cuda')
        except (AssertionError, RuntimeError) as e:
            print(f"⚠️ CUDA detected but not usable: {e}")
            print("   Falling back to CPU. Install PyTorch with CUDA support for GPU acceleration.")
    print(f"Using device: {device}")
    
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
    # Agent2: minimal buffer, no PBS (saves ~18% training time in early phases)
    # PBS will be enabled for Agent 2 when reaching Phase 4
    agent2 = RainbowAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=batch_size, num_envs=num_envs, buffer_size=10000, use_pbs=False)
    print("⚡ Agent 2 PBS disabled for early phases (will enable at Phase 4)")
    
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
    greedy_agent = GreedyAgent(device=device, player_id=-1, config=master_reward_config)
    heuristic_agent = HeuristicOpponent(device=device, player_id=-1)  # Frozen heuristic for Phase 2
    smart_heuristic_agent = SmartHeuristicOpponent(device=device, player_id=-1)  # Strong heuristic opponent
    
    # Initialize Curriculum Manager
    curriculum = None
    if CURRICULUM_ENABLED:
        curriculum = CurriculumManager(start_phase=CURRICULUM_START_PHASE, save_dir=model_save_path)
        print(f"📚 Curriculum enabled: Phase {curriculum.current_phase.value} ({curriculum.get_phase_config().name})")
        
        # Set initial observability based on phase
        if curriculum.should_use_full_observability():
            parallel_env.set_full_observability(True)
            print("   Full observability mode: ENABLED (Phase 1)")
    
    # Reward shaper is initialized per lane later
    
    # Initialize Piece Value Tracker (for convergence analysis)
    piece_tracker = None
    if PIECE_VALUE_TRACKING:
        piece_tracker = PieceValueTracker(save_path=os.path.join(model_save_path, "piece_value_tracking.json"))
        print(f"📊 Piece value tracking enabled ({piece_tracker.games_tracked} games loaded)")
    
    
    # --- Load Existing Models ---
    start_episode = 0
    
    def extract_episode(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1

    # Look for Rainbow models
    agent1_files = glob.glob(os.path.join(model_save_path, "agent1_rainbow_episode_*.pth"))
    agent2_files = glob.glob(os.path.join(model_save_path, "agent2_rainbow_episode_*.pth"))
    
    if agent1_files:
        agent1_files.sort(key=extract_episode, reverse=True)
        latest_file = agent1_files[0]
        try:
            agent1.load_model(latest_file)
            start_episode = extract_episode(latest_file)
            print(f"✅ Loaded Agent 1 Rainbow model from {latest_file}")
        except Exception as e:
            print(f"⚠️ Failed to load Agent 1 model: {e}")
            
    if agent2_files:
        agent2_files.sort(key=extract_episode, reverse=True)
        latest_file = agent2_files[0]
        try:
            agent2.load_model(latest_file)
            print(f"✅ Loaded Agent 2 Rainbow model from {latest_file}")
        except Exception as e:
            print(f"⚠️ Failed to load Agent 2 model: {e}")


    # (HeuristicSetupAgent doesn't need model loading)

    # Load PBS Evaluators (if available)
    pbs_eval1_files = glob.glob(os.path.join(model_save_path, "pbs_evaluator1_episode_*.pth"))
    pbs_eval2_files = glob.glob(os.path.join(model_save_path, "pbs_evaluator2_episode_*.pth"))
    
    if pbs_eval1_files and agent1.pbs and hasattr(agent1.pbs, 'evaluator') and agent1.pbs.evaluator:
        pbs_eval1_files.sort(key=extract_episode, reverse=True)
        try:
            agent1.pbs.evaluator.load_model(pbs_eval1_files[0])
            print(f"✅ Loaded PBS Evaluator 1 from {pbs_eval1_files[0]}")
        except Exception as e:
            print(f"⚠️ Could not load PBS Evaluator 1: {e}")
            
    if pbs_eval2_files and agent2.pbs and hasattr(agent2.pbs, 'evaluator') and agent2.pbs.evaluator:
        pbs_eval2_files.sort(key=extract_episode, reverse=True)
        try:
            agent2.pbs.evaluator.load_model(pbs_eval2_files[0])
            print(f"✅ Loaded PBS Evaluator 2 from {pbs_eval2_files[0]}")
        except Exception as e:
            print(f"⚠️ Could not load PBS Evaluator 2: {e}")

    # Metrics
    metrics = {
        # Core agent metrics
        'rewards_p1': [], 'rewards_p2': [],
        'wins_p1': 0, 'wins_p2': 0, 'draws': 0,
        'wins_p1_history': [], 'wins_p2_history': [],  # Track cumulative wins per episode
        'lengths': [],
        'losses_p1': [],
        'loss_steps_p1': [],  # Track global_step when each loss was recorded (for proper x-axis)
        'avg_loss_p1_history': [],  # Note: Agent 2 doesn't train, so no loss tracking
        
        # (Setup agent metrics removed - using HeuristicSetupAgent)
        
        # PBS evaluator metrics (Agent 1 only - Agent 2 has use_pbs=False)
        'pbs_eval1_losses': [],
        'pbs_eval1_buffer_sizes': [],
        'pbs_eval1_accuracy': [],
        
        # AAREN metrics (tracks belief state inference quality)
        'aaren_loss': [],        # Per-episode AAREN training loss
        'aaren_accuracy': [],    # Per-episode AAREN prediction accuracy
        'aaren_buffer_size': [], # AAREN training buffer size
        
        # Additional informative metrics
        'avg_q_values_p1': [],
        'avg_entropy_p1': [],
        'win_rate_100': [],  # Sliding window (last 100 episodes)
        
        # Curriculum phase tracking for visualization
        'phase_history': [],  # Phase value (1-5) per episode
        
        'pbs_accuracy': [],
        'episode_end_steps': []  # Track global_step when episode ends
    }
    
    if start_episode > 0:
        loaded_metrics = load_training_history(model_save_path)
        if loaded_metrics:
            metrics.update(loaded_metrics)
            print(f"Loaded training history.")

    print(f"🚀 Starting MULTI-LANE training from episode {start_episode}...")
    print(f"   🔀 Each environment is an independent lane - no idle time!")
    print(f"   📊 Episode count = completed games (not batches)")
    
    # Load global_step from metrics if resuming, otherwise start at 0
    global_step = metrics.get('global_step', 0)
    completed_episodes = start_episode  # Count of completed individual games
    if start_episode > 0 and global_step > 0:
        print(f"📊 Resuming from global step {global_step:,}, episode {completed_episodes}")
    
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
    lane_opponent_uses_pbs = [True] * num_envs  # Whether opponent uses PBS per lane
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
    
    # Agent Logic: Use Full Obs only if rate is 1.0 (or very close)
    # If mixed (e.g. 0.8), agent should use partial mode (PBS) to handle the hidden 20%
    use_full_obs = (obs_rate >= 0.99)
    
    # PBS optimization: track steps for interval-based updates
    from training_config import PBS_UPDATE_INTERVAL
    
    # Helper function to generate placements and reset a single lane

    def reset_lane(lane_idx):
        """Generate new placements and prepare reset command for a lane."""
        # Generate pieces
        p1_pieces = parallel_env.call_method('_generate_pieces')
        p1_pos = parallel_env.call_method('get_valid_placement_positions', 1)
        p1_place = setup_agent1.place_pieces(p1_pieces, p1_pos)
        
        p2_pieces = parallel_env.call_method('_generate_pieces')
        p2_pos = parallel_env.call_method('get_valid_placement_positions', -1)
        p2_place = setup_agent2.place_pieces(p2_pieces, p2_pos)
        
        # Random starting player (50% swap)
        if random.random() < 0.5:
            p1_place, p2_place = swap_placements(p1_place, p2_place)
        
        return {'p1_placement': p1_place, 'p2_placement': p2_place}
    
    def select_opponent_for_lane(lane_idx):
        """Select opponent type for a lane based on curriculum or opponent pool."""
        nonlocal use_full_obs, agent2
        
        opponent_type = "self"
        opponent_uses_pbs = True
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
            if opponent_type == "random":
                current_opponent = random_agent
                opponent_uses_pbs = False
            elif opponent_type in ["heuristic", "frozen_heuristic"]:
                current_opponent = heuristic_agent
                opponent_uses_pbs = False
            elif opponent_type == "smart_heuristic":
                current_opponent = smart_heuristic_agent
                opponent_uses_pbs = False
            elif opponent_type == "greedy":
                current_opponent = greedy_agent
                opponent_uses_pbs = False
            elif opponent_type == "league":
                path = league_manager.get_opponent()
                if path:
                    # Load league agent (shared agent2 weights)
                    agent2.load_model(path)
                    current_opponent = agent2
                    opponent_uses_pbs = True
                else:
                    opponent_type = "self"
                    agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                    agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                    current_opponent = agent2
                    opponent_uses_pbs = True
            elif opponent_type == "exploiters":
                current_opponent = get_random_exploiter(device, player_id=-1)
                opponent_uses_pbs = False
            else:  # self_500, self
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                current_opponent = agent2
                opponent_uses_pbs = True
        else:
            # Legacy mode: use opponent pool
            opponent_type, opponent_data = opponent_pool.select_opponent()
            if opponent_type == "league":
                agent2.load_model(opponent_data)
                current_opponent = agent2
                opponent_uses_pbs = True
            elif opponent_type == "random":
                current_opponent = random_agent
                opponent_uses_pbs = False
            elif opponent_type == "greedy":
                current_opponent = greedy_agent
                opponent_uses_pbs = False
            else:
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                current_opponent = agent2
                opponent_uses_pbs = True
        
        return opponent_type, opponent_uses_pbs, current_opponent
    
    # Initial reset for all lanes
    print(f"🔄 Initializing {num_envs} lanes...")
    initial_placements_p1 = []
    initial_placements_p2 = []
    for i in range(num_envs):
        placements = reset_lane(i)
        initial_placements_p1.append(placements['p1_placement'])
        initial_placements_p2.append(placements['p2_placement'])
        
        # Select opponent for each lane
        opp_type, opp_uses_pbs, opp = select_opponent_for_lane(i)
        lane_opponent_types[i] = opp_type
        lane_opponent_uses_pbs[i] = opp_uses_pbs
        lane_current_opponents[i] = opp
    
    # Reset all environments
    game_states, _, _, _, valid_moves = parallel_env.reset(initial_placements_p1, initial_placements_p2)
    for i in range(num_envs):
        lane_game_states[i] = game_states[i]
        lane_valid_moves[i] = valid_moves[i]
        lane_current_player[i] = 1  # P1 always starts
    
    # Loss tracking arrays
    lane_episode_loss_sum_p1 = np.zeros(num_envs)
    lane_episode_loss_count_p1 = np.zeros(num_envs, dtype=int)
    
    # Reset PBS for all lanes
    agent1.reset_pbs()
    agent2.reset_pbs()
    
    # State tracking for intervals
    last_replay_step = 0
    last_target_update_step = 0
    
    # Progress bar - updates based on completed episodes
    pbar = tqdm(total=num_episodes, initial=completed_episodes, desc="Training Episodes")
    
    # ==========================================================================
    # MAIN TRAINING LOOP - Step-based, not episode-based
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
                    
                    lane_pending_transitions[i] = None # Clear pending
            
            # Update PBS
            done_bool = dones[i].item() if hasattr(dones[i], 'item') else dones[i]
            if not done_bool:
                if current_player == 1 and lane_opponent_uses_pbs[i]:
                    agent2.update_pbs_batch([actions[i]], [lane_game_states[i]], acting_player=1)
                elif current_player == -1 and not use_full_obs:
                    agent1.update_pbs_batch([actions[i]], [lane_game_states[i]], acting_player=-1)
            
            # --- PBS TRAINING (Critical for Phase 2+) ---
            from training_config import PBS_UPDATE_INTERVAL
            if global_step % PBS_UPDATE_INTERVAL == 0:
                agent1.train_pbs(epochs=5)
            
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
                elif winner == -1:
                    metrics['wins_p2'] += 1
                else:
                    metrics['draws'] += 1
                
                metrics['rewards_p1'].append(lane_episode_rewards_p1[i])
                metrics['rewards_p2'].append(lane_episode_rewards_p2[i])
                metrics['lengths'].append(lane_step_counts[i])
                
                # Removed old sliding window loss code
                
                metrics['wins_p1_history'].append(metrics['wins_p1'])
                metrics['wins_p2_history'].append(metrics['wins_p2'])
                
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
                        'pbs_accuracy': agent1.pbs.avg_accuracy if agent1.pbs and hasattr(agent1.pbs, 'avg_accuracy') else 0.0
                    })
                
                # --- GRANULAR METRICS (Per Episode) ---
                # PBS evaluator metrics
                if agent1.pbs and hasattr(agent1.pbs, 'evaluator') and agent1.pbs.evaluator:
                    eval1 = agent1.pbs.evaluator
                    last_loss = eval1.training_losses[-1] if hasattr(eval1, 'training_losses') and eval1.training_losses else 0.0
                    buffer_size = len(eval1.memory) if hasattr(eval1, 'memory') else 0
                    metrics['pbs_eval1_losses'].append(last_loss)
                    metrics['pbs_eval1_buffer_sizes'].append(buffer_size)
                    metrics['pbs_eval1_accuracy'].append(getattr(eval1, 'avg_accuracy', 0.0))
                else:
                    metrics['pbs_eval1_losses'].append(0.0)
                    metrics['pbs_eval1_buffer_sizes'].append(0)
                    metrics['pbs_eval1_accuracy'].append(0.0)
                
                # AAREN metrics
                if agent1.pbs:
                    metrics['aaren_loss'].append(agent1.pbs.get_aaren_avg_loss() if hasattr(agent1.pbs, 'get_aaren_avg_loss') else 0.0)
                    metrics['aaren_accuracy'].append(agent1.pbs.get_aaren_accuracy() if hasattr(agent1.pbs, 'get_aaren_accuracy') else 0.0)
                    metrics['aaren_buffer_size'].append(agent1.pbs.get_aaren_buffer_size() if hasattr(agent1.pbs, 'get_aaren_buffer_size') else 0)
                
                # Q-value and entropy metrics
                avg_q = agent1.get_average_q() if hasattr(agent1, 'get_average_q') else 0.0
                if avg_q == 0.0 and hasattr(agent1, 'memory') and len(agent1.memory) > agent1.batch_size:
                     tqdm.write(f"⚠️ Avg Q-Value is 0.0. Memory: {len(agent1.memory)}")
                metrics['avg_q_values_p1'].append(avg_q)
                
                
                metrics['avg_entropy_p1'].append(agent1.get_exploration_entropy() if hasattr(agent1, 'get_exploration_entropy') else 0.0)
                
                # Track exact step when episode ended (for plotting alignment)
                metrics.get('episode_end_steps', []).append(global_step)
                
                completed_episodes += 1
                pbar.update(1)
                
                reset_commands[i] = reset_lane(i)
                opp_type, opp_uses_pbs, opp = select_opponent_for_lane(i)
                lane_opponent_types[i] = opp_type
                lane_opponent_uses_pbs[i] = opp_uses_pbs
                lane_current_opponents[i] = opp
                
                # Reset Lane State
                lane_episode_rewards_p1[i] = 0.0
                lane_episode_rewards_p2[i] = 0.0
                lane_episode_loss_sum_p1[i] = 0.0 # Reset loss sum
                lane_episode_loss_count_p1[i] = 0 # Reset loss count
                lane_step_counts[i] = 0
                lane_current_player[i] = 1
                lane_dist_rewards[i].reset()
                lane_p2_shapers[i].reset()
                lane_pending_transitions[i] = None
                
                if agent1.pbs_instances and i < len(agent1.pbs_instances):
                    agent1.pbs_instances[i].reset()
                if lane_opponent_uses_pbs[i] and agent2.pbs_instances and i < len(agent2.pbs_instances):
                    agent2.pbs_instances[i].reset()
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
                batch_next_game_states
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
            tqdm.write(f"🔄 Target Network Updated (Step {global_step})")
            last_target_update_step = global_step
        
        # --- CHECK CURRICULUM PHASE TRANSITION ---
        if curriculum and CURRICULUM_ENABLED and completed_episodes % 50 == 0:
            if curriculum.check_phase_transition():
                old_phase = curriculum.current_phase
                if curriculum.advance_phase():
                    tqdm.write(f"\n🎓 PHASE TRANSITION: {old_phase.name} → {curriculum.current_phase.name}")
                    
                    # Update observability
                    use_full_obs = curriculum.should_use_full_observability()
                    parallel_env.set_full_observability(use_full_obs)
                    
                    # Enable Agent 2 PBS at Phase 4
                    if curriculum.current_phase.value >= 4 and not agent2.use_pbs:
                        agent2.enable_pbs(num_envs)
                        tqdm.write("⚡ Agent 2 PBS ENABLED for Phase 4+")
        
        # --- LR SCHEDULER STEP (per fixed step interval) ---
        if agent1.scheduler and global_step % 1000 == 0:
            agent1.scheduler.step()
        
        # --- TRAIN PBS (periodically) ---
        if global_step % 500 == 0 and not use_full_obs:
            agent1.train_pbs(epochs=5)
            if agent2.use_pbs:
                agent2.train_pbs(epochs=5)
        
        # --- PERIODIC PLOTTING (every plot_interval episodes) ---
        if completed_episodes > 0 and completed_episodes % plot_interval == 0:
            plot_episode = (completed_episodes // plot_interval) * plot_interval
            
            if plot_episode != metrics.get('last_plot_episode', 0):
                metrics['last_plot_episode'] = plot_episode
                
                # Q-value and entropy metrics
                # Moved to per-episode block above for granularity
                pass
                
                # Win rate
                
                # Win rate
                if len(metrics['wins_p1_history']) >= 100:
                    wins_100 = metrics['wins_p1_history'][-1] - metrics['wins_p1_history'][-100]
                    win_rate = wins_100 / 100.0
                else:
                    win_rate = metrics['wins_p1'] / max(completed_episodes, 1)
                metrics['win_rate_100'].append(win_rate)
                
                # Plot progress graphs

                current_history_len = len(metrics['rewards_p1'])
                plot_episodes = list(range(1, current_history_len + 1))
                
                try:
                    plot_training_progress(
                        episode_history=plot_episodes,
                        rewards_history={'agent1': metrics['rewards_p1'], 'agent2': metrics['rewards_p2']},
                        wins_history={'agent1': metrics['wins_p1_history'], 'agent2': metrics['wins_p2_history']},
                        policy_loss_history={'agent1': metrics['losses_p1']},  # Raw losses (step-based)
                        save_path=os.path.join(model_save_path, f"training_progress_episode_{plot_episode}.png"),
                        total_episodes=plot_episode,
                        total_steps=global_step,
                        num_envs=num_envs,
                        phase_history =metrics.get('phase_history', []),
                        loss_steps=metrics.get('loss_steps_p1', None),  # Step numbers for uniform x-axis
                        episode_end_steps=metrics.get('episode_end_steps', None)
                    )
                    
                    # Plot additional metrics (Q-values, entropy, etc.)
                    avg_entropy_hist = metrics.get('avg_entropy_p1', [0.0] * len(plot_episodes))
                    
                    plot_additional_metrics(
                        episode_history=plot_episodes,
                        episode_lengths={'agent1': metrics.get('lengths', [0] * len(plot_episodes))}, # Pass lengths instead of epsilon
                        win_rate_history={'agent1': metrics.get('win_rate_100', [0.0] * len(plot_episodes))},
                        avg_q_history={'agent1': metrics.get('avg_q_values_p1', [0.0] * len(plot_episodes))},
                        entropy_history={'agent1': avg_entropy_hist},
                        save_path=os.path.join(model_save_path, f"additional_metrics_episode_{plot_episode}.png")
                    )
                    
                    tqdm.write(f"📊 Saved all plots for episode {plot_episode}")
                except Exception as e:
                    tqdm.write(f"⚠️ Could not plot training progress: {e}")
        

        # --- PERIODIC MODEL SAVES (every SAVE_INTERVAL episodes) ---
        if completed_episodes > 0 and completed_episodes % save_interval == 0 and completed_episodes != start_episode:
            save_episode = (completed_episodes // save_interval) * save_interval
            
            if save_episode != metrics.get('last_save_episode', 0):
                metrics['last_save_episode'] = save_episode
                
                # Save models
                agent1_path = os.path.join(model_save_path, f"agent1_rainbow_episode_{save_episode}.pth")
                agent1.save_model(agent1_path)
                
                # Export to league
                if save_episode % LEAGUE_SAVE_INTERVAL == 0:
                    league_manager.save_agent(agent1_path, save_episode)
                
                # Save PBS evaluators
                if agent1.pbs and hasattr(agent1.pbs, 'evaluator') and agent1.pbs.evaluator:
                    try:
                        agent1.pbs.evaluator.save_model(os.path.join(model_save_path, f"pbs_evaluator1_episode_{save_episode}.pth"))
                    except Exception:
                        pass
                
                # Save curriculum
                if curriculum and CURRICULUM_ENABLED:
                    curriculum.save_state()
                
                # Save metrics history
                metrics['global_step'] = global_step
                save_training_history(metrics, model_save_path)
                
                tqdm.write(f"💾 Saved models for episode {save_episode}")
                
                # Piece value tracking
                if piece_tracker is not None and save_episode % 500 == 0:
                    piece_tracker.log_comparison(save_episode)
                    piece_tracker.save()
                
        # --- UPDATE PROGRESS BAR ---

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

    print(f"\\n🎉 Training complete! Completed {completed_episodes} episodes, {global_step:,} steps")


if __name__ == "__main__":
    print("🎮 DQN Agent Training for Stratego")
    print("==================================================")
    print()
    
    if run_preflight_checks(model_save_path="dqn_models"):
        train_dqn_agents(
            num_episodes=NUM_EPISODES,
            save_interval=SAVE_INTERVAL,
            generate_gifs=GENERATE_GIFS
        )
    else:
        print("❌ Pre-flight checks failed. Aborting training.")
