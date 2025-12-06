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
from setup_agent import SetupAgent
from game_state import GameState
from training_visualizer import plot_training_progress, create_training_gif, create_episode_gif, plot_setup_agent_progress, plot_pbs_evaluator_progress, plot_additional_metrics
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
from setup_league import SetupLeague
from opponents import RandomAgent, GreedyAgent, OpponentPool
from training_config import *
from training_utils import save_training_history, load_training_history
from setup_evaluation import calculate_setup_agent_reward
from preflight_checks import run_preflight_checks

# Curriculum and Reward Shaping
from curriculum import CurriculumManager, TrainingPhase, HeuristicOpponent
from reward_shaping import RewardCalculator, RewardWeights, create_move_info
from exploiter_agents import get_random_exploiter, RusherAgent, TurtleAgent, FlankingAgent
from scenario_drills import get_scenario_drill, get_random_scenario

# Piece Value Tracking (for convergence analysis)
try:
    from piece_value_tracker import PieceValueTracker, ANALYTICAL_VALUES
    PIECE_VALUE_TRACKING = True
except ImportError:
    PIECE_VALUE_TRACKING = False
    print("⚠️ Piece value tracking disabled (module not found)")

def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     model_save_path: str = "dqn_models",
                     use_setup_agents: bool = True,
                     generate_gifs: bool = True):
    """
    Train Rainbow DQN agent with league-based diverse opponents.
    Early training uses self-play, then transitions to historical opponents.
    """
    # Set up device with robust CUDA check
    # torch.cuda.is_available() can return True even if PyTorch wasn't compiled with CUDA
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
    
    # Optimize GPU settings for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create model directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize Parallel Environment
    print(f"Initializing {NUM_ENVS} parallel environments...")
    parallel_env = ParallelStrategoEnvironment(num_envs=NUM_ENVS, device=device)
    
    # Initialize Agents
    print("Initializing Rainbow Agents...")
    agent1 = RainbowAgent(player_id=1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    # Agent2 uses minimal buffer - mainly loads from league, doesn't train
    agent2 = RainbowAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=10000)
    
    # Initialize Setup Agents
    setup_agent1 = SetupAgent(player_id=1, device=device)
    setup_agent2 = SetupAgent(player_id=-1, device=device)
    
    # Initialize League Manager and Opponent Pool
    league_dir = os.path.join(model_save_path, "league")
    league_manager = LeagueManager(league_dir=league_dir, max_agents=LEAGUE_MAX_AGENTS)
    
    opponent_pool = OpponentPool(
        league_manager=league_manager,
        device=device,
        league_prob=OPPONENT_LEAGUE_PROB,
        random_prob=OPPONENT_RANDOM_PROB,
        greedy_prob=OPPONENT_GREEDY_PROB,
        self_prob=OPPONENT_SELF_PROB
    )
    
    # Specialized opponents (for non-league matches)
    random_agent = RandomAgent()
    greedy_agent = GreedyAgent(device=device, player_id=-1)
    heuristic_agent = HeuristicOpponent(device=device, player_id=-1)  # Frozen heuristic for Phase 2
    
    # Initialize Curriculum Manager
    curriculum = None
    if CURRICULUM_ENABLED:
        curriculum = CurriculumManager(start_phase=CURRICULUM_START_PHASE, save_dir=model_save_path)
        print(f"📚 Curriculum enabled: Phase {curriculum.current_phase.value} ({curriculum.get_phase_config().name})")
        
        # Set initial observability based on phase
        if curriculum.should_use_full_observability():
            parallel_env.set_full_observability(True)
            print("   Full observability mode: ENABLED (Phase 1)")
    
    # Initialize Reward Calculator
    reward_weights = RewardWeights(
        outcome=REWARD_WEIGHT_OUTCOME,
        material=REWARD_WEIGHT_MATERIAL,
        epistemic=REWARD_WEIGHT_EPISTEMIC,
        positional=REWARD_WEIGHT_POSITIONAL
    )
    reward_calculator = RewardCalculator(device, reward_weights)
    
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

    # Load Setup Agents
    setup1_files = glob.glob(os.path.join(model_save_path, "setup_agent1_episode_*.pth"))
    setup2_files = glob.glob(os.path.join(model_save_path, "setup_agent2_episode_*.pth"))
    
    if setup1_files:
        setup1_files.sort(key=extract_episode, reverse=True)
        setup_agent1.load_model(setup1_files[0])
        print(f"✅ Loaded Setup Agent 1 from {setup1_files[0]}")
        
    if setup2_files:
        setup2_files.sort(key=extract_episode, reverse=True)
        setup_agent2.load_model(setup2_files[0])
        print(f"✅ Loaded Setup Agent 2 from {setup2_files[0]}")

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
        'avg_loss_p1_history': [],  # Note: Agent 2 doesn't train, so no loss tracking
        
        # Setup agent metrics (rewards, losses, convergence)
        'setup_agent1_rewards': [],
        'setup_agent2_rewards': [],
        'setup_agent1_losses': [],
        'setup_agent2_losses': [],
        
        # PBS evaluator metrics
        'pbs_eval1_losses': [],
        'pbs_eval2_losses': [],
        'pbs_eval1_buffer_sizes': [],
        'pbs_eval2_buffer_sizes': [],
        'pbs_eval1_accuracy': [],
        'pbs_eval2_accuracy': [],
        
        # Additional informative metrics
        'avg_q_values_p1': [],
        'avg_entropy_p1': [],
        'win_rate_100': [],  # Sliding window (last 100 episodes)
        
        'pbs_accuracy': []
    }
    
    if start_episode > 0:
        loaded_metrics = load_training_history(model_save_path)
        if loaded_metrics:
            metrics.update(loaded_metrics)
            print(f"Loaded training history.")

    print(f"🚀 Starting training from episode {start_episode + 1}...")
    
    global_step = 0
    
    pbar = tqdm(range(start_episode + 1, num_episodes + 1), desc="Training Episodes")
    
    for episode in pbar:
        # --- CURRICULUM-BASED OPPONENT SELECTION ---
        opponent_type = "self"  # Default
        opponent_uses_pbs = False
        
        if curriculum and CURRICULUM_ENABLED:
            phase = curriculum.current_phase
            
            # Set observability based on phase
            parallel_env.set_full_observability(curriculum.should_use_full_observability())
            
            # Get opponent distribution for current phase
            opponent_dist = curriculum.get_opponent_distribution()
            
            # Select opponent based on phase probabilities
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
            elif opponent_type in ["heuristic", "frozen_heuristic", "greedy"]:
                current_opponent = heuristic_agent
            elif opponent_type == "league":
                path = league_manager.get_opponent()
                if path:
                    agent2.load_model(path)
                    current_opponent = agent2
                    opponent_uses_pbs = True
                else:
                    # No league agents yet - fallback to self-play
                    opponent_type = "self"
                    agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                    agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                    current_opponent = agent2
                    opponent_uses_pbs = True
            elif opponent_type == "self_500" or opponent_type == "self":
                # Self-play: copy agent1 weights to agent2
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                current_opponent = agent2
                opponent_uses_pbs = True
            elif opponent_type == "exploiters":
                # Random exploiter agent
                current_opponent = get_random_exploiter(device, player_id=-1)
            elif opponent_type == "scenario":
                # Scenario drills (Phase 5) - handled separately
                current_opponent = heuristic_agent
            else:
                # Default to greedy
                current_opponent = greedy_agent
        else:
            # Legacy mode: use opponent pool
            opponent_type, opponent_data = opponent_pool.select_opponent()
            
            if opponent_type == "league":
                agent2.load_model(opponent_data)
                current_opponent = agent2
                opponent_uses_pbs = True
            elif opponent_type == "random":
                current_opponent = random_agent
            elif opponent_type == "greedy":
                current_opponent = greedy_agent
            else:  # self-play
                agent2.q_network.load_state_dict(agent1.q_network.state_dict())
                agent2.target_network.load_state_dict(agent1.target_network.state_dict())
                current_opponent = agent2
                opponent_uses_pbs = True
        
        # Reset reward calculator for new episode
        reward_calculator.reset()
        
        # 1. Generate Placements
        p1_placements = []
        p2_placements = []
        
        for _ in range(NUM_ENVS):
            p1_pieces = parallel_env.call_method('_generate_pieces')
            p1_pos = parallel_env.call_method('get_valid_placement_positions', 1)
            p1_place = setup_agent1.place_pieces(p1_pieces, p1_pos)
            p1_placements.append(p1_place)
            
            p2_pieces = parallel_env.call_method('_generate_pieces')
            p2_pos = parallel_env.call_method('get_valid_placement_positions', -1)
            p2_place = setup_agent2.place_pieces(p2_pieces, p2_pos)
            p2_placements.append(p2_place)
            
        # 2. Calculate setup agent evaluation rewards (before game starts)
        episode_setup_reward1 = 0.0
        episode_setup_reward2 = 0.0
        for i in range(NUM_ENVS):
            # Evaluate setup quality (flag protection, piece distribution, etc.)
            try:
                from setup_evaluation import evaluate_flag_protection, evaluate_piece_distribution
                setup_score1 = (evaluate_flag_protection(p1_placements[i], 1) + 
                               evaluate_piece_distribution(p1_placements[i], 1)) / 2.0
                setup_score2 = (evaluate_flag_protection(p2_placements[i], -1) + 
                               evaluate_piece_distribution(p2_placements[i], -1)) / 2.0
                episode_setup_reward1 += setup_score1
                episode_setup_reward2 += setup_score2
            except Exception:
                pass  # Silent fail to not impact training
        episode_setup_reward1 /= max(NUM_ENVS, 1)
        episode_setup_reward2 /= max(NUM_ENVS, 1)
        
        # 3. Reset Environments
        # parallel_env.reset returns (states, rewards, dones, infos, valid_moves)
        game_states, _, _, _, valid_moves = parallel_env.reset(p1_placements, p2_placements)
        
        # Reset Agents PBS (both for fair environment)
        agent1.reset_pbs()
        if opponent_uses_pbs:
            agent2.reset_pbs()
        
        episode_rewards = {1: np.zeros(NUM_ENVS), -1: np.zeros(NUM_ENVS)}
        active_envs = np.ones(NUM_ENVS, dtype=bool)
        
        # Track losses for this episode (Agent 1 only - Agent 2 doesn't train)
        episode_losses_p1 = []
        
        step_in_episode = 0
        
        # PBS optimization: track steps for interval-based updates
        from training_config import PBS_UPDATE_INTERVAL
        
        while np.any(active_envs):
            # 3. Get Actions for P1
            # valid_moves currently holds moves for the current player (P1 at start of loop)
            
            # P1 Actions
            actions_p1 = agent1.act_batch(
                [gs.board for gs in game_states],
                valid_moves,
                game_states,
                env_indices=list(range(NUM_ENVS))
            )
            
            # Step P1
            # parallel_env.step returns (next_states, rewards, dones, infos, valid_moves_for_next_player)
            next_states_p1, rewards_p1, dones_p1, infos_p1, valid_moves = parallel_env.step(actions_p1)
            
            # UPDATE P2's PBS with P1's moves (interval-based)
            # Always update if any action is a Scout move (2+ tiles), otherwise use interval
            should_update_p2_pbs = (step_in_episode % PBS_UPDATE_INTERVAL == 0)
            if not should_update_p2_pbs:
                # Check for Scout moves (must be updated immediately)
                for action in actions_p1:
                    if action:
                        (r_from, c_from), (r_to, c_to) = action
                        if abs(r_to - r_from) + abs(c_to - c_from) > 1:
                            should_update_p2_pbs = True
                            break
            if should_update_p2_pbs and opponent_uses_pbs:
                agent2.update_pbs_batch(actions_p1, game_states, acting_player=1)
            
            # Store P1 Experience (batched for efficiency)
            agent1.remember_batch(
                [gs.board for gs in game_states],
                actions_p1,
                rewards_p1,
                [ns.board for ns in next_states_p1],
                dones_p1,
                active_envs,
                game_states,
                next_states_p1
            )
            
            # Update episode rewards and track wins
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    episode_rewards[1][i] += rewards_p1[i]
                    if dones_p1[i]:
                        active_envs[i] = False
                        if infos_p1[i]['winner'] == 1: metrics['wins_p1'] += 1
                        elif infos_p1[i]['winner'] == -1: metrics['wins_p2'] += 1
                        else: metrics['draws'] += 1
            
            game_states = next_states_p1
            
            if not np.any(active_envs): break
            
            # P2 Actions (using current_opponent - may be league/random/greedy/self)
            actions_p2 = current_opponent.act_batch(
                [gs.board for gs in game_states],
                valid_moves,
                game_states,
                env_indices=list(range(NUM_ENVS))
            )
            
            # Step P2
            next_states_p2, rewards_p2, dones_p2, infos_p2, valid_moves = parallel_env.step(actions_p2)
            
            # UPDATE P1's PBS with P2's moves (interval-based)
            should_update_p1_pbs = (step_in_episode % PBS_UPDATE_INTERVAL == 0)
            if not should_update_p1_pbs:
                # Check for Scout moves (must be updated immediately)
                for action in actions_p2:
                    if action:
                        (r_from, c_from), (r_to, c_to) = action
                        if abs(r_to - r_from) + abs(c_to - c_from) > 1:
                            should_update_p1_pbs = True
                            break
            if should_update_p1_pbs:
                agent1.update_pbs_batch(actions_p2, game_states, acting_player=-1)
            
            # Store P2 Experience (batched for efficiency)
            agent2.remember_batch(
                [gs.board for gs in game_states],
                actions_p2,
                rewards_p2,
                [ns.board for ns in next_states_p2],
                dones_p2,
                active_envs,
                game_states,
                next_states_p2
            )
            
            # Update episode rewards and track wins
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    episode_rewards[-1][i] += rewards_p2[i]
                    if dones_p2[i]:
                        active_envs[i] = False
                        if infos_p2[i]['winner'] == 1: metrics['wins_p1'] += 1
                        elif infos_p2[i]['winner'] == -1: metrics['wins_p2'] += 1
                        else: metrics['draws'] += 1
                        
                        # Track piece values for convergence analysis
                        if piece_tracker is not None:
                            try:
                                # Get surviving pieces from final game state
                                board = game_states[i].board if hasattr(game_states[i], 'board') else game_states[i]
                                surviving_p1 = {}
                                surviving_p2 = {}
                                for r in range(10):
                                    for c in range(10):
                                        val = board[r, c].item() if hasattr(board[r, c], 'item') else board[r, c]
                                        if val > 0 and val <= 11:
                                            surviving_p1[val] = surviving_p1.get(val, 0) + 1
                                        elif val < 0 and val >= -11:
                                            surviving_p2[abs(val)] = surviving_p2.get(abs(val), 0) + 1
                                piece_tracker.record_game_end(infos_p2[i]['winner'], surviving_p1, surviving_p2)
                            except Exception:
                                pass  # Silent fail to not impact training

            game_states = next_states_p2
            step_in_episode += 1
            global_step += 1
            
            # 4. Training Step (only Agent1 trains)
            if global_step % REPLAY_UPDATE_INTERVAL == 0:
                loss1 = agent1.replay()
                
                if loss1: 
                    metrics['losses_p1'].append(loss1)
                    episode_losses_p1.append(loss1)
                
            # Update Target Networks (only Agent1)
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                agent1.update_target_network()
                print("🔄 Target Network Updated")

        # 5. Train PBS (AAREN) Models
        # Train on data collected during this episode
        agent1.train_pbs(epochs=5)
        agent2.train_pbs(epochs=5)

        # End of Episode Logging
        avg_reward_p1 = np.mean(episode_rewards[1])
        avg_reward_p2 = np.mean(episode_rewards[-1])
        metrics['rewards_p1'].append(avg_reward_p1)
        metrics['rewards_p2'].append(avg_reward_p2)
        metrics['lengths'].append(step_in_episode)
        
        # Update history metrics for plotting
        metrics['wins_p1_history'].append(metrics['wins_p1'])
        metrics['wins_p2_history'].append(metrics['wins_p2'])
        
        # Agent 1 loss tracking (Agent 2 doesn't train)
        avg_loss_p1 = np.mean(episode_losses_p1) if episode_losses_p1 else 0
        metrics['avg_loss_p1_history'].append(avg_loss_p1)
        
        # Setup agent metrics
        metrics['setup_agent1_rewards'].append(episode_setup_reward1)
        metrics['setup_agent2_rewards'].append(episode_setup_reward2)
        # Track setup agent losses if they have training losses
        setup_loss1 = setup_agent1.get_average_policy_loss(window=10) if hasattr(setup_agent1, 'get_average_policy_loss') else 0.0
        setup_loss2 = setup_agent2.get_average_policy_loss(window=10) if hasattr(setup_agent2, 'get_average_policy_loss') else 0.0
        metrics['setup_agent1_losses'].append(setup_loss1)
        metrics['setup_agent2_losses'].append(setup_loss2)
        
        # PBS evaluator metrics (use training_losses list and memory attribute)
        if agent1.pbs and hasattr(agent1.pbs, 'evaluator') and agent1.pbs.evaluator:
            eval1 = agent1.pbs.evaluator
            # Get last training loss if available
            last_loss1 = eval1.training_losses[-1] if hasattr(eval1, 'training_losses') and eval1.training_losses else 0.0
            buffer_size1 = len(eval1.memory) if hasattr(eval1, 'memory') else 0
            metrics['pbs_eval1_losses'].append(last_loss1)
            metrics['pbs_eval1_buffer_sizes'].append(buffer_size1)
            metrics['pbs_eval1_accuracy'].append(getattr(eval1, 'avg_accuracy', 0.0))
        else:
            metrics['pbs_eval1_losses'].append(0.0)
            metrics['pbs_eval1_buffer_sizes'].append(0)
            metrics['pbs_eval1_accuracy'].append(0.0)
            
        if agent2.pbs and hasattr(agent2.pbs, 'evaluator') and agent2.pbs.evaluator:
            eval2 = agent2.pbs.evaluator
            last_loss2 = eval2.training_losses[-1] if hasattr(eval2, 'training_losses') and eval2.training_losses else 0.0
            buffer_size2 = len(eval2.memory) if hasattr(eval2, 'memory') else 0
            metrics['pbs_eval2_losses'].append(last_loss2)
            metrics['pbs_eval2_buffer_sizes'].append(buffer_size2)
            metrics['pbs_eval2_accuracy'].append(getattr(eval2, 'avg_accuracy', 0.0))
        else:
            metrics['pbs_eval2_losses'].append(0.0)
            metrics['pbs_eval2_buffer_sizes'].append(0)
            metrics['pbs_eval2_accuracy'].append(0.0)
        
        # Additional informative metrics
        # Average Q-value from agent1
        avg_q = agent1.get_average_q() if hasattr(agent1, 'get_average_q') else 0.0
        metrics['avg_q_values_p1'].append(avg_q)
        
        # Action entropy (exploration diversity) - use noisy network sigma as proxy
        entropy = agent1.get_exploration_entropy() if hasattr(agent1, 'get_exploration_entropy') else 0.0
        metrics['avg_entropy_p1'].append(entropy)
        
        # Sliding window win rate (last 100 episodes)
        if len(metrics['wins_p1_history']) >= 100:
            # Calculate wins in last 100 episodes
            wins_100 = metrics['wins_p1_history'][-1] - metrics['wins_p1_history'][-100]
            win_rate_100 = wins_100 / 100.0
        else:
            # Not enough episodes yet
            win_rate_100 = metrics['wins_p1'] / max(len(metrics['wins_p1_history']), 1)
        metrics['win_rate_100'].append(win_rate_100)
        
        # --- CURRICULUM METRICS UPDATE ---
        if curriculum and CURRICULUM_ENABLED:
            # Determine winner from episode
            episode_winner = 0  # Draw by default
            for i in range(NUM_ENVS):
                # Check the final state from any environment
                if not np.any(active_envs):
                    # All done - check last info
                    break
            
            # Use metrics to determine winner (most common outcome)
            total_games = metrics['wins_p1'] + metrics['wins_p2'] + metrics['draws']
            if total_games > 0:
                last_win_p1 = metrics['wins_p1'] > metrics.get('prev_wins_p1', 0)
                last_win_p2 = metrics['wins_p2'] > metrics.get('prev_wins_p2', 0)
                if last_win_p1:
                    episode_winner = 1
                elif last_win_p2:
                    episode_winner = -1
            
            metrics['prev_wins_p1'] = metrics['wins_p1']
            metrics['prev_wins_p2'] = metrics['wins_p2']
            
            # Update curriculum metrics
            curriculum.update_metrics({
                'winner': episode_winner,
                'opponent_type': opponent_type,
                'pbs_accuracy': agent1.pbs.avg_accuracy if agent1.pbs and hasattr(agent1.pbs, 'avg_accuracy') else 0.0
            })
            
            # Check for phase transition
            if curriculum.check_phase_transition():
                old_phase = curriculum.current_phase
                if curriculum.advance_phase():
                    print(f"\n🎓 PHASE TRANSITION: {old_phase.name} → {curriculum.current_phase.name}")
                    
                    # Update environment observability for new phase
                    parallel_env.set_full_observability(curriculum.should_use_full_observability())
            
            # Save curriculum state periodically
            if episode % save_interval == 0:
                curriculum.save_state()
            
            # Enhanced progress bar with phase info
            pbar.set_postfix({
                'R1': f"{avg_reward_p1:.2f}",
                'W1': metrics['wins_p1'],
                'W2': metrics['wins_p2'],
                'Ph': curriculum.current_phase.value,
                'Opp': opponent_type[:4]
            })
        else:
            pbar.set_postfix({
                'R1': f"{avg_reward_p1:.2f}",
                'W1': metrics['wins_p1'],
                'W2': metrics['wins_p2'],
                'Opp': opponent_type[:4]
            })
        
        # Save Models
        if episode % save_interval == 0:
            agent1_path = os.path.join(model_save_path, f"agent1_rainbow_episode_{episode}.pth")
            agent1.save_model(agent1_path)
            
            # Add to league for diverse future opponents
            if episode % LEAGUE_SAVE_INTERVAL == 0:
                league_manager.save_agent(agent1_path, episode)
            
            # Save Setup Agents
            setup_agent1.save_model(os.path.join(model_save_path, f"setup_agent1_episode_{episode}.pth"))
            setup_agent2.save_model(os.path.join(model_save_path, f"setup_agent2_episode_{episode}.pth"))
            
            # Save PBS Evaluators (if available)
            if agent1.pbs and hasattr(agent1.pbs, 'evaluator') and agent1.pbs.evaluator:
                try:
                    agent1.pbs.evaluator.save_model(os.path.join(model_save_path, f"pbs_evaluator1_episode_{episode}.pth"))
                except Exception as e:
                    print(f"⚠️ Could not save PBS Evaluator 1: {e}")
            if agent2.pbs and hasattr(agent2.pbs, 'evaluator') and agent2.pbs.evaluator:
                try:
                    agent2.pbs.evaluator.save_model(os.path.join(model_save_path, f"pbs_evaluator2_episode_{episode}.pth"))
                except Exception as e:
                    print(f"⚠️ Could not save PBS Evaluator 2: {e}")
            
            save_training_history(metrics, model_save_path)
            
            # Prepare data for plotting
            episode_history = list(range(start_episode + 1, episode + 1))
            # If we loaded history, we need to adjust the range or just use the length of metrics
            # Ideally, metrics lists should align with total episodes.
            # Let's assume metrics lists are the source of truth for history length
            current_history_len = len(metrics['rewards_p1'])
            plot_episodes = list(range(1, current_history_len + 1))
            
            plot_training_progress(
                episode_history=plot_episodes,
                rewards_history={'agent1': metrics['rewards_p1'], 'agent2': metrics['rewards_p2']},
                wins_history={'agent1': metrics['wins_p1_history'], 'agent2': metrics['wins_p2_history']},
                policy_loss_history={'agent1': metrics['avg_loss_p1_history'], 'agent2': [0.0] * len(metrics['avg_loss_p1_history'])},  # Agent 2 doesn't train
                save_path=os.path.join(model_save_path, f"training_progress_episode_{episode}.png"),
                total_episodes=episode,
                total_steps=global_step
            )
            
            # Plot Setup Agent Progress
            try:
                plot_setup_agent_progress(
                    episode_history=plot_episodes,
                    setup_agent1_rewards=metrics['setup_agent1_rewards'],
                    setup_agent2_rewards=metrics['setup_agent2_rewards'],
                    setup_agent1_losses=metrics['setup_agent1_losses'],
                    setup_agent2_losses=metrics['setup_agent2_losses'],
                    save_path=os.path.join(model_save_path, f"setup_agent_progress_episode_{episode}.png")
                )
            except Exception as e:
                print(f"⚠️ Could not plot setup agent progress: {e}")
            
            # Plot PBS Evaluator Progress
            try:
                plot_pbs_evaluator_progress(
                    episode_history=plot_episodes,
                    evaluator1_losses=metrics['pbs_eval1_losses'],
                    evaluator2_losses=metrics['pbs_eval2_losses'],
                    evaluator1_buffer_sizes=metrics['pbs_eval1_buffer_sizes'],
                    evaluator2_buffer_sizes=metrics['pbs_eval2_buffer_sizes'],
                    save_path=os.path.join(model_save_path, f"pbs_evaluator_progress_episode_{episode}.png"),
                    total_episodes=episode
                )
            except Exception as e:
                print(f"⚠️ Could not plot PBS evaluator progress: {e}")
            
            # Plot Additional Metrics (Q-values, entropy, win rate)
            try:
                plot_additional_metrics(
                    episode_history=plot_episodes,
                    epsilon_history={'agent1': [0.0] * len(plot_episodes), 'agent2': [0.0] * len(plot_episodes)},  # Noisy networks, no epsilon
                    pbs_buffer_sizes={'agent1': metrics['pbs_eval1_buffer_sizes'], 'agent2': metrics['pbs_eval2_buffer_sizes']},
                    avg_q_history={'agent1': metrics['avg_q_values_p1'], 'agent2': [0.0] * len(metrics['avg_q_values_p1'])},
                    entropy_history={'agent1': metrics['avg_entropy_p1'], 'agent2': [0.0] * len(metrics['avg_entropy_p1'])},
                    save_path=os.path.join(model_save_path, f"additional_metrics_episode_{episode}.png")
                )
            except Exception as e:
                print(f"⚠️ Could not plot additional metrics: {e}")
            
            print(f"💾 Saved models and plots for episode {episode}")
            
            # Log piece value convergence (every 500 episodes)
            if piece_tracker is not None and episode % 500 == 0:
                piece_tracker.log_comparison(episode)
                piece_tracker.save()

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
