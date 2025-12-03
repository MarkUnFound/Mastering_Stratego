"""
Training Script for DQN Agents in Stratego
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
from dqn_agent import DQNAgent
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
from training_config import *
from training_utils import save_training_history, load_training_history
from setup_evaluation import calculate_setup_agent_reward






















def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     model_save_path: str = "dqn_models",
                     use_setup_agents: bool = True,
                     generate_gifs: bool = True):
    """
    Train two DQN agents through self-play
    
    Args:
        num_episodes: Number of training episodes
        save_interval: Interval for saving models and plots
        model_save_path: Path to save models and visualizations
        use_setup_agents: Whether to use setup agents for piece placement
        generate_gifs: Whether to generate GIFs (False to skip overhead)
    """
    REPLAY_UPDATE_INTERVAL = 4
    REPLAY_UPDATES_PER_STEP = 2
    PREFETCH_QUEUE_SIZE = 6
    TRAINING_BATCH_SIZE = 128
    TARGET_UPDATE_INTERVAL = 10000  # Update target network every 10000 steps (Increased for better stability)
    
    # Set up device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Optimize GPU settings for better performance
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    if device.type == 'cuda':
        # Enable TensorFloat32 (TF32) for faster float32 matrix multiplication on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Enable optimizations for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("⚠️  Using CPU - training will be slower")
    
    # Load persistent total episode and step counters (survives across training runs)
    total_episodes_file = os.path.join(model_save_path, "total_episodes.txt")
    total_steps_file = os.path.join(model_save_path, "total_steps.txt")
    
    total_episodes = 0
    total_steps = 0
    
    if os.path.exists(total_episodes_file):
        try:
            with open(total_episodes_file, 'r') as f:
                total_episodes = int(f.read().strip())
            print(f"📊 Loaded total episodes counter: {total_episodes}")
        except (ValueError, IOError) as e:
            print(f"⚠️  Could not load total episodes counter: {e}, starting from 0")
            total_episodes = 0
    else:
        print(f"📊 Starting total episodes counter from 0")
    
    if os.path.exists(total_steps_file):
        try:
            with open(total_steps_file, 'r') as f:
                total_steps = int(f.read().strip())
            print(f"📊 Loaded total steps counter: {total_steps}")
        except (ValueError, IOError) as e:
            print(f"⚠️  Could not load total steps counter: {e}, starting from 0")
            total_steps = 0
    else:
        print(f"📊 Starting total steps counter from 0")
    
    # Create environment
    # Create environment
    # env = StrategoEnvironment(device=device)
    env = ParallelStrategoEnvironment(num_envs=NUM_ENVS)
    
    # Create model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Load existing training history if available (for continuity across training sessions)
    loaded_history = load_training_history(model_save_path)
    
    # Create game-playing agents (with increased learning rate for CNN)
    agent1 = DQNAgent(player_id=1, device=device, lr=0.0001, batch_size=TRAINING_BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    agent2 = DQNAgent(player_id=-1, device=device, lr=0.0001, batch_size=TRAINING_BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    
    # Try to load the most recent saved models (separate files)
    try:
        # Find the most recent model checkpoint files (new separate format)
        agent1_dqn_files = glob.glob(os.path.join(model_save_path, "agent1_dqn_episode_*.pth"))
        agent2_dqn_files = glob.glob(os.path.join(model_save_path, "agent2_dqn_episode_*.pth"))
        
        # Also check for old combined format (backward compatibility)
        agent1_old_files = glob.glob(os.path.join(model_save_path, "agent1_episode_*.pth"))
        agent2_old_files = glob.glob(os.path.join(model_save_path, "agent2_episode_*.pth"))
        
        # Also check for final models
        final_agent1_dqn_path = os.path.join(model_save_path, "agent1_dqn_final.pth")
        final_agent2_dqn_path = os.path.join(model_save_path, "agent2_dqn_final.pth")
        final_agent1_old_path = os.path.join(model_save_path, "agent1_final.pth")
        final_agent2_old_path = os.path.join(model_save_path, "agent2_final.pth")
        
        def extract_episode(filepath):
            basename = os.path.basename(filepath)
            try:
                episode_num = int(basename.split('_episode_')[1].split('.')[0])
                return episode_num
            except:
                return 0
        
        # Determine which DQN models to load
        agent1_dqn_path = None
        agent2_dqn_path = None
        
        if agent1_dqn_files:
            agent1_dqn_files.sort(key=extract_episode, reverse=True)
            agent2_dqn_files.sort(key=extract_episode, reverse=True)
            agent1_dqn_path = agent1_dqn_files[0]
            if agent2_dqn_files:
                agent2_dqn_path = agent2_dqn_files[0]
        elif os.path.exists(final_agent1_dqn_path):
            agent1_dqn_path = final_agent1_dqn_path
            if os.path.exists(final_agent2_dqn_path):
                agent2_dqn_path = final_agent2_dqn_path
        elif agent1_old_files:
            # Fallback to old combined format
            agent1_old_files.sort(key=extract_episode, reverse=True)
            agent2_old_files.sort(key=extract_episode, reverse=True)
            agent1_dqn_path = agent1_old_files[0]
            if agent2_old_files:
                agent2_dqn_path = agent2_old_files[0]
        elif os.path.exists(final_agent1_old_path):
            agent1_dqn_path = final_agent1_old_path
            if os.path.exists(final_agent2_old_path):
                agent2_dqn_path = final_agent2_old_path
        
        # Load DQN models if found
        if agent1_dqn_path and os.path.exists(agent1_dqn_path):
            try:
                agent1.load_model(agent1_dqn_path)
                print(f"✅ Loaded Agent 1 DQN model from: {agent1_dqn_path}")
            except Exception as e:
                print(f"⚠️  Could not load Agent 1 DQN model from {agent1_dqn_path}: {e}")
                print(f"   File may be corrupted. Starting with fresh Agent 1.")
                agent1_dqn_path = None  # Mark as failed so we don't try to load PBS components
            
            # Check if old combined format has PBS components and load them (only if DQN loaded successfully)
            if agent1_dqn_path:
                try:
                    checkpoint = torch.load(agent1_dqn_path, map_location=device)
                    if 'pbs_aaren_state_dict' in checkpoint or 'pbs_lstm_state_dict' in checkpoint:
                        # Old combined format - try to load AAREN from it
                        if 'pbs_aaren_state_dict' in checkpoint:
                            agent1.pbs.aaren_model.load_state_dict(checkpoint['pbs_aaren_state_dict'])
                            if 'pbs_aaren_optimizer_state_dict' in checkpoint:
                                agent1.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_aaren_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 1 AAREN from combined checkpoint")
                        elif 'pbs_lstm_state_dict' in checkpoint:
                            agent1.pbs.aaren_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
                            if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                                agent1.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 1 AAREN (from LSTM) from combined checkpoint")
                    
                    if 'pbs_evaluator_state_dict' in checkpoint and agent1.pbs.evaluator is not None:
                        agent1.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
                        agent1.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
                        if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
                            agent1.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
                        agent1.pbs.evaluator.update_target_network()
                        print(f"✅ Loaded Agent 1 PBS Evaluator from combined checkpoint")
                except Exception as e:
                    # Not a combined checkpoint or error loading PBS components - that's okay
                    pass
        
        if agent2_dqn_path and os.path.exists(agent2_dqn_path):
            try:
                agent2.load_model(agent2_dqn_path)
                print(f"✅ Loaded Agent 2 DQN model from: {agent2_dqn_path}")
            except Exception as e:
                print(f"⚠️  Could not load Agent 2 DQN model from {agent2_dqn_path}: {e}")
                print(f"   File may be corrupted. Starting with fresh Agent 2.")
                agent2_dqn_path = None  # Mark as failed so we don't try to load PBS components
            
            # Check if old combined format has PBS components and load them (only if DQN loaded successfully)
            if agent2_dqn_path:
                try:
                    checkpoint = torch.load(agent2_dqn_path, map_location=device)
                    if 'pbs_aaren_state_dict' in checkpoint or 'pbs_lstm_state_dict' in checkpoint:
                        # Old combined format - try to load AAREN from it
                        if 'pbs_aaren_state_dict' in checkpoint:
                            agent2.pbs.aaren_model.load_state_dict(checkpoint['pbs_aaren_state_dict'])
                            if 'pbs_aaren_optimizer_state_dict' in checkpoint:
                                agent2.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_aaren_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 2 AAREN from combined checkpoint")
                        elif 'pbs_lstm_state_dict' in checkpoint:
                            agent2.pbs.aaren_model.load_state_dict(checkpoint['pbs_lstm_state_dict'])
                            if 'pbs_lstm_optimizer_state_dict' in checkpoint:
                                agent2.pbs.aaren_optimizer.load_state_dict(checkpoint['pbs_lstm_optimizer_state_dict'])
                            print(f"✅ Loaded Agent 2 AAREN (from LSTM) from combined checkpoint")
                    
                    if 'pbs_evaluator_state_dict' in checkpoint and agent2.pbs.evaluator is not None:
                        agent2.pbs.evaluator.evaluator_network.load_state_dict(checkpoint['pbs_evaluator_state_dict'])
                        agent2.pbs.evaluator.target_network.load_state_dict(checkpoint['pbs_evaluator_target_state_dict'])
                        if 'pbs_evaluator_optimizer_state_dict' in checkpoint:
                            agent2.pbs.evaluator.optimizer.load_state_dict(checkpoint['pbs_evaluator_optimizer_state_dict'])
                        agent2.pbs.evaluator.update_target_network()
                        print(f"✅ Loaded Agent 2 PBS Evaluator from combined checkpoint")
                except Exception as e:
                    # Not a combined checkpoint or error loading PBS components - that's okay
                    pass
        
        # Load AAREN models if found (separate files)
        agent1_aaren_files = glob.glob(os.path.join(model_save_path, "agent1_aaren_episode_*.pth"))
        agent2_aaren_files = glob.glob(os.path.join(model_save_path, "agent2_aaren_episode_*.pth"))
        final_agent1_aaren_path = os.path.join(model_save_path, "agent1_aaren_final.pth")
        final_agent2_aaren_path = os.path.join(model_save_path, "agent2_aaren_final.pth")
        
        agent1_aaren_path = None
        agent2_aaren_path = None
        
        if agent1_aaren_files:
            agent1_aaren_files.sort(key=extract_episode, reverse=True)
            agent2_aaren_files.sort(key=extract_episode, reverse=True)
            agent1_aaren_path = agent1_aaren_files[0]
            if agent2_aaren_files:
                agent2_aaren_path = agent2_aaren_files[0]
        elif os.path.exists(final_agent1_aaren_path):
            agent1_aaren_path = final_agent1_aaren_path
            if os.path.exists(final_agent2_aaren_path):
                agent2_aaren_path = final_agent2_aaren_path
        
        if agent1_aaren_path and os.path.exists(agent1_aaren_path):
            # try:
            #     agent1.load_aaren_model(agent1_aaren_path)
            # except Exception as e:
            #     print(f"⚠️  Could not load Agent 1 AAREN model: {e}")
            pass
        
        if agent2_aaren_path and os.path.exists(agent2_aaren_path):
            # try:
            #     print(f"DEBUG: Attempting to load Agent 2 AAREN model from {agent2_aaren_path}")
            #     agent2.load_aaren_model(agent2_aaren_path)
            #     print("DEBUG: Successfully loaded Agent 2 AAREN model")
            # except BaseException as e:
            #     print(f"⚠️  CRITICAL ERROR loading Agent 2 AAREN model: {type(e).__name__}: {e}")
            #     import traceback
            #     traceback.print_exc()
            pass
        
        print("DEBUG: Finished loading AAREN models")
        
        # Load PBS Evaluator models if found (separate files)
        agent1_evaluator_files = glob.glob(os.path.join(model_save_path, "agent1_pbs_evaluator_episode_*.pth"))
        agent2_evaluator_files = glob.glob(os.path.join(model_save_path, "agent2_pbs_evaluator_episode_*.pth"))
        final_agent1_evaluator_path = os.path.join(model_save_path, "agent1_pbs_evaluator_final.pth")
        final_agent2_evaluator_path = os.path.join(model_save_path, "agent2_pbs_evaluator_final.pth")
        
        agent1_evaluator_path = None
        agent2_evaluator_path = None
        
        if agent1_evaluator_files:
            agent1_evaluator_files.sort(key=extract_episode, reverse=True)
            agent2_evaluator_files.sort(key=extract_episode, reverse=True)
            agent1_evaluator_path = agent1_evaluator_files[0]
            if agent2_evaluator_files:
                agent2_evaluator_path = agent2_evaluator_files[0]
        elif os.path.exists(final_agent1_evaluator_path):
            agent1_evaluator_path = final_agent1_evaluator_path
            if os.path.exists(final_agent2_evaluator_path):
                agent2_evaluator_path = final_agent2_evaluator_path
        
        if agent1_evaluator_path and os.path.exists(agent1_evaluator_path):
            # try:
            #     agent1.load_pbs_evaluator(agent1_evaluator_path)
            # except Exception as e:
            #     print(f"⚠️  Could not load Agent 1 PBS Evaluator model: {e}")
            pass
        
        if agent2_evaluator_path and os.path.exists(agent2_evaluator_path):
            # try:
            #     agent2.load_pbs_evaluator(agent2_evaluator_path)
            # except Exception as e:
            #     print(f"⚠️  Could not load Agent 2 PBS Evaluator model: {e}")
            pass
        
    except Exception as e:
        print(f"⚠️  Could not load saved models: {e}")
        print("   Starting with fresh agents")
        traceback.print_exc()
    
    # Ensure learning rates are not too low (reset if decayed too much)
    # This prevents agents from being "stuck" if the LR was crushed in a previous run
    # CRITICAL FIX: Re-initialize optimizers completely to clear bad momentum/variance state
    # Optimizer re-initialization removed to preserve momentum from loaded models
    # Only re-initialize if we started fresh (no history) or if explicitly requested
    
    if not loaded_history:
        print("🧹 Initializing optimizers for fresh training...")
        pass
    
    # Create setup agents (for piece placement)
    setup_agent1 = SetupAgent(player_id=1, device=device) if use_setup_agents else None
    setup_agent2 = SetupAgent(player_id=-1, device=device) if use_setup_agents else None
    
    # Try to load setup agent models if available
    if use_setup_agents and setup_agent1 and setup_agent2:
        try:
            setup_agent1_files = glob.glob(os.path.join(model_save_path, "setup_agent1_episode_*.pth"))
            setup_agent2_files = glob.glob(os.path.join(model_save_path, "setup_agent2_episode_*.pth"))
            
            final_setup_agent1_path = os.path.join(model_save_path, "setup_agent1_final.pth")
            final_setup_agent2_path = os.path.join(model_save_path, "setup_agent2_final.pth")
            
            setup_agent1_path = None
            setup_agent2_path = None
            
            if setup_agent1_files:
                def extract_episode(filepath):
                    basename = os.path.basename(filepath)
                    try:
                        episode_num = int(basename.split('_episode_')[1].split('.')[0])
                        return episode_num
                    except:
                        return 0
                
                setup_agent1_files.sort(key=extract_episode, reverse=True)
                setup_agent2_files.sort(key=extract_episode, reverse=True)
                setup_agent1_path = setup_agent1_files[0]
                if setup_agent2_files:
                    setup_agent2_path = setup_agent2_files[0]
            elif os.path.exists(final_setup_agent1_path):
                setup_agent1_path = final_setup_agent1_path
                if os.path.exists(final_setup_agent2_path):
                    setup_agent2_path = final_setup_agent2_path
        
            if setup_agent1_path:
                setup_agent1.load_model(setup_agent1_path)
                print(f"✅ Loaded Setup Agent 1 from {setup_agent1_path}")
            if setup_agent2_path:
                setup_agent2.load_model(setup_agent2_path)
                print(f"✅ Loaded Setup Agent 2 from {setup_agent2_path}")
            
        except Exception as e:
            print(f"⚠️  Could not load setup agents: {e}")

    # Setup League is now run in a separate process (train_setup_league.py)
    # We just load the best agents produced by that process
    print("🏆 Setup League running in separate process. Will reload best agents periodically.")

    # Initialize League Manager
    league_manager = LeagueManager(league_dir=os.path.join(model_save_path, "league"))
    
    # Initialize History
    episode_history = []
    rewards_history = {'agent1': [], 'agent2': []}
    wins_history = {'agent1': [], 'agent2': [], 'draws': []}
    epsilon_history = {'agent1': [], 'agent2': []}
    policy_loss_history = {'agent1': [], 'agent2': []}
    setup_agent1_rewards = []
    setup_agent2_rewards = []
    setup_agent1_losses = []
    setup_agent2_losses = []
    pbs_evaluator1_losses = []
    pbs_evaluator2_losses = []
    pbs_evaluator1_buffer_sizes = []
    pbs_evaluator2_buffer_sizes = []
    avg_q_history = {'agent1': [], 'agent2': []}
    entropy_history = {'agent1': [], 'agent2': []}
    
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    total_rewards_agent1 = []
    total_rewards_agent2 = []
    agent1_losses = []
    agent2_losses = []
    
    if loaded_history:
        episode_history = loaded_history.get('episode_history', [])
        rewards_history = loaded_history.get('rewards_history', {'agent1': [], 'agent2': []})
        wins_history = loaded_history.get('wins_history', {'agent1': [], 'agent2': [], 'draws': []})
        if wins_history['agent1']: wins_agent1 = wins_history['agent1'][-1]
        if wins_history['agent2']: wins_agent2 = wins_history['agent2'][-1]
        if wins_history['draws']: draws = wins_history['draws'][-1]


    
    print("DEBUG: Ready to start training loop")
    
    # Reset all environments
    p1_placements = [None] * NUM_ENVS
    p2_placements = [None] * NUM_ENVS
    states_tuple, rewards, dones, infos, valid_moves_tuple = env.reset()
    states = list(states_tuple)
    valid_moves = list(valid_moves_tuple)
    
    episode_rewards_agent1 = [0.0] * NUM_ENVS
    episode_rewards_agent2 = [0.0] * NUM_ENVS
    episode_moves = [0] * NUM_ENVS
    pending_resets = [False] * NUM_ENVS
    placement_memory = {}
    
    completed_episodes = 0
    last_saved_episode = total_episodes
    last_plotted_episode = total_episodes
    last_reload_episode = 0
    
    pbar = tqdm(total=num_episodes, desc="Training Episodes")
    
    def save_checkpoint(episode_num, is_final=False):
        suffix = "final" if is_final else f"episode_{episode_num}"
        os.makedirs(model_save_path, exist_ok=True)
        agent1.save_model(f"{model_save_path}/agent1_dqn_{suffix}.pth")
        agent2.save_model(f"{model_save_path}/agent2_dqn_{suffix}.pth")
        if use_setup_agents:
            setup_agent1.save_model(f"{model_save_path}/setup_agent1_{suffix}.pth")
            setup_agent2.save_model(f"{model_save_path}/setup_agent2_{suffix}.pth")

    is_league_opponent = False
    
    try:
        while True:
            if completed_episodes >= num_episodes:
                break
            
            # --- Reload Best Setup Agents ---
            if use_setup_agents and total_episodes > 0 and total_episodes % 100 == 0 and total_episodes > last_reload_episode:
                last_reload_episode = total_episodes
                best_setup_path = os.path.join(model_save_path, "setup_agent_best.pth")
                if os.path.exists(best_setup_path):
                    try:
                        # Load the best agent into both setup agents (they share the best strategy)
                        # We use a helper to avoid code duplication
                        checkpoint = torch.load(best_setup_path, map_location=device)
                        
                        for agent in [setup_agent1, setup_agent2]:
                            agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                            # We might NOT want to load optimizer state if we aren't training them here
                            # But loading it is safer if we do decide to fine-tune
                            if 'optimizer_state_dict' in checkpoint:
                                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                        print(f"🔄 Reloaded best setup agents from {best_setup_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to reload best setup agent: {e}")

            # Action Selection
            actions_list = [None] * NUM_ENVS
            agent1_indices = [i for i in range(NUM_ENVS) if not pending_resets[i] and states[i].current_player == 1]
            agent2_indices = [i for i in range(NUM_ENVS) if not pending_resets[i] and states[i].current_player == -1]
        
            if agent1_indices:
                batch_states = [states[i] for i in agent1_indices]
                batch_moves = [valid_moves[i] for i in agent1_indices]
                batch_actions = agent1.act_batch(batch_states, batch_moves, game_states=batch_states)
                for idx, action in zip(agent1_indices, batch_actions):
                    actions_list[idx] = action
                
            if agent2_indices:
                batch_states = [states[i] for i in agent2_indices]
                batch_moves = [valid_moves[i] for i in agent2_indices]
                batch_actions = agent2.act_batch(batch_states, batch_moves, game_states=batch_states)
                for idx, action in zip(agent2_indices, batch_actions):
                    actions_list[idx] = action
                
            # PBS Updates
            if agent1_indices:
                agent2.update_pbs_batch([actions_list[i] for i in agent1_indices], [states[i] for i in agent1_indices], acting_player=1)
            if agent2_indices:
                agent1.update_pbs_batch([actions_list[i] for i in agent2_indices], [states[i] for i in agent2_indices], acting_player=-1)
            
            # Prepare Commands
            commands = []
            for i in range(NUM_ENVS):
                if pending_resets[i]:
                    p1_place = None
                    p2_place = None
                    if use_setup_agents:
                        # Use call_method to get pieces from the first worker
                        pieces = env.call_method('get_all_pieces')
                        p1_place = setup_agent1.place_pieces(pieces, env.call_method('get_valid_placement_positions', 1))
                        p2_place = setup_agent2.place_pieces(pieces, env.call_method('get_valid_placement_positions', -1))
                        placement_memory[i] = {'p1_placement': p1_place, 'p2_placement': p2_place}
                    commands.append(('reset', {'p1_placement': p1_place, 'p2_placement': p2_place}))
                else:
                    commands.append(actions_list[i])
                
            # Step
            next_states_tuple, step_rewards, step_dones, step_infos, next_valid_moves_tuple = env.step(commands)
            next_states = list(next_states_tuple)
            next_valid_moves = list(next_valid_moves_tuple)
        
            for i in range(NUM_ENVS):
                if pending_resets[i]:
                    states[i] = next_states[i]
                    valid_moves[i] = next_valid_moves[i]
                    pending_resets[i] = False
                    episode_rewards_agent1[i] = 0.0
                    episode_rewards_agent2[i] = 0.0
                    episode_moves[i] = 0
                    continue
                
                action = actions_list[i]
                reward = step_rewards[i].item()
                done = step_dones[i].item()
            
                current_agent = agent1 if states[i].current_player == 1 else agent2
                state_tensor = current_agent.get_state_representation(states[i])
                next_state_tensor = current_agent.get_state_representation(next_states[i])
                current_agent.remember(state_tensor, current_agent._move_to_action_index(action), reward, next_state_tensor, done)
            
                if states[i].current_player == 1: episode_rewards_agent1[i] += reward
                else: episode_rewards_agent2[i] += reward
                episode_moves[i] += 1
            
                states[i] = next_states[i]
                valid_moves[i] = next_valid_moves[i]
            
                if done:
                    pending_resets[i] = True
                    completed_episodes += 1
                    total_episodes += 1
                    pbar.update(1)
                
                    total_rewards_agent1.append(episode_rewards_agent1[i])
                    total_rewards_agent2.append(episode_rewards_agent2[i])
                
                    winner = step_infos[i].get('winner', 0)
                    if winner == 1: wins_agent1 += 1
                    elif winner == -1: wins_agent2 += 1
                    else: draws += 1
                
                    agent1.reset_pbs(i)
                    agent2.reset_pbs(i)
                
                    episode_history.append(total_episodes)
                    rewards_history['agent1'].append(episode_rewards_agent1[i])
                    rewards_history['agent2'].append(episode_rewards_agent2[i])
                    wins_history['agent1'].append(wins_agent1)
                    wins_history['agent2'].append(wins_agent2)
                    wins_history['draws'].append(draws)
                
                    # Record epsilon values
                    epsilon_history['agent1'].append(agent1.epsilon)
                    epsilon_history['agent2'].append(agent2.epsilon)
                
                    # Record recent average policy losses
                    recent_window = 10  # Average over last 10 training steps
                    avg_loss_agent1 = np.mean(agent1_losses[-recent_window:]) if len(agent1_losses) >= recent_window else (np.mean(agent1_losses) if agent1_losses else 0.0)
                    avg_loss_agent2 = np.mean(agent2_losses[-recent_window:]) if len(agent2_losses) >= recent_window else (np.mean(agent2_losses) if agent2_losses else 0.0)
                    policy_loss_history['agent1'].append(avg_loss_agent1)
                    policy_loss_history['agent2'].append(avg_loss_agent2)
                
                    # Record PBS evaluator metrics
                    pbs_eval_loss_1 = agent1.pbs.evaluator.get_average_loss() if agent1.pbs.evaluator else 0.0
                    pbs_eval_loss_2 = agent2.pbs.evaluator.get_average_loss() if agent2.pbs.evaluator else 0.0
                    pbs_evaluator1_losses.append(pbs_eval_loss_1)
                    pbs_evaluator2_losses.append(pbs_eval_loss_2)
                    pbs_evaluator1_buffer_sizes.append(len(agent1.pbs.evaluator.memory) if agent1.pbs.evaluator else 0)
                    pbs_evaluator2_buffer_sizes.append(len(agent2.pbs.evaluator.memory) if agent2.pbs.evaluator else 0)
                
                    # Train PBS Evaluator and AAREN (End of Episode)
                    # Train every episode to ensure data is used
                    agent1.train_pbs_evaluator(epochs=1)
                    agent2.train_pbs_evaluator(epochs=1)
                
                    # Record Average Q-Value and Entropy
                    avg_q_history['agent1'].append(agent1.get_average_q_value())
                    avg_q_history['agent2'].append(agent2.get_average_q_value())
                    entropy_history['agent1'].append(agent1.get_average_entropy())
                    entropy_history['agent2'].append(agent2.get_average_entropy())

                    # DEBUG: Print status every 100 episodes
                    if total_episodes % 100 == 0:
                        print(f"\n🔍 Diagnostics (Episode {total_episodes}):")
                        print(f"  Agent 1: Mem={len(agent1.memory)}, Loss={agent1_losses[-1] if agent1_losses else 'N/A'}, LR={agent1.optimizer.param_groups[0]['lr']:.2e}")
                        print(f"  Agent 2: Mem={len(agent2.memory)}, Loss={agent2_losses[-1] if agent2_losses else 'N/A'}, LR={agent2.optimizer.param_groups[0]['lr']:.2e}")
                        if len(agent2.memory) < agent2.batch_size:
                            print(f"  ⚠️ Agent 2 memory too low to train (< {agent2.batch_size})")
                        if agent2_losses and agent2_losses[-1] == 0.0:
                            print(f"  ⚠️ Agent 2 loss is exactly 0.0!")
                
                    # ============================================
                    # SETUP AGENT TRAINING
                    # ============================================
                    # Train setup agents if placements were generated by them
                    if i in placement_memory and use_setup_agents and setup_agent1 and setup_agent2:
                        # Calculate setup rewards based on game outcome
                        setup_reward_1 = calculate_setup_agent_reward(
                            placement_memory[i]['p1_placement'],
                            player_id=1,
                            winner=winner,
                            move_count=episode_moves[i]
                        )
                        setup_reward_2 = calculate_setup_agent_reward(
                            placement_memory[i]['p2_placement'],
                            player_id=-1,
                            winner=winner,
                            move_count=episode_moves[i]
                        )
                    
                        # Apply rewards to setup agent episode memory and store in replay buffer
                        # We don't train here anymore, just store experience if needed (though league handles training)
                        # setup_agent1.finish_episode(setup_reward_1)
                        # setup_agent2.finish_episode(setup_reward_2)
                    
                        # Train setup agents - REMOVED (Handled by Setup League)
                        # setup_loss_1 = setup_agent1.replay()
                        # setup_loss_2 = setup_agent2.replay()
                    
                        # Track setup agent performance for plotting
                        setup_agent1_rewards.append(setup_reward_1)
                        setup_agent2_rewards.append(setup_reward_2)
                        # Always append loss values (0 since we don't train here)
                        setup_agent1_losses.append(0.0)
                        setup_agent2_losses.append(0.0)
                    
                        # Clean up placement memory after training
                        del placement_memory[i]
                else:
                    # No setup agent data for this episode - append placeholders to maintain length match
                    if use_setup_agents:
                        setup_agent1_rewards.append(0.0)
                        setup_agent2_rewards.append(0.0)
                        setup_agent1_losses.append(0.0)
                        setup_agent2_losses.append(0.0)
                
                # ============================================
                # GENERATE PLOTS EVERY 50 EPISODES
                # ============================================
                # Prevent duplicate plot generation in parallel environment by checking last_plotted_episode
                if total_episodes % save_interval == 0 and total_episodes > last_plotted_episode:
                    # Generate training progress plots
                    try:
                        if len(episode_history) > 0:
                            # Plot DQN agent training progress
                            plot_path = f"{model_save_path}/training_progress_episode_{total_episodes}.png"
                            plot_training_progress(
                                episode_history,
                                rewards_history,
                                wins_history,
                                policy_loss_history,
                                plot_path,
                                total_episodes=total_episodes,
                                total_steps=total_steps
                            )
                            print(f"📊 Training progress plot saved: {plot_path}")
                            
                            # Plot setup agent progress (if using setup agents and have data)
                            if use_setup_agents and len(setup_agent1_rewards) > 0:
                                setup_plot_path = f"{model_save_path}/setup_agent_progress_episode_{total_episodes}.png"
                                plot_setup_agent_progress(
                                    episode_history,
                                    setup_agent1_rewards,
                                    setup_agent2_rewards,
                                    setup_agent1_losses,
                                    setup_agent2_losses,
                                    setup_plot_path
                                )
                                print(f"📊 Setup agent progress plot saved: {setup_plot_path}")
                            
                            # Plot PBS evaluator progress
                            if len(pbs_evaluator1_losses) > 0:
                                pbs_plot_path = f"{model_save_path}/pbs_evaluator_progress_episode_{total_episodes}.png"
                                plot_pbs_evaluator_progress(
                                    episode_history,
                                    pbs_evaluator1_losses,
                                    pbs_evaluator2_losses,
                                    pbs_evaluator1_buffer_sizes,
                                    pbs_evaluator2_buffer_sizes,
                                    pbs_plot_path,
                                    total_episodes=total_episodes
                                )
                                print(f"📊 PBS evaluator progress plot saved: {pbs_plot_path}")
                                
                            # Plot additional metrics (Epsilon, Buffer Size, Q-Value, Entropy)
                            additional_metrics_path = f"{model_save_path}/additional_metrics_episode_{total_episodes}.png"
                            plot_additional_metrics(
                                episode_history,
                                epsilon_history,
                                {'agent1': pbs_evaluator1_buffer_sizes, 'agent2': pbs_evaluator2_buffer_sizes},
                                avg_q_history,
                                entropy_history,
                                additional_metrics_path
                            )
                            print(f"📊 Additional metrics plot saved: {additional_metrics_path}")
                                
                            # Update last_plotted_episode to prevent duplicate plotting
                            last_plotted_episode = total_episodes
                                
                            # Save training history JSON for continuity
                            try:
                                save_training_history(
                                    model_save_path,
                                    episode_history,
                                    rewards_history,
                                    wins_history,
                                    epsilon_history,
                                    policy_loss_history,
                                    setup_agent1_rewards,
                                    setup_agent2_rewards,
                                    setup_agent1_losses,
                                    setup_agent2_losses,
                                    pbs_evaluator1_losses,
                                    pbs_evaluator2_losses,
                                    pbs_evaluator1_buffer_sizes,
                                    pbs_evaluator2_buffer_sizes,
                                    avg_q_history,
                                    entropy_history
                                )
                                print(f"💾 Training history JSON saved at episode {total_episodes}")
                            except Exception as json_err:
                                print(f"⚠️  Could not save training history JSON: {json_err}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not generate plots at episode {total_episodes}: {e}")
                        traceback.print_exc()
                     
                else:
                    states[i] = next_states[i]
                    valid_moves[i] = next_valid_moves[i]
                
                # 6. Train Agents (Batched)
                # Train every REPLAY_UPDATE_INTERVAL steps
                # Since we process NUM_ENVS steps per iteration, we train more frequently naturally
                if total_steps % REPLAY_UPDATE_INTERVAL == 0:
                     for _ in range(REPLAY_UPDATES_PER_STEP):
                         loss1 = agent1.replay()
                         if loss1 is not None:
                             agent1_losses.append(loss1)
                 
                         if not is_league_opponent:
                             loss2 = agent2.replay()
                             if loss2 is not None:
                                 agent2_losses.append(loss2)
                     
                     # Train PBS Evaluator (Once per update interval, not per step)
                     eval_loss1 = agent1.train_pbs_evaluator()
                     if eval_loss1 is not None:
                         pbs_evaluator1_losses.append(eval_loss1)
                 
                     if not is_league_opponent:
                         eval_loss2 = agent2.train_pbs_evaluator()
                         if eval_loss2 is not None:
                             pbs_evaluator2_losses.append(eval_loss2)
                 


                 
                # Update target networks
                # Update every TARGET_UPDATE_INTERVAL steps
                if total_steps % TARGET_UPDATE_INTERVAL == 0:
                     agent1.update_target_network()
                     agent2.update_target_network()
             
                total_steps += NUM_ENVS # Approx
            
                # Save models periodically (keep at save_interval)
                # Only save when we cross a new save interval threshold
                # Save models periodically (keep at save_interval)
                # Only save when we cross a new save interval threshold
                if total_episodes > 0 and total_episodes % save_interval == 0 and total_episodes > last_saved_episode:
                    save_checkpoint(total_episodes)
                    # Update last saved episode to prevent saving again
                    last_saved_episode = total_episodes
                
                    # Save to League
                    league_manager.save_agent(f"{model_save_path}/agent1_dqn_episode_{total_episodes}.pth", total_episodes)
            
                    # Save final persistent counters
                    try:
                        with open(total_episodes_file, 'w') as f:
                            f.write(str(total_episodes))
                        with open(total_steps_file, 'w') as f:
                            f.write(str(total_steps))
                        print(f"💾 Saved persistent counters: {total_episodes} episodes, {total_steps:,} steps")
                    except Exception as e:
                        print(f"⚠️  Could not save persistent counters: {e}")
    
                    # Save final training history for continuity
                    try:
                        save_training_history(
                            model_save_path,
                            episode_history,
                            rewards_history,
                            wins_history,
                            epsilon_history,
                            policy_loss_history,
                            setup_agent1_rewards,
                            setup_agent2_rewards,
                            setup_agent1_losses,
                            setup_agent2_losses,
                            pbs_evaluator1_losses,
                            pbs_evaluator2_losses,
                            pbs_evaluator1_buffer_sizes,
                            pbs_evaluator2_buffer_sizes,
                            avg_q_history,
                            entropy_history
                        )
                        print(f"💾 Saved final training history for continuity")
                    except Exception as e:
                        print(f"⚠️  Could not save final training history: {e}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user! Saving current state...")
        save_checkpoint(total_episodes)
        raise

    finally:
        # Ensure history is saved even if interrupted
        try:
            save_training_history(
                model_save_path,
                episode_history,
                rewards_history,
                wins_history,
                epsilon_history,
                policy_loss_history,
                setup_agent1_rewards,
                setup_agent2_rewards,
                setup_agent1_losses,
                setup_agent2_losses,
                pbs_evaluator1_losses,
                pbs_evaluator2_losses,
                pbs_evaluator1_buffer_sizes,
                pbs_evaluator2_buffer_sizes,
                avg_q_history,
                entropy_history
            )
            print(f"💾 Saved training history (finally block)")
        except Exception as e:
            print(f"⚠️  Could not save training history in finally block: {e}")
            
        print("Closing environment...")
        try:
            env.close()
        except:
            pass
            

    
    # Final training metrics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Episodes this run: {num_episodes}")
    print(f"Total Episodes (all runs): {total_episodes}")
    print(f"Total Steps (all runs): {total_steps:,}")
    print(f"Agent 1 Wins: {wins_agent1} ({wins_agent1/num_episodes*100:.1f}%)")
    print(f"Agent 2 Wins: {wins_agent2} ({wins_agent2/num_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.1f}%)")
    print(f"Average Reward Agent 1: {np.mean(total_rewards_agent1):.2f}")
    print(f"Average Reward Agent 2: {np.mean(total_rewards_agent2):.2f}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save final models (separate files)
    # Save final models (separate files)
    save_checkpoint(total_episodes, is_final=True)
    
    # Training completed
    return agent1, agent2


def main():
    """Main function to run DQN training"""
    print("🎮 DQN Agent Training for Stratego")
    print("=" * 50)
    
    # Training parameters
    model_save_path = "dqn_models"
    use_setup_agents = True  # Enable setup agents for piece placement
    
    try:
        # Train agents with parallel environment
        agent1, agent2 = train_dqn_agents(NUM_EPISODES, SAVE_INTERVAL, model_save_path,
                                          use_setup_agents=use_setup_agents,
                                          generate_gifs=GENERATE_GIFS)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
