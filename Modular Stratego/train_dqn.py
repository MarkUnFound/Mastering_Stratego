"""
Training Script for Rainbow DQN Agents in Stratego
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
from training_config import *
from training_utils import save_training_history, load_training_history
from setup_evaluation import calculate_setup_agent_reward
from preflight_checks import run_preflight_checks

def train_dqn_agents(num_episodes: int = 1000, save_interval: int = 100, 
                     model_save_path: str = "dqn_models",
                     use_setup_agents: bool = True,
                     generate_gifs: bool = True):
    """
    Train two Rainbow DQN agents through self-play
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    agent2 = RainbowAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    
    # Initialize Setup Agents
    setup_agent1 = SetupAgent(player_id=1, device=device)
    setup_agent2 = SetupAgent(player_id=-1, device=device)
    
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

    # Metrics
    metrics = {
        'rewards_p1': [], 'rewards_p2': [],
        'wins_p1': 0, 'wins_p2': 0, 'draws': 0,
        'lengths': [],
        'losses_p1': [], 'losses_p2': [],
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
            
        # 2. Reset Environments
        # parallel_env.reset returns (states, rewards, dones, infos, valid_moves)
        game_states, _, _, _, valid_moves = parallel_env.reset(p1_placements, p2_placements)
        
        # Reset Agents
        agent1.reset_pbs()
        agent2.reset_pbs()
        
        episode_rewards = {1: np.zeros(NUM_ENVS), -1: np.zeros(NUM_ENVS)}
        active_envs = np.ones(NUM_ENVS, dtype=bool)
        
        step_in_episode = 0
        
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
            
            # Store P1 Experience
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    agent1.remember(game_states[i].board, actions_p1[i], rewards_p1[i], next_states_p1[i].board, dones_p1[i])
                    episode_rewards[1][i] += rewards_p1[i]
                    
                    if dones_p1[i]:
                        active_envs[i] = False
                        if infos_p1[i]['winner'] == 1: metrics['wins_p1'] += 1
                        elif infos_p1[i]['winner'] == -1: metrics['wins_p2'] += 1
                        else: metrics['draws'] += 1
            
            game_states = next_states_p1
            
            if not np.any(active_envs): break
            
            # P2 Actions
            # valid_moves now holds moves for P2
            actions_p2 = agent2.act_batch(
                [gs.board for gs in game_states],
                valid_moves,
                game_states,
                env_indices=list(range(NUM_ENVS))
            )
            
            # Step P2
            next_states_p2, rewards_p2, dones_p2, infos_p2, valid_moves = parallel_env.step(actions_p2)
            
            # Store P2 Experience
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    agent2.remember(game_states[i].board, actions_p2[i], rewards_p2[i], next_states_p2[i].board, dones_p2[i])
                    episode_rewards[-1][i] += rewards_p2[i]
                    
                    if dones_p2[i]:
                        active_envs[i] = False
                        if infos_p2[i]['winner'] == 1: metrics['wins_p1'] += 1
                        elif infos_p2[i]['winner'] == -1: metrics['wins_p2'] += 1
                        else: metrics['draws'] += 1

            game_states = next_states_p2
            step_in_episode += 1
            global_step += 1
            
            # 4. Training Step
            if global_step % REPLAY_UPDATE_INTERVAL == 0:
                loss1 = agent1.replay()
                loss2 = agent2.replay()
                
                if loss1: metrics['losses_p1'].append(loss1)
                if loss2: metrics['losses_p2'].append(loss2)
                
            # Update Target Networks
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                agent1.update_target_network()
                agent2.update_target_network()
                print("🔄 Target Networks Updated")

        # End of Episode Logging
        avg_reward_p1 = np.mean(episode_rewards[1])
        avg_reward_p2 = np.mean(episode_rewards[-1])
        metrics['rewards_p1'].append(avg_reward_p1)
        metrics['rewards_p2'].append(avg_reward_p2)
        metrics['lengths'].append(step_in_episode)
        
        pbar.set_postfix({
            'R1': f"{avg_reward_p1:.2f}",
            'R2': f"{avg_reward_p2:.2f}",
            'Win1': metrics['wins_p1'],
            'Win2': metrics['wins_p2']
        })
        
        # Save Models
        if episode % save_interval == 0:
            agent1.save_model(os.path.join(model_save_path, f"agent1_rainbow_episode_{episode}.pth"))
            agent2.save_model(os.path.join(model_save_path, f"agent2_rainbow_episode_{episode}.pth"))
            save_training_history(metrics, model_save_path)
            plot_training_progress(metrics, model_save_path)
            print(f"💾 Saved models and plots for episode {episode}")

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
