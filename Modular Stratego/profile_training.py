"""
Profiling Script for DQN Training
Measures time spent in each component of the training loop
"""

import torch
import numpy as np
import time
import os
import sys
import glob
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import StrategoEnvironment
from parallel_environment import ParallelStrategoEnvironment
from drqn_agent import RainbowAgent
from setup_agent import SetupAgent
from game_state import GameState
from training_config import *

class Profiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        
    def start(self, name):
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)
            del self.start_times[name]
            
    def report(self):
        print("\n" + "="*60)
        print("PROFILING RESULTS")
        print("="*60)
        
        total_time = sum(sum(t) for t in self.timings.values())
        
        # Sort by total time spent
        sorted_items = sorted(
            [(k, v) for k, v in self.timings.items()],
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        for name, times in sorted_items:
            total = sum(times)
            avg = total / len(times) if times else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(f"{name:40s}: {total:8.3f}s ({pct:5.1f}%) | avg: {avg*1000:7.2f}ms | calls: {len(times)}")
        
        print("="*60)
        print(f"Total profiled time: {total_time:.2f}s")
        print("="*60)


def profile_training(num_episodes=2):
    """Profile training for a few episodes to find bottlenecks"""
    
    profiler = Profiler()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize
    profiler.start("init_parallel_env")
    parallel_env = ParallelStrategoEnvironment(num_envs=NUM_ENVS, device=device)
    profiler.stop("init_parallel_env")
    
    profiler.start("init_agents")
    agent1 = RainbowAgent(player_id=1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    agent2 = RainbowAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    profiler.stop("init_agents")
    
    profiler.start("init_setup_agents")
    setup_agent1 = SetupAgent(player_id=1, device=device)
    setup_agent2 = SetupAgent(player_id=-1, device=device)
    profiler.stop("init_setup_agents")
    
    # Load existing models if available
    model_save_path = "dqn_models"
    agent1_files = glob.glob(os.path.join(model_save_path, "agent1_rainbow_episode_*.pth"))
    if agent1_files:
        agent1_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        try:
            agent1.load_model(agent1_files[0])
        except: pass
            
    agent2_files = glob.glob(os.path.join(model_save_path, "agent2_rainbow_episode_*.pth"))
    if agent2_files:
        agent2_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        try:
            agent2.load_model(agent2_files[0])
        except: pass
    
    print(f"\nProfiling {num_episodes} episodes with {NUM_ENVS} parallel environments...")
    print("-"*60)
    
    global_step = 0
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Generate placements
        profiler.start("setup_placements")
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
        profiler.stop("setup_placements")
        
        # Reset environments
        profiler.start("env_reset")
        game_states, _, _, _, valid_moves = parallel_env.reset(p1_placements, p2_placements)
        profiler.stop("env_reset")
        
        # Reset agents
        profiler.start("agent_reset_pbs")
        agent1.reset_pbs()
        agent2.reset_pbs()
        profiler.stop("agent_reset_pbs")
        
        active_envs = np.ones(NUM_ENVS, dtype=bool)
        step_count = 0
        
        while np.any(active_envs) and step_count < 1000:  # Match train_dqn.py 1000 step limit
            # P1 Actions
            profiler.start("p1_act_batch")
            actions_p1 = agent1.act_batch(
                [gs.board for gs in game_states],
                valid_moves,
                game_states,
                env_indices=list(range(NUM_ENVS))
            )
            profiler.stop("p1_act_batch")
            
            # Step P1
            profiler.start("env_step_p1")
            next_states_p1, rewards_p1, dones_p1, infos_p1, valid_moves = parallel_env.step(actions_p1)
            profiler.stop("env_step_p1")
            
            # Update P2's PBS
            profiler.start("pbs_update_p2")
            agent2.update_pbs_batch(actions_p1, game_states, acting_player=1)
            profiler.stop("pbs_update_p2")
            
            # Store P1 Experience
            profiler.start("remember_p1")
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    agent1.remember(game_states[i].board, actions_p1[i], rewards_p1[i], next_states_p1[i].board, dones_p1[i])
                    if dones_p1[i]:
                        active_envs[i] = False
            profiler.stop("remember_p1")
            
            game_states = next_states_p1
            
            if not np.any(active_envs):
                break
            
            # P2 Actions
            profiler.start("p2_act_batch")
            actions_p2 = agent2.act_batch(
                [gs.board for gs in game_states],
                valid_moves,
                game_states,
                env_indices=list(range(NUM_ENVS))
            )
            profiler.stop("p2_act_batch")
            
            # Step P2
            profiler.start("env_step_p2")
            next_states_p2, rewards_p2, dones_p2, infos_p2, valid_moves = parallel_env.step(actions_p2)
            profiler.stop("env_step_p2")
            
            # Update P1's PBS
            profiler.start("pbs_update_p1")
            agent1.update_pbs_batch(actions_p2, game_states, acting_player=-1)
            profiler.stop("pbs_update_p1")
            
            # Store P2 Experience
            profiler.start("remember_p2")
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    agent2.remember(game_states[i].board, actions_p2[i], rewards_p2[i], next_states_p2[i].board, dones_p2[i])
                    if dones_p2[i]:
                        active_envs[i] = False
            profiler.stop("remember_p2")
            
            game_states = next_states_p2
            step_count += 1
            global_step += 1
            
            if global_step % REPLAY_UPDATE_INTERVAL == 0:
                profiler.start("replay_p1")
                agent1.replay()
                profiler.stop("replay_p1")
                # Note: Agent 2 does NOT train in train_dqn.py (opponent only)
            
            if global_step % TARGET_UPDATE_INTERVAL == 0:
                profiler.start("target_update")
                agent1.update_target_network()
                # Agent 2 doesn't train, so no target update
                profiler.stop("target_update")
        
        print(f"  Steps: {step_count}")
        
        # Train PBS
        profiler.start("train_pbs")
        agent1.train_pbs(epochs=5)
        agent2.train_pbs(epochs=5)
        profiler.stop("train_pbs")
        
        # New metrics tracking (matching train_dqn.py)
        profiler.start("metrics_tracking")
        # Q-value and entropy tracking
        avg_q = agent1.get_average_q() if hasattr(agent1, 'get_average_q') else 0.0
        entropy = agent1.get_exploration_entropy() if hasattr(agent1, 'get_exploration_entropy') else 0.0
        profiler.stop("metrics_tracking")
    
    # Report results
    profiler.report()
    
    # Cleanup
    parallel_env.close()


if __name__ == "__main__":
    profile_training(num_episodes=2)
