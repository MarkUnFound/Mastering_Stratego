
import time
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parallel_environment import ParallelStrategoEnvironment
from drqn_agent import DRQNAgent
from heuristic_setup import HeuristicSetupAgent
from training_config import *

def profile_training():
    print("Starting Performance Profiling...")
    
    # Configuration for profiling
    PROFILING_EPISODES = 5
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Environment
    print("Initializing Environment...")
    env = ParallelStrategoEnvironment(num_envs=NUM_ENVS)
    
    # Initialize Agents
    print("Initializing Agents...")
    agent1 = DRQNAgent(player_id=1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    agent2 = DRQNAgent(player_id=-1, device=device, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_envs=NUM_ENVS, buffer_size=MEMORY_SIZE)
    
    setup_agent1 = HeuristicSetupAgent(player_id=1, device=device)
    setup_agent2 = HeuristicSetupAgent(player_id=-1, device=device)
    
    # Timing stats
    stats = {
        'total_step_time': [],
        'act_batch_time': [],
        'history_update_time': [],
        'env_step_time': [],
        'remember_time': [],
        'training_time': [],
        'reset_time': []
    }
    
    # Game length stats
    episode_lengths = []
    current_lengths = [0] * NUM_ENVS
    
    # Initial Reset
    print("Performing initial reset...")

    states_tuple, rewards, dones, infos, valid_moves_tuple = env.reset()
    states = list(states_tuple)
    valid_moves = list(valid_moves_tuple)
    
    pending_resets = [False] * NUM_ENVS
    completed_episodes = 0
    total_steps = 0
    
    print(f"Profiling for {PROFILING_EPISODES} episodes...")
    
    pbar = tqdm(total=PROFILING_EPISODES, desc="Profiling Episodes")
    
    start_time = time.perf_counter()
    
    while completed_episodes < PROFILING_EPISODES:
        step_start = time.perf_counter()
        
        # 1. Action Selection
        t0 = time.perf_counter()
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
        stats['act_batch_time'].append(time.perf_counter() - t0)
        
        # 2. History Updates (AAREN)
        t0 = time.perf_counter()
        if agent1_indices:
            agent2.update_history_batch([actions_list[i] for i in agent1_indices], [states[i] for i in agent1_indices], acting_player=1)
        if agent2_indices:
            agent1.update_history_batch([actions_list[i] for i in agent2_indices], [states[i] for i in agent2_indices], acting_player=-1)
        stats['history_update_time'].append(time.perf_counter() - t0)
        
        # 3. Prepare Commands & Environment Step
        t0 = time.perf_counter()
        commands = []
        for i in range(NUM_ENVS):
            if pending_resets[i]:
                # Reset logic
                pieces = env.call_method('get_all_pieces')
                p1_place = setup_agent1.place_pieces(pieces, env.call_method('get_valid_placement_positions', 1))
                p2_place = setup_agent2.place_pieces(pieces, env.call_method('get_valid_placement_positions', -1))
                commands.append(('reset', {'p1_placement': p1_place, 'p2_placement': p2_place}))
            else:
                commands.append(actions_list[i])
        
        next_states_tuple, step_rewards, step_dones, step_infos, next_valid_moves_tuple = env.step(commands)
        next_states = list(next_states_tuple)
        next_valid_moves = list(next_valid_moves_tuple)
        next_valid_moves = list(next_valid_moves_tuple)
        stats['env_step_time'].append(time.perf_counter() - t0)
        
        # Update lengths
        for i in range(NUM_ENVS):
            current_lengths[i] += 1

        
        # 4. State Update & Memory
        t0 = time.perf_counter()
        for i in range(NUM_ENVS):
            if pending_resets[i]:
                states[i] = next_states[i]
                valid_moves[i] = next_valid_moves[i]
                pending_resets[i] = False
                continue
            
            action = actions_list[i]
            reward = step_rewards[i].item()
            done = step_dones[i].item()
            
            current_agent = agent1 if states[i].current_player == 1 else agent2
            state_tensor = current_agent.get_state_representation(states[i])
            next_state_tensor = current_agent.get_state_representation(next_states[i])
            current_agent.remember(state_tensor, current_agent._move_to_action_index(action), reward, next_state_tensor, done)
            
            states[i] = next_states[i]
            valid_moves[i] = next_valid_moves[i]
            
            if done:
                episode_lengths.append(current_lengths[i])
                current_lengths[i] = 0
                pending_resets[i] = True
                completed_episodes += 1
                pbar.update(1)

                agent1.reset_history()
                agent2.reset_history()
                
                # AAREN trains end-to-end with agent, no separate training needed
                t_hist = time.perf_counter()
                stats['history_update_time'][-1] += (time.perf_counter() - t_hist)

        stats['remember_time'].append(time.perf_counter() - t0)
        
        # 5. Training
        t0 = time.perf_counter()
        if total_steps % REPLAY_UPDATE_INTERVAL == 0:
            for _ in range(REPLAY_UPDATES_PER_STEP):
                agent1.replay()
                agent2.replay()
        stats['training_time'].append(time.perf_counter() - t0)
        
        total_steps += 1
        stats['total_step_time'].append(time.perf_counter() - step_start)
        
    total_time = time.perf_counter() - start_time
    pbar.close()
    env.close()
    
    print("\n" + "="*40)
    print("PROFILING RESULTS")
    print("="*40)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Steps: {total_steps}")
    print(f"Avg Time per Step: {np.mean(stats['total_step_time']):.4f}s")
    print("-" * 30)
    print(f"Avg Act Batch Time: {np.mean(stats['act_batch_time']):.4f}s")
    print(f"Avg History Update Time: {np.mean(stats['history_update_time']):.4f}s")
    print(f"Avg Env Step Time: {np.mean(stats['env_step_time']):.4f}s")
    print(f"Avg Remember Time: {np.mean(stats['remember_time']):.4f}s")
    print(f"Avg Training Time: {np.mean(stats['training_time']):.4f}s")
    print("-" * 30)
    if episode_lengths:
        print(f"Avg Game Length: {np.mean(episode_lengths):.1f} steps")
        print(f"Min Game Length: {np.min(episode_lengths)} steps")
        print(f"Max Game Length: {np.max(episode_lengths)} steps")
        
        # Calculate effective time per episode
        # Total time / number of episodes completed
        # This is rough because some episodes might be partial
        avg_step_time = np.mean(stats['total_step_time'])
        avg_episode_time = avg_step_time * (np.mean(episode_lengths) / NUM_ENVS)
        print(f"Est. Time per Episode: {avg_episode_time:.2f}s (assuming {NUM_ENVS} parallel envs)")

    print("="*40)

if __name__ == "__main__":
    profile_training()
