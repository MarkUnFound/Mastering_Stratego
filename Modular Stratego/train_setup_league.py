"""
Setup Optimization League for Stratego
Trains a population of SetupAgents to evolve robust starting formations.
"""

import torch
import numpy as np
import random
import os
import sys
import time
import copy
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import StrategoEnvironment
from dqn_agent import DQNAgent
from setup_agent import SetupAgent
from piece import PieceType
from train_dqn import calculate_setup_agent_reward

# Configuration
POPULATION_SIZE = 4  # Number of setup agents in the league
NUM_GENERATIONS = 100
GAMES_PER_MATCHUP = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "setup_league_models"

def main():
    print(f"🚀 Starting Setup Optimization League on {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Population
    population = []
    for i in range(POPULATION_SIZE):
        agent = SetupAgent(player_id=1, device=DEVICE, lr=0.0001, epsilon=0.5) # Shared ID, will flip when playing
        agent.name = f"SetupBot_{i}"
        population.append(agent)
        print(f"Created {agent.name}")

    # 2. Initialize Evaluator (Fixed Gameplay Agent)
    # We use a single DQNAgent to play BOTH sides to isolate setup quality
    gameplay_agent = DQNAgent(player_id=1, device=DEVICE)
    # Try to load a pre-trained model if available, otherwise use random/untrained
    model_path = "models/dqn_agent_1.pth"
    if os.path.exists(model_path):
        try:
            gameplay_agent.load_model(model_path)
            print("Loaded pre-trained gameplay agent.")
        except:
            print("Could not load pre-trained model. Using untrained agent.")
    else:
        print("No pre-trained model found. Using untrained agent.")
    
    # Ensure gameplay agent is in eval mode (no learning during setup training)
    gameplay_agent.epsilon = 0.1 

    # 3. Training Loop
    env = StrategoEnvironment()
    
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{NUM_GENERATIONS} ---")
        scores = {agent.name: 0 for agent in population}
        
        # Round Robin Tournament
        for i in range(POPULATION_SIZE):
            for j in range(i + 1, POPULATION_SIZE):
                agent_a = population[i]
                agent_b = population[j]
                
                # Play match
                wins_a = 0
                wins_b = 0
                
                for game_idx in range(GAMES_PER_MATCHUP):
                    # Swap sides halfway
                    if game_idx < GAMES_PER_MATCHUP // 2:
                        p1_setup, p2_setup = agent_a, agent_b
                    else:
                        p1_setup, p2_setup = agent_b, agent_a
                        
                    # Generate Setups
                    # Agent A is always Player 1 logic internally, but we map to board
                    pieces_p1 = p1_setup.place_pieces(env.get_all_pieces(), env.get_valid_placement_positions(1))
                    pieces_p2 = p2_setup.place_pieces(env.get_all_pieces(), env.get_valid_placement_positions(-1))
                    
                    # Reset Env
                    state = env.reset(pieces_p1, pieces_p2)
                    done = False
                    move_count = 0
                    
                    # Play out game
                    while not done and move_count < 200:
                        valid_moves = env.get_valid_moves(env.current_player)
                        if not valid_moves:
                            break
                            
                        # Gameplay agent plays for BOTH sides
                        # We just flip the state perspective for Player -1
                        gameplay_agent.player_id = env.current_player
                        action = gameplay_agent.act(state, valid_moves)
                        
                        state, reward, done, _ = env.step(action)
                        move_count += 1
                        
                    # Determine Winner
                    winner = env.winner
                    
                    # Calculate Rewards & Train Setup Agents
                    # P1 Setup Agent
                    r1 = calculate_setup_agent_reward(pieces_p1, 1, winner, move_count)
                    p1_setup.finish_episode(r1)
                    p1_setup.replay()
                    
                    # P2 Setup Agent
                    r2 = calculate_setup_agent_reward(pieces_p2, -1, winner, move_count)
                    p2_setup.finish_episode(r2)
                    p2_setup.replay()
                    
                    # Update Scores
                    if winner == 1:
                        if p1_setup == agent_a: wins_a += 1
                        else: wins_b += 1
                    elif winner == -1:
                        if p2_setup == agent_a: wins_a += 1
                        else: wins_b += 1
                        
                scores[agent_a.name] += wins_a
                scores[agent_b.name] += wins_b
                
        # End of Generation
        print("Scores:", scores)
        
        # Save best agent
        best_agent_name = max(scores, key=scores.get)
        best_agent = next(a for a in population if a.name == best_agent_name)
        best_agent.save_model(os.path.join(SAVE_DIR, f"best_setup_gen_{generation}.pth"))
        print(f"Saved best agent: {best_agent_name}")

if __name__ == "__main__":
    main()
