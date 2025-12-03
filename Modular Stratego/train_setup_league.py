"""
Setup League Training Script
Runs the Setup League evolution process independently from the main DQN training loop.
"""

import torch
import os
import sys
import time
import glob
import traceback
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import StrategoEnvironment
from drqn_agent import DRQNAgent
from setup_agent import SetupAgent
from setup_league import SetupLeague
from training_config import *

def train_setup_league(model_save_path: str = "dqn_models"):
    """
    Run the Setup League evolution loop.
    
    Args:
        model_save_path: Directory to save/load models
    """
    print("🏆 Starting Setup League Training Service")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment (local, for tournament)
    env = StrategoEnvironment(device=device)
    
    # Initialize Evaluator Agent (DQN)
    # We use a single agent to evaluate both sides
    evaluator_agent = DRQNAgent(player_id=1, device=device, lr=0.0001, batch_size=BATCH_SIZE)
    
    # Initialize Setup League
    print("Initializing Setup League...")
    setup_league = SetupLeague(population_size=4, device=device)
    
    # Try to load existing setup agents to seed the league
    try:
        setup_agent1_path = os.path.join(model_save_path, "setup_agent1_final.pth")
        if os.path.exists(setup_agent1_path):
            checkpoint = torch.load(setup_agent1_path, map_location=device)
            setup_league.population[0].q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print(f"✅ Seeded League Bot 0 from {setup_agent1_path}")
    except Exception as e:
        print(f"⚠️ Could not seed league: {e}")
        
    generation = 0
    history = []
    history_file = os.path.join(model_save_path, "setup_league_history.json")
    
    # Load existing history if available
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            generation = len(history)
            print(f"📊 Loaded setup league history: {generation} generations")
        except Exception as e:
            print(f"⚠️ Could not load history: {e}")
    
    while True:
        try:
            print(f"\n🔄 Starting League Generation {generation}...")
            
            # 1. Load latest DQN model for evaluation
            # We want the league to optimize against the CURRENT best strategy
            load_latest_dqn_model(evaluator_agent, model_save_path)
            
            # 2. Run Evolution
            # Run a few generations of evolution
            setup_league.run_evolution(env, evaluator_agent, generations=1, games_per_matchup=2)
            
            # 3. Metrics & History
            scores = list(setup_league.scores.values())
            stats = {
                "generation": generation,
                "best_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "best_agent": setup_league.get_best_agent().name
            }
            history.append(stats)
            
            # Save History
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
                
            # Plot Progress
            plot_league_progress(history, model_save_path)
            
            # 4. Save Best Agent
            best_agent = setup_league.get_best_agent()
            save_best_setup_agent(best_agent, model_save_path)
            
            generation += 1
            
            # Optional: Sleep briefly to allow DQN to train a bit? 
            # No, we want setup to evolve as fast as possible.
            
        except KeyboardInterrupt:
            print("\n🛑 Setup League interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in Setup League loop: {e}")
            traceback.print_exc()
            time.sleep(10) # Wait before retrying

def load_latest_dqn_model(agent: DRQNAgent, model_save_path: str):
    """Load the most recent DQN model to use as evaluator."""
    try:
        # Look for agent1 models (assuming agent1 and agent2 are similar/symmetric enough for eval)
        files = glob.glob(os.path.join(model_save_path, "agent1_dqn_episode_*.pth"))
        if not files:
            # Check for final
            final_path = os.path.join(model_save_path, "agent1_dqn_final.pth")
            if os.path.exists(final_path):
                files = [final_path]
        
        if files:
            # Sort by modification time to get the absolute latest file written
            latest_file = max(files, key=os.path.getmtime)
            
            # Load it
            agent.load_model(latest_file)
            print(f"✅ Loaded latest evaluator model: {os.path.basename(latest_file)}")
        else:
            print("⚠️ No DQN models found. Using random evaluator.")
            
    except Exception as e:
        print(f"⚠️ Failed to load evaluator model: {e}")

def save_best_setup_agent(agent: SetupAgent, model_save_path: str):
    """Save the best setup agent for the main training loop to pick up."""
    try:
        os.makedirs(model_save_path, exist_ok=True)
        
        # Save as 'best' which train_dqn.py will look for
        save_path = os.path.join(model_save_path, "setup_agent_best.pth")
        
        # We save it in a format that SetupAgent.load_model expects
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }, save_path)
        
        print(f"💾 Saved best setup agent to {save_path}")
        
    except Exception as e:
        print(f"❌ Failed to save best setup agent: {e}")

def plot_league_progress(history, save_path):
    """Generate a plot of the league's progress."""
    try:
        generations = [h['generation'] for h in history]
        best_scores = [h['best_score'] for h in history]
        avg_scores = [h['avg_score'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_scores, label='Best Score', color='green')
        plt.plot(generations, avg_scores, label='Avg Score', color='blue', linestyle='--')
        
        plt.title('Setup League Progress')
        plt.xlabel('Generation')
        plt.ylabel('Score (Wins)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_path, "setup_league_progress.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed to plot league progress: {e}")

if __name__ == "__main__":
    train_setup_league()
