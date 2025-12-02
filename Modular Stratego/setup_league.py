import torch
import numpy as np
import random
import os
import copy
from typing import List, Dict
from setup_agent import SetupAgent
from dqn_agent import DQNAgent
from setup_evaluation import calculate_setup_agent_reward
from environment import StrategoEnvironment

class SetupLeague:
    """
    Manages a league of SetupAgents to evolve better starting formations.
    """
    def __init__(self, population_size: int = 4, device: str = "cuda", 
                 lr: float = 0.0001, epsilon: float = 0.5):
        self.population_size = population_size
        self.device = device
        self.population: List[SetupAgent] = []
        self.scores: Dict[str, int] = {}
        
        # Initialize population
        for i in range(population_size):
            agent = SetupAgent(player_id=1, device=device, lr=lr, epsilon=epsilon)
            agent.name = f"SetupBot_{i}"
            self.population.append(agent)
            self.scores[agent.name] = 0
            
    def run_evolution(self, env, gameplay_agent: DQNAgent, generations: int = 1, games_per_matchup: int = 4):
        """
        Run evolution for a number of generations.
        
        Args:
            env: Ignored (we create a local environment for the tournament)
            gameplay_agent: The DQNAgent used to play out the games (evaluator)
            generations: Number of generations to run
            games_per_matchup: Number of games per pair of agents
        """
        print(f"🏆 Starting Setup League Evolution ({generations} generations)...")
        
        # Create a local environment for the tournament to avoid interference with the main parallel env
        # and to ensure we have a single-agent environment interface
        tournament_env = StrategoEnvironment(device=self.device)
        
        # Ensure gameplay agent is in eval mode
        was_training = gameplay_agent.q_network.training
        gameplay_agent.q_network.eval()
        original_epsilon = gameplay_agent.epsilon
        gameplay_agent.epsilon = 0.1 # Low exploration for evaluation
        
        for gen in range(generations):
            # Reset scores for this generation
            self.scores = {agent.name: 0 for agent in self.population}
            
            # Round Robin Tournament
            for i in range(self.population_size):
                for j in range(i + 1, self.population_size):
                    agent_a = self.population[i]
                    agent_b = self.population[j]
                    
                    self._play_matchup(tournament_env, gameplay_agent, agent_a, agent_b, games_per_matchup)
            
            # Log results
            best_name = max(self.scores, key=self.scores.get)
            print(f"   Gen {gen+1}: Best Agent = {best_name} (Score: {self.scores[best_name]})")
            
            # Evolution Step (Simple Elitism + Mutation)
            # Sort by score
            sorted_agents = sorted(self.population, key=lambda a: self.scores[a.name], reverse=True)
            
            # Keep top 50%, replace bottom 50% with mutated copies of top 50%
            half = self.population_size // 2
            for k in range(half):
                survivor = sorted_agents[k]
                loser = sorted_agents[k + half]
                
                # Copy weights from survivor to loser
                loser.q_network.load_state_dict(survivor.q_network.state_dict())
                loser.target_network.load_state_dict(survivor.target_network.state_dict())
                
                # Mutate loser (high exploration for next round)
                loser.epsilon = 0.5 
                
        # Restore gameplay agent state
        if was_training:
            gameplay_agent.q_network.train()
        gameplay_agent.epsilon = original_epsilon
        
    def _play_matchup(self, env, gameplay_agent, agent_a, agent_b, num_games):
        """Play a set of games between two setup agents."""
        wins_a = 0
        wins_b = 0
        
        for game_idx in range(num_games):
            # Swap sides
            if game_idx < num_games // 2:
                p1_setup, p2_setup = agent_a, agent_b
            else:
                p1_setup, p2_setup = agent_b, agent_a
            
            # Generate Setups
            try:
                pieces_p1 = p1_setup.place_pieces(env.get_all_pieces(), env.get_valid_placement_positions(1))
                pieces_p2 = p2_setup.place_pieces(env.get_all_pieces(), env.get_valid_placement_positions(-1))
                
                # Reset Env
                state = env.reset(pieces_p1, pieces_p2)
                done = False
                move_count = 0
                
                # Play out game
                while not done and move_count < 150: # Cap game length for speed
                    valid_moves = env.get_valid_moves()
                    if not valid_moves:
                        break
                    
                    # Gameplay agent plays for BOTH sides
                    gameplay_agent.player_id = env.current_player
                    action = gameplay_agent.act(state, valid_moves)
                    
                    state, reward, done, _ = env.step(action)
                    move_count += 1
                
                winner = env.winner
                
                # Train Setup Agents based on result
                r1 = calculate_setup_agent_reward(pieces_p1, 1, winner, move_count)
                p1_setup.finish_episode(r1)
                p1_setup.replay()
                
                r2 = calculate_setup_agent_reward(pieces_p2, -1, winner, move_count)
                p2_setup.finish_episode(r2)
                p2_setup.replay()
                
                # Update Match Score
                if winner == 1:
                    if p1_setup == agent_a: wins_a += 1
                    else: wins_b += 1
                elif winner == -1:
                    if p2_setup == agent_a: wins_a += 1
                    else: wins_b += 1
                    
            except Exception as e:
                print(f"Error in league match: {e}")
                continue
                
        self.scores[agent_a.name] += wins_a
        self.scores[agent_b.name] += wins_b

    def get_best_agent(self) -> SetupAgent:
        """Return the agent with the highest score."""
        best_name = max(self.scores, key=self.scores.get)
        return next(a for a in self.population if a.name == best_name)
