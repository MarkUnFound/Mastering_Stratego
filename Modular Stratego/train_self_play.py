import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from collections import deque
from typing import List, Tuple
import os

from environment import StrategoEnvironment
from hybrid_agent import HybridAgent
from dqn_evaluator import DQNEvaluator
from probabilistic_belief_state import ProbabilisticBeliefState

class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data # List of (state_seq, outcome)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SelfPlayTrainer:
    def __init__(self, 
                 save_dir="models",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 lr=1e-4,
                 batch_size=32,
                 buffer_size=10000):
        
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize Environment
        self.env = StrategoEnvironment(device=device)
        
        # Initialize Evaluator (Shared)
        self.evaluator = DQNEvaluator(device=device, use_history=True)
        self.optimizer = optim.Adam(self.evaluator.network.parameters(), lr=lr)
        
        # Initialize Agents
        # They share the same evaluator instance (and thus weights)
        self.agent_p1 = HybridAgent(player_id=1, device=device)
        self.agent_p1.evaluator = self.evaluator # Force share
        
        self.agent_p2 = HybridAgent(player_id=-1, device=device)
        self.agent_p2.evaluator = self.evaluator # Force share
        
        # Replay Buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Metrics
        self.games_played = 0
        self.loss_history = []

    def collect_self_play_data(self, num_games=1, temperature=1.0):
        """
        Play games and collect data.
        Data format: (state_sequence, final_outcome)
        """
        self.evaluator.network.eval()
        new_data = []
        
        for _ in range(num_games):
            state = self.env.reset()
            game_over = False
            
            # Track history for AAREN
            # We need to store the full sequence of states for training
            # For efficiency, we might store just the game trajectory and process it later
            trajectory = []
            
            while not game_over:
                # Get current player agent
                current_agent = self.agent_p1 if state.current_player == 1 else self.agent_p2
                valid_moves = self.env.get_valid_moves()
                
                if not valid_moves:
                    game_over = True
                    winner = -state.current_player
                    break
                
                # Select Action
                # In training, we might want to add noise/temperature to the agent's decision
                action = current_agent.act(state, valid_moves) # TODO: Add temperature support to act()
                
                # Store state (we need to convert it to tensor for storage/training)
                # Ideally, HybridAgent.act() already does this conversion. 
                # We should probably expose the tensor generation or do it here.
                # For now, let's assume we can reconstruct it or store the raw state.
                # Storing raw state is cheaper but requires conversion during training.
                trajectory.append((state, state.current_player))
                
                # Step
                state, reward, game_over, info = self.env.step(action)
                
                if game_over:
                    winner = info['winner']
            
            # Process Trajectory
            # Outcome z: +1 for winner, -1 for loser
            # We assign z to each state in the trajectory based on the player whose turn it was
            # V(s) should predict the probability of CURRENT player winning
            
            # If winner is 1:
            # States where player 1 moved -> Target = +1
            # States where player -1 moved -> Target = -1 (from their perspective)
            
            # Wait, V(s) usually represents value for the player whose turn it is.
            # If it's P1's turn and P1 wins, V(s) -> 1.
            # If it's P2's turn and P2 loses (P1 wins), V(s) -> -1.
            
            for i, (game_state, player) in enumerate(trajectory):
                if winner == 0: # Draw
                    target = 0.0
                elif winner == player:
                    target = 1.0
                else:
                    target = -1.0
                
                # We need to construct the tensor here to store it
                # This is expensive. In production, use a parallel data loader.
                # For this demo, we'll do it here.
                tensor = self.state_to_tensor(game_state)
                new_data.append((tensor, target))
                
            self.games_played += 1
            print(f"Game {self.games_played} finished. Winner: {winner}. Steps: {len(trajectory)}")
            
        self.replay_buffer.extend(new_data)

    def train_step(self):
        """
        Train the network on a batch from replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
            
        self.evaluator.network.train()
        
        # Sample Batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, targets = zip(*batch)
        
        # Convert to tensors
        # states: List of (41, 10, 10) tensors
        # targets: List of floats
        
        state_batch = torch.stack(states).to(self.device)
        target_batch = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Forward Pass
        # Note: We are training on single states here, not sequences, for simplicity in this first pass.
        # To train AAREN properly, we need to sample SEQUENCES (trajectories) from the buffer.
        # Let's adjust the data collection to store trajectories if we want to train history.
        
        # For now, let's assume we train the CNN part primarily.
        # If using HistoryAware, we pass seq_len=1
        
        if self.evaluator.use_history:
            # Add seq dim: (Batch, 1, 41, 10, 10)
            state_batch = state_batch.unsqueeze(1)
            
        values, _ = self.evaluator.network(state_batch)
        
        # Loss (MSE)
        loss = F.mse_loss(values, target_batch)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        return loss.item()

    def state_to_tensor(self, state):
        # Helper to convert GameState to Tensor
        # This logic should ideally be shared with KLUSSSolver
        # For now, placeholder implementation
        return torch.zeros((41, 10, 10), dtype=torch.float32)

    def run_training_loop(self, total_games=1000):
        for i in range(total_games):
            self.collect_self_play_data(num_games=1)
            self.train_step()
            
            if i % 100 == 0:
                self.save_model(f"checkpoint_{i}.pt")
                print(f"Saved checkpoint {i}")

    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.evaluator.network.state_dict(), path)

if __name__ == "__main__":
    trainer = SelfPlayTrainer()
    trainer.run_training_loop(total_games=10) # Short run for testing
