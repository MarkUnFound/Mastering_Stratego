import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from collections import deque
from typing import List, Tuple
import os
import time
import copy

from environment import StrategoEnvironment
from hybrid_agent import HybridAgent
from dqn_evaluator import DQNEvaluator

# Set start method to spawn for CUDA compatibility (though we use CPU for workers)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class SelfPlayWorker(mp.Process):
    def __init__(self, rank, shared_model, result_queue, league, device='cpu'):
        super(SelfPlayWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.result_queue = result_queue
        self.league = league # Shared list or manager list
        self.device = device # Worker device (usually CPU)
        
    def run(self):
        # Initialize Environment (Independent per worker)
        env = StrategoEnvironment(device=self.device)
        
        # Initialize Evaluator using Shared Model
        # We create a local wrapper but point to shared weights
        evaluator = DQNEvaluator(device=self.device, use_history=True)
        # We don't load state dict, we just use the shared network in the forward pass?
        # Or we copy weights? 
        # For A3C style, we use the shared model directly.
        # But DQNEvaluator wraps the network.
        evaluator.network = self.shared_model
        
        # Initialize Agents
        agent_p1 = HybridAgent(player_id=1, device=self.device)
        agent_p1.evaluator = evaluator
        
        agent_p2 = HybridAgent(player_id=-1, device=self.device)
        agent_p2.evaluator = evaluator
        
        while True:
            # Sync with latest weights? 
            # If shared_model is in shared_memory(), it's automatic for gradients, 
            # but for weights updated by optimizer in main process, we might need to reload?
            # Actually, if the main process updates the weights in-place on the shared model, it's visible.
            # But usually main process trains a GPU model.
            
            # Let's assume Main updates Shared Model periodically.
            
            state = env.reset()
            game_over = False
            trajectory = []
            
            # League Logic (Local to worker)
            is_league_game = False
            if len(self.league) > 0 and random.random() < 0.2:
                opponent_path = random.choice(self.league)
                try:
                    # Load opponent (this is slow, maybe cache?)
                    opp_eval = DQNEvaluator(model_path=opponent_path, device=self.device, use_history=True)
                    agent_p2.evaluator = opp_eval
                    is_league_game = True
                except:
                    agent_p2.evaluator = evaluator # Fallback
            else:
                agent_p2.evaluator = evaluator
            
            while not game_over:
                current_agent = agent_p1 if state.current_player == 1 else agent_p2
                valid_moves = env.get_valid_moves()
                
                if not valid_moves:
                    game_over = True
                    winner = -state.current_player
                    break
                
                # Action
                action = current_agent.act(state, valid_moves)
                trajectory.append((state, state.current_player))
                state, reward, game_over, info = env.step(action)
                
                if game_over:
                    winner = info['winner']
            
            # Process Data
            data_points = []
            for i, (game_state, player) in enumerate(trajectory):
                if winner == 0: target = 0.0
                elif winner == player: target = 1.0
                else: target = -1.0
                
                # Convert to tensor (CPU)
                # Use the same encoding as KLUSS/DQN via KLUSSSolver.state_to_tensor
                tensor = agent_p1.solver.state_to_tensor(game_state).cpu()

                data_points.append((tensor, target))
            
            # Send to Main
            self.result_queue.put(data_points)
            
            # Optional: Sleep to prevent CPU hogging if queue is full?
            # Queue handles blocking.

class ParallelSelfPlayTrainer:
    def __init__(self, num_workers=4, save_dir="models", lr=1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Main Training Model (GPU)
        self.train_evaluator = DQNEvaluator(device=self.device, use_history=True)
        self.train_evaluator.network.train()
        self.optimizer = optim.Adam(self.train_evaluator.network.parameters(), lr=lr)
        
        # 2. Shared Inference Model (CPU)
        # Workers will use this.
        self.shared_evaluator = DQNEvaluator(device='cpu', use_history=True)
        self.shared_evaluator.network.share_memory() # Magic for multiprocessing
        
        # Sync initially
        self.shared_evaluator.network.load_state_dict(self.train_evaluator.network.state_dict())
        
        # League (Managed List)
        self.manager = mp.Manager()
        self.league = self.manager.list()
        
        # Workers
        self.num_workers = num_workers
        self.result_queue = mp.Queue(maxsize=100)
        self.workers = []
        
        # Replay Buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.games_played = 0
        self.loss_history = []
        self.last_loss = None
        self.last_value_mean = None
        self.last_value_std = None

    def start_workers(self):
        for i in range(self.num_workers):
            w = SelfPlayWorker(i, self.shared_evaluator.network, self.result_queue, self.league, device='cpu')
            w.start()
            self.workers.append(w)
        print(f"Started {self.num_workers} worker processes.")

    def update_shared_model(self):
        # Copy weights from GPU Train model to CPU Shared model
        self.shared_evaluator.network.load_state_dict(self.train_evaluator.network.state_dict())

    def update_league(self):
        filename = f"league_model_{self.games_played}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save(self.train_evaluator.network.state_dict(), path)
        self.league.append(path)
        print(f"League Updated. Size: {len(self.league)}")

    def record_game(self, game_id):
        """
        Play a game and record it with visualization.
        """
        from visualization import StrategoVisualizer
        import pygame
        
        print(f"Recording Game {game_id}...")
        env = StrategoEnvironment(device=self.device)
        visualizer = StrategoVisualizer()
        
        # Agents
        agent_p1 = HybridAgent(player_id=1, device=self.device)
        agent_p1.evaluator = self.train_evaluator
        
        agent_p2 = HybridAgent(player_id=-1, device=self.device)
        agent_p2.evaluator = self.train_evaluator
        
        state = env.reset()
        game_over = False
        frame_count = 0
        save_dir = os.path.join(self.save_dir, f"game_{game_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        while not game_over:
            current_agent = agent_p1 if state.current_player == 1 else agent_p2
            valid_moves = env.get_valid_moves()
            
            if not valid_moves:
                break
                
            # Get Top Moves for Visualization
            top_moves = current_agent.get_top_moves(state, valid_moves, n=3)
            
            # Render and Save
            save_path = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
            visualizer.render_state(state, q_values=top_moves, save_path=save_path)
            frame_count += 1
            
            # Act
            action = current_agent.act(state, valid_moves)
            state, reward, game_over, info = env.step(action)
            
        print(f"Game recorded to {save_dir}")
        pygame.quit()

    def debug_probe_state(self, num_moves=5):
        """Probe a fresh starting state and print top moves with DQN values."""
        env = StrategoEnvironment(device=self.device)
        state = env.reset()

        agent = HybridAgent(player_id=state.current_player, device=self.device)
        agent.evaluator = self.train_evaluator

        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("No valid moves from initial state.")
            return

        move_values = agent.analyze_state(state, valid_moves)
        if not move_values:
            print("analyze_state returned no moves.")
            return

        move_values.sort(key=lambda x: x[1], reverse=True)
        print(f"Top {min(num_moves, len(move_values))} moves from initial state:")
        for (move, val) in move_values[:num_moves]:
            print(f"  Move {move}: value={float(val):.3f}")

    def train_loop(self, total_games=1000):
        self.start_workers()
        
        try:
            while self.games_played < total_games:
                # 1. Collect Data
                if not self.result_queue.empty():
                    new_data = self.result_queue.get()
                    self.replay_buffer.extend(new_data)
                    self.games_played += 1
                    
                    if self.games_played % 10 == 0:
                        if self.loss_history:
                            window = self.loss_history[-100:]
                            avg_loss = sum(window) / len(window)
                            if self.last_value_mean is not None and self.last_value_std is not None:
                                print(
                                    f"Games: {self.games_played}, Buffer: {len(self.replay_buffer)}, "
                                    f"LastLoss: {self.last_loss:.4f}, AvgLoss(100): {avg_loss:.4f}, "
                                    f"ValueMean: {self.last_value_mean:.3f}, ValueStd: {self.last_value_std:.3f}"
                                )
                            else:
                                print(
                                    f"Games: {self.games_played}, Buffer: {len(self.replay_buffer)}, "
                                    f"LastLoss: {self.last_loss:.4f}, AvgLoss(100): {avg_loss:.4f}"
                                )
                        else:
                            print(f"Games: {self.games_played}, Buffer: {len(self.replay_buffer)}")
                    
                    if self.games_played % 50 == 0:
                        self.update_league()
                        
                    if self.games_played % 100 == 0:
                        self.record_game(self.games_played)
                
                # 2. Train
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.train_step()
                    
                    if self.games_played % 10 == 0:
                        self.update_shared_model()
                
                if self.result_queue.empty() and len(self.replay_buffer) < self.batch_size:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            for w in self.workers:
                w.terminate()
                w.join()

    def train_step(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, targets = zip(*batch)
        
        state_batch = torch.stack(states).to(self.device)
        target_batch = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        if self.train_evaluator.use_history:
            state_batch = state_batch.unsqueeze(1)
            
        values, _ = self.train_evaluator.network(state_batch)
        loss = F.mse_loss(values, target_batch)
        
        value_mean = values.mean().item()
        value_std = values.std().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.last_loss = loss_value
        self.last_value_mean = value_mean
        self.last_value_std = value_std
        
        return loss_value

if __name__ == "__main__":
    # Run with 4 workers by default
    trainer = ParallelSelfPlayTrainer(num_workers=4)
    trainer.train_loop(total_games=100)

