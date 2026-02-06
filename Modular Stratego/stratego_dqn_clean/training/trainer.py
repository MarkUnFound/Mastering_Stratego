"""
Trainer for Double DQN + AAREN
Clean training loop with integrated self-checks and gradient monitoring.

Features:
- Hard batch size ceiling (32)
- Target network hard update every 1000 steps
- Gradient clipping
- Self-check integration
- VRAM monitoring
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add parent path for environment import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .selfcheck import SelfCheckSuite, compute_action_entropy
from .replay_buffer import UniformReplayBuffer
from .config import TrainingConfig





class DoubleDQNTrainer:
    """
    Trainer for Double DQN + AAREN.
    
    Integrates:
    - Clean training loop
    - Uniform replay buffer
    - Self-check protocols
    - Gradient monitoring
    - Vectorized training support
    """
    
    def __init__(
        self,
        agent,  # DoubleDQNAgent
        env,    # StrategoEnvironment or VectorStrategoEnv
        config: TrainingConfig = None,
        log_dir: str = "./logs",
        device: torch.device = None
    ):
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        self.log_dir = log_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Replay buffer
        self.buffer = UniformReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            device=self.device,
            use_gpu_storage=True,
            use_float16=True
        )
        
        # Self-check suite
        self.selfcheck = None
        if self.config.SELFCHECK_ENABLED:
            # Get underlying model from agent
            model = agent.online_network if hasattr(agent, 'online_network') else agent
            aaren = agent.aaren if hasattr(agent, 'aaren') else None
            self.selfcheck = SelfCheckSuite(model=model, aaren=aaren, enabled=True)
        
        # Training state
        self.total_steps = 0
        self.optimization_steps = 0
        self.episodes = 0
        self.last_saved_episode = 0
        
        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        self.win_rates = []
        self.training_history = []
        
        # Vector env state
        self.is_vector_env = hasattr(env, 'num_envs')
        if self.is_vector_env:
            self.current_obs, self.valid_moves = env.reset()
            self.current_episode_rewards = torch.zeros(env.num_envs, device=self.device)
            self.current_episode_steps = torch.zeros(env.num_envs, device=self.device)
            print(f"[DoubleDQNTrainer] Vector env detected ({env.num_envs} envs)")
        
        print(f"[DoubleDQNTrainer] Initialized")
        print(f"  - Batch size: {self.config.BATCH_SIZE}")
        print(f"  - Buffer size: {self.config.BUFFER_SIZE:,}")
        print(f"  - Target update: every {self.config.TARGET_UPDATE_FREQ} steps")
        print(f"  - Self-checks: {'ENABLED' if self.config.SELFCHECK_ENABLED else 'DISABLED'}")
    
    def train(self, max_steps: int = None, max_episodes: int = None) -> Dict[str, Any]:
        """
        Run training loop.
        
        Args:
            max_steps: Maximum total steps
            max_episodes: Maximum episodes (alternative stopping)
            
        Returns:
            Training results dictionary
        """
        max_steps = max_steps or self.config.MAX_STEPS
        
        print("\n" + "=" * 60)
        print("STARTING DOUBLE DQN + AAREN TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            while self.total_steps < max_steps:
                # Run steps
                if self.is_vector_env:
                    finished_rewards, steps_run = self._run_vector_steps()
                    
                    # Log finished episodes
                    for r in finished_rewards:
                        self.episodes += 1
                        self.episode_rewards.append(r)
                else:
                    episode_reward, episode_steps = self._run_episode()
                    self.episodes += 1
                    self.episode_rewards.append(episode_reward)
                
                # Logging
                if self.total_steps % self.config.LOG_FREQ < (self.env.num_envs if self.is_vector_env else 100):
                    # Log approximately at frequency
                    self._log_progress()
                
                # Evaluation
                if self.total_steps % self.config.EVAL_FREQ < (self.env.num_envs if self.is_vector_env else 100):
                    win_rate = self._evaluate()
                    self.win_rates.append((self.total_steps, win_rate))
                    
                    # Update learning validation
                    if self.selfcheck:
                        self.selfcheck.learning_validation.record_evaluation(
                            self.total_steps, win_rate, self.config.EVAL_GAMES
                        )
                
                # Save checkpoint (Steps)
                if self.total_steps % self.config.SAVE_FREQ < (self.env.num_envs if self.is_vector_env else 100):
                    self._save_checkpoint()

                # Save checkpoint (Episodes)
                if self.episodes > 0 and (self.episodes // self.config.CHECKPOINT_EPISODE_FREQ) > (self.last_saved_episode // self.config.CHECKPOINT_EPISODE_FREQ):
                     print(f"\n[SAVE] Triggered by episode count ({self.episodes})")
                     self._save_checkpoint()
                     self.last_saved_episode = self.episodes
                
                # Check for halt conditions
                if self.selfcheck:
                    should_halt, reason = self.selfcheck.should_halt()
                    if should_halt:
                        print(f"\n[CRITICAL] {reason}")
                        break
                
                # Episode limit
                if max_episodes and self.episodes >= max_episodes:
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
        
        # Final summary
        elapsed = time.time() - start_time
        results = self._finalize_training(elapsed)
        
        return results

    def _run_vector_steps(self) -> Tuple[List[float], int]:
        """
        Run one step for all vector environments.
        
        Returns:
            finished_rewards: List of rewards for episodes that finished in this step
            steps_run: Number of environment steps taken
        """
        # Encode current obs -> (B, 15, 10, 10)
        # Note: current_obs is a tensor or numpy array from env.reset/step
        if isinstance(self.current_obs, torch.Tensor):
            board_tensor = self.current_obs.to(dtype=torch.float32, device=self.device)
        else:
            board_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device)
            
        state_tensor = self._encode_board(board_tensor)
        
        # Select actions
        actions, action_indices = self.agent.act_batch(
             state_tensor, 
             self.valid_moves, 
             greedy=False
        )
        
        # Step envs
        # next_states (N,15,10,10), rewards (N,), dones (N,), infos (List), valid_moves (List)
        next_states, rewards, dones, infos, new_valid_moves = self.env.step(actions)
        
        # Accumulate rewards/steps
        self.current_episode_rewards += rewards
        self.current_episode_steps += 1
        
        finished_rewards = []
        
        # Process experiences
        num_envs = self.env.num_envs
        
        # We need to process each environment to handle terminal states correctly
        for i in range(num_envs):
             real_next_state = next_states[i] # This is RESET state if done
             
             if dones[i]:
                 # Episode finished
                 finished_rewards.append(self.current_episode_rewards[i].item())
                 self.current_episode_rewards[i] = 0
                 self.current_episode_steps[i] = 0
                 
                 # Get terminal state for buffer (the state BEFORE reset)
                 term_obs = infos[i]['terminal_observation']
                 if isinstance(term_obs, np.ndarray):
                     real_next_state = torch.as_tensor(term_obs, dtype=torch.float32, device=self.device)
                 else:
                     real_next_state = term_obs.to(dtype=torch.float32, device=self.device)
             
             # Encode next state
             # Ensure real_next_state is tensor
             if not isinstance(real_next_state, torch.Tensor):
                  real_next_state = torch.as_tensor(real_next_state, dtype=torch.float32, device=self.device)
             else:
                  real_next_state = real_next_state.to(dtype=torch.float32, device=self.device)
                  
             real_next_state_15ch = self._encode_board(real_next_state)
             
             # Add to buffer (unbatching)
             self.buffer.add(
                 state_tensor[i],
                 action_indices[i],
                 rewards[i],
                 real_next_state_15ch,
                 dones[i]
             )
             
        # Update state for next step
        self.current_obs = next_states # This contains RESET states where appropriate
        self.valid_moves = new_valid_moves
        
        # Update total steps
        self.total_steps += num_envs
        
        # Optimization step
        # Check if we crossed a multiple of TRAIN_FREQ
        # Simple heuristic: train if (steps - num_envs) // freq != steps // freq
        if (self.total_steps - num_envs) // self.config.TRAIN_FREQ != self.total_steps // self.config.TRAIN_FREQ:
             if self.buffer.is_ready(self.config.BATCH_SIZE):
                 loss = self._optimization_step()
                 self.losses.append(loss)
                 self.optimization_steps += 1
                 
        return finished_rewards, num_envs
    
    def _run_episode(self) -> Tuple[float, int]:
        """
        Run one training episode.
        
        Returns:
            episode_reward: Total reward for episode
            episode_steps: Number of steps
        """
        # Reset environment and agent
        state = self.env.reset()
        self.agent.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Get valid moves
            valid_moves = self.env.get_valid_moves()
            
            if not valid_moves:
                break
            
            # Get board tensor for state encoding
            board = self.env.board.get_visible_board(self.agent.player_id)
            board_tensor = torch.as_tensor(board, dtype=torch.float32, device=self.device)
            
            # Create state tensor (15-channel board only)
            # Agent's act() will internally add AAREN embedding
            state_15ch = self._encode_board(board_tensor)
            
            # Select action - agent will add AAREN embedding internally
            action, action_idx = self.agent.act(
                state_15ch.unsqueeze(0),  # (1, 15, 10, 10)
                valid_moves,
                greedy=False
            )
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Get next state tensor (15-channel only)
            next_board = self.env.board.get_visible_board(self.agent.player_id)
            next_board_tensor = torch.as_tensor(next_board, dtype=torch.float32, device=self.device)
            next_state_15ch = self._encode_board(next_board_tensor)
            
            # Store experience (15-channel states)
            # Training will add AAREN embeddings when sampling
            self.buffer.add(
                state_15ch,
                action_idx,
                reward,
                next_state_15ch,
                done
            )
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Training step
            if (self.total_steps % self.config.TRAIN_FREQ == 0 and 
                self.buffer.is_ready(self.config.BATCH_SIZE)):
                
                loss = self._optimization_step()
                self.losses.append(loss)
                self.optimization_steps += 1
                
                # Self-checks
                if (self.selfcheck and 
                    self.optimization_steps % self.config.SELFCHECK_FREQ == 0):
                    self._run_selfchecks()
        
        return episode_reward, episode_steps
    
    def _encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """Encode board to 15-channel tensor. Supports batch input."""
        # Handle batch dimension
        if board.dim() == 2:  # (10, 10)
            batch_mode = False
            board = board.unsqueeze(0)  # (1, 10, 10)
        elif board.dim() == 3:  # (B, 10, 10)
            batch_mode = True
        else:
            # Already 4D (B, 15, 10, 10) or 3D (15, 10, 10)?
            # If input is already encoded, return it
            if board.size(-3) == 15:
                return board
            raise ValueError(f"Invalid board shape: {board.shape}")

        batch_size = board.shape[0]
        features = torch.zeros((batch_size, 15, 10, 10), device=self.device)
        
        player_id = self.agent.player_id
        LAKE_SQUARE = -13
        
        if player_id == 1:
            for i in range(1, 13):
                features[:, i-1] = (board == i).float()
            features[:, 12] = ((board < 0) & (board > LAKE_SQUARE)).float()
        else:
            for i in range(1, 13):
                features[:, i-1] = (board == -i).float()
            features[:, 12] = (board > 0).float()
        
        features[:, 13] = (board == LAKE_SQUARE).float()
        features[:, 14] = (board == 0).float()
        
        if not batch_mode:
            return features.squeeze(0)
            
        return features
    
    def _optimization_step(self) -> float:
        """
        Perform one optimization step.
        
        Returns:
            loss: Training loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.BATCH_SIZE
        )
        
        # Compute loss and update
        loss = self.agent.train_step(states, actions, rewards, next_states, dones)
        
        return loss
    
    def _run_selfchecks(self):
        """Run self-check protocols."""
        if not self.selfcheck:
            return
        
        # Get sample data for checks
        if self.buffer.is_ready(self.config.BATCH_SIZE):
            states, actions, rewards, next_states, dones = self.buffer.sample(
                min(32, self.config.BATCH_SIZE)
            )
            
            # Add AAREN embeddings if states are 15-channel
            if states.size(1) == 15:
                states_79ch = self.agent.get_state_with_aaren(states, action_features=None)
            else:
                states_79ch = states
            
            # Get Q-values
            with torch.no_grad():
                q_values = self.agent.online_network(states_79ch)
            
            # Get online-target distance
            online_target_mse = self.agent.get_online_target_distance()
            
            # Get real AAREN output for monitoring
            with torch.no_grad():
                aaren_output = self.agent.aaren(torch.zeros(states.size(0), 24, device=self.device))
            
            # Run checks
            results = self.selfcheck.run_all_checks(
                step=self.optimization_steps,
                aaren_output=aaren_output,
                q_values=q_values,
                online_target_mse=online_target_mse
            )
            
            # Log any alerts
            for name, result in results.items():
                if not result.passed:
                    print(f"[ALERT] {name}: {result.message}")
    
    def _evaluate(self) -> float:
        """
        Evaluate against random opponent.
        
        Returns:
            win_rate: Win rate over evaluation games
        """
        wins = 0
        
        for _ in range(self.config.EVAL_GAMES):
            # Run evaluation game
            state = self.env.reset()
            self.agent.reset()
            
            done = False
            while not done:
                valid_moves = self.env.get_valid_moves()
                
                if not valid_moves:
                    break
                
                # Greedy action (no exploration)
                board = self.env.board.get_visible_board(self.agent.player_id)
                board_tensor = torch.as_tensor(board, dtype=torch.float32, device=self.device)
                state_15ch = self._encode_board(board_tensor)
                aaren_embedding = torch.zeros(64, device=self.device)
                state_tensor = torch.cat([
                    state_15ch,
                    aaren_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, 10, 10)
                ], dim=0)
                
                action, _ = self.agent.act(
                    state_tensor.unsqueeze(0),
                    valid_moves,
                    greedy=True
                )
                
                _, reward, done, info = self.env.step(action)
                
                if done and reward > 0:  # Win
                    wins += 1
        
        win_rate = wins / self.config.EVAL_GAMES
        print(f"[EVAL] Step {self.total_steps}: Win rate = {win_rate:.1%}")
        
        return win_rate
    
    def _log_progress(self):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        epsilon = self.agent.epsilon
        
        # VRAM usage
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
        
        print(f"[Step {self.total_steps:,}] "
              f"Episodes: {self.episodes} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"ε: {epsilon:.3f} | "
              f"VRAM: {vram_gb:.2f}GB")
        
        # Store in history
        self.training_history.append({
            'step': self.total_steps,
            'episodes': self.episodes,
            'avg_reward': float(avg_reward),
            'avg_loss': float(avg_loss),
            'epsilon': float(epsilon),
            'vram_gb': float(vram_gb)
        })
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, f"checkpoint_{self.total_steps}.pt")
        
        self.agent.save(checkpoint_path)
        
        # Save training history
        history_path = os.path.join(self.log_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
    
    def _finalize_training(self, elapsed_time: float) -> Dict[str, Any]:
        """Finalize training and return results."""
        
        # Print self-check summary
        if self.selfcheck:
            self.selfcheck.print_summary()
        
        # Final results
        results = {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'optimization_steps': self.optimization_steps,
            'elapsed_time': elapsed_time,
            'final_epsilon': self.agent.epsilon,
            'avg_episode_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0,
            'win_rates': self.win_rates,
            'training_history': self.training_history
        }
        
        # Save final checkpoint
        self._save_checkpoint()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total steps: {self.total_steps:,}")
        print(f"  Episodes: {self.episodes}")
        print(f"  Time: {elapsed_time/3600:.2f} hours")
        if self.win_rates:
            print(f"  Final win rate: {self.win_rates[-1][1]:.1%}")
        print("=" * 60 + "\n")
        
        return results
