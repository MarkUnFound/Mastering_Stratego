import torch
import numpy as np
import time
from stratego_env import StrategoEnv
from dqn_agent import DQNAgent, RandomAgent
from game_recorder import GameRecorder
from visualizer import TrainingVisualizer
import os
import argparse

class StrategoTrainer:
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        
        # Initialize environment
        self.env = StrategoEnv()
        state_size = self.env.get_state_space_size()
        action_size = self.env.get_action_space_size()
        
        # Initialize agents
        self.agent1 = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            player_id=0,
            **self.config['agent_params']
        )
        
        if self.config['opponent_type'] == 'dqn':
            self.agent2 = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                player_id=1,
                **self.config['agent_params']
            )
        else:
            self.agent2 = RandomAgent(player_id=1)
        
        # Initialize recorder and visualizer
        self.recorder = GameRecorder(self.config['log_dir'])
        self.visualizer = TrainingVisualizer()
        
        # Training statistics
        self.training_stats = {
            'episode': 0,
            'wins': [0, 0, 0],  # [agent1, agent2, draws]
            'total_rewards': [[], []],
            'losses': [[], []],
            'game_lengths': [],
            'training_time': 0
        }
        
        print("Stratego Trainer initialized")
        print(f"Agent 1: DQN")
        print(f"Agent 2: {self.config['opponent_type'].upper()}")
        print(f"Device: {self.agent1.device}")
    
    def _get_default_config(self):
        return {
            'episodes': 1000,
            'max_steps': 500,
            'opponent_type': 'random',  # 'random' or 'dqn'
            'save_interval': 100,
            'eval_interval': 50,
            'eval_episodes': 10,
            'log_dir': 'logs',
            'model_dir': 'models',
            'agent_params': {
                'lr': 1e-4,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
            'memory_size': args.memory_size,
            'batch_size': args.batch_size,
            'target_update': 1000
        }
    }
    
    # Initialize trainer
    trainer = StrategoTrainer(config)
    
    # Load models if specified
    if args.load_agent1 or args.load_agent2:
        trainer.load_models(args.load_agent1, args.load_agent2)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.recorder.print_summary()

if __name__ == "__main__":
    main().995,
                'memory_size': 100000,
                'batch_size': 64,
                'target_update': 1000
            }
        }
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['episodes']} episodes...")
        start_time = time.time()
        
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        for episode in range(self.config['episodes']):
            self.training_stats['episode'] = episode
            episode_start = time.time()
            
            # Run episode
            game_result = self._run_episode(episode)
            
            # Update statistics
            self._update_stats(game_result)
            
            # Training
            if isinstance(self.agent1, DQNAgent):
                loss1 = self.agent1.train()
                if loss1 is not None:
                    self.training_stats['losses'][0].append(loss1)
            
            if isinstance(self.agent2, DQNAgent):
                loss2 = self.agent2.train()
                if loss2 is not None:
                    self.training_stats['losses'][1].append(loss2)
            
            # Reset agents for new episode
            self.agent1.reset_episode()
            self.agent2.reset_episode()
            
            # Logging
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, time.time() - episode_start)
            
            # Evaluation
            if (episode + 1) % self.config['eval_interval'] == 0:
                self._evaluate(episode + 1)
            
            # Save models
            if (episode + 1) % self.config['save_interval'] == 0:
                self._save_models(episode + 1)
        
        self.training_stats['training_time'] = time.time() - start_time
        
        # Final evaluation and saving
        print("\nTraining completed!")
        self._evaluate(self.config['episodes'], final=True)
        self._save_models(self.config['episodes'], final=True)
        
        # Generate final visualizations
        self._generate_final_plots()
        
        # Print final summary
        self.recorder.print_summary()
    
    def _run_episode(self, episode):
        """Run a single episode"""
        state = self.env.reset()
        done = False
        step = 0
        episode_rewards = [0, 0]
        
        # Start recording
        agent1_type = "DQN" if isinstance(self.agent1, DQNAgent) else "Random"
        agent2_type = "DQN" if isinstance(self.agent2, DQNAgent) else "Random"
        self.recorder.start_game(agent1_type, agent2_type, episode)
        
        agents = [self.agent1, self.agent2]
        
        while not done and step < self.config['max_steps']:
            current_player = self.env.current_player
            current_agent = agents[current_player]
            
            # Get valid actions
            valid_actions = self.env.get_valid_actions(current_player)
            
            if not valid_actions:
                # No valid moves - game over
                done = True
                reward = -100  # Penalty for having no moves
                next_state = state
                info = {"winner": 1 - current_player, "no_moves": True}
            else:
                # Select action
                action = current_agent.select_action(state, valid_actions, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                current_agent.store_experience(state, action, reward, next_state, done)
                current_agent.add_reward(reward)
                
                episode_rewards[current_player] += reward
            
            # Record move
            self.recorder.record_move(
                current_player, state, action if not done or valid_actions else -1,
                reward, next_state, done, info
            )
            
            state = next_state
            step += 1
        
        # End recording
        self.recorder.end_game()
        
        return {
            'winner': info.get('winner') if done else None,
            'rewards': episode_rewards,
            'length': step,
            'invalid_moves': [0, 0]  # Could track this from recorder
        }
    
    def _update_stats(self, game_result):
        """Update training statistics"""
        if game_result['winner'] is not None:
            self.training_stats['wins'][game_result['winner']] += 1
        else:
            self.training_stats['wins'][2] += 1  # Draw
        
        self.training_stats['total_rewards'][0].append(game_result['rewards'][0])
        self.training_stats['total_rewards'][1].append(game_result['rewards'][1])
        self.training_stats['game_lengths'].append(game_result['length'])
    
    def _log_progress(self, episode, episode_time):
        """Log training progress"""
        recent_window = 50
        start_idx = max(0, len(self.training_stats['total_rewards'][0]) - recent_window)
        
        recent_rewards_1 = self.training_stats['total_rewards'][0][start_idx:]
        recent_rewards_2 = self.training_stats['total_rewards'][1][start_idx:]
        recent_lengths = self.training_stats['game_lengths'][start_idx:]
        
        # Win rates
        total_games = sum(self.training_stats['wins'])
        win_rate_1 = self.training_stats['wins'][0] / total_games if total_games > 0 else 0
        win_rate_2 = self.training_stats['wins'][1] / total_games if total_games > 0 else 0
        
        # Agent stats
        stats_1 = self.agent1.get_stats()
        stats_2 = self.agent2.get_stats() if hasattr(self.agent2, 'get_stats') else {}
        
        print(f"Episode {episode:4d} | "
              f"Win Rate: {win_rate_1:.3f}/{win_rate_2:.3f} | "
              f"Avg Reward: {np.mean(recent_rewards_1):6.2f}/{np.mean(recent_rewards_2):6.2f} | "
              f"Avg Length: {np.mean(recent_lengths):5.1f} | "
              f"ε: {stats_1.get('epsilon', 0):.3f} | "
              f"Time: {episode_time:.2f}s")
    
    def _evaluate(self, episode, final=False):
        """Evaluate agents"""
        print(f"\n{'='*20} EVALUATION {'='*20}")
        
        # Set agents to evaluation mode
        self.agent1.set_training_mode(False)
        if hasattr(self.agent2, 'set_training_mode'):
            self.agent2.set_training_mode(False)
        
        eval_wins = [0, 0, 0]
        eval_rewards = [[], []]
        eval_lengths = []
        
        for eval_ep in range(self.config['eval_episodes']):
            result = self._run_episode(f"eval_{episode}_{eval_ep}")
            
            if result['winner'] is not None:
                eval_wins[result['winner']] += 1
            else:
                eval_wins[2] += 1
            
            eval_rewards[0].append(result['rewards'][0])
            eval_rewards[1].append(result['rewards'][1])
            eval_lengths.append(result['length'])
        
        # Print evaluation results
        total_eval = sum(eval_wins)
        print(f"Evaluation Results (Episode {episode}):")
        print(f"Agent 1 wins: {eval_wins[0]:2d} ({eval_wins[0]/total_eval:.2%})")
        print(f"Agent 2 wins: {eval_wins[1]:2d} ({eval_wins[1]/total_eval:.2%})")
        print(f"Draws:        {eval_wins[2]:2d} ({eval_wins[2]/total_eval:.2%})")
        print(f"Avg rewards:  {np.mean(eval_rewards[0]):6.2f} / {np.mean(eval_rewards[1]):6.2f}")
        print(f"Avg length:   {np.mean(eval_lengths):5.1f}")
        print("=" * 50)
        
        # Set back to training mode
        self.agent1.set_training_mode(True)
        if hasattr(self.agent2, 'set_training_mode'):
            self.agent2.set_training_mode(True)
    
    def _save_models(self, episode, final=False):
        """Save agent models"""
        suffix = "_final" if final else f"_ep{episode}"
        
        if isinstance(self.agent1, DQNAgent):
            filepath1 = os.path.join(self.config['model_dir'], f"agent1{suffix}.pth")
            self.agent1.save_model(filepath1)
        
        if isinstance(self.agent2, DQNAgent):
            filepath2 = os.path.join(self.config['model_dir'], f"agent2{suffix}.pth")
            self.agent2.save_model(filepath2)
    
    def _generate_final_plots(self):
        """Generate final training plots"""
        print("Generating training visualizations...")
        
        # Get performance data from recorder
        perf_data = self.recorder.get_performance_data()
        
        if perf_data:
            # Create plots
            self.visualizer.plot_win_rates(perf_data)
            self.visualizer.plot_rewards(perf_data)
            self.visualizer.plot_game_length(perf_data)
            
            # Agent-specific plots
            if isinstance(self.agent1, DQNAgent):
                self.visualizer.plot_training_loss(self.agent1.losses, "Agent 1")
            
            if isinstance(self.agent2, DQNAgent):
                self.visualizer.plot_training_loss(self.agent2.losses, "Agent 2")
            
            # Save all plots
            plots_dir = os.path.join(self.config['log_dir'], 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            self.visualizer.save_all_plots(plots_dir)
        
        # Export data
        self.recorder.export_csv()
    
    def load_models(self, agent1_path=None, agent2_path=None):
        """Load pre-trained models"""
        if agent1_path and isinstance(self.agent1, DQNAgent):
            self.agent1.load_model(agent1_path)
        
        if agent2_path and isinstance(self.agent2, DQNAgent):
            self.agent2.load_model(agent2_path)

def main():
    parser = argparse.ArgumentParser(description='Train Stratego DQN agents')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--opponent', choices=['random', 'dqn'], default='random', 
                       help='Opponent type')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--memory-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--load-agent1', type=str, help='Path to agent 1 model')
    parser.add_argument('--load-agent2', type=str, help='Path to agent 2 model')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = {
        'episodes': args.episodes,
        'opponent_type': args.opponent,
        'save_interval': args.save_interval,
        'eval_interval': args.eval_interval,
        'eval_episodes': 10,
        'log_dir': 'logs',
        'model_dir': 'models',
        'agent_params': {
            'lr': args.lr,
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'epsilon_min': 0.01,
            'epsilon_decay': 0