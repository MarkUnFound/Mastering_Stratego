# main.py - Main entry point for Stratego DQN system

import os
import sys
import argparse
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    directories = ['logs', 'models', 'plots', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def train_agents():
    """Train DQN agents"""
    from training_loop import StrategoTrainer
    
    print("Starting training...")
    config = {
        'episodes': 2000,
        'opponent_type': 'random',  # Start with random opponent
        'save_interval': 100,
        'eval_interval': 50,
        'eval_episodes': 20,
        'log_dir': 'logs',
        'model_dir': 'models',
        'agent_params': {
            'lr': 3e-4,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.998,
            'memory_size': 200000,
            'batch_size': 128,
            'target_update': 1000
        }
    }
    
    trainer = StrategoTrainer(config)
    trainer.train()
    
    print("\nPhase 1 complete! Now training against DQN opponent...")
    
    # Phase 2: Self-play training
    config['opponent_type'] = 'dqn'
    config['episodes'] = 1000
    
    trainer2 = StrategoTrainer(config)
    # Load the trained agent as opponent
    trainer2.load_models('models/agent1_final.pth', 'models/agent1_final.pth')
    trainer2.train()

def play_human_vs_ai():
    """Launch human vs AI interface"""
    from player_interface import StrategoGUI
    
    print("Launching GUI interface...")
    app = StrategoGUI()
    app.run()

def analyze_games():
    """Analyze recorded games"""
    from game_recorder import GameRecorder
    from visualizer import TrainingVisualizer
    
    print("Analyzing game data...")
    
    # Load latest session
    log_dir = Path('logs')
    session_files = list(log_dir.glob('session_*.json'))
    
    if not session_files:
        print("No game sessions found!")
        return
    
    latest_session = max(session_files, key=os.path.getmtime)
    print(f"Analyzing session: {latest_session}")
    
    recorder = GameRecorder()
    # Load and analyze session data
    import json
    with open(latest_session, 'r') as f:
        session_data = json.load(f)
    
    recorder.games_data = session_data['games']
    recorder.print_summary()
    
    # Generate visualizations
    visualizer = TrainingVisualizer()
    perf_data = recorder.get_performance_data()
    
    if perf_data:
        visualizer.plot_win_rates(perf_data)
        visualizer.plot_rewards(perf_data)
        visualizer.plot_game_length(perf_data)
        
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        visualizer.save_all_plots(str(plots_dir))
        
        print(f"Plots saved to: {plots_dir}")

def benchmark_agents():
    """Benchmark different agent configurations"""
    from stratego_env import StrategoEnv
    from dqn_agent import DQNAgent, RandomAgent
    from game_recorder import GameRecorder
    import numpy as np
    
    print("Benchmarking agents...")
    
    env = StrategoEnv()
    state_size = env.get_state_space_size()
    action_size = env.get_action_space_size()
    
    # Load different models for comparison
    model_paths = ['models/agent1_final.pth', 'models/agent2_final.pth']
    agents = []
    
    # Add trained agents
    for i, path in enumerate(model_paths):
        if os.path.exists(path):
            agent = DQNAgent(state_size, action_size, i, epsilon=0.0)
            if agent.load_model(path):
                agent.set_training_mode(False)
                agents.append((f"DQN_Agent_{i+1}", agent))
    
    # Add random agent
    agents.append(("Random_Agent", RandomAgent(len(agents))))
    
    if len(agents) < 2:
        print("Need at least 2 agents for benchmarking!")
        return
    
    # Round-robin tournament
    results = {}
    num_games = 50
    
    for i, (name1, agent1) in enumerate(agents):
        for j, (name2, agent2) in enumerate(agents[i+1:], i+1):
            print(f"Playing {name1} vs {name2}...")
            
            wins = [0, 0, 0]  # [agent1, agent2, draws]
            
            for game in range(num_games):
                env.reset()
                current_agents = [agent1, agent2]
                
                while not env.game_over:
                    current_player = env.current_player
                    current_agent = current_agents[current_player]
                    
                    valid_actions = env.get_valid_actions(current_player)
                    if not valid_actions:
                        break
                    
                    action = current_agent.select_action(env.get_state(), valid_actions, training=False)
                    _, _, done, info = env.step(action)
                    
                    if done:
                        if info.get('winner') is not None:
                            wins[info['winner']] += 1
                        else:
                            wins[2] += 1
                        break
            
            # Store results
            results[f"{name1}_vs_{name2}"] = {
                'agent1_wins': wins[0],
                'agent2_wins': wins[1], 
                'draws': wins[2],
                'total_games': num_games
            }
            
            win_rate1 = wins[0] / num_games
            win_rate2 = wins[1] / num_games
            draw_rate = wins[2] / num_games
            
            print(f"Results: {name1} {win_rate1:.2%} - {name2} {win_rate2:.2%} - Draws {draw_rate:.2%}")
    
    # Save benchmark results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark complete! Results saved to benchmark_results.json")

def main():
    parser = argparse.ArgumentParser(description='Stratego DQN System')
    parser.add_argument('command', choices=['train', 'play', 'analyze', 'benchmark', 'setup'], 
                       help='Command to execute')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--model', type=str, help='Model path for playing')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_directories()
        print("Setup complete!")
        
    elif args.command == 'train':
        setup_directories()
        train_agents()
        
    elif args.command == 'play':
        play_human_vs_ai()
        
    elif args.command == 'analyze':
        analyze_games()
        
    elif args.command == 'benchmark':
        benchmark_agents()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# ============================================================================
# requirements.txt
"""
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.4.0
tkinter  # Usually included with Python
"""

# ============================================================================
# README.md content
README_CONTENT = """
# Stratego DQN Multi-Agent System

This project implements a multi-agent Deep Q-Network (DQN) system for playing the board game Stratego. The system includes training loops, game recording, performance visualization, and a human-playable interface.

## Features

- **Multi-Agent DQN Training**: Train agents to play against each other or random opponents
- **Game Recording**: Complete game history tracking with statistics and analysis
- **Performance Visualization**: Comprehensive charts and graphs of training progress
- **Human vs AI Interface**: Both GUI and console interfaces for playing against trained agents
- **Benchmarking System**: Compare different agent configurations

## Project Structure

```
├── main.py                 # Main entry point
├── stratego_env.py         # Stratego game environment
├── dqn_agent.py           # DQN agent implementation
├── training_loop.py       # Training orchestration
├── game_recorder.py       # Game recording and analysis
├── visualizer.py          # Training visualization
├── player_interface.py    # Human player interfaces
├── logs/                  # Training logs and data
├── models/                # Saved model checkpoints
├── plots/                 # Generated visualizations
└── requirements.txt       # Python dependencies
```

## Quick Start

1. **Setup**:
```bash
python main.py setup
pip install -r requirements.txt
```

2. **Train Agents**:
```bash
python main.py train
```

3. **Play Against AI**:
```bash
python main.py play
```

4. **Analyze Results**:
```bash
python main.py analyze
```

## Usage Examples

### Training
```bash
# Basic training
python training_loop.py --episodes 1000 --opponent random

# Advanced training with custom parameters
python training_loop.py --episodes 2000 --opponent dqn --lr 1e-4 --batch-size 128
```

### Playing
```bash
# GUI interface
python player_interface.py --interface gui --model models/agent1_final.pth

# Console interface  
python player_interface.py --interface console --model models/agent1_final.pth
```

### Analysis
```bash
# Generate performance plots
python -c "from visualizer import TrainingVisualizer; v = TrainingVisualizer(); v.show_all()"
```

## Game Rules (Simplified Stratego)

- 10x10 board with water obstacles
- Each player has 40 pieces with different ranks and abilities
- Goal: Capture the opponent's flag
- Higher-ranked pieces defeat lower-ranked pieces
- Special pieces: Spy defeats Marshal, Miners defeat Bombs
- Scouts can move multiple squares

## Training Process

1. **Phase 1**: Agent trains against random opponent to learn basic strategies
2. **Phase 2**: Self-play training where agents play against copies of themselves
3. **Evaluation**: Periodic evaluation games to measure progress
4. **Model Saving**: Regular checkpoints and final model saves

## Performance Metrics

- Win rates over time
- Average rewards per episode
- Game length statistics  
- Training loss curves
- Invalid move penalties

## Model Architecture

- Deep Q-Network with 3 hidden layers (512 units each)
- Experience replay with buffer size 100k+
- Target network updates every 1000 steps
- Epsilon-greedy exploration with decay

## Customization

The system is highly configurable through:
- Training parameters (learning rate, batch size, etc.)
- Network architecture
- Reward structure
- Game rules and board setup

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Pandas
- Tkinter (for GUI)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
"""

def create_readme():
    """Create README.md file"""
    with open('README.md', 'w') as f:
        f.write(README_CONTENT)
    print("README.md created")

def create_requirements():
    """Create requirements.txt file"""
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.4.0",
        "# tkinter usually included with Python"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    print("requirements.txt created")

if __name__ == "__main__":
    # Create additional files if run directly
    create_readme()
    create_requirements()
    main()