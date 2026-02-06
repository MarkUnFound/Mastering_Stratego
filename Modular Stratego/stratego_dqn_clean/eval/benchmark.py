"""
Benchmark for evaluating Double DQN + AAREN agent
Measures win rate against random agent over 100 games.
"""

import torch
import numpy as np
import sys
import os
import time
from typing import Optional, Tuple
from collections import defaultdict

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RandomAgent:
    """Simple random agent for benchmarking."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def act(self, valid_moves):
        """Select random action."""
        if not valid_moves:
            return None
        return np.random.choice(len(valid_moves))
    
    def reset(self):
        pass


class BenchmarkRunner:
    """
    Benchmarks agent against random opponent.
    
    Metrics:
    - Win rate
    - Average game length
    - Average reward
    - Action distribution entropy
    """
    
    def __init__(
        self,
        agent,
        env,
        opponent: Optional[RandomAgent] = None,
        device: torch.device = None
    ):
        self.agent = agent
        self.env = env
        self.opponent = opponent or RandomAgent(player_id=-1 * agent.player_id)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run_benchmark(
        self,
        num_games: int = 100,
        verbose: bool = True
    ) -> dict:
        """
        Run benchmark games.
        
        Args:
            num_games: Number of games to play
            verbose: Print progress
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_rewards': [],
            'game_lengths': [],
            'action_counts': defaultdict(int)
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"BENCHMARK: {num_games} games vs Random")
            print(f"{'='*50}")
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            outcome, reward, length, actions = self._play_game()
            
            if outcome == 'win':
                results['wins'] += 1
            elif outcome == 'loss':
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            results['total_rewards'].append(reward)
            results['game_lengths'].append(length)
            
            for action in actions:
                results['action_counts'][action] += 1
            
            if verbose and (game_idx + 1) % 10 == 0:
                win_rate = results['wins'] / (game_idx + 1)
                print(f"  Game {game_idx + 1}/{num_games}: Win rate = {win_rate:.1%}")
        
        elapsed = time.time() - start_time
        
        # Compute final statistics
        results['total_games'] = num_games
        results['win_rate'] = results['wins'] / num_games
        results['loss_rate'] = results['losses'] / num_games
        results['draw_rate'] = results['draws'] / num_games
        results['avg_reward'] = np.mean(results['total_rewards'])
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['elapsed_time'] = elapsed
        
        # Action entropy
        total_actions = sum(results['action_counts'].values())
        if total_actions > 0:
            probs = np.array(list(results['action_counts'].values())) / total_actions
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            results['action_entropy'] = entropy
        else:
            results['action_entropy'] = 0.0
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _play_game(self) -> Tuple[str, float, int, list]:
        """
        Play single game.
        
        Returns:
            outcome: 'win', 'loss', or 'draw'
            total_reward: Cumulative reward
            game_length: Number of moves
            actions: List of action indices taken
        """
        self.env.reset()
        self.agent.reset()
        self.opponent.reset()
        
        total_reward = 0.0
        game_length = 0
        actions_taken = []
        done = False
        
        current_player = 1  # Assume agent is player 1
        
        while not done:
            valid_moves = self.env.get_valid_moves()
            
            if not valid_moves:
                # No valid moves - draw or stalemate
                break
            
            if current_player == self.agent.player_id:
                # Agent's turn
                board = self.env.board.get_visible_board(self.agent.player_id)
                board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device)
                
                # Encode state (simplified - 15 channels + zeros for AAREN)
                state_tensor = self._encode_state(board_tensor)
                
                action, action_idx = self.agent.act(
                    state_tensor.unsqueeze(0),
                    valid_moves,
                    greedy=True  # Deterministic evaluation
                )
                
                actions_taken.append(action_idx)
            else:
                # Opponent's turn
                action_idx = self.opponent.act(valid_moves)
                action = valid_moves[action_idx]
            
            # Execute action
            _, reward, done, info = self.env.step(action)
            
            if current_player == self.agent.player_id:
                total_reward += reward
            
            game_length += 1
            
            # Switch player (simplified - actual Stratego has player tracking in env)
            current_player *= -1
            
            # Safety limit
            if game_length > 500:
                break
        
        # Determine outcome
        if total_reward > 5:
            outcome = 'win'
        elif total_reward < -5:
            outcome = 'loss'
        else:
            outcome = 'draw'
        
        return outcome, total_reward, game_length, actions_taken
    
    def _encode_state(self, board: torch.Tensor) -> torch.Tensor:
        """Encode board to 79-channel tensor."""
        LAKE_SQUARE = -13
        
        features = torch.zeros((15, 10, 10), device=self.device)
        player_id = self.agent.player_id
        
        if player_id == 1:
            for i in range(1, 13):
                features[i-1] = (board == i).float()
            features[12] = ((board < 0) & (board > LAKE_SQUARE)).float()
        else:
            for i in range(1, 13):
                features[i-1] = (board == -i).float()
            features[12] = (board > 0).float()
        
        features[13] = (board == LAKE_SQUARE).float()
        features[14] = (board == 0).float()
        
        # Add zero AAREN embedding (64 channels)
        aaren = torch.zeros((64, 10, 10), device=self.device)
        
        return torch.cat([features, aaren], dim=0)
    
    def _print_results(self, results: dict):
        """Print benchmark results."""
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"  Games played: {results['total_games']}")
        print(f"  Win rate:     {results['win_rate']:.1%}")
        print(f"  Loss rate:    {results['loss_rate']:.1%}")
        print(f"  Draw rate:    {results['draw_rate']:.1%}")
        print(f"  Avg reward:   {results['avg_reward']:.2f}")
        print(f"  Avg length:   {results['avg_game_length']:.1f} moves")
        print(f"  Action entropy: {results['action_entropy']:.2f} bits")
        print(f"  Time:         {results['elapsed_time']:.1f} seconds")
        print(f"{'='*50}\n")


def run_benchmark(
    agent,
    env,
    num_games: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Convenience function to run benchmark.
    
    Args:
        agent: Trained DoubleDQNAgent
        env: StrategoEnvironment
        num_games: Number of games to play
        device: PyTorch device
        
    Returns:
        Benchmark results dictionary
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    runner = BenchmarkRunner(agent, env, device=device)
    return runner.run_benchmark(num_games=num_games)


if __name__ == "__main__":
    print("Benchmark module - import and call run_benchmark()")
