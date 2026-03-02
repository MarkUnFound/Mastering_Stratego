"""
Stratego Piece Value Calculator

Calculates empirical piece values through Monte Carlo simulation and statistical analysis.
Similar to how chess piece values (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9) were derived,
this script determines the relative worth of each Stratego piece.

Methods used:
1. Win Correlation: How piece survival correlates with game wins
2. Capture Exchange: Net value when trading pieces
3. Mobility Value: Movement capabilities and board control
4. Special Ability: Flag capture, bomb defusal, Marshal assassination
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm

from piece import PieceType
from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from opponents import RandomAgent


@dataclass
class PieceStatistics:
    """Statistics collected for a single piece type."""
    games_started: int = 0
    games_survived: int = 0  # Piece still alive at game end
    games_won_with: int = 0  # Games won where this piece was present
    games_lost_with: int = 0  # Games lost where this piece was present
    
    total_captures: int = 0  # Times this piece captured another
    total_captured: int = 0  # Times this piece was captured
    capture_values: List[int] = field(default_factory=list)  # Values of pieces captured
    captured_by_values: List[int] = field(default_factory=list)  # Values of pieces that captured this
    
    moves_made: int = 0  # Total moves made by this piece type
    attacks_initiated: int = 0  # Attacks this piece started
    attacks_survived: int = 0  # Attacks where this piece won
    
    flag_captures: int = 0  # Times captured enemy flag
    bomb_defusals: int = 0  # Times defused a bomb (Miner only)
    marshal_kills: int = 0  # Times killed Marshal (Spy only)


class PieceValueCalculator:
    """
    Calculates empirical piece values through game simulation.
    
    The value calculation considers:
    - Win correlation: P(win | piece_survives) vs P(win | piece_dies)
    - Exchange value: Average net value when trading
    - Mobility: Average number of moves available
    - Special abilities: Flag capture, bomb defusal, etc.
    """
    
    # Base rank values (ordinal, not empirical)
    RANK_VALUES = {
        PieceType.FLAG: 0,      # Can't move
        PieceType.SPY: 1,       # Rank 1
        PieceType.SCOUT: 2,     # Rank 2
        PieceType.MINER: 3,     # Rank 3
        PieceType.SERGEANT: 4,  # Rank 4
        PieceType.LIEUTENANT: 5,# Rank 5
        PieceType.CAPTAIN: 6,   # Rank 6
        PieceType.MAJOR: 7,     # Rank 7
        PieceType.COLONEL: 8,   # Rank 8
        PieceType.GENERAL: 9,   # Rank 9
        PieceType.MARSHAL: 10,  # Rank 10
        PieceType.BOMB: 11,     # Immovable
    }
    
    # Piece counts in standard setup
    PIECE_COUNTS = {
        PieceType.FLAG: 1,
        PieceType.SPY: 1,
        PieceType.SCOUT: 8,
        PieceType.MINER: 5,
        PieceType.SERGEANT: 4,
        PieceType.LIEUTENANT: 4,
        PieceType.CAPTAIN: 4,
        PieceType.MAJOR: 3,
        PieceType.COLONEL: 2,
        PieceType.GENERAL: 1,
        PieceType.MARSHAL: 1,
        PieceType.BOMB: 6,
    }
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats: Dict[PieceType, PieceStatistics] = {
            pt: PieceStatistics() for pt in PieceType
        }
        self.games_played = 0
        self.calculated_values: Dict[PieceType, float] = {}
    
    def load_results(self, filepath: str = "piece_values.json") -> bool:
        """Load existing results from JSON file to continue from."""
        if not os.path.exists(filepath):
            print(f" No existing file found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.games_played = data.get("games_simulated", 0)
            
            # Load statistics for each piece type
            for pt_name, stats_data in data.get("statistics", {}).items():
                try:
                    pt = PieceType[pt_name]
                    self.stats[pt].games_started = stats_data.get("games_started", 0)
                    self.stats[pt].games_survived = stats_data.get("games_survived", 0)
                    self.stats[pt].games_won_with = stats_data.get("games_won_with", 0)
                    self.stats[pt].games_lost_with = stats_data.get("games_lost_with", 0)
                    self.stats[pt].total_captures = stats_data.get("total_captures", 0)
                    self.stats[pt].total_captured = stats_data.get("total_captured", 0)
                    self.stats[pt].moves_made = stats_data.get("moves_made", 0)
                    self.stats[pt].attacks_initiated = stats_data.get("attacks_initiated", 0)
                    self.stats[pt].attacks_survived = stats_data.get("attacks_survived", 0)
                    self.stats[pt].flag_captures = stats_data.get("flag_captures", 0)
                    self.stats[pt].bomb_defusals = stats_data.get("bomb_defusals", 0)
                    self.stats[pt].marshal_kills = stats_data.get("marshal_kills", 0)
                except KeyError:
                    pass
            
            print(f" Loaded {self.games_played} games from {filepath}")
            return True
        except Exception as e:
            print(f" Error loading {filepath}: {e}")
            return False
        
    def run_simulation(self, num_games: int = 1000, 
                       agent1_path: Optional[str] = None,
                       agent2_path: Optional[str] = None,
                       show_progress: bool = True) -> None:
        """
        Run Monte Carlo simulation games to collect piece statistics.
        
        Args:
            num_games: Number of games to simulate
            agent1_path: Path to trained agent 1 (uses random if None)
            agent2_path: Path to trained agent 2 (uses random if None)
            show_progress: Whether to show progress bar
        """
        print(f"\n{'='*60}")
        print(f"Running {num_games} simulation games...")
        print(f"{'='*60}\n")
        
        env = StrategoEnvironment(self.device)
        
        # Initialize agents
        if agent1_path:
            agent1 = RainbowAgent(player_id=1, device=self.device)
            agent1.load_model(agent1_path)
            print(f"Loaded Agent 1 from: {agent1_path}")
        else:
            agent1 = RandomAgent()
            print("Using Random Agent 1")
            
        if agent2_path:
            agent2 = RainbowAgent(player_id=-1, device=self.device)
            agent2.load_model(agent2_path)
            print(f"Loaded Agent 2 from: {agent2_path}")
        else:
            agent2 = RandomAgent()
            print("Using Random Agent 2")
        
        iterator = tqdm(range(num_games), desc="Simulating games") if show_progress else range(num_games)
        
        for game_idx in iterator:
            self._simulate_game(env, agent1, agent2)
            self.games_played += 1
            
        print(f"\n Completed {num_games} simulation games")
        
    def _simulate_game(self, env: StrategoEnvironment, agent1, agent2) -> None:
        """Simulate a single game and collect statistics."""
        state = env.reset()
        
        # Track pieces at game start
        initial_pieces = self._get_piece_counts(env.board.actual_board)
        for player in [1, -1]:
            for piece_type, count in initial_pieces[player].items():
                self.stats[piece_type].games_started += count
        
        # Track piece positions for this game
        piece_positions = {}  # (player, piece_type) -> list of (row, col)
        
        done = False
        turn = 0
        max_turns = 1000
        
        while not done and turn < max_turns:
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2
            
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                break
                
            # Get action - both agent types use act() method
            action = agent.act(state.board, valid_moves, state)
            
            if action is None:
                action = valid_moves[0] if valid_moves else None
                
            if action is None:
                done = True
                break
            
            # Track the move
            (r_from, c_from), (r_to, c_to) = action
            moving_piece_val = env.board.actual_board[r_from, c_from].item()
            target_piece_val = env.board.actual_board[r_to, c_to].item()
            
            if moving_piece_val != 0:
                moving_type = PieceType(abs(moving_piece_val))
                self.stats[moving_type].moves_made += 1
                
                # Check if this is an attack
                if target_piece_val != 0:
                    self.stats[moving_type].attacks_initiated += 1
                    target_type = PieceType(abs(target_piece_val))
                    
                    # Record pre-battle state
                    attacker_rank = self.RANK_VALUES[moving_type]
                    defender_rank = self.RANK_VALUES[target_type]
            
            # Execute move
            next_state, reward, done, info = env.step(action)
            
            # Track battle outcomes
            if target_piece_val != 0 and moving_piece_val != 0:
                self._record_battle(
                    env, 
                    moving_type, moving_piece_val > 0,
                    target_type, target_piece_val > 0,
                    action, info
                )
            
            state = next_state
            turn += 1
        
        # Record game outcome
        winner = env.winner
        final_pieces = self._get_piece_counts(env.board.actual_board)
        
        # Update survival and win statistics
        for player in [1, -1]:
            player_won = (winner == player)
            for piece_type, count in final_pieces[player].items():
                self.stats[piece_type].games_survived += count
                if player_won:
                    self.stats[piece_type].games_won_with += count
                else:
                    self.stats[piece_type].games_lost_with += count
    
    def _get_piece_counts(self, board: torch.Tensor) -> Dict[int, Dict[PieceType, int]]:
        """Get piece counts for each player from board state."""
        counts = {1: defaultdict(int), -1: defaultdict(int)}
        
        for r in range(10):
            for c in range(10):
                val = board[r, c].item()
                if val > 0 and val <= 11:  # Player 1
                    counts[1][PieceType(val)] += 1
                elif val < 0 and val >= -11:  # Player -1
                    counts[-1][PieceType(abs(val))] += 1
                    
        return counts
    
    def _record_battle(self, env, attacker_type, attacker_is_p1, 
                       defender_type, defender_is_p1, action, info) -> None:
        """Record battle outcome statistics."""
        (r_from, c_from), (r_to, c_to) = action
        
        # Check what happened at the target position
        new_val = env.board.actual_board[r_to, c_to].item()
        old_from = env.board.actual_board[r_from, c_from].item()  # Should be 0 if attacker moved
        
        attacker_rank = self.RANK_VALUES[attacker_type]
        defender_rank = self.RANK_VALUES[defender_type]
        
        # Determine outcome
        if old_from == 0 and new_val != 0:
            # Attacker won (moved to target position)
            self.stats[attacker_type].attacks_survived += 1
            self.stats[attacker_type].total_captures += 1
            self.stats[attacker_type].capture_values.append(defender_rank)
            self.stats[defender_type].total_captured += 1
            self.stats[defender_type].captured_by_values.append(attacker_rank)
            
            # Special cases
            if defender_type == PieceType.FLAG:
                self.stats[attacker_type].flag_captures += 1
            if defender_type == PieceType.BOMB and attacker_type == PieceType.MINER:
                self.stats[attacker_type].bomb_defusals += 1
            if defender_type == PieceType.MARSHAL and attacker_type == PieceType.SPY:
                self.stats[attacker_type].marshal_kills += 1
                
        elif old_from == 0 and new_val == 0:
            # Both died (mutual destruction)
            self.stats[attacker_type].total_captured += 1
            self.stats[attacker_type].captured_by_values.append(defender_rank)
            self.stats[defender_type].total_captured += 1
            self.stats[defender_type].captured_by_values.append(attacker_rank)
            
        else:
            # Defender won (attacker died, defender stayed)
            self.stats[defender_type].total_captures += 1
            self.stats[defender_type].capture_values.append(attacker_rank)
            self.stats[attacker_type].total_captured += 1
            self.stats[attacker_type].captured_by_values.append(defender_rank)
    
    def calculate_values(self) -> Dict[PieceType, float]:
        """
        Calculate empirical piece values from collected statistics.
        
        Value = win_correlation * w1 + exchange_value * w2 + mobility * w3 + special * w4
        
        Returns normalized values where Scout = 1.0 (like Pawn = 1 in chess)
        """
        print(f"\n{'='*60}")
        print("Calculating Piece Values...")
        print(f"{'='*60}\n")
        
        raw_values = {}
        
        for piece_type in PieceType:
            stats = self.stats[piece_type]
            
            # Skip if no data
            if stats.games_started == 0:
                raw_values[piece_type] = 0.0
                continue
            
            # 1. Win Correlation Value (0-10 scale)
            # How much does having this piece correlate with winning?
            total_outcomes = stats.games_won_with + stats.games_lost_with
            if total_outcomes > 0:
                win_rate = stats.games_won_with / total_outcomes
                win_value = win_rate * 10
            else:
                win_value = 5.0  # Neutral
            
            # 2. Survival Value (0-5 scale)
            # Pieces that survive more are more valuable
            survival_rate = stats.games_survived / max(1, stats.games_started)
            survival_value = survival_rate * 5
            
            # 3. Exchange Value (-5 to +5 scale)
            # Net value from trades
            if stats.capture_values:
                avg_captured = np.mean(stats.capture_values)
            else:
                avg_captured = 0
                
            if stats.captured_by_values:
                avg_captured_by = np.mean(stats.captured_by_values)
            else:
                avg_captured_by = 0
                
            exchange_value = (avg_captured - avg_captured_by) / 2
            
            # 4. Attack Efficiency (0-5 scale)
            if stats.attacks_initiated > 0:
                attack_success = stats.attacks_survived / stats.attacks_initiated
                attack_value = attack_success * 5
            else:
                attack_value = 0
            
            # 5. Special Ability Value (0-5 scale)
            special_value = 0
            if piece_type == PieceType.MINER and stats.bomb_defusals > 0:
                defusal_rate = stats.bomb_defusals / max(1, stats.attacks_survived)
                special_value = defusal_rate * 5 + 2  # Bonus for bomb defusal ability
            elif piece_type == PieceType.SPY and stats.marshal_kills > 0:
                special_value = 5  # High value for Marshal assassination
            elif piece_type == PieceType.SCOUT:
                special_value = 2  # Mobility bonus
            elif piece_type == PieceType.FLAG:
                special_value = 0  # Infinite value in a sense, but can't move
            elif piece_type == PieceType.BOMB:
                special_value = 1  # Defensive value
            
            # 6. Base Rank Value (0-10 scale)
            rank_value = self.RANK_VALUES[piece_type]
            
            # Combine with weights
            # win(0.30) + survival(0.15) + exchange(0.20) + attack(0.15) + special(0.10) + rank(0.10)
            total_value = (
                win_value * 0.30 +
                survival_value * 0.15 +
                exchange_value * 0.20 +
                attack_value * 0.15 +
                special_value * 0.10 +
                rank_value * 0.10
            )
            
            raw_values[piece_type] = max(0, total_value)
        
        # Normalize so Scout = 1.0 (like Pawn in chess)
        scout_value = raw_values.get(PieceType.SCOUT, 1.0)
        if scout_value == 0:
            scout_value = 1.0
            
        self.calculated_values = {
            pt: val / scout_value 
            for pt, val in raw_values.items()
        }
        
        return self.calculated_values
    
    def print_report(self) -> None:
        """Print a detailed report of piece values and statistics."""
        if not self.calculated_values:
            self.calculate_values()
            
        print(f"\n{'='*70}")
        print("STRATEGO EMPIRICAL PIECE VALUES")
        print(f"Based on {self.games_played} simulated games")
        print(f"{'='*70}\n")
        
        # Sort by value
        sorted_pieces = sorted(
            self.calculated_values.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"{'Piece':<15} {'Value':>8} {'Win%':>8} {'Surv%':>8} {'Capt':>6} {'Lost':>6}")
        print("-" * 55)
        
        for piece_type, value in sorted_pieces:
            stats = self.stats[piece_type]
            
            total_outcomes = stats.games_won_with + stats.games_lost_with
            win_pct = (stats.games_won_with / total_outcomes * 100) if total_outcomes > 0 else 0
            surv_pct = (stats.games_survived / stats.games_started * 100) if stats.games_started > 0 else 0
            
            print(f"{piece_type.name:<15} {value:>8.2f} {win_pct:>7.1f}% {surv_pct:>7.1f}% {stats.total_captures:>6} {stats.total_captured:>6}")
        
        print("\n" + "-" * 55)
        print("\nChess-style notation (Scout = 1 point):")
        print("-" * 40)
        
        for piece_type, value in sorted_pieces:
            if piece_type not in [PieceType.FLAG, PieceType.BOMB]:
                chess_val = round(value, 1)
                print(f"  {piece_type.name:<12} = {chess_val}")
        
        print("\nNotes:")
        print("  - Flag has infinite implicit value (game ends if captured)")
        print("  - Bomb value is defensive (can't move, but blocks)")
        print("  - Spy's value increases with Marshal presence on board")
        print("  - Miner value increases as more Bombs remain")
        
    def save_results(self, filepath: str = "piece_values.json") -> None:
        """Save calculated values to JSON file."""
        if not self.calculated_values:
            self.calculate_values()
            
        results = {
            "games_simulated": self.games_played,
            "values": {pt.name: val for pt, val in self.calculated_values.items()},
            "statistics": {}
        }
        
        for pt, stats in self.stats.items():
            results["statistics"][pt.name] = {
                "games_started": stats.games_started,
                "games_survived": stats.games_survived,
                "games_won_with": stats.games_won_with,
                "games_lost_with": stats.games_lost_with,
                "total_captures": stats.total_captures,
                "total_captured": stats.total_captured,
                "moves_made": stats.moves_made,
                "attacks_initiated": stats.attacks_initiated,
                "attacks_survived": stats.attacks_survived,
                "flag_captures": stats.flag_captures,
                "bomb_defusals": stats.bomb_defusals,
                "marshal_kills": stats.marshal_kills,
            }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Results saved to {filepath}")


def main():
    """Run piece value calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Stratego piece values')
    parser.add_argument('--games', type=int, default=2000, help='Number of games to simulate')
    parser.add_argument('--agent1', type=str, default=None, help='Path to agent 1 model')
    parser.add_argument('--agent2', type=str, default=None, help='Path to agent 2 model')
    parser.add_argument('--output', type=str, default='piece_values.json', help='Output file')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results file')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STRATEGO PIECE VALUE CALCULATOR")
    print("="*60)
    print("\nThis script calculates empirical piece values through simulation,")
    print("similar to how chess piece values were derived over centuries.")
    print()
    
    calculator = PieceValueCalculator()
    
    # Load existing results if resuming
    if args.resume:
        if calculator.load_results(args.output):
            print(f"Continuing from {calculator.games_played} games...")
        else:
            print("Starting fresh...")
    
    # Run simulation
    calculator.run_simulation(
        num_games=args.games,
        agent1_path=args.agent1,
        agent2_path=args.agent2
    )
    
    # Calculate and display results
    calculator.calculate_values()
    calculator.print_report()
    
    # Save to file
    calculator.save_results(args.output)
    

if __name__ == "__main__":
    main()
