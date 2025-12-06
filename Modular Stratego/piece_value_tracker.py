"""
Piece Value Tracker - Lightweight training integration

Tracks empirical piece statistics during DQN training to compare
with analytical values. Designed for minimal performance impact.

Usage in train_dqn.py:
    from piece_value_tracker import PieceValueTracker
    tracker = PieceValueTracker()
    
    # After each game step with a capture:
    tracker.record_battle(attacker_type, defender_type, attacker_won)
    
    # After each game ends:
    tracker.record_game_end(winner, surviving_pieces_p1, surviving_pieces_p2)
    
    # Periodically log comparison:
    if episode % 500 == 0:
        tracker.log_comparison()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum
import math
import json
import os


class PieceType(IntEnum):
    """Stratego piece types."""
    FLAG = 0
    SPY = 1
    SCOUT = 2
    MINER = 3
    SERGEANT = 4
    LIEUTENANT = 5
    CAPTAIN = 6
    MAJOR = 7
    COLONEL = 8
    GENERAL = 9
    MARSHAL = 10
    BOMB = 11


# Piece counts for standard Stratego
PIECE_COUNTS = {
    PieceType.FLAG: 1, PieceType.SPY: 1, PieceType.SCOUT: 8,
    PieceType.MINER: 5, PieceType.SERGEANT: 4, PieceType.LIEUTENANT: 4,
    PieceType.CAPTAIN: 4, PieceType.MAJOR: 3, PieceType.COLONEL: 2,
    PieceType.GENERAL: 1, PieceType.MARSHAL: 1, PieceType.BOMB: 6,
}

# Pre-computed analytical values (Scout = 1.0)
ANALYTICAL_VALUES = {
    PieceType.MARSHAL: 8.35, PieceType.GENERAL: 8.00, PieceType.COLONEL: 7.48,
    PieceType.MAJOR: 6.72, PieceType.CAPTAIN: 5.73, PieceType.LIEUTENANT: 4.75,
    PieceType.MINER: 4.31, PieceType.SERGEANT: 3.77, PieceType.BOMB: 2.00,
    PieceType.SPY: 1.11, PieceType.SCOUT: 1.00, PieceType.FLAG: 0.14,
}


@dataclass
class PieceStats:
    """Lightweight statistics for a single piece type."""
    games_won_with: int = 0
    games_lost_with: int = 0
    total_captures: int = 0
    total_captured: int = 0
    capture_value_sum: float = 0.0  # Sum of values of captured pieces
    captured_by_value_sum: float = 0.0  # Sum of values that captured this


class PieceValueTracker:
    """
    Lightweight tracker for piece value convergence analysis.
    
    Tracks minimal statistics to compare empirical vs analytical values
    with negligible performance impact on training.
    """
    
    def __init__(self, save_path: str = "piece_value_tracking.json"):
        self.save_path = save_path
        self.stats: Dict[PieceType, PieceStats] = {
            pt: PieceStats() for pt in PieceType
        }
        self.games_tracked = 0
        self._load_existing()
    
    def _load_existing(self):
        """Load existing stats if available."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.games_tracked = data.get("games_tracked", 0)
                for pt_name, stats in data.get("stats", {}).items():
                    try:
                        pt = PieceType[pt_name]
                        self.stats[pt].games_won_with = stats.get("games_won_with", 0)
                        self.stats[pt].games_lost_with = stats.get("games_lost_with", 0)
                        self.stats[pt].total_captures = stats.get("total_captures", 0)
                        self.stats[pt].total_captured = stats.get("total_captured", 0)
                        self.stats[pt].capture_value_sum = stats.get("capture_value_sum", 0.0)
                        self.stats[pt].captured_by_value_sum = stats.get("captured_by_value_sum", 0.0)
                    except KeyError:
                        pass
            except Exception:
                pass
    
    def record_battle(self, attacker: int, defender: int, attacker_won: bool, 
                      both_died: bool = False):
        """
        Record a battle outcome. Call this after each capture.
        
        Args:
            attacker: Piece type value of attacker (1-11, use abs())
            defender: Piece type value of defender (1-11, use abs())
            attacker_won: True if attacker captured defender
            both_died: True if both pieces were destroyed (equal rank)
        """
        try:
            atk_type = PieceType(abs(attacker))
            def_type = PieceType(abs(defender))
        except ValueError:
            return  # Invalid piece type
        
        atk_value = ANALYTICAL_VALUES.get(atk_type, 1.0)
        def_value = ANALYTICAL_VALUES.get(def_type, 1.0)
        
        if both_died:
            # Mutual destruction
            self.stats[atk_type].total_captured += 1
            self.stats[atk_type].captured_by_value_sum += def_value
            self.stats[def_type].total_captured += 1
            self.stats[def_type].captured_by_value_sum += atk_value
        elif attacker_won:
            # Attacker wins
            self.stats[atk_type].total_captures += 1
            self.stats[atk_type].capture_value_sum += def_value
            self.stats[def_type].total_captured += 1
            self.stats[def_type].captured_by_value_sum += atk_value
        else:
            # Defender wins
            self.stats[def_type].total_captures += 1
            self.stats[def_type].capture_value_sum += atk_value
            self.stats[atk_type].total_captured += 1
            self.stats[atk_type].captured_by_value_sum += def_value
    
    def record_game_end(self, winner: int, surviving_p1: Dict[int, int], 
                        surviving_p2: Dict[int, int]):
        """
        Record game end statistics.
        
        Args:
            winner: 1 for player 1, -1 for player 2, 0 for draw
            surviving_p1: Dict of {piece_type: count} for player 1
            surviving_p2: Dict of {piece_type: count} for player 2
        """
        self.games_tracked += 1
        
        # Record survival statistics for winning/losing players
        for piece_val, count in surviving_p1.items():
            try:
                pt = PieceType(abs(piece_val))
                if winner == 1:
                    self.stats[pt].games_won_with += count
                elif winner == -1:
                    self.stats[pt].games_lost_with += count
            except ValueError:
                pass
        
        for piece_val, count in surviving_p2.items():
            try:
                pt = PieceType(abs(piece_val))
                if winner == -1:
                    self.stats[pt].games_won_with += count
                elif winner == 1:
                    self.stats[pt].games_lost_with += count
            except ValueError:
                pass
    
    def calculate_empirical_values(self) -> Dict[PieceType, float]:
        """Calculate empirical piece values from tracked statistics."""
        raw_values = {}
        
        for pt in PieceType:
            stats = self.stats[pt]
            
            # Win correlation
            total_outcomes = stats.games_won_with + stats.games_lost_with
            if total_outcomes > 0:
                win_value = (stats.games_won_with / total_outcomes) * 10
            else:
                win_value = 5.0
            
            # Exchange value
            if stats.total_captures > 0:
                avg_captured = stats.capture_value_sum / stats.total_captures
            else:
                avg_captured = 0
            
            if stats.total_captured > 0:
                avg_captured_by = stats.captured_by_value_sum / stats.total_captured
            else:
                avg_captured_by = 0
            
            exchange_value = (avg_captured - avg_captured_by) / 2
            
            # Combine (simplified)
            raw_values[pt] = max(0.1, win_value * 0.4 + exchange_value * 0.3 + 
                                ANALYTICAL_VALUES.get(pt, 1.0) * 0.3)
        
        # Normalize to Scout = 1.0
        scout_value = raw_values.get(PieceType.SCOUT, 1.0)
        if scout_value == 0:
            scout_value = 1.0
        
        return {pt: val / scout_value for pt, val in raw_values.items()}
    
    def get_convergence_error(self) -> float:
        """Calculate mean squared error between empirical and analytical values."""
        empirical = self.calculate_empirical_values()
        
        total_error = 0.0
        count = 0
        for pt in PieceType:
            if pt in [PieceType.FLAG, PieceType.BOMB]:
                continue  # Skip immovable pieces
            analytical = ANALYTICAL_VALUES.get(pt, 1.0)
            emp = empirical.get(pt, 1.0)
            total_error += (analytical - emp) ** 2
            count += 1
        
        return (total_error / count) ** 0.5 if count > 0 else 0.0
    
    def log_comparison(self, episode: int = 0):
        """Print a comparison of empirical vs analytical values."""
        if self.games_tracked == 0:
            return
        
        empirical = self.calculate_empirical_values()
        error = self.get_convergence_error()
        
        print(f"\n📊 Piece Value Convergence (Episode {episode}, {self.games_tracked} games)")
        print(f"{'Piece':<12} {'Analytical':>10} {'Empirical':>10} {'Δ':>8}")
        print("-" * 42)
        
        for pt in sorted(PieceType, key=lambda x: ANALYTICAL_VALUES.get(x, 0), reverse=True):
            if pt in [PieceType.FLAG, PieceType.BOMB]:
                continue
            analytical = ANALYTICAL_VALUES.get(pt, 1.0)
            emp = empirical.get(pt, 1.0)
            delta = emp - analytical
            print(f"{pt.name:<12} {analytical:>10.2f} {emp:>10.2f} {delta:>+8.2f}")
        
        print(f"\nRMSE from Analytical: {error:.3f}")
    
    def save(self):
        """Save statistics to file."""
        data = {
            "games_tracked": self.games_tracked,
            "stats": {
                pt.name: {
                    "games_won_with": s.games_won_with,
                    "games_lost_with": s.games_lost_with,
                    "total_captures": s.total_captures,
                    "total_captured": s.total_captured,
                    "capture_value_sum": s.capture_value_sum,
                    "captured_by_value_sum": s.captured_by_value_sum,
                }
                for pt, s in self.stats.items()
            },
            "empirical_values": {
                pt.name: val for pt, val in self.calculate_empirical_values().items()
            },
            "analytical_values": {
                pt.name: val for pt, val in ANALYTICAL_VALUES.items()
            },
            "convergence_rmse": self.get_convergence_error()
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)


# Singleton instance for easy import
_tracker_instance = None

def get_tracker() -> PieceValueTracker:
    """Get the global tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PieceValueTracker()
    return _tracker_instance


if __name__ == "__main__":
    # Test the tracker
    tracker = PieceValueTracker()
    
    # Simulate some battles
    tracker.record_battle(10, 9, True)  # Marshal beats General
    tracker.record_battle(10, 1, False)  # Marshal loses to Spy (when attacked)
    tracker.record_battle(3, 11, True)  # Miner beats Bomb
    tracker.record_battle(2, 4, False)  # Scout loses to Sergeant
    
    # Simulate game end
    tracker.record_game_end(1, {10: 1, 9: 1}, {2: 3})  # P1 wins with Marshal, General
    
    tracker.log_comparison(episode=0)
    tracker.save()
