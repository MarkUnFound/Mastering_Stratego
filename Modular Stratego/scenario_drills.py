"""
Scenario Drills Module for Phase 5 Curriculum

Provides specialized endgame training scenarios:
- Miner vs Bombs: Practice bomb defusal
- Marshal Duel: Track and eliminate enemy Marshal
- Flag Hunt: Endgame flag search patterns
"""

import random
from typing import List, Tuple, Optional
from piece import PieceType


class ScenarioDrill:
    """
    Generator for specialized training scenarios.
    Creates custom piece placements for specific tactical situations.
    """
    
    def __init__(self):
        self.scenarios = {
            'miner_vs_bombs': self.setup_miner_vs_bombs,
            'marshal_duel': self.setup_marshal_duel,
            'flag_hunt': self.setup_flag_hunt,
            'scout_rush': self.setup_scout_rush,
            'endgame_standard': self.setup_endgame_standard
        }
    
    def get_scenario(self, scenario_name: str = None) -> Tuple[List, List]:
        """
        Get a scenario by name or random.
        
        Returns:
            Tuple of (p1_placement, p2_placement)
        """
        if scenario_name is None:
            scenario_name = random.choice(list(self.scenarios.keys()))
        
        if scenario_name in self.scenarios:
            return self.scenarios[scenario_name]()
        else:
            return self.setup_endgame_standard()
    
    def setup_miner_vs_bombs(self) -> Tuple[List, List]:
        """
        Scenario: Agent has 1 Miner, enemy Flag surrounded by Bombs.
        Goal: Practice bomb defusal and flag capture.
        """
        p1_placement = []
        p2_placement = []
        
        # P1: Minimal pieces for flag capture
        # Rows 6-9 for P1
        p1_pieces = [
            (PieceType.FLAG, (9, 5)),      # Our flag
            (PieceType.MINER, (7, 4)),     # Key piece
            (PieceType.MINER, (7, 5)),     # Backup miner
            (PieceType.SCOUT, (6, 3)),     # Scout for recon
            (PieceType.SCOUT, (6, 6)),
            (PieceType.CAPTAIN, (8, 4)),   # Protection
            (PieceType.CAPTAIN, (8, 5)),
            (PieceType.BOMB, (9, 4)),      # Flag protection
            (PieceType.BOMB, (9, 6)),
        ]
        
        # P2: Flag protected by bombs
        # Rows 0-3 for P2
        p2_pieces = [
            (PieceType.FLAG, (0, 5)),           # Target flag
            (PieceType.BOMB, (0, 4)),           # Bomb ring
            (PieceType.BOMB, (0, 6)),
            (PieceType.BOMB, (1, 4)),
            (PieceType.BOMB, (1, 5)),
            (PieceType.BOMB, (1, 6)),
            (PieceType.SERGEANT, (2, 5)),       # Light defense
            (PieceType.LIEUTENANT, (3, 4)),
            (PieceType.LIEUTENANT, (3, 6)),
        ]
        
        # Fill remaining positions with Scouts to reach 40 pieces each
        p1_placement = self._fill_remaining(p1_pieces, 1)
        p2_placement = self._fill_remaining(p2_pieces, -1)
        
        return p1_placement, p2_placement
    
    def setup_marshal_duel(self) -> Tuple[List, List]:
        """
        Scenario: Equal material, goal is to find and eliminate enemy Marshal.
        """
        # Symmetric setup with hidden Marshals
        common_pieces = [
            PieceType.FLAG,
            PieceType.MARSHAL,
            PieceType.GENERAL,
            PieceType.SPY,       # Can kill Marshal
            PieceType.CAPTAIN,
            PieceType.CAPTAIN,
            PieceType.LIEUTENANT,
            PieceType.LIEUTENANT,
            PieceType.SERGEANT,
            PieceType.SERGEANT,
            PieceType.SCOUT,
            PieceType.SCOUT,
            PieceType.MINER,
            PieceType.BOMB,
            PieceType.BOMB,
        ]
        
        # Random positions in valid areas
        p1_positions = self._get_random_positions(1, len(common_pieces))
        p2_positions = self._get_random_positions(-1, len(common_pieces))
        
        p1_placement = [(piece, pos) for piece, pos in zip(common_pieces, p1_positions)]
        p2_placement = [(piece, pos) for piece, pos in zip(common_pieces, p2_positions)]
        
        # Fill to 40
        p1_placement = self._fill_remaining(p1_placement, 1)
        p2_placement = self._fill_remaining(p2_placement, -1)
        
        return p1_placement, p2_placement
    
    def setup_flag_hunt(self) -> Tuple[List, List]:
        """
        Scenario: Reduced material, focus on finding hidden flag.
        """
        p1_pieces = [
            (PieceType.FLAG, (9, random.randint(0, 9))),
            (PieceType.MINER, (7, 3)),
            (PieceType.MINER, (7, 6)),
            (PieceType.SCOUT, (6, 1)),
            (PieceType.SCOUT, (6, 4)),
            (PieceType.SCOUT, (6, 5)),
            (PieceType.SCOUT, (6, 8)),
            (PieceType.CAPTAIN, (8, 4)),
            (PieceType.CAPTAIN, (8, 5)),
            (PieceType.BOMB, (9, 4)),
            (PieceType.BOMB, (9, 5)),
        ]
        
        # P2 flag in random corner
        flag_col = random.choice([0, 1, 8, 9])
        p2_pieces = [
            (PieceType.FLAG, (0, flag_col)),
            (PieceType.BOMB, (0, max(0, flag_col - 1))),
            (PieceType.BOMB, (0, min(9, flag_col + 1))),
            (PieceType.BOMB, (1, flag_col)),
            (PieceType.SERGEANT, (2, 4)),
            (PieceType.SERGEANT, (2, 5)),
            (PieceType.SCOUT, (3, 1)),
            (PieceType.SCOUT, (3, 8)),
            (PieceType.MINER, (3, 4)),
            (PieceType.MINER, (3, 5)),
        ]
        
        p1_placement = self._fill_remaining(p1_pieces, 1)
        p2_placement = self._fill_remaining(p2_pieces, -1)
        
        return p1_placement, p2_placement
    
    def setup_scout_rush(self) -> Tuple[List, List]:
        """
        Scenario: Practice Scout movement and reconnaissance.
        """
        p1_pieces = [
            (PieceType.FLAG, (9, 5)),
            (PieceType.SCOUT, (6, 0)),
            (PieceType.SCOUT, (6, 1)),
            (PieceType.SCOUT, (6, 2)),
            (PieceType.SCOUT, (6, 7)),
            (PieceType.SCOUT, (6, 8)),
            (PieceType.SCOUT, (6, 9)),
            (PieceType.CAPTAIN, (7, 4)),
            (PieceType.CAPTAIN, (7, 5)),
            (PieceType.MINER, (8, 4)),
            (PieceType.MINER, (8, 5)),
            (PieceType.BOMB, (9, 4)),
            (PieceType.BOMB, (9, 6)),
        ]
        
        p2_pieces = [
            (PieceType.FLAG, (0, 5)),
            (PieceType.LIEUTENANT, (1, 3)),
            (PieceType.LIEUTENANT, (1, 6)),
            (PieceType.SERGEANT, (2, 2)),
            (PieceType.SERGEANT, (2, 7)),
            (PieceType.CAPTAIN, (3, 4)),
            (PieceType.CAPTAIN, (3, 5)),
            (PieceType.BOMB, (0, 4)),
            (PieceType.BOMB, (0, 6)),
            (PieceType.BOMB, (1, 5)),
        ]
        
        p1_placement = self._fill_remaining(p1_pieces, 1)
        p2_placement = self._fill_remaining(p2_pieces, -1)
        
        return p1_placement, p2_placement
    
    def setup_endgame_standard(self) -> Tuple[List, List]:
        """
        Scenario: Standard endgame with reduced material.
        Simulates move 100+ game state.
        """
        p1_pieces = [
            (PieceType.FLAG, (9, 5)),
            (PieceType.MARSHAL, (6, 4)),
            (PieceType.GENERAL, (6, 5)),
            (PieceType.SPY, (7, 3)),
            (PieceType.MINER, (7, 5)),
            (PieceType.MINER, (7, 6)),
            (PieceType.CAPTAIN, (8, 4)),
            (PieceType.CAPTAIN, (8, 5)),
            (PieceType.SCOUT, (6, 2)),
            (PieceType.BOMB, (9, 4)),
            (PieceType.BOMB, (9, 6)),
        ]
        
        p2_pieces = [
            (PieceType.FLAG, (0, 5)),
            (PieceType.MARSHAL, (3, 5)),
            (PieceType.GENERAL, (3, 4)),
            (PieceType.SPY, (2, 6)),
            (PieceType.MINER, (2, 4)),
            (PieceType.MINER, (2, 5)),
            (PieceType.CAPTAIN, (1, 4)),
            (PieceType.CAPTAIN, (1, 5)),
            (PieceType.SCOUT, (3, 7)),
            (PieceType.BOMB, (0, 4)),
            (PieceType.BOMB, (0, 6)),
        ]
        
        p1_placement = self._fill_remaining(p1_pieces, 1)
        p2_placement = self._fill_remaining(p2_pieces, -1)
        
        return p1_placement, p2_placement
    
    def _get_random_positions(self, player_id: int, count: int) -> List[Tuple[int, int]]:
        """Get random valid positions for a player."""
        if player_id == 1:
            rows = range(6, 10)
        else:
            rows = range(0, 4)
        
        positions = [(r, c) for r in rows for c in range(10)]
        random.shuffle(positions)
        return positions[:count]
    
    def _fill_remaining(self, pieces: List[Tuple], player_id: int) -> List[Tuple]:
        """
        Fill remaining positions to reach 40 pieces.
        Uses Scouts and Sergeants as filler.
        """
        existing_positions = {pos for _, pos in pieces}
        
        if player_id == 1:
            rows = range(6, 10)
        else:
            rows = range(0, 4)
        
        all_positions = [(r, c) for r in rows for c in range(10)]
        available = [pos for pos in all_positions if pos not in existing_positions]
        random.shuffle(available)
        
        result = list(pieces)
        filler_types = [PieceType.SCOUT, PieceType.SERGEANT, PieceType.LIEUTENANT]
        
        while len(result) < 40 and available:
            pos = available.pop()
            piece_type = random.choice(filler_types)
            result.append((piece_type, pos))
        
        return result


# Singleton instance
_scenario_drill = None

def get_scenario_drill() -> ScenarioDrill:
    """Get the scenario drill singleton."""
    global _scenario_drill
    if _scenario_drill is None:
        _scenario_drill = ScenarioDrill()
    return _scenario_drill


def get_random_scenario() -> Tuple[List, List]:
    """Get a random training scenario."""
    return get_scenario_drill().get_scenario()
