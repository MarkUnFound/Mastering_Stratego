"""
Smart Heuristic Setup for Stratego
Fast piece placement with basic strategic rules:
1. Flag in back corner with bomb protection
2. Bombs surrounding flag
3. High-value pieces in back rows
4. Scouts in front rows for scouting
5. Miners scattered (for bomb defusal)
"""

import random
from typing import List, Tuple
from piece import PieceType


def smart_heuristic_setup(pieces: List[PieceType], 
                          valid_positions: List[Tuple[int, int]], 
                          player_id: int) -> List[Tuple[PieceType, Tuple[int, int]]]:
    """
    Fast heuristic setup that places pieces strategically without neural network.
    
    Args:
        pieces: List of 40 PieceType pieces to place
        valid_positions: List of valid (row, col) positions for this player
        player_id: 1 for player 1 (rows 6-9), -1 for player 2 (rows 0-3)
        
    Returns:
        List of (piece, position) tuples
    """
    placement = []
    remaining_positions = valid_positions.copy()
    
    # Group pieces by type
    piece_groups = {
        'flag': [p for p in pieces if p == PieceType.FLAG],
        'bombs': [p for p in pieces if p == PieceType.BOMB],
        'marshal': [p for p in pieces if p == PieceType.MARSHAL],
        'general': [p for p in pieces if p == PieceType.GENERAL],
        'miners': [p for p in pieces if p == PieceType.MINER],
        'scouts': [p for p in pieces if p == PieceType.SCOUT],
        'spy': [p for p in pieces if p == PieceType.SPY],
        'high_value': [p for p in pieces if p.value in [7, 8]],  # Colonel, Major
        'mid_value': [p for p in pieces if p.value in [5, 6]],   # Captain, Lieutenant
        'low_value': [p for p in pieces if p.value in [4]],      # Sergeant
    }
    
    # Determine back and front rows based on player
    if player_id == 1:
        # Player 1: back row is 6, front row is 9
        back_rows = [6, 7]
        front_rows = [8, 9]
        corner_positions = [(6, 0), (6, 9), (7, 0), (7, 9)]  # Corners
    else:
        # Player 2: back row is 3, front row is 0
        back_rows = [2, 3]
        front_rows = [0, 1]
        corner_positions = [(3, 0), (3, 9), (2, 0), (2, 9)]  # Corners
    
    def get_positions_in_rows(rows):
        return [p for p in remaining_positions if p[0] in rows]
    
    def get_adjacent_positions(pos):
        r, c = pos
        adj = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [p for p in adj if p in remaining_positions]
    
    def place_piece(piece, position):
        placement.append((piece, position))
        remaining_positions.remove(position)
    
    # 1. Place Flag in back corner
    flag = piece_groups['flag'][0]
    flag_candidates = [p for p in corner_positions if p in remaining_positions]
    if not flag_candidates:
        flag_candidates = get_positions_in_rows(back_rows)
    if flag_candidates:
        flag_pos = random.choice(flag_candidates)
        place_piece(flag, flag_pos)
    else:
        # Fallback: any position
        flag_pos = random.choice(remaining_positions)
        place_piece(flag, flag_pos)
    
    # 2. Place Bombs around Flag (2-3 bombs adjacent)
    bombs = piece_groups['bombs']
    adjacent_to_flag = get_adjacent_positions(flag_pos)
    bombs_placed_near_flag = 0
    
    for bomb in bombs[:3]:  # Place up to 3 bombs near flag
        if adjacent_to_flag and bombs_placed_near_flag < 3:
            pos = random.choice(adjacent_to_flag)
            place_piece(bomb, pos)
            adjacent_to_flag.remove(pos)
            bombs_placed_near_flag += 1
        else:
            # Place remaining bombs in back rows
            back_positions = get_positions_in_rows(back_rows)
            if back_positions:
                pos = random.choice(back_positions)
                place_piece(bomb, pos)
    
    # Place remaining bombs randomly in back
    for bomb in bombs[3:]:
        back_positions = get_positions_in_rows(back_rows)
        if back_positions:
            pos = random.choice(back_positions)
            place_piece(bomb, pos)
        elif remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(bomb, pos)
    
    # 3. Place Marshal and General in protected positions (back rows, not corners)
    for piece in piece_groups['marshal'] + piece_groups['general']:
        back_positions = get_positions_in_rows(back_rows)
        # Avoid corners (too predictable)
        back_positions = [p for p in back_positions if p[1] not in [0, 9]]
        if back_positions:
            pos = random.choice(back_positions)
            place_piece(piece, pos)
        elif remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    # 4. Place Spy near Marshal area (for Marshal hunting)
    for piece in piece_groups['spy']:
        back_positions = get_positions_in_rows(back_rows)
        if back_positions:
            pos = random.choice(back_positions)
            place_piece(piece, pos)
        elif remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    # 5. Place Scouts in front rows (for early scouting)
    for piece in piece_groups['scouts']:
        front_positions = get_positions_in_rows(front_rows)
        if front_positions:
            pos = random.choice(front_positions)
            place_piece(piece, pos)
        elif remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    # 6. Scatter Miners (important for bomb defusal) - mix of front and back
    for i, piece in enumerate(piece_groups['miners']):
        if i < 2:
            # 2 miners in front for aggressive bomb clearing
            front_positions = get_positions_in_rows(front_rows)
            if front_positions:
                pos = random.choice(front_positions)
                place_piece(piece, pos)
                continue
        # Rest scattered
        if remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    # 7. Place high-value pieces in back rows for protection
    for piece in piece_groups['high_value']:
        back_positions = get_positions_in_rows(back_rows)
        if back_positions:
            pos = random.choice(back_positions)
            place_piece(piece, pos)
        elif remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    # 8. Fill remaining pieces randomly
    remaining_pieces = []
    for p in pieces:
        already_placed = any(placed_piece == p for placed_piece, _ in placement)
        if not already_placed:
            remaining_pieces.append(p)
    
    # Actually, we need to track which pieces we've placed
    placed_pieces = [pp for pp, _ in placement]
    remaining_to_place = pieces.copy()
    for pp in placed_pieces:
        if pp in remaining_to_place:
            remaining_to_place.remove(pp)
    
    # Shuffle and place remaining
    random.shuffle(remaining_to_place)
    for piece in remaining_to_place:
        if remaining_positions:
            pos = random.choice(remaining_positions)
            place_piece(piece, pos)
    
    return placement


class HeuristicSetupAgent:
    """
    Fast setup agent using heuristics instead of neural network.
    Compatible with SetupAgent interface.
    """
    
    def __init__(self, player_id: int, device=None):
        self.player_id = player_id
        self.device = device
        self.name = "HeuristicSetup"
    
    def place_pieces(self, pieces: List[PieceType], 
                     available_positions: List[Tuple[int, int]]) -> List[Tuple[PieceType, Tuple[int, int]]]:
        """Place pieces using smart heuristic."""
        return smart_heuristic_setup(pieces, available_positions, self.player_id)
    
    def reset_noise(self):
        """No-op for compatibility."""
        pass
    
    def finish_episode(self, reward: float):
        """No-op for compatibility."""
        pass
    
    def replay(self):
        """No-op for compatibility."""
        return None
    
    def get_average_policy_loss(self, window: int = 100) -> float:
        """Returns 0 for compatibility."""
        return 0.0
    
    def save_model(self, path: str):
        """No-op for compatibility."""
        pass
    
    def load_model(self, path: str):
        """No-op for compatibility."""
        pass
