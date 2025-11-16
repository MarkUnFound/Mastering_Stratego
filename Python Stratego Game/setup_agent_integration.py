"""
Setup Agent Integration for Stratego Visual
Uses trained setup agent to create strategic piece placements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple

# Piece rank constants (matching stratego.py)
PIECE_COUNTS = {
    10: 1,  # Marshal
    9: 1,   # General
    8: 2,   # Colonel
    7: 3,   # Major
    6: 4,   # Captain
    5: 4,   # Lieutenant
    4: 4,   # Sergeant
    3: 5,   # Miner
    2: 8,   # Scout
    1: 1,   # Spy
    0: 6,   # Bomb
    -1: 1,  # Flag
}


class SetupNetwork(nn.Module):
    """Neural network for piece placement decisions"""
    
    def __init__(self, input_size: int = 400, hidden_size: int = 512, output_size: int = 400):
        super(SetupNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class StrategicSetupAgent:
    """
    Agent that creates strategic piece placements for Stratego
    Uses heuristics and learned strategies
    """
    
    def __init__(self, player_id: int, model_path: str = None, device=None):
        """
        Initialize setup agent
        
        Args:
            player_id: Player ID (1 or 2)
            model_path: Path to trained model (optional)
            device: PyTorch device
        """
        self.player_id = player_id
        self.device = device if device else torch.device('cpu')
        
        # Try to load trained model
        self.use_model = False
        if model_path:
            try:
                self.model = SetupNetwork(400, 512, 400).to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'q_network' in checkpoint:
                    self.model.load_state_dict(checkpoint['q_network'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.use_model = True
                print(f"Loaded setup model from {model_path}")
            except Exception as e:
                print(f"Could not load setup model: {e}")
                print("Using heuristic-based setup instead")
                self.use_model = False
    
    def create_setup(self, board) -> List[Tuple]:
        """
        Create strategic piece placement
        
        Args:
            board: Board object
            
        Returns:
            List of (position, piece) placements
        """
        if self.use_model:
            return self._create_learned_setup(board)
        else:
            return self._create_heuristic_setup(board)
    
    def _create_learned_setup(self, board) -> List[Tuple]:
        """
        Create setup using trained neural network
        
        Args:
            board: Board object
            
        Returns:
            List of (position, piece) placements
        """
        # Get available positions
        if self.player_id == 1:
            rows = [6, 7, 8, 9]
        else:
            rows = [0, 1, 2, 3]
        
        positions = [(r, c) for r in rows for c in range(10) if not board.is_lake(r, c)]
        
        # Create pieces
        pieces = []
        for rank, count in PIECE_COUNTS.items():
            for _ in range(count):
                pieces.append(rank)
        
        # If model fails or not enough positions, fall back to heuristic
        if len(positions) != 40 or len(pieces) != 40:
            return self._create_heuristic_setup(board)
        
        # Use greedy placement with model predictions
        placements = []
        remaining_positions = positions.copy()
        remaining_pieces = pieces.copy()
        
        # Place pieces iteratively using model predictions
        for i in range(min(40, len(remaining_pieces))):
            if not remaining_pieces or not remaining_positions:
                break
            
            # Select next piece to place (in order)
            piece = remaining_pieces[0]
            remaining_pieces = remaining_pieces[1:]
            
            # Create state representation for current placement state
            state = self._get_state_representation(pieces, positions, placements, piece, remaining_positions)
            
            # Get Q-values from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # Model outputs Q-values for 400 positions
                q_values = self.model(state_tensor).squeeze().cpu().numpy()
            
            # Map Q-values to available positions
            # The model was trained with positions mapped to indices 0-39
            # For player 1: rows 6-9 map to indices 0-39 (excluding lakes)
            # For player 2: rows 0-3 map to indices 0-39 (excluding lakes)
            position_scores = {}
            
            # Create position to index mapping
            pos_to_index = {}
            if self.player_id == 1:
                # Player 1: rows 6-9
                idx = 0
                for r in range(6, 10):
                    for c in range(10):
                        pos = (r, c)
                        if not board.is_lake(r, c):
                            pos_to_index[pos] = idx
                            idx += 1
            else:
                # Player 2: rows 0-3
                idx = 0
                for r in range(4):
                    for c in range(10):
                        pos = (r, c)
                        if not board.is_lake(r, c):
                            pos_to_index[pos] = idx
                            idx += 1
            
            # Get Q-values for remaining positions
            for pos in remaining_positions:
                if pos in pos_to_index:
                    q_idx = pos_to_index[pos]
                    if 0 <= q_idx < len(q_values):
                        position_scores[pos] = float(q_values[q_idx])
                    else:
                        position_scores[pos] = 0.0
                else:
                    position_scores[pos] = 0.0
            
            # Select best position based on Q-value (greedy)
            if position_scores:
                best_position = max(position_scores.keys(), key=lambda p: position_scores[p])
            else:
                # Fallback to random if mapping fails
                best_position = random.choice(remaining_positions)
            
            # Place piece and remove from available
            remaining_positions.remove(best_position)
            placements.append((best_position, piece))
            
        return placements
    
    def _get_state_representation(self, all_pieces: List[int], all_positions: List[Tuple[int, int]], 
                                  current_placements: List[Tuple], current_piece: int,
                                  remaining_positions: List[Tuple[int, int]]) -> List[float]:
        """
        Create state representation for model input
        
        Args:
            all_pieces: List of all pieces (40 pieces)
            all_positions: List of all available positions (40 positions)
            current_placements: List of (position, piece) tuples already placed
            current_piece: Current piece to place
            remaining_positions: List of remaining available positions
            
        Returns:
            State vector of 400 features
        """
        # Create feature vector matching training format
        # Format: For each of 40 pieces, we have: piece_type, row, col, piece_index (4 features each = 160, padded to 400)
        features = []
        
        # Track which pieces have been placed
        placed_pieces = [False] * len(all_pieces)
        piece_positions = {}
        
        for placed_pos, placed_piece in current_placements:
            # Find first unplaced instance of this piece type
            for i, piece in enumerate(all_pieces):
                if not placed_pieces[i] and piece == placed_piece:
                    placed_pieces[i] = True
                    piece_positions[i] = placed_pos
                    break
        
        # Build state representation for all 40 pieces
        for piece_index in range(40):
            if piece_index < len(all_pieces):
                piece_type = all_pieces[piece_index]
                
                # Check if this piece is placed
                if piece_index in piece_positions:
                    pos = piece_positions[piece_index]
                    row = float(pos[0]) / 9.0  # Normalize row to [0, 1]
                    col = float(pos[1]) / 9.0  # Normalize col to [0, 1]
                elif piece_type == current_piece and piece_index < len(current_placements) + 1:
                    # This is the piece we're currently placing (not yet placed)
                    row = 0.0
                    col = 0.0
                else:
                    # Piece not yet placed
                    row = 0.0
                    col = 0.0
                
                # Add features for this piece: [piece_type, row, col, piece_index]
                features.extend([
                    float(piece_type),
                    row,
                    col,
                    float(piece_index) / 40.0,  # Normalize index to [0, 1]
                ])
            else:
                # Pad with zeros if somehow we have fewer pieces
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Pad to exactly 400 features (matches model input size)
        while len(features) < 400:
            features.append(0.0)
        features = features[:400]
        
        return features
    
    def _create_heuristic_setup(self, board) -> List[Tuple]:
        """
        Create setup using strategic heuristics
        
        Strategy:
        - Flag in back corner with Bomb protection
        - High-value pieces (Marshal, General) near flag
        - Bombs around flag and in chokepoints
        - Scouts on flanks for mobility
        - Miners scattered (for bomb defusal)
        - Aggressive pieces (Captains, Lieutenants) in front
        
        Args:
            board: Board object
            
        Returns:
            List of (position, piece) placements
        """
        placements = []
        
        # Determine player's territory
        if self.player_id == 1:
            rows = [6, 7, 8, 9]  # Bottom rows
            back_row = 9
            mid_back = 8
            mid_front = 7
            front_row = 6
        else:
            rows = [0, 1, 2, 3]  # Top rows
            back_row = 0
            mid_back = 1
            mid_front = 2
            front_row = 3
        
        # Get available positions by row
        positions_by_row = {row: [] for row in rows}
        for r in rows:
            for c in range(10):
                if not board.is_lake(r, c):
                    positions_by_row[r].append(c)
        
        # Create pieces categorized by role
        high_value = [10, 9]  # Marshal, General
        bombs = [0] * 6
        flag = [-1]
        defensive = [8, 8, 7, 7, 7]  # Colonels, Majors
        miners = [3] * 5
        scouts = [2] * 8
        attackers = [6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4]  # Captains, Lieutenants, Sergeants
        spy = [1]
        
        used_positions = set()
        
        def place_piece(rank, row, col_preference=None):
            """Helper to place a piece"""
            if col_preference is not None and col_preference in positions_by_row[row] and (row, col_preference) not in used_positions:
                pos = (row, col_preference)
            else:
                # Find first available position in row
                available = [c for c in positions_by_row[row] if (row, c) not in used_positions]
                if not available:
                    # Try other rows
                    for r in rows:
                        available = [c for c in positions_by_row[r] if (r, c) not in used_positions]
                        if available:
                            pos = (r, random.choice(available))
                            break
                    else:
                        return False
                else:
                    pos = (row, random.choice(available))
            
            used_positions.add(pos)
            placements.append((pos, rank))
            return True
        
        # 1. Place Flag in back corner
        flag_col = random.choice([0, 1, 8, 9])  # Corner column
        place_piece(flag[0], back_row, flag_col)
        
        # 2. Place Bombs around flag and strategic positions
        # Bombs next to flag
        bomb_positions = []
        if flag_col <= 1:
            bomb_positions = [flag_col + 1, flag_col, flag_col]
        else:
            bomb_positions = [flag_col - 1, flag_col, flag_col]
        
        for i, bomb in enumerate(bombs[:3]):
            if i < len(bomb_positions):
                place_piece(bomb, back_row, bomb_positions[i])
            else:
                place_piece(bomb, back_row)
        
        # Remaining bombs in mid-back row
        for bomb in bombs[3:]:
            place_piece(bomb, mid_back)
        
        # 3. Place high-value pieces near flag
        for piece in high_value:
            place_piece(piece, back_row)
        
        # 4. Place defensive pieces in back rows
        for piece in defensive:
            place_piece(piece, mid_back)
        
        # 5. Place Spy in back for protection
        place_piece(spy[0], mid_back)
        
        # 6. Place Miners scattered (bomb defusal capability)
        for i, miner in enumerate(miners):
            if i < 2:
                place_piece(miner, mid_front)
            else:
                place_piece(miner, front_row)
        
        # 7. Place Scouts on flanks for mobility
        for i, scout in enumerate(scouts):
            if i < 4:
                # Flanks in front row
                if i % 2 == 0:
                    place_piece(scout, front_row, i // 2)
                else:
                    place_piece(scout, front_row, 9 - i // 2)
            else:
                place_piece(scout, mid_front)
        
        # 8. Place attackers in front rows
        for piece in attackers:
            if len([p for p in placements if p[0][0] == front_row]) < 8:
                place_piece(piece, front_row)
            else:
                place_piece(piece, mid_front)
        
        return placements
    
    def apply_setup_to_board(self, board, Piece):
        """
        Apply the setup to the board
        
        Args:
            board: Board object
            Piece: Piece class for creating piece objects
        """
        placements = self.create_setup(board)
        
        for pos, rank in placements:
            piece = Piece(owner=self.player_id, rank=rank)
            board.set(pos, piece)
        
        return len(placements)
