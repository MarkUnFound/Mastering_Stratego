"""
Setup Agent Integration for Stratego Visual
Uses trained setup agent to create strategic piece placements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    Uses trained neural network model (setup_agent_final.pth)
    """
    
    def __init__(self, player_id: int, model_path: str = "setup_agent_final.pth", device=None):
        """
        Initialize setup agent
        
        Args:
            player_id: Player ID (1 or 2)
            model_path: Path to trained model (default: setup_agent_final.pth)
            device: PyTorch device
            
        Raises:
            FileNotFoundError: If model file is not found
            Exception: If model loading fails
        """
        self.player_id = player_id
        self.device = device if device else torch.device('cpu')
        
        # Load trained model (required)
        self.model = SetupNetwork(400, 512, 400).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'q_network' in checkpoint:
            self.model.load_state_dict(checkpoint['q_network'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"Loaded setup model from {model_path}")
    
    def create_setup(self, board) -> List[Tuple]:
        """
        Create strategic piece placement using trained neural network
        
        Args:
            board: Board object
            
        Returns:
            List of (position, piece) placements
            
        Raises:
            ValueError: If board configuration is invalid
        """
        return self._create_learned_setup(board)
    
    def _create_learned_setup(self, board) -> List[Tuple]:
        """
        Create setup using trained neural network
        
        Args:
            board: Board object
            
        Returns:
            List of (position, piece) placements
            
        Raises:
            ValueError: If board configuration is invalid (not exactly 40 positions or pieces)
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
        
        # Validate board configuration
        if len(positions) != 40 or len(pieces) != 40:
            raise ValueError(
                f"Invalid board configuration: expected 40 positions and 40 pieces, "
                f"got {len(positions)} positions and {len(pieces)} pieces"
            )
        
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
            if not position_scores:
                raise ValueError(
                    f"Failed to map positions to Q-values. Available positions: {len(remaining_positions)}, "
                    f"Position mapping size: {len(pos_to_index)}"
                )
            best_position = max(position_scores.keys(), key=lambda p: position_scores[p])
            
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
