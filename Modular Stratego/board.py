
# stratego_modular/board.py

import torch
from typing import Tuple, List, Set
from piece import PieceType

BOARD_SIZE = 10
EMPTY_SQUARE = 0
LAKE_SQUARE = -13
HIDDEN_PIECE = -20  # Changed from -3 to avoid ambiguity with Scout (value 3, or -3 for Agent 2)

class Board:
    """Represents the game board with hidden information management."""
    
    def __init__(self, device):
        self.device = device
        self.board_size = BOARD_SIZE
        self.lakes = torch.tensor([(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)], device=device)
        self.actual_board = torch.zeros((self.board_size, self.board_size), dtype=torch.int8, device=device)
        self.visible_board_p1 = torch.zeros((self.board_size, self.board_size), dtype=torch.int8, device=device)
        self.visible_board_p2 = torch.zeros((self.board_size, self.board_size), dtype=torch.int8, device=device)
        
    def reset(self):
        """Reset the board to initial state."""
        self.actual_board.fill_(EMPTY_SQUARE)
        self.visible_board_p1.fill_(EMPTY_SQUARE)
        self.visible_board_p2.fill_(EMPTY_SQUARE)
        
        # Place lakes
        for r, c in self.lakes:
            self.actual_board[r, c] = LAKE_SQUARE
            self.visible_board_p1[r, c] = LAKE_SQUARE
            self.visible_board_p2[r, c] = LAKE_SQUARE
            
    def setup_pieces(self, p1_pieces: List[Tuple[PieceType, Tuple[int, int]]], 
                     p2_pieces: List[Tuple[PieceType, Tuple[int, int]]]):
        """Set up pieces on the board for both players."""
        # Place Player 1 pieces
        for piece_type, (r, c) in p1_pieces:
            self.actual_board[r, c] = piece_type.value
            self.visible_board_p1[r, c] = piece_type.value
            self.visible_board_p2[r, c] = HIDDEN_PIECE
            
        # Place Player 2 pieces
        for piece_type, (r, c) in p2_pieces:
            self.actual_board[r, c] = -piece_type.value
            self.visible_board_p2[r, c] = -piece_type.value
            self.visible_board_p1[r, c] = HIDDEN_PIECE
            
    def get_visible_board(self, player: int) -> torch.Tensor:
        """Get the board as visible to the specified player."""
        if player == 1:
            return self.visible_board_p1
        else:
            return self.visible_board_p2
            
    def move_piece(self, player: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """
        Move a piece on the board, updating visibility for both players.
        Enemy pieces remain hidden unless they've been revealed.
        """
        fr, fc = from_pos
        tr, tc = to_pos
        
        moving_piece = self.actual_board[fr, fc].item()
        target_piece = self.actual_board[tr, tc].item()
        
        # Check if moving piece has been revealed to opponent (check BEFORE updating actual board)
        # For player 1 moving: check if p2 can see it
        # For player -1 moving: check if p1 can see it
        if player == 1:
            # Check if opponent can see this piece (either revealed or it's their own piece)
            moving_piece_revealed_to_p2 = (self.visible_board_p2[fr, fc].item() == moving_piece)
        else:
            # Check if opponent can see this piece (either revealed or it's their own piece)
            moving_piece_revealed_to_p1 = (self.visible_board_p1[fr, fc].item() == moving_piece)
        
        # Update actual board
        self.actual_board[tr, tc] = moving_piece
        self.actual_board[fr, fc] = EMPTY_SQUARE
        
        # Update visible boards
        if player == 1:
            # Player 1's view: own piece moves (always visible to self)
            self.visible_board_p1[tr, tc] = moving_piece
            self.visible_board_p1[fr, fc] = EMPTY_SQUARE
            
            # Player 2's view: enemy piece moves
            self.visible_board_p2[fr, fc] = EMPTY_SQUARE
            if target_piece == EMPTY_SQUARE:
                # Moving to empty square - show as hidden unless revealed
                if moving_piece_revealed_to_p2:
                    self.visible_board_p2[tr, tc] = moving_piece
                else:
                    self.visible_board_p2[tr, tc] = HIDDEN_PIECE
            else:
                # Moving to occupied square (battle) - piece is already revealed by reveal_pieces
                # After reveal_pieces, the piece should be visible, so show it
                # reveal_pieces is called before move_piece in battle scenarios
                self.visible_board_p2[tr, tc] = moving_piece  # Already revealed by reveal_pieces
        else:
            # Player -1's view: own piece moves (always visible to self)
            self.visible_board_p2[tr, tc] = moving_piece
            self.visible_board_p2[fr, fc] = EMPTY_SQUARE
            
            # Player 1's view: enemy piece moves
            self.visible_board_p1[fr, fc] = EMPTY_SQUARE
            if target_piece == EMPTY_SQUARE:
                # Moving to empty square - show as hidden unless revealed
                if moving_piece_revealed_to_p1:
                    self.visible_board_p1[tr, tc] = moving_piece
                else:
                    self.visible_board_p1[tr, tc] = HIDDEN_PIECE
            else:
                # Moving to occupied square (battle) - piece is already revealed by reveal_pieces
                # After reveal_pieces, the piece should be visible, so show it
                # reveal_pieces is called before move_piece in battle scenarios
                self.visible_board_p1[tr, tc] = moving_piece  # Already revealed by reveal_pieces
            
    def reveal_pieces(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """Reveal pieces at two positions to both players after a battle."""
        r1, c1 = pos1
        r2, c2 = pos2
        
        piece1 = self.actual_board[r1, c1].item()
        piece2 = self.actual_board[r2, c2].item()
        
        # Reveal to both players, preserving ownership (sign)
        self.visible_board_p1[r1, c1] = piece1
        self.visible_board_p1[r2, c2] = piece2
        self.visible_board_p2[r1, c1] = piece1
        self.visible_board_p2[r2, c2] = piece2
        
    def is_valid_target(self, player: int, r: int, c: int) -> bool:
        """Check if a target square is valid for a move by the given player."""
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
            
        visible_board = self.get_visible_board(player)
        target_val = visible_board[r, c].item()
        
        # Target cannot be a lake
        if target_val == LAKE_SQUARE:
            return False
            
        # If square is empty, it's a valid target
        if target_val == EMPTY_SQUARE:
            return True
            
        # If square has an enemy piece (opposite sign), it's a valid target
        # Player 1 pieces are positive, Player -1 pieces are negative
        return (target_val * player) < 0

    def place_piece(self, r: int, c: int, value: int):
        """Place a piece on the board and update visibility."""
        self.actual_board[r, c] = value
        
        # Update visibility
        # If value > 0 (Player 1), visible to P1, hidden to P2
        # If value < 0 (Player 2), visible to P2, hidden to P1
        if value > 0:
            self.visible_board_p1[r, c] = value
            self.visible_board_p2[r, c] = HIDDEN_PIECE
        elif value < 0:
            self.visible_board_p2[r, c] = value
            self.visible_board_p1[r, c] = HIDDEN_PIECE