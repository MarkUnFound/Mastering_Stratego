# stratego_modular/board.py

import torch
from typing import Tuple, List, Set
from stratego_modular.piece import PieceType

BOARD_SIZE = 10
EMPTY_SQUARE = 0
LAKE_SQUARE = -13
HIDDEN_PIECE = -3

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
        """Move a piece on the board, updating visibility for both players."""
        fr, fc = from_pos
        tr, tc = to_pos
        
        moving_piece = self.actual_board[fr, fc].item()
        target_piece = self.actual_board[tr, tc].item()
        
        # Update actual board
        self.actual_board[tr, tc] = moving_piece
        self.actual_board[fr, fc] = EMPTY_SQUARE
        
        # Update visible boards
        if player == 1:
            self.visible_board_p1[tr, tc] = moving_piece
            self.visible_board_p1[fr, fc] = EMPTY_SQUARE
            self.visible_board_p2[tr, tc] = HIDDEN_PIECE if target_piece == EMPTY_SQUARE else self.visible_board_p2[tr, tc]
            self.visible_board_p2[fr, fc] = EMPTY_SQUARE
        else:
            self.visible_board_p2[tr, tc] = moving_piece
            self.visible_board_p2[fr, fc] = EMPTY_SQUARE
            self.visible_board_p1[tr, tc] = HIDDEN_PIECE if target_piece == EMPTY_SQUARE else self.visible_board_p1[tr, tc]
            self.visible_board_p1[fr, fc] = EMPTY_SQUARE
            
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