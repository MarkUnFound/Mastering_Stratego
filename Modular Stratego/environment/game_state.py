# stratego_modular/game_state.py

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import torch

@dataclass
class GameState:
    """Lightweight game state for GPU processing."""
    board: torch.Tensor
    current_player: int
    turn_count: int
    game_over: bool
    winner: Optional[int]
    move_history: List[Tuple]
    uncertainty_mask: torch.Tensor
    # Track revealed pieces for each player
    revealed_pieces_p1: Dict[Tuple[int, int], int]
    revealed_pieces_p2: Dict[Tuple[int, int], int]

    def get_player_view(self, player: int) -> torch.Tensor:
        """Returns the board as seen by the specified player."""
        if player == 1:
            return self.board
        else:
            # For player 2, flip the board perspective
            return -self.board

    def get_revealed_pieces(self, player: int) -> Dict[Tuple[int, int], int]:
        """Returns the pieces revealed to the specified player."""
        if player == 1:
            return self.revealed_pieces_p1
        else:
            return self.revealed_pieces_p2

    def clone(self):
        """
        Create a lightweight copy of the game state.
        Much faster than copy.deepcopy() for tensors and simple types.
        """
        # Clone tensor (fast)
        new_board = self.board.clone()
        
        # Shallow copy lists/dicts where possible, or create new ones
        # We need new containers but can share immutable elements
        new_move_history = list(self.move_history)
        new_revealed_p1 = self.revealed_pieces_p1.copy()
        new_revealed_p2 = self.revealed_pieces_p2.copy()
        
        # Uncertainty mask is a tensor
        new_uncertainty = self.uncertainty_mask.clone() if self.uncertainty_mask is not None else None
        
        return GameState(
            board=new_board,
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            winner=self.winner,
            move_history=new_move_history,
            uncertainty_mask=new_uncertainty,
            revealed_pieces_p1=new_revealed_p1,
            revealed_pieces_p2=new_revealed_p2
        )