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