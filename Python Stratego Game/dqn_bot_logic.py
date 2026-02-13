"""
DQN Bot Logic - Adapter for Rainbow DQN Agent in GUI

Bridges the GUI's stratego.Board (Piece objects) to the trained
RainbowDQN model from the Modular Stratego codebase.

Translation layers:
  - GUI Piece.rank -> Modular PieceType integer encoding
  - GUI owner (1,2) -> Modular player sign (+,-)
  - GUI Board.grid -> 10x10 integer tensor
  - Modular action index -> GUI (src, dst) tuple
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Add Modular Stratego to sys.path so we can import network definitions
# ---------------------------------------------------------------------------
MODULAR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Modular Stratego')
)
if MODULAR_DIR not in sys.path:
    sys.path.insert(0, MODULAR_DIR)

from networks import RainbowDQN
from history_aggregator import HistoryAggregator
from board import LAKE_SQUARE

# ---------------------------------------------------------------------------
# Rank mapping:  GUI rank  ->  Modular board value (absolute)
#
#  GUI                   Modular
#  -1  Flag              1  (PieceType.FLAG)
#   0  Bomb             12  (PieceType.BOMB)
#   1  Spy               2  (PieceType.SPY)
#   2  Scout             3  (PieceType.SCOUT)
#   3  Miner             4  (PieceType.MINER)
#   4  Sergeant          5
#   5  Lieutenant        6
#   6  Captain           7
#   7  Major             8
#   8  Colonel           9
#   9  General          10
#  10  Marshal          11
# ---------------------------------------------------------------------------
GUI_RANK_TO_MODULAR = {
    -1: 1,   # Flag
     0: 12,  # Bomb
     1: 2,   # Spy
     2: 3,   # Scout
     3: 4,   # Miner
     4: 5,   # Sergeant
     5: 6,   # Lieutenant
     6: 7,   # Captain
     7: 8,   # Major
     8: 9,   # Colonel
     9: 10,  # General
    10: 11,  # Marshal
}

# C51 distributional RL constants (must match training)
V_MIN = -30.0
V_MAX = 30.0
NUM_ATOMS = 51

# Lake positions on the 10×10 board (same in both codebases)
LAKE_POSITIONS = {(4, 2), (4, 3), (5, 2), (5, 3),
                  (4, 6), (4, 7), (5, 6), (5, 7)}


class DQNBotLogic:
    """
    Drop-in replacement for EnhancedBotLogic that uses the trained
    Rainbow DQN + AAREN model for move selection.

    Interface matches EnhancedBotLogic:
      - __init__(model_path, player_id)
      - choose_move(board, owner) -> (src, dst) | None
      - reset()

    Additional hooks for AAREN tracking:
      - notify_battle_result(pos, revealed_rank)
    """

    def __init__(self, model_path: str, player_id: int = 2):
        """
        Args:
            model_path: Path to .pth checkpoint (from Modular Stratego training)
            player_id: Bot's owner in the GUI (1 or 2)
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.move_count = 0

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # C51 support vector
        self.support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS, device=self.device)

        # AAREN settings
        self.history_embedding_size = 64
        self.input_channels = 15 + self.history_embedding_size  # 79

        # Create network with same architecture used during training
        self.q_network = RainbowDQN(
            input_shape=(self.input_channels, 10, 10),
            output_size=400,
            num_atoms=NUM_ATOMS,
        ).to(self.device)

        # Create AAREN history aggregator
        # Modular Stratego uses player_id 1 or -1 internally,
        # but HistoryAggregator just needs an integer id
        self.history = HistoryAggregator(player_id, self.device, hidden_size=64)

        # Provide legacy .pbs attribute so GUI code that touches bot_logic.pbs
        # won't crash (SimplifiedPBS compatibility shim)
        self.pbs = _PBSShim(self)

        # Load checkpoint
        self._load_checkpoint(model_path)

        # Put network in eval mode and disable noise for deterministic play
        self.q_network.eval()
        self._disable_noise()

        print(f"[DQN Bot] Loaded on {self.device} | "
              f"player_id={player_id} | channels={self.input_channels}")

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str):
        """Load weights from a Modular Stratego .pth checkpoint."""
        if not os.path.exists(path):
            print(f"[DQN Bot] WARNING: checkpoint not found at {path}")
            print("[DQN Bot] Bot will play with random weights (untrained).")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print(f"[DQN Bot] Q-network loaded from {path}")

            if 'history_state_dict' in checkpoint:
                self.history.load_state_dict(checkpoint['history_state_dict'])
                print(f"[DQN Bot] AAREN history state loaded")
        except Exception as e:
            print(f"[DQN Bot] WARNING: failed to load checkpoint: {e}")
            print("[DQN Bot] Bot will play with random weights (untrained).")

    def _disable_noise(self):
        """Set NoisyLinear layers to deterministic (sigma=0)."""
        for module in self.q_network.modules():
            if hasattr(module, 'weight_sigma') and hasattr(module, 'bias_sigma'):
                module.weight_sigma.data.zero_()
                module.bias_sigma.data.zero_()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def reset(self):
        """Reset state for a new game."""
        self.move_count = 0
        if self.history:
            self.history.reset()

    def choose_move(self, board, owner: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Choose the best move for the given board state.

        Args:
            board: GUI stratego.Board instance
            owner: Current player (should == self.player_id)

        Returns:
            ((r_from, c_from), (r_to, c_to))  or  None
        """
        # 1) Enumerate legal moves from the GUI board
        legal_moves = self._get_all_legal_moves(board, owner)
        if not legal_moves:
            return None

        # 2) Convert GUI board -> integer tensor
        board_tensor = self._board_to_tensor(board)

        # 3) Build state representation (15 board + 64 AAREN channels)
        state_tensor = self._get_state_representation(board_tensor)
        state_tensor = state_tensor.unsqueeze(0)  # add batch dim

        # 4) Forward pass -> Q-values
        with torch.no_grad():
            log_probs = self.q_network(state_tensor)
            probs = log_probs.exp()
            q_values = (probs * self.support).sum(dim=2).squeeze(0)  # (400,)

        # 5) Mask with legal moves and pick argmax
        best_move = None
        best_q = -float('inf')

        for src, dst in legal_moves:
            action_idx = self._move_to_action_index(src, dst)
            if action_idx is None:
                continue
            q = q_values[action_idx].item()
            if q > best_q:
                best_q = q
                best_move = (src, dst)

        self.move_count += 1
        return best_move if best_move else (legal_moves[0] if legal_moves else None)

    def notify_battle_result(self, pos: Tuple[int, int], revealed_rank: int):
        """
        Feed a reveal event to AAREN after a battle.

        Args:
            pos: Board position of the revealed piece
            revealed_rank: GUI rank of the revealed piece
        """
        if self.history is None:
            return
        modular_type_idx = GUI_RANK_TO_MODULAR.get(revealed_rank, None)
        if modular_type_idx is None:
            return
        # Determine game phase from move count
        if self.move_count < 50:
            game_phase = "early"
        elif self.move_count < 200:
            game_phase = "mid"
        else:
            game_phase = "end"
        try:
            from piece import PieceType
            piece_type = PieceType(modular_type_idx)
            self.history.update_from_reveal(pos, piece_type,
                                            game_phase=game_phase,
                                            turn_count=self.move_count)
        except Exception:
            pass  # Silently skip if AAREN update fails

    # ------------------------------------------------------------------
    # Board translation
    # ------------------------------------------------------------------
    def _board_to_tensor(self, board) -> torch.Tensor:
        """
        Convert a GUI stratego.Board to a 10×10 integer tensor
        matching the Modular Stratego encoding:
          - Player 1 pieces: positive values (1..12)
          - Player 2 pieces: negative values (-1..-12)
          - Lakes: LAKE_SQUARE constant
          - Empty: 0
        """
        tensor = torch.zeros(10, 10, device=self.device)

        for r in range(10):
            for c in range(10):
                if (r, c) in LAKE_POSITIONS:
                    tensor[r, c] = LAKE_SQUARE
                    continue

                piece = board.grid[r][c]
                if piece is None:
                    continue

                modular_val = GUI_RANK_TO_MODULAR.get(piece.rank, 1)
                # Sign convention: owner 1 -> positive, owner 2 -> negative
                if piece.owner == 1:
                    tensor[r, c] = modular_val
                else:
                    tensor[r, c] = -modular_val

        return tensor

    def _get_state_representation(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Build the 79-channel (15 board + 64 AAREN) state tensor,
        matching RainbowAgent.get_state_representation().
        """
        features = torch.zeros(15, 10, 10, device=self.device)

        if self.player_id == 1:
            # Own pieces (positive 1..12)
            for i in range(1, 13):
                features[i - 1] = (board_tensor == i).float()
            # Enemy pieces: negative, but not lake
            features[12] = ((board_tensor < 0) & (board_tensor > LAKE_SQUARE)).float()
        else:
            # Own pieces (negative -1..-12)
            for i in range(1, 13):
                features[i - 1] = (board_tensor == -i).float()
            # Enemy pieces: positive
            features[12] = (board_tensor > 0).float()

        # Obstacles (lakes)
        features[13] = (board_tensor == LAKE_SQUARE).float()
        # Empty squares
        features[14] = (board_tensor == 0).float()

        # AAREN embeddings (64 channels)
        if self.history is not None:
            embedding = self.history.get_embedding_tensor()
            if embedding.device != self.device:
                embedding = embedding.to(self.device)
        else:
            embedding = torch.zeros(self.history_embedding_size, 10, 10,
                                    device=self.device)

        return torch.cat([features, embedding], dim=0)  # (79, 10, 10)

    # ------------------------------------------------------------------
    # Move helpers
    # ------------------------------------------------------------------
    def _get_all_legal_moves(self, board, owner: int
                             ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Enumerate all legal moves for `owner` using the GUI board."""
        moves = []
        for r in range(10):
            for c in range(10):
                piece = board.grid[r][c]
                if piece and piece.owner == owner and piece.is_movable():
                    for dest in board.legal_moves_from((r, c)):
                        moves.append(((r, c), dest))
        return moves

    def _move_to_action_index(self, src: Tuple[int, int],
                              dst: Tuple[int, int]) -> Optional[int]:
        """
        Encode (src, dst) -> action index (0..399).
        Same encoding as RainbowAgent._move_to_action_index:
          index = (r1*10 + c1) * 4 + dir_idx
          Directions: 0=Right(0,+1)  1=Left(0,-1)  2=Down(+1,0)  3=Up(-1,0)

        For Scout moves (distance > 1) we map to the 1-step direction.
        """
        r1, c1 = src
        r2, c2 = dst
        dr = r2 - r1
        dc = c2 - c1
        dist = abs(dr) + abs(dc)

        # Scout multi-step: reduce to unit direction
        if dist > 1:
            if dr != 0:
                dr = 1 if dr > 0 else -1
                dc = 0
            else:
                dc = 1 if dc > 0 else -1
                dr = 0

        if dr == 0 and dc > 0:
            dir_idx = 0  # Right
        elif dr == 0 and dc < 0:
            dir_idx = 1  # Left
        elif dr > 0 and dc == 0:
            dir_idx = 2  # Down
        elif dr < 0 and dc == 0:
            dir_idx = 3  # Up
        else:
            return None

        return (r1 * 10 + c1) * 4 + dir_idx


# ---------------------------------------------------------------------------
# PBS compatibility shim
# ---------------------------------------------------------------------------
class _PBSShim:
    """
    Thin shim so that existing GUI code referencing bot_logic.pbs.update_from_reveal()
    does not crash.  Delegates to AAREN via DQNBotLogic.notify_battle_result().
    """
    def __init__(self, bot: DQNBotLogic):
        self._bot = bot

    def update_from_reveal(self, pos, rank):
        self._bot.notify_battle_result(pos, rank)

    def update_from_move(self, *args, **kwargs):
        pass  # Not needed for AAREN

    def reset(self):
        if self._bot.history:
            self._bot.history.reset()
