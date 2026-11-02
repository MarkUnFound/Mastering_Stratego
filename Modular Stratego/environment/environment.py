import torch
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from piece import PieceType
from battle import BattleResolver
from dqn_visualizer import DQNMoveVisualizer
from game_state import GameState
from training_config import REWARD_SCALE, DEFAULT_MAX_TURNS, NO_BATTLE_DRAW_PLIES

_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_IMMOBILE = (PieceType.FLAG.value, PieceType.BOMB.value)
_SCOUT = PieceType.SCOUT.value


def legal_moves(flat_board: List[int], player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Legal moves for `player` on a row-major list of 100 board values (the actual board)."""
    moves = []
    for i, v in enumerate(flat_board):
        rank = v * player
        if not 1 <= rank <= 12 or rank in _IMMOBILE:
            continue  # empty, lake, enemy, Flag or Bomb
        r, c = divmod(i, BOARD_SIZE)
        slides = rank == _SCOUT
        for dr, dc in _DIRECTIONS:
            tr, tc = r + dr, c + dc
            while 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
                target = flat_board[tr * BOARD_SIZE + tc]
                if target == LAKE_SQUARE or target * player > 0:
                    break  # lake or own piece
                moves.append(((r, c), (tr, tc)))
                if target != EMPTY_SQUARE or not slides:
                    break  # attack ends the move; non-Scouts move one square
                tr += dr
                tc += dc
    return moves


class StrategoEnvironment:
    def __init__(self, device, record_game=False, episode_num=None, full_observability=False, enable_anti_stall=False, max_turns=None,
                 no_battle_draw_plies=None):
        # Use configured default if not specified
        if max_turns is None:
            max_turns = DEFAULT_MAX_TURNS
        if no_battle_draw_plies is None:
            no_battle_draw_plies = NO_BATTLE_DRAW_PLIES
        self.device = device
        self.record_game = record_game
        self.episode_num = episode_num
        self.full_observability = full_observability  # For Phase 1 curriculum (cheating mode)
        self.enable_anti_stall = enable_anti_stall  # Only enable for agent-vs-agent games
        self.max_turns = max_turns  # Curriculum-based turn limit
        self.no_battle_draw_plies = no_battle_draw_plies  # Draw after this many plies without a battle (0 = off)
        self.plies_since_battle = 0
        self._moves_cache_key = None
        self._moves_cache = []
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.win_type = None  # Track win type: 'flag_capture', 'no_moves', 'timeout'
        self.revealed_pieces_p2 = {}
        
        self._flag_positions = {1: None, -1: None}
        self._cached_piece_counts = {1: 40, -1: 40}
        self._previous_piece_value = {1: 0, -1: 0}
        self._previous_move_count = {1: 0, -1: 0}
            
        self.board = Board(device)
        self.battle_resolver = BattleResolver()
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        self.dqn_visualizer = DQNMoveVisualizer()
        self.reset()

    def reset(self, p1_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None,
              p2_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None) -> GameState:

        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.win_type = None  # Reset win type tracking
        self.turn_count = 0
        self.plies_since_battle = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
        # Clear DQN visualizer history to prevent memory leak
        # (Each move stores cloned board tensors)
        self.dqn_visualizer.clear_history()
        
        self._flag_positions = {1: None, -1: None}
        self._cached_piece_counts = {1: 40, -1: 40}
        self._previous_piece_value = {1: 0, -1: 0}
        self._previous_move_count = {1: 0, -1: 0}
            
        # Track piece losses for exchange penalty mechanism
        self.piece_losses = {1: [], -1: []}
        self.exchanges = {1: [], -1: []}
        
        # Setup pieces in starting positions
        if p1_placement is None:
            p1_pieces = self._generate_pieces()
            p1_positions = self._get_p1_positions()
            p1_placement = [(piece, pos) for piece, pos in zip(p1_pieces, p1_positions)]
            
        if p2_placement is None:
            p2_pieces = self._generate_pieces()
            p2_positions = self._get_p2_positions()
            p2_placement = [(piece, pos) for piece, pos in zip(p2_pieces, p2_positions)]
            
        # Ensure we have exactly 40 pieces and 40 positions for each player
        assert len(p1_placement) == 40, f"Player 1 should have 40 pieces, got {len(p1_placement)}"
        assert len(p2_placement) == 40, f"Player 2 should have 40 pieces, got {len(p2_placement)}"
        
        # Verify no pieces in rows 4-5 (lake rows)
        for piece, (r, c) in p1_placement:
            assert r not in [4, 5], f"Player 1 piece in lake row: ({r}, {c})"
        for piece, (r, c) in p2_placement:
            assert r not in [4, 5], f"Player 2 piece in lake row: ({r}, {c})"

        self.place_pieces(1, p1_placement)
        self.place_pieces(-1, p2_placement)
        
        return self._get_game_state()

    def place_pieces(self, player_id: int, placement: List[Tuple[PieceType, Tuple[int, int]]]):
        """Place pieces on the board for a specific player."""
        for piece_type, (r, c) in placement:
            # Determine value based on player
            value = piece_type.value
            if player_id == -1:
                value = -value
            
            self.board.place_piece(r, c, value)
            
            # Track flag position
            if piece_type == PieceType.FLAG:
                self._flag_positions[player_id] = (r, c)

    def _generate_pieces(self) -> List[PieceType]:
        """Generate a list of pieces for one player. Must be exactly 40 pieces."""
        pieces = [PieceType.FLAG, PieceType.SPY] + [PieceType.BOMB]*6 + [PieceType.MARSHAL] + \
                 [PieceType.GENERAL] + [PieceType.COLONEL]*2 + [PieceType.MAJOR]*3 + \
                 [PieceType.CAPTAIN]*4 + [PieceType.LIEUTENANT]*4 + [PieceType.SERGEANT]*4 + \
                 [PieceType.MINER]*5 + [PieceType.SCOUT]*8
        # Ensure exactly 40 pieces
        if len(pieces) != 40:
            # Adjust scouts to make exactly 40
            current_count = len(pieces)
            if current_count < 40:
                pieces.extend([PieceType.SCOUT] * (40 - current_count))
            elif current_count > 40:
                # Remove extra scouts if any
                while len(pieces) > 40 and PieceType.SCOUT in pieces:
                    pieces.remove(PieceType.SCOUT)
        random.shuffle(pieces)
        assert len(pieces) == 40, f"Expected 40 pieces, got {len(pieces)}"
        return pieces

    def get_all_pieces(self) -> List[PieceType]:
        """Public method to get a full set of pieces (for setup agents)."""
        return self._generate_pieces()

    def get_valid_placement_positions(self, player_id: int) -> List[Tuple[int, int]]:
        """Get valid placement positions for a player."""
        positions = []
        if player_id == 1:
            # Player 1: Rows 6-9
            for r in range(6, 10):
                for c in range(BOARD_SIZE):
                    positions.append((r, c))
        else:
            # Player 2: Rows 0-3
            for r in range(4):
                for c in range(BOARD_SIZE):
                    positions.append((r, c))
        return positions

    def _get_p1_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 1. Must be exactly 40 positions in rows 6-9."""
        positions = self.get_valid_placement_positions(1)
        random.shuffle(positions)
        return positions

    def _get_p2_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 2. Must be exactly 40 positions in rows 0-3."""
        positions = self.get_valid_placement_positions(-1)
        random.shuffle(positions)
        return positions

    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all valid moves for the current player (Scouts slide any distance).
        Cached on the board contents and side to move, so repeated calls within a ply are free
        and a board edited from outside the environment is never served stale moves.
        """
        flat = self.board.actual_board.flatten().tolist()
        key = (self.current_player, tuple(flat))
        if key != self._moves_cache_key:
            self._moves_cache_key = key
            self._moves_cache = legal_moves(flat, self.current_player)
        return list(self._moves_cache)

    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute a move and return the new state.
        Assumes action is already validated by get_valid_moves() - no additional checks needed.
        """
        # Track revealed pieces for PBS training
        revealed_in_step = []

        if self.game_over:
            return self._get_game_state(), 0.0, True, {"winner": self.winner, "win_type": self.win_type}

        # Check for max turns (draw) - uses curriculum-based max_turns
        if self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = 0
            self.win_type = 'timeout'
            return self._get_game_state(), -1.0, True, {"winner": 0, "win_type": "timeout", "revealed_in_step": [], "game_phase": "end", "turn_count": self.turn_count}
            
        if action is None:
            # No valid moves possible - DRAW (both agents lose)
            # This prevents learning to win by starving opponent of moves
            self.game_over = True
            self.winner = 0  # Draw instead of opponent win
            self.win_type = 'timeout'
            return self._get_game_state(), -1.0, True, {"winner": 0, "win_type": "timeout", "revealed_in_step": [], "game_phase": "end", "turn_count": self.turn_count}

        (r_from, c_from), (r_to, c_to) = action
        
        # Get pieces involved in the move
        actual_board = self.board.actual_board
        moving_piece_value = actual_board[r_from, c_from].item()
        target_piece_value = actual_board[r_to, c_to].item()
        
        # ============= NORMALIZED REWARD STRUCTURE =============
        # All rewards normalized to approximately [-1, +1] range
        # Win/Loss handled separately in train_dqn.py for cleaner gradients
        
        reward = 0.0
        
        # Determine game phase
        game_phase = "early" if self.turn_count < 50 else ("mid" if self.turn_count < 200 else "end")
        
        # Handle battle or simple move
        if target_piece_value != EMPTY_SQUARE and target_piece_value != LAKE_SQUARE:
            # Battle occurs
            self.plies_since_battle = 0
            attacker_type = PieceType(abs(moving_piece_value))
            defender_type = PieceType(abs(target_piece_value))
            
            # Reveal pieces
            self.board.reveal_pieces((r_from, c_from), (r_to, c_to))
            self.revealed_pieces_p1[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p2[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p1[(r_to, c_to)] = abs(target_piece_value)
            self.revealed_pieces_p2[(r_to, c_to)] = abs(target_piece_value)

            revealed_in_step.append(((r_from, c_from), attacker_type))
            revealed_in_step.append(((r_to, c_to), defender_type))
            
            attacker_player = 1 if moving_piece_value > 0 else 2 if moving_piece_value < 0 else 0
            defender_player = 1 if target_piece_value > 0 else 2 if target_piece_value < 0 else 0
            result = self.battle_resolver.resolve_battle(attacker_type, defender_type, attacker_player, defender_player)
            
            attacker_rank = abs(moving_piece_value)
            defender_rank = abs(target_piece_value)
            
            if result == 1:  # Attacker wins
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                
                if hasattr(self, '_cached_piece_counts'):
                    self._cached_piece_counts[-self.current_player] = max(0, self._cached_piece_counts[-self.current_player] - 1)
                
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(self.current_player)
                
                # Flag capture = WIN
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    self.win_type = 'flag_capture'  # Track flag capture win
                    if hasattr(self, '_flag_positions'):
                        self._flag_positions[-self.current_player] = None
                        
            elif result == -1:  # Defender wins (attacker loses piece)
                actual_board[r_from, c_from] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                
                if hasattr(self, '_cached_piece_counts'):
                    self._cached_piece_counts[self.current_player] = max(0, self._cached_piece_counts[self.current_player] - 1)
                
                self._pending_loss = (self.current_player, attacker_rank, False)
                
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(-self.current_player)
                
            else:  # Both pieces die (mutual destruction)
                actual_board[r_from, c_from] = EMPTY_SQUARE
                actual_board[r_to, c_to] = EMPTY_SQUARE
                for p_board in [self.board.visible_board_p1, self.board.visible_board_p2]:
                    p_board[r_from, c_from] = EMPTY_SQUARE
                    p_board[r_to, c_to] = EMPTY_SQUARE
                
                if hasattr(self, '_cached_piece_counts'):
                    self._cached_piece_counts[self.current_player] = max(0, self._cached_piece_counts[self.current_player] - 1)
                    self._cached_piece_counts[-self.current_player] = max(0, self._cached_piece_counts[-self.current_player] - 1)
                
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(self.current_player)
                self._pending_exchanges.append(-self.current_player)
                
                if not hasattr(self, '_pending_losses'):
                    self._pending_losses = []
                self._pending_losses.append((self.current_player, attacker_rank, True))
                self._pending_losses.append((-self.current_player, defender_rank, True))
                
        else:
            # Simple move (no battle)
            self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
            self.plies_since_battle += 1

        self.turn_count += 1
        self.move_history.append(action)
        
        # Stalemate Prevention tracking (Mobility)
        avail_moves = len(self.get_valid_moves())
        
        # Process pending losses/exchanges
        if hasattr(self, '_pending_loss'):
            p, v, e = self._pending_loss
            self.piece_losses[p].append((self.turn_count, v, e))
            delattr(self, '_pending_loss')
            
        if hasattr(self, '_pending_losses'):
            for p, v, e in self._pending_losses:
                self.piece_losses[p].append((self.turn_count, v, e))
            delattr(self, '_pending_losses')
            
        if hasattr(self, '_pending_exchanges'):
            for p in self._pending_exchanges:
                self.exchanges[p].append(self.turn_count)
            delattr(self, '_pending_exchanges')
            
        # Switch player
        self.current_player *= -1
        
        # Record move
        self.dqn_visualizer.record_move(action, self._get_game_state(), self.current_player * -1)
        
        # Check for game end conditions
        self._check_game_end()
        
        info = {
            "winner": self.winner, 
            "win_type": self.win_type,
            "revealed_in_step": revealed_in_step, 
            "game_phase": game_phase, 
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "plies_since_battle": self.plies_since_battle,
            "num_valid_moves": avail_moves
        }
        
        return self._get_game_state(), 0.0, self.game_over, info

    def _check_game_end(self):
        # Checks for game-ending conditions.
        
        # If game is already over (e.g. from flag capture in step()), don't overwrite it
        if self.game_over:
            return

        # Flag capture (flag positions are tracked in place_pieces/step)
        if self._flag_positions[1] is None:
            self.game_over = True
            self.winner = -1
            self.win_type = 'flag_capture'
            return
        if self._flag_positions[-1] is None:
            self.game_over = True
            self.winner = 1
            self.win_type = 'flag_capture'
            return
        
        # Check if current player has any valid moves
        if not self.get_valid_moves():
            self.game_over = True
            # Player who cannot move loses - opponent wins
            # This follows standard Stratego rules and rewards material dominance
            self.winner = -self.current_player
            self.win_type = 'no_moves'  # Track immobilization win
            return

        # No-battle draw: too many plies without an attack
        if self.no_battle_draw_plies and self.plies_since_battle >= self.no_battle_draw_plies:
            self.game_over = True
            self.winner = 0
            self.win_type = 'no_battle_draw'
            return
        
        # Early draw detection - ONLY if anti_stall is enabled (agent-vs-agent)
        # Disabled for random opponents since they don't cause intentional stalls
        if self.enable_anti_stall and self.turn_count > 100:
            # Check for repetitive positions (aggressive check)
            if self._is_position_repetitive():
                self.game_over = True
                self.winner = 0
                self.win_type = 'stall_repetition'
                return
        
        # Stalemate check - ONLY if anti_stall is enabled
        if self.enable_anti_stall and self.turn_count > 200:
            if self._is_stalemate():
                self.game_over = True
                self.winner = 0
                self.win_type = 'stall_few_pieces'
                return
        
        # Hard limit - uses curriculum-based max_turns
        if self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = 0
            self.win_type = 'timeout'
            
    def _is_position_repetitive(self):
        # Check if the same position has occurred multiple times (aggressive detection)
        if len(self.move_history) < 6:
            return False
        # Check last 6 moves for back-and-forth patterns (deadlock)
        recent_moves = self.move_history[-6:]
        reversal_count = 0
        for i in range(1, len(recent_moves)):
            current_move = recent_moves[i]
            previous_move = recent_moves[i-1]
            if (current_move[0] == previous_move[1] and 
                current_move[1] == previous_move[0]):
                reversal_count += 1
        return reversal_count >= 2  # 2 reversals in 6 moves = deadlock

    def _is_stalemate(self):
        # Check if very few pieces are moving
        if len(self.move_history) < 50:
            return False
        # Check if moves are only involving a few pieces
        recent_moves = self.move_history[-50:]
        moved_pieces = set()
        for move in recent_moves:
            moved_pieces.add(move[0])  # Add starting positions
        # If only 2-3 pieces are doing all the moving, likely stalemate
        return len(moved_pieces) <= 3
            
    def _get_game_state(self) -> GameState:
        # Observability Logic
        # 1. Full Observability (True or 1.0) -> Show actual board
        # 2. No Observability (False or 0.0) -> Show visible board
        # 3. Mixed Observability (float 0.0 < x < 1.0) -> Show visible + x% of hidden pieces
        
        observability = self.full_observability
        if observability is True:
            observability = 1.0
        elif observability is False:
            observability = 0.0
            
        if observability >= 1.0:
            board = self.board.actual_board.clone()
        elif observability <= 0.0:
            board = self.board.get_visible_board(self.current_player).clone()
        else:
            # Mixed Mode: Start with visible board
            board = self.board.get_visible_board(self.current_player).clone() # Clone to avoid modifying cache
            actual = self.board.actual_board
            
            # Identify hidden pieces (where board is 'unknown' but actual has a piece)
            # Unknown is usually 0 (empty) or LAKE in visible board, but pieces are hidden.
            # Actually, hidden pieces appear as 0 or simply don't appear?
            # get_visible_board returns the board from perspective. Hidden pieces are 0 (empty) unless revealed.
            # But empty squares are also 0.
            # We want to reveal pieces that are NOT 0 in actual, but ARE 0 (or unknown) in visible.
            
            # Values in actual: +/- Rank (or Lake)
            # Values in visible: +/- Rank (if known), or Lake, or 0 (if empty OR hidden)
            
            # Find pieces that are hidden
            # (Actual != 0) AND (Actual != Lake) AND (Visible == 0)
            hidden_mask = (actual != 0) & (actual != LAKE_SQUARE) & (board == 0)
            
            # Get indices of hidden pieces
            hidden_indices = torch.nonzero(hidden_mask)
            num_hidden = len(hidden_indices)
            
            if num_hidden > 0:
                # Randomly select subset to reveal based on observability rate
                num_to_reveal = int(num_hidden * observability)
                if num_to_reveal > 0:
                    # Shuffle indices to pick random ones
                    perm = torch.randperm(num_hidden, device=self.device)
                    indices_to_reveal = hidden_indices[perm[:num_to_reveal]]
                    
                    # Reveal them in the observation (copy from actual)
                    r_rev, c_rev = indices_to_reveal[:, 0], indices_to_reveal[:, 1]
                    board[r_rev, c_rev] = actual[r_rev, c_rev]

        game_state = GameState(
            board=board,
            current_player=self.current_player,
            turn_count=self.turn_count,
            plies_since_battle=self.plies_since_battle,
            game_over=self.game_over,
            winner=self.winner,
            move_history=self.move_history.copy() if len(self.move_history) < 1000 else self.move_history[-100:],  # Only copy recent moves if history is long
            uncertainty_mask=torch.zeros(BOARD_SIZE, BOARD_SIZE, device=self.device),  # Simplified, use device
            revealed_pieces_p1=self.revealed_pieces_p1.copy() if len(self.revealed_pieces_p1) < 100 else dict(list(self.revealed_pieces_p1.items())[-50:]),  # Only copy recent if large
            revealed_pieces_p2=self.revealed_pieces_p2.copy() if len(self.revealed_pieces_p2) < 100 else dict(list(self.revealed_pieces_p2.items())[-50:])  # Only copy recent if large
        )
        # Add actual board for visualization purposes (needed for PBS visualization)
        # clone() is necessary here to avoid reference issues when board changes
        game_state.actual_board = self.board.actual_board.clone()
        return game_state
    
    def get_state(self) -> GameState:
        """Public alias for _get_game_state() for dashboard/GUI compatibility."""
        return self._get_game_state()

    def set_full_observability(self, enabled: object):
        """
        Set full observability mode.
        Args:
            enabled: bool or float.
                     True/1.0 = Full Observability
                     False/0.0 = Partial Observability
                     0.0 < x < 1.0 = Mixed Observability (reveal x% of hidden pieces)
        """
        self.full_observability = enabled
