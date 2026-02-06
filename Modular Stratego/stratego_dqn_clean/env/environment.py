"""
Stratego Environment for Double DQN + AAREN

Clean implementation without legacy debug/reward code.
Rewards are handled externally in distributional_reward.py.
"""

import torch
import random
import numpy as np
from typing import List, Tuple, Optional, Dict

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from piece import PieceType
from battle import BattleResolver
from game_state import GameState


class StrategoEnvironment:
    """
    Clean Stratego environment for Double DQN training.
    
    Key design decisions:
    - Rewards are NOT calculated here (use external reward shaper)
    - Returns raw game info in step() for external reward calculation
    - No debug visualizers (add externally if needed)
    - Minimal state tracking
    """
    
    def __init__(
        self,
        device,
        full_observability: bool = False,
        max_turns: int = 1000,
        strict_validation: bool = False,
        safe_guards: bool = True
    ):
        """
        Args:
            device: PyTorch device (cuda/cpu)
            full_observability: If True, reveals all pieces (curriculum mode)
            max_turns: Maximum turns before draw
            strict_validation: If True, checks move against full valid move list (SLOW)
            safe_guards: If True, checks basic rules like lakes/friendly fire (FAST)
        """
        self.device = device
        self.full_observability = full_observability
        self.max_turns = max_turns
        self.strict_validation = strict_validation
        self.safe_guards = safe_guards
        
        # Game state
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.win_type = None
        self.turn_count = 0
        self.move_history = []
        
        # Visibility tracking
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
        # Flag position cache
        self._flag_positions = {1: None, -1: None}
        
        # Core components
        self.board = Board(device)
        self.battle_resolver = BattleResolver()
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        
        self.reset()

    def reset(
        self,
        p1_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None,
        p2_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None
    ) -> GameState:
        """Reset environment to initial state."""
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.win_type = None
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        self._flag_positions = {1: None, -1: None}
        
        # Setup pieces
        if p1_placement is None:
            p1_pieces = self._generate_pieces()
            p1_positions = self._get_p1_positions()
            p1_placement = list(zip(p1_pieces, p1_positions))
            
        if p2_placement is None:
            p2_pieces = self._generate_pieces()
            p2_positions = self._get_p2_positions()
            p2_placement = list(zip(p2_pieces, p2_positions))
        
        assert len(p1_placement) == 40, f"P1 should have 40 pieces, got {len(p1_placement)}"
        assert len(p2_placement) == 40, f"P2 should have 40 pieces, got {len(p2_placement)}"
        
        self.place_pieces(1, p1_placement)
        self.place_pieces(-1, p2_placement)
        
        return self._get_game_state()

    def place_pieces(self, player_id: int, placement: List[Tuple[PieceType, Tuple[int, int]]]):
        """Place pieces on the board for a specific player."""
        for piece_type, (r, c) in placement:
            value = piece_type.value if player_id == 1 else -piece_type.value
            self.board.place_piece(r, c, value)
            
            if piece_type == PieceType.FLAG:
                self._flag_positions[player_id] = (r, c)

    def _generate_pieces(self) -> List[PieceType]:
        """Generate standard 40-piece Stratego army."""
        pieces = (
            [PieceType.FLAG] +
            [PieceType.SPY] +
            [PieceType.BOMB] * 6 +
            [PieceType.MARSHAL] +
            [PieceType.GENERAL] +
            [PieceType.COLONEL] * 2 +
            [PieceType.MAJOR] * 3 +
            [PieceType.CAPTAIN] * 4 +
            [PieceType.LIEUTENANT] * 4 +
            [PieceType.SERGEANT] * 4 +
            [PieceType.MINER] * 5 +
            [PieceType.SCOUT] * 8
        )
        random.shuffle(pieces)
        return pieces

    def get_all_pieces(self) -> List[PieceType]:
        """Public method to get a full set of pieces (for setup agents)."""
        return self._generate_pieces()

    def get_valid_placement_positions(self, player_id: int) -> List[Tuple[int, int]]:
        """Get valid placement positions for a player."""
        if player_id == 1:
            return [(r, c) for r in range(6, 10) for c in range(BOARD_SIZE)]
        else:
            return [(r, c) for r in range(4) for c in range(BOARD_SIZE)]

    def _get_p1_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 1 (rows 6-9)."""
        positions = self.get_valid_placement_positions(1)
        random.shuffle(positions)
        return positions

    def _get_p2_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 2 (rows 0-3)."""
        positions = self.get_valid_placement_positions(-1)
        random.shuffle(positions)
        return positions

    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all valid moves for current player (vectorized)."""
        moves = []
        actual_board = self.board.actual_board
        
        # Find current player's pieces
        if self.current_player == 1:
            player_pieces_mask = (actual_board > 0) & (actual_board != LAKE_SQUARE)
        else:
            player_pieces_mask = (actual_board < 0) & (actual_board != LAKE_SQUARE)
        
        piece_indices = torch.nonzero(player_pieces_mask)
        if len(piece_indices) == 0:
            return []
        
        r_indices = piece_indices[:, 0]
        c_indices = piece_indices[:, 1]
        piece_values = actual_board[r_indices, c_indices]
        piece_types = torch.abs(piece_values)
        
        # Filter immobile pieces (Flag, Bomb)
        movable_mask = (piece_types != PieceType.FLAG.value) & (piece_types != PieceType.BOMB.value)
        r_movable = r_indices[movable_mask]
        c_movable = c_indices[movable_mask]
        types_movable = piece_types[movable_mask]
        
        if len(r_movable) == 0:
            return []
        
        # Non-Scout moves (1 step)
        non_scout_mask = types_movable != PieceType.SCOUT.value
        r_non_scout = r_movable[non_scout_mask]
        c_non_scout = c_movable[non_scout_mask]
        
        if len(r_non_scout) > 0:
            moves.extend(self._get_single_step_moves(r_non_scout, c_non_scout, actual_board))
        
        # Scout moves (multiple steps)
        scout_mask = types_movable == PieceType.SCOUT.value
        r_scout = r_movable[scout_mask]
        c_scout = c_movable[scout_mask]
        
        if len(r_scout) > 0:
            moves.extend(self._get_scout_moves(r_scout, c_scout, actual_board))
        
        return moves

    def _get_single_step_moves(self, r_pieces, c_pieces, board):
        """Get single-step moves for non-scout pieces."""
        moves = []
        dr = torch.tensor([0, 0, 1, -1], device=self.device)
        dc = torch.tensor([1, -1, 0, 0], device=self.device)
        
        # Broadcast: (N, 1) + (1, 4) -> (N, 4)
        r_targets = r_pieces.unsqueeze(1) + dr.unsqueeze(0)
        c_targets = c_pieces.unsqueeze(1) + dc.unsqueeze(0)
        
        # Flatten
        r_targets_flat = r_targets.view(-1)
        c_targets_flat = c_targets.view(-1)
        r_sources_flat = r_pieces.unsqueeze(1).repeat(1, 4).view(-1)
        c_sources_flat = c_pieces.unsqueeze(1).repeat(1, 4).view(-1)
        
        # Bounds check
        bounds_mask = (
            (r_targets_flat >= 0) & (r_targets_flat < BOARD_SIZE) &
            (c_targets_flat >= 0) & (c_targets_flat < BOARD_SIZE)
        )
        
        r_valid = r_targets_flat[bounds_mask]
        c_valid = c_targets_flat[bounds_mask]
        r_src_valid = r_sources_flat[bounds_mask]
        c_src_valid = c_sources_flat[bounds_mask]
        
        if len(r_valid) > 0:
            target_values = board[r_valid, c_valid]
            
            # Valid: not lake, not friendly
            not_lake = target_values != LAKE_SQUARE
            if self.current_player == 1:
                not_friendly = target_values <= 0
            else:
                not_friendly = target_values >= 0
            
            valid_mask = not_lake & not_friendly
            
            r_final_src = r_src_valid[valid_mask]
            c_final_src = c_src_valid[valid_mask]
            r_final_dst = r_valid[valid_mask]
            c_final_dst = c_valid[valid_mask]
            
            for i in range(len(r_final_src)):
                moves.append((
                    (int(r_final_src[i]), int(c_final_src[i])),
                    (int(r_final_dst[i]), int(c_final_dst[i]))
                ))
        
        return moves

    def _get_scout_moves(self, r_scout, c_scout, board):
        """Get sliding moves for scouts."""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            active_mask = torch.ones(len(r_scout), dtype=torch.bool, device=self.device)
            
            for dist in range(1, BOARD_SIZE):
                if not active_mask.any():
                    break
                
                r_target = r_scout + dr * dist
                c_target = c_scout + dc * dist
                
                # Bounds check
                bounds_mask = (
                    (r_target >= 0) & (r_target < BOARD_SIZE) &
                    (c_target >= 0) & (c_target < BOARD_SIZE)
                )
                active_mask = active_mask & bounds_mask
                
                if not active_mask.any():
                    break
                
                active_indices = torch.nonzero(active_mask).squeeze(1)
                r_active = r_target[active_indices]
                c_active = c_target[active_indices]
                target_values = board[r_active, c_active]
                
                # Check blocking
                is_lake = target_values == LAKE_SQUARE
                is_occupied = target_values != EMPTY_SQUARE
                if self.current_player == 1:
                    is_friendly = target_values > 0
                else:
                    is_friendly = target_values < 0
                
                # Valid: not lake, not friendly
                valid_step_mask = (~is_lake) & (~is_friendly)
                valid_indices = active_indices[valid_step_mask]
                
                if len(valid_indices) > 0:
                    r_src_valid = r_scout[valid_indices]
                    c_src_valid = c_scout[valid_indices]
                    r_dst_valid = r_target[valid_indices]
                    c_dst_valid = c_target[valid_indices]
                    
                    for i in range(len(valid_indices)):
                        moves.append((
                            (int(r_src_valid[i]), int(c_src_valid[i])),
                            (int(r_dst_valid[i]), int(c_dst_valid[i]))
                        ))
                
                # Update active mask (stop on lake or occupied)
                continue_mask = (~is_lake) & (~is_occupied)
                full_continue_mask = torch.zeros_like(active_mask)
                full_continue_mask[active_indices] = continue_mask
                active_mask = active_mask & full_continue_mask
        
        return moves

    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute a move.
        
        Returns:
            game_state: Current game state
            reward: Always 0.0 (use external reward shaper)
            done: Whether game is over
            info: Dict with battle info, winner, etc.
        """
        if self.game_over:
            return self._get_game_state(), 0.0, True, {
                "winner": self.winner,
                "win_type": self.win_type
            }
        
        # Timeout check
        if self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = 0
            self.win_type = 'timeout'
            return self._get_game_state(), 0.0, True, {
                "winner": 0,
                "win_type": "timeout",
                "game_phase": "end",
                "turn_count": self.turn_count
            }
        
        # No action = stalemate
        if action is None:
            self.game_over = True
            self.winner = 0
            self.win_type = 'timeout'
            return self._get_game_state(), 0.0, True, {
                "winner": 0,
                "win_type": "timeout"
            }
        
        (r_from, c_from), (r_to, c_to) = action
        actual_board = self.board.actual_board
        
        moving_piece_value = actual_board[r_from, c_from].item()
        target_piece_value = actual_board[r_to, c_to].item()
        
        # --- VALIDATION START ---
        if self.safe_guards:
            self._check_safe_guards(r_from, c_from, r_to, c_to, moving_piece_value, target_piece_value)
            
        if self.strict_validation:
            self._check_strict_validation(action)
        # --- VALIDATION END ---
        
        # Track reveals for info
        revealed_in_step = []
        battle_result = None
        
        # Determine game phase
        game_phase = "early" if self.turn_count < 50 else ("mid" if self.turn_count < 200 else "end")
        
        # Handle battle or simple move
        if target_piece_value != EMPTY_SQUARE and target_piece_value != LAKE_SQUARE:
            # Battle
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
            
            attacker_player = 1 if moving_piece_value > 0 else -1
            defender_player = 1 if target_piece_value > 0 else -1
            result = self.battle_resolver.resolve_battle(
                attacker_type, defender_type, attacker_player, defender_player
            )
            
            if result == 1:  # Attacker wins
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                battle_result = "attacker_wins"
                
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    self.win_type = 'flag_capture'
                    self._flag_positions[-self.current_player] = None
                    
            elif result == -1:  # Defender wins
                actual_board[r_from, c_from] = EMPTY_SQUARE
                self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                battle_result = "defender_wins"
                
            else:  # Both die
                actual_board[r_from, c_from] = EMPTY_SQUARE
                actual_board[r_to, c_to] = EMPTY_SQUARE
                for p_board in [self.board.visible_board_p1, self.board.visible_board_p2]:
                    p_board[r_from, c_from] = EMPTY_SQUARE
                    p_board[r_to, c_to] = EMPTY_SQUARE
                battle_result = "mutual_destruction"
        else:
            # Simple move
            self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
        
        self.turn_count += 1
        self.move_history.append(action)
        
        # Count valid moves for info
        num_valid_moves = len(self.get_valid_moves())
        
        # Switch player
        self.current_player *= -1
        
        # Check game end
        self._check_game_end()
        
        info = {
            "winner": self.winner,
            "win_type": self.win_type,
            "revealed_in_step": revealed_in_step,
            "game_phase": game_phase,
            "turn_count": self.turn_count,
            "num_valid_moves": num_valid_moves,
            "battle_result": battle_result,
            "attacker_rank": abs(moving_piece_value) if battle_result else None,
            "defender_rank": abs(target_piece_value) if battle_result else None
        }
        
        return self._get_game_state(), 0.0, self.game_over, info

    def _check_game_end(self):
        """Check for game-ending conditions."""
        # Check flag capture
        if self._flag_positions[1] is None or self._flag_positions[-1] is None:
            if not self.game_over:
                self.game_over = True
                self.winner = -self.current_player
        
        # Check if current player can move
        if not self.game_over:
            if not self.get_valid_moves():
                self.game_over = True
                self.winner = -self.current_player
                self.win_type = 'no_moves'
        
        # Timeout
        if not self.game_over and self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = 0
            self.win_type = 'timeout'

    def _get_game_state(self) -> GameState:
        """Get current game state with appropriate visibility."""
        if self.full_observability:
            board = self.board.actual_board.clone()
        else:
            board = self.board.get_visible_board(self.current_player).clone()
        
        game_state = GameState(
            board=board,
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            winner=self.winner,
            move_history=self.move_history[-100:] if len(self.move_history) > 100 else self.move_history.copy(),
            uncertainty_mask=torch.zeros(BOARD_SIZE, BOARD_SIZE, device=self.device),
            revealed_pieces_p1=dict(list(self.revealed_pieces_p1.items())[-50:]) if len(self.revealed_pieces_p1) > 50 else self.revealed_pieces_p1.copy(),
            revealed_pieces_p2=dict(list(self.revealed_pieces_p2.items())[-50:]) if len(self.revealed_pieces_p2) > 50 else self.revealed_pieces_p2.copy()
        )
        game_state.actual_board = self.board.actual_board.clone()
        return game_state

    def set_full_observability(self, enabled: bool):
        """Set full observability mode for curriculum learning."""
        self.full_observability = enabled

    def _check_safe_guards(self, r_from, c_from, r_to, c_to, moving_val, target_val):
        """Fast checks for obvious illegal moves."""
        # 1. Check ownership
        if self.current_player == 1:
            if moving_val <= 0:
                raise ValueError(f"P1 tried to move non-owned piece at {r_from},{c_from} (val={moving_val})")
            if target_val != 0 and target_val != LAKE_SQUARE and target_val > 0:
                raise ValueError(f"P1 tried to attack friendly piece at {r_to},{c_to} (val={target_val})")
        else:
            if moving_val >= 0:
                raise ValueError(f"P2 tried to move non-owned piece at {r_from},{c_from} (val={moving_val})")
            if target_val != 0 and target_val != LAKE_SQUARE and target_val < 0:
                raise ValueError(f"P2 tried to attack friendly piece at {r_to},{c_to} (val={target_val})")
                
        # 2. Check Lake
        if target_val == LAKE_SQUARE:
            raise ValueError(f"Player {self.current_player} tried to move into LAKE at {r_to},{c_to}")
            
        # 3. Check Distance (unless Scout)
        piece_type = abs(moving_val)
        if piece_type != PieceType.SCOUT.value:
            dist = abs(r_to - r_from) + abs(c_to - c_from)
            if dist != 1:
                raise ValueError(f"Non-Scout piece moved distance {dist} from {r_from},{c_from} to {r_to},{c_to}")

    def _check_strict_validation(self, action):
        """Slow check against full valid move list."""
        valid_moves = self.get_valid_moves()
        # Convert tuple of tuples to list of tuples for comparison if needed, 
        # but valid_moves is list of ((r1,c1), (r2,c2))
        
        if action not in valid_moves:
             raise ValueError(f"INVALID MOVE DETECTED: {action} is not in get_valid_moves().")
