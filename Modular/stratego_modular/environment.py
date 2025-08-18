# stratego_modular/environment.py

import torch
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from stratego_modular.board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from stratego_modular.piece import PieceType
from stratego_modular.battle import BattleResolver
from stratego_modular.game_state import GameState
from stratego_modular.dqn_visualizer import DQNMoveVisualizer

class StrategoEnvironment:
    """Stratego environment with hidden information management."""
    
    def __init__(self, device, record_game=False, episode_num=None):
        self.device = device
        self.record_game = record_game
        self.episode_num = episode_num
        self.board = Board(device)
        self.battle_resolver = BattleResolver()
        self.directions = torch.tensor([(0, 1), (0, -1), (1, 0), (-1, 0)], device=device)
        self.dqn_visualizer = DQNMoveVisualizer()
        self.reset()
        
    def reset(self) -> GameState:
        """Reset the environment to start a new game."""
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
        # Setup pieces in starting positions
        p1_pieces = self._generate_pieces()
        p2_pieces = self._generate_pieces()
        
        # Place pieces on the board
        self.board.setup_pieces(
            [(piece, (r, c)) for piece, (r, c) in zip(p1_pieces, self._get_p1_positions())],
            [(piece, (r, c)) for piece, (r, c) in zip(p2_pieces, self._get_p2_positions())]
        )
        
    def visualize_moves(self, move_index=None, save_path=None):
        """Visualize recorded moves using the DQNMoveVisualizer."""
        if move_index is not None:
            self.dqn_visualizer.visualize_move(move_index, save_path)
        else:
            self.dqn_visualizer.print_move_history()
            
    def clear_move_history(self):
        """Clear the recorded move history."""
        self.dqn_visualizer.clear_history()
        
        return self._get_game_state()
        
    def _generate_pieces(self) -> List[PieceType]:
        """Generate a list of pieces for one player."""
        pieces = [PieceType.FLAG, PieceType.SPY] + [PieceType.BOMB]*6 + [PieceType.MARSHAL] + \
                 [PieceType.GENERAL] + [PieceType.COLONEL]*2 + [PieceType.MAJOR]*3 + \
                 [PieceType.CAPTAIN]*4 + [PieceType.LIEUTENANT]*4 + [PieceType.SERGEANT]*4 + \
                 [PieceType.MINER]*5 + [PieceType.SCOUT]*8
        random.shuffle(pieces)
        return pieces
        
    def _get_p1_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 1."""
        positions = [(r, c) for r in range(6, 10) for c in range(10)]
        # Remove lake positions
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        random.shuffle(positions)
        return positions[:40]
        
    def _get_p2_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 2."""
        positions = [(r, c) for r in range(0, 4) for c in range(10)]
        # Remove lake positions
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        random.shuffle(positions)
        return positions[:40]
        
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all valid moves for the current player."""
        moves = []
        visible_board = self.board.get_visible_board(self.current_player)
        player_pieces = torch.nonzero((visible_board * self.current_player > 0) & (visible_board != LAKE_SQUARE))
        
        for r_from, c_from in player_pieces:
            r, c = r_from.item(), c_from.item()
            piece_value = visible_board[r, c].item()
            piece_type = PieceType(abs(piece_value))
            
            # Flags and bombs cannot move
            if piece_type in [PieceType.FLAG, PieceType.BOMB]:
                continue
                
            # Scout can move any distance in a straight line
            if piece_type == PieceType.SCOUT:
                for dr, dc in self.directions:
                    for i in range(1, BOARD_SIZE):
                        r_to, c_to = r + i * dr.item(), c + i * dc.item()
                        if not self.board.is_valid_target(self.current_player, r_to, c_to):
                            break
                        moves.append(((r, c), (r_to, c_to)))
                        # Stop if square is occupied
                        if visible_board[r_to, c_to].item() != EMPTY_SQUARE:
                            break
            else:
                # Other pieces move one square
                for dr, dc in self.directions:
                    r_to, c_to = r + dr.item(), c + dc.item()
                    if self.board.is_valid_target(self.current_player, r_to, c_to):
                        moves.append(((r, c), (r_to, c_to)))
                        
        return moves
        
    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[GameState, float, bool, Dict]:
        """Execute a move and return the new state."""
        if self.game_over:
            return self._get_game_state(), 0.0, True, {"winner": self.winner}
            
        (r_from, c_from), (r_to, c_to) = action
        reward = -0.01  # Small penalty for each move
        
        # Add penalty for repeating moves
        move_penalty = self.dqn_visualizer.get_move_penalty(action, self.current_player)
        reward += move_penalty
        
        # Get pieces involved in the move
        moving_piece_value = self.board.actual_board[r_from, c_from].item()
        target_piece_value = self.board.actual_board[r_to, c_to].item()
        
        # Handle battle or simple move
        if target_piece_value != EMPTY_SQUARE and target_piece_value != LAKE_SQUARE:
            # Battle occurs
            attacker_type = PieceType(abs(moving_piece_value))
            defender_type = PieceType(abs(target_piece_value))
            
            # Reveal pieces to both players
            self.board.reveal_pieces((r_from, c_from), (r_to, c_to))
            self.revealed_pieces_p1[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p2[(r_from, c_from)] = abs(moving_piece_value)
            self.revealed_pieces_p1[(r_to, c_to)] = abs(target_piece_value)
            self.revealed_pieces_p2[(r_to, c_to)] = abs(target_piece_value)
            
            # Determine player ownership for battle resolution
            # Player 1 has positive values, Player 2 has negative values
            attacker_player = 1 if moving_piece_value > 0 else 2 if moving_piece_value < 0 else 0
            defender_player = 1 if target_piece_value > 0 else 2 if target_piece_value < 0 else 0
            result = self.battle_resolver.resolve_battle(attacker_type, defender_type, attacker_player, defender_player)
            
            if result == 1:  # Attacker wins
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                reward += 0.1 * abs(target_piece_value)
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    reward += 1.0
            elif result == -1:  # Defender wins
                # Remove attacker
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                reward -= 0.1 * abs(moving_piece_value)
            else:  # Draw
                # Remove both pieces
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                self.board.actual_board[r_to, c_to] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_to, c_to] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_to, c_to] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_to, c_to] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_to, c_to] = EMPTY_SQUARE
        else:
            # Simple move to empty square
            self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
            
        self.turn_count += 1
        self.move_history.append(action)
        self.current_player *= -1
        
        # Record move for DQN visualization
        current_state = self._get_game_state()
        # Record the player who made the move (before switching)
        self.dqn_visualizer.record_move(action, current_state, self.current_player * -1)
        
        # Check for game end conditions
        self._check_game_end()
        
        return self._get_game_state(), reward, self.game_over, {"winner": self.winner}
        
    def _check_game_end(self):
        """Checks for game-ending conditions."""
        # Check if any flag exists on the board
        flags_exist = any(abs(p.item()) == PieceType.FLAG.value for p in self.board.actual_board.flatten())
        if not flags_exist:
             self.game_over = True
             self.winner = -self.current_player  # Winner is the player who captured the flag
        # Check if current player has any valid moves
        if not self.get_valid_moves():
            self.game_over = True
            self.winner = -self.current_player # Player who cannot move loses
        # Smart draw detection
        if self.turn_count > 300:  # Start checking after 300 moves
            # Check for repetitive positions (same piece arrangement)
            if self._is_position_repetitive():
                self.game_over = True
                self.winner = 0
                return
            # Check for minimal piece movement (stalemate)
            if self._is_stalemate():
                self.game_over = True
                self.winner = 0
                return
        # Ultimate limit
        if self.turn_count > 1000:
            self.game_over = True
            self.winner = 0
            
    def _is_position_repetitive(self):
        """Check if the same position has occurred multiple times"""
        if len(self.move_history) < 10:
            return False
        # Simple check: last 10 moves are mostly back-and-forth
        recent_moves = self.move_history[-10:]
        # Count how many moves are reversals of previous moves
        reversal_count = 0
        for i in range(1, len(recent_moves)):
            current_move = recent_moves[i]
            previous_move = recent_moves[i-1]
            if (current_move[0] == previous_move[1] and 
                current_move[1] == previous_move[0]):
                reversal_count += 1
        return reversal_count >= 3  # 3 or more reversals indicate stalemate

    def _is_stalemate(self):
        """Check if very few pieces are moving"""
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
        """Create a GameState object from the current environment state."""
        game_state = GameState(
            board=self.board.get_visible_board(self.current_player),
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            winner=self.winner,
            move_history=self.move_history.copy(),
            uncertainty_mask=torch.zeros(BOARD_SIZE, BOARD_SIZE),  # Simplified
            revealed_pieces_p1=self.revealed_pieces_p1.copy(),
            revealed_pieces_p2=self.revealed_pieces_p2.copy()
        )
        # Add actual board for visualization purposes
        game_state.actual_board = self.board.actual_board.clone()
        return game_state