# stratego_modular/environment.py

import torch
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from .board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from .piece import PieceType
from .battle import BattleResolver
from .game_state import GameState
from .dqn_visualizer import DQNMoveVisualizer

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
        
    def reset(self, p1_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None,
              p2_placement: Optional[List[Tuple[PieceType, Tuple[int, int]]]] = None) -> GameState:
        """
        Reset the environment to start a new game.
        
        Args:
            p1_placement: Optional custom placement for Player 1 (list of (piece, position) tuples)
            p2_placement: Optional custom placement for Player 2 (list of (piece, position) tuples)
        """
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
        # Track piece losses for exchange penalty mechanism
        # Format: {player: [(turn_count, piece_value, was_exchange), ...]}
        self.piece_losses = {1: [], -1: []}
        
        # Track exchanges (equal trades or captures) for each player
        # Format: {player: [turn_count, ...]}
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
        
        # Place pieces on the board
        self.board.setup_pieces(p1_placement, p2_placement)
        
        return self._get_game_state()
        
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
        
    def _get_p1_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 1. Must be exactly 40 positions in rows 6-9."""
        # Player 1 is in rows 6-9 (bottom 4 rows)
        # Lakes are in rows 4-5, so no overlap
        positions = [(r, c) for r in range(6, 10) for c in range(10)]
        # Remove lake positions (shouldn't be any, but check anyway)
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        # Ensure we have exactly 40 positions
        if len(positions) < 40:
            raise ValueError(f"Not enough positions for Player 1: {len(positions)} < 40")
        random.shuffle(positions)
        positions = positions[:40]  # Take exactly 40
        assert len(positions) == 40, f"Player 1 should have 40 positions, got {len(positions)}"
        return positions
        
    def _get_p2_positions(self) -> List[Tuple[int, int]]:
        """Get starting positions for Player 2. Must be exactly 40 positions in rows 0-3."""
        # Player 2 is in rows 0-3 (top 4 rows)
        # Lakes are in rows 4-5, so no overlap
        positions = [(r, c) for r in range(0, 4) for c in range(10)]
        # Remove lake positions (shouldn't be any, but check anyway)
        lake_positions = set((r.item(), c.item()) for r, c in self.board.lakes)
        positions = [pos for pos in positions if pos not in lake_positions]
        # Ensure we have exactly 40 positions
        if len(positions) < 40:
            raise ValueError(f"Not enough positions for Player 2: {len(positions)} < 40")
        random.shuffle(positions)
        positions = positions[:40]  # Take exactly 40
        assert len(positions) == 40, f"Player 2 should have 40 positions, got {len(positions)}"
        return positions
        
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all valid moves for the current player.
        Only returns moves that are actually valid:
        - Only pieces belonging to current player can move
        - Bombs and flags cannot move
        - Pieces cannot attack their own allies (friendly fire prevention)
        - Enemy pieces are only visible if revealed
        """
        moves = []
        actual_board = self.board.actual_board
        
        # CRITICAL: Use actual_board to get pieces belonging to current player
        # Player 1 has positive values, Player -1 has negative values
        if self.current_player == 1:
            # Player 1: get all positive pieces (excluding lakes)
            player_pieces = torch.nonzero((actual_board > 0) & (actual_board != LAKE_SQUARE))
        else:
            # Player -1: get all negative pieces (excluding lakes)
            player_pieces = torch.nonzero((actual_board < 0) & (actual_board != LAKE_SQUARE))
        
        for r_from, c_from in player_pieces:
            r, c = r_from.item(), c_from.item()
            piece_value_actual = actual_board[r, c].item()
            
            # Verify this piece actually belongs to current player
            if self.current_player == 1 and piece_value_actual <= 0:
                continue  # Not player 1's piece
            if self.current_player == -1 and piece_value_actual >= 0:
                continue  # Not player -1's piece
            
            piece_type = PieceType(abs(piece_value_actual))
            
            # Rule 1: Flags and bombs cannot move - exclude them completely
            if piece_type in [PieceType.FLAG, PieceType.BOMB]:
                continue
            
            # Rule 2: Scout can move any distance in a straight line
            if piece_type == PieceType.SCOUT:
                for dr, dc in self.directions:
                    for i in range(1, BOARD_SIZE):
                        r_to, c_to = r + i * dr.item(), c + i * dc.item()
                        
                        # Check bounds
                        if not (0 <= r_to < BOARD_SIZE and 0 <= c_to < BOARD_SIZE):
                            break
                        
                        # Check actual board to prevent friendly fire
                        target_actual = actual_board[r_to, c_to].item()
                        if target_actual == LAKE_SQUARE:
                            break
                        
                        # Friendly fire check: cannot attack own pieces
                        if target_actual != EMPTY_SQUARE:
                            # Check if target is same team (same sign)
                            if (piece_value_actual > 0 and target_actual > 0) or \
                               (piece_value_actual < 0 and target_actual < 0):
                                # Same team - cannot attack (friendly fire)
                                break
                        
                        # Valid target: empty or enemy
                        if target_actual == EMPTY_SQUARE:
                            moves.append(((r, c), (r_to, c_to)))
                        else:
                            # Enemy piece - can attack
                            moves.append(((r, c), (r_to, c_to)))
                            break  # Stop after capturing
            else:
                # Rule 3: Other pieces move one square only
                for dr, dc in self.directions:
                    r_to, c_to = r + dr.item(), c + dc.item()
                    
                    # Check bounds
                    if not (0 <= r_to < BOARD_SIZE and 0 <= c_to < BOARD_SIZE):
                        continue
                    
                    # Check actual board to prevent friendly fire
                    target_actual = actual_board[r_to, c_to].item()
                    if target_actual == LAKE_SQUARE:
                        continue
                    
                    # Friendly fire check: cannot attack own pieces
                    if target_actual != EMPTY_SQUARE:
                        # Check if target is same team (same sign)
                        if (piece_value_actual > 0 and target_actual > 0) or \
                           (piece_value_actual < 0 and target_actual < 0):
                            # Same team - cannot attack (friendly fire)
                            continue
                    
                    # Valid target: empty or enemy
                    if target_actual == EMPTY_SQUARE:
                        moves.append(((r, c), (r_to, c_to)))
                    else:
                        # Enemy piece - can attack
                        moves.append(((r, c), (r_to, c_to)))
                        
        return moves
        
    def step(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute a move and return the new state.
        Assumes action is already validated by get_valid_moves() - no additional checks needed.
        """
        if self.game_over:
            return self._get_game_state(), 0.0, True, {"winner": self.winner}
            
        (r_from, c_from), (r_to, c_to) = action
        
        # Get pieces involved in the move
        moving_piece_value = self.board.actual_board[r_from, c_from].item()
        target_piece_value = self.board.actual_board[r_to, c_to].item()
        
        # Calculate reward
        reward = -0.01  # Small penalty for each move
        
        # Add penalty for repeating moves
        move_penalty = self.dqn_visualizer.get_move_penalty(action, self.current_player)
        reward += move_penalty
        
        # Determine game phase for reward scaling
        game_phase = "early" if self.turn_count < 50 else ("mid" if self.turn_count < 200 else "end")
        phase_multiplier = 1.2 if game_phase == "early" else (1.0 if game_phase == "mid" else 0.8)
        
        # 1. IMPROVED: Reward for moving forward (toward enemy flag) with distance scaling
        # Player 1 (rows 6-9) moves forward when moving up (decreasing row)
        # Player 2 (rows 0-3) moves forward when moving down (increasing row)
        row_change = r_to - r_from
        col_change = abs(c_to - c_from)
        distance_moved = abs(row_change) + col_change
        
        if self.current_player == 1:
            # Player 1 moves forward when row decreases (moving up toward enemy)
            if row_change < 0:
                # Base reward + distance bonus (scouts can move multiple squares)
                forward_reward = 0.05 + (0.02 * min(distance_moved, 3))  # Cap at 3 squares
                reward += forward_reward * phase_multiplier
        else:  # Player -1
            # Player 2 moves forward when row increases (moving down toward enemy)
            if row_change > 0:
                # Base reward + distance bonus
                forward_reward = 0.05 + (0.02 * min(distance_moved, 3))
                reward += forward_reward * phase_multiplier
        
        # 2. NEW: Strategic positioning rewards
        # Center control (rows 4-5 are center, excluding lakes)
        center_rows = [4, 5]
        if r_to in center_rows and (r_to, c_to) not in [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]:
            reward += 0.05 * phase_multiplier  # Reward for controlling center
        
        # Territory control (pieces in enemy half)
        if self.current_player == 1:
            # Player 1 in enemy territory (rows 0-3)
            if r_to <= 3:
                reward += 0.02 * phase_multiplier
        else:
            # Player 2 in enemy territory (rows 6-9)
            if r_to >= 6:
                reward += 0.02 * phase_multiplier
        
        # Handle battle or simple move
        if target_piece_value != EMPTY_SQUARE and target_piece_value != LAKE_SQUARE:
            # Battle occurs - pieces are revealed during battle
            attacker_type = PieceType(abs(moving_piece_value))
            defender_type = PieceType(abs(target_piece_value))
            
            # Reveal pieces to both players (battle reveals both pieces)
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
            
            # 3. IMPROVED: Penalty for revealing own high-value pieces (scaled by piece value)
            attacker_rank = abs(moving_piece_value)
            if attacker_rank >= 8:  # High value piece
                # Scale penalty by piece value (MARSHAL=11 gets highest penalty)
                reveal_penalty = 0.3 * (attacker_rank / 11.0)
                reward -= reveal_penalty * phase_multiplier
            
            # 4. IMPROVED: Reward for revealing enemy high-value pieces (scaled by piece value)
            defender_rank = abs(target_piece_value)
            if defender_rank >= 8:  # High value piece
                # Scale reward by piece value
                reveal_reward = 0.3 * (defender_rank / 11.0)
                reward += reveal_reward * phase_multiplier
            
            # 5. NEW: Tactical rewards (special battle outcomes)
            # Miner defusing bomb
            if attacker_type == PieceType.MINER and defender_type == PieceType.BOMB:
                reward += 0.5 * phase_multiplier  # Significant reward for defusing bomb
            
            # Spy capturing Marshal
            if attacker_type == PieceType.SPY and defender_type == PieceType.MARSHAL:
                reward += 1.0 * phase_multiplier  # Big reward for spy capturing marshal
            
            # Scout reconnaissance (reward for revealing enemy pieces with scouts)
            if attacker_type == PieceType.SCOUT:
                reward += 0.1 * phase_multiplier  # Reward for scouting/revealing enemy
            
            if result == 1:  # Attacker wins
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                
                # IMPROVED: Better piece value scaling for captures
                captured_value = abs(target_piece_value)
                # Scale reward by piece rank (higher rank = more reward)
                capture_reward = 0.15 * (captured_value / 11.0)  # Normalized to 0-0.15
                reward += capture_reward * phase_multiplier
                
                # NEW: Trade evaluation (favorable vs unfavorable trades)
                lost_value = abs(moving_piece_value)
                value_difference = captured_value - lost_value
                if value_difference > 0:
                    # Favorable trade: captured higher value piece
                    trade_bonus = 0.2 * (value_difference / 11.0)
                    reward += trade_bonus * phase_multiplier
                elif value_difference < 0:
                    # Unfavorable trade: lost higher value piece
                    trade_penalty = 0.15 * (abs(value_difference) / 11.0)
                    reward -= trade_penalty * phase_multiplier
                else:
                    # Equal trade: small bonus
                    reward += 0.05 * phase_multiplier
                
                # Track exchange: attacker captured enemy piece
                # Store temporarily to record after turn increment
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(self.current_player)
                
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    # Flag capture reward (keep this, but remove duplicate at end)
                    reward += 10.0
            elif result == -1:  # Defender wins
                # Remove attacker
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                
                # IMPROVED: Better piece loss penalty (scaled by piece value)
                lost_value = abs(moving_piece_value)
                loss_penalty = 0.15 * (lost_value / 11.0)  # Normalized to 0-0.15
                reward -= loss_penalty * phase_multiplier
                
                # Track piece loss: attacker lost a piece (not an exchange for attacker)
                # Note: We'll record this after incrementing turn_count
                was_exchange = False
                # Store temporarily to record after turn increment
                self._pending_loss = (self.current_player, abs(moving_piece_value), was_exchange)
                
                # Defender captured attacker's piece - this is an exchange for the defender
                defender_player = -self.current_player
                # Store temporarily to record after turn increment
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(defender_player)
            else:  # Draw - both removed
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
                
                # Draw is an equal trade - both players lose a piece
                # Store temporarily to record after turn increment
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(self.current_player)
                self._pending_exchanges.append(-self.current_player)
                
                # Track piece loss for both players (equal trade)
                # Store temporarily to record after turn increment
                if not hasattr(self, '_pending_losses'):
                    self._pending_losses = []
                self._pending_losses.append((self.current_player, abs(moving_piece_value), True))
                self._pending_losses.append((-self.current_player, abs(target_piece_value), True))
        else:
            # Simple move to empty square
            self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
            
            # NEW: Piece preservation reward (reward for keeping high-value pieces alive)
            # Check if moving piece is high-value and moving to safer position
            moving_rank = abs(moving_piece_value)
            if moving_rank >= 8:  # High value piece
                # Reward for moving high-value piece (preservation)
                preservation_reward = 0.05 * (moving_rank / 11.0)
                reward += preservation_reward * phase_multiplier
        
        # Only increment turn and add to history for valid moves
        # (Invalid moves already returned above)
        self.turn_count += 1
        self.move_history.append(action)
        
        # Record pending losses and exchanges (now that turn_count is incremented)
        if hasattr(self, '_pending_loss'):
            player, piece_value, was_exchange = self._pending_loss
            self.piece_losses[player].append((self.turn_count, piece_value, was_exchange))
            delattr(self, '_pending_loss')
        
        if hasattr(self, '_pending_losses'):
            for player, piece_value, was_exchange in self._pending_losses:
                self.piece_losses[player].append((self.turn_count, piece_value, was_exchange))
            delattr(self, '_pending_losses')
        
        if hasattr(self, '_pending_exchanges'):
            for player in self._pending_exchanges:
                self.exchanges[player].append(self.turn_count)  # Use current turn count (after incrementing)
            delattr(self, '_pending_exchanges')
        
        # 2. Check for piece losses without exchange in the next 3 moves
        # Apply penalty if a piece was lost and no exchange occurred within 3 moves after the loss
        # Note: We check for the player who just moved (before switching)
        current_turn = self.turn_count
        player_who_moved = self.current_player
        
        # Check losses for the player who just moved
        # We check if a loss from exactly 3 turns ago didn't have an exchange in the 3 turns after it
        for loss_turn, piece_value, was_exchange in self.piece_losses[player_who_moved]:
            # Check if exactly 3 turns have passed since the loss
            if not was_exchange and (current_turn - loss_turn) == 3:
                # Check if there was an exchange in the 3 turns after the loss (loss_turn+1 to loss_turn+3)
                had_exchange = any(exchange_turn > loss_turn and exchange_turn <= (loss_turn + 3)
                                 for exchange_turn in self.exchanges[player_who_moved])
                if not had_exchange:
                    # Penalty for giving away a piece without exchange within 3 moves
                    reward -= 0.2 * (piece_value / 12.0)  # Scale penalty by piece value
        
        # Clean up old losses (older than 3 turns, already processed)
        self.piece_losses[player_who_moved] = [(t, v, e) for t, v, e in self.piece_losses[player_who_moved] 
                                               if (current_turn - t) <= 3]
        
        self.current_player *= -1
        
        # Record move for DQN visualization
        current_state = self._get_game_state()
        # Record the player who made the move (before switching)
        self.dqn_visualizer.record_move(action, current_state, self.current_player * -1)
        
        # NEW: Defensive rewards (flag protection and defensive positioning)
        # Check if piece is protecting flag or other high-value pieces
        if not self.game_over:
            # Find flag position for current player
            flag_pos = None
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    piece_val = self.board.actual_board[r, c].item()
                    if self.current_player == 1 and piece_val == PieceType.FLAG.value:
                        flag_pos = (r, c)
                        break
                    elif self.current_player == -1 and piece_val == -PieceType.FLAG.value:
                        flag_pos = (r, c)
                        break
                if flag_pos:
                    break
            
            # Check if moved piece is now protecting flag
            if flag_pos:
                flag_r, flag_c = flag_pos
                # Check if piece is adjacent to flag
                if abs(r_to - flag_r) <= 1 and abs(c_to - flag_c) <= 1:
                    moving_rank = abs(moving_piece_value)
                    if moving_rank >= 8:  # High value piece protecting flag
                        reward += 0.1 * phase_multiplier  # Reward for flag protection
            
            # NEW: Defensive positioning (strong pieces protecting weaker ones)
            # Check if moved piece is now adjacent to weaker friendly pieces
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_r, adj_c = r_to + dr, c_to + dc
                    if 0 <= adj_r < BOARD_SIZE and 0 <= adj_c < BOARD_SIZE:
                        adj_piece_val = self.board.actual_board[adj_r, adj_c].item()
                        # Check if adjacent piece is friendly and weaker
                        if (self.current_player == 1 and adj_piece_val > 0 and 
                            adj_piece_val < abs(moving_piece_value) and adj_piece_val != PieceType.BOMB.value):
                            reward += 0.05 * phase_multiplier  # Reward for defensive positioning
                        elif (self.current_player == -1 and adj_piece_val < 0 and 
                              abs(adj_piece_val) < abs(moving_piece_value) and abs(adj_piece_val) != PieceType.BOMB.value):
                            reward += 0.05 * phase_multiplier
        
        # NEW: Piece advantage reward (reward for having more pieces than opponent)
        if not self.game_over:
            p1_pieces = torch.sum((self.board.actual_board > 0) & (self.board.actual_board != LAKE_SQUARE)).item()
            p2_pieces = torch.sum((self.board.actual_board < 0) & (self.board.actual_board != LAKE_SQUARE)).item()
            
            if self.current_player == 1:
                piece_advantage = p1_pieces - p2_pieces
            else:
                piece_advantage = p2_pieces - p1_pieces
            
            if piece_advantage > 0:
                # Reward for piece advantage (scaled by advantage size)
                advantage_reward = 0.05 * min(piece_advantage / 10.0, 1.0)  # Cap at 10 piece advantage
                reward += advantage_reward * phase_multiplier
        
        # Check for game end conditions
        self._check_game_end()
        
        # FIXED: Remove duplicate win/loss rewards (these are handled in train_dqn.py)
        # Only keep flag capture reward which is already added above
        # Note: Win/loss rewards are now only in train_dqn.py to avoid duplication
        
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