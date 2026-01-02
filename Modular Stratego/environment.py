import torch
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from piece import PieceType
from battle import BattleResolver
from dqn_visualizer import DQNMoveVisualizer
from game_state import GameState
from training_config import REWARD_SCALE

class StrategoEnvironment:
    def __init__(self, device, record_game=False, episode_num=None, full_observability=False, enable_anti_stall=False, max_turns=500):
        self.device = device
        self.record_game = record_game
        self.episode_num = episode_num
        self.full_observability = full_observability  # For Phase 1 curriculum (cheating mode)
        self.enable_anti_stall = enable_anti_stall  # Only enable for agent-vs-agent games
        self.max_turns = max_turns  # Curriculum-based turn limit
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
        Get all valid moves for the current player using vectorized operations.
        """
        moves = []
        actual_board = self.board.actual_board
        
        # 1. Identify current player's pieces
        if self.current_player == 1:
            # Player 1: positive pieces
            player_pieces_mask = (actual_board > 0) & (actual_board != LAKE_SQUARE)
        else:
            # Player -1: negative pieces
            player_pieces_mask = (actual_board < 0) & (actual_board != LAKE_SQUARE)
            
        # Get indices of all pieces
        piece_indices = torch.nonzero(player_pieces_mask)
        if len(piece_indices) == 0:
            return []
            
        r_indices = piece_indices[:, 0]
        c_indices = piece_indices[:, 1]
        piece_values = actual_board[r_indices, c_indices]
        piece_types = torch.abs(piece_values)
        
        # Filter out Flags and Bombs (cannot move)
        movable_mask = (piece_types != PieceType.FLAG.value) & (piece_types != PieceType.BOMB.value)
        
        r_movable = r_indices[movable_mask]
        c_movable = c_indices[movable_mask]
        types_movable = piece_types[movable_mask]
        
        if len(r_movable) == 0:
            return []
            
        # 2. Handle Non-Scout Pieces (Step 1 square)
        non_scout_mask = (types_movable != PieceType.SCOUT.value)
        r_non_scout = r_movable[non_scout_mask]
        c_non_scout = c_movable[non_scout_mask]
        
        if len(r_non_scout) > 0:
            # Directions: (0, 1), (0, -1), (1, 0), (-1, 0)
            dr = torch.tensor([0, 0, 1, -1], device=self.device)
            dc = torch.tensor([1, -1, 0, 0], device=self.device)
            
            # Broadcasting: (N, 1) + (1, 4) -> (N, 4)
            r_targets = r_non_scout.unsqueeze(1) + dr.unsqueeze(0)
            c_targets = c_non_scout.unsqueeze(1) + dc.unsqueeze(0)
            
            # Flatten to (N*4) for easier filtering
            r_targets_flat = r_targets.view(-1)
            c_targets_flat = c_targets.view(-1)
            
            # Repeat source coordinates
            r_sources_flat = r_non_scout.unsqueeze(1).repeat(1, 4).view(-1)
            c_sources_flat = c_non_scout.unsqueeze(1).repeat(1, 4).view(-1)
            
            # Check bounds
            bounds_mask = (r_targets_flat >= 0) & (r_targets_flat < BOARD_SIZE) & \
                          (c_targets_flat >= 0) & (c_targets_flat < BOARD_SIZE)
            
            # Filter out of bounds first to avoid index error
            r_valid = r_targets_flat[bounds_mask]
            c_valid = c_targets_flat[bounds_mask]
            r_src_valid = r_sources_flat[bounds_mask]
            c_src_valid = c_sources_flat[bounds_mask]
            
            if len(r_valid) > 0:
                target_values = actual_board[r_valid, c_valid]
                
                # Check valid targets: Empty or Enemy
                # Invalid: Lake or Friendly
                
                # Lake check
                not_lake_mask = (target_values != LAKE_SQUARE)
                
                # Friendly fire check
                if self.current_player == 1:
                    # Invalid if target > 0 (friendly)
                    not_friendly_mask = (target_values <= 0)
                else:
                    # Invalid if target < 0 (friendly)
                    not_friendly_mask = (target_values >= 0)
                    
                valid_move_mask = not_lake_mask & not_friendly_mask
                
                # Extract final valid moves
                r_final_src = r_src_valid[valid_move_mask]
                c_final_src = c_src_valid[valid_move_mask]
                r_final_dst = r_valid[valid_move_mask]
                c_final_dst = c_valid[valid_move_mask]
                
                # Convert to list of tuples
                for i in range(len(r_final_src)):
                    moves.append(((int(r_final_src[i]), int(c_final_src[i])), 
                                  (int(r_final_dst[i]), int(c_final_dst[i]))))

        # 3. Handle Scout Pieces (Slide)
        scout_mask = (types_movable == PieceType.SCOUT.value)
        r_scout = r_movable[scout_mask]
        c_scout = c_movable[scout_mask]
        
        if len(r_scout) > 0:
            # For scouts, we iterate distances 1..9
            # This is harder to fully vectorize without a large memory footprint
            # But we can vectorize per direction
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in directions:
                # Start with all scouts active for this direction
                active_mask = torch.ones(len(r_scout), dtype=torch.bool, device=self.device)
                
                for dist in range(1, BOARD_SIZE):
                    if not active_mask.any():
                        break
                        
                    # Calculate targets at distance 'dist'
                    r_target = r_scout + dr * dist
                    c_target = c_scout + dc * dist
                    
                    # Check bounds
                    bounds_mask = (r_target >= 0) & (r_target < BOARD_SIZE) & \
                                  (c_target >= 0) & (c_target < BOARD_SIZE)
                    
                    # Update active mask (out of bounds stops the ray)
                    active_mask = active_mask & bounds_mask
                    
                    if not active_mask.any():
                        break
                        
                    # Check content for active rays
                    # We only check content where active_mask is True to avoid index errors
                    # But to keep tensors aligned, we get values for all, but use a safe index for inactive
                    # Alternatively, just filter indices
                    
                    active_indices = torch.nonzero(active_mask).squeeze(1)
                    r_active = r_target[active_indices]
                    c_active = c_target[active_indices]
                    
                    target_values = actual_board[r_active, c_active]
                    
                    # Check blocking conditions
                    is_lake = (target_values == LAKE_SQUARE)
                    is_occupied = (target_values != EMPTY_SQUARE)
                    
                    # Friendly fire
                    if self.current_player == 1:
                        is_friendly = (target_values > 0)
                    else:
                        is_friendly = (target_values < 0)
                        
                    # Valid move conditions:
                    # 1. Not Lake
                    # 2. Not Friendly
                    # 3. If Occupied (Enemy), it's valid but stops the ray (capture)
                    # 4. If Empty, it's valid and ray continues
                    
                    # Identify valid moves at this distance
                    valid_step_mask = (~is_lake) & (~is_friendly)
                    
                    # Add valid moves
                    valid_indices = active_indices[valid_step_mask]
                    if len(valid_indices) > 0:
                        r_src_valid = r_scout[valid_indices]
                        c_src_valid = c_scout[valid_indices]
                        r_dst_valid = r_target[valid_indices]
                        c_dst_valid = c_target[valid_indices]
                        
                        for i in range(len(valid_indices)):
                            moves.append(((int(r_src_valid[i]), int(c_src_valid[i])), 
                                          (int(r_dst_valid[i]), int(c_dst_valid[i]))))
                    
                    # Update active_mask for NEXT distance
                    # Ray stops if: Lake OR Occupied (even if enemy capture)
                    # So active if: Not Lake AND Not Occupied
                    # We need to map back to the original active_mask size
                    
                    # Determine which of the currently active rays should continue
                    continue_mask = (~is_lake) & (~is_occupied)
                    
                    # Update the main active_mask
                    # We only update the bits that were already True
                    # active_mask[active_indices] = continue_mask
                    # But we need to be careful with indexing.
                    
                    # Create a full-size continue mask (default False)
                    full_continue_mask = torch.zeros_like(active_mask)
                    full_continue_mask[active_indices] = continue_mask
                    
                    active_mask = active_mask & full_continue_mask

        return moves

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
            "num_valid_moves": avail_moves
        }
        
        return self._get_game_state(), 0.0, self.game_over, info

    def _check_game_end(self):
        # Checks for game-ending conditions.
        # Check if any flag exists on the board
        if hasattr(self, '_flag_positions'):
            flags_exist = (self._flag_positions[1] is not None) or (self._flag_positions[-1] is not None)
        else:
            # Fallback: scan board only if cache doesn't exist
            flags_exist = any(abs(p.item()) == PieceType.FLAG.value for p in self.board.actual_board.flatten())
        
        if not flags_exist:
             self.game_over = True
             self.winner = -self.current_player  # Winner is the player who captured the flag
             # Update flag cache
             if hasattr(self, '_flag_positions'):
                 self._flag_positions[self.current_player] = None
        
        if not self.game_over:
            # Check if current player has any valid moves
            if not self.get_valid_moves():
                self.game_over = True
                # Player who cannot move loses - opponent wins
                # This follows standard Stratego rules and rewards material dominance
                self.winner = -self.current_player
                self.win_type = 'no_moves'  # Track immobilization win
        
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
