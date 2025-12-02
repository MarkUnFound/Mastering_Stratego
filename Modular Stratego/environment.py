import torch
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from board import Board, BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE
from piece import PieceType
from battle import BattleResolver
from dqn_visualizer import DQNMoveVisualizer
from game_state import GameState

class StrategoEnvironment:
    def __init__(self, device, record_game=False, episode_num=None):
        self.device = device
        self.record_game = record_game
        self.episode_num = episode_num
        self.current_player = 1
        self.game_over = False
        self.winner = None
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
        self.turn_count = 0
        self.move_history = []
        self.revealed_pieces_p1 = {}
        self.revealed_pieces_p2 = {}
        
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
        
        directions_list = [(int(dr.item()), int(dc.item())) for dr, dc in self.directions]
        
        player_pieces_list = [(int(r.item()), int(c.item())) for r, c in player_pieces]
        
        for r, c in player_pieces_list:
            piece_value_actual = int(actual_board[r, c].item())
            
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
                for dr, dc in directions_list:
                    for i in range(1, BOARD_SIZE):
                        r_to, c_to = r + i * dr, c + i * dc
                        
                        # Check bounds
                        if not (0 <= r_to < BOARD_SIZE and 0 <= c_to < BOARD_SIZE):
                            break
                        
                        target_actual = int(actual_board[r_to, c_to].item())
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
                for dr, dc in directions_list:
                    r_to, c_to = r + dr, c + dc
                    
                    # Check bounds
                    if not (0 <= r_to < BOARD_SIZE and 0 <= c_to < BOARD_SIZE):
                        continue
                    
                    target_actual = int(actual_board[r_to, c_to].item())
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
        # Track revealed pieces for PBS training
        revealed_in_step = []

        if self.game_over:
            return self._get_game_state(), 0.0, True, {"winner": self.winner}

        # Check for max turns (draw) - reduced to encourage faster games
        if self.turn_count >= 400:
            self.game_over = True
            self.winner = 0
            return self._get_game_state(), -1.0, True, {"winner": 0, "revealed_in_step": [], "game_phase": "end", "turn_count": self.turn_count}
            
        if action is None:
            # No valid moves possible - player loses
            self.game_over = True
            self.winner = -self.current_player
            return self._get_game_state(), -1.0, True, {"winner": self.winner, "revealed_in_step": [], "game_phase": "end", "turn_count": self.turn_count}

        (r_from, c_from), (r_to, c_to) = action
        
        # Get pieces involved in the move
        moving_piece_value = self.board.actual_board[r_from, c_from].item()
        target_piece_value = self.board.actual_board[r_to, c_to].item()
        
        # Calculate reward
        # REWARD SCALING: Scale down by 10x (divide by 10) to prevent reward explosion
        REWARD_SCALE = 1.0  # Changed from 10.0 to 1.0 (divide by 10)
        
        reward = -0.01 * REWARD_SCALE  # Small penalty for each move (was -0.01, now -0.1)
        
        # Add penalty for repeating moves
        move_penalty = self.dqn_visualizer.get_move_penalty(action, self.current_player)
        reward += move_penalty * REWARD_SCALE  # Scale move penalty
        
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
                forward_reward = (0.05 + (0.02 * min(distance_moved, 3))) * REWARD_SCALE  # Cap at 3 squares
                reward += forward_reward * phase_multiplier
        else:  # Player -1
            # Player 2 moves forward when row increases (moving down toward enemy)
            if row_change > 0:
                # Base reward + distance bonus
                forward_reward = (0.05 + (0.02 * min(distance_moved, 3))) * REWARD_SCALE
                reward += forward_reward * phase_multiplier
        
        # 2. NEW: Strategic positioning rewards
        # Center control (rows 4-5 are center, excluding lakes)
        center_rows = [4, 5]
        if r_to in center_rows and (r_to, c_to) not in [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]:
            reward += 0.05 * REWARD_SCALE * phase_multiplier  # Reward for controlling center (was 0.05, now 0.5)
        
        # Territory control (pieces in enemy half)
        if self.current_player == 1:
            # Player 1 in enemy territory (rows 0-3)
            if r_to <= 3:
                reward += 0.02 * REWARD_SCALE * phase_multiplier  # (was 0.02, now 0.2)
        else:
            # Player 2 in enemy territory (rows 6-9)
            if r_to >= 6:
                reward += 0.02 * REWARD_SCALE * phase_multiplier  # (was 0.02, now 0.2)
        
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

            # Add to revealed_in_step for PBS training
            revealed_in_step.append(((r_from, c_from), attacker_type))
            revealed_in_step.append(((r_to, c_to), defender_type))
            
            # Determine player ownership for battle resolution
            # Player 1 has positive values, Player 2 has negative values
            attacker_player = 1 if moving_piece_value > 0 else 2 if moving_piece_value < 0 else 0
            defender_player = 1 if target_piece_value > 0 else 2 if target_piece_value < 0 else 0
            result = self.battle_resolver.resolve_battle(attacker_type, defender_type, attacker_player, defender_player)
            
            # 3. IMPROVED: Penalty for revealing own high-value pieces (scaled by piece value)
            attacker_rank = abs(moving_piece_value)
            if attacker_rank >= 8:  # High value piece
                # Scale penalty by piece value (MARSHAL=11 gets highest penalty)
                reveal_penalty = 0.3 * (attacker_rank / 11.0) * REWARD_SCALE
                reward -= reveal_penalty * phase_multiplier  # (was 0.3, now 3.0)
            
            # 4. IMPROVED: Reward for revealing enemy high-value pieces (scaled by piece value)
            defender_rank = abs(target_piece_value)
            if defender_rank >= 8:  # High value piece
                # Scale reward by piece value
                reveal_reward = 0.3 * (defender_rank / 11.0) * REWARD_SCALE
                reward += reveal_reward * phase_multiplier  # (was 0.3, now 3.0)
            
            # 5. NEW: Tactical rewards (special battle outcomes)
            # Miner defusing bomb
            if attacker_type == PieceType.MINER and defender_type == PieceType.BOMB:
                reward += 0.5 * REWARD_SCALE * phase_multiplier  # Significant reward for defusing bomb (was 0.5, now 5.0)
            
            # Spy capturing Marshal
            if attacker_type == PieceType.SPY and defender_type == PieceType.MARSHAL:
                reward += 1.0 * REWARD_SCALE * phase_multiplier  # Big reward for spy capturing marshal (was 1.0, now 10.0)
            
            # Scout reconnaissance (reward for revealing enemy pieces with scouts)
            if attacker_type == PieceType.SCOUT:
                reward += 0.1 * REWARD_SCALE * phase_multiplier  # Reward for scouting/revealing enemy (was 0.1, now 1.0)
                
                # 14. NEW: Scout Mobility Bonus
                distance = abs(row_change) + abs(col_change)
                if distance >= 3:
                    scout_mobility_bonus = 0.1 * (distance / 8.0) * REWARD_SCALE
                    reward += scout_mobility_bonus * phase_multiplier
            
            # 8. NEW: Information Gathering Rewards (Enhanced)
            # Reward for revealing enemy pieces with low-value pieces (good trade)
            attacker_rank = abs(moving_piece_value)
            defender_rank = abs(target_piece_value)
            if attacker_rank <= 3 and defender_rank >= 5:
                info_gathering_reward = 0.2 * (defender_rank / 11.0) * REWARD_SCALE
                reward += info_gathering_reward * phase_multiplier
            
            # 15. NEW: Piece Value Discovery Rewards
            # Extra reward for discovering enemy marshal/general
            if defender_rank >= 9:
                # Check if this piece was previously revealed
                was_previously_revealed = (r_to, c_to) in self.revealed_pieces_p1 or \
                                         (r_to, c_to) in self.revealed_pieces_p2
                if not was_previously_revealed:
                    discovery_reward = 0.3 * (defender_rank / 11.0) * REWARD_SCALE
                    reward += discovery_reward * phase_multiplier
            
            if result == 1:  # Attacker wins
                self.board.move_piece(self.current_player, (r_from, c_from), (r_to, c_to))
                
                if hasattr(self, '_cached_piece_counts'):
                    # Piece was captured, update cache
                    self._cached_piece_counts[-self.current_player] = max(0, self._cached_piece_counts[-self.current_player] - 1)
                
                # IMPROVED: Better piece value scaling for captures
                captured_value = abs(target_piece_value)
                # Scale reward by piece rank (higher rank = more reward)
                capture_reward = 0.15 * (captured_value / 11.0) * REWARD_SCALE  # Normalized to 0-1.5 (was 0-0.15)
                reward += capture_reward * phase_multiplier
                
                lost_value = abs(moving_piece_value)
                value_difference = captured_value - lost_value
                if value_difference > 0:
                    # Favorable trade: captured higher value piece
                    trade_bonus = 0.2 * (value_difference / 11.0) * REWARD_SCALE
                    reward += trade_bonus * phase_multiplier  # (was 0.2, now 2.0)
                elif value_difference < 0:
                    # Unfavorable trade: lost higher value piece
                    trade_penalty = 0.15 * (abs(value_difference) / 11.0) * REWARD_SCALE
                    reward -= trade_penalty * phase_multiplier  # (was 0.15, now 1.5)
                else:
                    # Equal trade: small bonus
                    reward += 0.05 * REWARD_SCALE * phase_multiplier  # (was 0.05, now 0.5)
                
                # Track exchange: attacker captured enemy piece
                # Store temporarily to record after turn increment
                if not hasattr(self, '_pending_exchanges'):
                    self._pending_exchanges = []
                self._pending_exchanges.append(self.current_player)
                
                if defender_type == PieceType.FLAG:
                    self.game_over = True
                    self.winner = self.current_player
                    # Flag capture reward - make it MUCH larger to incentivize winning
                    # Keep it large relative to other rewards to encourage aggressive play
                    reward += 50.0  # Increased from 10.0 to 50.0 (5x larger) to incentivize winning
                    if hasattr(self, '_flag_positions'):
                        self._flag_positions[-self.current_player] = None
            elif result == -1:  # Defender wins
                # Remove attacker
                self.board.actual_board[r_from, c_from] = EMPTY_SQUARE
                if self.current_player == 1:
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                else:
                    self.board.visible_board_p2[r_from, c_from] = EMPTY_SQUARE
                    self.board.visible_board_p1[r_from, c_from] = EMPTY_SQUARE
                
                if hasattr(self, '_cached_piece_counts'):
                    self._cached_piece_counts[self.current_player] = max(0, self._cached_piece_counts[self.current_player] - 1)
                
                # IMPROVED: Better piece loss penalty (scaled by piece value)
                lost_value = abs(moving_piece_value)
                loss_penalty = 0.15 * (lost_value / 11.0) * REWARD_SCALE  # Normalized to 0-1.5 (was 0-0.15)
                
                # 5. NEW: Endgame Behavior Rewards - Heavier penalty for losing pieces in endgame
                if game_phase == "end":
                    endgame_loss_penalty = 0.3 * (lost_value / 11.0) * REWARD_SCALE
                    reward -= endgame_loss_penalty
                else:
                    reward -= loss_penalty * phase_multiplier
                
                # Track piece loss: attacker lost a piece (not an exchange for attacker)
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
                
                if hasattr(self, '_cached_piece_counts'):
                    self._cached_piece_counts[self.current_player] = max(0, self._cached_piece_counts[self.current_player] - 1)
                    self._cached_piece_counts[-self.current_player] = max(0, self._cached_piece_counts[-self.current_player] - 1)
                
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
            
            # Check if moving piece is high-value and moving to safer position
            moving_rank = abs(moving_piece_value)
            if moving_rank >= 8:  # High value piece
                # Reward for moving high-value piece (preservation)
                preservation_reward = 0.05 * (moving_rank / 11.0) * REWARD_SCALE
                reward += preservation_reward * phase_multiplier  # (was 0.05, now 0.5)
            
            if game_phase == "end":
                enemy_flag_pos = self._flag_positions[-self.current_player] if hasattr(self, '_flag_positions') else None
                if enemy_flag_pos:
                    enemy_flag_r, enemy_flag_c = enemy_flag_pos
                    distance_after = abs(r_to - enemy_flag_r) + abs(c_to - enemy_flag_c)
                    distance_before = abs(r_from - enemy_flag_r) + abs(c_from - enemy_flag_c)
                    if distance_after < distance_before:
                        endgame_aggression_reward = 0.5 * REWARD_SCALE  # Increased from 0.2 to 0.5 (more reward for aggressive play)
                        reward += endgame_aggression_reward
                    # Additional reward for being very close to flag
                    if distance_after < 3:
                        proximity_bonus = 0.3 * REWARD_SCALE  # Bonus for being within 3 squares
                        reward += proximity_bonus
            
            # 9. NEW: Tactical Support Rewards
            # Count pieces that can support the moved piece in battle
            supporting_pieces = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    supp_r, supp_c = r_to + dr, c_to + dc
                    if 0 <= supp_r < BOARD_SIZE and 0 <= supp_c < BOARD_SIZE:
                        supp_piece = int(self.board.actual_board[supp_r, supp_c].item())
                        if (self.current_player == 1 and supp_piece > 0) or \
                           (self.current_player == -1 and supp_piece < 0):
                            supporting_pieces += 1
            
            if supporting_pieces >= 2:
                support_reward = 0.1 * min(supporting_pieces / 3.0, 1.0) * REWARD_SCALE
                reward += support_reward * phase_multiplier
        
        # Only increment turn and add to history for valid moves
        # (Invalid moves already returned above)
        self.turn_count += 1
        self.move_history.append(action)
        
        # 10. NEW: Piece Economy / Value Preservation
        # Track piece value over time (every 10 moves)
        if self.turn_count % 10 == 0:
            if not hasattr(self, '_previous_piece_value'):
                self._previous_piece_value = {1: 0, -1: 0}
            
            # Calculate current total piece value
            p1_pieces_mask = (self.board.actual_board > 0) & (self.board.actual_board != LAKE_SQUARE)
            p2_pieces_mask = (self.board.actual_board < 0) & (self.board.actual_board != LAKE_SQUARE)
            current_p1_value = torch.sum(torch.abs(self.board.actual_board[p1_pieces_mask])).item()
            current_p2_value = torch.sum(torch.abs(self.board.actual_board[p2_pieces_mask])).item()
            
            if self.current_player == 1:
                current_value = current_p1_value
                previous_value = self._previous_piece_value[1]
            else:
                current_value = current_p2_value
                previous_value = self._previous_piece_value[-1]
            
            value_change = current_value - previous_value
            if value_change > 0:
                # Reward for gaining value (captures)
                value_gain_reward = 0.15 * (value_change / 50.0) * REWARD_SCALE
                reward += value_gain_reward
            elif value_change < -5:
                # Penalty for losing significant value
                value_loss_penalty = 0.2 * (abs(value_change) / 50.0) * REWARD_SCALE
                reward -= value_loss_penalty
            
            # Update stored values
            self._previous_piece_value[1] = current_p1_value
            self._previous_piece_value[-1] = current_p2_value
        
        # 11. NEW: Stalemate Prevention
        # Check if move reduces available moves significantly (check before switching players)
        if not hasattr(self, '_previous_move_count'):
            self._previous_move_count = {1: 0, -1: 0}
        
        # Count available moves after this move (before player switch)
        available_moves_after = len(self.get_valid_moves())
        player_who_moved = self.current_player  # Store before switching
        available_moves_before = self._previous_move_count.get(player_who_moved, available_moves_after)
        
        if available_moves_after < available_moves_before * 0.5 and available_moves_before > 0:
            stalemate_penalty = 0.05 * REWARD_SCALE  # Reduced from 0.15 to 0.05 (less penalty to encourage aggressive play)
            reward -= stalemate_penalty * phase_multiplier
        
        # Store current move count for next turn
        self._previous_move_count[player_who_moved] = available_moves_after
        
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
        current_turn = self.turn_count
        player_who_moved = self.current_player
        
        # Check losses for the player who just moved
        # We check if a loss from exactly 3 turns ago didn't have an exchange in the 3 turns after the loss (loss_turn+1 to loss_turn+3)
        for loss_turn, piece_value, was_exchange in self.piece_losses[player_who_moved]:
            # Check if exactly 3 turns have passed since the loss
            if not was_exchange and (current_turn - loss_turn) == 3:
                # Check if there was an exchange in the 3 turns after the loss (loss_turn+1 to loss_turn+3)
                had_exchange = any(exchange_turn > loss_turn and exchange_turn <= (loss_turn + 3)
                                 for exchange_turn in self.exchanges[player_who_moved])
                if not had_exchange:
                    # Penalty for giving away a piece without exchange within 3 moves
                    reward -= 0.2 * (piece_value / 12.0) * REWARD_SCALE  # Scale penalty by piece value (was 0.2, now 2.0)
        
        # Clean up old losses (older than 3 turns, already processed)
        self.piece_losses[player_who_moved] = [(t, v, e) for t, v, e in self.piece_losses[player_who_moved] 
                                               if (current_turn - t) <= 3]
        
        self.current_player *= -1
        
        # Record move for DQN visualization
        current_state = self._get_game_state()
        # Record the player who made the move (before switching)
        self.dqn_visualizer.record_move(action, current_state, self.current_player * -1)
        
        # Check if piece is protecting flag or other high-value pieces
        if not self.game_over:
            flag_pos = None
            if not hasattr(self, '_flag_positions'):
                self._flag_positions = {1: None, -1: None}
            
            # Check if we need to find flag position (only if not cached or might have changed)
            if self._flag_positions[self.current_player] is None or self.turn_count % 10 == 0:
                if self.current_player == 1:
                    # Find positive FLAG value
                    flag_mask = (self.board.actual_board == PieceType.FLAG.value)
                    flag_positions = torch.nonzero(flag_mask)
                    if len(flag_positions) > 0:
                        r, c = int(flag_positions[0, 0].item()), int(flag_positions[0, 1].item())
                        flag_pos = (r, c)
                        self._flag_positions[1] = flag_pos
                    else:
                        flag_pos = None
                else:
                    # Find negative FLAG value
                    flag_mask = (self.board.actual_board == -PieceType.FLAG.value)
                    flag_positions = torch.nonzero(flag_mask)
                    if len(flag_positions) > 0:
                        r, c = int(flag_positions[0, 0].item()), int(flag_positions[0, 1].item())
                        flag_pos = (r, c)
                        self._flag_positions[-1] = flag_pos
                    else:
                        flag_pos = None
            else:
                flag_pos = self._flag_positions[self.current_player]
            
            if flag_pos:
                flag_r, flag_c = flag_pos
                # Count pieces protecting flag (adjacent pieces)
                flag_protection_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        prot_r, prot_c = flag_r + dr, flag_c + dc
                        if 0 <= prot_r < BOARD_SIZE and 0 <= prot_c < BOARD_SIZE:
                            prot_piece = int(self.board.actual_board[prot_r, prot_c].item())
                            if (self.current_player == 1 and prot_piece > 0) or \
                               (self.current_player == -1 and prot_piece < 0):
                                flag_protection_count += 1
                
                if flag_protection_count >= 2:
                    protection_reward = 0.1 * min(flag_protection_count / 4.0, 1.0) * REWARD_SCALE
                    reward += protection_reward * phase_multiplier
                elif flag_protection_count == 0:
                    # Penalty if flag is unprotected
                    protection_penalty = 0.2 * REWARD_SCALE
                    reward -= protection_penalty * phase_multiplier
                
                # Check if moved piece is now protecting flag
                if abs(r_to - flag_r) <= 1 and abs(c_to - flag_c) <= 1:
                    moving_rank = abs(moving_piece_value)
                    if moving_rank >= 8:  # High value piece protecting flag
                        reward += 0.1 * REWARD_SCALE * phase_multiplier  # Reward for flag protection
            
            # Check if moved piece is now adjacent to weaker friendly pieces
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_r, adj_c = r_to + dr, c_to + dc
                    if 0 <= adj_r < BOARD_SIZE and 0 <= adj_c < BOARD_SIZE:
                        adj_piece_val = int(self.board.actual_board[adj_r, adj_c].item())
                        # Check if adjacent piece is friendly and weaker
                        if (self.current_player == 1 and adj_piece_val > 0 and 
                            adj_piece_val < abs(moving_piece_value) and adj_piece_val != PieceType.BOMB.value):
                            reward += 0.05 * REWARD_SCALE * phase_multiplier  # Reward for defensive positioning
                        elif (self.current_player == -1 and adj_piece_val < 0 and 
                              abs(adj_piece_val) < abs(moving_piece_value) and abs(adj_piece_val) != PieceType.BOMB.value):
                            reward += 0.05 * REWARD_SCALE * phase_multiplier
            
            # Check if moved piece is now vulnerable (adjacent to enemy pieces)
            vulnerable_adjacent_enemies = 0
            moving_rank = abs(moving_piece_value)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_r, adj_c = r_to + dr, c_to + dc
                    if 0 <= adj_r < BOARD_SIZE and 0 <= adj_c < BOARD_SIZE:
                        adj_piece = int(self.board.actual_board[adj_r, adj_c].item())
                        # Check if adjacent piece is enemy
                        if (self.current_player == 1 and adj_piece < 0) or \
                           (self.current_player == -1 and adj_piece > 0):
                            vulnerable_adjacent_enemies += 1
            
            if vulnerable_adjacent_enemies > 0:
                # Penalty increases with number of threats and piece value
                vulnerability_penalty = 0.1 * vulnerable_adjacent_enemies * (moving_rank / 11.0) * REWARD_SCALE
                reward -= vulnerability_penalty * phase_multiplier
            
            # 2. NEW: Piece Coordination / Formation Rewards
            # Check if multiple pieces are in enemy territory together
            pieces_in_enemy_territory = 0
            if self.current_player == 1:
                # Player 1 in enemy territory (rows 0-3)
                enemy_territory_mask = (self.board.actual_board > 0) & (self.board.actual_board != LAKE_SQUARE)
                for r in range(4):  # Rows 0-3
                    pieces_in_enemy_territory += torch.sum(enemy_territory_mask[r, :]).item()
            else:
                # Player 2 in enemy territory (rows 6-9)
                enemy_territory_mask = (self.board.actual_board < 0) & (self.board.actual_board != LAKE_SQUARE)
                for r in range(6, 10):  # Rows 6-9
                    pieces_in_enemy_territory += torch.sum(enemy_territory_mask[r, :]).item()
            
            if pieces_in_enemy_territory >= 3:
                coordination_reward = 0.15 * min(pieces_in_enemy_territory / 5.0, 1.0) * REWARD_SCALE
                reward += coordination_reward * phase_multiplier
            
            # 7. NEW: Control of Key Squares
            # Calculate distance to enemy flag
            enemy_flag_pos = self._flag_positions[-self.current_player] if hasattr(self, '_flag_positions') else None
            if enemy_flag_pos:
                enemy_flag_r, enemy_flag_c = enemy_flag_pos
                distance_to_enemy_flag = abs(r_to - enemy_flag_r) + abs(c_to - enemy_flag_c)
                if distance_to_enemy_flag <= 2:
                    key_square_reward = 0.15 * (1.0 - distance_to_enemy_flag / 2.0) * REWARD_SCALE
                    reward += key_square_reward * phase_multiplier
            
            # Check if controlling chokepoint (around lakes)
            lakes = [(4,2), (4,3), (5,2), (5,3), (4,6), (4,7), (5,6), (5,7)]
            chokepoints = [(3,2), (3,3), (6,2), (6,3), (3,6), (3,7), (6,6), (6,7),
                          (4,1), (4,4), (5,1), (5,4), (4,5), (4,8), (5,5), (5,8)]
            if (r_to, c_to) in chokepoints:
                chokepoint_reward = 0.1 * REWARD_SCALE
                reward += chokepoint_reward * phase_multiplier
            
            # 6. NEW: Piece Mobility Rewards
            # Count valid moves from new position
            mobility = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                r_new, c_new = r_to + dr, c_to + dc
                if 0 <= r_new < BOARD_SIZE and 0 <= c_new < BOARD_SIZE:
                    target_val = int(self.board.actual_board[r_new, c_new].item())
                    if target_val == EMPTY_SQUARE or target_val == LAKE_SQUARE:
                        mobility += 1
                    elif (self.current_player == 1 and target_val < 0) or \
                         (self.current_player == -1 and target_val > 0):
                        mobility += 1  # Can attack
            
            if mobility >= 3:
                mobility_reward = 0.05 * (mobility / 8.0) * REWARD_SCALE
                reward += mobility_reward * phase_multiplier
            elif mobility == 0:
                # Penalty for getting piece stuck
                stuck_penalty = 0.1 * (moving_rank / 11.0) * REWARD_SCALE
                reward -= stuck_penalty * phase_multiplier
            
            # 13. NEW: Bomb Protection Rewards
            # Check if moved piece is now protecting a bomb
            adjacent_bombs = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_r, adj_c = r_to + dr, c_to + dc
                    if 0 <= adj_r < BOARD_SIZE and 0 <= adj_c < BOARD_SIZE:
                        adj_piece = int(self.board.actual_board[adj_r, adj_c].item())
                        if abs(adj_piece) == PieceType.BOMB.value:
                            if (self.current_player == 1 and adj_piece > 0) or \
                               (self.current_player == -1 and adj_piece < 0):
                                adjacent_bombs += 1
            
            if adjacent_bombs > 0 and moving_rank >= 5:
                bomb_protection_reward = 0.1 * adjacent_bombs * REWARD_SCALE
                reward += bomb_protection_reward * phase_multiplier
        
        if not self.game_over:
            if not hasattr(self, '_cached_piece_counts') or self.turn_count % 5 == 0:
                self._cached_piece_counts = {
                    1: torch.sum((self.board.actual_board > 0) & (self.board.actual_board != LAKE_SQUARE)).item(),
                    -1: torch.sum((self.board.actual_board < 0) & (self.board.actual_board != LAKE_SQUARE)).item()
                }
            p1_pieces = self._cached_piece_counts[1]
            p2_pieces = self._cached_piece_counts[-1]
            
            if self.current_player == 1:
                piece_advantage = p1_pieces - p2_pieces
            else:
                piece_advantage = p2_pieces - p1_pieces
            
            if piece_advantage > 0:
                # Reward for piece advantage (scaled by advantage size)
                advantage_reward = 0.05 * min(piece_advantage / 10.0, 1.0) * REWARD_SCALE  # Cap at 10 piece advantage
                reward += advantage_reward * phase_multiplier
            
            # Calculate total piece value for each player
            p1_value = torch.sum(torch.abs(self.board.actual_board[(self.board.actual_board > 0) & (self.board.actual_board != LAKE_SQUARE)])).item()
            p2_value = torch.sum(torch.abs(self.board.actual_board[(self.board.actual_board < 0) & (self.board.actual_board != LAKE_SQUARE)])).item()
            
            if self.current_player == 1:
                value_advantage = p1_value - p2_value
            else:
                value_advantage = p2_value - p1_value
            
            if value_advantage > 0:
                # Reward for value advantage (scaled by advantage)
                value_reward = 0.1 * min(value_advantage / 50.0, 1.0) * REWARD_SCALE
                reward += value_reward * phase_multiplier
        
        # Check for game end conditions
        self._check_game_end()
        
        # Win/loss rewards are now only in train_dqn.py to avoid duplication
        
        return self._get_game_state(), reward, self.game_over, {"winner": self.winner, "revealed_in_step": revealed_in_step, "game_phase": game_phase, "turn_count": self.turn_count}
        
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
                self.winner = -self.current_player # Player who cannot move loses
        # Smart draw detection (relaxed to allow more decisive games)
        if self.turn_count > 400:  # Increased from 300 to 400 moves before checking (gives more time for decisive play)
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
        # Ultimate limit (increased to allow longer games)
        if self.turn_count > 1500:  # Increased from 1000 to 1500
            self.game_over = True
            self.winner = 0
            
    def _is_position_repetitive(self):
        # Check if the same position has occurred multiple times
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

        game_state = GameState(
            board=self.board.get_visible_board(self.current_player),
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
