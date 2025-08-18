import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum

class Piece(Enum):
    EMPTY = 0
    WATER = -1
    # Red pieces (1-12)
    FLAG_R = 1
    SPY_R = 2
    SCOUT_R = 3
    MINER_R = 4
    SERGEANT_R = 5
    LIEUTENANT_R = 6
    CAPTAIN_R = 7
    MAJOR_R = 8
    COLONEL_R = 9
    GENERAL_R = 10
    MARSHAL_R = 11
    BOMB_R = 12
    # Blue pieces (13-24)
    FLAG_B = 13
    SPY_B = 14
    SCOUT_B = 15
    MINER_B = 16
    SERGEANT_B = 17
    LIEUTENANT_B = 18
    CAPTAIN_B = 19
    MAJOR_B = 20
    COLONEL_B = 21
    GENERAL_B = 22
    MARSHAL_B = 23
    BOMB_B = 24

class StrategoEnv:
    def __init__(self):
        self.board_size = 10
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 0  # 0 for red, 1 for blue
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.max_moves = 500  # Prevent infinite games
        
        # Water squares (lakes)
        self.water_squares = [(4, 2), (4, 3), (5, 2), (5, 3), 
                             (4, 6), (4, 7), (5, 6), (5, 7)]
        
        # Initialize piece counts
        self.red_pieces = {
            Piece.FLAG_R.value: 1, Piece.SPY_R.value: 1, Piece.SCOUT_R.value: 8,
            Piece.MINER_R.value: 5, Piece.SERGEANT_R.value: 4, Piece.LIEUTENANT_R.value: 4,
            Piece.CAPTAIN_R.value: 4, Piece.MAJOR_R.value: 3, Piece.COLONEL_R.value: 2,
            Piece.GENERAL_R.value: 1, Piece.MARSHAL_R.value: 1, Piece.BOMB_R.value: 6
        }
        
        self.blue_pieces = {
            Piece.FLAG_B.value: 1, Piece.SPY_B.value: 1, Piece.SCOUT_B.value: 8,
            Piece.MINER_B.value: 5, Piece.SERGEANT_B.value: 4, Piece.LIEUTENANT_B.value: 4,
            Piece.CAPTAIN_B.value: 4, Piece.MAJOR_B.value: 3, Piece.COLONEL_B.value: 2,
            Piece.GENERAL_B.value: 1, Piece.MARSHAL_B.value: 1, Piece.BOMB_B.value: 6
        }
        
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.move_count = 0
        
        # Set water squares
        for r, c in self.water_squares:
            self.board[r, c] = Piece.WATER.value
        
        # Place pieces randomly for both players
        self._place_pieces_randomly()
        
        return self.get_state()
    
    def _place_pieces_randomly(self):
        """Randomly place pieces for both players"""
        # Red pieces (bottom 4 rows)
        red_positions = [(r, c) for r in range(6, 10) for c in range(10)]
        random.shuffle(red_positions)
        
        piece_idx = 0
        for piece_type, count in self.red_pieces.items():
            for _ in range(count):
                if piece_idx < len(red_positions):
                    r, c = red_positions[piece_idx]
                    self.board[r, c] = piece_type
                    piece_idx += 1
        
        # Blue pieces (top 4 rows)
        blue_positions = [(r, c) for r in range(4) for c in range(10)]
        random.shuffle(blue_positions)
        
        piece_idx = 0
        for piece_type, count in self.blue_pieces.items():
            for _ in range(count):
                if piece_idx < len(blue_positions):
                    r, c = blue_positions[piece_idx]
                    self.board[r, c] = piece_type
                    piece_idx += 1
    
    def get_state(self):
        """Get the current state of the game"""
        # Return a flattened view of the board plus current player
        state = self.board.flatten()
        return np.append(state, self.current_player)
    
    def get_valid_actions(self, player=None):
        """Get all valid actions for the current player"""
        if player is None:
            player = self.current_player
        
        valid_actions = []
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board[r, c]
                
                # Check if this is the player's piece and can move
                if self._is_player_piece(piece, player) and self._can_piece_move(piece):
                    # Check all possible moves
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        
                        # Scout can move multiple squares
                        if self._is_scout(piece):
                            distance = 1
                            while self._is_valid_move(r, c, nr, nc, player):
                                action = self._encode_action(r, c, nr, nc)
                                valid_actions.append(action)
                                
                                # If we hit an enemy piece, can't go further
                                if self.board[nr, nc] != Piece.EMPTY.value:
                                    break
                                
                                # Continue in same direction
                                distance += 1
                                nr, nc = r + dr * distance, c + dc * distance
                        else:
                            if self._is_valid_move(r, c, nr, nc, player):
                                action = self._encode_action(r, c, nr, nc)
                                valid_actions.append(action)
        
        return valid_actions
    
    def step(self, action):
        """Execute an action and return (next_state, reward, done, info)"""
        if self.game_over:
            return self.get_state(), 0, True, {"winner": self.winner}
        
        # Decode action
        from_r, from_c, to_r, to_c = self._decode_action(action)
        
        # Validate action
        if not self._is_valid_move(from_r, from_c, to_r, to_c, self.current_player):
            # Invalid move penalty
            return self.get_state(), -10, False, {"invalid_move": True}
        
        # Execute move
        reward = self._execute_move(from_r, from_c, to_r, to_c)
        
        # Check win conditions
        self._check_game_over()
        
        # Switch players
        if not self.game_over:
            self.current_player = 1 - self.current_player
        
        self.move_count += 1
        if self.move_count >= self.max_moves:
            self.game_over = True
            self.winner = None  # Draw
        
        return self.get_state(), reward, self.game_over, {"winner": self.winner}
    
    def _is_valid_move(self, from_r, from_c, to_r, to_c, player):
        """Check if a move is valid"""
        # Bounds check
        if not (0 <= to_r < self.board_size and 0 <= to_c < self.board_size):
            return False
        
        # Can't move to water
        if self.board[to_r, to_c] == Piece.WATER.value:
            return False
        
        # Must be player's piece
        if not self._is_player_piece(self.board[from_r, from_c], player):
            return False
        
        # Can't move immobile pieces
        if not self._can_piece_move(self.board[from_r, from_c]):
            return False
        
        # Can't capture own pieces
        if self._is_player_piece(self.board[to_r, to_c], player):
            return False
        
        # Check movement distance (scout can move multiple squares)
        piece = self.board[from_r, from_c]
        if self._is_scout(piece):
            # Scout can move in straight line until blocked
            dr = 0 if to_r == from_r else (1 if to_r > from_r else -1)
            dc = 0 if to_c == from_c else (1 if to_c > from_c else -1)
            
            # Must be straight line
            if dr != 0 and dc != 0:
                return False
            
            # Check path is clear
            r, c = from_r + dr, from_c + dc
            while r != to_r or c != to_c:
                if self.board[r, c] != Piece.EMPTY.value:
                    return False
                r, c = r + dr, c + dc
        else:
            # Other pieces can only move one square
            if abs(to_r - from_r) + abs(to_c - from_c) != 1:
                return False
        
        return True
    
    def _execute_move(self, from_r, from_c, to_r, to_c):
        """Execute a move and return reward"""
        attacking_piece = self.board[from_r, from_c]
        defending_piece = self.board[to_r, to_c]
        reward = 0
        
        if defending_piece == Piece.EMPTY.value:
            # Simple move
            self.board[to_r, to_c] = attacking_piece
            self.board[from_r, from_c] = Piece.EMPTY.value
            reward = 0.1  # Small reward for movement
        else:
            # Combat
            result = self._resolve_combat(attacking_piece, defending_piece)
            
            if result == "attacker_wins":
                self.board[to_r, to_c] = attacking_piece
                self.board[from_r, from_c] = Piece.EMPTY.value
                reward = self._get_piece_value(defending_piece)
            elif result == "defender_wins":
                self.board[from_r, from_c] = Piece.EMPTY.value
                reward = -self._get_piece_value(attacking_piece)
            else:  # Both die
                self.board[to_r, to_c] = Piece.EMPTY.value
                self.board[from_r, from_c] = Piece.EMPTY.value
                reward = 0
        
        return reward
    
    def _resolve_combat(self, attacker, defender):
        """Resolve combat between two pieces"""
        attacker_rank = self._get_piece_rank(attacker)
        defender_rank = self._get_piece_rank(defender)
        
        # Special cases
        if self._is_flag(defender):
            return "attacker_wins"
        
        if self._is_bomb(defender):
            if self._is_miner(attacker):
                return "attacker_wins"
            else:
                return "defender_wins"
        
        if self._is_spy(attacker) and self._is_marshal(defender):
            return "attacker_wins"
        
        # Normal combat
        if attacker_rank > defender_rank:
            return "attacker_wins"
        elif attacker_rank < defender_rank:
            return "defender_wins"
        else:
            return "both_die"
    
    def _check_game_over(self):
        """Check if game is over"""
        # Check if flag is captured
        red_flag_exists = np.any(self.board == Piece.FLAG_R.value)
        blue_flag_exists = np.any(self.board == Piece.FLAG_B.value)
        
        if not red_flag_exists:
            self.game_over = True
            self.winner = 1  # Blue wins
        elif not blue_flag_exists:
            self.game_over = True
            self.winner = 0  # Red wins
        
        # Check if player has no movable pieces
        if not self.game_over:
            red_can_move = False
            blue_can_move = False
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    piece = self.board[r, c]
                    if self._is_player_piece(piece, 0) and self._can_piece_move(piece):
                        if self._has_valid_moves(r, c, 0):
                            red_can_move = True
                    if self._is_player_piece(piece, 1) and self._can_piece_move(piece):
                        if self._has_valid_moves(r, c, 1):
                            blue_can_move = True
            
            if not red_can_move:
                self.game_over = True
                self.winner = 1
            elif not blue_can_move:
                self.game_over = True
                self.winner = 0
    
    def _has_valid_moves(self, r, c, player):
        """Check if piece at (r,c) has any valid moves"""
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self._is_valid_move(r, c, nr, nc, player):
                return True
        return False
    
    def _encode_action(self, from_r, from_c, to_r, to_c):
        """Encode move as single integer"""
        return from_r * 1000 + from_c * 100 + to_r * 10 + to_c
    
    def _decode_action(self, action):
        """Decode move from integer"""
        to_c = action % 10
        action //= 10
        to_r = action % 10
        action //= 10
        from_c = action % 10
        action //= 10
        from_r = action % 10
        return from_r, from_c, to_r, to_c
    
    def _is_player_piece(self, piece, player):
        """Check if piece belongs to player"""
        if player == 0:  # Red
            return 1 <= piece <= 12
        else:  # Blue
            return 13 <= piece <= 24
    
    def _can_piece_move(self, piece):
        """Check if piece can move"""
        return not (self._is_flag(piece) or self._is_bomb(piece))
    
    def _is_scout(self, piece):
        return piece == Piece.SCOUT_R.value or piece == Piece.SCOUT_B.value
    
    def _is_flag(self, piece):
        return piece == Piece.FLAG_R.value or piece == Piece.FLAG_B.value
    
    def _is_bomb(self, piece):
        return piece == Piece.BOMB_R.value or piece == Piece.BOMB_B.value
    
    def _is_miner(self, piece):
        return piece == Piece.MINER_R.value or piece == Piece.MINER_B.value
    
    def _is_spy(self, piece):
        return piece == Piece.SPY_R.value or piece == Piece.SPY_B.value
    
    def _is_marshal(self, piece):
        return piece == Piece.MARSHAL_R.value or piece == Piece.MARSHAL_B.value
    
    def _get_piece_rank(self, piece):
        """Get piece rank for combat"""
        rank_map = {
            # Red pieces
            Piece.SPY_R.value: 1, Piece.SCOUT_R.value: 2, Piece.MINER_R.value: 3,
            Piece.SERGEANT_R.value: 4, Piece.LIEUTENANT_R.value: 5, Piece.CAPTAIN_R.value: 6,
            Piece.MAJOR_R.value: 7, Piece.COLONEL_R.value: 8, Piece.GENERAL_R.value: 9,
            Piece.MARSHAL_R.value: 10,
            # Blue pieces
            Piece.SPY_B.value: 1, Piece.SCOUT_B.value: 2, Piece.MINER_B.value: 3,
            Piece.SERGEANT_B.value: 4, Piece.LIEUTENANT_B.value: 5, Piece.CAPTAIN_B.value: 6,
            Piece.MAJOR_B.value: 7, Piece.COLONEL_B.value: 8, Piece.GENERAL_B.value: 9,
            Piece.MARSHAL_B.value: 10,
        }
        return rank_map.get(piece, 0)
    
    def _get_piece_value(self, piece):
        """Get piece value for rewards"""
        value_map = {
            # Flags
            Piece.FLAG_R.value: 100, Piece.FLAG_B.value: 100,
            # High value pieces
            Piece.MARSHAL_R.value: 10, Piece.MARSHAL_B.value: 10,
            Piece.GENERAL_R.value: 9, Piece.GENERAL_B.value: 9,
            # Medium value pieces
            Piece.COLONEL_R.value: 8, Piece.COLONEL_B.value: 8,
            Piece.MAJOR_R.value: 7, Piece.MAJOR_B.value: 7,
            Piece.CAPTAIN_R.value: 6, Piece.CAPTAIN_B.value: 6,
            Piece.LIEUTENANT_R.value: 5, Piece.LIEUTENANT_B.value: 5,
            Piece.SERGEANT_R.value: 4, Piece.SERGEANT_B.value: 4,
            # Special pieces
            Piece.MINER_R.value: 5, Piece.MINER_B.value: 5,  # Valuable for bombs
            Piece.SPY_R.value: 3, Piece.SPY_B.value: 3,
            Piece.SCOUT_R.value: 2, Piece.SCOUT_B.value: 2,
            # Static pieces
            Piece.BOMB_R.value: 3, Piece.BOMB_B.value: 3,
        }
        return value_map.get(piece, 1)
    
    def render(self):
        """Render the current board state"""
        piece_symbols = {
            0: ' .', -1: '~~',  # Empty, Water
            # Red pieces
            1: 'RF', 2: 'Rs', 3: 'Rc', 4: 'Rm', 5: 'RG', 6: 'RL',
            7: 'RC', 8: 'RJ', 9: 'RO', 10: 'Rg', 11: 'RM', 12: 'RB',
            # Blue pieces  
            13: 'BF', 14: 'Bs', 15: 'Bc', 16: 'Bm', 17: 'BG', 18: 'BL',
            19: 'BC', 20: 'BJ', 21: 'BO', 22: 'Bg', 23: 'BM', 24: 'BB'
        }
        
        print(f"\nCurrent Player: {'Red' if self.current_player == 0 else 'Blue'}")
        print("   ", end="")
        for c in range(self.board_size):
            print(f"{c:2}", end=" ")
        print()
        
        for r in range(self.board_size):
            print(f"{r:2}:", end="")
            for c in range(self.board_size):
                symbol = piece_symbols.get(self.board[r, c], '??')
                print(f"{symbol}", end=" ")
            print()
        print()

    def get_action_space_size(self):
        """Get the size of action space"""
        return 10000  # from_r * 1000 + from_c * 100 + to_r * 10 + to_c

    def get_state_space_size(self):
        """Get the size of state space"""
        return self.board_size * self.board_size + 1  # Board + current player