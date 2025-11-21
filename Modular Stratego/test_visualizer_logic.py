import unittest
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from live_visualizer import LiveVisualizer
from environment import StrategoEnvironment
from piece import PieceType
from board import LAKE_SQUARE, BOARD_SIZE

class TestVisualizerLogic(unittest.TestCase):
    def setUp(self):
        # Mock pygame to avoid display issues
        import pygame
        pygame.init = lambda: None
        pygame.display.set_caption = lambda x: None
        pygame.display.set_mode = lambda x: None
        pygame.time.Clock = lambda: None
        pygame.font.SysFont = lambda *args, **kwargs: None
        
        self.vis = LiveVisualizer()
        
    def test_piece_mapping(self):
        """Test that piece values map to correct text labels (Bug #1)"""
        print("\nTesting Piece Mapping...")
        
        # Check Flag (1)
        flag_text = self.vis.get_piece_text(1)
        print(f"Value 1 (Flag) -> '{flag_text}'")
        self.assertEqual(flag_text, "F", "Flag (1) should map to 'F'")
        
        # Check Bomb (12)
        bomb_text = self.vis.get_piece_text(12)
        print(f"Value 12 (Bomb) -> '{bomb_text}'")
        self.assertEqual(bomb_text, "B", "Bomb (12) should map to 'B'")
        
        # Check Scout (3)
        scout_text = self.vis.get_piece_text(3)
        print(f"Value 3 (Scout) -> '{scout_text}'")
        self.assertEqual(scout_text, "2", "Scout (3) should map to '2'")
        
        # Check Marshal (11)
        marshal_text = self.vis.get_piece_text(11)
        print(f"Value 11 (Marshal) -> '{marshal_text}'")
        self.assertEqual(marshal_text, "M", "Marshal (11) should map to 'M'")
        
    def test_lake_movement(self):
        """Test that pieces cannot move into lakes (Bug #2)"""
        print("\nTesting Lake Movement...")
        env = self.vis.env
        
        # Place a piece next to a lake
        # Lake at (4, 2), place piece at (3, 2)
        env.board.actual_board[3, 2] = PieceType.SCOUT.value # Scout
        env.current_player = 1
        
        moves = env.get_valid_moves()
        
        # Check if any move goes to (4, 2)
        lake_move = False
        for start, end in moves:
            if start == (3, 2) and end == (4, 2):
                lake_move = True
                break
                
        print(f"Can move from (3,2) to lake (4,2)? {lake_move}")
        self.assertFalse(lake_move, "Should not be able to move into lake")
        
    def test_multi_tile_movement(self):
        """Test that only Scouts can move multiple tiles (Bug #3)"""
        print("\nTesting Multi-Tile Movement...")
        env = self.vis.env
        env.board.actual_board.fill_(0) # Clear board
        
        # 1. Test Scout (Value 3)
        env.board.actual_board[1, 1] = PieceType.SCOUT.value
        env.current_player = 1
        
        moves = env.get_valid_moves()
        scout_moves = [m for m in moves if m[0] == (1, 1)]
        
        # Scout should be able to move to (1, 3) (2 tiles away)
        can_move_2_tiles = any(m[1] == (1, 3) for m in scout_moves)
        print(f"Scout at (1,1) can move to (1,3)? {can_move_2_tiles}")
        self.assertTrue(can_move_2_tiles, "Scout should be able to move multiple tiles")
        
        # 2. Test Major (Value 8) - Non-Scout
        env.board.actual_board.fill_(0)
        env.board.actual_board[1, 1] = PieceType.MAJOR.value
        
        moves = env.get_valid_moves()
        major_moves = [m for m in moves if m[0] == (1, 1)]
        
        # Major should NOT be able to move to (1, 3)
        can_move_2_tiles = any(m[1] == (1, 3) for m in major_moves)
        print(f"Major at (1,1) can move to (1,3)? {can_move_2_tiles}")
        self.assertFalse(can_move_2_tiles, "Non-Scout should NOT be able to move multiple tiles")
        
        # Major should be able to move to (1, 2) (1 tile away)
        can_move_1_tile = any(m[1] == (1, 2) for m in major_moves)
        print(f"Major at (1,1) can move to (1,2)? {can_move_1_tile}")
        self.assertTrue(can_move_1_tile, "Major should be able to move 1 tile")

if __name__ == '__main__':
    unittest.main()
