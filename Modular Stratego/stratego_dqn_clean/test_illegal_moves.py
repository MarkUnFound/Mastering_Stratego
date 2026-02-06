
import torch
import unittest
import sys
import os

# Add parent directory to path to find 'board.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import StrategoEnvironment, PieceType, LAKE_SQUARE

class TestMoveValidation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        # Enable safe guards for testing
        self.env = StrategoEnvironment(
            device=self.device, 
            strict_validation=False, 
            safe_guards=True
        )
        self.env.reset()
        
    def test_move_into_lake(self):
        """Test that moving into a lake raises ValueError."""
        # Find a piece next to a lake (Lakes are at rows 4,5, cols 2,3 & 6,7)
        # Lake at (4,2). Try to move from (3,2) to (4,2)
        
        # Manually place a piece at (3,2) for Player 1
        # P1 pieces are positive.
        self.env.board.actual_board[3, 2] = PieceType.SERGEANT.value 
        self.env.current_player = 1
        
        # Valid move attempt: (3,2) -> (4,2) which is LAKE
        action = ((3, 2), (4, 2))
        
        print("\n[TEST] Testing Move into Lake...")
        try:
            self.env.step(action)
            self.fail("Environment should have raised ValueError for moving into lake")
        except ValueError as e:
            print(f"  Caught expected error: {e}")
            self.assertIn("LAKE", str(e))

    def test_attack_friendly(self):
        """Test that attacking friendly piece raises ValueError."""
        # Place P1 pieces at (0,0) and (0,1)
        self.env.board.actual_board[0, 0] = PieceType.SERGEANT.value
        self.env.board.actual_board[0, 1] = PieceType.SCOUT.value
        self.env.current_player = 1
        
        action = ((0, 0), (0, 1))
        
        print("\n[TEST] Testing Friendly Fire...")
        try:
            self.env.step(action)
            self.fail("Environment should have raised ValueError for friendly fire")
        except ValueError as e:
            print(f"  Caught expected error: {e}")
            self.assertIn("friendly", str(e))
            
    def test_move_opponent_piece(self):
        """Test that moving opponent's piece raises ValueError."""
        # Place P2 piece at (9,9) (Negative value)
        self.env.board.actual_board[9, 9] = -PieceType.SCOUT.value
        self.env.current_player = 1 # It's P1's turn
        
        action = ((9, 9), (9, 8))
        
        print("\n[TEST] Testing Moving Opponent Piece...")
        try:
            self.env.step(action)
            self.fail("Environment should have raised ValueError for moving opponent piece")
        except ValueError as e:
            print(f"  Caught expected error: {e}")
            self.assertIn("non-owned", str(e))

    def test_teleport(self):
        """Test that non-scout moving > 1 step raises ValueError."""
        self.env.board.actual_board[0, 0] = PieceType.SERGEANT.value # Not a scout
        self.env.current_player = 1
        
        # Try to move 2 squares
        action = ((0, 0), (0, 2))
        
        print("\n[TEST] Testing Teleportation...")
        try:
            self.env.step(action)
            self.fail("Environment should have raised ValueError for teleportation")
        except ValueError as e:
            print(f"  Caught expected error: {e}")
            self.assertIn("distance", str(e))

if __name__ == '__main__':
    unittest.main()
