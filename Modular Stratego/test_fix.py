
import sys
import os
import torch
# Add the repository root to sys.path
sys.path.append(r'd:\repo\Research\Modular Stratego')

from live_visualizer import LiveVisualizer
from piece import PieceType

def test_piece_mapping():
    print("Testing LiveVisualizer piece mapping...")
    try:
        # We don't want to init pygame window, so we mock it or just instantiate the class if possible
        # But LiveVisualizer __init__ calls pygame.init() and set_mode. 
        # We can just import the class and test the method if it was static, but it's an instance method.
        # However, we can subclass or mock.
        # Or simpler: just check the code logic by importing the function if it was standalone.
        # Since it's a method, let's try to instantiate it. It might fail due to no display.
        # Alternatively, we can just verify the logic by inspecting the file, but running is better.
        # Let's try to mock pygame.
        import unittest.mock as mock
        import pygame
        
        with mock.patch('pygame.display.set_mode'), mock.patch('pygame.init'):
            vis = LiveVisualizer()
            
            # Test Flag (1)
            txt = vis.get_piece_text(1)
            print(f"Piece 1 (Flag) text: {txt}")
            assert txt == "F", f"Expected 'F' for Flag, got '{txt}'"
            
            # Test Bomb (11)
            txt = vis.get_piece_text(11)
            print(f"Piece 11 (Bomb) text: {txt}")
            assert txt == "B", f"Expected 'B' for Bomb, got '{txt}'"
            
            # Test Scout (2)
            txt = vis.get_piece_text(2)
            print(f"Piece 2 (Scout) text: {txt}")
            assert txt == "2", f"Expected '2' for Scout, got '{txt}'"
            
            print("Piece mapping test PASSED!")
            
    except Exception as e:
        print(f"Test FAILED: {e}")

if __name__ == "__main__":
    test_piece_mapping()
