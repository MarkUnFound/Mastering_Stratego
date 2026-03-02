
import sys
import os
import random

# Add parent directory to path
sys.path.append(os.getcwd())

from piece import PieceType, PIECE_NAMES
from heuristic_setup import HeuristicSetupAgent

# Lake Positions (Standard Stratego)
LAKES = {(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)}

def get_standard_army():
    pieces = []
    pieces.extend([PieceType.FLAG] * 1)
    pieces.extend([PieceType.MARSHAL] * 1)
    pieces.extend([PieceType.GENERAL] * 1)
    pieces.extend([PieceType.COLONEL] * 2)
    pieces.extend([PieceType.MAJOR] * 3)
    pieces.extend([PieceType.CAPTAIN] * 4)
    pieces.extend([PieceType.LIEUTENANT] * 4)
    pieces.extend([PieceType.SERGEANT] * 4)
    pieces.extend([PieceType.MINER] * 5)
    pieces.extend([PieceType.SCOUT] * 8)
    pieces.extend([PieceType.SPY] * 1)
    pieces.extend([PieceType.BOMB] * 6)
    return pieces

def print_full_board(p1_placement, p2_placement):
    # Initialize 10x10 grid with empty strings
    grid = [['..' for _ in range(10)] for _ in range(10)]
    
    # Mark Lakes
    for r, c in LAKES:
        grid[r][c] = "~~"
        
    # Place Player 1 (Bottom, Rows 6-9)
    # P1 pieces are positive, but names are same
    for piece, (r, c) in p1_placement:
        name = PIECE_NAMES[piece]
        # P1 is usually Blue/Red. We'll mark with + or simply uppercase
        grid[r][c] = f"{name:>2}"
        
    # Place Player 2 (Top, Rows 0-3)
    # P2 pieces are negative in engine, but here just placement
    for piece, (r, c) in p2_placement:
        name = PIECE_NAMES[piece]
        # Mark P2 with a prefix or lowercase to distinguish if needed
        # But Stratego pieces look same on board usually. 
        # Let's wrap P2 in braces or lowercase to ensure we know who is who.
        # Actually P2 is opponent. Let's maximize readability.
        # P1 = Standard
        # P2 = Lowercase
        grid[r][c] = f"{name.lower():>2}"

    print("\nFULL BOARD CONFIGURATION (P2=Top/Lower, P1=Bottom/Upper)")
    print("   0  1  2  3  4  5  6  7  8  9")
    print("  -----------------------------")
    
    for r in range(10):
        row_str = ' '.join(grid[r])
        print(f"{r}| {row_str}")

def main():
    print("INSPECTING HEURISTIC SETUP AGENT QUALITY...")
    
    agent1 = HeuristicSetupAgent(player_id=1)
    agent2 = HeuristicSetupAgent(player_id=-1)
    
    all_pieces = get_standard_army()
    valid_p1 = [(r, c) for r in range(6, 10) for c in range(10)]
    valid_p2 = [(r, c) for r in range(0, 4) for c in range(10)]
    
    print("\n" + "="*50)
    print("GENERATING 3 FULL GAME SETUPS")
    print("="*50)
    
    for i in range(3):
        print(f"\n--- Game Sample {i+1} ---")
        try:
            # Generate both sides
            p1_setup = agent1.place_pieces(all_pieces.copy(), valid_p1.copy())
            p2_setup = agent2.place_pieces(all_pieces.copy(), valid_p2.copy())
            
            print_full_board(p1_setup, p2_setup)
            
            # Brief check on Flags
            p1_flag = next(pos for p, pos in p1_setup if p == PieceType.FLAG)
            p2_flag = next(pos for p, pos in p2_setup if p == PieceType.FLAG)
            print(f"P1 Flag: {p1_flag} | P2 Flag: {p2_flag}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
