import os
import sys
import time
import pygame

# Mock pygame to avoid window creation if possible, or just let it open
# We will just import the class
from live_visualizer_enhanced import LiveVisualizer

def verify_history():
    print("Initializing Visualizer...")
    vis = LiveVisualizer()
    
    print("Stepping game 10 times...")
    for i in range(10):
        vis.step_game()
        # Wait a bit for animation if needed, but step_game sets animation
        # We need to call _execute_step to finish the move if animation is set
        if vis.animating_move:
            # Force finish animation
            vis._execute_step()
            
        print(f"Step {i+1} complete.")
        
    print("\nMove History Log:")
    for entry in vis.move_history:
        print(entry)
        
    if len(vis.move_history) == 10:
        print("\nSUCCESS: History has 10 entries.")
    else:
        print(f"\nFAILURE: History has {len(vis.move_history)} entries (expected 10).")
        
    # Check content of last entry
    last = vis.move_history[-1]
    if 'q_val' in last and 'max_q' in last and 'player' in last:
        print("SUCCESS: Log entry contains required fields.")
    else:
        print("FAILURE: Log entry missing fields.")

    pygame.quit()

if __name__ == "__main__":
    verify_history()
