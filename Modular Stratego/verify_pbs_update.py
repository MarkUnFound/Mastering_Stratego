
import torch
import numpy as np
from drqn_agent import RainbowAgent
from game_state import GameState

def verify_pbs_update():
    print("🧪 Verifying PBS Update Mechanism...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. Create Agents
    print("   Creating Agents...")
    agent1 = RainbowAgent(player_id=1, device=device, num_envs=1)
    agent2 = RainbowAgent(player_id=-1, device=device, num_envs=1)
    
    # 2. Setup Mock Game State
    print("   Setting up mock game state...")
    # Create a simple board with a piece at (6, 5) for Player 1
    board = np.zeros((10, 10), dtype=int)
    board[6, 5] = 2 # Scout (Player 1)
    
    # Initialize GameState with required arguments (using defaults where possible)
    game_state = GameState(
        board=torch.tensor(board, device=device),
        current_player=1,
        turn_count=0,
        game_over=False,
        winner=0,
        move_history=[],
        uncertainty_mask=None,
        revealed_pieces_p1={},
        revealed_pieces_p2={}
    )
    game_state.actual_board = torch.tensor(board, device=device) # PBS needs actual board to check values
    
    # 3. Simulate P1 Move: (6, 5) -> (5, 5) (Moving UP)
    # This is a 1-step move, so it could be any piece, but we know it's a Scout.
    action_p1 = ((6, 5), (5, 5))
    
    # 4. Check Agent 2's PBS before update
    print("   Checking Agent 2 PBS before update...")
    pbs = agent2.pbs
    # Position (6, 5) should be empty in history
    history_before = len(pbs.piece_action_history.get((6, 5), []))
    print(f"   History length for (6, 5) before: {history_before}")
    
    # 5. Call update_pbs_batch
    print("   Calling update_pbs_batch on Agent 2...")
    actions = [action_p1]
    game_states = [game_state]
    
    agent2.update_pbs_batch(actions, game_states, acting_player=1)
    
    # 6. Verify Update
    print("   Verifying update...")
    
    # Check history
    history_after = len(pbs.piece_action_history.get((6, 5), []))
    print(f"   History length for (6, 5) after: {history_after}")
    
    if history_after > history_before:
        print("   ✅ SUCCESS: PBS history updated!")
        
        # Check if belief tensor updated (optional, but good to know)
        # Since it moved 1 step, rule-based inference might not trigger "Scout" certainty,
        # but behavioral patterns might apply.
        
        return True
    else:
        print("   ❌ FAILURE: PBS history NOT updated!")
        return False

if __name__ == "__main__":
    try:
        success = verify_pbs_update()
        if success:
            print("\n✅ Verification Passed!")
            exit(0)
        else:
            print("\n❌ Verification Failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Verification Crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
