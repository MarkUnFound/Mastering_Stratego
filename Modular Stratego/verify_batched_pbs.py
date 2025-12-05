
import torch
import numpy as np
from drqn_agent import RainbowAgent
from game_state import GameState

def verify_batched_pbs():
    print("🧪 Verifying Batched PBS Update...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    NUM_ENVS = 4
    print(f"   Creating Agent with {NUM_ENVS} environments...")
    agent = RainbowAgent(player_id=1, device=device, num_envs=NUM_ENVS)
    
    # Setup mock game states and actions for all envs
    actions = []
    game_states = []
    
    # We will use different moves for each env to ensure batching works with diverse inputs
    # Env 0: (6, 5) -> (5, 5)
    # Env 1: (6, 6) -> (5, 6)
    # Env 2: None (no move)
    # Env 3: (6, 7) -> (5, 7)
    
    moves = [
        ((6, 5), (5, 5)),
        ((6, 6), (5, 6)),
        None,
        ((6, 7), (5, 7))
    ]
    
    print("   Setting up mock environments...")
    for i in range(NUM_ENVS):
        board = np.zeros((10, 10), dtype=int)
        if moves[i]:
            start_pos = moves[i][0]
            # Agent is Player 1.
            # PBS tracks OPPONENT pieces.
            # Opponent is Player -1.
            # So pieces should be negative (e.g. -2 for Scout).
            board[start_pos] = -2 # Scout (Player -1)
            
        gs = GameState(
            board=torch.tensor(board, device=device),
            current_player=-1, # Opponent is moving
            turn_count=0,
            game_over=False,
            winner=0,
            move_history=[],
            uncertainty_mask=None,
            revealed_pieces_p1={},
            revealed_pieces_p2={}
        )
        gs.actual_board = torch.tensor(board, device=device)
        
        game_states.append(gs)
        actions.append(moves[i])
        
    # Check history before update
    print("   Checking history before update...")
    history_counts_before = []
    for i in range(NUM_ENVS):
        if moves[i]:
            start_pos = moves[i][0]
            count = len(agent.pbs_instances[i].piece_action_history.get(start_pos, []))
            history_counts_before.append(count)
        else:
            history_counts_before.append(0)
            
    # Call update_pbs_batch
    # acting_player should be the opponent (-1)
    print("   Calling update_pbs_batch...")
    agent.update_pbs_batch(actions, game_states, acting_player=-1)
    
    # Verify updates
    print("   Verifying updates...")
    success = True
    
    for i in range(NUM_ENVS):
        if moves[i]:
            start_pos = moves[i][0]
            count_after = len(agent.pbs_instances[i].piece_action_history.get(start_pos, []))
            print(f"   Env {i}: History {history_counts_before[i]} -> {count_after}")
            
            if count_after <= history_counts_before[i]:
                print(f"   ❌ Env {i} failed to update!")
                success = False
        else:
            print(f"   Env {i}: No move (skipped)")
            
    if success:
        print("   ✅ SUCCESS: All active environments updated correctly!")
        return True
    else:
        print("   ❌ FAILURE: Some environments failed to update.")
        return False

if __name__ == "__main__":
    try:
        if verify_batched_pbs():
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
