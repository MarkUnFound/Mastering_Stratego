import torch
import numpy as np
from dqn_agent import DQNAgent
from search_agent import SearchAgent
from environment import StrategoEnvironment
from piece import PieceType

def mock_get_state_value(state):
    """Mock heuristic: count pieces + random noise."""
    # Simple heuristic: more pieces = better
    board = state.board
    if isinstance(board, torch.Tensor):
        count = (board > 0).sum().item()
    else:
        count = np.count_nonzero(board > 0)
    return float(count) / 10.0

def verify_search():
    print("🔍 Verifying SearchAgent...")
    
    # 1. Setup Mock Agent
    dqn_agent = DQNAgent(player_id=1, device=torch.device("cpu"))
    # Mock the heuristic to avoid needing a trained model
    dqn_agent.get_state_value = mock_get_state_value
    
    # 2. Enable Search on DQNAgent
    dqn_agent.enable_search(depth=2, endgame_threshold=10)
    
    # 3. Create Mock Game State (Endgame)
    class MockState:
        def __init__(self):
            self.board = np.zeros((10, 10), dtype=int)
            self.current_player = 1
            self.turn_count = 100 # Late game
            
            # Place some pieces
            # Us (1): Scout at (5, 5)
            self.board[5, 5] = PieceType.SCOUT.value
            # Opponent (-1): Bomb at (6, 5)
            self.board[6, 5] = -PieceType.BOMB.value
            # Opponent (-1): Scout at (0, 0) (Movable, so they don't lose by no moves)
            self.board[0, 0] = -PieceType.SCOUT.value
            
    state = MockState()
    
    # 4. Test Search via DQNAgent.act
    print("  Testing Search via DQNAgent.act...")
    
    # Valid moves
    valid_moves = [
        ((5, 5), (5, 4)),
        ((5, 5), (5, 6)),
        ((5, 5), (4, 5)),
        ((5, 5), (6, 5)) # Attack Bomb -> Loss
    ]
    
    try:
        # Pass game_state to act (required for search)
        best_move = dqn_agent.act(state, valid_moves, game_state=state)
        print(f"  Best move found: {best_move}")
        
        if best_move == ((5, 5), (6, 5)):
            print("⚠️  Warning: Search chose to attack Bomb (suicide).")
        else:
            print("✅ Search avoided immediate suicide (Bomb).")
            
    except Exception as e:
        print(f"❌ Search failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_search()
