
import time
import torch
import numpy as np
import copy
from drqn_agent import DQNAgent
from game_state import GameState
from piece import PieceType
from history_aggregator import HistoryAggregator

def benchmark_history_update():
    print("--- Benchmarking History Aggregator Update ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize Agent with AAREN history
    agent = DQNAgent(player_id=1, device=device, num_envs=32)
    
    # Create mock batch
    batch_size = 32
    actions = []
    game_states = []
    
    # Create a dummy game state with tensors
    board = torch.zeros((10, 10), dtype=torch.int32, device=device)
    # Add some pieces
    board[0, 0] = 2 # Scout
    board[9, 9] = -2 # Enemy Scout
    
    dummy_state = GameState(
        board=board,
        current_player=1,
        turn_count=10,
        game_over=False,
        winner=None,
        move_history=[],
        uncertainty_mask=torch.zeros((10, 10), device=device),
        revealed_pieces_p1={},
        revealed_pieces_p2={}
    )
    
    # Add actual_board for history to see "hidden" pieces
    dummy_state.actual_board = board
    
    for i in range(batch_size):
        # Action: (0,0) -> (0,1)
        action = ((0, 0), (0, 1))
        actions.append(action)
        game_states.append(dummy_state)
        
    # Warmup
    agent.update_history_batch(actions, game_states, acting_player=-1)
    
    # Benchmark
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        agent.update_history_batch(actions, game_states, acting_player=-1)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    print(f"Batched Update (Batch Size {batch_size}): {avg_time*1000:.2f} ms per batch")
    print(f"Per Item: {(avg_time/batch_size)*1000:.2f} ms")

def benchmark_gamestate_clone():
    print("\n--- Benchmarking GameState Clone ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy game state with tensors
    board = torch.randn((10, 10), device=device)
    uncertainty = torch.randn((10, 10), device=device)
    
    state = GameState(
        board=board,
        current_player=1,
        turn_count=100,
        game_over=False,
        winner=None,
        move_history=[(1, 2, 3, 4)] * 50, # Some history
        uncertainty_mask=uncertainty,
        revealed_pieces_p1={(0,0): 1},
        revealed_pieces_p2={(9,9): -1}
    )
    
    # Warmup
    _ = state.clone()
    _ = copy.deepcopy(state)
    
    iterations = 1000
    
    # Benchmark Clone
    start_time = time.time()
    for _ in range(iterations):
        _ = state.clone()
    end_time = time.time()
    clone_time = (end_time - start_time) / iterations
    print(f"GameState.clone(): {clone_time*1000:.4f} ms")
    
    # Benchmark Deepcopy
    start_time = time.time()
    for _ in range(iterations):
        _ = copy.deepcopy(state)
    end_time = time.time()
    deepcopy_time = (end_time - start_time) / iterations
    print(f"copy.deepcopy(): {deepcopy_time*1000:.4f} ms")
    
    print(f"Speedup: {deepcopy_time / clone_time:.2f}x")

if __name__ == "__main__":
    benchmark_gamestate_clone()
    benchmark_history_update()
