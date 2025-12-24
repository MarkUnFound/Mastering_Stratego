
import torch
import numpy as np
from environment import StrategoEnvironment
from opponents import RandomAgent, RandomSetupAgent
from heuristic_setup import HeuristicSetupAgent
from piece import PieceType
import tqdm

def debug_random_wins(num_episodes=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    env = StrategoEnvironment(device, full_observability=False)
    
    # Agent 1: Random (representing an untrained agent that acts randomly)
    # We use RandomAgent for Agent 1 too to simulate "bad/initial" play,
    # OR we can load the actual model if we had it. 
    # But to test "Is Random beating Heuristic?", let's pitch Random vs Heuristic.
    
    # The user says "Random Agent wins... against a well defended flag".
    # So Agent 2 (Random) vs Agent 1 (Heuristic Setup).
    # Since we don't have the trained weights loaded easily, let's assume Agent 1 is behaving poorly (randomly).
    # A "dumb" Agent 1 vs Random Agent 2.
    
    agent1 = RandomAgent() 
    setup1 = HeuristicSetupAgent(player_id=1, device=device)
    
    agent2 = RandomAgent()
    setup2 = RandomSetupAgent(player_id=-1) # Random Setup for Random Agent
    
    wins = {1: 0, -1: 0, 0: 0}
    reasons = {1: [], -1: []}
        
    for ep in tqdm.tqdm(range(num_episodes)):
        # Setup
        p1_pieces = env.get_all_pieces()
        p1_pos = env.get_valid_placement_positions(1)
        p1_place = setup1.place_pieces(p1_pieces, p1_pos)
        
        p2_pieces = env.get_all_pieces()
        p2_pos = env.get_valid_placement_positions(-1)
        p2_place = setup2.place_pieces(p2_pieces, p2_pos)
        
        state = env.reset(p1_place, p2_place)
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            if env.current_player == 1:
                action = agent1.act(state.board, valid_moves)
            else:
                action = agent2.act(state.board, valid_moves)
                
            state, reward, done, info = env.step(action)
            
            if done:
                winner = info['winner']
                wins[winner] += 1
                
                if winner != 0:
                    # Determine reason
                    loser = -winner
                    
                    # Check if Loser's Flag is missing
                    flag_exists = False
                    board_flat = env.board.actual_board.flatten()
                    for val in board_flat:
                        if abs(val.item()) == PieceType.FLAG.value:
                            # Check owner
                            if (val.item() > 0 and loser == 1) or (val.item() < 0 and loser == -1):
                                flag_exists = True
                                break
                    
                    reason = "Flag Capture" if not flag_exists else "No Valid Moves / Annihilation"
                    reasons[winner].append(reason)
    
    print("\nResults:")
    print(f"Agent 1 (Heuristic Setup) Wins: {wins[1]}")
    print(f"Agent 2 (Random Agent) Wins: {wins[-1]}")
    print(f"Draws: {wins[0]}")
    
    print("\nWin Reasons for Random Agent (Agent 2):")
    capture = reasons[-1].count("Flag Capture")
    annihilation = reasons[-1].count("No Valid Moves / Annihilation")
    print(f"  Flag Capture: {capture}")
    print(f"  Annihilation/No Moves: {annihilation}")

if __name__ == "__main__":
    debug_random_wins(20)
