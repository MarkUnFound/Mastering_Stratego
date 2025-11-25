import sys
import os
import torch
import numpy as np

# Add Modular Stratego to path
current_dir = os.path.dirname(os.path.abspath(__file__))
modular_stratego_path = os.path.join(current_dir, "..", "Modular Stratego")
sys.path.append(modular_stratego_path)

from hybrid_agent import HybridAgent
from game_state import GameState
from piece import PieceType

class KlussBot:
    def __init__(self, player_id=2):
        self.player_id = player_id
        # Map GUI player (2) to Agent player (-1) usually
        # In GUI: P1=1, P2=2.
        # In Agent: P1=1, P2=-1.
        self.agent_player_id = 1 if player_id == 1 else -1
        
        self.agent = HybridAgent(player_id=self.agent_player_id, device='cpu')
        # Load model if exists
        model_path = os.path.join(modular_stratego_path, "models", "best_model.pt")
        if os.path.exists(model_path):
            try:
                self.agent.evaluator.network.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded KLUSS model from {model_path}")
            except:
                print("Failed to load model, using random weights.")
        else:
            print("No model found, using random weights.")

        self.last_q_values = [] # Store for visualization

    def reset(self):
        self.last_q_values = []

    def choose_move(self, gui_board, owner):
        # Convert GUI board to GameState
        game_state = self.convert_board(gui_board, owner)
        
        # Get valid moves from GUI board to filter agent's output
        # (Agent generates its own valid moves based on its internal board representation)
        # We need to ensure they match.
        
        # Actually, HybridAgent.act() takes (state, valid_moves).
        # We should generate valid moves using the Agent's environment logic or convert GUI moves.
        # Let's convert GUI moves to Agent format ((r1,c1), (r2,c2)).
        
        gui_moves = []
        for src in gui_board.owner_positions(owner):
            for dst in gui_board.legal_moves_from(src):
                gui_moves.append((src, dst))
                
        if not gui_moves:
            return None
            
        # Get Q-values for visualization
        self.last_q_values = self.agent.get_top_moves(game_state, n=3)
        
        # Select Action
        action = self.agent.act(game_state, gui_moves)
        return action

    def convert_board(self, gui_board, current_player):
        # Create 10x10 tensor
        board_tensor = torch.zeros((10, 10), dtype=torch.float32)
        
        for r in range(10):
            for c in range(10):
                p = gui_board.grid[r][c]
                if p:
                    # Map Rank
                    rank_val = self.map_rank(p.rank)
                    
                    # Map Owner to Sign
                    # Agent expects: P1 (1) -> Positive, P2 (-1) -> Negative
                    # GUI: P1=1, P2=2
                    sign = 1 if p.owner == 1 else -1
                    
                    board_tensor[r, c] = rank_val * sign
                    
        # Create GameState
        # We need to construct other fields like revealed_pieces
        # GUI has p.revealed.
        revealed_p1 = {}
        revealed_p2 = {}
        
        for r in range(10):
            for c in range(10):
                p = gui_board.grid[r][c]
                if p and p.revealed:
                    rank_val = self.map_rank(p.rank)
                    if p.owner == 1:
                        revealed_p2[(r,c)] = rank_val # P1 piece revealed to P2
                    else:
                        revealed_p1[(r,c)] = rank_val # P2 piece revealed to P1
                        
        # Turn count (approximate or track)
        turn_count = 0 
        
        state = GameState(
            board=board_tensor,
            current_player=1 if current_player == 1 else -1,
            turn_count=turn_count,
            game_over=False,
            winner=None,
            move_history=[],
            uncertainty_mask=torch.zeros((10, 10)), # Placeholder
            revealed_pieces_p1=revealed_p1,
            revealed_pieces_p2=revealed_p2
        )
        return state

    def map_rank(self, gui_rank):
        # GUI -> Agent (PieceType enum value)
        mapping = {
            10: 11, # Marshal
            9: 10,  # General
            8: 9,   # Colonel
            7: 8,   # Major
            6: 7,   # Captain
            5: 6,   # Lieutenant
            4: 5,   # Sergeant
            3: 4,   # Miner
            2: 3,   # Scout
            1: 2,   # Spy
            0: 12,  # Bomb
            -1: 1   # Flag
        }
        return mapping.get(gui_rank, 0)
