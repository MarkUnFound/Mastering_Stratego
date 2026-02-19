import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

# Access the MARQ framework from the sibling directory
MODULAR_STRATEGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modular Stratego"))
if MODULAR_STRATEGO_PATH not in sys.path:
    sys.path.append(MODULAR_STRATEGO_PATH)

from drqn_agent import RainbowAgent
from history_aggregator import HistoryAggregator
from piece import PieceType, NUM_PIECE_TYPES
from board import LAKE_SQUARE

class ExpectamaxSearch:
    """
    Formal Expectamax search algorithm for Stratego.
    
    Structure:
    - Max Node: Bot's turn (select best expected move)
    - Chance Node: Target piece identity (sum over AAREN probabilities)
    - Min Node: Opponent's turn (assumed optimal for 1-ply depth)
    """
    
    def __init__(self, agent: RainbowAgent):
        self.agent = agent
        self.device = agent.device
        
    def evaluate_move(self, move: Tuple, gui_board, prior_q: float) -> float:
        """
        Evaluate a move using the formal Expectamax formula.
        """
        src, dst = move
        target_piece = gui_board.get(dst)
        
        # 1. If not an attack, use the DQN prior (or simple heuristic)
        if not target_piece:
            return prior_q
            
        # 2. If it is an attack, target identity is a CHANCE NODE
        # Get AAREN probabilities for the target square
        probabilities = self.agent.history.get_piece_predictions(dst)
        
        if probabilities is None:
            # No history for this piece - fall back to uniform/prior
            return prior_q
            
        # 3. Sum over all possible ranks (Chance Node calculation)
        expected_utility = 0.0
        attacker_rank = gui_board.get(src).rank
        
        for p_type, prob in probabilities.items():
            rank = p_type.value
            outcome_utility = self._calculate_outcome_utility(attacker_rank, rank)
            expected_utility += prob * outcome_utility
            
        # 4. Mix with prior Q-value for tactical context
        # Prior Q contains info about positional gain, flag proximity, etc.
        # We use Expectamax primarily as a "sanity/safety" filter for attacks.
        return 0.3 * prior_q + 0.7 * expected_utility

    def _calculate_outcome_utility(self, attacker_rank: int, defender_rank: int) -> float:
        """
        Calculate scalar utility for a specific combat outcome.
        Ranges from -1.0 (loss) to 1.0 (win).
        """
        # Stratego combat logic
        # 11 = Bomb, 12 = Spy, 1 = Marshal, ..., 10 = Flag
        
        # Tie
        if attacker_rank == defender_rank:
            return -0.1 # Slight penalty for trading pieces
            
        # Special cases
        if defender_rank == 11: # Bomb
            return 1.0 if attacker_rank == 8 else -1.0 # Miner wins, others lose
        if defender_rank == 12: # Spy
            return 0.8 # Any piece wins against Spy if Spy is the defender
        if attacker_rank == 12 and defender_rank == 1: # Spy attacks Marshal
            return 1.0
        if defender_rank == 10: # Flag
            return 10.0 # Huge win
            
        # Normal ranks (lower number is stronger)
        if attacker_rank < defender_rank:
            # Win - return utility based on defender's value
            return 0.5 + (defender_rank / 12.0) * 0.5
        else:
            # Loss - return negative utility based on attacker's value
            return -0.5 - ((12 - attacker_rank) / 12.0) * 0.5

class DQNBotLogic:
    """
    High-fidelity adapter bridging the Pygame GUI with the MARQ Rainbow DQN.
    Implements Formal Expectamax to refine move selection.
    """
    
    def __init__(self, model_path: str, player_id: int = 2, device: str = None):
        self.player_id = player_id
        self.opponent_id = 1 if player_id == 2 else 2
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the full RainbowAgent
        self.agent = RainbowAgent(
            player_id=player_id, 
            device=self.device, 
            use_pbs=True
        )
        
        # Load the checkpoint
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
            print(f"🚀 [MARQ Bot] Model loaded: {os.path.basename(model_path)}")
        else:
            print(f"⚠️ [MARQ Bot] Warning: {model_path} not found.")

        # Initialize formal Expectamax engine
        self.expectamax = ExpectamaxSearch(self.agent)
        
    def reset(self):
        self.agent.reset_history()
        
    def choose_move(self, gui_board, current_player: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # 1. Get all legal moves from the GUI
        legal_moves = self._get_gui_legal_moves(gui_board, current_player)
        if not legal_moves:
            return None
            
        # 2. Get state tensor
        state_tensor = self._get_state_representation(gui_board)
        
        # 3. Get departmental Q-values
        self.agent.q_network.eval()
        with torch.no_grad():
            log_probs = self.agent.q_network(state_tensor.unsqueeze(0))
            probs = log_probs.exp()
            support = self.agent.support if hasattr(self.agent, 'support') else torch.linspace(-10, 10, 51).to(self.device)
            expected_q = (probs * support).sum(dim=2).squeeze(0) # (400)
            
        # 4. Evaluate all legal moves using Expectamax
        move_scores = []
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move)
            if action_idx is not None:
                prior_q = expected_q[action_idx].item()
                # Refine with Expectamax Chance Nodes
                score = self.expectamax.evaluate_move(move, gui_board, prior_q)
                move_scores.append((move, score))
            else:
                move_scores.append((move, -float('inf')))
        
        # Select best move
        best_move, best_score = max(move_scores, key=lambda x: x[1])
        
        # Diagnostic logging for attacks
        if gui_board.get(best_move[1]):
            print(f"🗡️ [Expectamax] Attack selected! Score: {best_score:.3f}")
            probs = self.agent.history.get_piece_predictions(best_move[1])
            if probs:
                top_rank = max(probs.items(), key=lambda x: x[1])
                print(f"   Confidence in identity: {top_rank[0].name} ({top_rank[1]*100:.1f}%)")
        
        return best_move
    
    def update_from_opponent_move(self, gui_board, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """
        Update AAREN history aggregator after opponent move.
        """
        marq_grid = self._gui_board_to_marq_grid(gui_board)
        move = (from_pos, to_pos)
        self.agent.update_history_batch([move], [marq_grid], acting_player=self.opponent_id)

    def _get_state_representation(self, gui_board) -> torch.Tensor:
        marq_grid = self._gui_board_to_marq_grid(gui_board)
        return self.agent.get_state_representation(marq_grid, pbs_instance=self.agent.history)

    def _gui_board_to_marq_grid(self, gui_board) -> np.ndarray:
        grid = np.zeros((10, 10), dtype=np.int32)
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                if piece:
                    val = piece.rank
                    if piece.owner == 2: val = -val
                    grid[r, c] = val
                elif gui_board.is_lake(r, c):
                    grid[r, c] = LAKE_SQUARE
        return grid

    def _get_gui_legal_moves(self, gui_board, player_id: int) -> List[Tuple]:
        moves = []
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                if piece and piece.owner == player_id:
                    legal_dsts = gui_board.legal_moves_from((r, c))
                    for dst in legal_dsts:
                        moves.append(((r, c), dst))
        return moves
