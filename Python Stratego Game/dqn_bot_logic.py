import os
import sys
import time
import copy
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

class TimeoutException(Exception):
    pass

class ExpectamaxSearch:
    """
    Formal Expectamax search algorithm for Stratego.
    
    Structure:
    - Max Node: Bot's turn (select best expected move)
    - Chance Node: Target piece identity (sum over AAREN probabilities)
    - Min Node: Opponent's turn (minimize Bot's utility)
    """
    
    def __init__(self, agent: RainbowAgent, max_depth: int = 4, max_time: float = 4.0):
        self.agent = agent
        self.device = agent.device
        self.max_depth = max_depth
        self.max_time = max_time
        
    def _check_timeout(self, start_time: float):
        if time.time() - start_time > self.max_time:
            raise TimeoutException()

    def search(self, gui_board, current_player: int, DQN_adapter) -> Tuple[Optional[Tuple], float]:
        """
        Initiates the recursive Expectamax search.
        Returns the best move and its expected utility.
        """
        start_time = time.time()
        
        legal_moves = DQN_adapter._get_gui_legal_moves(gui_board, current_player)
        if not legal_moves:
            return None, -float('inf')
            
        # Sort initial moves by prior Q to prioritize promising branches
        move_scores = []
        expected_q = self._get_expected_q(gui_board, DQN_adapter)
        
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move)
            score = expected_q[action_idx].item() if action_idx is not None else -float('inf')
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [m[0] for m in move_scores]

        best_move = None
        best_score = -float('inf')
        
        try:
            for move in sorted_moves:
                action_idx = self.agent._move_to_action_index(move)
                prior_q = expected_q[action_idx].item() if action_idx is not None else 0.0
                
                score = self._expectamax_recursive(
                    gui_board, move, current_player, 
                    depth=1, start_time=start_time, 
                    prior_q=prior_q, DQN_adapter=DQN_adapter
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
        except TimeoutException:
            print(f" [Expectamax] Timeout hit at {time.time() - start_time:.2f}s, bounded at depth limits.")
            
        if best_move is None and sorted_moves:
            best_move = sorted_moves[0]
            
        return best_move, best_score

    def _expectamax_recursive(self, board, move: Tuple, player: int, depth: int, 
                              start_time: float, prior_q: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        src, dst = move
        target_piece = board.get(dst)
        
        # 1. Handle non-attacks (simple move)
        if not target_piece:
            sim_board = self._simulate_move_simple(board, move)
            return self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
        # 2. Handle attacks (Chance Node)
        probabilities = self.agent.history.get_piece_predictions(dst)
        
        if not probabilities:
            sim_board = self._simulate_move_simple(board, move)
            return self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
        expected_utility = 0.0
        attacker_rank = board.get(src).rank
        
        # Sort probabilities to explore top 3 highest likelihoods (pruning chance branching factor)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_n = min(3, len(sorted_probs))
        
        subset_prob_sum = sum(p for pt, p in sorted_probs[:top_n])
        if subset_prob_sum == 0: subset_prob_sum = 1.0
        
        for p_type, prob in sorted_probs[:top_n]:
            norm_prob = prob / subset_prob_sum
            rank = p_type.value
            
            outcome, sim_board = self._simulate_combat(board, move, attacker_rank, rank)
            combat_utility = self._calculate_outcome_utility(attacker_rank, rank)
            
            branch_utility = self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
            # Combine immediate tactical outcome with deeper state utility
            total_branch_utility = 0.5 * combat_utility + 0.5 * branch_utility
            expected_utility += norm_prob * total_branch_utility
            
        return 0.3 * prior_q + 0.7 * expected_utility

    def _min_node(self, board, player: int, depth: int, start_time: float, prior_q: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        if depth >= self.max_depth:
            # Enemy turn evaluated from BOT's perspective (hence 3-player for Bot if player is Enemy)
            return DQN_adapter.get_state_evaluation(board, 3 - player)
            
        legal_moves = DQN_adapter._get_gui_legal_moves(board, player)
        if not legal_moves:
            return 10.0 # Huge win for bot
            
        # Opponent moves (limit branching factor to top 2 for speed)
        expected_q = self._get_expected_q(board, DQN_adapter)
        move_scores = []
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move)
            score = expected_q[action_idx].item() if action_idx is not None else -float('inf')
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        min_utility = float('inf')
        for move_tuple in move_scores[:2]: 
            move, opp_prior_q = move_tuple
            sim_board = self._simulate_move_simple(board, move)
            
            utility = self._max_node(sim_board, 3 - player, depth + 1, start_time, DQN_adapter)
            if utility < min_utility:
                min_utility = utility
                
        return min_utility if min_utility != float('inf') else prior_q

    def _max_node(self, board, player: int, depth: int, start_time: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        if depth >= self.max_depth:
            return DQN_adapter.get_state_evaluation(board, player)
            
        legal_moves = DQN_adapter._get_gui_legal_moves(board, player)
        if not legal_moves:
            return -10.0 # Loss
            
        expected_q = self._get_expected_q(board, DQN_adapter)
        move_scores = []
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move)
            score = expected_q[action_idx].item() if action_idx is not None else -float('inf')
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        max_utility = -float('inf')
        for move_tuple in move_scores[:2]:
            move, prior_q = move_tuple
            utility = self._expectamax_recursive(board, move, player, depth + 1, start_time, prior_q, DQN_adapter)
            if utility > max_utility:
                max_utility = utility
                
        return max_utility

    def _get_expected_q(self, board, DQN_adapter):
        state_tensor = DQN_adapter._get_state_representation(board)
        self.agent.q_network.eval()
        with torch.no_grad():
            log_probs = self.agent.q_network(state_tensor.unsqueeze(0))
            probs = log_probs.exp()
            support = self.agent.support if hasattr(self.agent, 'support') else torch.linspace(-10, 10, 51).to(self.device)
            expected_q = (probs * support).sum(dim=2).squeeze(0)
        return expected_q

    def _simulate_move_simple(self, board, move: Tuple):
        sim_board = copy.deepcopy(board)
        src, dst = move
        piece = sim_board.get(src)
        sim_board.set(dst, piece)
        sim_board.set(src, None)
        return sim_board
        
    def _simulate_combat(self, board, move: Tuple, attacker_rank: int, defender_rank: int):
        sim_board = copy.deepcopy(board)
        src, dst = move
        attacker = sim_board.get(src)
        
        outcome = "tie"
        if attacker_rank == defender_rank:
            sim_board.set(src, None)
            sim_board.set(dst, None)
            outcome = "tie"
        elif defender_rank == 11: # Bomb
            if attacker_rank == 8: # Miner wins
                sim_board.set(dst, attacker)
                sim_board.set(src, None)
                outcome = "attacker"
            else:
                sim_board.set(src, None)
                outcome = "defender"
        elif defender_rank == 12: # Spy
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif attacker_rank == 12 and defender_rank == 1: # Spy vs Marshal
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif defender_rank == 10: # Flag
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif attacker_rank < defender_rank:
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        else:
            sim_board.set(src, None)
            outcome = "defender"
            
        return outcome, sim_board

    def _calculate_outcome_utility(self, attacker_rank: int, defender_rank: int) -> float:
        if attacker_rank == defender_rank: return -0.1
        if defender_rank == 11: return 1.0 if attacker_rank == 8 else -1.0
        if defender_rank == 12: return 0.8
        if attacker_rank == 12 and defender_rank == 1: return 1.0
        if defender_rank == 10: return 10.0
        if attacker_rank < defender_rank: return 0.5 + (defender_rank / 12.0) * 0.5
        else: return -0.5 - ((12 - attacker_rank) / 12.0) * 0.5

class DQNBotLogic:
    """
    High-fidelity adapter bridging the Pygame GUI with the MARQ Rainbow DQN.
    Implements Formal Expectamax to refine move selection.
    """
    
    def __init__(self, model_path: str, player_id: int = 2, device: str = None):
        self.player_id = player_id
        self.opponent_id = 1 if player_id == 2 else 2
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the full RainbowAgent
        self.agent = RainbowAgent(
            player_id=player_id, 
            device=self.device, 
            use_pbs=True
        )
        
        # Load the checkpoint (.pt prioritized)
        possible_pt = model_path.replace('.pth', '.pt') if model_path else None
        
        load_success = False
        for path in [possible_pt, model_path]:
            if path and os.path.exists(path):
                self.agent.load_model(path)
                print(f" [MARQ Bot] Model loaded: {os.path.basename(path)}")
                load_success = True
                break
        
        if not load_success:
            print(f" [MARQ Bot] Warning: No valid model found at {model_path} or its .pt variant.")

        # Initialize formal Expectamax engine (4 ply depth bounded by 4 seconds)
        self.expectamax = ExpectamaxSearch(self.agent, max_depth=4, max_time=4.0)
        
    def reset(self):
        self.agent.reset_history()
        
    def choose_move(self, gui_board, current_player: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # Delegate directly to the recursive expectamax search bounded by 4 seconds
        best_move, best_score = self.expectamax.search(gui_board, current_player, self)
        
        if best_move:
            # Diagnostic logging for attacks
            if gui_board.get(best_move[1]):
                print(f" [Expectamax] Attack selected! Score: {best_score:.3f}")
                probs = self.agent.history.get_piece_predictions(best_move[1])
                if probs:
                    top_rank = max(probs.items(), key=lambda x: x[1])
                    print(f"   Confidence in identity: {top_rank[0].name} ({top_rank[1]*100:.1f}%)")
                    
        return best_move
    
    def get_state_evaluation(self, gui_board, player_id: int) -> float:
        """
        Calculates board advantage using the Dueling Value Head.
        Positive = Advantageous for player_id.
        """
        self.agent.q_network.eval()
        with torch.no_grad():
            state = self._get_state_representation(gui_board).unsqueeze(0).to(self.device)
            # Use the new evaluate_state_value method
            v_probs = self.agent.q_network.evaluate_state_value(state)
            
            # Expected value of the support
            state_value = torch.sum(v_probs * self.agent.support, dim=1).item()
            
            return state_value

    def get_multi_recommendations(self, gui_board, player_id: int, top_n: int = 3) -> List[Dict]:
        """
        Returns multiple move suggestions with chess.com-style classifications.
        """
        # Temporarily switch agent ID
        original_id = self.agent.player_id
        self.agent.player_id = player_id
        
        recommendations = []
        try:
            self.agent.q_network.eval()
            with torch.no_grad():
                state = self._get_state_representation(gui_board).unsqueeze(0).to(self.device)
                log_probs = self.agent.q_network(state)
                expected_q = torch.sum(torch.exp(log_probs) * self.agent.support, dim=2).squeeze(0)
            
            legal_moves = self._get_gui_legal_moves(gui_board, player_id)
            move_scores = []
            
            for move in legal_moves:
                action_idx = self.agent._move_to_action_index(move) # Use agent's internal method
                if action_idx is not None:
                    score = expected_q[action_idx].item()
                    move_scores.append((move, score))
            
            # Sort by score descending
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            if not move_scores:
                return []
                
            best_score = move_scores[0][1]
            
            for i, (move, score) in enumerate(move_scores[:top_n]):
                gap = best_score - score
                
                # Classify based on gap (Chess.com style)
                if i == 0:
                    label = "Best"
                    color = (52, 211, 153) # Emerald
                elif gap < 0.1:
                    label = "Excellent"
                    color = (96, 165, 250) # Blue
                elif gap < 0.3:
                    label = "Good"
                    color = (134, 239, 172) # Light Green
                else:
                    label = "Inaccuracy"
                    color = (251, 191, 36) # Orange
                
                recommendations.append({
                    'move': move,
                    'score': score,
                    'label': label,
                    'color': color
                })
                
        finally:
            self.agent.player_id = original_id
            
        return recommendations

    def get_move_recommendation(self, gui_board, player_id: int) -> Optional[Tuple]:
        """
        Suggests the best move for the human player.
        """
        # Temporarily switch agent ID if needed or just use current logic
        original_id = self.agent.player_id
        self.agent.player_id = player_id
        
        try:
            move = self.choose_move(gui_board, player_id)
        finally:
            self.agent.player_id = original_id
            
        return move

    def get_rank_probabilities(self, pos: Tuple[int, int]) -> Optional[Dict]:
        """
        Expose AAREN's rank probabilities for a specific square.
        """
        probs = self.agent.history.get_piece_predictions(pos)
        if probs:
            # Convert PieceType keys to simple names/ints for GUI
            return {pt.name: float(p) for pt, p in probs.items() if p > 0.01}
        return None
    
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
