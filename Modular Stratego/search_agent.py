import copy
import random
import torch
from typing import Tuple, List, Optional
from piece import PieceType

class SearchAgent:
    """
    Wraps a DQNAgent to add Minimax search capabilities for the endgame.
    Uses the DQN's Q-value as a heuristic evaluation function.
    """
    
    def __init__(self, dqn_agent, search_depth: int = 3, endgame_threshold: int = 15):
        """
        Initialize the SearchAgent.
        
        Args:
            dqn_agent: The underlying DQNAgent to use for heuristics and non-endgame play
            search_depth: Depth of the Minimax search
            endgame_threshold: Number of pieces below which search is enabled
        """
        self.dqn_agent = dqn_agent
        self.search_depth = search_depth
        self.endgame_threshold = endgame_threshold
        self.name = f"Search_{dqn_agent.name}"
        
    def is_endgame(self, game_state) -> bool:
        """Check if the game is in the endgame phase."""
        # Count total pieces on board
        total_pieces = 0
        if hasattr(game_state, 'board'):
            board = game_state.board
            # Assuming board is 10x10 numpy array or tensor
            # Count non-zero elements (pieces)
            if isinstance(board, torch.Tensor):
                total_pieces = (board != 0).sum().item()
            else:
                import numpy as np
                total_pieces = np.count_nonzero(board)
        
        return total_pieces <= self.endgame_threshold
        
    def act(self, state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], game_state=None):
        """
        Choose an action, using Search if in endgame, otherwise DQN.
        """
        if not valid_moves:
            return None
            
        # Check if endgame
        if self.is_endgame(game_state if game_state else state):
            # Run Minimax Search
            best_move = self.minimax_search(game_state if game_state else state, valid_moves, self.search_depth)
            return best_move
        else:
            # Use standard DQN policy
            return self.dqn_agent.act(state, valid_moves)
            
    def minimax_search(self, game_state, valid_moves, depth) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Perform Minimax search with Alpha-Beta pruning.
        Returns the best move found.
        """
        best_score = float('-inf')
        best_move = random.choice(valid_moves) # Default to random valid move
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort moves by immediate heuristic (e.g. captures) to improve pruning
        # For now, just shuffle to avoid bias
        random.shuffle(valid_moves)
        
        for move in valid_moves:
            # Simulate move
            next_state = self._simulate_move(game_state, move)
            
            # Recursive call (minimize opponent's score)
            score = self._min_value(next_state, depth - 1, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break # Beta cut-off
                
        return best_move

    def _max_value(self, game_state, depth, alpha, beta) -> float:
        if depth == 0 or self._is_terminal(game_state):
            return self.dqn_agent.get_state_value(game_state)
            
        valid_moves = self._get_valid_moves(game_state, player=1) # Assuming self is player 1
        if not valid_moves:
            return float('-inf') # Loss
            
        v = float('-inf')
        for move in valid_moves:
            next_state = self._simulate_move(game_state, move)
            v = max(v, self._min_value(next_state, depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _min_value(self, game_state, depth, alpha, beta) -> float:
        if depth == 0 or self._is_terminal(game_state):
            return self.dqn_agent.get_state_value(game_state) # Value is always from self perspective?
            # Wait, get_state_value returns V(s) for the current player.
            # If it's opponent's turn, V(s) is good for opponent.
            # So we want to Minimize Opponent's V(s).
            # Yes, standard Minimax.
            
        valid_moves = self._get_valid_moves(game_state, player=-1) # Opponent
        if not valid_moves:
            return float('inf') # Win for us (opponent has no moves)
            
        v = float('inf')
        for move in valid_moves:
            next_state = self._simulate_move(game_state, move)
            # Opponent tries to maximize THEIR value, which is bad for us?
            # Actually, if V(s) is always "Win Probability for Current Player",
            # Then:
            # Max node (Us): Maximize V(next_state_us)
            # Min node (Opponent): Opponent chooses move to Maximize V(next_state_opp).
            # Since V(next_state_opp) ~= 1 - V(next_state_us) (in zero-sum),
            # Minimizing V(next_state_opp) is equivalent to Maximizing V(next_state_us).
            # BUT, get_state_value returns value for the player whose turn it is.
            
            # Let's assume get_state_value returns value for the agent calling it.
            # No, DQNAgent.get_state_value takes state, gets representation (relative to current player), and returns value.
            # So it returns "How good is this state for the player whose turn it is".
            
            # So:
            # Root (Us): We want to pick move leading to state where V(state_opp) is LOW (bad for opponent).
            # Wait, after we move, it becomes Opponent's turn.
            # So we want to Minimize V(state_opp).
            
            # Opponent (Min node): Opponent picks move. After opponent moves, it becomes Our turn.
            # Opponent wants to pick move leading to state where V(state_us) is LOW (bad for us).
            # So Opponent Minimizes V(state_us).
            
            # So:
            # Max (Us): Minimize (Opponent's V after our move)
            # Min (Opponent): Minimize (Our V after their move)
            
            # This is confusing. Let's stick to standard Minimax with NegaMax or similar.
            # Or simpler:
            # We want to Maximize Our Win Probability.
            # Value function V(s) = P(Win | s).
            # If s is our turn, V(s) is high if we can win.
            # If s is opponent turn, V(s) is high if THEY can win.
            # So P(We Win) = 1 - V(s_opponent).
            
            # So:
            # We want to choose move m such that V(next_state) (opponent's view) is MINIMIZED.
            # Opponent wants to choose move m such that V(next_next_state) (our view) is MINIMIZED.
            
            # So both are Minimizers of the *next state's value*?
            # Yes, in this "Value is always for current player" setup.
            
            # Let's implement that.
            
            # Recursive step:
            # Value of move = - minimax(next_state) ? 
            # If range is [-1, 1], then yes, Negamax.
            # If range is [0, 1] (win prob), then Value = 1 - minimax(next_state).
            
            # Let's use Negamax formulation assuming V is roughly [-1, 1] or centered.
            # Our DQN uses rewards -1, 0, 1. So V is roughly [-1, 1].
            
            score = self._negamax(next_state, depth - 1, -beta, -alpha, -1)
            v = min(v, score) # Wait, Negamax handles the sign flip.
            
            # Let's write explicit Negamax helper.
            pass
            
        return best_move

    def _negamax(self, game_state, depth, alpha, beta, color) -> float:
        """
        Negamax search.
        Color: 1 for us, -1 for opponent.
        Returns value from perspective of player 'color'.
        """
        if depth == 0 or self._is_terminal(game_state):
            # Heuristic value for current player
            # get_state_value returns value for current player
            val = self.dqn_agent.get_state_value(game_state)
            return val
            
        valid_moves = self._get_valid_moves(game_state, player=1 if game_state.current_player == 1 else -1)
        if not valid_moves:
            return -1.0 # Loss
            
        value = float('-inf')
        for move in valid_moves:
            next_state = self._simulate_move(game_state, move)
            # Value of this move is -(Value of next state for opponent)
            v = -self._negamax(next_state, depth - 1, -beta, -alpha, -color)
            value = max(value, v)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def _simulate_move(self, game_state, move):
        """
        Simulate a move on a clone of the game state.
        NOTE: This requires a robust GameState clone/step method that doesn't rely on full Environment.
        """
        # This is the tricky part. We need a lightweight forward model.
        # We can try deepcopy, but it might be slow.
        # For now, assume game_state has a 'clone' and 'apply_move' method.
        # If not, we might need to implement a lightweight one here.
        
        # Fallback: Deepcopy
        import copy
        new_state = copy.deepcopy(game_state)
        
        # Apply move logic (simplified)
        # We need to handle captures, flag, etc.
        # This logic duplicates Environment logic. Ideally, GameState has this.
        # Let's assume we can use a helper from environment.py or board.py
        
        # For this implementation plan, I will assume a placeholder _apply_move
        self._apply_move_logic(new_state, move)
        return new_state

    def _apply_move_logic(self, game_state, move):
        """Apply move to state (in-place)."""
        (r1, c1), (r2, c2) = move
        board = game_state.board
        
        # Move piece
        piece_val = board[r1][c1]
        target_val = board[r2][c2]
        
        # Get piece types (absolute values)
        attacker = abs(int(piece_val))
        defender = abs(int(target_val))
        
        # Battle logic
        if target_val == 0:
            # Move to empty square
            board[r2][c2] = piece_val
            board[r1][c1] = 0
        else:
            # Battle
            # We need to resolve battle.
            # Assuming standard Stratego rules.
            # 1: Spy, 2: Scout, 3: Miner, ..., 10: Marshal, 11: Bomb, 12: Flag
            # Higher rank wins, except Spy vs Marshal, Miner vs Bomb.
            
            # Simple resolution (can import BattleResolver if needed, but keep it self-contained for speed)
            winner_val = 0
            
            # Special cases
            if attacker == 3 and defender == 11: # Miner vs Bomb
                winner_val = piece_val # Miner wins
            elif attacker == 1 and defender == 10: # Spy vs Marshal
                winner_val = piece_val # Spy wins
            elif attacker == 11: # Bomb attacking? (Illegal usually, but if allowed, it loses)
                winner_val = target_val # Defender wins
            elif defender == 11: # Attacking Bomb (non-Miner)
                winner_val = target_val # Bomb wins
            elif attacker > defender:
                winner_val = piece_val # Attacker wins
            elif defender > attacker:
                winner_val = target_val # Defender wins
            else:
                winner_val = 0 # Draw (both removed)
                
            board[r2][c2] = winner_val
            board[r1][c1] = 0
        
        # Switch turn
        game_state.current_player *= -1
        
    def _is_terminal(self, game_state) -> bool:
        """Check if game is over."""
        # Check flag capture or no moves
        return False # Simplified
        
    def _get_valid_moves(self, game_state, player):
        """Get valid moves for player."""
        # Need move generation logic
        return [] # Placeholder
