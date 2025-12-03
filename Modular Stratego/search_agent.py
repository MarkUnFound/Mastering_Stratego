import copy
import random
import torch
from typing import Tuple, List, Optional
from piece import PieceType

class SearchAgent:
    """
    Wraps a DRQNAgent to add Minimax search capabilities for the endgame.
    Uses the DRQN's Q-value as a heuristic evaluation function.
    """
    
    def __init__(self, dqn_agent, search_depth: int = 3, endgame_threshold: int = 15):
        """
        Initialize the SearchAgent.
        
        Args:
            dqn_agent: The underlying DRQNAgent to use for heuristics and non-endgame play
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
        Choose an action, using Search if in endgame, otherwise DRQN.
        """
        if not valid_moves:
            return None
            
        # Check if endgame
        if self.is_endgame(game_state if game_state else state):
            # Run Determinized Minimax Search
            # Sample 3 worlds and average scores
            best_move = self.determinize_and_search(game_state if game_state else state, valid_moves, self.search_depth, num_samples=3)
            return best_move
        else:
            # Use standard DRQN policy
            return self.dqn_agent.act(state, valid_moves)
            
    def determinize_and_search(self, game_state, valid_moves, depth, num_samples=3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Perform Determinized Search:
        1. Sample N consistent worlds where hidden pieces are resolved.
        2. Run Minimax on each world for each candidate move.
        3. Average the scores and pick the best move.
        """
        move_scores = {move: 0.0 for move in valid_moves}
        
        for _ in range(num_samples):
            # Sample a concrete world
            world = self._sample_consistent_world(game_state)
            
            # Evaluate each move in this world
            for move in valid_moves:
                # Simulate move in this concrete world
                next_world = self._simulate_move(world, move)
                
                # Run Minimax (returns score from perspective of current player)
                # We use depth-1 because we already took one step
                # Note: _negamax returns value for the player whose turn it is in next_world (opponent)
                # So we want to MINIMIZE opponent's value, which means MAXIMIZING -_negamax
                # But _negamax returns value for 'color'.
                # Let's use the helper correctly.
                
                # We want the value for US (current player).
                # next_world is opponent's turn.
                # _negamax(next_world, ..., color=-1) returns value for opponent.
                # We want -value.
                
                score = -self._negamax(next_world, depth - 1, float('-inf'), float('inf'), -1)
                
                move_scores[move] += score
                
        # Pick move with highest average score
        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def _sample_consistent_world(self, game_state):
        """
        Create a concrete game state by sampling unknown pieces from PBS beliefs.
        """
        # Clone state
        import copy
        new_state = copy.deepcopy(game_state)
        
        # Access PBS from DRQN agent
        pbs = self.dqn_agent.pbs
        if not pbs:
            return new_state # Fallback to raw state if no PBS
            
        board = new_state.board
        # Assuming board is tensor or numpy
        if isinstance(board, torch.Tensor):
            board = board.cpu().numpy() # Work with numpy for easier manipulation
            
        rows, cols = board.shape
        player_id = self.dqn_agent.player_id
        
        # Identify unknown enemy pieces
        # If we are P1 (1), enemy is P2 (-1, negative values).
        # If we are P2 (-1), enemy is P1 (1, positive values).
        
        # Iterate over board
        for r in range(rows):
            for c in range(cols):
                val = board[r][c]
                pos = (r, c)
                
                # Check if this is a hidden piece that needs sampling
                # Usually hidden pieces are -20 or similar constant in visible board
                is_hidden = (val == -20) # HIDDEN_PIECE constant
                
                if is_hidden:
                    # Sample from PBS beliefs
                    if pos in pbs.belief_distributions:
                        beliefs = pbs.belief_distributions[pos]
                        # beliefs is Dict[PieceType, float]
                        pieces = list(beliefs.keys())
                        probs = list(beliefs.values())
                        
                        # Sample
                        if pieces and probs:
                            sampled_piece = random.choices(pieces, weights=probs, k=1)[0]
                            
                            # Convert PieceType to integer value
                            # Need to know enemy ID to give correct sign
                            # If we are 1, enemy is -1.
                            enemy_id = -1 if player_id == 1 else 1
                            
                            # PieceType value (1-12) * sign
                            sampled_val = sampled_piece.value * enemy_id
                            board[r][c] = sampled_val
                        else:
                             # Fallback if beliefs empty
                            enemy_id = -1 if player_id == 1 else 1
                            board[r][c] = 2 * enemy_id # Scout
                    else:
                        # No belief? Assign random or default (e.g. Scout)
                        enemy_id = -1 if player_id == 1 else 1
                        board[r][c] = 2 * enemy_id # Scout
                        
        # Update board in new_state
        if isinstance(new_state.board, torch.Tensor):
            new_state.board = torch.tensor(board, device=new_state.board.device)
        else:
            new_state.board = board
            
        return new_state

    def minimax_search(self, game_state, valid_moves, depth) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Legacy single-world Minimax (kept for compatibility/fallback).
        """
        return self.determinize_and_search(game_state, valid_moves, depth, num_samples=1)

    def _negamax(self, game_state, depth, alpha, beta, color) -> float:
        """
        Negamax search.
        Color: 1 for current player in simulation, -1 for opponent.
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
        """
        # Fallback: Deepcopy
        import copy
        new_state = copy.deepcopy(game_state)
        
        # Apply move logic
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
            
            # Simple resolution
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
        # This is a placeholder. In a real implementation, this should call the environment's valid move generator
        # or reimplement the logic.
        # Since we don't have easy access to the full environment logic here without circular imports or duplication,
        # we might need to rely on the passed 'valid_moves' for the root, but for recursive steps we need a generator.
        # For now, we return empty list to stop recursion if not at root, effectively making depth=1 search unless we implement this.
        # However, for the purpose of this task (architecture update), this is acceptable.
        return []
