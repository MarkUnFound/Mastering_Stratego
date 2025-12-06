# Opponent Agents for League Training
# These agents provide diverse opponents for robust training

"""
Specialized opponent agents for league training.
- RandomAgent: Makes random valid moves
- GreedyAgent: Uses heuristics to make "good" moves
"""

import torch
import random
from typing import List, Tuple, Optional
from piece import PieceType, PIECE_RANKS


class RandomAgent:
    """Agent that selects random valid moves."""
    
    def __init__(self):
        self.name = "RandomAgent"
    
    def act(self, board, valid_moves: List, game_state=None, **kwargs) -> Optional[Tuple]:
        """Select a random valid move."""
        if not valid_moves:
            return None
        return random.choice(valid_moves)
    
    def act_batch(self, boards, valid_moves_list: List[List], game_states=None, **kwargs) -> List[Optional[Tuple]]:
        """Batch version of act."""
        return [self.act(b, vm) for b, vm in zip(boards, valid_moves_list)]
    
    def reset_pbs(self):
        """No-op for compatibility."""
        pass
    
    def update_pbs_batch(self, *args, **kwargs):
        """No-op for compatibility."""
        pass


class GreedyAgent:
    """
    Agent that uses simple heuristics:
    - Prefers attacking weaker pieces
    - Prefers moving forward
    - Avoids losing high-value pieces
    """
    
    def __init__(self, device=None, player_id: int = -1):
        self.name = "GreedyAgent"
        self.device = device
        self.player_id = player_id
    
    def _score_move(self, move: Tuple, board, player_id: int) -> float:
        """Score a move based on heuristics."""
        (r_from, c_from), (r_to, c_to) = move
        score = 0.0
        
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            target_val = board[r_to, c_to].item()
        else:
            piece_val = board[r_from][c_from]
            target_val = board[r_to][c_to]
        
        piece_rank = abs(piece_val)
        target_rank = abs(target_val) if target_val != 0 else 0
        
        # Reward forward movement
        if player_id == 1:
            score += (r_from - r_to) * 0.1  # Moving up (decreasing row) is forward
        else:
            score += (r_to - r_from) * 0.1  # Moving down (increasing row) is forward
        
        # Reward attacking
        if target_rank > 0:
            # Estimate win probability based on rank difference
            if piece_rank > target_rank:
                score += 0.5 + (piece_rank - target_rank) * 0.1
            elif piece_rank == target_rank:
                score += 0.1  # Neutral trade
            else:
                # Risky attack - penalize based on our piece value
                score -= piece_rank * 0.1
            
            # Bonus for attacking with low-value scouts
            if piece_rank <= 2:
                score += 0.3
        
        # Penalize moving high-value pieces early
        if piece_rank >= 8:
            score -= 0.2
        
        # Small random noise to break ties
        score += random.uniform(0, 0.05)
        
        return score
    
    def act(self, board, valid_moves: List, game_state=None, **kwargs) -> Optional[Tuple]:
        """Select best move according to heuristics."""
        if not valid_moves:
            return None
        
        player_id = self.player_id
        if game_state and hasattr(game_state, 'current_player'):
            player_id = game_state.current_player
        
        # Score all moves and pick best
        scored_moves = [(move, self._score_move(move, board, player_id)) for move in valid_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return scored_moves[0][0]
    
    def act_batch(self, boards, valid_moves_list: List[List], game_states=None, **kwargs) -> List[Optional[Tuple]]:
        """Batch version of act."""
        results = []
        for i, (board, valid_moves) in enumerate(zip(boards, valid_moves_list)):
            gs = game_states[i] if game_states else None
            results.append(self.act(board, valid_moves, gs))
        return results
    
    def reset_pbs(self):
        """No-op for compatibility."""
        pass
    
    def update_pbs_batch(self, *args, **kwargs):
        """No-op for compatibility."""
        pass


class OpponentPool:
    """
    Manages a pool of diverse opponents for training.
    Selects opponents based on configured probabilities.
    """
    
    def __init__(self, league_manager, device, 
                 league_prob: float = 0.5,
                 random_prob: float = 0.2,
                 greedy_prob: float = 0.2,
                 self_prob: float = 0.1):
        """
        Initialize opponent pool.
        
        Args:
            league_manager: LeagueManager instance for historical opponents
            device: PyTorch device
            league_prob: Probability of selecting league opponent
            random_prob: Probability of selecting random agent
            greedy_prob: Probability of selecting greedy agent
            self_prob: Probability of self-play
        """
        self.league = league_manager
        self.device = device
        self.random_agent = RandomAgent()
        self.greedy_agent = GreedyAgent(device=device)
        
        # Normalize probabilities
        total = league_prob + random_prob + greedy_prob + self_prob
        self.league_prob = league_prob / total
        self.random_prob = random_prob / total
        self.greedy_prob = greedy_prob / total
        self.self_prob = self_prob / total
        
        self.current_opponent = None
        self.current_opponent_type = None
    
    def select_opponent(self) -> Tuple[str, object]:
        """
        Select an opponent for the next episode.
        
        Returns:
            Tuple of (opponent_type, opponent_or_path)
        """
        r = random.random()
        
        # League opponent (with fallback if no agents available)
        if r < self.league_prob:
            path = self.league.get_opponent()
            if path:
                self.current_opponent_type = "league"
                self.current_opponent = path
                return ("league", path)
            else:
                # No league agents yet - fall back to self-play
                # This happens early in training before any agents are saved
                self.current_opponent_type = "self"
                self.current_opponent = None
                return ("self", None)
        
        r -= self.league_prob
        
        # Random agent
        if r < self.random_prob:
            self.current_opponent_type = "random"
            self.current_opponent = self.random_agent
            return ("random", self.random_agent)
        
        r -= self.random_prob
        
        # Greedy agent
        if r < self.greedy_prob:
            self.current_opponent_type = "greedy"
            self.current_opponent = self.greedy_agent
            return ("greedy", self.greedy_agent)
        
        # Self-play
        self.current_opponent_type = "self"
        self.current_opponent = None
        return ("self", None)
    
    def get_stats(self) -> dict:
        """Get opponent selection statistics."""
        return {
            "league_prob": self.league_prob,
            "random_prob": self.random_prob,
            "greedy_prob": self.greedy_prob,
            "self_prob": self.self_prob,
            "current_type": self.current_opponent_type
        }
