"""
Enhanced Bot Logic (Wrapper)
Delegates move selection to the formal Expectamax agent in dqn_bot_logic.py
"""

import os
import random
from typing import Tuple, Optional, List

# Import the actual agent wrapper from dqn_bot_logic
try:
    from dqn_bot_logic import DQNBotLogic
except ImportError:
    print(" Error: Could not import DQNBotLogic from dqn_bot_logic.py")
    DQNBotLogic = None


class EnhancedBotLogic:
    """
    Enhanced bot that acts as a bridge for the GUI to use the formal Expectamax search
    provided by the fully trained Rainbow DQN.
    """
    
    def __init__(self, model_path: str, player_id: int = 2):
        """
        Initialize the bot wrapper
        
        Args:
            model_path: Path to trained model weights
            player_id: Bot's player ID (1 or 2)
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        
        print(f" [BotLogic] Initializing Expectamax search with model: {model_path}")
        
        if DQNBotLogic is not None:
            # The DQNBotLogic automatically handles loading the full model & init Expectamax
            self.agent_backend = DQNBotLogic(model_path=model_path, player_id=player_id)
        else:
            self.agent_backend = None
            print(" [BotLogic] Initialization failed because DQNBotLogic is missing.")
            
    def reset(self):
        """Reset for a new game"""
        if self.agent_backend:
            self.agent_backend.reset()
    
    def choose_move(self, board, owner: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Choose best move using the neural network + Expectamax search
        
        Args:
            board: Current GUI board state
            owner: Current player
            
        Returns:
            (source, destination) tuple or None
        """
        if self.agent_backend:
            # Delegate directly to the expectamax test-time search
            return self.agent_backend.choose_move(board, owner)
        else:
            # Fallback for safety if backend failed to load
            legal_moves = self._get_all_legal_moves(board, owner)
            return random.choice(legal_moves) if legal_moves else None

    def update_from_opponent_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                                  board, revealed_rank: Optional[int] = None):
        """
        Update AAREN belief state history based on opponent's move
        """
        if self.agent_backend:
            self.agent_backend.update_from_opponent_move(board, from_pos, to_pos)
            
    def _get_all_legal_moves(self, board, owner: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Fallback legal moves getter if agent fails"""
        moves = []
        for src in board.owner_positions(owner):
            piece = board.get(src)
            if piece and piece.is_movable():
                legal_dsts = board.legal_moves_from(src)
                for dst in legal_dsts:
                    moves.append((src, dst))
        return moves
