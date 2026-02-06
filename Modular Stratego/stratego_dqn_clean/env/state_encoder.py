"""
State Encoder for Double DQN
Extracts 15-channel board tensor from Stratego game state.

Ported from drqn_agent.py get_state_representation()

Channels:
- 0-11: Own piece types (1-12 for P1, -1 to -12 for P2)
- 12: Enemy pieces (hidden)  
- 13: Obstacles (lakes)
- 14: Empty squares
"""

import torch
import numpy as np
from typing import Union, Optional

# Board constants
BOARD_SIZE = 10
LAKE_SQUARE = -13
EMPTY_SQUARE = 0


class StateEncoder:
    """
    Encodes Stratego game state into 15-channel tensor.
    
    This is the BOARD-ONLY encoder. AAREN embeddings are added separately.
    Total channels after AAREN fusion: 15 + 64 = 79
    """
    
    NUM_CHANNELS = 15
    
    def __init__(self, player_id: int, device: torch.device):
        """
        Initialize state encoder.
        
        Args:
            player_id: 1 or -1
            device: PyTorch device
        """
        self.player_id = player_id
        self.device = device
    
    def encode(
        self, 
        board: Union[torch.Tensor, np.ndarray],
        aaren_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode board state to feature tensor.
        
        Args:
            board: (10, 10) game board
            aaren_embedding: Optional (64,) AAREN embedding to append
            
        Returns:
            state: (15, 10, 10) or (79, 10, 10) if AAREN provided
        """
        # Convert to tensor
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).to(self.device)
        board = board.to(self.device)
        
        # Initialize 15 channels
        features = torch.zeros((self.NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), 
                               device=self.device, dtype=torch.float32)
        
        if self.player_id == 1:
            # Player 1: Own pieces are positive (1-12)
            for i in range(1, 13):
                features[i-1] = (board == i).float()
            # Enemy pieces: Negative values > LAKE_SQUARE
            features[12] = ((board < 0) & (board > LAKE_SQUARE)).float()
        else:
            # Player 2: Own pieces are negative (-1 to -12)
            for i in range(1, 13):
                features[i-1] = (board == -i).float()
            # Enemy pieces: Positive values
            features[12] = (board > 0).float()
        
        # Channel 13: Obstacles (lakes)
        features[13] = (board == LAKE_SQUARE).float()
        
        # Channel 14: Empty squares
        features[14] = (board == EMPTY_SQUARE).float()
        
        # Append AAREN embedding if provided
        if aaren_embedding is not None:
            # Expand (64,) to (64, 10, 10)
            aaren_spatial = aaren_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, 10, 10)
            features = torch.cat([features, aaren_spatial], dim=0)
        
        return features
    
    def encode_batch(
        self,
        boards: list,
        aaren_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode batch of board states.
        
        Args:
            boards: List of (10, 10) boards
            aaren_embeddings: Optional (B, 64) AAREN embeddings
            
        Returns:
            states: (B, C, 10, 10) where C is 15 or 79
        """
        batch_tensors = []
        
        for i, board in enumerate(boards):
            aaren_emb = None
            if aaren_embeddings is not None:
                aaren_emb = aaren_embeddings[i]
            
            tensor = self.encode(board, aaren_emb)
            batch_tensors.append(tensor)
        
        return torch.stack(batch_tensors)
    
    def encode_full_observability(
        self,
        board: Union[torch.Tensor, np.ndarray],
        embedding_size: int = 64
    ) -> torch.Tensor:
        """
        Encode with full observability (ground truth enemy types).
        Used for curriculum learning Phase 1.
        
        Args:
            board: (10, 10) TRUE game board with revealed enemy types
            embedding_size: Size of embedding channels to create
            
        Returns:
            state: (15 + embedding_size, 10, 10)
        """
        # Get base 15-channel encoding
        features = self.encode(board, aaren_embedding=None)
        
        # Create embedding from ground truth
        embedding = torch.zeros((embedding_size, 10, 10), device=self.device)
        
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).to(self.device)
        
        if self.player_id == 1:
            # Find enemy pieces and encode their true types
            enemy_mask = (board < 0) & (board > LAKE_SQUARE)
            enemy_positions = torch.nonzero(enemy_mask)
            for pos in enemy_positions:
                r, c = pos[0].item(), pos[1].item()
                piece_type_idx = abs(int(board[r, c].item())) - 1
                if 0 <= piece_type_idx < 12:
                    # One-hot in first 12 channels
                    embedding[piece_type_idx, r, c] = 1.0
        else:
            enemy_mask = board > 0
            enemy_positions = torch.nonzero(enemy_mask)
            for pos in enemy_positions:
                r, c = pos[0].item(), pos[1].item()
                piece_type_idx = int(board[r, c].item()) - 1
                if 0 <= piece_type_idx < 12:
                    embedding[piece_type_idx, r, c] = 1.0
        
        # Concatenate
        return torch.cat([features, embedding], dim=0)


def create_state_encoder(player_id: int, device: torch.device) -> StateEncoder:
    """Factory function to create state encoder."""
    return StateEncoder(player_id, device)
