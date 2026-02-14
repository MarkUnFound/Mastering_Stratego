"""
History Aggregator - Lightweight AAREN wrapper for action history tracking.

Replaces the complex PBS (Probabilistic Belief State) system with a simpler
approach: AAREN produces embeddings from action history, and the agent network
learns to interpret these embeddings for piece inference.

Based on DeepNash's end-to-end approach which achieved human-level Stratego
without explicit belief state computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List, Dict
from collections import defaultdict, deque

from aaren.network import PieceActionAaren
from piece import PieceType, NUM_PIECE_TYPES

# Default configuration
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
MAX_HISTORY_LENGTH = 50  # Max actions to track per position (supports 200-500 move games)


class HistoryAggregator:
    """
    Lightweight wrapper around AAREN for action history aggregation.
    
    Instead of computing belief distributions, this class:
    1. Tracks action history per enemy piece position
    2. Produces fixed-size embeddings via AAREN
    3. Lets the agent network learn piece inference implicitly
    
    This follows DeepNash's model-free approach where the network
    learns to handle hidden information without explicit beliefs.
    """
    
    def __init__(self, player_id: int, device, hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 shared_aaren_model: Optional[nn.Module] = None,
                 input_size: int = 24):
        """
        Initialize the History Aggregator.
        
        Args:
            player_id: Player ID (1 or -1)
            device: PyTorch device
            hidden_size: AAREN hidden size (also embedding output size)
            num_layers: Number of AAREN layers
            shared_aaren_model: Optional shared AAREN model for multi-env
            input_size: Size of action feature vector
        """
        self.player_id = player_id
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # AAREN model (shared or owned)
        if shared_aaren_model is not None:
            self.aaren = shared_aaren_model
            self.owns_aaren = False
        else:
            self.aaren = PieceActionAaren(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=NUM_PIECE_TYPES,
                device=device
            ).to(device)
            self.owns_aaren = True
        
        # Optimizer (only if we own the model)
        if self.owns_aaren:
            self.optimizer = optim.AdamW(self.aaren.parameters(), lr=0.001, weight_decay=0.01)
        else:
            self.optimizer = None  # Use shared optimizer
        
        # Per-position tracking
        self.position_histories: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
        self.position_hidden_states: Dict[Tuple[int, int], Optional[List[Tuple]]] = defaultdict(lambda: None)
        
        # Training data collection
        self.training_buffer: List[Tuple[List[List[float]], int]] = []  # (sequence, true_type)
        self.training_losses = deque(maxlen=1000)  # Bounded to prevent memory growth
        self.max_buffer_size = 10000
        
        # Cached embedding tensor (updated on demand)
        self._embedding_cache: Optional[torch.Tensor] = None
        self._cache_valid = False
        
        # Metrics
        self.predictions_correct = 0
        self.predictions_total = 0
    
    def reset(self):
        """Reset state for a new game."""
        self.position_histories.clear()
        self.position_hidden_states.clear()
        self._embedding_cache = None
        self._cache_valid = False
    
    def _extract_action_features(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                                  game_state, pos: Optional[Tuple[int, int]] = None) -> List[float]:
        """
        Extract features from an action for AAREN input.
        
        Args:
            action: ((from_row, from_col), (to_row, to_col))
            game_state: Current game state
            pos: Position of the piece (if None, use action source)
            
        Returns:
            Feature vector as list of floats
        """
        (r1, c1), (r2, c2) = action
        if pos is None:
            pos = (r1, c1)
        
        board = game_state.board if hasattr(game_state, 'board') else game_state
        
        # Direction and distance
        dr = r2 - r1
        dc = c2 - c1
        dist = abs(dr) + abs(dc)
        
        # Normalize direction
        dir_r = dr / max(abs(dr), 1) if dr != 0 else 0
        dir_c = dc / max(abs(dc), 1) if dc != 0 else 0
        
        # Target square info
        target_val = board[r2, c2].item() if hasattr(board[r2, c2], 'item') else board[r2, c2]
        is_attack = (target_val != 0 and target_val != -13)  # Not empty and not lake
        
        # Position features (normalized)
        pos_r = r1 / 9.0
        pos_c = c1 / 9.0
        
        # Turn info
        turn = game_state.turn_count if hasattr(game_state, 'turn_count') else 0
        turn_norm = min(turn / 1000.0, 1.0)  # Aligned with DEFAULT_MAX_TURNS=1000
        
        # Scout indicator (distance > 1)
        is_scout_move = 1.0 if dist > 1 else 0.0
        
        # Movement pattern features
        moves_toward_flag = 1.0 if (self.player_id == 1 and dr < 0) or (self.player_id == -1 and dr > 0) else 0.0
        lateral_move = 1.0 if dr == 0 else 0.0
        
        # Piece activity (how many times this position has moved)
        activity = min(len(self.position_histories.get(pos, [])) / 10.0, 1.0)
        
        # Build feature vector (24 features to match AAREN default)
        features = [
            dir_r, dir_c,           # Direction (2)
            dist / 9.0,             # Distance normalized (1)
            is_attack * 1.0,        # Attack flag (1)
            pos_r, pos_c,           # Position (2)
            turn_norm,              # Turn (1)
            is_scout_move,          # Scout indicator (1)
            moves_toward_flag,      # Advance indicator (1)
            lateral_move,           # Lateral move (1)
            activity,               # Piece activity (1)
            float(r1 < 5),          # Top half (1)
            float(c1 < 5),          # Left half (1)
        ]
        
        # Pad to input_size
        while len(features) < self.input_size:
            features.append(0.0)
        
        return features[:self.input_size]
    
    def update(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
               game_state, acting_player: int):
        """
        Update history for a piece that moved.
        
        Args:
            action: The move taken
            game_state: Current game state
            acting_player: Player who acted (1 or -1)
        """
        (r1, c1), (r2, c2) = action
        pos = (r1, c1)
        
        # Only track enemy pieces
        if acting_player == self.player_id:
            return
        
        # Extract features
        features = self._extract_action_features(action, game_state, pos)
        
        # Add to history (with limit)
        if len(self.position_histories[pos]) >= MAX_HISTORY_LENGTH:
            self.position_histories[pos].pop(0)
        self.position_histories[pos].append(features)
        
        # Update hidden state using recurrent mode
        feature_tensor = torch.tensor([features], device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            prev_state = self.position_hidden_states[pos]
            _, new_state = self.aaren.forward_sequential(feature_tensor, prev_state)
            self.position_hidden_states[pos] = new_state
        
        # Update position tracking (piece moved from pos to new_pos)
        new_pos = (r2, c2)
        if pos in self.position_histories:
            self.position_histories[new_pos] = self.position_histories.pop(pos)
            self.position_hidden_states[new_pos] = self.position_hidden_states.pop(pos)
        
        # Invalidate cache
        self._cache_valid = False
    
    def update_from_reveal(self, pos: Tuple[int, int], piece_type: PieceType,
                           game_phase: str = 'middle', turn_count: int = 0):
        """
        Record revealed piece for training data collection.
        
        Args:
            pos: Position of revealed piece
            piece_type: True piece type
            game_phase: Current game phase
            turn_count: Current turn number
        """
        # Get history for this position
        history = self.position_histories.get(pos, [])
        
        if len(history) >= 2:  # Need at least 2 actions
            # Add to training buffer
            true_type_idx = piece_type.value - 1  # Convert to 0-indexed
            self.training_buffer.append((history.copy(), true_type_idx))
            
            # Limit buffer size
            if len(self.training_buffer) > self.max_buffer_size:
                self.training_buffer = self.training_buffer[-self.max_buffer_size:]
            
            # Update metrics
            if pos in self.position_hidden_states and self.position_hidden_states[pos] is not None:
                # Get prediction from current state
                with torch.no_grad():
                    feature = torch.zeros(1, self.input_size, device=self.device)
                    probs, _ = self.aaren.forward_sequential(feature, self.position_hidden_states[pos])
                    predicted = probs.argmax(dim=1).item()
                    if predicted == true_type_idx:
                        self.predictions_correct += 1
                    self.predictions_total += 1
    
    def get_embedding_tensor(self) -> torch.Tensor:
        """
        Get embedding tensor for all positions.
        
        Returns:
            Tensor of shape (hidden_size, 10, 10) with AAREN embeddings
        """
        if self._cache_valid and self._embedding_cache is not None:
            return self._embedding_cache
        
        # Create embedding tensor
        embedding = torch.zeros((self.hidden_size, 10, 10), device=self.device)
        
        for pos, hidden_state in self.position_hidden_states.items():
            r, c = pos
            if hidden_state is not None and 0 <= r < 10 and 0 <= c < 10:
                # Extract last layer's hidden state
                last_layer_state = hidden_state[-1]  # (h, c) tuple
                if isinstance(last_layer_state, tuple):
                    h = last_layer_state[0]  # Use h from (h, c)
                else:
                    h = last_layer_state
                
                # h shape: (1, hidden_size) -> (hidden_size,)
                if h.dim() == 2:
                    h = h.squeeze(0)
                
                embedding[:, r, c] = h
        
        self._embedding_cache = embedding
        self._cache_valid = True
        
        return embedding
    
    def train(self, epochs: int = 1) -> Optional[float]:
        """
        Train AAREN on collected reveal data.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Average loss over training, or None if insufficient data
        """
        if len(self.training_buffer) < 32:
            return None
        
        if self.optimizer is None:
            return None  # Using shared model
        
        self.aaren.train()
        total_loss = 0
        num_batches = 0
        
        batch_size = min(32, len(self.training_buffer))
        
        for _ in range(epochs):
            # Sample batch
            indices = torch.randint(0, len(self.training_buffer), (batch_size,))
            
            batch_seqs = []
            batch_targets = []
            max_len = 0
            
            for idx in indices:
                seq, target = self.training_buffer[idx.item()]
                batch_seqs.append(seq)
                batch_targets.append(target)
                max_len = max(max_len, len(seq))
            
            # Pad sequences
            padded_seqs = []
            for seq in batch_seqs:
                padded = seq + [[0.0] * self.input_size] * (max_len - len(seq))
                padded_seqs.append(padded)
            
            # Create tensors
            input_tensor = torch.tensor(padded_seqs, device=self.device, dtype=torch.float32)
            target_tensor = torch.tensor(batch_targets, device=self.device, dtype=torch.long)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.aaren.forward_parallel(input_tensor)
            
            # Loss
            loss = nn.functional.cross_entropy(logits, target_tensor)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.aaren.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def get_accuracy(self) -> float:
        """Get prediction accuracy (0.0 to 1.0)."""
        if self.predictions_total == 0:
            return 0.0
        return self.predictions_correct / self.predictions_total
    
    def get_avg_loss(self, window: int = 50) -> float:
        """Get average training loss over recent window."""
        if not self.training_losses:
            return 0.0
        recent = self.training_losses[-window:]
        return sum(recent) / len(recent)
    
    def get_buffer_size(self) -> int:
        """Get current training buffer size."""
        return len(self.training_buffer)
    
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'aaren_state_dict': self.aaren.state_dict() if self.owns_aaren else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_losses': self.training_losses,
            'predictions_correct': self.predictions_correct,
            'predictions_total': self.predictions_total,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        if state_dict.get('aaren_state_dict') and self.owns_aaren:
            self.aaren.load_state_dict(state_dict['aaren_state_dict'])
        if state_dict.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.training_losses = state_dict.get('training_losses', [])
        self.predictions_correct = state_dict.get('predictions_correct', 0)
        self.predictions_total = state_dict.get('predictions_total', 0)
