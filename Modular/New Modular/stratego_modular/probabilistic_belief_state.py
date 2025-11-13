# stratego_modular/probabilistic_belief_state.py

"""
Probabilistic Belief State (PBS) for Stratego
Tracks piece actions and infers possible piece types/values using:
1. Rule-based inference (e.g., multi-tile moves = Scout)
2. LSTM-based pattern learning from action sequences
3. Confidence scores for each possible piece value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from .piece import PieceType, PIECE_RANKS

# Number of piece types (excluding FLAG and BOMB for some inferences)
NUM_PIECE_TYPES = len(PieceType)

# Import PBS evaluator (optional, to avoid circular imports)
try:
    from .pbs_evaluator import PBSEvaluator
    PBS_EVALUATOR_AVAILABLE = True
except ImportError:
    PBS_EVALUATOR_AVAILABLE = False


class PieceActionLSTM(nn.Module):
    """LSTM network to learn piece value patterns from action sequences."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = NUM_PIECE_TYPES, device=None):
        """
        Initialize LSTM for piece value inference.
        
        Args:
            input_size: Size of action feature vector (8 features)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            output_size: Number of possible piece types
            device: PyTorch device
        """
        super(PieceActionLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size) with piece type probabilities
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Softmax to get probabilities
        return F.softmax(x, dim=1)


class ProbabilisticBeliefState:
    """Probabilistic belief state that infers piece values from actions."""
    
    def __init__(self, player_id: int, device, lstm_hidden_size: int = 64):
        """
        Initialize the Probabilistic Belief State.
        
        Args:
            player_id: Player ID (1 or -1)
            device: PyTorch device
            lstm_hidden_size: Hidden size for LSTM network
        """
        self.player_id = player_id
        self.device = device
        
        # Track actions for each unknown piece position
        # Key: (row, col), Value: deque of action features
        self.piece_action_history: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=20)  # Keep last 20 actions per piece
        )
        
        # Belief distributions: position -> {piece_type: confidence}
        # Use sorted PieceType to ensure deterministic default (FLAG first)
        sorted_piece_types = sorted(PieceType, key=lambda pt: pt.value)
        default_beliefs = {pt: 1.0 / NUM_PIECE_TYPES for pt in sorted_piece_types}
        self.belief_distributions: Dict[Tuple[int, int], Dict[PieceType, float]] = defaultdict(
            lambda: default_beliefs.copy()
        )
        
        # LSTM model for learning action patterns
        self.lstm_model = PieceActionLSTM(
            input_size=8,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            output_size=NUM_PIECE_TYPES,
            device=device
        ).to(device)
        
        self.lstm_optimizer = torch.optim.AdamW(self.lstm_model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Track revealed pieces to update beliefs
        self.revealed_pieces: Dict[Tuple[int, int], PieceType] = {}
        
        # PBS evaluator (optional, for RL-based evaluation)
        self.evaluator = None
        if PBS_EVALUATOR_AVAILABLE:
            self.evaluator = PBSEvaluator(device=device)
        
    def reset(self):
        """Reset the belief state for a new game."""
        self.piece_action_history.clear()
        self.belief_distributions.clear()
        self.revealed_pieces.clear()
    
    def _extract_action_features(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                                 game_state) -> np.ndarray:
        """
        Extract features from an action for LSTM input.
        
        Features:
        0: Move distance (tiles)
        1: Is attack (1 if attacking, 0 otherwise)
        2: Direction (0=N, 1=S, 2=E, 3=W)
        3: Distance from center
        4: Is forward move (toward enemy)
        5: Is backward move (away from enemy)
        6: Is lateral move
        7: Aggressiveness score (based on position and action)
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Feature 0: Move distance
        distance = max(abs(r_to - r_from), abs(c_to - c_from))
        
        # Feature 1: Is attack
        is_attack = 0.0
        if hasattr(game_state, 'board'):
            board = game_state.board
            if isinstance(board, torch.Tensor):
                target_val = board[r_to, c_to].item()
                # Check if target is enemy piece
                if self.player_id == 1:
                    is_attack = 1.0 if target_val < 0 else 0.0
                else:
                    is_attack = 1.0 if target_val > 0 else 0.0
        
        # Feature 2: Direction (simplified)
        if r_to > r_from:
            direction = 1.0  # South
        elif r_to < r_from:
            direction = 0.0  # North
        elif c_to > c_from:
            direction = 2.0  # East
        else:
            direction = 3.0  # West
        
        # Feature 3: Distance from center
        center_r, center_c = 4.5, 4.5
        dist_from_center = np.sqrt((r_to - center_r)**2 + (c_to - center_c)**2) / 10.0
        
        # Feature 4-6: Move direction relative to enemy
        # For player 1, enemy is in rows 0-3, for player -1, enemy is in rows 6-9
        # Note: acting_player is the one making the move, we're tracking their pieces
        # For player 1, forward is toward row 0 (decreasing row)
        # For player -1, forward is toward row 9 (increasing row)
        if self.player_id == 1:
            # Tracking opponent (player -1), so forward for them is increasing row
            is_forward = 1.0 if r_to > r_from else 0.0
            is_backward = 1.0 if r_to < r_from else 0.0
        else:
            # Tracking opponent (player 1), so forward for them is decreasing row
            is_forward = 1.0 if r_to < r_from else 0.0
            is_backward = 1.0 if r_to > r_from else 0.0
        
        is_lateral = 1.0 if r_from == r_to or c_from == c_to else 0.0
        
        # Feature 7: Aggressiveness score
        # Higher if attacking, moving forward, or holding position near front
        aggressiveness = is_attack * 0.5
        if is_forward:
            aggressiveness += 0.3
        if distance == 0:  # Holding position
            aggressiveness += 0.2
        
        return np.array([
            distance / 10.0,  # Normalize distance
            is_attack,
            direction / 3.0,  # Normalize direction
            dist_from_center,
            is_forward,
            is_backward,
            is_lateral,
            aggressiveness
        ], dtype=np.float32)
    
    def update_from_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                          game_state, acting_player: int):
        """
        Update belief state based on an action.
        
        Args:
            action: The action taken ((from_pos), (to_pos))
            game_state: Current game state
            acting_player: Player who took the action
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Only track opponent's pieces
        if acting_player == self.player_id:
            return
        
        # Check if this is an unknown piece
        # CRITICAL: game_state.board is the visible board for the current player
        # For Agent 2, Agent 1's pieces show as HIDDEN_PIECE (-3), not positive values
        # Use actual_board if available, otherwise use visible board
        board = None
        if hasattr(game_state, 'actual_board'):
            board = game_state.actual_board
        elif hasattr(game_state, 'board'):
            board = game_state.board
        else:
            return
        
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            # If piece is revealed, don't track it
            if (r_from, c_from) in self.revealed_pieces:
                return
            
            # Check if it's an unknown enemy piece
            # CRITICAL: Use actual board values:
            # - Agent 1's pieces are positive (> 0)
            # - Agent 2's pieces are negative (< 0)
            # - HIDDEN_PIECE = -3 (only in visible boards, not actual board)
            # - LAKE_SQUARE = -13
            # - EMPTY_SQUARE = 0
            if self.player_id == 1:
                # Agent 1 tracking Agent 2's pieces (negative values in actual board)
                if piece_val < 0 and piece_val != -13:  # Not empty, not lake, enemy piece
                    pos = (r_from, c_from)
                else:
                    return
            elif self.player_id == -1:
                # Agent 2 tracking Agent 1's pieces (positive values in actual board)
                if piece_val > 0:  # Enemy piece (Agent 1's pieces are positive in actual board)
                    pos = (r_from, c_from)
                else:
                    return
            else:
                return
        else:
            return
        
        # Extract action features
        action_features = self._extract_action_features(action, game_state)
        self.piece_action_history[pos].append(action_features)
        
        # Rule-based inference
        self._apply_rule_based_inference(pos, action)
        
        # LSTM-based inference
        if len(self.piece_action_history[pos]) >= 3:
            self._apply_lstm_inference(pos)
        
        # Update position tracking: if piece moved, transfer beliefs to new position
        if (r_from, c_from) != (r_to, c_to):
            # Check if target position has an unknown piece (after move)
            # This will be handled in the next state update, but we can prepare
            # by transferring the belief distribution if the piece moved to an empty square
            if (r_from, c_from) in self.belief_distributions:
                # Store the old position's beliefs temporarily
                old_beliefs = self.belief_distributions.get((r_from, c_from), {})
                old_history = self.piece_action_history.get((r_from, c_from), deque())
                
                # If the piece moved (not a battle), transfer beliefs to new position
                # Note: This is a simplification - in reality, we'd need to check
                # if the move was successful and the piece is still unknown
                # For now, we'll keep tracking at the old position until we confirm
                # the piece moved to the new position in the next state update
    
    def _apply_rule_based_inference(self, pos: Tuple[int, int], 
                                   action: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Apply rule-based inference to update beliefs.
        
        Rules:
        - Moving more than 1 tile = Scout
        - Not moving = Flag or Bomb
        - Aggressive behavior = High value piece
        """
        (r_from, c_from), (r_to, c_to) = action
        distance = max(abs(r_to - r_from), abs(c_to - c_from))
        
        # Get current beliefs
        beliefs = self.belief_distributions[pos]
        
        # Rule 1: Multi-tile move = Scout
        if distance > 1:
            # Strong evidence for Scout (ensure Python float)
            beliefs[PieceType.SCOUT] = float(min(0.9, float(beliefs[PieceType.SCOUT]) + 0.3))
            # Reduce probability for non-moving pieces (ensure Python float)
            beliefs[PieceType.FLAG] = float(beliefs[PieceType.FLAG] * 0.5)
            beliefs[PieceType.BOMB] = float(beliefs[PieceType.BOMB] * 0.5)
            # Normalize (ensure all values are Python floats)
            total = float(sum(beliefs.values()))
            if total > 0:
                for pt in PieceType:
                    beliefs[pt] = float(beliefs[pt] / total)
        
        # Rule 2: Single tile move = Not Scout, Flag, or Bomb
        elif distance == 1:
            # Reduce Scout probability (ensure Python float)
            beliefs[PieceType.SCOUT] = float(beliefs[PieceType.SCOUT] * 0.3)
            beliefs[PieceType.FLAG] = float(beliefs[PieceType.FLAG] * 0.5)
            beliefs[PieceType.BOMB] = float(beliefs[PieceType.BOMB] * 0.5)
            # Normalize (ensure all values are Python floats)
            total = float(sum(beliefs.values()))
            if total > 0:
                for pt in PieceType:
                    beliefs[pt] = float(beliefs[pt] / total)
    
    def _apply_lstm_inference(self, pos: Tuple[int, int]):
        """
        Apply LSTM-based inference to update beliefs from action sequence.
        """
        if pos not in self.piece_action_history:
            return
        
        action_sequence = list(self.piece_action_history[pos])
        if len(action_sequence) < 1:
            return
        
        # Prepare input for LSTM (batch_size=1, sequence_length, features)
        # Pad or truncate to fixed length
        seq_length = min(len(action_sequence), 10)
        action_tensor = torch.zeros(1, 10, 8, device=self.device)
        
        for i, action_feat in enumerate(action_sequence[-seq_length:]):
            action_tensor[0, i] = torch.tensor(action_feat, device=self.device)
        
        # Get LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            predictions = self.lstm_model(action_tensor)
        
        # Convert predictions to piece types
        # LSTM outputs probabilities for each piece type index
        pred_probs = predictions[0].cpu().numpy()
        
        # Map indices to PieceType - use sorted order to match default distribution
        # CRITICAL: Use sorted PieceType to ensure consistent mapping
        piece_types = sorted(PieceType, key=lambda pt: pt.value)
        beliefs = self.belief_distributions[pos]
        
        # Combine LSTM predictions with existing beliefs (weighted average)
        alpha = 0.3  # Weight for LSTM predictions
        for i, piece_type in enumerate(piece_types):
            if i < len(pred_probs):
                # Convert to Python float to avoid numpy type issues
                new_belief = float((1 - alpha) * float(beliefs[piece_type]) + alpha * float(pred_probs[i]))
                beliefs[piece_type] = new_belief
        
        # Normalize (ensure all values are Python floats)
        total = float(sum(beliefs.values()))
        if total > 0:
            for pt in PieceType:
                beliefs[pt] = float(beliefs[pt] / total)
    
    def update_from_reveal(self, pos: Tuple[int, int], piece_type: PieceType, 
                          game_phase: str = 'middle', turn_count: int = 0):
        """
        Update beliefs when a piece is revealed (e.g., after battle).
        Also collect data for PBS evaluator training if evaluator is available.
        
        Args:
            pos: Position of the revealed piece
            piece_type: Actual piece type (ground truth)
            game_phase: 'middle' or 'end' game phase
            turn_count: Current turn number
        """
        # Get PBS prediction before updating (for evaluator training)
        if self.evaluator is not None and pos in self.belief_distributions:
            pbs_prediction = self.belief_distributions[pos].copy()
            # Only collect data from middle/end game (skip early game)
            if game_phase in ['middle', 'end']:
                self.evaluator.remember(
                    pbs_prediction=pbs_prediction,
                    ground_truth=piece_type,
                    position=pos,
                    game_phase=game_phase,
                    turn_count=turn_count
                )
        
        self.revealed_pieces[pos] = piece_type
        # Set belief to 1.0 for revealed type, 0.0 for others
        self.belief_distributions[pos] = {
            pt: 1.0 if pt == piece_type else 0.0 for pt in PieceType
        }
    
    def get_evaluator_feedback(self, pos: Tuple[int, int], 
                               ground_truth: Optional[PieceType] = None) -> Optional[Dict]:
        """
        Get feedback from PBS evaluator on prediction quality.
        
        Args:
            pos: Position to evaluate
            ground_truth: Optional ground truth piece type
            
        Returns:
            Feedback dictionary or None if evaluator not available
        """
        if self.evaluator is None or pos not in self.belief_distributions:
            return None
        
        pbs_prediction = self.belief_distributions[pos]
        return self.evaluator.get_feedback(pbs_prediction, ground_truth)
    
    def train_evaluator(self, epochs: int = 1) -> Optional[float]:
        """
        Train the PBS evaluator on collected data.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Average loss or None if evaluator not available
        """
        if self.evaluator is None:
            return None
        
        return self.evaluator.train(epochs=epochs)
    
    def get_belief_distribution(self, pos: Tuple[int, int]) -> Dict[PieceType, float]:
        """
        Get the belief distribution for a piece at a given position.
        
        Returns:
            Dictionary mapping PieceType to confidence score
        """
        return self.belief_distributions[pos].copy()
    
    def get_expected_value(self, pos: Tuple[int, int]) -> float:
        """
        Get the expected piece value based on belief distribution.
        
        Returns:
            Expected value (weighted average of piece ranks)
        """
        beliefs = self.belief_distributions[pos]
        expected_value = 0.0
        
        for piece_type, confidence in beliefs.items():
            rank = PIECE_RANKS.get(piece_type, 0)
            expected_value += confidence * rank
        
        return expected_value
    
    def get_confidence_scores(self, pos: Tuple[int, int]) -> torch.Tensor:
        """
        Get confidence scores for all piece types at a position.
        
        Returns:
            Tensor of shape (NUM_PIECE_TYPES,) with confidence scores
        """
        beliefs = self.belief_distributions[pos]
        piece_types = list(PieceType)
        scores = torch.zeros(NUM_PIECE_TYPES, device=self.device)
        
        for i, piece_type in enumerate(piece_types):
            scores[i] = beliefs.get(piece_type, 0.0)
        
        return scores
    
    def get_belief_enhanced_state(self, game_state, board_size: int = 10) -> torch.Tensor:
        """
        Enhance game state with belief information.
        
        Returns:
            Enhanced state tensor with belief probabilities
        """
        if not hasattr(game_state, 'board'):
            return None
        
        board = game_state.board
        if not isinstance(board, torch.Tensor):
            return None
        
        # Create belief-enhanced board
        # For each position, add belief probabilities
        enhanced_state = board.clone().float()
        
        # Add belief layer: for each unknown enemy piece, add expected value
        for r in range(board_size):
            for c in range(board_size):
                pos = (r, c)
                piece_val = board[r, c].item()
                
                # If it's an unknown enemy piece
                if (self.player_id == 1 and piece_val < 0 and piece_val != -13) or \
                   (self.player_id == -1 and piece_val > 0):
                    if pos not in self.revealed_pieces:
                        # Add expected value as additional information
                        expected_val = float(self.get_expected_value(pos))
                        # Store in a normalized form (0-1 range)
                        # Calculate enhancement and assign as Python float (PyTorch accepts this)
                        enhancement_value = float((expected_val / 12.0) * 0.1)
                        enhanced_state[r, c] = float(piece_val) + enhancement_value
        
        return enhanced_state
    
    def train_lstm(self, action_sequences: List[List[np.ndarray]], 
                   true_piece_types: List[PieceType], epochs: int = 10):
        """
        Train the LSTM on labeled action sequences.
        
        Args:
            action_sequences: List of action feature sequences
            true_piece_types: List of true piece types for each sequence
            epochs: Number of training epochs
        """
        if len(action_sequences) == 0:
            return
        
        self.lstm_model.train()
        
        # Prepare training data
        max_seq_len = max(len(seq) for seq in action_sequences)
        max_seq_len = min(max_seq_len, 10)
        
        batch_size = min(len(action_sequences), 32)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Create batches
            for batch_start in range(0, len(action_sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(action_sequences))
                batch_seqs = action_sequences[batch_start:batch_end]
                batch_labels = true_piece_types[batch_start:batch_end]
                
                # Prepare batch tensor
                batch_tensor = torch.zeros(len(batch_seqs), max_seq_len, 8, device=self.device)
                batch_labels_tensor = torch.zeros(len(batch_seqs), NUM_PIECE_TYPES, device=self.device)
                
                for i, (seq, label) in enumerate(zip(batch_seqs, batch_labels)):
                    # Pad sequence
                    seq_padded = seq[-max_seq_len:] if len(seq) >= max_seq_len else seq
                    for j, feat in enumerate(seq_padded):
                        if j < max_seq_len:
                            batch_tensor[i, j] = torch.tensor(feat, device=self.device)
                    
                    # Create one-hot label
                    piece_types = list(PieceType)
                    label_idx = piece_types.index(label)
                    batch_labels_tensor[i, label_idx] = 1.0
                
                # Forward pass
                predictions = self.lstm_model(batch_tensor)
                
                # Compute loss
                loss = F.cross_entropy(predictions, batch_labels_tensor.argmax(dim=1))
                
                # Backward pass
                self.lstm_optimizer.zero_grad()
                loss.backward()
                self.lstm_optimizer.step()
                
                total_loss += loss.item()
        
        self.lstm_model.eval()
    
    def save_lstm_model(self, filepath: str):
        """
        Save the LSTM model separately.
        
        Args:
            filepath: Path to save the LSTM model
        """
        torch.save({
            'lstm_state_dict': self.lstm_model.state_dict(),
            'lstm_optimizer_state_dict': self.lstm_optimizer.state_dict(),
        }, filepath)
    
    def load_lstm_model(self, filepath: str):
        """
        Load the LSTM model separately.
        
        Args:
            filepath: Path to load the LSTM model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        self.lstm_optimizer.load_state_dict(checkpoint['lstm_optimizer_state_dict'])

