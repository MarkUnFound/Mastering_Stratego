# PBS Module - Probabilistic Belief State
# Main belief state class extracted from probabilistic_belief_state.py

"""
ProbabilisticBeliefState: Tracks piece actions and infers possible piece types/values.

Uses:
1. Rule-based inference (e.g., multi-tile moves = Scout)
2. AAREN-based pattern learning from action sequences
3. Confidence scores for each possible piece value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict

# Local imports
from piece import PieceType, PIECE_RANKS, NUM_PIECE_TYPES

# Import AAREN components from aaren module
from aaren import PieceActionAaren

# Import utilities
from .utils import extract_action_features, calculate_entropy, normalize_beliefs

# Check if PBS evaluator is available
try:
    from pbs_evaluator import PBSEvaluator
    PBS_EVALUATOR_AVAILABLE = True
except ImportError:
    PBS_EVALUATOR_AVAILABLE = False


class ProbabilisticBeliefState:
    """Probabilistic belief state that infers piece values from actions."""
    
    def __init__(self, player_id: int, device, aaren_hidden_size: int = 64, 
                 shared_aaren_model: Optional[nn.Module] = None,
                 shared_evaluator: Optional[object] = None):
        """
        Initialize the Probabilistic Belief State.
        
        Args:
            player_id: Player ID (1 or -1)
            device: PyTorch device
            aaren_hidden_size: Hidden size for Aaren network
            shared_aaren_model: Optional shared AAREN model (for parallel envs)
            shared_evaluator: Optional shared PBS evaluator (for parallel envs)
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
        
        # Track revealed piece counts for constraints
        self.revealed_piece_counts = {pt: 0 for pt in PieceType}
        # Standard Stratego piece counts
        self.total_piece_counts = {
            PieceType.FLAG: 1, PieceType.SPY: 1, PieceType.BOMB: 6,
            PieceType.MARSHAL: 1, PieceType.GENERAL: 1, PieceType.COLONEL: 2,
            PieceType.MAJOR: 3, PieceType.CAPTAIN: 4, PieceType.LIEUTENANT: 4,
            PieceType.SERGEANT: 4, PieceType.MINER: 5, PieceType.SCOUT: 8
        }
        
        # Track piece observation times for temporal features
        self.piece_observation_times: Dict[Tuple[int, int], int] = {}
        self.turn_count = 0
        
        # Track prediction accuracy for confidence calibration
        self.prediction_history: List[Tuple[Dict[PieceType, float], PieceType, bool]] = []
        self.accuracy_by_piece_type: Dict[PieceType, List[bool]] = defaultdict(list)
        
        # Track multi-piece patterns
        self.piece_coordination: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        
        # AAREN model for learning action patterns
        if shared_aaren_model is not None:
            self.aaren_model = shared_aaren_model
            # Optimizer should be managed externally if model is shared
            self.aaren_optimizer = None 
        else:
            # Expanded input size for enhanced features (8 -> 24)
            self.aaren_model = PieceActionAaren(
                input_size=24,  # Enhanced from 8 to 24 features
                hidden_size=aaren_hidden_size,
                num_layers=3,  # Increased from 2 to 3 for better capacity
                output_size=NUM_PIECE_TYPES,
                device=device
            ).to(device)
            self.aaren_optimizer = torch.optim.AdamW(self.aaren_model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Track Aaren states for sequential inference (per position)
        # Key: (row, col), Value: List of states (one per layer)
        self.aaren_states: Dict[Tuple[int, int], List[Tuple]] = {}
        self.piece_hidden_states: Dict[Tuple[int, int], Optional[List[Tuple]]] = {}  # Explicitly for recurrent inference
        
        # Track revealed pieces to update beliefs
        self.revealed_pieces: Dict[Tuple[int, int], PieceType] = {}
        
        # Store evaluator feedback for AAREN training (weights and positions)
        self._aaren_training_weights: Dict[Tuple[int, int], float] = {}
        self._aaren_training_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # PBS evaluator (optional, for RL-based evaluation)
        if shared_evaluator is not None:
            self.evaluator = shared_evaluator
        else:
            self.evaluator = None
            if PBS_EVALUATOR_AVAILABLE:
                self.evaluator = PBSEvaluator(device=device)
        
        # Active learning: track uncertain positions that need more observation
        self.uncertain_positions: set = set()
        
        # AAREN Training Buffer
        # Stores tuples of (action_sequence, true_piece_type, position)
        self.aaren_training_buffer: deque = deque(maxlen=5000)
        
        # AAREN Metrics Tracking
        self.aaren_training_losses: List[float] = []  # Training loss history
        self.aaren_predictions_correct: int = 0  # Correct predictions count
        self.aaren_predictions_total: int = 0  # Total predictions count
        
        # Cached Belief Tensor (Optimization)
        # Shape: (NUM_PIECE_TYPES, 10, 10)
        # We maintain this on the device to avoid re-creating it every step
        self.belief_tensor = torch.zeros((NUM_PIECE_TYPES, 10, 10), device=device, dtype=torch.float32)
        
        # Pre-allocated buffer for belief prob updates (Optimization)
        # Avoids creating new tensor each update
        self._probs_buffer = torch.zeros(NUM_PIECE_TYPES, device=device, dtype=torch.float32)
        
        # Pre-compute sorted piece types once (Optimization)
        self._sorted_piece_types = sorted(PieceType, key=lambda pt: pt.value)
        
        # Uncertainty Map Cache (Optimization)
        self._cached_uncertainty_map: Dict[Tuple[int, int], float] = {}
        self._uncertainty_map_dirty: bool = True

        
    def reset(self):
        """Reset the belief state for a new game."""
        self.piece_action_history.clear()
        self.belief_distributions.clear()
        self.revealed_pieces.clear()
        self.aaren_states.clear()
        self._aaren_training_positions.clear()
        self.revealed_piece_counts = {pt: 0 for pt in PieceType}
        self.piece_observation_times.clear()
        self.turn_count = 0
        self.piece_coordination.clear()
        self.uncertain_positions.clear()
        self.prediction_history.clear()
        self.accuracy_by_piece_type.clear()
        if hasattr(self, 'piece_q_value_history'):
            self.piece_q_value_history.clear()
        # Use in-place zero_ to avoid allocation
        if hasattr(self, 'belief_tensor'):
            self.belief_tensor.zero_()
        if hasattr(self, '_probs_buffer'):
            self._probs_buffer.zero_()
        # Reset uncertainty cache
        self._cached_uncertainty_map = {}
        self._uncertainty_map_dirty = True

    
    def _update_belief_tensor(self, pos: Tuple[int, int]):
        """
        Update the cached belief tensor for a specific position.
        Should be called whenever beliefs for a position change.
        OPTIMIZED: Uses pre-allocated buffer to avoid tensor allocation.
        """
        if pos not in self.belief_distributions:
            return
            
        r, c = pos
        beliefs = self.belief_distributions[pos]
        
        # Use pre-allocated buffer and pre-computed piece type order
        for i, pt in enumerate(self._sorted_piece_types):
            self._probs_buffer[i] = beliefs.get(pt, 0.0)
        
        # Copy buffer to belief tensor (in-place operation)
        self.belief_tensor[:, r, c].copy_(self._probs_buffer)
        
        # Invalidate uncertainty cache
        self._uncertainty_map_dirty = True

    
    def _extract_action_features(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                                 game_state, pos: Optional[Tuple[int, int]] = None,
                                 apply_feature_weights: bool = True) -> np.ndarray:
        """Extract enhanced features from an action for Aaren input."""
        return extract_action_features(
            action=action,
            game_state=game_state,
            player_id=self.player_id,
            pos=pos,
            belief_distributions=self.belief_distributions,
            piece_action_history=self.piece_action_history,
            piece_observation_times=self.piece_observation_times,
            turn_count=self.turn_count,
            evaluator=self.evaluator,
            apply_feature_weights=apply_feature_weights
        )

    def prepare_aaren_update(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                            game_state, acting_player: int, q_value: Optional[float] = None) -> Optional[Tuple[Tuple[int, int], List[List[float]]]]:
        """
        Prepare for AAREN update. Returns (pos, sequence) if AAREN inference is needed.
        Also performs initial processing (history update, etc.).
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Only track opponent's pieces
        if acting_player == self.player_id:
            return None
        
        # Check if this is an unknown piece
        board = None
        if hasattr(game_state, 'actual_board'):
            board = game_state.actual_board
        elif hasattr(game_state, 'board'):
            board = game_state.board
        else:
            return None
        
        pos = None
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            # If piece is revealed, don't track it
            if (r_from, c_from) in self.revealed_pieces:
                return None
            
            if self.player_id == 1:
                if piece_val < 0 and piece_val != -13:
                    pos = (r_from, c_from)
            elif self.player_id == -1:
                if piece_val > 0:
                    pos = (r_from, c_from)
        
        if pos is None:
            return None
            
        # Track observation time
        if pos not in self.piece_observation_times:
            self.piece_observation_times[pos] = self.turn_count
            
        # Store Q-value
        if q_value is not None:
            if not hasattr(self, 'piece_q_value_history'):
                self.piece_q_value_history = {}
            if pos not in self.piece_q_value_history:
                self.piece_q_value_history[pos] = []
            self.piece_q_value_history[pos].append(q_value)
        
        # Initialize priors
        if pos not in self.belief_distributions or len(self.belief_distributions[pos]) == 0:
            self._initialize_position_priors(pos)
        
        # Extract features and update history
        action_features = self._extract_action_features(action, game_state, pos=pos)
        self.piece_action_history[pos].append(action_features)
        
        # Prepare AAREN input if history exists
        if len(self.piece_action_history[pos]) >= 1:
            history = list(self.piece_action_history[pos])
            seq_len = min(len(history), 10)
            sequence = history[-seq_len:]
            return pos, sequence
            
        return pos, None

    def prepare_recurrent_update(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                                game_state, acting_player: int) -> Optional[Tuple[Tuple[int, int], List[float], Optional[List[Tuple]]]]:
        """
        Prepare for Recurrent AAREN update (O(1)).
        Returns (pos, latest_feature, hidden_state) if inference is needed.
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Only track opponent's pieces
        if acting_player == self.player_id:
            return None
        
        # Check if this is an unknown piece
        board = None
        if hasattr(game_state, 'actual_board'):
            board = game_state.actual_board
        elif hasattr(game_state, 'board'):
            board = game_state.board
        else:
            return None
        
        pos = None
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            # If piece is revealed, don't track it
            if (r_from, c_from) in self.revealed_pieces:
                return None
            
            if self.player_id == 1:
                if piece_val < 0 and piece_val != -13:
                    pos = (r_from, c_from)
            elif self.player_id == -1:
                if piece_val > 0:
                    pos = (r_from, c_from)
        
        if pos is None:
            return None
            
        # Track observation time
        if pos not in self.piece_observation_times:
            self.piece_observation_times[pos] = game_state.turn_count
            
        # Initialize priors if needed
        if pos not in self.belief_distributions or len(self.belief_distributions[pos]) == 0:
            self._initialize_position_priors(pos)
        
        # Extract features and update history
        action_features = self._extract_action_features(action, game_state, pos=pos)
        self.piece_action_history[pos].append(action_features)
        
        # Get current hidden state
        hidden_state = self.piece_hidden_states.get(pos, None)
        
        return pos, action_features, hidden_state

    def apply_recurrent_update(self, pos: Tuple[int, int], 
                              aaren_probs: Optional[torch.Tensor], 
                              new_hidden_state: Optional[List[Tuple]],
                              action: Tuple[Tuple[int, int], Tuple[int, int]], 
                              game_state):
        """Apply Recurrent AAREN results and update hidden state."""
        # Update hidden state
        if new_hidden_state is not None:
            self.piece_hidden_states[pos] = new_hidden_state
            
        # Rule-based
        self._apply_rule_based_inference(pos, action)
        
        # Behavioral
        self._apply_behavioral_patterns(pos, action, game_state)
        
        # Apply AAREN results
        if aaren_probs is not None:
            # We trust AAREN more as sequence grows
            history_len = len(self.piece_action_history[pos])
            aaren_weight = min(0.8, history_len * 0.1)
            current_weight = 1.0 - aaren_weight
            
            beliefs = self.belief_distributions[pos]
            piece_types = list(PieceType)
            
            for k, pt in enumerate(piece_types):
                aaren_prob = aaren_probs[k].item()
                current_prob = beliefs.get(pt, 0.0)
                beliefs[pt] = current_weight * current_prob + aaren_weight * aaren_prob
            
            # Renormalize
            total = sum(beliefs.values())
            if total > 0:
                for pt in beliefs:
                    beliefs[pt] /= total
        
        # Evaluator bias correction
        if self.evaluator is not None and pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            corrected_beliefs = {}
            for piece_type in beliefs:
                correction = self.evaluator.get_bias_correction(piece_type)
                corrected_beliefs[piece_type] = beliefs[piece_type] * correction
            
            total = sum(corrected_beliefs.values())
            if total > 0:
                for piece_type in corrected_beliefs:
                    corrected_beliefs[piece_type] /= total
                self.belief_distributions[pos] = corrected_beliefs
        
        # Active learning
        if self.evaluator is not None and pos in self.belief_distributions:
            if self.evaluator.should_gather_more_info(self.belief_distributions[pos]):
                self.uncertain_positions.add(pos)
            else:
                self.uncertain_positions.discard(pos)
        
        # Constraints
        self._apply_piece_count_constraints(pos)
        
        # Update tensor
        self._update_belief_tensor(pos)
        
        # Coordination
        self._update_piece_coordination(pos, action, game_state)
        
        # Turn count
        if hasattr(game_state, 'turn_count'):
            self.turn_count = game_state.turn_count

    def apply_aaren_update(self, pos: Tuple[int, int], 
                          aaren_probs: Optional[torch.Tensor],
                          action: Tuple[Tuple[int, int], Tuple[int, int]], 
                          game_state):
        """Apply AAREN results and other inference rules."""
        # Rule-based
        self._apply_rule_based_inference(pos, action)
        
        # Behavioral
        self._apply_behavioral_patterns(pos, action, game_state)
        
        # Apply AAREN results
        if aaren_probs is not None:
            history_len = len(self.piece_action_history[pos])
            aaren_weight = min(0.8, history_len * 0.1)
            current_weight = 1.0 - aaren_weight
            
            beliefs = self.belief_distributions[pos]
            piece_types = list(PieceType)
            
            for k, pt in enumerate(piece_types):
                aaren_prob = aaren_probs[k].item()
                current_prob = beliefs.get(pt, 0.0)
                beliefs[pt] = current_weight * current_prob + aaren_weight * aaren_prob
            
            # Renormalize
            total = sum(beliefs.values())
            if total > 0:
                for pt in beliefs:
                    beliefs[pt] /= total
            
            self._update_belief_tensor(pos)
        
        # Update piece coordination
        self._update_piece_coordination(pos, action, game_state)
        
        # Update position map if piece moved
        (r_from, c_from), (r_to, c_to) = action
        if (r_from, c_from) == pos:
            new_pos = (r_to, c_to)
            # Move beliefs
            self.belief_distributions[new_pos] = self.belief_distributions.pop(pos)
            self.piece_action_history[new_pos] = self.piece_action_history.pop(pos)
            self.piece_observation_times[new_pos] = self.piece_observation_times.pop(pos, 0)
            
            # Move hidden state
            if pos in self.piece_hidden_states:
                self.piece_hidden_states[new_pos] = self.piece_hidden_states.pop(pos)
            
            # Clear old tensor pos
            self._update_belief_tensor(pos)  # Clears it because beliefs are gone
            self._update_belief_tensor(new_pos)

    def update_from_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                          game_state, acting_player: int, q_value: Optional[float] = None):
        """Update belief state based on an action."""
        result = self.prepare_aaren_update(action, game_state, acting_player, q_value)
        if result:
            pos, sequence = result
            
            # Run AAREN inference if sequence available
            probs = None
            if sequence is not None and self.aaren_model is not None:
                # Single inference
                seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.aaren_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
            
            self.apply_aaren_update(pos, probs, action, game_state)

    def update_from_reveal(self, pos: Tuple[int, int], piece_type: PieceType, 
                          game_phase: str = 'middle', turn_count: int = 0):
        """
        Update beliefs when a piece is revealed (e.g., after battle).
        Also collect data for PBS evaluator training if evaluator is available.
        """
        # Get PBS prediction before updating (for evaluator training)
        evaluator_feedback = None
        if self.evaluator is not None and pos in self.belief_distributions:
            pbs_prediction = self.belief_distributions[pos].copy()
            # Only collect data from middle/end game (skip early game)
            if game_phase in ['middle', 'end']:
                action_features = None
                if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
                    action_features = self.piece_action_history[pos][-1]
                
                q_value = None
                if hasattr(self, 'piece_q_value_history') and pos in self.piece_q_value_history and len(self.piece_q_value_history[pos]) > 0:
                    q_value = self.piece_q_value_history[pos][-1]
                
                self.evaluator.remember(
                    pbs_prediction=pbs_prediction,
                    ground_truth=piece_type,
                    position=pos,
                    game_phase=game_phase,
                    turn_count=turn_count,
                    action_features=action_features,
                    q_value=q_value
                )
                evaluator_feedback = self.get_evaluator_feedback(pos, ground_truth=piece_type)
        
        # Store evaluator feedback for AAREN training
        if evaluator_feedback and pos in self.piece_action_history:
            quality_score = evaluator_feedback.get('quality_score', 0.0)
            piece_value = PIECE_RANKS.get(piece_type, 1)
            quality_weight = 1.0 - (1.0 / (1.0 + math.exp(-quality_score / 10.0)))
            value_weight = 0.5 + (piece_value / 12.0)
            combined_weight = quality_weight * value_weight
            self._aaren_training_weights[pos] = combined_weight
            self._aaren_training_positions[pos] = pos
        
        # Ensure piece_type is a PieceType enum
        if not isinstance(piece_type, PieceType):
            if isinstance(piece_type, int):
                piece_type = PieceType(piece_type)
            else:
                raise ValueError(f"piece_type must be PieceType enum, got {type(piece_type)}: {piece_type}")
        
        self.revealed_pieces[pos] = piece_type
        self.revealed_piece_counts[piece_type] = self.revealed_piece_counts.get(piece_type, 0) + 1
        
        # Track prediction accuracy
        if pos in self.belief_distributions:
            prediction = self.belief_distributions[pos]
            predicted_type = max(prediction.items(), key=lambda x: x[1])[0]
            was_correct = (predicted_type == piece_type)
            self.prediction_history.append((prediction.copy(), piece_type, was_correct))
            self.accuracy_by_piece_type[piece_type].append(was_correct)
            
            # Track AAREN prediction accuracy specifically
            self.aaren_predictions_total += 1
            if was_correct:
                self.aaren_predictions_correct += 1
        
        # Set belief to 1.0 for revealed type
        self.belief_distributions[pos] = {
            pt: 1.0 if pt == piece_type else 0.0 for pt in PieceType
        }
        self._update_belief_tensor(pos)
        
        # Apply piece count constraints to other positions
        for other_pos in self.belief_distributions:
            if other_pos != pos and other_pos not in self.revealed_pieces:
                self._apply_piece_count_constraints(other_pos)
                
        # Collect data for AAREN training
        if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
            action_sequence = list(self.piece_action_history[pos])
            self.aaren_training_buffer.append((action_sequence, piece_type, pos))
    
    def get_evaluator_feedback(self, pos: Tuple[int, int], 
                               ground_truth: Optional[PieceType] = None) -> Optional[Dict]:
        """Get feedback from PBS evaluator on prediction quality."""
        if self.evaluator is None or pos not in self.belief_distributions:
            return None
        
        pbs_prediction = self.belief_distributions[pos]
        return self.evaluator.get_feedback(pbs_prediction, ground_truth)
    
    def train_evaluator(self, epochs: int = 1) -> Optional[float]:
        """Train the PBS evaluator on collected data."""
        if self.evaluator is None:
            return None
        
        self.evaluator.train_feature_importance(epochs=epochs)
        evaluator_loss = self.evaluator.train(epochs=epochs)
        
        # Train AAREN model if enough data
        if len(self.aaren_training_buffer) >= 32:
            batch_size = 64
            if len(self.aaren_training_buffer) > batch_size:
                batch = random.sample(self.aaren_training_buffer, batch_size)
            else:
                batch = list(self.aaren_training_buffer)
            
            sequences = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            positions = [item[2] for item in batch]
            
            aaren_epochs = max(1, epochs // 2)
            self.train_aaren(
                action_sequences=sequences,
                true_piece_types=labels,
                epochs=aaren_epochs,
                positions=positions
            )
            
        return evaluator_loss
    
    def train_aaren(self, action_sequences: List[List[List[float]]], 
                   true_piece_types: List[PieceType], 
                   epochs: int = 1,
                   positions: Optional[List[Tuple[int, int]]] = None) -> Optional[float]:
        """Train the AAREN model on collected action sequences. Returns final loss."""
        if not action_sequences or not self.aaren_model:
            return None
            
        # Prepare batch data
        max_len = max(len(seq) for seq in action_sequences)
        batch_size = len(action_sequences)
        if batch_size == 0 or max_len == 0:
            return None
            
        input_size = len(action_sequences[0][0])
        
        # Create padded tensor
        x_batch = torch.zeros(batch_size, max_len, input_size, device=self.device)
        for i, seq in enumerate(action_sequences):
            seq_len = len(seq)
            if seq_len > 0:
                x_batch[i, :seq_len, :] = torch.tensor(seq, dtype=torch.float32, device=self.device)
                
        # Create labels tensor
        y_batch = torch.tensor([pt.value for pt in true_piece_types], dtype=torch.long, device=self.device)
        
        # Get weights if positions provided
        weights = None
        if positions and self._aaren_training_weights:
            weights = torch.ones(batch_size, device=self.device)
            for i, pos in enumerate(positions):
                if pos in self._aaren_training_weights:
                    weights[i] = self._aaren_training_weights[pos]
        
        # Training loop
        self.aaren_model.train()
        final_loss = None
        for _ in range(epochs):
            self.aaren_optimizer.zero_grad()
            
            logits = self.aaren_model(x_batch)
            
            if weights is not None:
                loss = F.cross_entropy(logits, y_batch, reduction='none')
                loss = (loss * weights).mean()
            else:
                loss = F.cross_entropy(logits, y_batch)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.aaren_model.parameters(), max_norm=1.0)
            self.aaren_optimizer.step()
            final_loss = loss.item()
            
        self.aaren_model.eval()
        
        # Track the final loss
        if final_loss is not None:
            self.aaren_training_losses.append(final_loss)
            # Keep only last 1000 losses to avoid memory bloat
            if len(self.aaren_training_losses) > 1000:
                self.aaren_training_losses = self.aaren_training_losses[-1000:]
        
        return final_loss
    
    def get_uncertain_positions(self) -> set:
        """Get positions that need more information gathering."""
        return self.uncertain_positions.copy()
    
    def get_bias_summary(self) -> Optional[Dict[str, float]]:
        """Get summary of detected PBS biases from evaluator."""
        if self.evaluator is None:
            return None
        return self.evaluator.bias_tracker.get_bias_summary()
    
    def get_aaren_accuracy(self) -> float:
        """Get AAREN prediction accuracy (0.0 to 1.0)."""
        if self.aaren_predictions_total == 0:
            return 0.0
        return self.aaren_predictions_correct / self.aaren_predictions_total
    
    def get_aaren_avg_loss(self, window: int = 50) -> float:
        """Get average AAREN training loss over recent window."""
        if not self.aaren_training_losses:
            return 0.0
        recent = self.aaren_training_losses[-window:]
        return sum(recent) / len(recent)
    
    def get_aaren_buffer_size(self) -> int:
        """Get current AAREN training buffer size."""
        return len(self.aaren_training_buffer)
    
    def get_belief_distribution(self, pos: Tuple[int, int]) -> Dict[PieceType, float]:
        """Get the belief distribution for a piece at a given position."""
        return self.belief_distributions[pos].copy()
    
    def get_expected_value(self, pos: Tuple[int, int]) -> float:
        """Get the expected piece value based on belief distribution."""
        beliefs = self.belief_distributions[pos]
        expected_value = 0.0
        
        for piece_type, confidence in beliefs.items():
            rank = PIECE_RANKS.get(piece_type, 0)
            expected_value += confidence * rank
        
        return expected_value
    
    def get_confidence_scores(self, pos: Tuple[int, int]) -> torch.Tensor:
        """Get confidence scores for all piece types at a position."""
        beliefs = self.belief_distributions[pos]
        piece_types = list(PieceType)
        scores = torch.zeros(NUM_PIECE_TYPES, device=self.device)
        
        for i, piece_type in enumerate(piece_types):
            scores[i] = beliefs.get(piece_type, 0.0)
        
        return scores
    
    def get_calibrated_confidence(self, pos: Tuple[int, int], piece_type: PieceType) -> float:
        """Get calibrated confidence score for a piece type at a position."""
        if pos not in self.belief_distributions:
            return 0.0
        
        beliefs = self.belief_distributions[pos]
        base_confidence = beliefs.get(piece_type, 0.0)
        
        if piece_type in self.accuracy_by_piece_type:
            accuracy_history = self.accuracy_by_piece_type[piece_type]
            if len(accuracy_history) > 0:
                accuracy_rate = sum(accuracy_history) / len(accuracy_history)
                calibration_factor = 0.5 + 0.5 * accuracy_rate
                calibrated_confidence = base_confidence * calibration_factor
                return float(min(1.0, calibrated_confidence))
        
        return float(base_confidence)
    
    def get_uncertainty_map(self, game_state, board_size: int = 10) -> Dict[Tuple[int, int], float]:
        """Get uncertainty map for all positions on the board."""
        try:
            from training_config import PBS_CACHE_UNCERTAINTY
        except ImportError:
            PBS_CACHE_UNCERTAINTY = True
            
        if PBS_CACHE_UNCERTAINTY and not self._uncertainty_map_dirty and self._cached_uncertainty_map:
            return self._cached_uncertainty_map
        
        uncertainty_map = {}
        
        for r in range(board_size):
            for c in range(board_size):
                pos = (r, c)
                if pos in self.belief_distributions:
                    beliefs = self.belief_distributions[pos]
                    if beliefs:
                        probs = [p for p in beliefs.values() if p > 0]
                        if probs:
                            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                            max_entropy = math.log(len(beliefs))
                            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                            uncertainty_map[pos] = float(normalized_entropy)
                        else:
                            uncertainty_map[pos] = 1.0
                    else:
                        uncertainty_map[pos] = 1.0
                else:
                    if hasattr(game_state, 'board'):
                        board = game_state.board
                        if isinstance(board, torch.Tensor):
                            piece_val = board[r, c].item()
                        else:
                            piece_val = board[r][c]
                        
                        from board import HIDDEN_PIECE
                        if piece_val == HIDDEN_PIECE or (self.player_id == 1 and piece_val < 0) or (self.player_id == -1 and piece_val > 0):
                            uncertainty_map[pos] = 1.0
                        else:
                            uncertainty_map[pos] = 0.0
        
        if PBS_CACHE_UNCERTAINTY:
            self._cached_uncertainty_map = uncertainty_map
            self._uncertainty_map_dirty = False
        
        return uncertainty_map
    
    def get_position_uncertainty(self, pos: Tuple[int, int]) -> float:
        """Get uncertainty for a specific position."""
        if pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            if beliefs:
                probs = [p for p in beliefs.values() if p > 0]
                if probs:
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    max_entropy = math.log(len(beliefs))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                    return float(normalized_entropy)
        return 1.0
    
    def _initialize_position_priors(self, pos: Tuple[int, int]):
        """Initialize beliefs for a new position with priors."""
        r, c = pos
        beliefs = {}
        
        is_back_row = (r <= 2) if self.player_id == -1 else (r >= 7)
        
        for piece_type in PieceType:
            prob = 1.0 / len(PieceType)
            
            if is_back_row:
                if piece_type in [PieceType.FLAG, PieceType.BOMB]:
                    prob *= 2.0
                elif piece_type == PieceType.SCOUT:
                    prob *= 0.5
            
            beliefs[piece_type] = prob
        
        total = sum(beliefs.values())
        for pt in beliefs:
            beliefs[pt] /= total
            
        self.belief_distributions[pos] = beliefs
        self._update_belief_tensor(pos)


    def _apply_rule_based_inference(self, pos: Tuple[int, int], 
                                   action: Tuple[Tuple[int, int], Tuple[int, int]]):
        """Apply hard rules to update beliefs."""
        (r_from, c_from), (r_to, c_to) = action
        dist = abs(r_to - r_from) + abs(c_to - c_from)
        
        # Rule 1: If piece moves > 1 square, it MUST be a Scout
        if dist > 1:
            self.belief_distributions[pos] = {
                pt: 1.0 if pt == PieceType.SCOUT else 0.0 for pt in PieceType
            }
            self._update_belief_tensor(pos)
            return

        # Rule 2: Bombs and Flags cannot move
        if dist > 0:
            beliefs = self.belief_distributions[pos]
            beliefs[PieceType.BOMB] = 0.0
            beliefs[PieceType.FLAG] = 0.0
            
            total = sum(beliefs.values())
            if total > 0:
                for pt in beliefs:
                    beliefs[pt] /= total
            else:
                self._initialize_position_priors(pos)
            
            self._update_belief_tensor(pos)


    def _apply_behavioral_patterns(self, pos: Tuple[int, int], 
                                   action: Tuple[Tuple[int, int], Tuple[int, int]],
                                   game_state):
        """Apply soft behavioral patterns to update beliefs."""
        (r_from, c_from), (r_to, c_to) = action
        beliefs = self.belief_distributions[pos]
        
        is_advance = (r_to < r_from) if self.player_id == 1 else (r_to > r_from)
        
        if is_advance:
            for pt in [PieceType.MINER, PieceType.GENERAL, PieceType.MARSHAL]:
                if pt in beliefs:
                    beliefs[pt] *= 1.2
        
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total
        
        self._update_belief_tensor(pos)


    def _apply_piece_count_constraints(self, pos: Tuple[int, int]):
        """Constrain beliefs based on known remaining pieces."""
        revealed_counts = self.revealed_piece_counts.copy()
        
        beliefs = self.belief_distributions[pos]
        for pt in PieceType:
            if revealed_counts.get(pt, 0) >= self.total_piece_counts.get(pt, 0):
                beliefs[pt] = 0.0
        
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total
        
        self._update_belief_tensor(pos)


    def _apply_aaren_inference(self, pos: Tuple[int, int]):
        """Apply AAREN model inference to update beliefs."""
        if self.aaren_model is None:
            return
            
        history = list(self.piece_action_history.get(pos, []))
        if not history:
            return
            
        seq_len = min(len(history), 10)
        sequence = history[-seq_len:]
        
        seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.aaren_model(seq_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        aaren_weight = min(0.8, len(history) * 0.1)
        current_weight = 1.0 - aaren_weight
        
        beliefs = self.belief_distributions[pos]
        piece_types = list(PieceType)
        
        for i, pt in enumerate(piece_types):
            aaren_prob = probs[i].item()
            current_prob = beliefs.get(pt, 0.0)
            beliefs[pt] = current_weight * current_prob + aaren_weight * aaren_prob
            
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total
        
        self._update_belief_tensor(pos)


    def _update_piece_coordination(self, pos: Tuple[int, int],
                                   action: Tuple[Tuple[int, int], Tuple[int, int]],
                                   game_state):
        """Track multi-piece coordination patterns."""
        (r_from, c_from), (r_to, c_to) = action
        
        nearby_pieces = []
        for other_pos in self.belief_distributions:
            if other_pos == pos or other_pos in self.revealed_pieces:
                continue
            
            other_r, other_c = other_pos
            distance = abs(other_r - r_from) + abs(other_c - c_from)
            
            if distance <= 2:
                nearby_pieces.append(other_pos)
        
        if len(nearby_pieces) >= 2:
            self.piece_coordination[pos].extend(nearby_pieces[:3])
    
    def get_multi_channel_state(self, game_state, board_size: int = 10) -> torch.Tensor:
        """Get a multi-channel tensor representation of the game state."""
        if not hasattr(game_state, 'board'):
            return torch.zeros((15, board_size, board_size), device=self.device)
        
        board = game_state.board
        if not isinstance(board, torch.Tensor):
            board = torch.tensor(board, device=self.device)
        else:
            board = board.to(self.device)
            
        channels = 15
        state_tensor = torch.zeros((channels, board_size, board_size), device=self.device, dtype=torch.float32)
        
        lakes_mask = (board == -13)
        state_tensor[1] = lakes_mask.float()
        
        if self.player_id == 1:
            own_pieces_mask = (board > 0)
            state_tensor[0][own_pieces_mask] = board[own_pieces_mask].float()
        else:
            own_pieces_mask = (board < 0) & (board != -13) & (board != -20)
            state_tensor[0][own_pieces_mask] = board[own_pieces_mask].abs().float()
            
        if hasattr(self, 'belief_tensor'):
            state_tensor[2:14] = self.belief_tensor
            
        if self.player_id == 1:
            enemy_mask = (board < 0) & (board != -13)
        else:
            enemy_mask = (board > 0)
            
        state_tensor[14][enemy_mask] = 1.0
        
        for pos in self.revealed_pieces:
            r, c = pos
            state_tensor[14, r, c] = 0.0
            
        belief_sum = state_tensor[2:14].sum(dim=0)
        missing_belief_mask = enemy_mask & (belief_sum < 0.01)
        
        if missing_belief_mask.any():
            uniform_prob = 1.0 / 12.0
            for i in range(12):
                state_tensor[2+i][missing_belief_mask] = uniform_prob
                            
        return state_tensor
    
    def train_aaren_with_evaluator_feedback(self, epochs: int = 10):
        """Train AAREN using stored action sequences and evaluator feedback weights."""
        action_sequences = []
        true_piece_types = []
        evaluator_weights = []
        positions = []
        
        for pos, piece_type in self.revealed_pieces.items():
            if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
                action_sequences.append(list(self.piece_action_history[pos]))
                true_piece_types.append(piece_type)
                positions.append(pos)
                weight = self._aaren_training_weights.get(pos, 1.0)
                evaluator_weights.append(weight)
        
        if len(action_sequences) > 0:
            self.train_aaren(
                action_sequences=action_sequences,
                true_piece_types=true_piece_types,
                epochs=epochs,
                evaluator_weights=evaluator_weights,
                positions=positions
            )
    
    def get_aaren_training_data(self) -> Tuple[List, List, List, List]:
        """Collect training data for shared AAREN training."""
        action_sequences = []
        true_piece_types = []
        evaluator_weights = []
        positions = []
        
        for pos, piece_type in self.revealed_pieces.items():
            if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
                action_sequences.append(list(self.piece_action_history[pos]))
                true_piece_types.append(piece_type)
                positions.append(pos)
                weight = self._aaren_training_weights.get(pos, 1.0)
                evaluator_weights.append(weight)
                
        return action_sequences, true_piece_types, evaluator_weights, positions
    
    def save_aaren_model(self, filepath: str):
        """Save the Aaren model separately."""
        torch.save({
            'aaren_state_dict': self.aaren_model.state_dict(),
            'aaren_optimizer_state_dict': self.aaren_optimizer.state_dict(),
        }, filepath)
    
    def load_aaren_model(self, filepath: str):
        """Load the Aaren model separately."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.aaren_model.load_state_dict(checkpoint['aaren_state_dict'])
        self.aaren_optimizer.load_state_dict(checkpoint['aaren_optimizer_state_dict'])
    
    def state_dict(self) -> Dict:
        """Returns a dictionary containing the whole state of the module."""
        state = {}
        if self.aaren_model:
            state['aaren_model'] = self.aaren_model.state_dict()
        if self.aaren_optimizer:
            state['aaren_optimizer'] = self.aaren_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict):
        """Copies parameters and buffers from state_dict into this module."""
        if 'aaren_model' in state_dict and self.aaren_model:
            self.aaren_model.load_state_dict(state_dict['aaren_model'])
        if 'aaren_optimizer' in state_dict and self.aaren_optimizer:
            self.aaren_optimizer.load_state_dict(state_dict['aaren_optimizer'])
