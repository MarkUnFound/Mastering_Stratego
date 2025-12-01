# stratego_modular/probabilistic_belief_state.py

"""
Probabilistic Belief State (PBS) for Stratego
Tracks piece actions and infers possible piece types/values using:
1. Rule-based inference (e.g., multi-tile moves = Scout)
2. AAREN-based pattern learning from action sequences
3. Confidence scores for each possible piece value

AAREN (Attention as a Recurrent Neural Network) provides:
- Parallel training (like Transformers)
- Efficient O(1) inference updates (like RNNs)
- Constant memory usage
- No vanishing gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from piece import PieceType, PIECE_RANKS, NUM_PIECE_TYPES

# Check if PBS evaluator is available
try:
    from pbs_evaluator import PBSEvaluator
    PBS_EVALUATOR_AVAILABLE = True
except ImportError:
    PBS_EVALUATOR_AVAILABLE = False



# 1. Define the JIT-compiled scan kernel
@torch.jit.script
def aaren_scan_kernel(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT-compiled kernel for Hillis-Steele parallel scan.
    Input shapes: 
      k, v: (batch, seq_len, hidden)
      q: (hidden)
    """
    batch_size, seq_len, hidden_size = k.size()
    
    # Pre-compute scores s = q * k
    # q is (hidden), k is (B, L, H) -> broadcast multiply then sum
    s = (k * q).sum(dim=2, keepdim=True) # (B, L, 1)

    # Initialize state (m, u, w)
    # Using the correct initialization mapping from the paper:
    # m_i = s_i 
    # u_i = v_i  (Variable name 'u' in code maps to 'w' in paper)
    # w_i = 1    (Variable name 'w' in code maps to 'u' in paper)
    
    curr_m = s
    curr_u = v
    curr_w = torch.ones_like(s)

    # Prepare identity elements for padding
    # Max identity = -inf
    # Sum identity = 0
    pad_m_val = -float('inf')
    
    # Calculate number of steps: ceil(log2(L))
    # For L=20, steps=5
    # math.log2 is not supported in TorchScript, use change of base
    num_steps = int(math.ceil(math.log(float(seq_len)) / math.log(2.0)))
    
    for i in range(num_steps):
        offset = 1 << i  # 2^i
        
        # -----------------------------------------------------------
        # Optimization: View-based slicing instead of creating new tensors
        # where possible, though JIT optimizes cat significantly.
        # -----------------------------------------------------------
        
        # Create shifted tensors (Efficient padding logic)
        # Shift Right by offset
        
        # 1. Pad m (Max)
        # Construct prev_m: [ -inf padding, curr_m[:-offset] ]
        pad_m = torch.full((batch_size, offset, 1), pad_m_val, device=curr_m.device, dtype=curr_m.dtype)
        prev_m = torch.cat((pad_m, curr_m[:, :-offset, :]), dim=1)

        # 2. Pad u (Value Sums)
        pad_u = torch.zeros((batch_size, offset, hidden_size), device=curr_u.device, dtype=curr_u.dtype)
        prev_u = torch.cat((pad_u, curr_u[:, :-offset, :]), dim=1)

        # 3. Pad w (Normalization Sums)
        pad_w = torch.zeros((batch_size, offset, 1), device=curr_w.device, dtype=curr_w.dtype)
        prev_w = torch.cat((pad_w, curr_w[:, :-offset, :]), dim=1)

        # -----------------------------------------------------------
        # Associative Operator Core
        # -----------------------------------------------------------
        
        # Update Max
        m_new = torch.maximum(prev_m, curr_m)
        
        # Stable Exponentials
        # exp(prev - new) and exp(curr - new)
        exp_prev = torch.exp(prev_m - m_new)
        exp_curr = torch.exp(curr_m - m_new)
        
        # Update Weighted Sums
        curr_u = prev_u * exp_prev + curr_u * exp_curr
        curr_w = prev_w * exp_prev + curr_w * exp_curr
        curr_m = m_new

    # Final division output
    return curr_u, curr_w, curr_m


class AarenCell(nn.Module):
    """
    Aaren (Attention as a Recurrent Neural Network) Cell.
    
    Implements attention-based recurrent computation with:
    - State: (a_t, c_t, m_t) where:
      - a_t: Weighted sum of encoded values
      - c_t: Normalization constant
      - m_t: Cumulative maximum (for numerical stability)
    - Query vector q: Learned parameter (what to attend to)
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize Aaren cell.
        
        Args:
            hidden_size: Hidden state dimension
        """
        super(AarenCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Learned query vector (what to attend to)
        self.q = nn.Parameter(torch.randn(hidden_size))
        
        # Key and value projections
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x_t: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass for single observation (sequential inference).
        
        Args:
            x_t: Current observation (batch, hidden_size)
            prev_state: Previous state (a_{t-1}, c_{t-1}, m_{t-1}) or None for initial state
            
        Returns:
            output: Normalized attention output (batch, hidden_size)
            new_state: Updated state (a_t, c_t, m_t)
        """
        batch_size = x_t.size(0)
        
        # Compute key and value
        k_t = self.W_k(x_t)  # (batch, hidden_size)
        v_t = self.W_v(x_t)  # (batch, hidden_size)
        
        # Compute attention score: s_t = dot(q, k_t)
        s_t = torch.sum(self.q.unsqueeze(0) * k_t, dim=1, keepdim=True)  # (batch, 1)
        
        if prev_state is None:
            # Initial state
            # For numerical stability, we track m_t = max(s_1...s_t)
            # At t=1, m_t = s_t
            # a_t = v_t * exp(s_t - m_t) = v_t * 1
            # c_t = exp(s_t - m_t) = 1
            m_t = s_t  # (batch, 1)
            a_t = v_t  # (batch, hidden_size)
            c_t = torch.ones_like(s_t)  # (batch, 1)
        else:
            a_prev, c_prev, m_prev = prev_state
            
            # Update cumulative maximum
            m_t = torch.maximum(m_prev, s_t)  # (batch, 1)
            
            # Update weighted sum and normalization with numerical stability
            exp_prev = torch.exp(m_prev - m_t)  # (batch, 1)
            exp_curr = torch.exp(s_t - m_t)  # (batch, 1)
            
            a_t = a_prev * exp_prev + v_t * exp_curr  # (batch, hidden_size)
            c_t = c_prev * exp_prev + exp_curr  # (batch, 1)
        
        # Normalize output
        output = a_t / (c_t + 1e-8)  # (batch, hidden_size)
        
        return output, (a_t, c_t, m_t)


class PieceActionAaren(nn.Module):
    """
    AAREN-based network to learn piece value patterns from action sequences.
    
    AAREN provides:
    - Parallel training (no sequential bottleneck)
    - Efficient O(1) inference updates
    - Constant memory usage
    - Better gradient flow
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = NUM_PIECE_TYPES, device=None):
        """
        Initialize Aaren for piece value inference.
        
        Args:
            input_size: Size of action feature vector (8 features)
            hidden_size: Hidden state size
            num_layers: Number of Aaren layers
            output_size: Number of possible piece types
            device: PyTorch device
        """
        super(PieceActionAaren, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection to hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Stack of Aaren cells (one per layer)
        self.aaren_cells = nn.ModuleList([
            AarenCell(hidden_size) for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward_parallel(self, x: torch.Tensor):
        """
        Forward pass for training (parallel processing of full sequences).
        
        Uses parallel prefix scan algorithm for efficient parallel computation.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, sequence_length, output_size) with piece type logits
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden size
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Process through each Aaren layer
        for layer_idx, (aaren_cell, layer_norm) in enumerate(zip(self.aaren_cells, self.layer_norms)):
            # Parallel prefix scan for this layer
            h = self._parallel_prefix_scan(h, aaren_cell, layer_norm)
        
        # Use last timestep for prediction
        last_hidden = h[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return logits (softmax applied in loss function)
        return x
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass (defaults to parallel mode for training).
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size) with piece type probabilities
        """
        logits = self.forward_parallel(x)
        return F.softmax(logits, dim=1)
    
    def forward_sequential(self, x_t: torch.Tensor, prev_states: Optional[List[Tuple]] = None):
        """
        Forward pass for inference (sequential processing, O(1) per timestep).
        
        Args:
            x_t: Current observation (batch, input_size) or (batch, 1, input_size)
            prev_states: List of previous states for each layer, or None for initial state
            
        Returns:
            output: Piece type probabilities (batch, output_size)
            new_states: Updated states for each layer
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)  # (batch, input_size)
        
        batch_size = x_t.size(0)
        
        # Project input
        h = self.input_proj(x_t)  # (batch, hidden_size)
        
        new_states = []
        
        # Process through each Aaren layer sequentially
        for layer_idx, (aaren_cell, layer_norm) in enumerate(zip(self.aaren_cells, self.layer_norms)):
            prev_state = prev_states[layer_idx] if prev_states is not None else None
            
            # Sequential update
            h, new_state = aaren_cell(h, prev_state)
            h = layer_norm(h)
            new_states.append(new_state)
        
        # Fully connected layers
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Softmax to get probabilities
        probs = F.softmax(x, dim=1)
        
        return probs, new_states
    
    def _parallel_prefix_scan(self, h: torch.Tensor, aaren_cell: AarenCell, layer_norm: nn.LayerNorm):
        """
        Optimized version using JIT kernel and batched projections.
        """
        # 1. Compute Projections (Batched MatMul is highly optimized)
        k = aaren_cell.W_k(h)
        v = aaren_cell.W_v(h)
        
        # 2. Run JIT Scan
        # Note: We pass aaren_cell.q directly
        u_final, w_final, _ = aaren_scan_kernel(k, v, aaren_cell.q)
        
        # 3. Finalize
        output = u_final / (w_final + 1e-8)
        output = layer_norm(output)
        
        return output


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
        
    def reset(self):
        """Reset the belief state for a new game."""
        self.piece_action_history.clear()
        self.belief_distributions.clear()
        self.revealed_pieces.clear()
        self.aaren_states.clear()  # Clear Aaren states
        self._aaren_training_positions.clear()  # Clear training positions
        self.revealed_piece_counts = {pt: 0 for pt in PieceType}
        self.piece_observation_times.clear()
        self.turn_count = 0
        self.piece_coordination.clear()
        self.uncertain_positions.clear()
        self.prediction_history.clear()
        self.accuracy_by_piece_type.clear()
    
    def _extract_action_features(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                                 game_state, pos: Optional[Tuple[int, int]] = None,
                                 apply_feature_weights: bool = True) -> np.ndarray:
        """
        Extract enhanced features from an action for Aaren input.
        
        Original Features (0-7):
        0: Move distance (tiles)
        1: Is attack (1 if attacking, 0 otherwise)
        2: Direction (0=N, 1=S, 2=E, 3=W)
        3: Distance from center
        4: Is forward move (toward enemy)
        5: Is backward move (away from enemy)
        6: Is lateral move
        7: Aggressiveness score (based on position and action)
        
        Enhanced Features (8-23):
        8: Piece value estimate (from current beliefs)
        9: Confidence in prediction (1 - entropy)
        10: Number of previous moves by this piece
        11: Time since piece was first observed (normalized)
        12: Position row (normalized)
        13: Position column (normalized)
        14: Distance to own flag (normalized)
        15: Distance to enemy flag (normalized)
        16: Number of adjacent friendly pieces
        17: Number of adjacent enemy pieces
        18: Is piece in enemy territory
        19: Is piece in center (rows 4-5)
        20: Game phase (early=0, mid=0.5, end=1.0)
        21: Turn count (normalized)
        22: Piece mobility estimate
        23: Threat level (adjacent enemies)
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Original features (0-7)
        distance = max(abs(r_to - r_from), abs(c_to - c_from))
        
        is_attack = 0.0
        if hasattr(game_state, 'board'):
            board = game_state.board
            if isinstance(board, torch.Tensor):
                target_val = board[r_to, c_to].item()
                if self.player_id == 1:
                    is_attack = 1.0 if target_val < 0 else 0.0
                else:
                    is_attack = 1.0 if target_val > 0 else 0.0
        
        if r_to > r_from:
            direction = 1.0  # South
        elif r_to < r_from:
            direction = 0.0  # North
        elif c_to > c_from:
            direction = 2.0  # East
        else:
            direction = 3.0  # West
        
        center_r, center_c = 4.5, 4.5
        dist_from_center = np.sqrt((r_to - center_r)**2 + (c_to - center_c)**2) / 10.0
        
        if self.player_id == 1:
            # Player 1 is at bottom (rows 6-9), moves UP (decreasing r) to advance
            is_forward = 1.0 if r_to < r_from else 0.0
            is_backward = 1.0 if r_to > r_from else 0.0
        else:
            # Player 2 is at top (rows 0-3), moves DOWN (increasing r) to advance
            is_forward = 1.0 if r_to > r_from else 0.0
            is_backward = 1.0 if r_to < r_from else 0.0
        
        is_lateral = 1.0 if r_from == r_to or c_from == c_to else 0.0
        aggressiveness = is_attack * 0.5
        if is_forward:
            aggressiveness += 0.3
        if distance == 0:
            aggressiveness += 0.2
        
        # Enhanced features (8-23)
        if pos and pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            # Feature 8: Piece value estimate
            piece_value_estimate = sum(PIECE_RANKS.get(pt, 0) * conf for pt, conf in beliefs.items()) / 12.0
            
            # Feature 9: Confidence (1 - entropy)
            entropy = -sum(conf * np.log(conf + 1e-10) for conf in beliefs.values())
            max_entropy = np.log(len(beliefs))
            confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            # Feature 10: Number of previous moves
            num_moves = len(self.piece_action_history.get(pos, [])) / 20.0  # Normalize by max history
            
            # Feature 11: Time since first observed
            if pos in self.piece_observation_times:
                time_since_observed = (self.turn_count - self.piece_observation_times[pos]) / 500.0  # Normalize
            else:
                time_since_observed = 0.0
        else:
            piece_value_estimate = 0.5  # Default
            confidence = 0.0
            num_moves = 0.0
            time_since_observed = 0.0
        
        # Feature 12-13: Position
        pos_row = r_from / 10.0
        pos_col = c_from / 10.0
        
        # Feature 14-15: Distance to flags (simplified - would need flag positions)
        dist_to_own_flag = 0.5  # Placeholder
        dist_to_enemy_flag = 0.5  # Placeholder
        
        # Feature 16-17: Adjacent pieces (simplified - would need board access)
        adjacent_friendly = 0.0
        adjacent_enemy = 0.0
        
        # Feature 18: Is in enemy territory
        if self.player_id == 1:
            is_in_enemy_territory = 1.0 if r_from <= 3 else 0.0
        else:
            is_in_enemy_territory = 1.0 if r_from >= 6 else 0.0
        
        # Feature 19: Is in center
        is_in_center = 1.0 if 4 <= r_from <= 5 else 0.0
        
        # Feature 20: Game phase
        if hasattr(game_state, 'turn_count'):
            turn = game_state.turn_count
        else:
            turn = self.turn_count
        if turn < 50:
            game_phase = 0.0  # Early
        elif turn < 200:
            game_phase = 0.5  # Mid
        else:
            game_phase = 1.0  # End
        
        # Feature 21: Turn count
        turn_count_norm = turn / 500.0
        
        # Feature 22: Mobility estimate (simplified)
        mobility_estimate = 1.0 - (distance / 10.0) if distance > 0 else 0.5
        
        # Feature 23: Threat level (simplified)
        threat_level = is_attack * 0.5 + (adjacent_enemy / 4.0)
        
        features = np.array([
            distance / 10.0,  # 0
            is_attack,  # 1
            direction / 3.0,  # 2
            dist_from_center,  # 3
            is_forward,  # 4
            is_backward,  # 5
            is_lateral,  # 6
            aggressiveness,  # 7
            piece_value_estimate,  # 8
            confidence,  # 9
            num_moves,  # 10
            time_since_observed,  # 11
            pos_row,  # 12
            pos_col,  # 13
            dist_to_own_flag,  # 14
            dist_to_enemy_flag,  # 15
            adjacent_friendly,  # 16
            adjacent_enemy,  # 17
            is_in_enemy_territory,  # 18
            is_in_center,  # 19
            game_phase,  # 20
            turn_count_norm,  # 21
            mobility_estimate,  # 22
            threat_level  # 23
        ], dtype=np.float32)
        
        # Apply feature importance weighting if evaluator is available
        if apply_feature_weights and self.evaluator is not None:
            try:
                importance_weights = self.evaluator.get_feature_importance(features)
                # Ensure weights match features shape
                if importance_weights.shape == features.shape:
                    features = features * importance_weights
            except Exception:
                # If feature importance fails, use unweighted features
                pass
        
        return features
    
    def update_from_action(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                          game_state, acting_player: int, q_value: Optional[float] = None):
        """
        Update belief state based on an action.
        
        Args:
            action: The action taken ((from_pos), (to_pos))
            game_state: Current game state
            acting_player: Player who took the action
            q_value: Q-value of the action (optional, for evaluator feedback)
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
        
        # Track observation time if first time seeing this piece
        if pos not in self.piece_observation_times:
            self.piece_observation_times[pos] = self.turn_count
            
        # Store Q-value if provided (for later evaluator feedback)
        if q_value is not None:
            if not hasattr(self, 'piece_q_value_history'):
                self.piece_q_value_history = {}
            if pos not in self.piece_q_value_history:
                self.piece_q_value_history[pos] = []
            self.piece_q_value_history[pos].append(q_value)
        
        # Initialize beliefs with position-based priors if not already set
        if pos not in self.belief_distributions or len(self.belief_distributions[pos]) == 0:
            self._initialize_position_priors(pos)
        
        # Extract action features (with position for enhanced features)
        action_features = self._extract_action_features(action, game_state, pos=pos)
        self.piece_action_history[pos].append(action_features)
        
        # Rule-based inference
        self._apply_rule_based_inference(pos, action)
        
        # Behavioral pattern recognition
        self._apply_behavioral_patterns(pos, action, game_state)
        
        # Aaren-based inference (sequential update for efficiency)
        if len(self.piece_action_history[pos]) >= 1:
            self._apply_aaren_inference(pos)
        
        # Apply bias correction from evaluator
        if self.evaluator is not None and pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            # Apply correction factors to each piece type
            corrected_beliefs = {}
            for piece_type in beliefs:
                correction = self.evaluator.get_bias_correction(piece_type)
                corrected_beliefs[piece_type] = beliefs[piece_type] * correction
            
            # Normalize to ensure probabilities sum to 1
            total = sum(corrected_beliefs.values())
            if total > 0:
                for piece_type in corrected_beliefs:
                    corrected_beliefs[piece_type] /= total
                self.belief_distributions[pos] = corrected_beliefs
        
        # Check if we need more information (active learning)
        if self.evaluator is not None and pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            if self.evaluator.should_gather_more_info(beliefs):
                self.uncertain_positions.add(pos)
            else:
                self.uncertain_positions.discard(pos)
        
        # Apply piece count constraints
        self._apply_piece_count_constraints(pos)
        
        # Update turn count
        if hasattr(game_state, 'turn_count'):
            self.turn_count = game_state.turn_count
        
        # 3. NEW: Multi-Piece Pattern Recognition
        # Track pieces moving together or coordinating
        self._update_piece_coordination(pos, action, game_state)
        
        # Update position tracking: if piece moved, transfer beliefs to new position
        if (r_from, c_from) != (r_to, c_to):
            # Check if target position has an unknown piece (after move)
            # This will be handled in the next state update, but we can prepare
            # by transferring the belief distribution if the piece moved to an empty square
            if (r_from, c_from) in self.belief_distributions:
                # Store the old position's beliefs temporarily
                old_beliefs = self.belief_distributions.get((r_from, c_from), {})
                # We don't move beliefs here because the move might fail or be a battle
                # The environment update will handle the actual move
                pass
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
        evaluator_feedback = None
        if self.evaluator is not None and pos in self.belief_distributions:
            pbs_prediction = self.belief_distributions[pos].copy()
            # Only collect data from middle/end game (skip early game)
            if game_phase in ['middle', 'end']:
                # Get action features for this position if available
                action_features = None
                if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
                    # Use the most recent action features
                    action_features = self.piece_action_history[pos][-1]
                
                # Get Q-value if available
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
                # Get evaluator feedback for AAREN training
                evaluator_feedback = self.get_evaluator_feedback(pos, ground_truth=piece_type)
        
        # Store evaluator feedback for AAREN training (if available)
        # This will be used when training AAREN on this position's action sequence
        if evaluator_feedback and pos in self.piece_action_history:
            quality_score = evaluator_feedback.get('quality_score', 0.0)
            # Store weight for AAREN training (higher quality = more important)
            # Also consider piece value: high-value pieces get more weight
            from piece import PIECE_RANKS
            piece_value = PIECE_RANKS.get(piece_type, 1)
            # Combine quality score and piece value for training weight
            # Quality score normalized to [0, 1], piece value normalized to [0.5, 1.5]
            # INVERTED LOGIC: Focus on LOW quality (errors) -> High weight
            quality_weight = 1.0 - (1.0 / (1.0 + math.exp(-quality_score / 10.0)))  # [0, 1], low score -> high weight
            value_weight = 0.5 + (piece_value / 12.0)  # [0.5, 1.5]
            combined_weight = quality_weight * value_weight
            self._aaren_training_weights[pos] = combined_weight
            self._aaren_training_positions[pos] = pos
        
        # 9. NEW: Reveal-Based Learning - Track revealed pieces and learn from outcomes
        # CRITICAL: Store the piece_type directly (it's already a PieceType enum from update_from_reveal)
        # Ensure we're storing the correct PieceType, not a converted value
        if not isinstance(piece_type, PieceType):
            # Safety check: if piece_type is not a PieceType enum, convert it
            if isinstance(piece_type, int):
                piece_type = PieceType(piece_type)
            else:
                raise ValueError(f"piece_type must be PieceType enum, got {type(piece_type)}: {piece_type}")
        
        self.revealed_pieces[pos] = piece_type
        self.revealed_piece_counts[piece_type] = self.revealed_piece_counts.get(piece_type, 0) + 1
        
        # Track prediction accuracy for confidence calibration
        if pos in self.belief_distributions:
            prediction = self.belief_distributions[pos]
            # Check if prediction was correct (highest confidence matches ground truth)
            predicted_type = max(prediction.items(), key=lambda x: x[1])[0]
            was_correct = (predicted_type == piece_type)
            self.prediction_history.append((prediction.copy(), piece_type, was_correct))
            self.accuracy_by_piece_type[piece_type].append(was_correct)
        
        # Set belief to 1.0 for revealed type, 0.0 for others
        self.belief_distributions[pos] = {
            pt: 1.0 if pt == piece_type else 0.0 for pt in PieceType
        }
        
        # Apply piece count constraints to all other positions
        for other_pos in self.belief_distributions:
            if other_pos != pos and other_pos not in self.revealed_pieces:
                self._apply_piece_count_constraints(other_pos)
                
        # 10. NEW: Collect data for AAREN training
        # If we have action history for this piece, add it to the training buffer
        if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
            # Convert deque to list for storage
            action_sequence = list(self.piece_action_history[pos])
            self.aaren_training_buffer.append((action_sequence, piece_type, pos))
    
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
        
        # Train feature importance network
        self.evaluator.train_feature_importance(epochs=epochs)
        # Train main evaluator network
        evaluator_loss = self.evaluator.train(epochs=epochs)
        
        # 11. NEW: Train AAREN model if we have enough data
        # We train AAREN whenever we train the evaluator
        if len(self.aaren_training_buffer) >= 32: # Minimum batch size
            # Sample a batch instead of using everything
            batch_size = 64
            if len(self.aaren_training_buffer) > batch_size:
                batch = random.sample(self.aaren_training_buffer, batch_size)
            else:
                batch = list(self.aaren_training_buffer)
            
            # Extract sequences and labels from batch
            sequences = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            positions = [item[2] for item in batch]
            
            # Get evaluator weights if available
            evaluator_weights = None
            if self.evaluator is not None:
                # We can get weights based on position quality
                # This is handled inside train_aaren if we pass positions
                pass
                
            # Train AAREN
            # Use fewer epochs for AAREN to prevent overfitting to small buffers
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
                   positions: Optional[List[Tuple[int, int]]] = None):
        """
        Train the AAREN model on collected action sequences.
        
        Args:
            action_sequences: List of action feature sequences
            true_piece_types: List of ground truth piece types
            epochs: Number of training epochs
            positions: Optional list of positions for weighting
        """
        if not action_sequences or not self.aaren_model:
            return
            
        # Prepare batch data
        # Pad sequences to same length
        max_len = max(len(seq) for seq in action_sequences)
        batch_size = len(action_sequences)
        if batch_size == 0 or max_len == 0:
            return
            
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
        for _ in range(epochs):
            self.aaren_optimizer.zero_grad()
            
            # Forward pass (parallel)
            logits = self.aaren_model(x_batch) # (batch, output_size)
            
            # Compute loss
            if weights is not None:
                loss = F.cross_entropy(logits, y_batch, reduction='none')
                loss = (loss * weights).mean()
            else:
                loss = F.cross_entropy(logits, y_batch)
                
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.aaren_model.parameters(), max_norm=1.0)
            self.aaren_optimizer.step()
            
        self.aaren_model.eval()
        
        # Do NOT clear buffer after training (Replay Buffer behavior)
        # self.aaren_training_buffer.clear()
        # self._aaren_training_weights.clear()
        # self._aaren_training_positions.clear()
    
    def get_uncertain_positions(self) -> set:
        """
        Get positions that need more information gathering (active learning).
        
        Returns:
            Set of positions that are uncertain and should be prioritized for observation
        """
        return self.uncertain_positions.copy()
    
    def get_bias_summary(self) -> Optional[Dict[str, float]]:
        """
        Get summary of detected PBS biases from evaluator.
        
        Returns:
            Dictionary mapping piece type names to correction factors, or None if evaluator not available
        """
        if self.evaluator is None:
            return None
        return self.evaluator.bias_tracker.get_bias_summary()
    
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
    
    def get_calibrated_confidence(self, pos: Tuple[int, int], piece_type: PieceType) -> float:
        """
        Get calibrated confidence score for a piece type at a position.
        
        Uses historical accuracy to calibrate confidence.
        
        Returns:
            Calibrated confidence score (0.0 to 1.0)
        """
        if pos not in self.belief_distributions:
            return 0.0
        
        beliefs = self.belief_distributions[pos]
        base_confidence = beliefs.get(piece_type, 0.0)
        
        # Calibrate based on historical accuracy for this piece type
        if piece_type in self.accuracy_by_piece_type:
            accuracy_history = self.accuracy_by_piece_type[piece_type]
            if len(accuracy_history) > 0:
                accuracy_rate = sum(accuracy_history) / len(accuracy_history)
                # Adjust confidence based on accuracy: if we're often wrong, reduce confidence
                calibration_factor = 0.5 + 0.5 * accuracy_rate  # Range: 0.5 to 1.0
                calibrated_confidence = base_confidence * calibration_factor
                return float(min(1.0, calibrated_confidence))
        
        return float(base_confidence)
    
    def get_uncertainty_map(self, game_state, board_size: int = 10) -> Dict[Tuple[int, int], float]:
        """
        Get uncertainty map for all positions on the board.
        Uncertainty is calculated as entropy of belief distribution.
        
        Args:
            game_state: Current game state
            board_size: Size of the board
            
        Returns:
            Dictionary mapping positions to uncertainty values (0.0 = certain, 1.0 = uncertain)
        """
        uncertainty_map = {}
        
        for r in range(board_size):
            for c in range(board_size):
                pos = (r, c)
                if pos in self.belief_distributions:
                    beliefs = self.belief_distributions[pos]
                    if beliefs:
                        # Calculate entropy (uncertainty)
                        probs = [p for p in beliefs.values() if p > 0]
                        if probs:
                            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                            max_entropy = math.log(len(beliefs))
                            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                            uncertainty_map[pos] = float(normalized_entropy)
                        else:
                            uncertainty_map[pos] = 1.0  # No beliefs = maximum uncertainty
                    else:
                        uncertainty_map[pos] = 1.0  # Empty beliefs = maximum uncertainty
                else:
                    # Position not in beliefs - check if it's an enemy piece
                    if hasattr(game_state, 'board'):
                        board = game_state.board
                        if isinstance(board, torch.Tensor):
                            piece_val = board[r, c].item()
                        else:
                            piece_val = board[r][c]
                        
                        # If it's an enemy piece (hidden), it has uncertainty
                        from board import HIDDEN_PIECE
                        if piece_val == HIDDEN_PIECE or (self.player_id == 1 and piece_val < 0) or (self.player_id == -1 and piece_val > 0):
                            uncertainty_map[pos] = 1.0  # Unknown enemy piece = maximum uncertainty
                        else:
                            uncertainty_map[pos] = 0.0  # Known piece = no uncertainty
        
        return uncertainty_map
    
    def get_position_uncertainty(self, pos: Tuple[int, int]) -> float:
        """
        Get uncertainty for a specific position.
        
        Args:
            pos: Position tuple (row, col)
            
        Returns:
            Uncertainty value (0.0 = certain, 1.0 = uncertain)
        """
        if pos in self.belief_distributions:
            beliefs = self.belief_distributions[pos]
            if beliefs:
                probs = [p for p in beliefs.values() if p > 0]
                if probs:
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    max_entropy = math.log(len(beliefs))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                    return float(normalized_entropy)
        return 1.0  # No beliefs = maximum uncertainty
    
    def _initialize_position_priors(self, pos: Tuple[int, int]):
        """
        Initialize beliefs for a new position with priors.
        """
        r, c = pos
        beliefs = {}
        
        # Heuristic priors based on position
        # Back rows (0-2 or 7-9) more likely to have Flag/Bombs
        is_back_row = (r <= 2) if self.player_id == -1 else (r >= 7)
        
        for piece_type in PieceType:
            prob = 1.0 / len(PieceType)  # Uniform prior
            
            # Adjust for back row
            if is_back_row:
                if piece_type in [PieceType.FLAG, PieceType.BOMB]:
                    prob *= 2.0
                elif piece_type == PieceType.SCOUT:
                    prob *= 0.5
            
            beliefs[piece_type] = prob
        
        # Normalize
        total = sum(beliefs.values())
        for pt in beliefs:
            beliefs[pt] /= total
            
        self.belief_distributions[pos] = beliefs

    def _apply_rule_based_inference(self, pos: Tuple[int, int], 
                                   action: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Apply hard rules to update beliefs.
        """
        (r_from, c_from), (r_to, c_to) = action
        dist = abs(r_to - r_from) + abs(c_to - c_from)
        
        # Rule 1: If piece moves > 1 square, it MUST be a Scout (2)
        if dist > 1:
            self.belief_distributions[pos] = {
                pt: 1.0 if pt == PieceType.SCOUT else 0.0 for pt in PieceType
            }
            return

        # Rule 2: Bombs and Flags cannot move
        # If it moved, probability of Bomb/Flag is 0
        if dist > 0:
            beliefs = self.belief_distributions[pos]
            beliefs[PieceType.BOMB] = 0.0
            beliefs[PieceType.FLAG] = 0.0
            
            # Renormalize
            total = sum(beliefs.values())
            if total > 0:
                for pt in beliefs:
                    beliefs[pt] /= total
            else:
                # Fallback if all became 0 (shouldn't happen if logic is sound)
                self._initialize_position_priors(pos)

    def _apply_behavioral_patterns(self, pos: Tuple[int, int], 
                                   action: Tuple[Tuple[int, int], Tuple[int, int]],
                                   game_state):
        """
        Apply soft behavioral patterns to update beliefs.
        """
        (r_from, c_from), (r_to, c_to) = action
        beliefs = self.belief_distributions[pos]
        
        # Pattern 1: Aggressive moves (towards enemy side) suggest higher rank
        # Agent 1 starts bottom (6-9), moves up (decreasing r)
        # Agent 2 starts top (0-3), moves down (increasing r)
        is_advance = (r_to < r_from) if self.player_id == 1 else (r_to > r_from)
        
        if is_advance:
            # Slightly increase probability of higher ranks (Miner, General, Marshal)
            for pt in [PieceType.MINER, PieceType.GENERAL, PieceType.MARSHAL]:
                if pt in beliefs:
                    beliefs[pt] *= 1.2
        
        # Renormalize
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total

    def _apply_piece_count_constraints(self, pos: Tuple[int, int]):
        """
        Constrain beliefs based on known remaining pieces.
        """
        # This is computationally expensive to do exactly, so we use a simplified approach
        # If we know all pieces of type X are revealed, prob(X) = 0
        
        # Count revealed pieces
        revealed_counts = self.revealed_piece_counts.copy()
        
        # Max counts per piece type (standard Stratego)
        # Use self.total_piece_counts instead of importing
        
        beliefs = self.belief_distributions[pos]
        for pt in PieceType:
            if revealed_counts.get(pt, 0) >= self.total_piece_counts.get(pt, 0):
                beliefs[pt] = 0.0
        
        # Renormalize
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total

    def _apply_aaren_inference(self, pos: Tuple[int, int]):
        """
        Apply AAREN model inference to update beliefs.
        """
        if self.aaren_model is None:
            return
            
        # Convert deque to list for slicing
        history = list(self.piece_action_history.get(pos, []))
        if not history:
            return
            
        # Prepare input
        # Take last N actions (up to sequence length)
        seq_len = min(len(history), 10)
        sequence = history[-seq_len:]
        
        # Convert to tensor
        seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32, device=self.device).unsqueeze(0) # (1, seq_len, features)
        
        # Inference
        with torch.no_grad():
            logits = self.aaren_model(seq_tensor) # (1, num_classes)
            probs = torch.softmax(logits, dim=1).squeeze(0) # (num_classes)
            
        # Update beliefs (weighted average with current beliefs)
        # We trust AAREN more as sequence grows
        aaren_weight = min(0.8, len(history) * 0.1)
        current_weight = 1.0 - aaren_weight
        
        beliefs = self.belief_distributions[pos]
        piece_types = list(PieceType)
        
        for i, pt in enumerate(piece_types):
            aaren_prob = probs[i].item()
            current_prob = beliefs.get(pt, 0.0)
            beliefs[pt] = current_weight * current_prob + aaren_weight * aaren_prob
            
        # Renormalize
        total = sum(beliefs.values())
        if total > 0:
            for pt in beliefs:
                beliefs[pt] /= total

    def _update_piece_coordination(self, pos: Tuple[int, int],
                                   action: Tuple[Tuple[int, int], Tuple[int, int]],
                                   game_state):
        """
        Track multi-piece coordination patterns.
        
        Updates beliefs based on pieces moving together or coordinating.
        """
        (r_from, c_from), (r_to, c_to) = action
        
        # Find nearby pieces that might be coordinating
        nearby_pieces = []
        for other_pos in self.belief_distributions:
            if other_pos == pos or other_pos in self.revealed_pieces:
                continue
            
            other_r, other_c = other_pos
            distance = abs(other_r - r_from) + abs(other_c - c_from)
            
            if distance <= 2:  # Within 2 squares
                nearby_pieces.append(other_pos)
        
        # If multiple pieces are nearby and moving, might indicate coordination
        if len(nearby_pieces) >= 2:
            # Pieces coordinating might be similar value (e.g., scouts together)
            # or complementary (e.g., strong + weak)
            beliefs = self.belief_distributions[pos]
            
            # Check if nearby pieces have similar movement patterns
            # (simplified - would need more context)
            # For now, just track coordination
            self.piece_coordination[pos].extend(nearby_pieces[:3])  # Keep last 3
    
    def get_multi_channel_state(self, game_state, board_size: int = 10) -> torch.Tensor:
        """
        Get a multi-channel tensor representation of the game state.
        
        Channels:
        0: Own pieces (positive values for rank)
        1: Lakes (binary)
        2-13: Belief probabilities for each piece type (Flag, Spy, Scout... Bomb)
        14: Unknown enemy piece mask
        
        Returns:
            Tensor of shape (15, board_size, board_size)
        """
        if not hasattr(game_state, 'board'):
            return torch.zeros((15, board_size, board_size), device=self.device)
        
        board = game_state.board
        if not isinstance(board, torch.Tensor):
            # Convert to tensor if needed (though it should be a tensor)
            board = torch.tensor(board, device=self.device)
            
        # Initialize multi-channel tensor
        # 15 channels: 1 (Own) + 1 (Lakes) + 12 (Beliefs) + 1 (Unknown Mask)
        channels = 15
        state_tensor = torch.zeros((channels, board_size, board_size), device=self.device, dtype=torch.float32)
        
        # Channel 0: Own pieces and known enemy pieces (ground truth for own, revealed for enemy)
        # We want to represent ranks. 
        # Own pieces: Positive rank (1-12)
        # Lakes: -13 -> Handled in Channel 1
        # Unknown/Hidden: Handled in Beliefs
        
        # Create masks
        lakes_mask = (board == -13)
        
        # Channel 1: Lakes
        state_tensor[1] = lakes_mask.float()
        
        # Channel 0: Known pieces
        # Iterate to fill channel 0 and belief channels
        # Vectorized approach is harder due to dictionary lookups for beliefs, 
        # but we can vectorize the "known" parts.
        
        # Fill Channel 0 with raw board values first, then clean up
        # We only want positive ranks for own pieces here.
        # If player_id == 1, own pieces are > 0.
        # If player_id == -1, own pieces are < 0 (need to take abs).
        
        if self.player_id == 1:
            own_pieces_mask = (board > 0)
            state_tensor[0][own_pieces_mask] = board[own_pieces_mask].float()
        else:
            own_pieces_mask = (board < 0) & (board != -13) & (board != -20) # Exclude lakes/hidden
            state_tensor[0][own_pieces_mask] = board[own_pieces_mask].abs().float()
            
        # Channel 14: Unknown enemy piece mask
        # And Channels 2-13: Beliefs
        
        # Identify unknown enemy pieces
        # If player 1: Enemy pieces are < 0. Unknown if not in revealed_pieces.
        # If player -1: Enemy pieces are > 0. Unknown if not in revealed_pieces.
        
        # We need to iterate to handle the dictionary lookups for beliefs efficiently
        # Since board is small (10x10), a loop is acceptable, but we can optimize if needed.
        
        sorted_piece_types = sorted(PieceType, key=lambda pt: pt.value)
        
        for r in range(board_size):
            for c in range(board_size):
                pos = (r, c)
                piece_val = board[r, c].item()
                
                is_enemy = False
                if self.player_id == 1:
                    if piece_val < 0 and piece_val != -13: # Enemy
                        is_enemy = True
                else:
                    if piece_val > 0: # Enemy
                        is_enemy = True
                        
                if is_enemy:
                    if pos in self.revealed_pieces:
                        # Known enemy piece - put in Channel 0 (as negative rank? or just rank?)
                        # Let's put known enemy pieces in Channel 0 as NEGATIVE rank to distinguish?
                        # Or maybe better: Channel 0 is OWN, Channel 1 is LAKES.
                        # Let's add a channel for KNOWN ENEMY?
                        # The prompt plan said:
                        # 0: Own pieces
                        # 1: Lakes
                        # 2-13: Beliefs (which covers unknown enemy)
                        # 14: Unknown mask
                        
                        # What about KNOWN enemy pieces?
                        # If we know it's a Marshal (10), probability of Marshal is 1.0.
                        # So we can just fill the belief channels with 1.0 for known pieces!
                        
                        # Get the piece type
                        actual_type = self.revealed_pieces[pos]
                        # Find index in sorted_piece_types
                        type_idx = sorted_piece_types.index(actual_type)
                        # Set probability 1.0
                        state_tensor[2 + type_idx, r, c] = 1.0
                        
                    else:
                        # Unknown enemy piece
                        state_tensor[14, r, c] = 1.0
                        
                        # Fill belief channels
                        if pos in self.belief_distributions:
                            beliefs = self.belief_distributions[pos]
                            for i, pt in enumerate(sorted_piece_types):
                                prob = beliefs.get(pt, 0.0)
                                state_tensor[2 + i, r, c] = prob
                        else:
                            # Default uniform belief
                            uniform_prob = 1.0 / 12.0
                            state_tensor[2:14, r, c] = uniform_prob
                            
        return state_tensor
    
    def train_aaren(self, action_sequences: List[List[np.ndarray]], 
                   true_piece_types: List[PieceType], epochs: int = 10,
                   evaluator_weights: Optional[List[float]] = None,
                   positions: Optional[List[Tuple[int, int]]] = None):
        """
        Train the Aaren on labeled action sequences using parallel training.
        Now supports evaluator feedback weighting for improved learning.
        
        Args:
            action_sequences: List of action feature sequences
            true_piece_types: List of true piece types for each sequence
            epochs: Number of training epochs
            evaluator_weights: Optional list of weights from PBS evaluator (higher = more important)
            positions: Optional list of positions corresponding to each sequence (for evaluator feedback)
        """
        if len(action_sequences) == 0:
            return
        
        self.aaren_model.train()
        
        # Prepare training data
        max_seq_len = max(len(seq) for seq in action_sequences)
        max_seq_len = min(max_seq_len, 10)
        
        batch_size = min(len(action_sequences), 32)
        
        # Get evaluator feedback weights if available
        sample_weights = None
        if evaluator_weights is not None and len(evaluator_weights) == len(action_sequences):
            # Normalize weights to [0.5, 2.0] range (focus on high-value examples but don't ignore others)
            weights_array = np.array(evaluator_weights)
            if weights_array.max() > weights_array.min():
                # Normalize to [0.5, 2.0]
                normalized = 0.5 + 1.5 * (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min())
            else:
                normalized = np.ones_like(weights_array)  # All equal if no variation
            sample_weights = torch.tensor(normalized, device=self.device, dtype=torch.float32)
        elif self.evaluator is not None and positions is not None:
            # Get evaluator feedback for each position
            sample_weights = []
            for pos, piece_type in zip(positions, true_piece_types):
                if pos in self.belief_distributions:
                    feedback = self.get_evaluator_feedback(pos, ground_truth=piece_type)
                    if feedback:
                        # Use quality score as weight (normalize to [0.5, 2.0])
                        quality = feedback.get('quality_score', 0.0)
                        # Convert quality score to weight: negative scores -> 0.5, positive -> up to 2.0
                        weight = 0.5 + 1.5 * (1.0 / (1.0 + math.exp(-quality / 10.0)))
                        sample_weights.append(weight)
                    else:
                        sample_weights.append(1.0)  # Default weight
                else:
                    sample_weights.append(1.0)
            sample_weights = torch.tensor(sample_weights, device=self.device, dtype=torch.float32)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Create batches
            for batch_start in range(0, len(action_sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(action_sequences))
                batch_seqs = action_sequences[batch_start:batch_end]
                batch_labels = true_piece_types[batch_start:batch_end]
                batch_weights = sample_weights[batch_start:batch_end] if sample_weights is not None else None
                
                # Prepare batch tensor
                batch_tensor = torch.zeros(len(batch_seqs), max_seq_len, 24, device=self.device)
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
                
                # Forward pass (parallel training mode)
                predictions = self.aaren_model(batch_tensor)
                
                # Compute loss with optional sample weighting
                if batch_weights is not None:
                    # Weighted cross-entropy loss
                    loss_per_sample = F.cross_entropy(predictions, batch_labels_tensor.argmax(dim=1), reduction='none')
                    loss = (loss_per_sample * batch_weights).mean()
                else:
                    # Standard cross-entropy loss
                    loss = F.cross_entropy(predictions, batch_labels_tensor.argmax(dim=1))
                
                # Backward pass
                self.aaren_optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.aaren_model.parameters(), max_norm=1.0)
                self.aaren_optimizer.step()
                
                total_loss += loss.item()
        
        self.aaren_model.eval()
    
    def save_aaren_model(self, filepath: str):
        """
        Save the Aaren model separately.
        
        Args:
            filepath: Path to save the Aaren model
        """
        torch.save({
            'aaren_state_dict': self.aaren_model.state_dict(),
            'aaren_optimizer_state_dict': self.aaren_optimizer.state_dict(),
        }, filepath)
    
    def load_aaren_model(self, filepath: str):
        """
        Load the Aaren model separately.
        
        Args:
            filepath: Path to load the Aaren model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.aaren_model.load_state_dict(checkpoint['aaren_state_dict'])
        self.aaren_optimizer.load_state_dict(checkpoint['aaren_optimizer_state_dict'])
    
    def train_aaren_with_evaluator_feedback(self, epochs: int = 10):
        """
        Train AAREN using stored action sequences and evaluator feedback weights.
        This method automatically collects training data from revealed pieces and
        uses evaluator feedback to weight training examples.
        
        Args:
            epochs: Number of training epochs
        """
        # Collect training data from revealed pieces with action history
        action_sequences = []
        true_piece_types = []
        evaluator_weights = []
        positions = []
        
        for pos, piece_type in self.revealed_pieces.items():
            if pos in self.piece_action_history and len(self.piece_action_history[pos]) > 0:
                action_sequences.append(list(self.piece_action_history[pos]))
                true_piece_types.append(piece_type)
                positions.append(pos)
                # Get evaluator weight if available, otherwise use default
                weight = self._aaren_training_weights.get(pos, 1.0)
                evaluator_weights.append(weight)
        
        if len(action_sequences) > 0:
            # Train with evaluator-weighted samples
            self.train_aaren(
                action_sequences=action_sequences,
                true_piece_types=true_piece_types,
                epochs=epochs,
                evaluator_weights=evaluator_weights,
                positions=positions
            )
    
    def get_aaren_training_data(self) -> Tuple[List, List, List, List]:
        """
        Collect training data for shared AAREN training.
        Returns:
            Tuple of (action_sequences, true_piece_types, evaluator_weights, positions)
        """
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
    

