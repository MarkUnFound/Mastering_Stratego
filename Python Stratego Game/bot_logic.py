"""
Enhanced Bot Logic with PBS (Probabilistic Belief State) Integration
Combines neural network decision making with belief state tracking for improved play
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque
import math

class StrategoNet(nn.Module):
    def __init__(self):
        super(StrategoNet, self).__init__()
        self.fc1 = nn.Linear(200, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1000)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimplifiedPBS:
    """
    Simplified Probabilistic Belief State for Stratego
    Tracks opponent piece probabilities based on observed actions
    """
    
    def __init__(self, player_id: int):
        """
        Initialize PBS for tracking opponent pieces
        
        Args:
            player_id: The bot's player ID (1 or 2)
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        
        # Track belief distributions for each unknown enemy position
        # pos -> {rank: probability}
        self.beliefs: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
        
        # Track action history for each position
        self.action_history: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Track revealed pieces
        self.revealed: Dict[Tuple[int, int], int] = {}
        
        # Standard Stratego piece counts
        self.piece_counts = {
            10: 1,  # Marshal
            9: 1,   # General
            8: 2,   # Colonel
            7: 3,   # Major
            6: 4,   # Captain
            5: 4,   # Lieutenant
            4: 4,   # Sergeant
            3: 5,   # Miner
            2: 8,   # Scout
            1: 1,   # Spy
            0: 6,   # Bomb
            -1: 1,  # Flag
        }
        self.remaining_counts = self.piece_counts.copy()
        
    def reset(self):
        """Reset beliefs for a new game"""
        self.beliefs.clear()
        self.action_history.clear()
        self.revealed.clear()
        self.remaining_counts = self.piece_counts.copy()
    
    def initialize_position(self, pos: Tuple[int, int]):
        """Initialize uniform belief distribution for a position"""
        if pos not in self.beliefs:
            # Uniform distribution over all possible pieces
            total_pieces = sum(self.remaining_counts.values())
            if total_pieces > 0:
                self.beliefs[pos] = {
                    rank: count / total_pieces 
                    for rank, count in self.remaining_counts.items()
                }
            else:
                # Fallback to uniform if all revealed
                num_ranks = len(self.piece_counts)
                self.beliefs[pos] = {rank: 1.0 / num_ranks for rank in self.piece_counts.keys()}
    
    def update_from_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                        board, is_attack: bool = False):
        """
        Update beliefs based on a move
        
        Args:
            from_pos: Source position
            to_pos: Destination position
            board: Current board state
            is_attack: Whether this move resulted in an attack
        """
        # Initialize if needed
        self.initialize_position(from_pos)
        
        # Calculate move distance
        distance = max(abs(to_pos[0] - from_pos[0]), abs(to_pos[1] - from_pos[1]))
        
        # Rule-based inference
        beliefs = self.beliefs[from_pos]
        
        # Scout detection: moves > 1 square
        if distance > 1:
            # Strong evidence for Scout
            for rank in beliefs:
                if rank == 2:  # Scout
                    beliefs[rank] = min(0.95, beliefs[rank] * 3.0)
                elif rank in [0, -1]:  # Bomb, Flag can't move
                    beliefs[rank] = 0.0
                else:
                    beliefs[rank] *= 0.1
        
        # Single square move: not Scout, Bomb, or Flag
        elif distance == 1:
            for rank in beliefs:
                if rank == 2:  # Scout
                    beliefs[rank] *= 0.3
                elif rank in [0, -1]:  # Bomb, Flag
                    beliefs[rank] = 0.0
                else:
                    beliefs[rank] *= 1.2
        
        # Normalize
        total = sum(beliefs.values())
        if total > 0:
            for rank in beliefs:
                beliefs[rank] /= total
        
        # Transfer beliefs to new position if piece moved
        if to_pos != from_pos and not is_attack:
            self.beliefs[to_pos] = beliefs.copy()
            if from_pos in self.beliefs:
                del self.beliefs[from_pos]
    
    def update_from_reveal(self, pos: Tuple[int, int], rank: int):
        """
        Update beliefs when a piece is revealed
        
        Args:
            pos: Position of revealed piece
            rank: Actual rank of the piece
        """
        # Mark as revealed
        self.revealed[pos] = rank
        
        # Set belief to certainty
        self.beliefs[pos] = {r: 1.0 if r == rank else 0.0 for r in self.piece_counts.keys()}
        
        # Update remaining counts
        if rank in self.remaining_counts and self.remaining_counts[rank] > 0:
            self.remaining_counts[rank] -= 1
        
        # Update all other positions to account for reduced count
        for other_pos in self.beliefs:
            if other_pos != pos and other_pos not in self.revealed:
                beliefs = self.beliefs[other_pos]
                if rank in beliefs and self.remaining_counts[rank] == 0:
                    beliefs[rank] = 0.0
                    # Renormalize
                    total = sum(beliefs.values())
                    if total > 0:
                        for r in beliefs:
                            beliefs[r] /= total
    
    def get_expected_rank(self, pos: Tuple[int, int]) -> float:
        """
        Get expected rank value for a position
        
        Args:
            pos: Position to evaluate
            
        Returns:
            Expected rank (weighted average)
        """
        if pos in self.revealed:
            return float(self.revealed[pos])
        
        if pos not in self.beliefs:
            self.initialize_position(pos)
        
        beliefs = self.beliefs[pos]
        expected = sum(rank * prob for rank, prob in beliefs.items())
        return expected
    
    def get_win_probability(self, attacker_rank: int, defender_pos: Tuple[int, int]) -> float:
        """
        Estimate probability of winning an attack
        
        Args:
            attacker_rank: Rank of attacking piece
            defender_pos: Position of defending piece
            
        Returns:
            Probability of winning (0.0 to 1.0)
        """
        if defender_pos in self.revealed:
            defender_rank = self.revealed[defender_pos]
            return self._combat_outcome(attacker_rank, defender_rank)
        
        if defender_pos not in self.beliefs:
            self.initialize_position(defender_pos)
        
        beliefs = self.beliefs[defender_pos]
        win_prob = 0.0
        
        for rank, prob in beliefs.items():
            outcome = self._combat_outcome(attacker_rank, rank)
            win_prob += prob * outcome
        
        return win_prob
    
    def _combat_outcome(self, attacker: int, defender: int) -> float:
        """
        Determine combat outcome
        
        Args:
            attacker: Attacker rank
            defender: Defender rank
            
        Returns:
            1.0 if attacker wins, 0.0 if defender wins, 0.5 if tie
        """
        # Bomb logic
        if defender == 0:  # Bomb
            return 1.0 if attacker == 3 else 0.0  # Only Miner defeats Bomb
        if attacker == 0:  # Attacking with Bomb (shouldn't happen)
            return 0.0
        
        # Spy vs Marshal
        if attacker == 1 and defender == 10:
            return 1.0
        
        # Flag
        if defender == -1:
            return 1.0
        if attacker == -1:  # Shouldn't happen
            return 0.0
        
        # Standard combat
        if attacker > defender:
            return 1.0
        elif attacker < defender:
            return 0.0
        else:
            return 0.5  # Tie


class EnhancedBotLogic:
    """
    Enhanced bot that combines neural network with PBS for better decision making
    """
    
    def __init__(self, model_path: str, player_id: int = 2):
        """
        Initialize enhanced bot
        
        Args:
            model_path: Path to trained model weights
            player_id: Bot's player ID (1 or 2)
        """
        self.model = StrategoNet()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['q_network_state_dict'])
        self.model.eval()
        
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        
        # Initialize PBS
        self.pbs = SimplifiedPBS(player_id)
        
        # Track game state
        self.move_count = 0
        
    def reset(self):
        """Reset for a new game"""
        self.pbs.reset()
        self.move_count = 0
    
    def choose_move(self, board, owner: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Choose best move using combination of neural network and PBS
        
        Args:
            board: Current board state
            owner: Current player (should match self.player_id)
            
        Returns:
            (source, destination) tuple or None
        """
        # Get all legal moves
        legal_moves = self._get_all_legal_moves(board, owner)
        if not legal_moves:
            return None
        
        # Evaluate moves using PBS-enhanced scoring
        best_move = None
        best_score = -float('inf')
        
        for src, dst in legal_moves:
            score = self._evaluate_move(board, src, dst, owner)
            if score > best_score:
                best_score = score
                best_move = (src, dst)
        
        # Use neural network as tiebreaker/validation
        if best_move and len(legal_moves) > 1:
            nn_move = self._get_nn_move(board, owner, legal_moves)
            # Blend: 70% PBS-based, 30% neural network
            if random.random() < 0.3 and nn_move:
                best_move = nn_move
        
        return best_move
    
    def _evaluate_move(self, board, src: Tuple[int, int], dst: Tuple[int, int], owner: int) -> float:
        """
        Evaluate move quality using PBS
        
        Args:
            board: Board state
            src: Source position
            dst: Destination position
            owner: Current player
            
        Returns:
            Move score (higher is better)
        """
        score = 0.0
        
        piece = board.get(src)
        if not piece:
            return -1000.0
        
        target = board.get(dst)
        
        # Attack evaluation
        if target and target.owner != owner:
            # Use PBS to estimate win probability
            win_prob = self.pbs.get_win_probability(piece.rank, dst)
            
            # High-value attacks
            expected_rank = self.pbs.get_expected_rank(dst)
            if expected_rank >= 8:  # High value target
                score += 50.0 * win_prob
            elif expected_rank <= 0:  # Bomb or Flag
                if piece.rank == 3:  # Miner vs Bomb
                    score += 100.0
                elif expected_rank == -1:  # Flag
                    score += 1000.0  # Win condition!
            else:
                score += 20.0 * win_prob
            
            # Penalty for risky attacks
            if win_prob < 0.3:
                score -= 30.0
        
        # Positional scoring
        r, c = dst
        
        # Forward movement (toward opponent)
        if owner == 1:  # Player 1 moves up
            score += (10 - r) * 2.0
        else:  # Player 2 moves down
            score += r * 2.0
        
        # Center control
        center_dist = abs(r - 4.5) + abs(c - 4.5)
        score += (10 - center_dist) * 1.5
        
        # Scout special: long-range moves
        if piece.rank == 2:
            distance = max(abs(dst[0] - src[0]), abs(dst[1] - src[1]))
            if distance > 1:
                score += distance * 3.0  # Reward scout mobility
        
        # Flag protection: keep high-value pieces near flag
        if piece.rank >= 8:
            # Prefer staying in back rows
            if (owner == 1 and r >= 6) or (owner == 2 and r <= 3):
                score += 10.0
        
        return score
    
    def _get_nn_move(self, board, owner: int, legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get move suggestion from neural network
        
        Args:
            board: Board state
            owner: Current player
            legal_moves: List of legal moves
            
        Returns:
            Suggested move or None
        """
        board_state = self.get_board_state(board, owner)
        
        with torch.no_grad():
            output = self.model(board_state)
            legal_mask = self.get_legal_moves_mask(board, owner, output.shape[-1], legal_moves)
            masked_output = output * legal_mask
            move_index = torch.argmax(masked_output).item()
        
        return self.index_to_move(move_index, board, owner, legal_moves)
    
    def update_from_opponent_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                                  board, revealed_rank: Optional[int] = None):
        """
        Update PBS based on opponent's move
        
        Args:
            from_pos: Source position
            to_pos: Destination position
            board: Current board state
            revealed_rank: If piece was revealed, its rank
        """
        target = board.get(to_pos)
        is_attack = target is not None
        
        # Update PBS
        self.pbs.update_from_move(from_pos, to_pos, board, is_attack)
        
        # If piece revealed, update
        if revealed_rank is not None:
            self.pbs.update_from_reveal(to_pos, revealed_rank)
    
    def _get_all_legal_moves(self, board, owner: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all legal moves for the current player"""
        moves = []
        for src in board.owner_positions(owner):
            piece = board.get(src)
            if piece and piece.is_movable():
                legal_dsts = board.legal_moves_from(src)
                for dst in legal_dsts:
                    moves.append((src, dst))
        return moves
    
    # Original neural network support methods
    def get_board_state(self, board, owner: int) -> torch.Tensor:
        """Create board state representation for neural network"""
        player_state = np.zeros((10, 10), dtype=np.float32)
        opponent_state = np.zeros((10, 10), dtype=np.float32)
        
        for r in range(10):
            for c in range(10):
                piece = board.get((r, c))
                if piece:
                    if piece.owner == owner:
                        player_state[r, c] = piece.rank
                    else:
                        # Use PBS expected rank for hidden pieces
                        if piece.revealed:
                            opponent_state[r, c] = piece.rank
                        else:
                            expected = self.pbs.get_expected_rank((r, c))
                            opponent_state[r, c] = expected

        state_vector = np.concatenate(
            (player_state.flatten(), opponent_state.flatten()), axis=0
        )
        return torch.from_numpy(state_vector).unsqueeze(0)
    
    def get_legal_moves_mask(self, board, owner: int, output_size: int, 
                            legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> torch.Tensor:
        """Create mask for legal moves"""
        mask = torch.zeros(output_size)
        for src, dst in legal_moves:
            move_index = self.move_to_index(src, dst)
            if move_index < output_size:
                mask[move_index] = 1.0
        return mask
    
    def move_to_index(self, src: Tuple[int, int], dst: Tuple[int, int]) -> int:
        """Convert move to index"""
        src_index = src[0] * 10 + src[1]
        dst_row = dst[0]
        return src_index * 10 + dst_row
    
    def index_to_move(self, index: int, board, owner: int,
                     legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Convert index to move"""
        src_index = index // 10
        dst_row = index % 10
        src_r, src_c = src_index // 10, src_index % 10
        src = (src_r, src_c)

        legal_from_src = [move for move in legal_moves if move[0] == src]
        for move in legal_from_src:
            if move[1][0] == dst_row:
                return move

        # Fallback
        if legal_moves:
            return random.choice(legal_moves)
        return None
