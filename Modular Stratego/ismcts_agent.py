import math
import random
import copy
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

class ISMCTSNode:
    """
    Represents a node in the Information Set MCTS tree.
    Each node corresponds to a sequence of moves (history) from the root.
    Note: In IS-MCTS, nodes represent Information Sets (what we know), 
    not concrete game states (which include hidden info).
    """
    def __init__(self, move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, parent=None):
        self.move = move  # The move that led to this node
        self.parent = parent
        self.children: Dict[Tuple[Tuple[int, int], Tuple[int, int]], ISMCTSNode] = {}
        self.visits = 0
        self.total_value = 0.0
        self.avail_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [] # Stores valid moves from this state

    def ucb_score(self, c_puct: float = 1.414) -> float:
        """Calculate Upper Confidence Bound for Trees (UCT) score."""
        if self.visits == 0:
            return float('inf')
        
        # Standard UCB1
        exploitation = self.total_value / self.visits
        exploration = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """Check if all available moves have been expanded."""
        return len(self.children) == len(self.avail_moves) and len(self.avail_moves) > 0

    def best_child(self, c_puct: float = 0.0) -> 'ISMCTSNode':
        """Return the child with the highest UCB score (or visit count if c_puct=0)."""
        if not self.children:
            return None
        
        # If c_puct is 0, we are choosing the final action (robust child)
        if c_puct == 0:
            return max(self.children.values(), key=lambda node: node.visits)
        
        # Otherwise, use UCB for selection during search
        return max(self.children.values(), key=lambda node: node.ucb_score(c_puct))


class ISMCTSAgent:
    """
    Information Set Monte Carlo Tree Search Agent.
    Uses Probabilistic Belief State (PBS) to sample consistent worlds (Determinization).
    Uses DQN as a heuristic to evaluate leaf nodes (AlphaZero-style).
    """
    def __init__(self, dqn_agent, num_simulations: int = 50, c_puct: float = 1.414):
        self.dqn_agent = dqn_agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.name = f"ISMCTS_{dqn_agent.name}"

    def act(self, game_state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Choose the best move using IS-MCTS.
        """
        if not valid_moves:
            return None
            
        # If only one move, don't search
        if len(valid_moves) == 1:
            return valid_moves[0]

        root = ISMCTSNode(parent=None)
        root.avail_moves = valid_moves
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            self._run_simulation(root, game_state)
            
        # Select best move (most visited child)
        best_child = root.best_child(c_puct=0.0)
        if best_child:
            return best_child.move
        else:
            # Fallback if search failed (shouldn't happen)
            return random.choice(valid_moves)

    def _run_simulation(self, root: ISMCTSNode, real_game_state):
        """
        Run a single MCTS simulation.
        1. Determinize: Sample a concrete world from PBS.
        2. Select: Traverse tree to a leaf.
        3. Expand: Add a new child.
        4. Evaluate: Use DQN to estimate value.
        5. Backpropagate: Update values up the tree.
        """
        node = root
        
        # 1. Determinize
        # Sample a concrete state where hidden pieces are resolved
        # We use the helper from SearchAgent (re-implemented here or imported)
        # Since we are in a new file, let's implement a helper or assume PBS has it.
        # We will use the PBS directly if available, or a simplified sampler.
        
        determinized_state = self._sample_consistent_world(real_game_state)
        
        # 2. Selection
        # Traverse down until we hit a node that is not fully expanded or is terminal
        # Note: We must check if moves are valid in the *determinized* state.
        # IS-MCTS nuance: A node exists in the tree, but might not be valid in THIS determinization.
        # However, for Stratego, moves are usually valid regardless of hidden pieces (except specific interactions).
        # We'll assume tree structure is consistent with legal moves.
        
        while node.is_fully_expanded() and not self._is_terminal(determinized_state):
            node = node.best_child(self.c_puct)
            self._apply_move_logic(determinized_state, node.move)
            
        # 3. Expansion
        # If not terminal, expand one unexpanded child
        if not self._is_terminal(determinized_state):
            # Get valid moves in this determinized state
            # In a real implementation, we'd filter by what's actually possible in this sampled world
            # For simplicity, we use the node's stored available moves (which came from the root/parent)
            
            unexpanded_moves = [m for m in node.avail_moves if m not in node.children]
            
            if unexpanded_moves:
                move = random.choice(unexpanded_moves)
                new_node = ISMCTSNode(move=move, parent=node)
                
                # We need to know available moves from this new state to populate the new node
                # Simulate move to get next state
                next_state = copy.deepcopy(determinized_state)
                self._apply_move_logic(next_state, move)
                
                # Get valid moves for the NEXT player
                # Placeholder: In real usage, we'd call environment.get_valid_moves(next_state)
                # Since we don't have the full env here, we might need a generator.
                # For now, we'll leave avail_moves empty for leaves, meaning they will be evaluated but not expanded further in this sim.
                # This effectively makes it a 1-step lookahead expansion per sim, which is standard.
                new_node.avail_moves = [] # TODO: Implement move generator for deeper search
                
                node.children[move] = new_node
                node = new_node
                determinized_state = next_state # Update state to the expanded node

        # 4. Evaluation
        # Use DQN to get value of this state
        # Value is from perspective of current player in determinized_state
        value = self.dqn_agent.get_state_value(determinized_state)
        
        # 5. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
            # Flip value for opponent (zero-sum)
            value = -value

    def _sample_consistent_world(self, game_state):
        """
        Create a concrete game state by sampling unknown pieces from PBS beliefs.
        (Adapted from SearchAgent logic)
        """
        new_state = copy.deepcopy(game_state)
        pbs = self.dqn_agent.pbs
        
        if not pbs:
            return new_state
            
        board = new_state.board
        if isinstance(board, torch.Tensor):
            board = board.cpu().numpy()
            
        rows, cols = board.shape
        player_id = self.dqn_agent.player_id
        
        for r in range(rows):
            for c in range(cols):
                val = board[r][c]
                pos = (r, c)
                
                # Check if hidden (usually -20 or similar constant)
                is_hidden = (val == -20) 
                
                if is_hidden:
                    if pos in pbs.belief_distributions:
                        beliefs = pbs.belief_distributions[pos]
                        pieces = list(beliefs.keys())
                        probs = list(beliefs.values())
                        
                        if pieces and probs:
                            sampled_piece = random.choices(pieces, weights=probs, k=1)[0]
                            enemy_id = -1 if player_id == 1 else 1
                            sampled_val = sampled_piece.value * enemy_id
                            board[r][c] = sampled_val
                        else:
                            enemy_id = -1 if player_id == 1 else 1
                            board[r][c] = 2 * enemy_id # Scout default
                    else:
                        enemy_id = -1 if player_id == 1 else 1
                        board[r][c] = 2 * enemy_id # Scout default
                        
        if isinstance(new_state.board, torch.Tensor):
            new_state.board = torch.tensor(board, device=new_state.board.device)
        else:
            new_state.board = board
            
        return new_state

    def _apply_move_logic(self, game_state, move):
        """Apply move to state (in-place). Simplified battle logic."""
        (r1, c1), (r2, c2) = move
        board = game_state.board
        
        piece_val = board[r1][c1]
        target_val = board[r2][c2]
        
        attacker = abs(int(piece_val))
        defender = abs(int(target_val))
        
        if target_val == 0:
            board[r2][c2] = piece_val
            board[r1][c1] = 0
        else:
            # Simplified Battle
            winner_val = 0
            if attacker == 3 and defender == 11: winner_val = piece_val # Miner vs Bomb
            elif attacker == 1 and defender == 10: winner_val = piece_val # Spy vs Marshal
            elif attacker > defender: winner_val = piece_val
            elif defender > attacker: winner_val = target_val
            
            board[r2][c2] = winner_val
            board[r1][c1] = 0
            
        game_state.current_player *= -1

    def _is_terminal(self, game_state) -> bool:
        """Check if game is over (simplified)."""
        # In real implementation, check for flag capture or no moves
        return False
