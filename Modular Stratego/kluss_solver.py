import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import deque, defaultdict
import random
import copy
from dqn_evaluator import DQNEvaluator
from piece import PieceType

# Constants
BOARD_SIZE = 10
LAKE_SQUARE = -13
EMPTY_SQUARE = 0

class Node:
    def __init__(self, state, player, depth, node_id: str, is_chance=False):
        self.state = state
        self.player = player  # 0 for chance nodes, 1 or -1 for player nodes
        self.depth = depth
        self.id = node_id
        
        # Tree structure
        self.children: Dict[Any, 'Node'] = {} # Action/Outcome -> Node
        self.parents: List['Node'] = []
        
        # Graph structure (KLUSS)
        self.infoset_neighbors: Set['Node'] = set()
        
        # KLUSS Metadata
        self.knowledge_distance = float('inf')
        self.in_subgame = False
        self.is_unfrozen = False
        self.is_terminal = False
        self.payoff = 0.0
        
        # STRATEGO SPECIFIC: Chance node support
        self.is_chance = is_chance
        self.outcome_probs: Dict[Any, float] = {}  # For chance nodes: outcome -> probability
        
        # CFR Data
        self.regret_sum = {}
        self.strategy_sum = {}
        self.strategy = {}
        self.reach_prob = 1.0

    def add_infoset_neighbor(self, node: 'Node'):
        if node != self:
            self.infoset_neighbors.add(node)
            node.infoset_neighbors.add(self)

    def get_strategy(self, realization_weight):
        actions = list(self.children.keys())
        if not actions:
            return {}
        regrets = {a: max(self.regret_sum.get(a, 0.0), 0.0) for a in actions}
        sum_regret = sum(regrets.values())
        if sum_regret > 0:
            return {a: r / sum_regret for a, r in regrets.items()}
        else:
            return {a: 1.0 / len(actions) for a in actions}

class BeliefSampler:
    """
    CRITICAL FOR STRATEGO: Replaces Obscuro's enumeration of all consistent states.
    Generates samples consistent with game history using constraint satisfaction.
    """
    def __init__(self):
        pass
    
    def generate_consistent_sample(self, game_state, history=None):
        """
        Generate a concrete board configuration consistent with:
        1. Known piece locations (my pieces)
        2. Revealed enemy pieces (from battles)
        3. Movement constraints (Bombs/Flags haven't moved)
        4. Piece count constraints (standard Stratego setup)
        """
        # For now, return a copy of the current state
        # TODO: Implement full CSP-based sampling that:
        # - Shuffles unknown enemy pieces into unknown slots
        # - Respects revealed pieces from history
        # - Ensures immobile pieces (FLAG, BOMB) are in valid positions
        return copy.deepcopy(game_state)

class KLUSSSolver:
    def __init__(self, dqn_evaluator: DQNEvaluator, max_depth=4, k=2, iterations=1000, belief_state=None):
        self.dqn = dqn_evaluator
        self.max_depth = max_depth
        self.k = k
        self.iterations = iterations
        self.nodes: Dict[str, Node] = {}
        self.root_infoset_nodes: List[Node] = []
        self.infoset_map = defaultdict(lambda: defaultdict(list))
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # STRATEGO SPECIFIC: Particle filter for state sampling (not enumeration)
        self.belief_sampler = BeliefSampler()
        self.belief_state = belief_state  # For probability queries

    def solve(self, current_game_state, belief_state):
        self.nodes.clear()
        self.root_infoset_nodes.clear()
        self.infoset_map.clear()
        
        # 1. Sample States & Build Graph
        sampled_states = self.sample_states(current_game_state, belief_state, num_samples=5)
        self.build_connectivity_graph(sampled_states)
        
        # 2. Compute Distances
        self.compute_knowledge_distances()
        
        # 3. Mark Subgame Nodes
        self.mark_subgame_nodes()
        
        # 4. Verify Implementation (optional, can disable for performance)
        if len(self.nodes) > 0:
            try:
                self.verify_kluss_implementation()
            except AssertionError as e:
                print(f"⚠️ KLUSS verification failed: {e}")
        
        # 5. Run CFR
        for i in range(self.iterations):
            if not self.root_infoset_nodes: break
            root = random.choice(self.root_infoset_nodes)
            self.cfr(root, 1.0, 1.0)
            
        return self.get_aggregated_strategy()

    def build_connectivity_graph(self, start_states: List[Any]):
        """
        STRATEGO ADAPTATION: Creates chance nodes for uncertain battle outcomes.
        Unlike Obscuro (deterministic outcomes), Stratego battles branch based on belief probabilities.
        """
        queue = deque()
        for i, state in enumerate(start_states):
            node_id = f"root_{i}"
            node = Node(state, state.current_player, 0, node_id)
            self.nodes[node_id] = node
            self.root_infoset_nodes.append(node)
            queue.append(node)

        nodes_expanded = 0
        MAX_NODES = 5000  # Increased for Stratego due to chance branching
        
        while queue and nodes_expanded < MAX_NODES:
            node = queue.popleft()
            
            # Skip chance nodes for infoset mapping (they're not decision nodes)
            if not node.is_chance:
                key_p1 = self.get_infoset_key(node.state, 1)
                key_p2 = self.get_infoset_key(node.state, -1)
                self.infoset_map[1][key_p1].append(node)
                self.infoset_map[-1][key_p2].append(node)
            
            if node.depth >= self.max_depth or self.is_terminal(node.state):
                node.is_terminal =True
                continue
            
            # Chance nodes: expand all outcomes
            if node.is_chance:
                for outcome_idx, child in node.children.items():
                    if child not in [n for n in queue]:
                        queue.append(child)
                continue
                
            # Decision nodes: get actions
            actions = self.get_legal_actions(node.state)
            for action in actions:
                # STRATEGO SPECIFIC: Check if this action leads to battle
                outcomes = self.get_battle_outcomes(node.state, action)
                
                if len(outcomes) == 1:
                    # Deterministic move (empty square)
                    prob, next_state = outcomes[0]
                    child_id = f"{node.id}_a{nodes_expanded}"
                    child = Node(next_state, next_state.current_player, node.depth + 1, child_id)
                    node.children[action] = child
                    child.parents.append(node)
                    self.nodes[child_id] = child
                    queue.append(child)
                    nodes_expanded += 1
                else:
                    # Battle with uncertain outcome -> Create CHANCE NODE
                    chance_id = f"{node.id}_chance{nodes_expanded}"
                    chance_node = Node(node.state, player=0, depth=node.depth, node_id=chance_id, is_chance=True)
                    
                    self.nodes[chance_id] = chance_node
                    node.children[action] = chance_node  # Parent -> Chance
                    chance_node.parents.append(node)
                    nodes_expanded += 1
                    
                    # Create outcome nodes
                    for outcome_idx, (prob, next_state) in enumerate(outcomes):
                        outcome_id = f"{chance_id}_out{outcome_idx}"
                        outcome_child = Node(next_state, next_state.current_player, node.depth + 1, outcome_id)
                        outcome_child.reach_prob = prob
                        chance_node.children[outcome_idx] = outcome_child
                        chance_node.outcome_probs[outcome_idx] = prob
                        outcome_child.parents.append(chance_node)
                        self.nodes[outcome_id] = outcome_child
                        queue.append(outcome_child)
                        nodes_expanded += 1
            
        # Build Edges (skip chance nodes)
        for player in [1, -1]:
            for key, nodes in self.infoset_map[player].items():
                if len(nodes) > 1:
                    base_node = nodes[0]
                    for other in nodes[1:]:
                        base_node.add_infoset_neighbor(other)

    def compute_knowledge_distances(self):
        """
        Phase 1: BFS through connectivity graph using infoset_neighbors.
        Phase 2: Propagate distances down the game tree to descendants.
        """
        # Phase 1: Initialize all distances
        for node in self.nodes.values():
            node.knowledge_distance = float('inf')
        
        # Phase 2: BFS through connectivity graph
        queue = deque()
        for node in self.root_infoset_nodes:
            node.knowledge_distance = 0
            queue.append(node)
        
        visited = set()
        
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            
            current_distance = u.knowledge_distance
            
            # CRITICAL FIX: Use the pre-built infoset_neighbors, not re-querying infoset_map
            for v in u.infoset_neighbors:
                if v.knowledge_distance > current_distance + 1:
                    v.knowledge_distance = current_distance + 1
                    queue.append(v)
        
        # Phase 3: Propagate distances DOWN the game tree
        # Children inherit AT MOST the parent's distance
        def propagate_to_descendants(node):
            for child in node.children.values():
                # Child can't be closer than parent
                if child.knowledge_distance > node.knowledge_distance:
                    child.knowledge_distance = node.knowledge_distance
                    propagate_to_descendants(child)
        
        for root in self.root_infoset_nodes:
            propagate_to_descendants(root)

    def mark_subgame_nodes(self):
        """
        For k=2 (2-KLUSS):
        - Cutoff distance: k+1 = 3 (keep nodes at distance <= 3)
        - Unfrozen: nodes at distance <= k = 2 (I_1, I_2, I_3 in paper notation)
        - Include downward closure (all descendants of core nodes)
        """
        # For k=2: cutoff is k+1 = 3
        cutoff_distance = self.k + 1  # 3
        
        # Step 1: Find all nodes within distance cutoff
        core_nodes = [n for n in self.nodes.values() 
                      if n.knowledge_distance <= cutoff_distance]
        
        # Step 2: Include downward closure (all descendants)
        queue = deque(core_nodes)
        subgame_nodes = set(core_nodes)
        
        while queue:
            node = queue.popleft()
            for child in node.children.values():
                if child not in subgame_nodes:
                    subgame_nodes.add(child)
                    queue.append(child)
        
        # Step 3: Mark all as in subgame and set unfrozen status
        for node in subgame_nodes:
            node.in_subgame = True
            
            # For 2-KLUSS: unfrozen if distance <= k = 2
            # This includes distance 0, 1, 2 (I_1, I_2, I_3)
            if node.knowledge_distance <= self.k:
                node.is_unfrozen = True
            else:
                # Distance > 2 (but <= 3 since in subgame)
                # These are boundary nodes, treated as frozen/evaluated
                node.is_unfrozen = False
        
        # Step 4: Identify opponent boundary infosets (for gadget construction)
        self.opponent_boundary_infosets = self._identify_opponent_boundary_infosets(subgame_nodes)

    def get_infoset_key(self, state, observer_player):
        board_desc = []
        board = state.board
        if hasattr(board, 'cpu'): board = board.cpu().numpy()
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = board[r, c]
                if hasattr(p, 'item'): p = p.item()
                
                if p == 0: board_desc.append("0")
                elif p == -13: board_desc.append("L")
                else:
                    owner = 1 if p > 0 else -1
                    if owner == observer_player:
                        board_desc.append(str(p))
                    else:
                        # Check revealed
                        is_revealed = False
                        if observer_player == 1 and hasattr(state, 'revealed_pieces_p1'):
                            is_revealed = (r, c) in state.revealed_pieces_p1
                        elif observer_player == -1 and hasattr(state, 'revealed_pieces_p2'):
                            is_revealed = (r, c) in state.revealed_pieces_p2
                        
                        if is_revealed: board_desc.append(str(p))
                        else: board_desc.append("?")
        return f"{observer_player}|{''.join(board_desc)}"

    def sample_states(self, game_state, belief_state, num_samples=100):
        """
        CRITICAL CHANGE FOR STRATEGO (vs Obscuro):
        Obscuro stores 'P' (all consistent states). We cannot (|P| ~ 10^33).
        We must use a generative approach (Particle Filter).
        """
        samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        # Store belief_state for probability queries
        if belief_state is not None:
            self.belief_state = belief_state
        
        # If belief_state has a sample method, use it
        if belief_state is not None and hasattr(belief_state, 'sample_concrete_state'):
            while len(samples) < num_samples and attempts < max_attempts:
                sample = belief_state.sample_concrete_state(game_state)
                if sample is not None:
                    samples.append(sample)
                attempts += 1
        else:
            # Fallback: Use constraint sampler
            while len(samples) < num_samples and attempts < max_attempts:
                sample = self.belief_sampler.generate_consistent_sample(game_state, history=None)
                if sample is not None:
                    samples.append(sample)
                attempts += 1
        
        # Ensure we have at least one sample (the current state)
        if len(samples) == 0:
            samples.append(copy.deepcopy(game_state))
        
        return samples

    def get_legal_actions(self, state):
        moves = []
        board = state.board
        if hasattr(board, 'cpu'): board = board.cpu().numpy()
        
        player = state.current_player
        
        # Identify player pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = board[r, c]
                if hasattr(p, 'item'): p = p.item()
                
                if p == 0 or p == LAKE_SQUARE: continue
                if (player == 1 and p <= 0) or (player == -1 and p >= 0): continue
                
                piece_type = abs(p)
                if piece_type in [PieceType.FLAG.value, PieceType.BOMB.value]: continue
                
                # Generate moves
                if piece_type == PieceType.SCOUT.value:
                    for dr, dc in self.directions:
                        for i in range(1, BOARD_SIZE):
                            nr, nc = r + i*dr, c + i*dc
                            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE): break
                            target = board[nr, nc]
                            if hasattr(target, 'item'): target = target.item()
                            
                            if target == LAKE_SQUARE: break
                            if target == 0:
                                moves.append(((r, c), (nr, nc)))
                            else:
                                # Attack?
                                if (player == 1 and target < 0) or (player == -1 and target > 0):
                                    moves.append(((r, c), (nr, nc)))
                                break # Blocked by piece (friend or foe)
                else:
                    for dr, dc in self.directions:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE): continue
                        target = board[nr, nc]
                        if hasattr(target, 'item'): target = target.item()
                        
                        if target == LAKE_SQUARE: continue
                        if target == 0:
                            moves.append(((r, c), (nr, nc)))
                        elif (player == 1 and target < 0) or (player == -1 and target > 0):
                            moves.append(((r, c), (nr, nc)))
                            
        return moves

    def apply_action(self, state, action):
        """
        Apply action to state, handling battle stochasticity with chance nodes.
        For KLUSS, we need to branch on possible battle outcomes when piece identities are unknown.
        """
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = action
        
        board = new_state.board
        if hasattr(board, 'cpu'): board = board.cpu().numpy()
        
        attacker = board[r1, c1]
        if hasattr(attacker, 'item'): attacker = attacker.item()
        
        defender = board[r2, c2]
        if hasattr(defender, 'item'): defender = defender.item()
        
        # Check if this is a battle (target square occupied by enemy)
        is_battle = False
        if defender != 0 and defender != LAKE_SQUARE:
            attacker_owner = 1 if attacker > 0 else -1
            defender_owner = 1 if defender > 0 else -1
            if attacker_owner != defender_owner:
                is_battle = True
        
        if is_battle:
            # CRITICAL: For KLUSS, we should create chance nodes here
            # For now, we use a simplified deterministic resolution
            # TODO: Implement full chance node branching based on belief state
            
            # Simplified: Assume attacker wins (deterministic approximation)
            # In full implementation, this would create multiple child nodes
            # with probabilities based on belief state
            new_state.board[r1, c1] = 0
            new_state.board[r2, c2] = attacker
        else:
            # Simple move
            new_state.board[r1, c1] = 0
            new_state.board[r2, c2] = attacker
        
        new_state.current_player *= -1
        return new_state
    
    def get_battle_outcomes(self, state, action):
        """
        STRATEGO SPECIFIC: Returns list of (probability, next_state) for battle outcomes.
        If attacking an unknown piece, branch based on belief probabilities.
        """
        (r1, c1), (r2, c2) = action
        board = state.board
        if hasattr(board, 'cpu'): board = board.cpu().numpy()
        
        attacker_val = board[r1, c1]
        if hasattr(attacker_val, 'item'): attacker_val = attacker_val.item()
        
        defender_val = board[r2, c2]
        if hasattr(defender_val, 'item'): defender_val = defender_val.item()
        
        # Case 1: Moving to empty square (deterministic)
        if defender_val == 0 or defender_val == LAKE_SQUARE:
            next_state = self._apply_deterministic_move(state, action)
            return [(1.0, next_state)]
        
        # Case 2: Battle
        attacker_owner = 1 if attacker_val > 0 else -1
        defender_owner = 1 if defender_val > 0 else -1
        
        if attacker_owner == defender_owner:
            return []  # Same team
        
        # Query belief state for defender probabilities
        if self.belief_state and hasattr(self.belief_state, 'belief_distributions'):
            # Get belief distribution for this position
            if (r2, c2) in self.belief_state.belief_distributions:
                defender_probs = self.belief_state.belief_distributions[(r2, c2)]
                # Convert PieceType keys to values
                defender_probs = {pt.value: prob for pt, prob in defender_probs.items()}
            else:
                defender_probs = {abs(defender_val): 1.0}
        else:
            defender_probs = {abs(defender_val): 1.0}
        
        # Calculate outcome probabilities
        attacker_type = abs(attacker_val)
        win_prob = 0.0
        loss_prob = 0.0
        draw_prob = 0.0
        
        for defender_type, type_prob in defender_probs.items():
            if type_prob < 0.01:
                continue
            
            result = self._resolve_battle(attacker_type, defender_type)
            if result == 1:
                win_prob += type_prob
            elif result == -1:
                loss_prob += type_prob
            else:
                draw_prob += type_prob
        
        outcomes = []
        if win_prob > 0:
            outcomes.append((win_prob, self._apply_battle_win(state, action)))
        if loss_prob > 0:
            outcomes.append((loss_prob, self._apply_battle_loss(state, action)))
        if draw_prob > 0:
            outcomes.append((draw_prob, self._apply_battle_draw(state, action)))
        
        return outcomes if outcomes else [(1.0, self._apply_deterministic_move(state, action))]
    
    def _resolve_battle(self, attacker_type, defender_type):
        """Returns: 1 (attacker wins), -1 (defender wins), 0 (draw)"""
        # Special cases
        if defender_type == PieceType.BOMB.value:
            return 1 if attacker_type == PieceType.MINER.value else -1
        if defender_type == PieceType.FLAG.value:
            return 1
        if attacker_type == PieceType.SPY.value and defender_type == PieceType.MARSHAL.value:
            return 1
        
        # Standard ranking
        if attacker_type > defender_type:
            return 1
        elif attacker_type < defender_type:
            return -1
        else:
            return 0
    
    def _apply_deterministic_move(self, state, action):
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = action
        new_state.board[r1, c1] = 0
        new_state.board[r2, c2] = state.board[r1, c1]
        new_state.current_player *= -1
        return new_state
    
    def _apply_battle_win(self, state, action):
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = action
        new_state.board[r2, c2] = state.board[r1, c1]
        new_state.board[r1, c1] = 0
        new_state.current_player *= -1
        return new_state
    
    def _apply_battle_loss(self, state, action):
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = action
        new_state.board[r1, c1] = 0
        new_state.current_player *= -1
        return new_state
    
    def _apply_battle_draw(self, state, action):
        new_state = copy.deepcopy(state)
        (r1, c1), (r2, c2) = action
        new_state.board[r1, c1] = 0
        new_state.board[r2, c2] = 0
        new_state.current_player *= -1
        return new_state

    def state_to_tensor(self, state):
        tensor = torch.zeros((41, 10, 10), device=self.dqn.device)
        board = state.board
        if hasattr(board, 'cpu'): board = board.cpu().numpy()
        
        # Channels 0-11: My Pieces (P1 if current is P1)
        # Channels 12-23: Known Enemy Pieces
        # Channel 24: Unknown Enemy
        # Channel 25: Lakes
        
        current_p = state.current_player
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = board[r, c]
                if hasattr(p, 'item'): p = p.item()
                
                if p == LAKE_SQUARE:
                    tensor[25, r, c] = 1
                    continue
                if p == 0: continue
                
                piece_type = abs(p)
                owner = 1 if p > 0 else -1
                
                if owner == current_p:
                    # My piece (0-11)
                    # PieceType enum is 1-12 (Flag=1, Spy=2, ...)
                    # Map to 0-11
                    idx = piece_type - 1
                    if 0 <= idx < 12:
                        tensor[idx, r, c] = 1
                else:
                    # Enemy piece
                    # Check if revealed
                    is_revealed = False
                    if current_p == 1 and hasattr(state, 'revealed_pieces_p2'):
                        is_revealed = (r, c) in state.revealed_pieces_p2
                    elif current_p == -1 and hasattr(state, 'revealed_pieces_p1'):
                        is_revealed = (r, c) in state.revealed_pieces_p1
                        
                    if is_revealed:
                        # Known enemy (12-23)
                        idx = 12 + (piece_type - 1)
                        if 12 <= idx < 24:
                            tensor[idx, r, c] = 1
                    else:
                        # Unknown enemy
                        tensor[24, r, c] = 1
                        
        return tensor

    def is_terminal(self, state):
        # Check flag capture or no moves
        # Simplified check
        return False

    def cfr(self, node, p0, p1):
        """
        CFR with STRATEGO ADAPTATION: Handle chance nodes for probabilistic battles.
        """
        if not node.in_subgame: return 0.0
        if node.is_terminal: return 0.0
        
        # Evaluation: Use DQN if node is "Frozen" or max depth
        if node.depth >= self.max_depth or not node.is_unfrozen:
            return self.dqn.evaluate(self.state_to_tensor(node.state))
        
        # CHANCE NODE HANDLING (New for Stratego)
        if node.is_chance:
            expected_value = 0.0
            for outcome_idx, child in node.children.items():
                # Traverse child, weighting by probability
                prob = node.outcome_probs.get(outcome_idx, child.reach_prob)
                val = self.cfr(child, p0, p1)
                expected_value += prob * val
            return expected_value
        
        # STANDARD CFR (External Sampling / MCCFR)
        current_player = node.player
        strategy = node.get_strategy(realization_weight=1.0)
        
        if not strategy:  # No children
            return 0.0
        
        util = {}
        node_util = 0.0
        
        for action, child in node.children.items():
            if current_player == 1:
                util[action] = self.cfr(child, p0 * strategy.get(action, 0), p1)
            else:
                util[action] = self.cfr(child, p0, p1 * strategy.get(action, 0))
            node_util += strategy.get(action, 0) * util[action]
        
        # Regret Update (Only for decision nodes, not chance nodes)
        prob_weight = p1 if current_player == 1 else p0
        for action in node.children:
            regret = util.get(action, 0) - node_util
            node.regret_sum[action] = node.regret_sum.get(action, 0) + (regret * prob_weight)
        
        return node_util

    def get_aggregated_strategy(self):
        if self.root_infoset_nodes:
            return self.root_infoset_nodes[0].strategy
        return {}
    
    def _identify_opponent_boundary_infosets(self, subgame_nodes):
        """
        Identify opponent infosets at the subgame boundary.
        These are used for setting alternate values in Resolve/Maxmargin gadgets.
        """
        opponent_infosets = defaultdict(list)
        
        for node in subgame_nodes:
            # Check if this is an opponent decision node at the boundary
            # (distance = k+1, which is 3 for k=2)
            if node.knowledge_distance == self.k + 1:
                # Get opponent's infoset key
                opponent_player = -node.player  # Opponent of current player
                infoset_key = self.get_infoset_key(node.state, opponent_player)
                opponent_infosets[infoset_key].append(node)
        
        return opponent_infosets
    
    def verify_kluss_implementation(self):
        """
        Verification tests to ensure KLUSS is correctly implemented.
        Run after build_connectivity_graph and compute_knowledge_distances.
        """
        print("Running KLUSS verification tests...")
        
        # Test 1: Distance 0 nodes should be root infoset
        for node in self.root_infoset_nodes:
            assert node.knowledge_distance == 0, f"Root node {node.id} has distance {node.knowledge_distance}, expected 0"
        print("✓ Test 1 passed: Root infoset has distance 0")
        
        # Test 2: Connected nodes differ by at most 1 in distance
        for node in self.nodes.values():
            for neighbor in node.infoset_neighbors:
                distance_diff = abs(node.knowledge_distance - neighbor.knowledge_distance)
                assert distance_diff <= 1, f"Nodes {node.id} and {neighbor.id} have distance diff {distance_diff}"
        print("✓ Test 2 passed: Connected nodes differ by at most 1 in distance")
        
        # Test 3: Children have distance >= parent distance
        for node in self.nodes.values():
            for child in node.children.values():
                assert child.knowledge_distance >= node.knowledge_distance, \
                    f"Child {child.id} (dist={child.knowledge_distance}) closer than parent {node.id} (dist={node.knowledge_distance})"
        print("✓ Test 3 passed: Children have distance >= parent distance")
        
        # Test 4: Subgame contains downward closure
        subgame_nodes = {n for n in self.nodes.values() if n.in_subgame}
        for node in subgame_nodes:
            for child in node.children.values():
                assert child.in_subgame, f"Child {child.id} not in subgame but parent {node.id} is"
        print("✓ Test 4 passed: Subgame contains downward closure")
        
        # Test 5: For k=2, unfrozen nodes are distance <= 2
        for node in self.nodes.values():
            if node.in_subgame and node.is_unfrozen:
                assert node.knowledge_distance <= self.k, \
                    f"Unfrozen node {node.id} has distance {node.knowledge_distance} > k={self.k}"
        print("✓ Test 5 passed: Unfrozen nodes have distance <= k")
        
        # Test 6: All nodes in subgame have finite distance
        for node in subgame_nodes:
            assert node.knowledge_distance != float('inf'), \
                f"Subgame node {node.id} has infinite distance"
        print("✓ Test 6 passed: All subgame nodes have finite distance")
        
        print(f"\n✅ All KLUSS verification tests passed!")
        print(f"   Total nodes: {len(self.nodes)}")
        print(f"   Subgame nodes: {len(subgame_nodes)}")
        print(f"   Unfrozen nodes: {sum(1 for n in subgame_nodes if n.is_unfrozen)}")
        print(f"   Opponent boundary infosets: {len(self.opponent_boundary_infosets)}")
