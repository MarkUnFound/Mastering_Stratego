# stratego_modular/mcts.py

import math
from typing import Dict, List, Tuple, Optional

class MCTSNode:
    """Simplified MCTS node optimized for GPU batch processing."""
    def __init__(self, parent=None, action=None, prior_prob=0.0):
        self.parent = parent
        self.action = action
        self.children: Dict[Tuple, MCTSNode] = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False

    def ucb_score(self, c_puct=1.4):
        """Calculates the Upper Confidence Bound for Trees (UCT) score."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_value / self.visits
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration

    def select_child(self):
        """Selects the child with the highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score())

    def expand(self, action_probs: List[Tuple[Tuple, float]]):
        """Expands the node by creating children for valid actions."""
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action, prior_prob=prob)
        self.is_expanded = True

    def backup(self, value: float):
        """Backpropagates the simulation result up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            # The value is from the perspective of the player who made the move *into* the state.
            # So, we negate it for the parent.
            node.total_value += value
            value = -value
            node = node.parent