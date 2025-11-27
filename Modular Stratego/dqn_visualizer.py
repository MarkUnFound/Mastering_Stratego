import os
from typing import List, Tuple, Dict, Optional


class DQNMoveVisualizer:
    """Tracks move history and provides simple repetition penalties and hooks for visualization.

    This implementation is intentionally lightweight: it only tracks moves and players,
    computes a basic repetition penalty, and exposes stubs for frame-based visualization
    used by StrategoEnvironment.
    """

    def __init__(self):
        self.move_history: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]] = []

    def record_move(self, action: Tuple[Tuple[int, int], Tuple[int, int]], player: int) -> None:
        """Record a move taken by the given player."""
        self.move_history.append((action, player))

    def get_move_penalty(self, action: Tuple[Tuple[int, int], Tuple[int, int]], player: int) -> float:
        """Return a small penalty if the player is repeating recent moves.

        Strategy:
        - If the exact same action by the same player occurred in the last few turns,
          apply a negative penalty to discourage loops.
        - Otherwise return 0.0.
        """
        if not self.move_history:
            return 0.0

        # Look back over a short horizon
        horizon = 6
        recent = self.move_history[-horizon:]
        repeats = sum(1 for (a, p) in recent if p == player and a == action)

        if repeats <= 1:
            return 0.0

        # Penalty grows mildly with repeats, capped
        penalty = -0.02 * min(repeats - 1, 3)
        return penalty

    def visualize_move(self, move_index: int, save_path: Optional[str] = None) -> None:
        """Placeholder for per-move visualization.

        The environment already has a separate frame visualizer; this method is kept
        for compatibility and can be expanded later if finer-grained DQN move
        visualization is needed.
        """
        # No-op for now; hook for future extensions.
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def print_move_history(self) -> None:
        """Print a simple textual summary of the recorded move history."""
        for idx, (action, player) in enumerate(self.move_history):
            (r1, c1), (r2, c2) = action
            print(f"{idx}: P{player} move ({r1}, {c1}) -> ({r2}, {c2})")

    def clear_history(self) -> None:
        """Clear the stored move history."""
        self.move_history.clear()
