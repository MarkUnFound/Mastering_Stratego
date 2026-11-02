"""
Turn bookkeeping for two-player training lanes.

A transition for side X starts at X's own view when X moves and ends at X's own view the next
time X is to move, i.e. after the opponent's reply. When the game ends, the open transitions of
both sides are closed as terminal, whichever side made the last move.
"""

from typing import Dict, List, Optional


def outcome_reward(winner: Optional[int], player: int) -> float:
    """Terminal reward for `player`: +1 win, -1 loss, 0 draw."""
    if winner == player:
        return 1.0
    if winner == -player:
        return -1.0
    return 0.0


class TurnTracker:
    """Pairs each side's move with the opponent's reply, separately for every lane."""

    def __init__(self, num_lanes: int):
        self.pending: List[Dict[int, Optional[dict]]] = [{1: None, -1: None} for _ in range(num_lanes)]

    def reset(self, lane: int):
        """Forget open moves (call when the lane starts a new game)."""
        self.pending[lane] = {1: None, -1: None}

    def record(self, lane: int, player: int, state, action, next_state, done: bool, info: dict,
               snapshot=None, next_moves=None) -> List[dict]:
        """
        Side `player` played `action` in `state` (its own view); the env returned `next_state`
        (the view of the side now to move), `done`, `info` and `next_moves` (the legal moves of
        the side now to move).

        Returns the transitions this completes, each a dict with lane, player, state, action,
        next_state, next_moves (the transition's side's legal moves in next_state; None when
        done), done, winner, info (the mover's info, with the battles of both the move and the
        reply in revealed_in_step) and snapshot (AAREN history when `state` was acted on).
        """
        completed = []
        waiting = self.pending[lane][-player]
        if waiting is not None:
            # The opponent's open move is answered by this move: it is now the opponent's turn
            # again (next_state is its view), or the game is over
            completed.append(self._close(waiting, next_state, done, info, next_moves))
            self.pending[lane][-player] = None

        if action is not None:
            move = {'lane': lane, 'player': player, 'state': state, 'action': action,
                    'info': info, 'snapshot': snapshot}
            if done:
                completed.append(self._close(move, next_state, True, info, None))
            else:
                self.pending[lane][player] = move
        return completed

    @staticmethod
    def _close(move: dict, next_state, done: bool, reply_info: dict, next_moves) -> dict:
        info = dict(move['info'])
        if reply_info is not move['info']:
            info['revealed_in_step'] = (list(move['info'].get('revealed_in_step') or []) +
                                        list(reply_info.get('revealed_in_step') or []))
        return {**move, 'next_state': next_state, 'next_moves': None if done else next_moves, 'done': done,
                'winner': reply_info.get('winner') if done else None, 'info': info}
