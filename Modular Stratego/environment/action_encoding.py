"""
Lossless action encoding shared by the agent, the opponents that use a network, the GUI bot and
Expectamax.

A move is ((from_row, from_col), (to_row, to_col)) on the real board. Its action index is taken in
the mover's *canonical frame*: Player 2's board is rotated 180 degrees, so both sides see their
own pieces at the bottom and one network can play either side.

    index = from_square * 36 + direction * 9 + (distance - 1)

from_square = row * 10 + col in the canonical frame; direction is up, down, left, right
(DIRECTIONS order); distance is 1..9. Every legal move, long Scout moves included, gets its own
index: from-square x to-square for every square in a straight line. Most of the 3600 indices can
never be legal (they leave the board); mask them with legal_action_mask().

(The thesis code used 400 actions = from_square x direction and played a long Scout move as a
one-square step, so long Scout moves were unreachable and shared a Q-value with the step.)
"""

from typing import Iterable, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 10
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right (canonical frame)
MAX_DISTANCE = BOARD_SIZE - 1
ACTIONS_PER_SQUARE = len(DIRECTIONS) * MAX_DISTANCE  # 36
NUM_ACTIONS = NUM_SQUARES * ACTIONS_PER_SQUARE       # 3600

Move = Tuple[Tuple[int, int], Tuple[int, int]]


def is_first_player(player) -> bool:
    """Player 1 is 1; Player 2 is -1 in the trainer and 2 in the GUI."""
    return player == 1


def canonical_coords(r: int, c: int, player) -> Tuple[int, int]:
    """Map a square to `player`'s canonical frame and back (the mapping is its own inverse)."""
    if is_first_player(player):
        return r, c
    return BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c


def encode_move(move: Move, player) -> int:
    """Action index of `move` played by `player`. Raises ValueError for a non-straight move."""
    (r1, c1), (r2, c2) = move
    r1, c1 = canonical_coords(int(r1), int(c1), player)
    r2, c2 = canonical_coords(int(r2), int(c2), player)
    if c1 == c2 and r1 != r2:
        direction, distance = (0 if r2 < r1 else 1), abs(r2 - r1)
    elif r1 == r2 and c1 != c2:
        direction, distance = (2 if c2 < c1 else 3), abs(c2 - c1)
    else:
        raise ValueError(f"{move} is not a straight-line move")
    return (r1 * BOARD_SIZE + c1) * ACTIONS_PER_SQUARE + direction * MAX_DISTANCE + distance - 1


def decode_action(index: int, player) -> Optional[Move]:
    """Move for action `index` played by `player`, or None if the index leaves the board."""
    index = int(index)
    if not 0 <= index < NUM_ACTIONS:
        return None
    square, rest = divmod(index, ACTIONS_PER_SQUARE)
    direction, distance = divmod(rest, MAX_DISTANCE)
    distance += 1
    r1, c1 = divmod(square, BOARD_SIZE)
    dr, dc = DIRECTIONS[direction]
    r2, c2 = r1 + dr * distance, c1 + dc * distance
    if not (0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE):
        return None
    return canonical_coords(r1, c1, player), canonical_coords(r2, c2, player)


def legal_action_indices(moves: Iterable[Move], player) -> List[int]:
    return [encode_move(m, player) for m in moves]


def legal_action_mask(moves: Iterable[Move], player) -> np.ndarray:
    """Boolean (NUM_ACTIONS,) mask that is True exactly at the indices of `moves`."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    idx = legal_action_indices(moves, player)
    if idx:
        mask[idx] = True
    return mask


def mirror_action(index: int) -> int:
    """Index of the same move reflected left-right (columns c -> 9 - c) in the canonical frame.

    The lakes are symmetric under this reflection, so it is a symmetry of the game; the replay
    buffer's "flip" augmentation uses it together with a flip of the observation's last axis.
    """
    (r1, c1), (r2, c2) = decode_action(index, 1)
    return encode_move(((r1, MAX_DISTANCE - c1), (r2, MAX_DISTANCE - c2)), 1)


def rotate_action(index: int) -> int:
    """Index of the same move rotated 180 degrees in the canonical frame (the "rotate" augmentation)."""
    (r1, c1), (r2, c2) = decode_action(index, 1)
    return encode_move(((MAX_DISTANCE - r1, MAX_DISTANCE - c1), (MAX_DISTANCE - r2, MAX_DISTANCE - c2)), 1)
