"""
Observation planes: what one side can see, as a (30, 10, 10) float tensor in that side's
canonical frame (Player 2's board rotated 180 degrees, see action_encoding.py), so one network
can play either side and its action indices line up with the squares.

    0-11   own pieces, one plane per rank (PieceType value - 1)
    12-23  enemy pieces whose rank this side knows, one plane per rank (revealed in battle, or
           shown by the curriculum's observability)
    24     enemy pieces of unknown rank
    25     lakes
    26     own pieces whose rank the enemy knows
    27     observability flag: share of hidden enemy ranks the curriculum is showing
           (1 = full observability, 0 = normal fog of war), constant over the board
    28     no-battle counter: plies since the last battle / the no-battle draw limit (0 if off)
    29     game progress: turns played / turn limit (0 if unknown)

Empty squares are the squares with no plane set. The thesis encoding had 15 planes: own pieces
by rank, one plane for every enemy piece (ranks dropped, even revealed ones and even under full
observability), lakes and empty squares, and no counters.

Planes 26-29 come from the GameState (the GUI bot passes a GuiObservation with the same
fields); a bare board leaves them at 0.
"""

from typing import Optional, Sequence

import torch

from action_encoding import BOARD_SIZE, is_first_player
from board import HIDDEN_PIECE, LAKE_SQUARE

NUM_RANKS = 12
CH_OWN = 0
CH_ENEMY_KNOWN = 12
CH_ENEMY_HIDDEN = 24
CH_LAKE = 25
CH_OWN_REVEALED = 26
CH_OBSERVABILITY = 27
CH_NO_BATTLE = 28
CH_PROGRESS = 29
NUM_OBS_PLANES = 30


def _board_tensor(item, device) -> torch.Tensor:
    board = getattr(item, 'board', item)
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    return board.to(device=device, dtype=torch.int16)


def _ratio(numerator, limit) -> float:
    if not limit or limit <= 0:
        return 0.0
    return min(float(numerator) / float(limit), 1.0)


def encode_observations(items: Sequence, players, device=None) -> torch.Tensor:
    """
    Observation planes for a batch.

    Args:
        items: GameState objects (planes 26-29 read from them) or bare 10x10 boards in the
            board.py value convention (Player 1 positive, Player 2 negative, LAKE_SQUARE,
            HIDDEN_PIECE for an enemy of unknown rank).
        players: the viewing side for each item (1, or -1/2 for Player 2), or one value for all.
        device: device of the result (default: the first board's device).

    Returns:
        (B, NUM_OBS_PLANES, 10, 10) float32 tensor in each viewer's canonical frame.
    """
    if device is None:
        first = getattr(items[0], 'board', items[0])
        device = first.device if isinstance(first, torch.Tensor) else torch.device('cpu')
    n = len(items)
    if not isinstance(players, (list, tuple)):
        players = [players] * n

    boards = torch.stack([_board_tensor(it, device) for it in items])          # (B, 10, 10)
    sign = torch.tensor([1 if is_first_player(p) else -1 for p in players],
                        dtype=torch.int16, device=device).view(n, 1, 1)

    lake = boards == LAKE_SQUARE
    hidden = boards == HIDDEN_PIECE
    piece = (boards != 0) & ~lake & ~hidden
    v = boards * sign                                                          # own > 0, enemy < 0
    own_rank = torch.where(piece & (v > 0), v, torch.zeros_like(v)).long().clamp(0, NUM_RANKS)
    enemy_rank = torch.where(piece & (v < 0), -v, torch.zeros_like(v)).long().clamp(0, NUM_RANKS)

    obs = torch.zeros((n, NUM_OBS_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
    obs[:, CH_OWN:CH_OWN + NUM_RANKS] = torch.nn.functional.one_hot(own_rank, NUM_RANKS + 1)[..., 1:].permute(0, 3, 1, 2)
    obs[:, CH_ENEMY_KNOWN:CH_ENEMY_KNOWN + NUM_RANKS] = \
        torch.nn.functional.one_hot(enemy_rank, NUM_RANKS + 1)[..., 1:].permute(0, 3, 1, 2)
    obs[:, CH_ENEMY_HIDDEN] = hidden
    obs[:, CH_LAKE] = lake

    for i, it in enumerate(items):
        own_revealed = getattr(it, 'own_revealed', None)
        if own_revealed is not None:
            obs[i, CH_OWN_REVEALED] = torch.as_tensor(own_revealed, device=device).float() * (own_rank[i] > 0)
        obs[i, CH_OBSERVABILITY] = float(getattr(it, 'observability', 0.0) or 0.0)
        obs[i, CH_NO_BATTLE] = _ratio(getattr(it, 'plies_since_battle', 0), getattr(it, 'no_battle_draw_plies', 0))
        obs[i, CH_PROGRESS] = _ratio(getattr(it, 'turn_count', 0), getattr(it, 'max_turns', 0))

    second = (sign.view(n) < 0)
    if second.any():
        obs[second] = obs[second].flip(-2, -1)  # rotate Player 2's view by 180 degrees
    return obs


def encode_observation(item, player, device=None) -> torch.Tensor:
    """(NUM_OBS_PLANES, 10, 10) observation of one GameState or bare board for `player`."""
    return encode_observations([item], player, device=device)[0]


def to_canonical(planes: torch.Tensor, player) -> torch.Tensor:
    """Rotate (..., 10, 10) planes given in board coordinates into `player`'s canonical frame."""
    return planes if is_first_player(player) else planes.flip(-2, -1)
