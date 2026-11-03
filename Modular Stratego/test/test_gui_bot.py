"""The GUI bot (Python Stratego Game/dqn_bot_logic.py) gives the network the board the training
environment would: PieceType values, the human's unrevealed pieces hidden, and a search whose
human replies fight real battles with what the bot can know."""
import collections
import os
import random
import sys
import time

import pytest
import torch

GUI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'Python Stratego Game')
sys.path.insert(0, GUI)

import dqn_bot_logic as D  # noqa: E402
import stratego as S  # noqa: E402
from piece import PieceType  # noqa: E402
from board import HIDDEN_PIECE  # noqa: E402

ARMY = {PieceType.MARSHAL: 1, PieceType.GENERAL: 1, PieceType.COLONEL: 2, PieceType.MAJOR: 3,
        PieceType.CAPTAIN: 4, PieceType.LIEUTENANT: 4, PieceType.SERGEANT: 4, PieceType.MINER: 5,
        PieceType.SCOUT: 8, PieceType.SPY: 1, PieceType.BOMB: 6, PieceType.FLAG: 1}


def adapter(player_id=2):
    bot = D.DQNBotLogic.__new__(D.DQNBotLogic)  # the board conversion needs no network
    bot.player_id, bot.opponent_id = player_id, 3 - player_id
    return bot


def full_board(seed=0):
    random.seed(seed)
    b = S.Board()
    S.auto_setup(b, 1)
    S.auto_setup(b, 2)
    return b


def test_the_bot_reads_its_own_army_in_piece_type_values():
    """The thesis code copied the GUI ranks (Spy 1 ... Marshal 10, Bomb 0, Flag -1): the network
    saw 8 Spies, 2 Flags, no Marshal and no Bombs."""
    b = full_board()
    grid = adapter(2)._gui_board_to_marq_grid(b)
    own = collections.Counter(PieceType(-int(grid[p])) for p in b.owner_positions(2))
    assert own == ARMY
    grid1 = adapter(1)._gui_board_to_marq_grid(b)
    assert collections.Counter(PieceType(int(grid1[p])) for p in b.owner_positions(1)) == ARMY


def test_the_other_sides_unrevealed_pieces_are_hidden():
    b = full_board()
    human = b.owner_positions(1)
    shown = human[0]
    b.get(shown).revealed = True
    grid = adapter(2)._gui_board_to_marq_grid(b)
    assert all(grid[p] == HIDDEN_PIECE for p in human if p != shown)
    assert grid[shown] == D.GUI_RANK_TO_PIECETYPE[b.get(shown).rank].value
    # and the human's view (the recommendation feature) hides the bot's army
    assert all(adapter(2)._gui_board_to_marq_grid(b, 1)[p] == HIDDEN_PIECE for p in b.owner_positions(2))


class StubAgent:
    device = torch.device('cpu')
    history = None


def two_piece_board(attacker, defender, revealed=False):
    b = S.Board()
    b.set((5, 0), S.Piece(owner=1, rank=attacker, revealed=revealed))
    b.set((4, 0), S.Piece(owner=2, rank=defender))
    return b


def test_a_human_attack_in_the_search_is_a_battle():
    """The thesis search moved the human's piece onto the defender: every attack won."""
    search = D.ExpectamaxSearch(StubAgent())
    board = two_piece_board(attacker=2, defender=10, revealed=True)  # revealed Scout hits the Marshal
    after = search._simulate_opponent_move(board, ((5, 0), (4, 0)))
    assert after.get((4, 0)).rank == 10 and after.get((5, 0)) is None
    board = two_piece_board(attacker=10, defender=0, revealed=True)  # Marshal hits a Bomb
    after = search._simulate_opponent_move(board, ((5, 0), (4, 0)))
    assert after.get((4, 0)).rank == 0 and after.get((5, 0)) is None


def test_the_search_moves_the_humans_pieces_by_what_the_bot_knows():
    """An unrevealed piece may be anything movable (a Bomb gives itself away no more); a revealed
    one moves by its rules. The thesis search moved the human's true pieces."""
    search = D.ExpectamaxSearch(StubAgent())
    b = S.Board()
    b.set((9, 0), S.Piece(owner=1, rank=0))                  # unrevealed Bomb
    b.set((9, 9), S.Piece(owner=1, rank=0, revealed=True))   # revealed Bomb
    b.set((6, 5), S.Piece(owner=1, rank=2))                  # unrevealed Scout: one step only
    moves = search._opponent_moves(b, 1)
    assert {m for m in moves if m[0] == (9, 0)} == {((9, 0), (8, 0)), ((9, 0), (9, 1))}
    assert not [m for m in moves if m[0] == (9, 9)]
    assert {m[1] for m in moves if m[0] == (6, 5)} == {(5, 5), (7, 5), (6, 4), (6, 6)}
    # an unrevealed attacker fights as the most likely movable type of the setup prior (Scout)
    board = two_piece_board(attacker=10, defender=3)          # hidden Marshal hits a Miner
    after = search._simulate_opponent_move(board, ((5, 0), (4, 0)))
    assert after.get((4, 0)).rank == 3


def test_the_bot_chooses_a_legal_move_without_reading_the_humans_army(cpu, tiny_network, monkeypatch):
    from network import RainbowAgent
    bot = adapter(2)
    bot.device = cpu
    bot.agent = RainbowAgent(player_id=2, device=cpu, use_pbs=True, buffer_size=100)
    bot.expectamax = D.ExpectamaxSearch(bot.agent, max_depth=2, max_time=60.0)
    seen = []
    original = bot._gui_board_to_marq_grid
    monkeypatch.setattr(bot, '_gui_board_to_marq_grid',
                        lambda board, viewer=None: seen.append(original(board, viewer)) or seen[-1])
    b = full_board(1)
    move = bot.choose_move(b, 2)
    assert move in bot._get_gui_legal_moves(b, 2)
    # Player 1's pieces are positive values: none of the human's (all unrevealed) may show
    assert seen and all(g.max() <= 0 for g in seen), "a network input showed an unrevealed human piece"
