"""The GUI's rules engine (Python Stratego Game/stratego.py) plays by the training environment's
rules: games played in the environment are replayed on the GUI board, which must offer the same
legal moves and reach the same position after every move."""
import os
import random
import sys

import torch

GUI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'Python Stratego Game')
sys.path.insert(0, GUI)

import stratego as S  # noqa: E402
from environment import StrategoEnvironment, HeuristicSetupAgent  # noqa: E402
from board import LAKE_SQUARE  # noqa: E402
from setup_agent_integration import GUI_RANK  # noqa: E402

OWNER = {1: 1, -1: 2}  # environment player -> GUI owner


def gui_values(board):
    """The GUI board in the environment's true-board values."""
    from dqn_bot_logic import GUI_RANK_TO_PIECETYPE
    out = []
    for r in range(10):
        row = []
        for c in range(10):
            p = board.get((r, c))
            if p is None:
                row.append(LAKE_SQUARE if board.is_lake(r, c) else 0)
            else:
                v = GUI_RANK_TO_PIECETYPE[p.rank].value
                row.append(v if p.owner == 1 else -v)
        out.append(row)
    return out


def test_gui_engine_matches_the_environment():
    battles = flags = 0
    for seed in range(6):
        random.seed(seed)
        torch.manual_seed(seed)
        env = StrategoEnvironment(torch.device('cpu'), full_observability=True, max_turns=400,
                                  no_battle_draw_plies=0)
        place = {p: HeuristicSetupAgent(player_id=p).place_pieces(env._generate_pieces(),
                                                                  env.get_valid_placement_positions(p))
                 for p in (1, -1)}
        state = env.reset(place[1], place[-1], starting_player=1 if seed % 2 == 0 else -1)
        gui = S.Board()
        for p in (1, -1):
            for pt, pos in place[p]:
                gui.set(tuple(pos), S.Piece(owner=OWNER[p], rank=GUI_RANK[pt]))
        assert gui_values(gui) == env.board.actual_board.tolist()

        done, info = False, {}
        while not done:
            player = state.current_player
            env_moves = {(tuple(a), tuple(b)) for a, b in env.get_valid_moves()}
            gui_moves = {(src, dst) for src in gui.owner_positions(OWNER[player])
                         for dst in gui.legal_moves_from(src)}
            assert gui_moves == env_moves, (seed, env.turn_count)
            attacks = [m for m in env_moves if gui.get(m[1]) is not None]
            move = random.choice(sorted(attacks)) if attacks and random.random() < 0.5 \
                else random.choice(sorted(env_moves))
            battles += gui.get(move[1]) is not None
            _, gui_winner = gui.move_and_resolve(*move)
            state, _, done, info = env.step(move)
            assert gui_values(gui) == env.board.actual_board.tolist(), (seed, env.turn_count, move)
            if gui_winner is not None:
                flags += 1
                assert done and info['winner'] == (1 if gui_winner == 1 else -1)
    assert battles > 50 and flags >= 1, (battles, flags)
