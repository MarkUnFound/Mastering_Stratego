"""The GUI's rules engine (Python Stratego Game/stratego.py) plays by the training environment's
rules: games played in the environment are replayed on the GUI board, which must offer the same
legal moves and reach the same position after every move. The GUI bot's network input for a
replayed position is the environment's own observation of it."""
import collections
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
from observation import encode_observations, CH_OWN_REVEALED, CH_NO_BATTLE, CH_PROGRESS  # noqa: E402
from setup_agent_integration import GUI_RANK  # noqa: E402
from training_config import DEFAULT_MAX_TURNS, EVAL_NO_BATTLE_DRAW_PLIES  # noqa: E402

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


def replayed_plies(seeds, **env_kwargs):
    """Environment games with heuristic setups and attack-biased random moves, replayed on a GUI
    board. Yields (env, state, gui, move) before each move is played on both, then (env, state,
    gui, None) once the game is over."""
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        env = StrategoEnvironment(torch.device('cpu'), **env_kwargs)
        place = {p: HeuristicSetupAgent(player_id=p).place_pieces(env._generate_pieces(),
                                                                  env.get_valid_placement_positions(p))
                 for p in (1, -1)}
        state = env.reset(place[1], place[-1], starting_player=1 if seed % 2 == 0 else -1)
        gui = S.Board(max_turns=env.max_turns, no_battle_draw_plies=env.no_battle_draw_plies)
        gui.last_winner = None
        for p in (1, -1):
            for pt, pos in place[p]:
                gui.set(tuple(pos), S.Piece(owner=OWNER[p], rank=GUI_RANK[pt]))
        done = False
        while not done:
            env_moves = sorted((tuple(a), tuple(b)) for a, b in env.get_valid_moves())
            attacks = [m for m in env_moves if gui.get(m[1]) is not None]
            move = random.choice(attacks) if attacks and random.random() < 0.5 else random.choice(env_moves)
            yield env, state, gui, move
            _, gui_winner = gui.move_and_resolve(*move)
            state, _, done, info = env.step(move)
            gui.last_winner = gui_winner
        yield env, state, gui, None


ENV_WINNER = {1: 1, 2: -1, 0: 0}  # GUI winner (owner, 0 = draw) -> environment winner


def check_the_game_ends_together(env, gui, move):
    """The GUI ends the game on the same move as the environment, with the same result and rule."""
    if move is not None:
        assert gui.last_winner is None, (env.turn_count, gui.end_reason)
        return None
    assert gui.last_winner is not None, (env.turn_count, env.win_type)
    assert (ENV_WINNER[gui.last_winner], gui.end_reason) == (env.winner, env.win_type)
    return gui.end_reason


def test_gui_engine_matches_the_environment():
    battles = flags = 0
    for env, state, gui, move in replayed_plies(range(6), full_observability=True, max_turns=400,
                                                no_battle_draw_plies=0):
        if check_the_game_ends_together(env, gui, move) == 'flag_capture':
            flags += 1
        if move is None:
            continue
        if env.turn_count == 0:
            assert gui_values(gui) == env.board.actual_board.tolist()
        player = state.current_player
        env_moves = {(tuple(a), tuple(b)) for a, b in env.get_valid_moves()}
        gui_moves = {(src, dst) for src in gui.owner_positions(OWNER[player])
                     for dst in gui.legal_moves_from(src)}
        assert gui_moves == env_moves, (env.turn_count,)
        # the GUI counts plies and plies without a battle as the environment does
        assert (gui.turn_count, gui.plies_since_battle) == (env.turn_count, env.plies_since_battle)
        battles += gui.get(move[1]) is not None
        assert gui_values(gui) == env.board.actual_board.tolist(), (env.turn_count, move)
    assert battles > 50 and flags >= 1, (battles, flags)


def test_the_gui_ends_games_by_the_environments_rules():
    """No-battle draws, the turn limit and a side left without a legal move end a GUI game as they
    end a training game (the GUI only knew captured flags: long games went on, planes 28-29 sat at
    1, and a side with nothing left to move got stuck)."""
    assert (S.NO_BATTLE_DRAW_PLIES, S.MAX_TURNS) == (EVAL_NO_BATTLE_DRAW_PLIES, DEFAULT_MAX_TURNS)
    ends = collections.Counter()
    for limits in ({'no_battle_draw_plies': 30, 'max_turns': 1000}, {'no_battle_draw_plies': 0, 'max_turns': 60}):
        for env, state, gui, move in replayed_plies(range(4), full_observability=True, **limits):
            reason = check_the_game_ends_together(env, gui, move)
            if reason:
                ends[reason] += 1
    assert ends['no_battle_draw'] >= 1 and ends['timeout'] >= 1, ends


def test_a_side_with_no_legal_move_loses():
    b = S.Board()
    b.set((9, 0), S.Piece(owner=1, rank=-1))   # Flag
    b.set((9, 1), S.Piece(owner=1, rank=0))    # Bomb
    b.set((8, 0), S.Piece(owner=1, rank=0))    # Bomb
    b.set((6, 5), S.Piece(owner=1, rank=4))    # Sergeant: its last movable piece
    b.set((5, 5), S.Piece(owner=2, rank=10))   # Marshal
    b.set((0, 0), S.Piece(owner=2, rank=-1))
    msg, winner = b.move_and_resolve((5, 5), (6, 5))
    assert (winner, b.end_reason) == (2, 'no_moves')
    assert S.result_message(winner, b.end_reason, human_side=1) == "Bot wins: the other side cannot move"
    assert S.result_message(0, 'no_battle_draw') == "Draw: 200 moves without a battle"


class EncodingAgent:
    """Stands in for RainbowAgent.get_state_representation with AAREN off: the 30 planes only."""
    history = None

    def get_state_representation(self, board, pbs_instance=None, player=None):
        return encode_observations([board], player)[0]


def test_the_gui_bots_input_is_the_environments_observation():
    """All 30 planes, fog of war, with the limits the agent is evaluated under. Before this the
    GUI bot left planes 26-29 at 0: it never knew which of its pieces the human had seen, and
    every position looked like the first move."""
    import dqn_bot_logic as D
    bot = D.DQNBotLogic.__new__(D.DQNBotLogic)
    bot.agent = EncodingAgent()
    checked = 0
    for env, state, gui, move in replayed_plies(range(3), full_observability=False,
                                                max_turns=DEFAULT_MAX_TURNS,
                                                no_battle_draw_plies=EVAL_NO_BATTLE_DRAW_PLIES):
        if move is None or env.turn_count % 7:
            continue
        player = state.current_player
        bot.player_id = OWNER[player]
        ours = bot._get_state_representation(gui, OWNER[player])
        theirs = encode_observations([state], player)[0]
        assert torch.equal(ours, theirs), (env.turn_count, [i for i in range(30) if not torch.equal(ours[i], theirs[i])])
        checked += 1
        if env.turn_count > 50:
            assert ours[CH_PROGRESS].max() > 0
        if ours[CH_OWN_REVEALED].any() and ours[CH_NO_BATTLE].max() > 0:
            checked += 1000  # a position with both a seen piece and a running no-battle counter
    assert checked > 1000, checked
