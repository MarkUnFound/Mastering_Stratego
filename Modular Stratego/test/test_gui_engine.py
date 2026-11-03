"""The GUI's rules engine (Python Stratego Game/stratego.py) plays by the training environment's
rules: games played in the environment are replayed on the GUI board, which must offer the same
legal moves and reach the same position after every move. The GUI bot's network input for a
replayed position is the environment's own observation of it."""
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
        gui = S.Board()
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


def test_gui_engine_matches_the_environment():
    battles = flags = 0
    for env, state, gui, move in replayed_plies(range(6), full_observability=True, max_turns=400,
                                                no_battle_draw_plies=0):
        if move is None:
            if gui.last_winner is not None:
                flags += 1
                assert env.game_over and env.winner == (1 if gui.last_winner == 1 else -1)
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
