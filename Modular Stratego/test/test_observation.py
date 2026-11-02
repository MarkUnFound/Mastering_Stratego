"""The network's view of the game: 30 planes in the mover's canonical frame.

The thesis encoding had 15 planes and one plane for every enemy piece, so revealed enemy ranks
were dropped (and full observability showed the network nothing more than fog of war).
Plane layout (environment/observation.py):
    0-11 own by rank, 12-23 known enemy by rank, 24 unknown enemy, 25 lakes,
    26 own pieces the enemy knows, 27 observability flag, 28 no-battle counter, 29 progress.
"""

import torch

from board import HIDDEN_PIECE, LAKE_SQUARE
from drqn_agent import RainbowAgent
from environment import StrategoEnvironment
from piece import PieceType

NUM_PLANES = 30
OWN, ENEMY_KNOWN, ENEMY_HIDDEN, LAKE, OWN_REVEALED, OBSERVABILITY, NO_BATTLE, PROGRESS = 0, 12, 24, 25, 26, 27, 28, 29


def _agent(cpu, player=1):
    return RainbowAgent(player_id=player, device=cpu, num_envs=1, use_pbs=False)


def _rank_plane(base, piece_type):
    return base + piece_type.value - 1


def test_observation_has_30_planes(cpu, tiny_network):
    agent = _agent(cpu)
    x = agent.get_state_representation(StrategoEnvironment(cpu).reset())
    assert x.shape == (NUM_PLANES, 10, 10)
    assert agent.q_network.conv_in.in_channels == NUM_PLANES


def test_revealed_enemy_ranks_reach_the_network(cpu, tiny_network):
    agent = _agent(cpu)
    board = torch.zeros(10, 10, dtype=torch.int8)
    board[0, 0] = -PieceType.MARSHAL.value   # revealed enemy Marshal
    board[0, 1] = -PieceType.SCOUT.value     # revealed enemy Scout
    board[0, 2] = HIDDEN_PIECE               # enemy of unknown rank
    board[9, 0] = PieceType.SPY.value        # own Spy
    board[4, 2] = LAKE_SQUARE
    x = agent.get_state_representation(board)
    assert x[_rank_plane(ENEMY_KNOWN, PieceType.MARSHAL), 0, 0] == 1
    assert x[_rank_plane(ENEMY_KNOWN, PieceType.SCOUT), 0, 1] == 1
    assert x[ENEMY_HIDDEN, 0, 2] == 1 and x[ENEMY_KNOWN:ENEMY_KNOWN + 12, 0, 2].sum() == 0
    assert x[_rank_plane(OWN, PieceType.SPY), 9, 0] == 1
    assert x[LAKE, 4, 2] == 1
    # every square holds at most one piece plane
    assert x[:LAKE + 1].sum(0).max() == 1


def test_fog_of_war_hides_enemy_ranks_and_full_observability_shows_them(cpu, tiny_network):
    agent = _agent(cpu)
    fog = StrategoEnvironment(cpu).reset()
    x = agent.get_state_representation(fog)
    assert x[ENEMY_KNOWN:ENEMY_KNOWN + 12].sum() == 0
    assert x[ENEMY_HIDDEN].sum() == 40
    assert x[OBSERVABILITY].max() == 0

    full = StrategoEnvironment(cpu, full_observability=True).reset()
    x = agent.get_state_representation(full)
    assert x[ENEMY_KNOWN:ENEMY_KNOWN + 12].sum() == 40
    assert x[_rank_plane(ENEMY_KNOWN, PieceType.FLAG)].sum() == 1
    assert x[ENEMY_HIDDEN].sum() == 0
    assert x[OBSERVABILITY].min() == 1


def test_mixed_observability_shows_the_requested_share_of_enemy_ranks(cpu):
    """The thesis env looked for hidden enemies as 0 (they are HIDDEN_PIECE), so 50% showed none."""
    state = StrategoEnvironment(cpu, full_observability=0.5).reset()
    hidden = int((state.board == HIDDEN_PIECE).sum())
    shown = int(((state.board < 0) & (state.board > LAKE_SQUARE)).sum())
    assert (hidden, shown) == (20, 20)


def test_both_sides_see_mirrored_positions_identically(cpu, tiny_network):
    """Player 2's view is rotated 180 degrees and sign-flipped: the same position seen from
    either side, with the roles swapped, gives the same planes."""
    p1, p2 = _agent(cpu, 1), _agent(cpu, -1)
    board = torch.zeros(10, 10, dtype=torch.int8)
    for r, c in [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]:
        board[r, c] = LAKE_SQUARE
    board[9, 1] = PieceType.FLAG.value
    board[6, 3] = PieceType.MARSHAL.value
    board[1, 8] = HIDDEN_PIECE
    board[2, 5] = -PieceType.GENERAL.value
    mirrored = torch.zeros_like(board)
    for r in range(10):
        for c in range(10):
            v = int(board[r, c])
            mirrored[9 - r, 9 - c] = v if v in (LAKE_SQUARE, HIDDEN_PIECE, 0) else -v
    assert torch.equal(p1.get_state_representation(board), p2.get_state_representation(mirrored))


def test_player_two_sees_its_own_pieces_at_the_bottom(cpu, tiny_network):
    state = StrategoEnvironment(cpu).reset(starting_player=-1)
    x = _agent(cpu, -1).get_state_representation(state)
    own = x[OWN:OWN + 12].sum(0)
    assert own[6:].sum() == 40 and own[:4].sum() == 0
    assert x[ENEMY_HIDDEN, :4].sum() == 40


def _play_until_battle(env, rng):
    for _ in range(2000):
        moves = env.get_valid_moves()
        attacks = [m for m in moves if env.board.actual_board[m[1]].item() != 0]
        move = rng.choice(attacks) if attacks else rng.choice(moves)
        state, _, done, info = env.step(move)
        if info.get('revealed_in_step'):
            return state, info
        assert not done
    raise AssertionError("no battle")


def test_own_pieces_the_enemy_has_seen_are_marked(cpu, tiny_network):
    import random
    rng = random.Random(3)
    env = StrategoEnvironment(cpu)
    env.reset()
    marked = 0
    for _ in range(10):
        state, info = _play_until_battle(env, rng)
        player = state.current_player
        x = _agent(cpu, player).get_state_representation(state)
        actual = env.board.actual_board
        enemy_view = env.board.get_visible_board(-player)
        expected = ((actual * player >= 1) & (actual * player <= 12) & (enemy_view == actual)).float()
        if player != 1:
            expected = expected.flip(0, 1)
        assert torch.equal(x[OWN_REVEALED], expected)
        marked += int(expected.sum())
        if env.game_over:
            break
    assert marked > 0


def test_counter_planes(cpu, tiny_network):
    env = StrategoEnvironment(cpu, max_turns=200, no_battle_draw_plies=100)
    state = env.reset()
    for _ in range(10):
        moves = [m for m in env.get_valid_moves() if env.board.actual_board[m[1]].item() == 0]
        state, _, _, _ = env.step(moves[0])
    x = _agent(cpu, state.current_player).get_state_representation(state)
    assert torch.allclose(x[NO_BATTLE], torch.full((10, 10), 10 / 100))
    assert torch.allclose(x[PROGRESS], torch.full((10, 10), 10 / 200))


def test_own_revealed_mask_follows_direct_board_edits(cpu):
    """The mask is cached per position; editing a board from outside must not serve a stale one."""
    env = StrategoEnvironment(cpu)
    env.reset()
    assert env._get_game_state().own_revealed.sum() == 0
    r, c = 6, 0
    env.board.visible_board_p2[r, c] = env.board.actual_board[r, c]  # Player 2 now knows this piece
    assert env._get_game_state().own_revealed[r, c]
