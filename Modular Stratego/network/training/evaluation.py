"""
Evaluation games: the learning agent plays greedily (NoisyNet off) against a fixed opponent,
outside the training lanes. The curriculum advances on these results, not on training games
(which mix in exploration noise and accumulate over the whole phase).
"""

from typing import Dict, Optional

from environment import StrategoEnvironment
from heuristic_setup import HeuristicSetupAgent
from training_config import EVAL_NO_BATTLE_DRAW_PLIES


def play_evaluation_games(agent, opponent, num_games: int, device, full_observability=False,
                          max_turns: Optional[int] = None, no_battle_draw_plies: int = EVAL_NO_BATTLE_DRAW_PLIES
                          ) -> Dict[str, float]:
    """
    `agent` (Player 1) plays `num_games` games against `opponent` (Player 2). Both sides use the
    heuristic setup, as in training; the starting player alternates. The agent acts greedily on
    its mean weights and reads no AAREN history (a lane's history belongs to its training game).

    Returns wins, losses, draws, games and win_rate = wins / games (draws count as non-wins).
    """
    env = StrategoEnvironment(device, full_observability=full_observability, max_turns=max_turns,
                              no_battle_draw_plies=no_battle_draw_plies)
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}
    noisy = getattr(agent, 'noisy_acting', False)
    agent.noisy_acting = False
    wins = losses = draws = 0
    try:
        for game in range(num_games):
            placements = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p))
                          for p in (1, -1)}
            state = env.reset(placements[1], placements[-1], starting_player=1 if game % 2 == 0 else -1)
            if hasattr(opponent, 'reset_history'):
                opponent.reset_history()
            done, info = False, {}
            while not done:
                moves = env.get_valid_moves()
                if not moves:
                    break  # the environment ends a game without moves; guard against a stuck loop
                actor = agent if state.current_player == 1 else opponent
                action = actor.act_batch([state.board], [moves], [state], env_indices=[None],
                                         full_observability=full_observability)[0]
                state, _, done, info = env.step(action)
            winner = info.get('winner', 0) if done else 0
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws += 1
    finally:
        agent.noisy_acting = noisy
    return {'wins': wins, 'losses': losses, 'draws': draws, 'games': num_games,
            'win_rate': wins / max(1, num_games)}
