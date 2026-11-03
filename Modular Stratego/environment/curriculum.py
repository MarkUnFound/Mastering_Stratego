"""
Curriculum Learning Module for Stratego DQN Training

Implements 5-phase curriculum with automatic phase transitions:
- Phase 1: Physics of War (Full Observability)
- Phase 2: Memory Gap (Partial Observability, Frozen Heuristic)
- Phase 3: Self-Play (Nash Ascent)
- Phase 4: League Training (Ensemble)
- Phase 5: Scenario Drills (Endgame Specialization)
"""

import os
import json
import random
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import IntEnum

import training_config
from training_config import (
    PHASE_MAX_TURNS, DEFAULT_MAX_TURNS,
    PHASE_1_WIN_THRESHOLD_RANDOM, PHASE_1_WIN_THRESHOLD_HEURISTIC,
    PHASE_2_WIN_THRESHOLD, PHASE_3_WIN_THRESHOLD
)
from battle import battle_from_values, is_enemy_value
from piece import PieceType

# Training games per opponent that PFSP looks back over
PFSP_WINDOW = 100


class TrainingPhase(IntEnum):
    """Training curriculum phases."""
    PHYSICS_OF_WAR = 1      # Full observability, learn basics
    MEMORY_GAP = 2          # Fog of war, use memory (PBS)
    SELF_PLAY = 3           # Play against past self
    LEAGUE_TRAINING = 4     # Diverse opponent pool (the last phase of the sequence)
    SCENARIO_DRILLS = 5     # Endgame specialization (not part of the 1 -> 4 sequence)


def phase_gates(phase) -> Dict[str, float]:
    """Evaluation gates of a phase: opponent -> win rate (wins / games, draws count as non-wins)
    the greedy agent must reach in evaluation games to advance. Empty: the phase does not end on
    evaluation. Read from training_config when asked, so overrides take effect."""
    tc = training_config
    return {
        TrainingPhase.PHYSICS_OF_WAR: {'random': tc.PHASE_1_WIN_THRESHOLD_RANDOM,
                                       'heuristic': tc.PHASE_1_WIN_THRESHOLD_HEURISTIC},
        # Phases 2 and 3 are gated on the heuristic, the stronger scripted opponent (40 games each
        # way: heuristic 16W/5L as P1 and 19W/5L as P2 against smart_heuristic). The thesis gates
        # were on smart_heuristic, which the formal_1 run could only draw against.
        TrainingPhase.MEMORY_GAP: {'heuristic': tc.PHASE_2_WIN_THRESHOLD},
        TrainingPhase.SELF_PLAY: {'heuristic': tc.PHASE_3_WIN_THRESHOLD},
    }.get(TrainingPhase(phase), {})


def phase_episode_cap(phase) -> int:
    """Training episodes after which a phase ends even if its gates are not met (0 = no cap)."""
    return int(getattr(training_config, f'PHASE_{int(phase)}_MAX_EPISODES', 0) or 0)


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    name: str
    full_observability: bool
    min_episodes: int
    max_episodes: int
    opponents: List[str]
    reward_focus: str  # Which reward component to emphasize
    success_metrics: Dict[str, float]
    max_turns: int = 0  # Max steps per episode (0 = use DEFAULT_MAX_TURNS)


# PHASE CONFIGURATIONS
# A phase ends when the greedy agent passes its evaluation gates (phase_gates) or after
# PHASE_<n>_MAX_EPISODES training episodes (phase_episode_cap). min_episodes / max_episodes and
# success_metrics below are for display only.
PHASE_CONFIGS = {
    TrainingPhase.PHYSICS_OF_WAR: PhaseConfig(
        name="Physics of War",
        full_observability=True,
        min_episodes=0,  # NOT ENFORCED - dynamic transition
        max_episodes=0,  # NOT ENFORCED - dynamic transition
        opponents=["random", "heuristic", "smart_heuristic"],
        reward_focus="material",
        success_metrics={  # evaluation gates: see phase_gates()
            "random": PHASE_1_WIN_THRESHOLD_RANDOM,
            "heuristic": PHASE_1_WIN_THRESHOLD_HEURISTIC
        },
        max_turns=PHASE_MAX_TURNS.get(1, DEFAULT_MAX_TURNS),
    ),
    TrainingPhase.MEMORY_GAP: PhaseConfig(
        name="Memory Gap",
        full_observability=False,
        min_episodes=0,  # NOT ENFORCED
        max_episodes=0,  # NOT ENFORCED
        opponents=["frozen_heuristic", "smart_heuristic"],
        reward_focus="epistemic",
        success_metrics={  # evaluation gates: see phase_gates()
            "heuristic": PHASE_2_WIN_THRESHOLD
        },
        max_turns=PHASE_MAX_TURNS.get(2, DEFAULT_MAX_TURNS),
    ),
    TrainingPhase.SELF_PLAY: PhaseConfig(
        name="Simple Self-Play",
        full_observability=False,
        min_episodes=0,  # NOT ENFORCED
        max_episodes=0,  # NOT ENFORCED
        opponents=["self_500", "smart_heuristic"],
        reward_focus="material",
        success_metrics={  # evaluation gates: see phase_gates()
            "heuristic": PHASE_3_WIN_THRESHOLD
        },
        max_turns=PHASE_MAX_TURNS.get(3, DEFAULT_MAX_TURNS),
    ),
    TrainingPhase.LEAGUE_TRAINING: PhaseConfig(
        name="League Training",
        full_observability=False,
        min_episodes=0,  # NOT ENFORCED
        max_episodes=0,  # NOT ENFORCED
        opponents=["league", "self"], # Pure MARL, no cheating heuristics
        reward_focus="balanced",
        success_metrics={
            # Dynamic criteria: ELO-based
            "league_elo": 1500,
            "win_rate_stable": True
        },
        max_turns=PHASE_MAX_TURNS.get(4, DEFAULT_MAX_TURNS),
    ),
    TrainingPhase.SCENARIO_DRILLS: PhaseConfig(
        name="Scenario Drills",
        full_observability=False,
        min_episodes=0,  # NOT ENFORCED
        max_episodes=0,  # NOT ENFORCED
        opponents=["scenario_specific"],
        reward_focus="positional",
        success_metrics={
            # Dynamic criteria: scenario completion
            "scenario_completion_rate": 0.80
        },
        max_turns=DEFAULT_MAX_TURNS,
    )
}


@dataclass
class PhaseMetrics:
    """Tracked metrics for phase transition decisions."""
    episodes_in_phase: int = 0
    total_wins: int = 0
    total_games: int = 0
    total_losses: int = 0
    wins_vs_random: int = 0
    losses_vs_random: int = 0
    games_vs_random: int = 0
    wins_vs_heuristic: int = 0
    losses_vs_heuristic: int = 0
    games_vs_heuristic: int = 0
    recent_win_rates: List[float] = None
    # Per-opponent adaptive tracking (PFSP)
    # Maps opponent_name -> {wins, losses, games, recent: last PFSP_WINDOW results (1 win, 0 not)}
    opponent_stats: Dict[str, Dict[str, Any]] = None
    # Latest evaluation in this phase: opponent -> win rate (wins / games)
    last_eval: Dict[str, float] = None
    eval_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recent_win_rates is None:
            self.recent_win_rates = []
        if self.opponent_stats is None:
            self.opponent_stats = {}
        if self.last_eval is None:
            self.last_eval = {}
        if self.eval_history is None:
            self.eval_history = []

    # ------------------------------------------------------------------
    # Aggregate win-rate helpers
    # ------------------------------------------------------------------
    @property
    def overall_win_rate(self) -> float:
        decisive_games = self.total_wins + self.total_losses
        return self.total_wins / max(1, decisive_games)

    @property
    def win_rate_vs_random(self) -> float:
        decisive_games = self.wins_vs_random + self.losses_vs_random
        return self.wins_vs_random / max(1, decisive_games)

    @property
    def win_rate_vs_heuristic(self) -> float:
        decisive_games = self.wins_vs_heuristic + self.losses_vs_heuristic
        return self.wins_vs_heuristic / max(1, decisive_games)

    # ------------------------------------------------------------------
    # Per-opponent helpers (PFSP)
    # ------------------------------------------------------------------
    def record_opponent_result(self, opponent_name: str, winner: int):
        """Record a win (1) or loss (-1) or draw (0) against a named opponent."""
        if opponent_name not in self.opponent_stats:
            self.opponent_stats[opponent_name] = {'wins': 0, 'losses': 0, 'games': 0}
        s = self.opponent_stats[opponent_name]
        s['games'] += 1
        if winner == 1:
            s['wins'] += 1
        elif winner == -1:
            s['losses'] += 1
        recent = s.setdefault('recent', [])
        recent.append(1 if winner == 1 else 0)
        del recent[:-PFSP_WINDOW]

    def win_rate_vs(self, opponent_name: str, min_games: int = 30) -> Optional[float]:
        """Win rate (draws count as non-wins) over the last PFSP_WINDOW games vs a specific
        opponent; None if fewer than min_games. (The thesis rate was decisive games only, over
        the whole phase, so an early losing streak weighed on PFSP for good.)"""
        s = self.opponent_stats.get(opponent_name)
        recent = s.get('recent', []) if s else []
        if len(recent) < min_games:
            return None
        return sum(recent) / len(recent)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PhaseMetrics':
        # Drop legacy PBS fields that were removed after PBS→AAREN migration.
        data.pop('pbs_accuracy_sum', None)
        data.pop('pbs_accuracy_count', None)
        return cls(**data)


class CurriculumManager:
    """
    Manages the 5-phase curriculum for Stratego training.
    
    Tracks progress, determines phase transitions, and provides
    phase-specific configurations for the training loop.
    """
    
    def __init__(self, start_phase: int = 1, save_dir: str = "dqn_models"):
        self.current_phase = TrainingPhase(start_phase)
        self.save_dir = save_dir
        self.phase_start_episode = 0
        self.total_episodes = 0
        
        # Metrics per phase
        self.metrics = {phase: PhaseMetrics() for phase in TrainingPhase}
        
        # Load saved state if exists
        self._load_state()
        
    def _save_path(self) -> str:
        return os.path.join(self.save_dir, "curriculum_state.json")
    
    def _load_state(self):
        """Load curriculum state from disk."""
        path = self._save_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.current_phase = TrainingPhase(data['current_phase'])
                self.phase_start_episode = data['phase_start_episode']
                self.total_episodes = data['total_episodes']
                
                for phase_val, metrics_data in data.get('metrics', {}).items():
                    phase = TrainingPhase(int(phase_val))
                    self.metrics[phase] = PhaseMetrics.from_dict(metrics_data)
                    
                print(f" Loaded curriculum state: Phase {self.current_phase.value} ({self.get_phase_config().name})")
            except Exception as e:
                print(f" Could not load curriculum state: {e}")
    
    def save_state(self):
        """Save curriculum state to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        data = {
            'current_phase': self.current_phase.value,
            'phase_start_episode': self.phase_start_episode,
            'total_episodes': self.total_episodes,
            'metrics': {
                phase.value: metrics.to_dict() 
                for phase, metrics in self.metrics.items()
            }
        }
        with open(self._save_path(), 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_phase_config(self) -> PhaseConfig:
        """Get configuration for current phase."""
        return PHASE_CONFIGS[self.current_phase]
    
    def get_max_turns(self) -> int:
        """Get max turns for the current curriculum phase."""
        config = self.get_phase_config()
        return config.max_turns if config.max_turns > 0 else DEFAULT_MAX_TURNS
    
    def should_use_full_observability(self) -> bool:
        """
        Check if current phase uses full observability.
        Legacy method - prefer get_observability_rate()
        """
        return self.get_observability_rate() > 0.0
    
    def get_observability_rate(self) -> float:
        """
        Get the percentage of hidden pieces to reveal (0.0 to 1.0).
        - 1.0: Full Observability (Physics of War)
        - 0.0: Partial Observability (Memory Gap+)
        - 0.3-0.8: Mixed Observability (Transition/Fog)
        """
        config = self.get_phase_config()
        
        # Base setting from config
        if not config.full_observability:
            return 0.0
            
        # Refined Phase 1 Logic:
        # If we are in Phase 1 (Physics of War) but getting good (win rate > 80%),
        # start introducing "Fog of War" to prepare for Phase 2.
        if self.current_phase == TrainingPhase.PHYSICS_OF_WAR:
            metrics = self.metrics[self.current_phase]
            win_rate = metrics.overall_win_rate
            
            # Transition Zone: 85% Win Rate -> Start hiding 50% of pieces
            if win_rate > 0.85:
                return 0.5  # 50% Fog
            
            # Late Stage: 75% Win Rate -> Start hiding 20% of pieces
            if win_rate > 0.75:
                return 0.8  # 20% Fog
                
        return 1.0
    
    # ------------------------------------------------------------------
    # PFSP Scheduler (Prioritized Fictitious Self-Play)
    # ------------------------------------------------------------------
    def _pfsp_weights(self, candidates: List[str]) -> Dict[str, float]:
        """
        Compute opponent sampling weights using inverse-mastery prioritization.

        For each candidate:
          - If win rate vs that opponent is >= 0.90 (mastered): weight = FLOOR (0.05)
          - Otherwise: weight = (1 - win_rate), ensuring harder opponents get more time.

        The result is normalized so weights sum to 1.0.
        This mirrors the PFSP scheme used in AlphaStar and OpenAI Five.
        """
        MASTERY_THRESHOLD = 0.90   # above this → de-prioritize
        MASTERY_FLOOR     = 0.05   # minimum sampling share even if mastered
        MIN_GAMES         = 30     # need this many games before PFSP kicks in

        metrics = self.metrics[self.current_phase]
        raw: Dict[str, float] = {}

        for opp in candidates:
            wr = metrics.win_rate_vs(opp, min_games=MIN_GAMES)
            if wr is None:
                # Not enough data yet → treat as max priority (1.0) to encourage exploration
                raw[opp] = 1.0
            elif wr >= MASTERY_THRESHOLD:
                raw[opp] = MASTERY_FLOOR
            else:
                raw[opp] = max(MASTERY_FLOOR, 1.0 - wr)

        total = sum(raw.values())
        weights = {opp: w / total for opp, w in raw.items()}
        
        # --- PFSP MINIMUM FLOORING ---
        # Ensure the agent never entirely drops the easiest opponent ('random').
        # This prevents catastrophic forgetting of basic mechanics when faced
        # with an overwhelming wall like 'heuristic'.
        if 'random' in weights and weights['random'] < 0.15:
            deficit = 0.15 - weights['random']
            weights['random'] = 0.15
            
            other_opps = [o for o in weights if o != 'random']
            if other_opps:
                other_total = sum(weights[o] for o in other_opps)
                if other_total > 0:
                    for o in other_opps:
                        weights[o] -= deficit * (weights[o] / other_total)
        
        return weights

    def get_opponent_distribution(self) -> Dict[str, float]:
        """
        Get opponent selection probabilities for current phase.

        Phase 1 uses PFSP-based adaptive difficulty: opponents the agent has mastered
        (>= 90% decisive win rate) are de-prioritized; those it struggles against
        receive proportionally more matchups.

        Other phases retain fixed distributions.
        """
        config = self.get_phase_config()
        metrics = self.metrics[self.current_phase]

        if self.current_phase == TrainingPhase.PHYSICS_OF_WAR:
            # --- PFSP adaptive scheduling ---
            # Recent training win rates (draws count as non-wins) decide the candidate pool
            win_vs_heur = metrics.win_rate_vs('heuristic', min_games=20) or 0.0

            # Decide the active candidate pool based on overall progress
            # The heuristic is in the pool from the start: the Phase-1 gate needs wins against it,
            # and the thesis pool held it back until 70% training wins against random, which the
            # agent may never reach in its turn limit (the phase then only ended at its cap).
            if win_vs_heur < 0.30:
                candidates = ["random", "heuristic"]
            else:
                # Agent is competent — full pool, PFSP distributes weight
                candidates = ["random", "heuristic", "smart_heuristic"]

            return self._pfsp_weights(candidates)

        elif self.current_phase == TrainingPhase.MEMORY_GAP:
            return self._pfsp_weights(["frozen_heuristic", "smart_heuristic"])

        elif self.current_phase == TrainingPhase.SELF_PLAY:
            return self._pfsp_weights(["self_500", "smart_heuristic"])

        elif self.current_phase == TrainingPhase.LEAGUE_TRAINING:
            return self._pfsp_weights(config.opponents)

        elif self.current_phase == TrainingPhase.SCENARIO_DRILLS:
            return {"scenario": 1.0}

        # Default fallback (Pure MARL)
        return {"self": 1.0}

    def sample_opponent(self, rng=None) -> str:
        """Opponent type for one game, drawn with the PFSP weights. (The thesis trainer cycled
        through the sorted types in fixed blocks, whatever the weights.)"""
        weights = self.get_opponent_distribution()
        types = sorted(weights)
        return (rng or random).choices(types, weights=[weights[t] for t in types])[0]

    # ------------------------------------------------------------------
    # Evaluation-based advancement
    # ------------------------------------------------------------------
    def evaluation_gates(self) -> Dict[str, float]:
        """Opponents (-> required win rate) the agent is evaluated against in this phase."""
        return phase_gates(self.current_phase)

    def record_evaluation(self, results: Dict[str, float]):
        """Store an evaluation of the greedy agent in the current phase: opponent -> win rate."""
        metrics = self.metrics[self.current_phase]
        metrics.last_eval = dict(results)
        metrics.eval_history.append({'episode': self.total_episodes,
                                     'episodes_in_phase': metrics.episodes_in_phase,
                                     'results': dict(results)})
    
    def update_metrics(self, episode_result: Dict):
        """
        Update metrics after an episode.
        
        Args:
            episode_result: Dict with keys like 'winner', 'opponent_type', 'pbs_accuracy'
        """
        metrics = self.metrics[self.current_phase]
        metrics.episodes_in_phase += 1
        metrics.total_games += 1
        self.total_episodes += 1
        
        winner = episode_result.get('winner', 0)
        opponent_type = episode_result.get('opponent_type', 'unknown')
        
        if winner == 1:  # Agent1 wins
            metrics.total_wins += 1
            
            # Track wins by opponent type (true_random counts as random for curriculum purposes)
            if opponent_type in ['random', 'true_random']:
                metrics.wins_vs_random += 1
            elif opponent_type in ['heuristic', 'greedy', 'frozen_heuristic', 'smart_heuristic']:
                metrics.wins_vs_heuristic += 1
        elif winner == -1:  # Agent1 loses / Agent2 wins
            metrics.total_losses += 1
            
            if opponent_type in ['random', 'true_random']:
                metrics.losses_vs_random += 1
            elif opponent_type in ['heuristic', 'greedy', 'frozen_heuristic', 'smart_heuristic']:
                metrics.losses_vs_heuristic += 1
                
        # Track games played by opponent type
        if opponent_type in ['random', 'true_random']:
            metrics.games_vs_random += 1
        elif opponent_type in ['heuristic', 'greedy', 'frozen_heuristic', 'smart_heuristic']:
            metrics.games_vs_heuristic += 1

        # Per-opponent PFSP tracking
        metrics.record_opponent_result(opponent_type, winner)

        # Track recent win rates (sliding window)
        metrics.recent_win_rates.append(1 if winner == 1 else 0)
        if len(metrics.recent_win_rates) > 100:
            metrics.recent_win_rates.pop(0)
    
    def check_phase_transition(self) -> bool:
        """
        Whether to advance to the next phase: the latest evaluation in this phase passes every
        gate of phase_gates(), or the phase has run phase_episode_cap() training episodes.
        League Training (the last phase) does not end. The reason is in last_transition_reason.

        (The thesis check used training games only, counted draws out of the win rate and
        accumulated over the whole phase; Phase 4 was the configured start.)
        """
        self.last_transition_reason = None
        if self.current_phase >= TrainingPhase.LEAGUE_TRAINING:
            return False
        metrics = self.metrics[self.current_phase]
        gates = self.evaluation_gates()
        if gates and all(metrics.last_eval.get(opp, -1.0) >= threshold for opp, threshold in gates.items()):
            self.last_transition_reason = 'evaluation ' + ', '.join(
                f'{opp} {metrics.last_eval[opp]:.0%} >= {threshold:.0%}' for opp, threshold in gates.items())
            return True
        cap = phase_episode_cap(self.current_phase)
        if cap and metrics.episodes_in_phase >= cap:
            self.last_transition_reason = f'episode cap {cap} reached'
            return True
        return False

    def advance_phase(self) -> bool:
        """
        Advance to the next curriculum phase.
        
        Returns:
            True if successfully advanced, False if already at final phase
        """
        if self.current_phase >= TrainingPhase.LEAGUE_TRAINING:
            # Don't auto-advance beyond League Training
            # Scenario Drills are periodic, not sequential
            return False
        
        old_phase = self.current_phase
        self.current_phase = TrainingPhase(self.current_phase.value + 1)
        self.phase_start_episode = self.total_episodes
        
        print(f" CURRICULUM PHASE TRANSITION: {old_phase.name} -> {self.current_phase.name}")
        print(f"   Total episodes: {self.total_episodes}")
        
        self.save_state()
        return True
    
    def should_run_scenario_drill(self) -> bool:
        """
        Check if we should run scenario drills.
        Scenario drills run periodically during League Training.
        """
        if self.current_phase != TrainingPhase.LEAGUE_TRAINING:
            return False
        
        # Run scenarios every 1000 episodes
        episodes_in_league = self.metrics[TrainingPhase.LEAGUE_TRAINING].episodes_in_phase
        return episodes_in_league > 0 and episodes_in_league % 1000 == 0
    
    def get_status_string(self) -> str:
        """Get a short status string for progress bar."""
        config = self.get_phase_config()
        metrics = self.metrics[self.current_phase]
        
        return f"P{self.current_phase.value}:{config.name[:8]}({metrics.episodes_in_phase}ep)"
    
    def get_detailed_status(self) -> Dict:
        """Get detailed status for logging."""
        config = self.get_phase_config()
        metrics = self.metrics[self.current_phase]

        return {
            'phase': self.current_phase.value,
            'phase_name': config.name,
            'episodes_in_phase': metrics.episodes_in_phase,
            'min_episodes': config.min_episodes,
            'max_episodes': config.max_episodes,
            'full_observability': config.full_observability,
            'win_rate': metrics.overall_win_rate,
            'win_rate_vs_random': metrics.win_rate_vs_random,
            'win_rate_vs_heuristic': metrics.win_rate_vs_heuristic,
        }


class TrueRandomOpponent:
    """
    Truly random opponent - picks uniformly from valid moves.
    Easier than HeuristicOpponent for early training.
    """
    
    def __init__(self, device, player_id: int = -1):
        self.device = device
        self.player_id = player_id
        self.name = "TrueRandom"
        
    def act(self, board, valid_moves, game_state=None, **kwargs):
        """Select a uniformly random valid move."""
        import random
        if not valid_moves:
            return None
        return random.choice(valid_moves)
    
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        import random
        return [random.choice(vm) if vm else None for vm in valid_moves_list]
    
    def reset_pbs(self):
        pass
    
    def update_pbs_batch(self, *args, **kwargs):
        pass


class HeuristicOpponent:
    """
    Frozen heuristic opponent for Phase 2.
    Fixed move scoring without learning (ties between the best moves broken at random).
    """
    
    def __init__(self, device, player_id: int = -1):
        self.device = device
        self.player_id = player_id
        self.name = "FrozenHeuristic"
        
    def act(self, board, valid_moves, game_state=None, **kwargs):
        """Select move using fixed heuristics."""
        if not valid_moves:
            return None
        
        import torch
        import random
        
        player_id = self.player_id
        if game_state and hasattr(game_state, 'current_player'):
            player_id = game_state.current_player
        
        if isinstance(board, torch.Tensor):
            board = board.tolist()  # scored as nested lists: no per-square .item()
        scored_moves = []
        for move in valid_moves:
            score = self._score_move(move, board, player_id)
            scored_moves.append((move, score))
        
        # Ties go to a random one of the best moves: always taking the first one walked the same
        # piece back and forth
        best = max(s for _, s in scored_moves)
        return random.choice([m for m, s in scored_moves if s >= best - 1e-9])
    
    def _score_move(self, move, board, player_id):
        """Deterministic move scoring."""
        import torch
        
        (r_from, c_from), (r_to, c_to) = move
        score = 0.0
        
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            target_val = board[r_to, c_to].item()
        else:
            piece_val = board[r_from][c_from]
            target_val = board[r_to][c_to]
        
        piece_rank = abs(piece_val)  # board values are PieceType values, which order the ranks

        # Forward movement
        if player_id == 1:
            score += (r_from - r_to) * 0.1
        else:
            score += (r_to - r_from) * 0.1

        # Attacking preference, with the real battle rules (Spy takes Marshal, Miner takes Bomb,
        # Bomb beats other attackers). The thesis code compared absolute values, so a hidden
        # piece (HIDDEN = -20) always looked unbeatable and the special battles looked lost.
        if is_enemy_value(target_val, player_id):
            outcome = battle_from_values(piece_val, target_val)
            if outcome is None:
                # Hidden defender: cheap pieces probe, valuable ones keep away. (The thesis
                # penalty, -rank * 0.05, made every probe but the Spy's worse than a sideways
                # move, so this opponent shuffled until the no-battle draw.)
                score += -0.3 if piece_rank == PieceType.SPY.value else 0.4 - piece_rank * 0.04
            elif outcome == 1:
                score += 0.5
            elif outcome == 0:
                score += 0.1
            else:
                score -= piece_rank * 0.1
        
        return score
    
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        return [self.act(b, vm, gs) for b, vm, gs in 
                zip(boards, valid_moves_list, game_states or [None]*len(boards))]
    
    def reset_pbs(self):
        pass
    
    def update_pbs_batch(self, *args, **kwargs):
        pass


class SmartHeuristicOpponent:
    """
    Strong heuristic opponent for challenging the DQN agent.
    Uses sophisticated multi-factor move scoring:
    - Material advantage (rank comparison)
    - Positional control (center, advancement)
    - Piece protection (don't sacrifice high-value pieces)
    - Flag hunting behavior
    - Scout reconnaissance
    - Miner prioritization for bombs
    - Spy tactical deployment
    """
    
    def __init__(self, device, player_id: int = -1):
        self.device = device
        self.player_id = player_id
        self.name = "SmartHeuristic"
        
        # Piece value hierarchy (higher = more valuable), keyed by board value (PieceType value).
        # (The thesis table was keyed by rank 0-11, so every piece got its neighbour's value:
        # the Marshal 0, the General 100.)
        self.piece_values = {
            PieceType.MARSHAL.value: 100,
            PieceType.GENERAL.value: 80,
            PieceType.COLONEL.value: 50,
            PieceType.MAJOR.value: 45,
            PieceType.CAPTAIN.value: 40,
            PieceType.LIEUTENANT.value: 30,
            PieceType.SERGEANT.value: 25,
            PieceType.MINER.value: 20,
            PieceType.SCOUT.value: 15,
            PieceType.FLAG.value: 200,   # Flag (must protect!)
            PieceType.BOMB.value: 0,     # Bomb (no movement value)
            PieceType.SPY.value: -10,    # Spy (special tactics)
        }
        
        # Track revealed enemy pieces (simple memory)
        self.revealed_enemies = {}  # position -> rank
    
    def act(self, board, valid_moves, game_state=None, **kwargs):
        """Select move using sophisticated heuristics."""
        if not valid_moves:
            return None
        
        import torch
        import random
        
        player_id = self.player_id
        if game_state and hasattr(game_state, 'current_player'):
            player_id = game_state.current_player
        
        if isinstance(board, torch.Tensor):
            board = board.tolist()  # scored as nested lists: no per-square .item()
        scored_moves = []
        for move in valid_moves:
            score = self._score_move(move, board, valid_moves, player_id)
            scored_moves.append((move, score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Add small randomization among top moves to avoid predictability
        top_score = scored_moves[0][1]
        top_moves = [m for m, s in scored_moves if s >= top_score - 0.1]
        
        if len(top_moves) > 1:
            return random.choice(top_moves)
        return scored_moves[0][0]
    
    def _score_move(self, move, board, all_valid_moves, player_id):
        """Multi-factor move scoring."""
        import torch
        
        (r_from, c_from), (r_to, c_to) = move
        score = 0.0
        
        if isinstance(board, torch.Tensor):
            piece_val = board[r_from, c_from].item()
            target_val = board[r_to, c_to].item()
        else:
            piece_val = board[r_from][c_from]
            target_val = board[r_to][c_to]
        
        piece_rank = abs(piece_val)  # board values are PieceType values
        target_rank = abs(target_val)
        my_value = self.piece_values.get(piece_rank, 10)

        # ===== 1. COMBAT EVALUATION (real battle rules) =====
        if is_enemy_value(target_val, player_id):
            outcome = battle_from_values(piece_val, target_val)

            if outcome is None:
                # Hidden defender: cheap pieces probe, valuable ones keep away. (With the thesis
                # weights, 0.5 - value / 25, only the Spy, which loses every attack but one,
                # probed: the Scout scored -0.1, so this opponent hardly ever attacked.)
                if piece_rank == PieceType.SPY.value:
                    score -= 1.0
                else:
                    score += 1.0 - (my_value / 40)
            else:
                target_value = self.piece_values.get(target_rank, 10)

                # Winning combat: big bonus
                if outcome == 1:
                    score += 2.0 + (target_value / 20)  # Higher value targets worth more

                # Equal combat: slight penalty (mutual destruction)
                elif outcome == 0:
                    score += 0.3 - (my_value / 100)  # Avoid equal trades with valuable pieces

                # Losing combat: big penalty unless we're expendable
                else:
                    score -= 1.5 + (my_value / 20)  # Don't sacrifice valuable pieces

            # Special: Spy vs Marshal
            if piece_rank == PieceType.SPY.value and target_rank == PieceType.MARSHAL.value:
                score += 5.0  # High priority kill

            # Special: Miner probes the enemy's back rows (where the bombs are)
            if piece_rank == PieceType.MINER.value:
                if player_id == 1 and r_to <= 3:  # Enemy side for P1 (the thesis code had r_to >= 6, its own side)
                    score += 0.5
                elif player_id == -1 and r_to >= 6:
                    score += 0.5
        
        # ===== 2. POSITIONAL EVALUATION =====
        # Forward advancement (toward enemy flag)
        if player_id == 1:
            advancement = (r_from - r_to)  # Moving up (lower row number)
        else:
            advancement = (r_to - r_from)  # Moving down (higher row number)
        
        # Don't overvalue advancement for high-value pieces
        advancement_weight = 0.15 if my_value < 50 else 0.05
        score += advancement * advancement_weight
        
        # Center control (columns 4-5 are valuable mid-board)
        center_bonus = 0.1 if 3 <= c_to <= 6 else 0
        center_row_bonus = 0.1 if 3 <= r_to <= 6 else 0
        score += center_bonus + center_row_bonus
        
        # ===== 3. PIECE PROTECTION =====
        # Don't move high-value pieces into danger
        if my_value >= 50:  # Marshal, General, Colonel
            # Check if destination puts us at risk (enemy adjacent)
            risk = self._check_adjacent_enemies(r_to, c_to, board, player_id)
            score -= risk * (my_value / 40)
        
        # ===== 4. SCOUT TACTICS =====
        if piece_rank == PieceType.SCOUT.value:
            move_distance = abs(r_to - r_from) + abs(c_to - c_from)
            if move_distance > 1:
                # Scouts should use multi-square moves for reconnaissance. (The thesis weights,
                # 0.3 per square plus 0.5 in enemy territory, outscored every probe, so scouts
                # roamed and the games timed out: 20 of 40 against random.)
                score += 0.05 * move_distance
                
                # Prefer scouting enemy territory
                if (player_id == 1 and r_to < 4) or (player_id == -1 and r_to > 5):
                    score += 0.1
        
        # ===== 5. FLAG PROTECTION =====
        # Penalize moves that leave flag area exposed
        # (Implicit: if we're in back 2 rows, prefer staying near corners/edges)
        if player_id == 1 and r_from >= 8:
            if my_value >= 30:  # Keep defenders near flag
                score -= 0.2
        elif player_id == -1 and r_from <= 1:
            if my_value >= 30:
                score -= 0.2
        
        # ===== 6. MOBILITY FACTOR =====
        # Slight bonus for moves that maintain piece flexibility
        # (avoids getting cornered)
        if c_to == 0 or c_to == 9:  # Edge columns
            score -= 0.05
        
        return score
    
    def _check_adjacent_enemies(self, r, c, board, player_id):
        """Check how many enemy pieces are adjacent to position."""
        import torch
        
        adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

        count = 0
        for ar, ac in adjacent:
            if 0 <= ar < 10 and 0 <= ac < 10:
                if isinstance(board, torch.Tensor):
                    val = board[ar, ac].item()
                else:
                    val = board[ar][ac]

                # Enemy piece (hidden or not); the thesis sign test also counted lakes for P1
                if is_enemy_value(val, player_id):
                    count += 1
        
        return count
    
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        return [self.act(b, vm, gs) for b, vm, gs in 
                zip(boards, valid_moves_list, game_states or [None]*len(boards))]
    
    def reset_pbs(self):
        self.revealed_enemies = {}
    
    def update_pbs_batch(self, *args, **kwargs):
        pass
