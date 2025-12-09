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
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import IntEnum


class TrainingPhase(IntEnum):
    """Training curriculum phases."""
    PHYSICS_OF_WAR = 1      # Full observability, learn basics
    MEMORY_GAP = 2          # Fog of war, use memory (PBS)
    SELF_PLAY = 3           # Play against past self
    LEAGUE_TRAINING = 4     # Diverse opponent pool
    SCENARIO_DRILLS = 5     # Endgame specialization


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


# Default phase configurations with episode durations
PHASE_CONFIGS = {
    TrainingPhase.PHYSICS_OF_WAR: PhaseConfig(
        name="Physics of War",
        full_observability=True,
        min_episodes=500,
        max_episodes=2000,
        opponents=["random", "heuristic"],
        reward_focus="material",
        success_metrics={
            "win_rate_vs_random": 0.90,
            "win_rate_vs_heuristic": 0.60
        }
    ),
    TrainingPhase.MEMORY_GAP: PhaseConfig(
        name="Memory Gap",
        full_observability=False,
        min_episodes=1000,
        max_episodes=3000,
        opponents=["frozen_heuristic"],
        reward_focus="epistemic",
        success_metrics={
            "pbs_accuracy": 0.70,
            "win_rate": 0.55
        }
    ),
    TrainingPhase.SELF_PLAY: PhaseConfig(
        name="Simple Self-Play",
        full_observability=False,
        min_episodes=2000,
        max_episodes=5000,
        opponents=["self_500"],  # Self from 500 episodes ago
        reward_focus="material",
        success_metrics={
            "strategy_diversity": 0.3,  # Variance in win patterns
            "win_rate_stable": True
        }
    ),
    TrainingPhase.LEAGUE_TRAINING: PhaseConfig(
        name="League Training",
        full_observability=False,
        min_episodes=5000,
        max_episodes=50000,  # Main training phase
        opponents=["league", "exploiters", "random", "greedy"],
        reward_focus="balanced",
        success_metrics={
            "league_elo": 1500,
            "no_exploit_vulnerability": True
        }
    ),
    TrainingPhase.SCENARIO_DRILLS: PhaseConfig(
        name="Scenario Drills",
        full_observability=False,
        min_episodes=500,
        max_episodes=1000,  # Periodic, not continuous
        opponents=["scenario_specific"],
        reward_focus="positional",
        success_metrics={
            "scenario_completion_rate": 0.80
        }
    )
}


@dataclass
class PhaseMetrics:
    """Tracked metrics for phase transition decisions."""
    episodes_in_phase: int = 0
    total_wins: int = 0
    total_games: int = 0
    wins_vs_random: int = 0
    games_vs_random: int = 0
    wins_vs_heuristic: int = 0
    games_vs_heuristic: int = 0
    pbs_accuracy_sum: float = 0.0
    pbs_accuracy_count: int = 0
    recent_win_rates: List[float] = None
    
    def __post_init__(self):
        if self.recent_win_rates is None:
            self.recent_win_rates = []
    
    @property
    def overall_win_rate(self) -> float:
        return self.total_wins / max(1, self.total_games)
    
    @property
    def win_rate_vs_random(self) -> float:
        return self.wins_vs_random / max(1, self.games_vs_random)
    
    @property
    def win_rate_vs_heuristic(self) -> float:
        return self.wins_vs_heuristic / max(1, self.games_vs_heuristic)
    
    @property
    def avg_pbs_accuracy(self) -> float:
        return self.pbs_accuracy_sum / max(1, self.pbs_accuracy_count)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhaseMetrics':
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
                    
                print(f"📚 Loaded curriculum state: Phase {self.current_phase.value} ({self.get_phase_config().name})")
            except Exception as e:
                print(f"⚠️ Could not load curriculum state: {e}")
    
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
    
    def should_use_full_observability(self) -> bool:
        """Check if current phase uses full observability."""
        return self.get_phase_config().full_observability
    
    def get_opponent_distribution(self) -> Dict[str, float]:
        """
        Get opponent selection probabilities for current phase.
        Returns dict mapping opponent type to probability.
        
        For Phase 1: Uses ADAPTIVE difficulty based on win rate against random.
        - Win rate < 50%: 80% random, 20% heuristic (easy mode)
        - Win rate >= 50%: 20% random, 80% heuristic (hard mode)
        """
        config = self.get_phase_config()
        opponents = config.opponents
        metrics = self.metrics[self.current_phase]
        
        if self.current_phase == TrainingPhase.PHYSICS_OF_WAR:
            # DYNAMIC OPPONENT SCALING based on win rate
            win_rate = metrics.overall_win_rate
            games_played = metrics.total_games
            
            # Need at least 50 games to have meaningful statistics
            if games_played >= 50 and win_rate >= 0.50:
                # Agent is dominating random - increase difficulty!
                # 80% heuristic to make it struggle and learn
                return {"random": 0.2, "heuristic": 0.8}
            else:
                # Still learning basics - keep it easy
                return {"random": 0.6, "heuristic": 0.4}
                
        elif self.current_phase == TrainingPhase.MEMORY_GAP:
            return {"frozen_heuristic": 1.0}
        elif self.current_phase == TrainingPhase.SELF_PLAY:
            return {"self_500": 1.0}
        elif self.current_phase == TrainingPhase.LEAGUE_TRAINING:
            return {"league": 0.5, "random": 0.2, "greedy": 0.2, "self": 0.1}
        elif self.current_phase == TrainingPhase.SCENARIO_DRILLS:
            return {"scenario": 1.0}
        
        # Default fallback
        return {"self": 1.0}
    
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
            
            if opponent_type == 'random':
                metrics.wins_vs_random += 1
            elif opponent_type in ['heuristic', 'greedy', 'frozen_heuristic']:
                metrics.wins_vs_heuristic += 1
                
        if opponent_type == 'random':
            metrics.games_vs_random += 1
        elif opponent_type in ['heuristic', 'greedy', 'frozen_heuristic']:
            metrics.games_vs_heuristic += 1
            
        # PBS accuracy tracking
        if 'pbs_accuracy' in episode_result:
            metrics.pbs_accuracy_sum += episode_result['pbs_accuracy']
            metrics.pbs_accuracy_count += 1
        
        # Track recent win rates (sliding window)
        metrics.recent_win_rates.append(1 if winner == 1 else 0)
        if len(metrics.recent_win_rates) > 100:
            metrics.recent_win_rates.pop(0)
    
    def check_phase_transition(self) -> bool:
        """
        Check if we should advance to the next phase.
        
        Returns:
            True if phase transition should occur
        """
        config = self.get_phase_config()
        metrics = self.metrics[self.current_phase]
        
        # Always require minimum episodes
        if metrics.episodes_in_phase < config.min_episodes:
            return False
        
        # Force transition after max episodes
        if metrics.episodes_in_phase >= config.max_episodes:
            return True
        
        # Phase-specific success criteria
        if self.current_phase == TrainingPhase.PHYSICS_OF_WAR:
            return (metrics.win_rate_vs_random >= 0.90 and 
                    metrics.win_rate_vs_heuristic >= 0.60 and
                    metrics.games_vs_random >= 100 and
                    metrics.games_vs_heuristic >= 100)
                    
        elif self.current_phase == TrainingPhase.MEMORY_GAP:
            return (metrics.avg_pbs_accuracy >= 0.70 and
                    metrics.overall_win_rate >= 0.55 and
                    metrics.pbs_accuracy_count >= 50)
                    
        elif self.current_phase == TrainingPhase.SELF_PLAY:
            # Check for stable win rate (low variance = strategies are converging)
            if len(metrics.recent_win_rates) >= 100:
                variance = sum((x - 0.5)**2 for x in metrics.recent_win_rates) / 100
                return variance < 0.1  # Low variance indicates stable play
            return False
            
        elif self.current_phase == TrainingPhase.LEAGUE_TRAINING:
            # Main training phase - typically doesn't auto-transition
            # Could add ELO-based criteria here
            return False
            
        elif self.current_phase == TrainingPhase.SCENARIO_DRILLS:
            # Scenario completion rate
            return metrics.overall_win_rate >= 0.80
        
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
        
        print(f"🎓 CURRICULUM PHASE TRANSITION: {old_phase.name} → {self.current_phase.name}")
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
            'pbs_accuracy': metrics.avg_pbs_accuracy
        }


class HeuristicOpponent:
    """
    Frozen heuristic opponent for Phase 2.
    Uses deterministic strategy without learning.
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
        
        scored_moves = []
        for move in valid_moves:
            score = self._score_move(move, board, player_id)
            scored_moves.append((move, score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return scored_moves[0][0]
    
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
        
        piece_rank = abs(piece_val)
        target_rank = abs(target_val) if target_val != 0 else 0
        
        # Forward movement
        if player_id == 1:
            score += (r_from - r_to) * 0.1
        else:
            score += (r_to - r_from) * 0.1
        
        # Attacking preference
        if target_rank > 0:
            if piece_rank > target_rank:
                score += 0.5
            elif piece_rank == target_rank:
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
