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
        min_episodes=5000,
        max_episodes=10000,
        opponents=["random", "heuristic", "smart_heuristic"],
        reward_focus="material",
        success_metrics={
            "win_rate_vs_random": 0.95,
            "win_rate_vs_heuristic": 0.75,
            "win_rate_vs_smart_heuristic": 0.60
        }
    ),
    TrainingPhase.MEMORY_GAP: PhaseConfig(
        name="Memory Gap",
        full_observability=False,
        min_episodes=5000,
        max_episodes=10000,
        opponents=["frozen_heuristic", "smart_heuristic"],
        reward_focus="epistemic",
        success_metrics={
            "pbs_accuracy": 0.75,
            "win_rate": 0.60
        }
    ),
    TrainingPhase.SELF_PLAY: PhaseConfig(
        name="Simple Self-Play",
        full_observability=False,
        min_episodes=5000,
        max_episodes=15000,
        opponents=["self_500", "smart_heuristic"],  # Mix with strong opponent
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
        
        For Phase 1: Uses ADAPTIVE difficulty based on win rate.
        Progressive opponent scaling:
        - Win rate < 60%: 100% random (pure learning mode)
        - Win rate 60-70%: 60% random, 40% heuristic (gradual introduction)
        - Win rate 70-80%: 30% random, 50% heuristic, 20% smart_heuristic (moderate challenge)
        - Win rate 80-90%: 10% random, 40% heuristic, 50% smart_heuristic (hard mode)
        - Win rate >= 90%: 20% heuristic, 80% smart_heuristic (expert mode - learn optimal play)
        """
        config = self.get_phase_config()
        opponents = config.opponents
        metrics = self.metrics[self.current_phase]
        
        if self.current_phase == TrainingPhase.PHYSICS_OF_WAR:
            # PROGRESSIVE OPPONENT SCALING based on win rate
            win_rate = metrics.overall_win_rate
            games_played = metrics.total_games
            
            # Need at least 50 games to have meaningful statistics
            if games_played < 50:
                # Still warming up - 100% random to learn basics
                return {"random": 1.0, "heuristic": 0.0, "smart_heuristic": 0.0}
            elif win_rate < 0.60:
                # Agent struggling - keep it 100% random until 60% win rate
                return {"random": 1.0, "heuristic": 0.0, "smart_heuristic": 0.0}
            elif win_rate < 0.70:
                # Just crossed 60% - gradual heuristic introduction (40%)
                return {"random": 0.6, "heuristic": 0.4, "smart_heuristic": 0.0}
            elif win_rate < 0.80:
                # Strong against random - introduce smart_heuristic (20%)
                return {"random": 0.3, "heuristic": 0.5, "smart_heuristic": 0.2}
            elif win_rate < 0.90:
                # Dominating heuristic - increase smart_heuristic (50%)
                return {"random": 0.1, "heuristic": 0.4, "smart_heuristic": 0.5}
            else:
                # Expert level - focus on optimal play against strongest (80%)
                return {"random": 0.0, "heuristic": 0.2, "smart_heuristic": 0.8}
                
        elif self.current_phase == TrainingPhase.MEMORY_GAP:
            # Mix frozen heuristic with smart heuristic for better learning
            return {"frozen_heuristic": 0.4, "smart_heuristic": 0.6}
        elif self.current_phase == TrainingPhase.SELF_PLAY:
            # Mix self-play with smart heuristic to prevent strategy collapse
            return {"self_500": 0.6, "smart_heuristic": 0.4}
        elif self.current_phase == TrainingPhase.LEAGUE_TRAINING:
            return {"league": 0.4, "smart_heuristic": 0.3, "greedy": 0.2, "self": 0.1}
        elif self.current_phase == TrainingPhase.SCENARIO_DRILLS:
            return {"scenario": 1.0}
        
        # Default fallback
        return {"smart_heuristic": 1.0}
    
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
        
        # Piece value hierarchy (higher = more valuable)
        self.piece_values = {
            10: 100,  # Marshal
            9: 80,    # General
            8: 50,    # Colonel
            7: 45,    # Major
            6: 40,    # Captain
            5: 30,    # Lieutenant
            4: 25,    # Sergeant
            3: 20,    # Miner
            2: 15,    # Scout
            1: 200,   # Flag (must protect!)
            11: 0,    # Bomb (no movement value)
            0: -10,   # Spy (special tactics)
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
        
        piece_rank = abs(piece_val)
        target_rank = abs(target_val) if target_val != 0 else 0
        my_value = self.piece_values.get(piece_rank, 10)
        
        # ===== 1. COMBAT EVALUATION =====
        if target_rank > 0:
            target_value = self.piece_values.get(target_rank, 10)
            
            # Winning combat: big bonus
            if piece_rank > target_rank:
                score += 2.0 + (target_value / 20)  # Higher value targets worth more
                
            # Equal combat: slight penalty (mutual destruction)
            elif piece_rank == target_rank:
                score += 0.3 - (my_value / 100)  # Avoid equal trades with valuable pieces
                
            # Losing combat: big penalty unless we're expendable
            else:
                score -= 1.5 + (my_value / 20)  # Don't sacrifice valuable pieces
            
            # Special: Spy vs Marshal
            if piece_rank == 0 and target_rank == 10:  # Spy attacks Marshal
                score += 5.0  # High priority kill
            
            # Special: Miner vs suspected Bomb location (back row enemy)
            if piece_rank == 3:
                if player_id == 1 and r_to >= 6:  # Enemy back row for P1
                    score += 0.5  # Miners should probe back rows
                elif player_id == -1 and r_to <= 3:
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
        if piece_rank == 2:  # Scout
            move_distance = abs(r_to - r_from) + abs(c_to - c_from)
            if move_distance > 1:
                # Scouts should use multi-square moves for reconnaissance
                score += 0.3 * move_distance
                
                # Prefer scouting enemy territory
                if (player_id == 1 and r_to < 4) or (player_id == -1 and r_to > 5):
                    score += 0.5
        
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
        
        enemy_sign = 1 if player_id == -1 else -1
        adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        
        count = 0
        for ar, ac in adjacent:
            if 0 <= ar < 10 and 0 <= ac < 10:
                if isinstance(board, torch.Tensor):
                    val = board[ar, ac].item()
                else:
                    val = board[ar][ac]
                
                # Check if it's an enemy piece
                if (enemy_sign > 0 and val > 0) or (enemy_sign < 0 and val < 0):
                    count += 1
        
        return count
    
    def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
        return [self.act(b, vm, gs) for b, vm, gs in 
                zip(boards, valid_moves_list, game_states or [None]*len(boards))]
    
    def reset_pbs(self):
        self.revealed_enemies = {}
    
    def update_pbs_batch(self, *args, **kwargs):
        pass
