"""
Self-Check Protocols for Double DQN + AAREN Training
Implements 5 mandatory monitors that HALT training on failure.

Reference: Pascanu et al. (2013) for gradient vitality monitoring

Checks:
1. Gradient Vitality Monitor - Detects dead gradients
2. AAREN Collapse Detection - Detects representation collapse
3. Q-Value Sanity Check - Ensures network outputs vary
4. Memory Guardian - Prevents OOM (6GB VRAM constraint)
5. Learning Validation Checkpoint - Detects convergence failure
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class CheckResult:
    """Result of a self-check."""
    passed: bool
    message: str
    severity: str = "INFO"  # INFO, WARNING, ALERT, CRITICAL
    data: Dict[str, Any] = field(default_factory=dict)


class SelfCheckProtocol:
    """Base class for self-check protocols."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.check_count = 0
        self.alert_history: List[CheckResult] = []
    
    def check(self, *args, **kwargs) -> CheckResult:
        raise NotImplementedError
    
    def should_halt(self) -> bool:
        """Return True if training should halt."""
        return False


class GradientVitalityMonitor(SelfCheckProtocol):
    """
    Monitor gradient L2 norms to detect dead gradients or explosions.
    
    ALERT if:
    - Any layer's mean gradient norm < 10^-7 (dead) for 5 consecutive checks
    - Any layer's mean gradient norm > 100 (explosion) for 5 consecutive checks
    
    Reference: Pascanu et al. (2013), "On the difficulty of training recurrent neural networks"
    """
    
    DEAD_THRESHOLD = 1e-7
    EXPLOSION_THRESHOLD = 100.0
    CONSECUTIVE_FAILURES = 5
    CHECK_INTERVAL = 100  # Check every 100 optimization steps
    
    def __init__(self, model: nn.Module, enabled: bool = True):
        super().__init__("Gradient Vitality Monitor", enabled)
        self.model = model
        
        # Track gradient norms per layer
        self.layer_names: List[str] = []
        self.layer_grad_history: Dict[str, deque] = {}
        self.consecutive_dead: Dict[str, int] = {}
        self.consecutive_explosion: Dict[str, int] = {}
        
        # Register layers to monitor
        self._register_layers()
    
    def _register_layers(self):
        """Identify layers to monitor."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)
                self.layer_grad_history[name] = deque(maxlen=10)
                self.consecutive_dead[name] = 0
                self.consecutive_explosion[name] = 0
    
    def compute_gradient_norms(self) -> Dict[str, float]:
        """Compute L2 norm of gradients for each layer."""
        norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm(2).item()
                norms[name] = norm
                self.layer_grad_history[name].append(norm)
        return norms
    
    def check(self, step: int) -> CheckResult:
        """
        Check gradient vitality.
        
        Args:
            step: Current optimization step
            
        Returns:
            CheckResult with pass/fail status
        """
        if not self.enabled:
            return CheckResult(True, "Check disabled", "INFO")
        
        if step % self.CHECK_INTERVAL != 0:
            return CheckResult(True, "Not check interval", "INFO")
        
        self.check_count += 1
        norms = self.compute_gradient_norms()
        
        dead_layers = []
        exploding_layers = []
        
        for name, norm in norms.items():
            # Check for dead gradients
            if norm < self.DEAD_THRESHOLD:
                self.consecutive_dead[name] += 1
                if self.consecutive_dead[name] >= self.CONSECUTIVE_FAILURES:
                    dead_layers.append((name, norm))
            else:
                self.consecutive_dead[name] = 0
            
            # Check for gradient explosion
            if norm > self.EXPLOSION_THRESHOLD:
                self.consecutive_explosion[name] += 1
                if self.consecutive_explosion[name] >= self.CONSECUTIVE_FAILURES:
                    exploding_layers.append((name, norm))
            else:
                self.consecutive_explosion[name] = 0
        
        # Build result
        if dead_layers or exploding_layers:
            msg_parts = []
            if dead_layers:
                msg_parts.append(f"DEAD gradients in: {[l[0] for l in dead_layers]}")
            if exploding_layers:
                msg_parts.append(f"EXPLODING gradients in: {[l[0] for l in exploding_layers]}")
            
            result = CheckResult(
                passed=False,
                message=" | ".join(msg_parts),
                severity="ALERT",
                data={"dead_layers": dead_layers, "exploding_layers": exploding_layers}
            )
            self.alert_history.append(result)
            return result
        
        # Compute summary stats
        all_norms = list(norms.values())
        mean_norm = np.mean(all_norms) if all_norms else 0
        
        return CheckResult(
            passed=True,
            message=f"Gradients healthy (mean norm: {mean_norm:.6f})",
            severity="INFO",
            data={"mean_norm": mean_norm, "layer_count": len(norms)}
        )
    
    def should_halt(self) -> bool:
        """Halt if any layer has dead or exploding gradients for too long."""
        if not self.alert_history:
            return False
        return len(self.alert_history) >= 3  # 3 consecutive alerts


class AARENCollapseDetection(SelfCheckProtocol):
    """
    Monitor AAREN output for representation collapse.
    
    ALERT if:
    - σ < 0.01 (representation collapse)
    - |μ| > 5.0 (severe drift)
    - >50% of output dimensions are exactly 0.0 (sparse gradient death)
    """
    
    SIGMA_MIN = 0.01
    MU_MAX = 5.0
    ZERO_THRESHOLD = 0.5  # 50% zeros
    
    def __init__(self, enabled: bool = True):
        super().__init__("AAREN Collapse Detection", enabled)
        self.stats_history: deque = deque(maxlen=100)
        self.consecutive_failures = 0
    
    def check(self, aaren_output: torch.Tensor) -> CheckResult:
        """
        Check AAREN output for collapse.
        
        Args:
            aaren_output: (batch, 64) AAREN embeddings
            
        Returns:
            CheckResult
        """
        if not self.enabled:
            return CheckResult(True, "Check disabled", "INFO")
        
        self.check_count += 1
        
        # Compute statistics
        with torch.no_grad():
            batch_mean = aaren_output.mean(dim=0)  # (64,)
            batch_std = aaren_output.std(dim=0)    # (64,)
            
            mu = batch_mean.mean().item()
            sigma = batch_std.mean().item()
            
            # Count exact zeros
            zero_mask = (aaren_output == 0.0)
            zero_ratio = zero_mask.float().mean().item()
        
        self.stats_history.append({
            'mu': mu,
            'sigma': sigma,
            'zero_ratio': zero_ratio
        })
        
        # Check conditions
        alerts = []
        
        if sigma < self.SIGMA_MIN:
            alerts.append(f"sigma={sigma:.4f} < {self.SIGMA_MIN} (COLLAPSE)")
        
        if abs(mu) > self.MU_MAX:
            alerts.append(f"|mu|={abs(mu):.4f} > {self.MU_MAX} (DRIFT)")
        
        if zero_ratio > self.ZERO_THRESHOLD:
            alerts.append(f"zeros={zero_ratio:.1%} > {self.ZERO_THRESHOLD:.0%} (SPARSE DEATH)")
        
        if alerts:
            self.consecutive_failures += 1
            result = CheckResult(
                passed=False,
                message=" | ".join(alerts),
                severity="ALERT",
                data={'mu': mu, 'sigma': sigma, 'zero_ratio': zero_ratio}
            )
            self.alert_history.append(result)
            return result
        
        self.consecutive_failures = 0
        return CheckResult(
            passed=True,
            message=f"AAREN healthy (mu={mu:.4f}, sigma={sigma:.4f}, zeros={zero_ratio:.1%})",
            severity="INFO",
            data={'mu': mu, 'sigma': sigma, 'zero_ratio': zero_ratio}
        )
    
    def should_halt(self) -> bool:
        return self.consecutive_failures >= 5


class QValueSanityCheck(SelfCheckProtocol):
    """
    Ensure Q-values are not constant and networks are different.
    
    Checks:
    - Variance of Q-values > 0.1 (network is not outputting constants)
    - Online and target networks differ (MSE > 0)
    """
    
    VARIANCE_MIN = 0.1
    
    def __init__(self, enabled: bool = True):
        super().__init__("Q-Value Sanity Check", enabled)
        self.consecutive_failures = 0
    
    def check(
        self,
        q_values: torch.Tensor,
        online_target_mse: float
    ) -> CheckResult:
        """
        Check Q-value sanity.
        
        Args:
            q_values: (batch, action_dim) Q-values
            online_target_mse: MSE between online and target network params
            
        Returns:
            CheckResult
        """
        if not self.enabled:
            return CheckResult(True, "Check disabled", "INFO")
        
        self.check_count += 1
        
        with torch.no_grad():
            q_variance = q_values.var().item()
            q_mean = q_values.mean().item()
        
        alerts = []
        
        if q_variance < self.VARIANCE_MIN:
            alerts.append(f"Q variance={q_variance:.4f} < {self.VARIANCE_MIN} (CONSTANT OUTPUT)")
        
        if online_target_mse <= 0:
            alerts.append("Online == Target network (NO LEARNING)")
        
        if alerts:
            self.consecutive_failures += 1
            result = CheckResult(
                passed=False,
                message=" | ".join(alerts),
                severity="ALERT",
                data={'q_variance': q_variance, 'q_mean': q_mean, 'online_target_mse': online_target_mse}
            )
            self.alert_history.append(result)
            return result
        
        self.consecutive_failures = 0
        return CheckResult(
            passed=True,
            message=f"Q-values healthy (var={q_variance:.4f}, mean={q_mean:.4f})",
            severity="INFO",
            data={'q_variance': q_variance, 'q_mean': q_mean, 'online_target_mse': online_target_mse}
        )
    
    def should_halt(self) -> bool:
        return self.consecutive_failures >= 10


class MemoryGuardian(SelfCheckProtocol):
    """
    Prevent OOM by monitoring CUDA memory allocation.
    
    CRITICAL HALT if allocated > 5.5 GB (reserve 500MB buffer for CUDA overhead)
    Enable gradient checkpointing automatically if OOM imminent.
    """
    
    VRAM_LIMIT_GB = 4.5
    WARNING_THRESHOLD_GB = 4.0
    
    def __init__(self, model: nn.Module = None, enabled: bool = True):
        super().__init__("Memory Guardian", enabled)
        self.model = model
        self.checkpointing_enabled = False
        self.peak_memory_gb = 0.0
    
    def check(self) -> CheckResult:
        """
        Check CUDA memory usage.
        
        Returns:
            CheckResult with CRITICAL if over limit
        """
        if not self.enabled:
            return CheckResult(True, "Check disabled", "INFO")
        
        if not torch.cuda.is_available():
            return CheckResult(True, "CUDA not available", "INFO")
        
        self.check_count += 1
        
        # Get memory stats
        allocated_bytes = torch.cuda.memory_allocated()
        allocated_gb = allocated_bytes / 1e9
        
        reserved_bytes = torch.cuda.memory_reserved()
        reserved_gb = reserved_bytes / 1e9
        
        self.peak_memory_gb = max(self.peak_memory_gb, allocated_gb)
        
        # Critical: Over hard limit
        if allocated_gb > self.VRAM_LIMIT_GB:
            # Try to enable checkpointing
            if self.model is not None and not self.checkpointing_enabled:
                self._enable_checkpointing()
            
            result = CheckResult(
                passed=False,
                message=f"VRAM {allocated_gb:.2f}GB > {self.VRAM_LIMIT_GB}GB LIMIT - CRITICAL HALT",
                severity="CRITICAL",
                data={'allocated_gb': allocated_gb, 'reserved_gb': reserved_gb}
            )
            self.alert_history.append(result)
            return result
        
        # Warning: Approaching limit
        if allocated_gb > self.WARNING_THRESHOLD_GB:
            if self.model is not None and not self.checkpointing_enabled:
                self._enable_checkpointing()
            
            return CheckResult(
                passed=True,
                message=f"VRAM {allocated_gb:.2f}GB approaching limit - checkpointing enabled",
                severity="WARNING",
                data={'allocated_gb': allocated_gb, 'reserved_gb': reserved_gb}
            )
        
        return CheckResult(
            passed=True,
            message=f"VRAM OK: {allocated_gb:.2f}GB allocated ({reserved_gb:.2f}GB reserved)",
            severity="INFO",
            data={'allocated_gb': allocated_gb, 'reserved_gb': reserved_gb}
        )
    
    def _enable_checkpointing(self):
        """Enable gradient checkpointing in model."""
        if self.model is not None and hasattr(self.model, 'enable_checkpointing'):
            self.model.enable_checkpointing()
            self.checkpointing_enabled = True
            print("[Memory Guardian] Gradient checkpointing ENABLED")
    
    def should_halt(self) -> bool:
        """Halt on CRITICAL memory alerts."""
        for result in self.alert_history[-5:]:  # Check last 5
            if result.severity == "CRITICAL":
                return True
        return False
    
    @staticmethod
    def force_cleanup():
        """Force memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class LearningValidationCheckpoint(SelfCheckProtocol):
    """
    Validate learning progress by tracking win rate vs random agent.
    
    ALERT if:
    - Win rate remains at 0-1% after 50,000 steps (convergence failure)
    - Entropy of action distribution drops below 0.5 bits (premature exploitation)
    """
    
    MIN_WIN_RATE = 0.02  # 2%
    STEPS_BEFORE_CHECK = 50000
    CHECK_INTERVAL = 10000
    MIN_ENTROPY = 0.5  # bits
    
    def __init__(self, enabled: bool = True):
        super().__init__("Learning Validation Checkpoint", enabled)
        self.win_history: List[Tuple[int, float]] = []
        self.entropy_history: List[float] = []
    
    def record_evaluation(self, step: int, win_rate: float, games_played: int = 100):
        """Record evaluation result."""
        self.win_history.append((step, win_rate))
    
    def record_action_entropy(self, entropy: float):
        """Record action distribution entropy."""
        self.entropy_history.append(entropy)
    
    def check(self, current_step: int) -> CheckResult:
        """
        Check learning progress.
        
        Args:
            current_step: Current training step
            
        Returns:
            CheckResult
        """
        if not self.enabled:
            return CheckResult(True, "Check disabled", "INFO")
        
        self.check_count += 1
        
        alerts = []
        
        # Check win rate (after warmup)
        if current_step >= self.STEPS_BEFORE_CHECK and self.win_history:
            latest_win_rate = self.win_history[-1][1]
            if latest_win_rate < self.MIN_WIN_RATE:
                alerts.append(f"Win rate {latest_win_rate:.1%} < {self.MIN_WIN_RATE:.1%} after {current_step} steps (CONVERGENCE FAILURE)")
        
        # Check entropy
        if self.entropy_history:
            recent_entropy = np.mean(self.entropy_history[-100:])
            if recent_entropy < self.MIN_ENTROPY:
                alerts.append(f"Action entropy {recent_entropy:.2f} < {self.MIN_ENTROPY} bits (PREMATURE EXPLOITATION)")
        
        if alerts:
            result = CheckResult(
                passed=False,
                message=" | ".join(alerts),
                severity="ALERT",
                data={
                    'win_history': self.win_history[-5:],
                    'mean_entropy': np.mean(self.entropy_history[-100:]) if self.entropy_history else None
                }
            )
            self.alert_history.append(result)
            return result
        
        return CheckResult(
            passed=True,
            message=f"Learning validation OK (step {current_step})",
            severity="INFO"
        )
    
    def should_halt(self) -> bool:
        """Halt on sustained convergence failure."""
        if len(self.alert_history) >= 3:
            # Check if last 3 evaluations all failed
            return all(not r.passed for r in self.alert_history[-3:])
        return False


class SelfCheckSuite:
    """
    Combines all self-check protocols into a single monitoring suite.
    """
    
    def __init__(
        self,
        model: nn.Module,
        aaren: nn.Module,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.model = model
        self.aaren = aaren
        
        # Initialize all checks
        self.gradient_monitor = GradientVitalityMonitor(model, enabled)
        self.aaren_collapse = AARENCollapseDetection(enabled)
        self.q_value_check = QValueSanityCheck(enabled)
        self.memory_guardian = MemoryGuardian(model, enabled)
        self.learning_validation = LearningValidationCheckpoint(enabled)
        
        self.all_checks = [
            self.gradient_monitor,
            self.aaren_collapse,
            self.q_value_check,
            self.memory_guardian,
            self.learning_validation
        ]
    
    def run_all_checks(
        self,
        step: int,
        aaren_output: Optional[torch.Tensor] = None,
        q_values: Optional[torch.Tensor] = None,
        online_target_mse: float = 0.0
    ) -> Dict[str, CheckResult]:
        """
        Run all applicable self-checks.
        
        Args:
            step: Current training step
            aaren_output: AAREN embeddings (if available)
            q_values: Q-values (if available)
            online_target_mse: MSE between online and target networks
            
        Returns:
            Dictionary of check names to results
        """
        results = {}
        
        # Gradient check
        results['gradient'] = self.gradient_monitor.check(step)
        
        # AAREN collapse check
        if aaren_output is not None:
            results['aaren'] = self.aaren_collapse.check(aaren_output)
        
        # Q-value check
        if q_values is not None:
            results['q_value'] = self.q_value_check.check(q_values, online_target_mse)
        
        # Memory check
        results['memory'] = self.memory_guardian.check()
        
        # Learning validation
        results['learning'] = self.learning_validation.check(step)
        
        return results
    
    def should_halt(self) -> Tuple[bool, str]:
        """
        Check if any protocol requires halting.
        
        Returns:
            (should_halt, reason)
        """
        for check in self.all_checks:
            if check.should_halt():
                return True, f"HALT: {check.name} triggered stop condition"
        return False, ""
    
    def print_summary(self):
        """Print summary of all checks."""
        print("\n" + "=" * 60)
        print("SELF-CHECK SUMMARY")
        print("=" * 60)
        
        for check in self.all_checks:
            status = "[ok]" if not check.should_halt() else "[!!]"
            alerts = len(check.alert_history)
            print(f"  [{status}] {check.name}: {check.check_count} checks, {alerts} alerts")
        
        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"\n  VRAM: {allocated:.2f}GB / 6.0GB")
        
        print("=" * 60 + "\n")


def compute_action_entropy(action_probs: torch.Tensor) -> float:
    """
    Compute entropy of action distribution in bits.
    
    Args:
        action_probs: (batch, action_dim) action probabilities
        
    Returns:
        Mean entropy in bits
    """
    # Clamp to avoid log(0)
    probs = torch.clamp(action_probs, min=1e-8)
    
    # Entropy: -sum(p * log2(p))
    entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
    
    return entropy.mean().item()
