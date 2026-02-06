"""
DQN Learning Diagnostics - Phase 1: Learning Verification

Systematic debugging tools to identify why a Rainbow DQN agent fails to learn.
Based on the diagnostic protocol for open-information Stratego environments.

Usage:
    from diagnostics import DiagnosticTracker
    tracker = DiagnosticTracker()
    tracker.log_step(episode, loss, grad_norm, q_delta)
    failures = tracker.check_failure_conditions()
"""

import numpy as np
from collections import deque
from typing import List, Optional, Dict, Any
import torch


class SignalTracker:
    """
    Phase 2: Signal Verification
    
    Tracks reward signals to ensure critical events (like Flag Capture)
    are actually reaching the agent's memory and driving learning.
    """
    def __init__(self):
        self.flag_captures = 0
        self.max_reward_seen = -float('inf')
        self.high_TD_errors = 0
        self.last_high_reward_step = 0
        
    def log_experience(self, reward, done):
        """Log rewards entering memory (called from remember)"""
        if reward > self.max_reward_seen:
            self.max_reward_seen = reward
            
        if reward > 5.0: # Significant positive reward
            print(f"[DIAG SIGNAL] High Reward Encountered: {reward:.2f}")
            if reward > 20.0:
                self.flag_captures += 1
                print(f"[DIAG SIGNAL] !!! FLAG CAPTURE DETECTED ({reward:.2f}) !!!")
                
    def log_td_error(self, td_errors: np.ndarray, rewards: np.ndarray):
        """Verify high rewards cause high TD errors (learning opportunity)"""
        # Check if batch contains high rewards
        max_batch_reward = rewards.max()
        if max_batch_reward > 5.0:
            max_td = td_errors.max()
            mean_td = td_errors.mean()
            print(f"[DIAG SIGNAL] Replaying High Reward ({max_batch_reward:.2f}). "
                  f"Max TD: {max_td:.4f}, Mean TD: {mean_td:.4f}")
            
            if max_td > 1.0:
                self.high_TD_errors += 1


class DiagnosticTracker:
    """
    Phase 1 & 2 Diagnostics
    """
    
    def __init__(self, log_interval: int = 100, 
                 dead_grad_threshold: float = 1e-6,
                 dead_grad_consecutive: int = 10,
                 history_size: int = 1000):
        # ... (same init args)
        self.log_interval = log_interval
        self.dead_grad_threshold = dead_grad_threshold
        self.dead_grad_consecutive = dead_grad_consecutive
        
        # Metric histories
        self.grad_norm_history = deque(maxlen=history_size)
        self.q_delta_history = deque(maxlen=history_size)
        self.loss_history = deque(maxlen=history_size)
        self.episode_history = deque(maxlen=history_size)
        

class RepresentationTracker:
    """
    Phase 3: Representation Verification
    
    Checks if the AAREN memory embeddings are collapsing or vanishing,
    which would explain why the agent fails to generalize winning signals.
    """
    def __init__(self):
        self.embedding_history = deque(maxlen=100)
        self.collapse_detected = False
        
    def log_embedding_stats(self, states):
        """Check AAREN embedding statistics (states: Batch x Channels x H x W)"""
        if states.shape[1] <= 15:
            return None # No embeddings present
            
        # Extract embedding part of state (channels 15-79)
        # Assuming first 15 are board, rest are embeddings
        embeddings = states[:, 15:, :, :] 
        
        # Calculate stats across batch and spatial dims
        mean = embeddings.mean().item()
        std = embeddings.std().item()
        zero_fraction = (embeddings == 0).float().mean().item()
        
        if std < 1e-4 and not self.collapse_detected:
            print(f"[DIAG REP FAILURE] Representation Collapse Detected! Std: {std:.8f}")
            self.collapse_detected = True
            
        return {"rep_mean": mean, "rep_std": std, "rep_zero": zero_fraction}


class DiagnosticTracker:
    """
    Phase 1 & 2 & 3 Diagnostics
    """
    
    def __init__(self, log_interval: int = 100, 
                 dead_grad_threshold: float = 1e-6,
                 dead_grad_consecutive: int = 10,
                 history_size: int = 1000):
        # ... (same init args)
        self.log_interval = log_interval
        self.dead_grad_threshold = dead_grad_threshold
        self.dead_grad_consecutive = dead_grad_consecutive
        
        # Metric histories
        self.grad_norm_history = deque(maxlen=history_size)
        self.q_delta_history = deque(maxlen=history_size)
        self.loss_history = deque(maxlen=history_size)
        self.episode_history = deque(maxlen=history_size)
        

class PolicyTracker:
    """
    Phase 4: Policy Verification
    
    Checks if the agent is learning a confident policy (low entropy, high Q-gap)
    or just staying random/confused (high entropy, low Q-gap).
    """
    def __init__(self):
        self.entropy_history = deque(maxlen=100)
        self.q_gap_history = deque(maxlen=100)
        
    def log_policy_stats(self, q_values):
        """
        Analyze Policy derived from Q-values (Batch x Actions).
        q_values should be Expected Q-values.
        """
        # q_values: (Batch, Actions)
        if q_values.dim() == 3: # Handle if passed distribution by mistake, take mean?? No, assume expected Q.
             pass 
             
        # 1. Calculate Q-Value Separation (Gap)
        # Sort Q-values to find top 2
        topk, _ = torch.topk(q_values, k=2, dim=1)
        # Gap = Best - SecondBest
        gap = (topk[:, 0] - topk[:, 1]).mean().item()
        
        # 2. Calculate Softmax Entropy (Proxy for confidence)
        # Use temperature=1.0 for analysis
        probs = torch.softmax(q_values, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1).mean().item()
        
        print(f"[DIAG POLICY] Entropy: {entropy:.4f} | Q-Gap: {gap:.4f}")
        
        return {"policy_entropy": entropy, "policy_q_gap": gap}


class DiagnosticTracker:
    """
    Phase 1 & 2 & 3 & 4 Diagnostics
    """
    
    def __init__(self, log_interval: int = 100, 
                 dead_grad_threshold: float = 1e-6,
                 dead_grad_consecutive: int = 10,
                 history_size: int = 1000):
        # ... (same init args)
        self.log_interval = log_interval
        self.dead_grad_threshold = dead_grad_threshold
        self.dead_grad_consecutive = dead_grad_consecutive
        
        # Metric histories
        self.grad_norm_history = deque(maxlen=history_size)
        self.q_delta_history = deque(maxlen=history_size)
        self.loss_history = deque(maxlen=history_size)
        self.episode_history = deque(maxlen=history_size)
        
        # Failure tracking
        self.consecutive_dead_grads = 0
        self.consecutive_frozen_q = 0
        self.consecutive_stagnant_loss = 0
        
        # Phase 2: Signal Tracking
        self.signal_tracker = SignalTracker()
        
        # Phase 3: Representation Tracking
        self.representation_tracker = RepresentationTracker()
        
        # Phase 4: Policy Tracking
        self.policy_tracker = PolicyTracker()
        
        # Summary stats
        self.total_steps = 0
        self.last_logged_episode = -1
        
    def log_step(self, episode: int, loss: float, grad_norm: float, 
                 q_delta: float, verbose: bool = True) -> None:
        """
        Log a training step's diagnostics.
        
        Args:
            episode: Current episode number
            loss: Training loss value
            grad_norm: Total gradient norm
            q_delta: Max absolute change in Q-values
            verbose: Print to console
        """
        self.total_steps += 1
        self.episode_history.append(episode)
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.q_delta_history.append(q_delta)
        
        # Track consecutive failures
        if grad_norm < self.dead_grad_threshold:
            self.consecutive_dead_grads += 1
        else:
            self.consecutive_dead_grads = 0
            
        if q_delta < 1e-8:
            self.consecutive_frozen_q += 1
        else:
            self.consecutive_frozen_q = 0
            
        # Check loss stagnation (compare to 10 steps ago)
        if len(self.loss_history) >= 10:
            old_loss = self.loss_history[-10]
            if abs(loss - old_loss) < 1e-6:
                self.consecutive_stagnant_loss += 1
            else:
                self.consecutive_stagnant_loss = 0
        
        if verbose:
            print(f"[DIAG][Ep {episode}] Loss: {loss:.6f} | "
                  f"Grad Norm: {grad_norm:.6f} | Q-delta: {q_delta:.6f}")
            
        self.last_logged_episode = episode
    
    def check_failure_conditions(self) -> List[str]:
        """
        Check for common DQN training failures.
        
        Returns:
            List of failure condition strings (empty if all OK)
        """
        failures = []
        
        # Dead gradients
        if self.consecutive_dead_grads >= self.dead_grad_consecutive:
            failures.append(
                f"Dead gradients: Grad norm < {self.dead_grad_threshold} for "
                f"{self.consecutive_dead_grads} consecutive logs. "
                "Check: ReLU death, LR too low, loss scale issue."
            )
        
        # Frozen Q-values
        if self.consecutive_frozen_q >= 5:
            failures.append(
                f"Q-values frozen: Zero Q-delta for {self.consecutive_frozen_q} "
                "consecutive logs. Check: Optimizer not stepping, zero gradients."
            )
        
        # Stagnant loss
        if self.consecutive_stagnant_loss >= 20:
            failures.append(
                f"Loss stagnant: Constant to 6+ decimal places for "
                f"{self.consecutive_stagnant_loss} logs. Check: No learning signal, "
                "reward structure, or network capacity."
            )
            
        return failures
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        if len(self.loss_history) == 0:
            return {"status": "No data yet", "total_steps": self.total_steps}
            
        return {
            "total_steps": self.total_steps,
            "recent_loss_mean": np.mean(list(self.loss_history)[-100:]),
            "recent_loss_std": np.std(list(self.loss_history)[-100:]),
            "recent_grad_norm_mean": np.mean(list(self.grad_norm_history)[-100:]),
            "recent_q_delta_mean": np.mean(list(self.q_delta_history)[-100:]),
            "consecutive_dead_grads": self.consecutive_dead_grads,
            "consecutive_frozen_q": self.consecutive_frozen_q,
            "consecutive_stagnant_loss": self.consecutive_stagnant_loss,
        }


class DetailedDiagnostics:
    """
    Extended diagnostics for deeper analysis.
    
    Captures per-layer gradient norms, activation statistics,
    and distribution shift metrics.
    """
    
    def __init__(self, network: torch.nn.Module, device: str = 'cuda'):
        self.network = network
        self.device = device
        self.layer_grad_norms = {}
        self.activation_stats = {}
        
    def capture_layer_gradients(self) -> Dict[str, float]:
        """Capture gradient norm for each layer."""
        layer_norms = {}
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                layer_norms[name] = param.grad.norm().item()
            else:
                layer_norms[name] = 0.0
        self.layer_grad_norms = layer_norms
        return layer_norms
    
    def check_relu_death(self) -> float:
        """
        Estimate ReLU death rate using gradient proxy.
        
        Returns:
            Fraction of layers with very small gradients (potential dead neurons)
        """
        if not self.layer_grad_norms:
            self.capture_layer_gradients()
            
        dead_layers = sum(1 for v in self.layer_grad_norms.values() if v < 1e-7)
        total_layers = len(self.layer_grad_norms)
        
        return dead_layers / max(total_layers, 1)
    
    def diagnose_q_value_issue(self, q_network, states: torch.Tensor, 
                                support: torch.Tensor) -> Dict[str, Any]:
        """
        Detailed Q-value diagnostics.
        
        Args:
            q_network: The Q-network to analyze
            states: Batch of states
            support: C51 support atoms
            
        Returns:
            Dict with Q-value statistics
        """
        q_network.eval()
        with torch.no_grad():
            log_probs = q_network(states)
            probs = log_probs.exp()
            expected_q = (probs * support).sum(dim=2)
            
            return {
                "q_mean": expected_q.mean().item(),
                "q_std": expected_q.std().item(),
                "q_min": expected_q.min().item(),
                "q_max": expected_q.max().item(),
                "q_range": (expected_q.max() - expected_q.min()).item(),
                "max_probs_entropy": -(probs * log_probs).sum(dim=2).mean().item(),
            }
        
    def print_layer_report(self):
        """Print human-readable layer gradient report."""
        if not self.layer_grad_norms:
            print("[DIAG] No layer gradients captured yet.")
            return
            
        print("\n" + "="*60)
        print("LAYER GRADIENT REPORT")
        print("="*60)
        
        for name, norm in sorted(self.layer_grad_norms.items()):
            status = "✓" if norm > 1e-6 else "⚠ DEAD"
            print(f"  {status} {name}: {norm:.2e}")
            
        dead_rate = self.check_relu_death()
        print(f"\nDead layer rate: {dead_rate*100:.1f}%")
        print("="*60 + "\n")


def run_preflight_diagnostics(agent, device='cuda') -> bool:
    """
    Run quick sanity checks before training starts.
    
    Args:
        agent: RainbowAgent instance
        device: Device to use
        
    Returns:
        True if all checks pass
    """
    print("\n[DIAG] Running Pre-flight Diagnostics...")
    
    all_ok = True
    
    # 1. Check network is on correct device
    first_param = next(agent.q_network.parameters())
    if str(first_param.device) != device and device not in str(first_param.device):
        print(f"  [WARN] Network on {first_param.device}, expected {device}")
        all_ok = False
    else:
        print(f"  [OK] Network on {first_param.device}")
    
    # 2. Check optimizer has parameters
    if len(agent.optimizer.param_groups) == 0:
        print("  [FAIL] Optimizer has no parameter groups!")
        all_ok = False
    else:
        total_params = sum(len(g['params']) for g in agent.optimizer.param_groups)
        print(f"  [OK] Optimizer tracking {total_params} parameter groups")
    
    # 3. Quick forward pass test
    try:
        dummy_state = torch.randn(1, agent.input_channels, 10, 10).to(first_param.device)
        with torch.no_grad():
            output = agent.q_network(dummy_state)
        print(f"  [OK] Forward pass works, output shape: {output.shape}")
    except Exception as e:
        print(f"  [FAIL] Forward pass error: {e}")
        all_ok = False
    
    # 4. Check memory has correct structure
    if hasattr(agent, 'memory') and agent.memory is not None:
        print(f"  [OK] Memory buffer initialized, size: {len(agent.memory)}")
    else:
        print("  [WARN] No memory buffer found")
    
    # 5. Check support vector matches V_MIN/V_MAX
    if hasattr(agent, 'support'):
        v_min = agent.support[0].item()
        v_max = agent.support[-1].item()
        print(f"  [OK] C51 support: [{v_min:.1f}, {v_max:.1f}] with {len(agent.support)} atoms")
    
    print(f"\n[DIAG] Pre-flight {'PASSED' if all_ok else 'FAILED'}\n")
    return all_ok
