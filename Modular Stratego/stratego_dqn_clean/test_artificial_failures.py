"""
Artificial Failure Tests for Self-Check Protocols

GATE 2: All artificial failure tests must trigger alerts.
No false negatives allowed.

Tests:
1. Zero gradients (via p.grad.zero_())
2. Constant Q-values 
3. AAREN collapse (σ < 0.01)
4. Memory overflow simulation
5. Convergence failure simulation
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import numpy as np

from models.aaren import AAREN
from models.double_dqn import DoubleDQN
from training.selfcheck import (
    SelfCheckSuite,
    GradientVitalityMonitor,
    AARENCollapseDetection,
    QValueSanityCheck,
    MemoryGuardian,
    LearningValidationCheckpoint
)


def test_zero_gradients():
    """
    Test 1: Inject zero gradients and verify Gradient Vitality Monitor triggers.
    """
    print("\n" + "=" * 60)
    print("TEST 1: ZERO GRADIENTS (Artificial Dead Gradients)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DoubleDQN(action_dim=100).to(device)
    monitor = GradientVitalityMonitor(model, enabled=True)
    
    # Run forward/backward to create gradients
    fake_input = torch.randn(32, 79, 10, 10, device=device)
    output = model(fake_input)
    loss = output.mean()
    loss.backward()
    
    # Zero out all gradients artificially
    print("  Injecting zero gradients via p.grad.zero_()...")
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Run checks multiple times to trigger consecutive failure threshold
    alert_triggered = False
    for i in range(10):  # More than CONSECUTIVE_FAILURES (5)
        result = monitor.check(step=(i + 1) * 100)
        if not result.passed:
            alert_triggered = True
            print(f"  Check {i+1}: ALERT - {result.message}")
    
    if alert_triggered:
        print("[PASS] Zero gradient alert triggered correctly")
        return True
    else:
        print("[FAIL] Zero gradient alert NOT triggered!")
        return False


def test_constant_q_values():
    """
    Test 2: Inject constant Q-values and verify Q-Value Sanity Check triggers.
    """
    print("\n" + "=" * 60)
    print("TEST 2: CONSTANT Q-VALUES")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    check = QValueSanityCheck(enabled=True)
    
    # Create constant Q-values (all same value)
    print("  Injecting constant Q-values (variance = 0)...")
    constant_q = torch.ones(32, 100, device=device) * 5.0  # All 5.0
    
    result = check.check(q_values=constant_q, online_target_mse=0.1)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] Constant Q-value alert triggered correctly")
        return True
    else:
        print("[FAIL] Constant Q-value alert NOT triggered!")
        return False


def test_aaren_collapse():
    """
    Test 3: Inject collapsed AAREN output and verify detection.
    """
    print("\n" + "=" * 60)
    print("TEST 3: AAREN COLLAPSE (σ < 0.01)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    check = AARENCollapseDetection(enabled=True)
    
    # Create collapsed output (very low variance)
    print("  Injecting collapsed AAREN output (σ ≈ 0.001)...")
    collapsed_output = torch.ones(32, 64, device=device) * 0.5  # All same
    collapsed_output += torch.randn(32, 64, device=device) * 0.001  # Tiny variance
    
    result = check.check(collapsed_output)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] AAREN collapse alert triggered correctly")
        return True
    else:
        print("[FAIL] AAREN collapse alert NOT triggered!")
        return False


def test_aaren_drift():
    """
    Test 4: Inject drifted AAREN output (|μ| > 5.0) and verify detection.
    """
    print("\n" + "=" * 60)
    print("TEST 4: AAREN DRIFT (|μ| > 5.0)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    check = AARENCollapseDetection(enabled=True)
    
    # Create drifted output (high mean)
    print("  Injecting drifted AAREN output (μ ≈ 10.0)...")
    drifted_output = torch.randn(32, 64, device=device) + 10.0  # Mean ≈ 10
    
    result = check.check(drifted_output)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] AAREN drift alert triggered correctly")
        return True
    else:
        print("[FAIL] AAREN drift alert NOT triggered!")
        return False


def test_sparse_death():
    """
    Test 5: Inject sparse AAREN output (>50% zeros) and verify detection.
    """
    print("\n" + "=" * 60)
    print("TEST 5: SPARSE GRADIENT DEATH (>50% zeros)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    check = AARENCollapseDetection(enabled=True)
    
    # Create sparse output (many zeros)
    print("  Injecting sparse AAREN output (60% zeros)...")
    sparse_output = torch.randn(32, 64, device=device)
    mask = torch.rand(32, 64, device=device) < 0.6  # 60% zeros
    sparse_output[mask] = 0.0
    
    result = check.check(sparse_output)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] Sparse death alert triggered correctly")
        return True
    else:
        print("[FAIL] Sparse death alert NOT triggered!")
        return False


def test_network_identical():
    """
    Test 6: Make online == target network and verify detection.
    """
    print("\n" + "=" * 60)
    print("TEST 6: IDENTICAL NETWORKS (online == target)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    check = QValueSanityCheck(enabled=True)
    
    # Pass MSE = 0 to simulate identical networks
    print("  Simulating online == target (MSE = 0)...")
    q_values = torch.randn(32, 100, device=device)  # Normal Q-values with variance
    
    result = check.check(q_values=q_values, online_target_mse=0.0)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] Identical network alert triggered correctly")
        return True
    else:
        print("[FAIL] Identical network alert NOT triggered!")
        return False


def test_convergence_failure():
    """
    Test 7: Simulate convergence failure (0% win rate after 50k steps).
    """
    print("\n" + "=" * 60)
    print("TEST 7: CONVERGENCE FAILURE (0% win rate after 50k steps)")
    print("=" * 60)
    
    check = LearningValidationCheckpoint(enabled=True)
    
    # Record 0% win rate
    print("  Simulating 0% win rate at step 60000...")
    check.record_evaluation(step=60000, win_rate=0.0, games_played=100)
    
    result = check.check(current_step=60000)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] Convergence failure alert triggered correctly")
        return True
    else:
        print("[FAIL] Convergence failure alert NOT triggered!")
        return False


def test_premature_exploitation():
    """
    Test 8: Simulate premature exploitation (entropy < 0.5 bits).
    """
    print("\n" + "=" * 60)
    print("TEST 8: PREMATURE EXPLOITATION (entropy < 0.5 bits)")
    print("=" * 60)
    
    check = LearningValidationCheckpoint(enabled=True)
    
    # Record low entropy values
    print("  Simulating low action entropy (0.3 bits)...")
    for _ in range(100):
        check.record_action_entropy(0.3)
    
    result = check.check(current_step=10000)
    
    if not result.passed:
        print(f"  ALERT: {result.message}")
        print("[PASS] Premature exploitation alert triggered correctly")
        return True
    else:
        print("[FAIL] Premature exploitation alert NOT triggered!")
        return False


def main():
    """Run all artificial failure tests."""
    print("\n" + "=" * 60)
    print("GATE 2: ARTIFICIAL FAILURE TESTS")
    print("All tests must trigger alerts. No false negatives allowed.")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results['zero_gradients'] = test_zero_gradients()
    results['constant_q'] = test_constant_q_values()
    results['aaren_collapse'] = test_aaren_collapse()
    results['aaren_drift'] = test_aaren_drift()
    results['sparse_death'] = test_sparse_death()
    results['identical_networks'] = test_network_identical()
    results['convergence_failure'] = test_convergence_failure()
    results['premature_exploitation'] = test_premature_exploitation()
    
    # Summary
    print("\n" + "=" * 60)
    print("GATE 2 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ GATE 2 PASSED: All artificial failures correctly detected.\n")
        return 0
    else:
        print("\n✗ GATE 2 FAILED: Some failures were not detected!\n")
        return 1


if __name__ == "__main__":
    exit(main())
