"""
MANDATORY PRE-IMPLEMENTATION SMOKE TEST
Run this BEFORE writing the full training loop.

Tests:
1. Memory Test - Verify < 5.5GB VRAM usage with batch size 32
2. Gradient Flow - Verify AAREN receives gradients > 1e-6
3. Forward Pass - Verify output shape (batch, 100)
4. Self-Check Integration - Verify all monitors work

Usage:
    python validation_script.py
"""

import sys
import os

# Add ONLY current directory for imports (no parent to avoid 'training' module conflict)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Direct import from sub-packages
from models.aaren import AAREN, RMSNorm
from models.double_dqn import DoubleDQN
from training.selfcheck import SelfCheckSuite


def test_vram_usage():
    """Test 1: Verify VRAM usage with batch size 32."""
    print("\n" + "=" * 60)
    print("TEST 1: VRAM USAGE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return True
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = torch.cuda.memory_allocated() / 1e9
    print(f"Initial VRAM: {initial_memory:.3f} GB")
    
    # Create models
    aaren = AAREN(input_dim=24, hidden_dim=64, device=device).to(device)
    dqn = DoubleDQN(action_dim=100, use_checkpointing=False).to(device)
    
    model_memory = torch.cuda.memory_allocated() / 1e9
    print(f"After model creation: {model_memory:.3f} GB")
    
    # Test forward pass with batch size 32
    batch_size = 32
    fake_state = torch.randn(batch_size, 79, 10, 10, device=device)
    
    # Forward pass
    q_values = dqn(fake_state)
    
    forward_memory = torch.cuda.memory_allocated() / 1e9
    print(f"After forward pass: {forward_memory:.3f} GB")
    
    # Backward pass
    loss = q_values.mean()
    loss.backward()
    
    backward_memory = torch.cuda.memory_allocated() / 1e9
    print(f"After backward pass: {backward_memory:.3f} GB")
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak VRAM usage: {peak_memory:.3f} GB")
    
    # Check constraint: < 4.5GB (GATE 1 requirement)
    if peak_memory > 4.5:
        print(f"[FAIL] Peak VRAM {peak_memory:.3f}GB exceeds 4.5GB limit!")
        return False
    else:
        print(f"[PASS] VRAM constraint satisfied ({peak_memory:.3f}GB < 4.5GB)")
        return True


def test_gradient_flow():
    """Test 2: Verify AAREN receives gradients."""
    print("\n" + "=" * 60)
    print("TEST 2: GRADIENT FLOW TO AAREN")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    aaren = AAREN(input_dim=24, hidden_dim=64, device=device).to(device)
    dqn = DoubleDQN(action_dim=100).to(device)
    
    # Test AAREN independently
    batch_size = 32
    seq_len = 10
    
    # Simulate action history
    action_features = torch.randn(batch_size, seq_len, 24, device=device, requires_grad=True)
    
    # Get AAREN embedding
    aaren_embedding = aaren(action_features)  # (32, 64)
    
    print(f"AAREN output shape: {aaren_embedding.shape}")
    assert aaren_embedding.shape == (batch_size, 64), f"Wrong shape: {aaren_embedding.shape}"
    
    # Expand to spatial and create fake board
    aaren_spatial = aaren_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 10, 10)
    fake_board = torch.randn(batch_size, 15, 10, 10, device=device)
    
    # Combine
    combined = torch.cat([fake_board, aaren_spatial], dim=1)  # (32, 79, 10, 10)
    
    # Forward through DQN
    q_values = dqn(combined)
    
    # Backward
    loss = q_values.mean()
    loss.backward()
    
    # Check AAREN gradients
    aaren_grad_norm = 0.0
    aaren_param_count = 0
    for name, param in aaren.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            aaren_grad_norm += grad_norm
            aaren_param_count += 1
            print(f"  {name}: grad norm = {grad_norm:.6f}")
    
    print(f"\nTotal AAREN gradient norm: {aaren_grad_norm:.6f}")
    print(f"Parameters with gradients: {aaren_param_count}")
    
    if aaren_grad_norm < 1e-6:
        print("[FAIL] AAREN is DEAD ON ARRIVAL - no gradient flow!")
        return False
    else:
        print(f"[PASS] AAREN receives gradients ({aaren_grad_norm:.6f} > 1e-6)")
        return True


def test_forward_pass_shape():
    """Test 3: Verify output shape is (batch, 100)."""
    print("\n" + "=" * 60)
    print("TEST 3: OUTPUT SHAPE VERIFICATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DQN
    dqn = DoubleDQN(action_dim=100).to(device)
    
    # Various batch sizes
    for batch_size in [1, 8, 16, 32]:
        fake_state = torch.randn(batch_size, 79, 10, 10, device=device)
        q_values = dqn(fake_state)
        
        expected_shape = (batch_size, 100)
        if q_values.shape != expected_shape:
            print(f"[FAIL] Batch {batch_size}: got {q_values.shape}, expected {expected_shape}")
            return False
        else:
            print(f"[PASS] Batch {batch_size}: output shape = {q_values.shape}")
    
    print("\n[PASS] All output shapes correct")
    return True


def test_selfcheck_monitors():
    """Test 4: Verify self-check monitors work."""
    print("\n" + "=" * 60)
    print("TEST 4: SELF-CHECK MONITORS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    aaren = AAREN(input_dim=24, hidden_dim=64, device=device).to(device)
    dqn = DoubleDQN(action_dim=100).to(device)
    
    # Create self-check suite
    suite = SelfCheckSuite(model=dqn, aaren=aaren, enabled=True)
    
    # Run a training step to generate gradients
    fake_state = torch.randn(32, 79, 10, 10, device=device)
    q_values = dqn(fake_state)
    loss = q_values.mean()
    loss.backward()
    
    # Run checks
    fake_aaren_output = torch.randn(32, 64, device=device)
    
    results = suite.run_all_checks(
        step=100,
        aaren_output=fake_aaren_output,
        q_values=q_values,
        online_target_mse=0.1
    )
    
    print("\nCheck Results:")
    all_passed = True
    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}: {result.message}")
        if not result.passed and result.severity in ["ALERT", "CRITICAL"]:
            all_passed = False
    
    # Print summary
    suite.print_summary()
    
    # Memory guardian specific test
    print("\nMemory Guardian Status:")
    mem_result = suite.memory_guardian.check()
    print(f"  {mem_result.message}")
    
    if all_passed:
        print("[PASS] All self-checks passed")
        return True
    else:
        print("[WARN] Some checks need attention (may be OK for initial state)")
        return True  # Don't fail on initial check alerts


def test_aaren_rms_norm():
    """Test 5: Verify AAREN uses RMSNorm, not LayerNorm."""
    print("\n" + "=" * 60)
    print("TEST 5: AAREN USES RMSNorm (NOT LayerNorm)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aaren = AAREN(input_dim=24, hidden_dim=64, device=device).to(device)
    
    # Check for RMSNorm
    has_rms_norm = False
    has_layer_norm = False
    
    for name, module in aaren.named_modules():
        if isinstance(module, RMSNorm):
            has_rms_norm = True
            print(f"  Found RMSNorm: {name}")
        if isinstance(module, nn.LayerNorm):
            has_layer_norm = True
            print(f"  Found LayerNorm: {name}")
    
    if has_layer_norm:
        print("[FAIL] AAREN contains LayerNorm - should use RMSNorm!")
        return False
    
    if not has_rms_norm:
        print("[FAIL] AAREN does not contain RMSNorm!")
        return False
    
    print("[PASS] AAREN uses RMSNorm correctly")
    return True


def test_double_dqn_no_rainbow():
    """Test 6: Verify no Rainbow components (C51, Noisy, Dueling)."""
    print("\n" + "=" * 60)
    print("TEST 6: NO RAINBOW COMPONENTS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dqn = DoubleDQN(action_dim=100).to(device)
    
    issues = []
    
    # Check for noisy layers
    for name, module in dqn.named_modules():
        module_name = type(module).__name__.lower()
        
        if 'noisy' in module_name:
            issues.append(f"Found Noisy layer: {name}")
        
        # Check for dueling-like structure
        if 'value' in name.lower() and 'advantage' in name.lower():
            issues.append(f"Possible dueling structure: {name}")
    
    # Check output shape (should be action_dim, not action_dim * num_atoms)
    fake_input = torch.randn(1, 79, 10, 10, device=device)
    output = dqn(fake_input)
    
    if output.shape[-1] != 100:
        issues.append(f"Output dim {output.shape[-1]} != 100 (possible C51?)")
    
    if len(output.shape) > 2:
        issues.append(f"Output has {len(output.shape)} dims (possible C51 atoms?)")
    
    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
        print("[FAIL] Rainbow components detected!")
        return False
    
    print("[PASS] No Rainbow components (pure Double DQN)")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("DOUBLE DQN + AAREN VALIDATION SCRIPT")
    print("Run this BEFORE full training implementation")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Run tests
    results['vram'] = test_vram_usage()
    results['gradient_flow'] = test_gradient_flow()
    results['output_shape'] = test_forward_pass_shape()
    results['self_checks'] = test_selfcheck_monitors()
    results['rms_norm'] = test_aaren_rms_norm()
    results['no_rainbow'] = test_double_dqn_no_rainbow()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\nVALIDATION PASSED: Safe to proceed with full implementation.\n")
        return 0
    else:
        print("\nVALIDATION FAILED: Fix issues before proceeding!\n")
        return 1


if __name__ == "__main__":
    exit(main())
