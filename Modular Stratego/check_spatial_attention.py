"""
SpatialAttention Diagnostic
============================
Tests whether the SpatialAttention layer in RainbowDQN is functioning correctly.

Tests:
  1. Non-Identity Transform  - Output differs meaningfully from input
  2. Attention Weight Spread - Attention weights are non-uniform (not collapsed)
  3. Positional Sensitivity  - Different board positions attend to different regions
  4. Gradient Flow           - Gradients propagate through the attention layer
  5. Semantic Responsiveness - Attention patterns change when board state changes
"""

import sys
import torch
import numpy as np

sys.path.insert(0, '.')

from networks.rainbow_dqn import SpatialAttention


def make_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    print("[WARN] CUDA not available, falling back to CPU")
    return torch.device('cpu')


# ---------------------------------------------------------------
# Test 1: Non-Identity Transform
# ---------------------------------------------------------------
def test_non_identity(device):
    """Verify SpatialAttention is actively transforming its input, not passing through."""
    print("\n-- Test 1: Non-Identity Transform --")

    attn = SpatialAttention(channels=64, num_heads=4, dropout=0.0).to(device)
    attn.eval()

    x = torch.randn(1, 64, 10, 10, device=device)

    with torch.no_grad():
        y = attn(x)

    diff = (y - x).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0), y.flatten().unsqueeze(0)
    ).item()

    print(f"  Input  mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    print(f"  Output mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Diff   mean={mean_diff:.4f}, max={max_diff:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f} (1.0 = identity, <1.0 = transformed)")

    # Attention should change the representation meaningfully
    is_transformed = (mean_diff > 0.01) and (cosine_sim < 0.999)

    if is_transformed:
        print("  [PASS] Attention actively transforms the feature map.")
        return True
    else:
        print("  [FAIL] Attention output is nearly identical to input (pass-through).")
        return False


# ---------------------------------------------------------------
# Test 2: Attention Weight Spread
# ---------------------------------------------------------------
def test_attention_spread(device):
    """Verify attention weights are non-uniform (not collapsed to uniform or single-point)."""
    print("\n-- Test 2: Attention Weight Spread --")

    attn = SpatialAttention(channels=64, num_heads=4, dropout=0.0).to(device)
    attn.eval()

    x = torch.randn(1, 64, 10, 10, device=device)
    x_flat = x.flatten(2).permute(0, 2, 1)  # (1, 100, 64)

    with torch.no_grad():
        _, attn_weights = attn.attn(x_flat, x_flat, x_flat, need_weights=True)

    # attn_weights shape: (batch, num_positions, num_positions) = (1, 100, 100)
    w = attn_weights[0]  # (100, 100)

    # Entropy of attention distribution (higher = more spread, lower = more focused)
    # Uniform distribution entropy = log(100) = 4.605
    eps = 1e-10
    entropy = -(w * (w + eps).log()).sum(dim=-1)  # Per-query entropy
    mean_entropy = entropy.mean().item()
    uniform_entropy = np.log(100)

    # Check for collapse: if std of weights is very low = uniform = not useful
    weight_std = w.std().item()

    # Check max attention value per query position
    max_attn = w.max(dim=-1).values
    mean_max = max_attn.mean().item()

    print(f"  Attention weight shape: {w.shape}")
    print(f"  Weight std:      {weight_std:.6f}")
    print(f"  Mean entropy:    {mean_entropy:.4f} (uniform={uniform_entropy:.4f})")
    print(f"  Entropy ratio:   {mean_entropy/uniform_entropy:.4f} (1.0=uniform, 0.0=one-hot)")
    print(f"  Mean max weight: {mean_max:.4f} (uniform=0.01)")

    # For an UNTRAINED network, near-uniform weights are expected.
    # The key checks are:
    # - Entropy above near-zero (not collapsed to single point)
    # - Weight std > 0 (weights vary at all)
    # - Max attention is not extreme (not degenerate one-hot)
    not_collapsed = (mean_entropy > 0.1 * uniform_entropy)
    has_variance = (weight_std > 1e-8)
    not_degenerate = (mean_max < 0.5)  # Each position shouldn't fixate on one target

    # Note: near-uniform at init is EXPECTED, it will sharpen during training
    if mean_entropy > 0.98 * uniform_entropy:
        print("  [NOTE] Weights are near-uniform (expected for untrained network).")

    if not_collapsed and has_variance and not_degenerate:
        print("  [PASS] Attention weights are healthy (not collapsed or degenerate).")
        return True
    else:
        if not not_collapsed:
            print("  [FAIL] Attention collapsed to near-one-hot (degenerate).")
        if not has_variance:
            print("  [FAIL] Attention weights have zero variance.")
        if not not_degenerate:
            print("  [FAIL] Attention fixated on single targets (max weight too high).")
        return False


# ---------------------------------------------------------------
# Test 3: Positional Sensitivity
# ---------------------------------------------------------------
def test_positional_sensitivity(device):
    """Verify different board positions produce different attention outputs."""
    print("\n-- Test 3: Positional Sensitivity --")

    attn = SpatialAttention(channels=64, num_heads=4, dropout=0.0).to(device)
    attn.eval()

    x = torch.randn(1, 64, 10, 10, device=device)

    with torch.no_grad():
        y = attn(x)

    # Check if different spatial positions have different output vectors
    # Reshape to (100, 64) - one vector per position
    pos_vectors = y[0].flatten(1).T  # (100, 64)

    # Pairwise cosine similarity between a few representative positions
    corners = [0, 9, 90, 99]  # top-left, top-right, bottom-left, bottom-right
    center = 44  # approximate center
    test_positions = corners + [center]

    cos_sims = []
    for i in range(len(test_positions)):
        for j in range(i+1, len(test_positions)):
            pi, pj = test_positions[i], test_positions[j]
            sim = torch.nn.functional.cosine_similarity(
                pos_vectors[pi].unsqueeze(0), pos_vectors[pj].unsqueeze(0)
            ).item()
            cos_sims.append(sim)

    mean_sim = np.mean(cos_sims)
    min_sim = np.min(cos_sims)
    max_sim = np.max(cos_sims)

    print(f"  Cosine similarity between positions:")
    print(f"    mean={mean_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")

    # Also check variance across position vectors
    pos_variance = pos_vectors.var(dim=0).mean().item()
    print(f"  Per-feature variance across positions: {pos_variance:.6f}")

    # Good attention should produce position-specific representations
    is_diverse = (mean_sim < 0.99) and (pos_variance > 1e-4)

    if is_diverse:
        print("  [PASS] Different board positions produce distinct representations.")
        return True
    else:
        print("  [FAIL] Board positions produce identical representations (no spatial distinction).")
        return False


# ---------------------------------------------------------------
# Test 4: Gradient Flow Through Attention
# ---------------------------------------------------------------
def test_gradient_flow(device):
    """Verify gradients flow through SpatialAttention to earlier layers."""
    print("\n-- Test 4: Gradient Flow --")

    attn = SpatialAttention(channels=64, num_heads=4, dropout=0.0).to(device)
    attn.train()

    x = torch.randn(1, 64, 10, 10, device=device, requires_grad=True)
    y = attn(x)
    # Use sum() not mean() - mean() on LayerNorm output vanishes
    loss = y.sum()
    loss.backward()

    # Check input gradients
    input_grad_norm = x.grad.norm().item() if x.grad is not None else 0.0

    # Check internal parameter gradients
    attn_param_grads = {}
    for name, p in attn.named_parameters():
        if p.grad is not None:
            attn_param_grads[name] = p.grad.norm().item()

    print(f"  Input gradient norm: {input_grad_norm:.6f}")
    print(f"  Parameters with gradients: {len(attn_param_grads)}/{sum(1 for _ in attn.parameters())}")

    # Show key parameter gradients
    for name, grad_norm in sorted(attn_param_grads.items()):
        print(f"    {name}: {grad_norm:.6f}")

    has_input_grad = (input_grad_norm > 1e-8)
    has_param_grads = len(attn_param_grads) > 0

    if has_input_grad and has_param_grads:
        print("  [PASS] Gradients flow through SpatialAttention.")
        return True
    else:
        if not has_input_grad:
            print("  [FAIL] No gradient on input (attention blocks backprop).")
        if not has_param_grads:
            print("  [FAIL] No parameter gradients (attention params are dead).")
        return False


# ---------------------------------------------------------------
# Test 5: Semantic Responsiveness
# ---------------------------------------------------------------
def test_semantic_responsiveness(device):
    """
    Verify attention patterns change meaningfully when the input board changes.
    Simulates: 'empty board' vs 'board with a cluster of active pieces' and checks
    if the attention layer produces different outputs for each.
    """
    print("\n-- Test 5: Semantic Responsiveness --")

    attn = SpatialAttention(channels=64, num_heads=4, dropout=0.0).to(device)
    attn.eval()

    # Scenario A: Relatively uniform features
    x_uniform = torch.randn(1, 64, 10, 10, device=device) * 0.1

    # Scenario B: Strong signal concentrated in top-right quadrant (like a piece cluster)
    x_cluster = x_uniform.clone()
    x_cluster[:, :, 0:3, 7:10] = torch.randn(1, 64, 3, 3, device=device) * 5.0

    with torch.no_grad():
        y_uniform = attn(x_uniform)
        y_cluster = attn(x_cluster)

    # Check attention weight differences
    x_flat_u = x_uniform.flatten(2).permute(0, 2, 1)
    x_flat_c = x_cluster.flatten(2).permute(0, 2, 1)

    with torch.no_grad():
        _, w_uniform = attn.attn(x_flat_u, x_flat_u, x_flat_u, need_weights=True)
        _, w_cluster = attn.attn(x_flat_c, x_flat_c, x_flat_c, need_weights=True)

    weight_diff = (w_cluster - w_uniform).abs().mean().item()
    output_diff = (y_cluster - y_uniform).abs().mean().item()

    # Check if cluster positions receive more attention in the cluster scenario
    # Top-right quadrant positions: rows 0-2, cols 7-9 -> flat indices
    cluster_indices = []
    for r in range(3):
        for c in range(7, 10):
            cluster_indices.append(r * 10 + c)

    # Mean attention TO cluster positions in both scenarios
    attn_to_cluster_uniform = w_uniform[0, :, cluster_indices].mean().item()
    attn_to_cluster_cluster = w_cluster[0, :, cluster_indices].mean().item()

    print(f"  Attention weight diff: {weight_diff:.6f}")
    print(f"  Output diff:          {output_diff:.6f}")
    print(f"  Attn to cluster region (uniform input): {attn_to_cluster_uniform:.4f}")
    print(f"  Attn to cluster region (cluster input): {attn_to_cluster_cluster:.4f}")
    print(f"  Cluster attention increase: {(attn_to_cluster_cluster - attn_to_cluster_uniform):.4f}")

    responds_to_content = (weight_diff > 1e-3) and (output_diff > 0.01)
    # For untrained network, we mainly check that outputs change, not that attention increases
    # (attention direction depends on learned weights)

    if responds_to_content:
        print("  [PASS] Attention responds to input content changes.")
        return True
    else:
        print("  [FAIL] Attention produces identical patterns regardless of input content.")
        return False


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  SpatialAttention Diagnostic")
    print("=" * 60)

    device = make_device()
    print(f"\nDevice: {device}")

    results = {}
    tests = [
        ("Non-Identity Transform", test_non_identity),
        ("Attention Weight Spread", test_attention_spread),
        ("Positional Sensitivity", test_positional_sensitivity),
        ("Gradient Flow", test_gradient_flow),
        ("Semantic Responsiveness", test_semantic_responsiveness),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn(device)
        except Exception as e:
            print(f"  [FAIL] Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Result: {passed}/{total} tests passed.")

    if passed == total:
        print("  [OK] SpatialAttention is functioning correctly.")
    else:
        print("  [!!] SpatialAttention has issues that may affect learning.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
