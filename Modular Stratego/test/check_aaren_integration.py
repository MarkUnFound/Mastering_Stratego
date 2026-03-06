"""
AAREN → Rainbow DQN Integration Check
=======================================
Verifies that AAREN correctly inputs to and outputs from the Rainbow DQN model.
If any test fails, the agent will not learn from AAREN embeddings.

Tests:
  1. Dimension Match       - AAREN hidden_size matches DQN input channels
  2. Embedding Generation  - AAREN produces non-zero embeddings after updates
  3. Channel Concatenation - Board + AAREN channels correctly assembled
  4. DQN Forward Pass      - RainbowDQN accepts the concatenated tensor
  5. Gradient Flow         - DQN backward pass produces gradients on conv_in (AAREN channels)
  6. Sparse Death          - Multiple updates produce active embeddings (not all zeros)
"""

import sys
import torch
import numpy as np

# Ensure project modules are importable
sys.path.insert(0, '.')

from drqn_agent import DQNAgent
from history_aggregator import HistoryAggregator
from aaren.network import PieceActionAaren
from networks.rainbow_dqn import RainbowDQN
from training_config import AAREN_HIDDEN_SIZE, AAREN_NUM_LAYERS, HISTORY_EMBEDDING_SIZE
from game_state import GameState
from environment import StrategoEnvironment


def make_device():
    """Use CUDA when available."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    print("[WARN] CUDA not available, falling back to CPU")
    return torch.device('cpu')


def make_dummy_game_state(device):
    """Create a minimal game state with pieces on the board for testing."""
    env = StrategoEnvironment(device=device)
    game_state = env.reset()  # reset() returns a GameState directly
    return game_state, env


def make_dummy_action():
    """Return a simple valid-looking action tuple."""
    return ((6, 0), (5, 0))  # Move from row 6 to row 5


# ─────────────────────────────────────────────────────────────
# Test 1: Dimension Match
# ─────────────────────────────────────────────────────────────
def test_dimension_match(device):
    """Verify AAREN hidden_size = HISTORY_EMBEDDING_SIZE = DQN expected channels."""
    print("\n── Test 1: Dimension Match ──")

    aaren = PieceActionAaren(
        input_size=24, hidden_size=AAREN_HIDDEN_SIZE,
        num_layers=AAREN_NUM_LAYERS, output_size=12, device=device
    ).to(device)

    expected_input_channels = 15 + HISTORY_EMBEDDING_SIZE
    dqn = RainbowDQN(input_shape=(expected_input_channels, 10, 10), output_size=400, num_atoms=51).to(device)

    # Check AAREN hidden matches embedding config
    aaren_ok = (aaren.hidden_size == HISTORY_EMBEDDING_SIZE)
    # Check DQN first conv accepts the right channels
    conv_in_channels = dqn.conv_in.in_channels
    dqn_ok = (conv_in_channels == expected_input_channels)

    print(f"  AAREN hidden_size:        {aaren.hidden_size}")
    print(f"  HISTORY_EMBEDDING_SIZE:   {HISTORY_EMBEDDING_SIZE}")
    print(f"  DQN conv_in.in_channels:  {conv_in_channels}")
    print(f"  Expected input_channels:  {expected_input_channels} (15 board + {HISTORY_EMBEDDING_SIZE} AAREN)")

    if aaren_ok and dqn_ok:
        print("  [PASS] All dimensions match correctly.")
        return True
    else:
        if not aaren_ok:
            print(f"  [FAIL] AAREN hidden_size ({aaren.hidden_size}) != HISTORY_EMBEDDING_SIZE ({HISTORY_EMBEDDING_SIZE})")
        if not dqn_ok:
            print(f"  [FAIL] DQN conv_in expects {conv_in_channels} channels but should be {expected_input_channels}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 2: Embedding Generation
# ─────────────────────────────────────────────────────────────
def test_embedding_generation(device):
    """Verify AAREN produces non-zero embeddings after update() calls."""
    print("\n── Test 2: Embedding Generation ──")

    ha = HistoryAggregator(player_id=1, device=device, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
    game_state, _ = make_dummy_game_state(device)

    # Before any updates, embedding should be all zeros
    emb_before = ha.get_embedding_tensor()
    all_zero_before = (emb_before.abs().sum().item() == 0.0)
    print(f"  Before update - all zeros: {all_zero_before} (expected: True)")

    # Simulate opponent action (player -1 is enemy for player_id=1)
    action = make_dummy_action()
    ha.update(action, game_state, acting_player=-1)

    emb_after = ha.get_embedding_tensor()
    has_nonzero = (emb_after.abs().sum().item() > 1e-8)
    print(f"  After update  - has non-zero values: {has_nonzero}")
    print(f"  Embedding shape: {emb_after.shape}")
    print(f"  Embedding sum:   {emb_after.abs().sum().item():.6f}")
    print(f"  Embedding max:   {emb_after.abs().max().item():.6f}")

    # Check the updated position specifically
    dest_pos = (5, 0)  # Piece moved to this position
    pos_emb = emb_after[:, dest_pos[0], dest_pos[1]]
    pos_nonzero = (pos_emb.abs().sum().item() > 1e-8)
    print(f"  Position {dest_pos} embedding non-zero: {pos_nonzero}")

    if all_zero_before and has_nonzero and pos_nonzero:
        print("  [PASS] AAREN generates non-zero embeddings after update.")
        return True
    else:
        print("  [FAIL] AAREN embedding generation is broken.")
        if not all_zero_before:
            print("    → Embedding was non-zero BEFORE any updates (unexpected)")
        if not has_nonzero:
            print("    → Embedding is STILL all zeros after update (sparse death)")
        if not pos_nonzero:
            print("    → Updated position has zero embedding (position tracking issue)")
        return False


# ─────────────────────────────────────────────────────────────
# Test 3: Channel Concatenation
# ─────────────────────────────────────────────────────────────
def test_channel_concatenation(device):
    """Verify get_state_representation() returns correct shape with AAREN channels."""
    print("\n── Test 3: Channel Concatenation ──")

    agent = DQNAgent(player_id=1, device=device, num_envs=1, use_pbs=True)
    game_state, env = make_dummy_game_state(device)

    # Feed some history so AAREN has data
    action = make_dummy_action()
    if agent.history:
        agent.history.update(action, game_state, acting_player=-1)

    # Get state representation
    state_tensor = agent.get_state_representation(game_state.board, pbs_instance=agent.history)
    expected_channels = 15 + HISTORY_EMBEDDING_SIZE

    print(f"  State tensor shape: {state_tensor.shape}")
    print(f"  Expected shape:     ({expected_channels}, 10, 10)")

    shape_ok = (state_tensor.shape == (expected_channels, 10, 10))

    # Check board channels (first 15) are populated
    board_channels = state_tensor[:15]
    board_has_data = (board_channels.abs().sum().item() > 0)
    print(f"  Board channels (0-14) have data: {board_has_data}")

    # Check AAREN channels (15+) have data
    aaren_channels = state_tensor[15:]
    aaren_has_data = (aaren_channels.abs().sum().item() > 1e-8)
    print(f"  AAREN channels (15-{expected_channels-1}) have data: {aaren_has_data}")
    print(f"  AAREN channels sum: {aaren_channels.abs().sum().item():.6f}")

    if shape_ok and board_has_data and aaren_has_data:
        print("  [PASS] Channel concatenation is correct.")
        return True
    else:
        if not shape_ok:
            print(f"  [FAIL] Wrong shape: got {state_tensor.shape}, expected ({expected_channels}, 10, 10)")
        if not board_has_data:
            print("  [FAIL] Board channels are empty.")
        if not aaren_has_data:
            print("  [FAIL] AAREN channels are all zeros after update.")
        return False


# ─────────────────────────────────────────────────────────────
# Test 4: DQN Forward Pass
# ─────────────────────────────────────────────────────────────
def test_dqn_forward_pass(device):
    """Verify RainbowDQN accepts the concatenated tensor and produces valid output."""
    print("\n── Test 4: DQN Forward Pass ──")

    input_channels = 15 + HISTORY_EMBEDDING_SIZE
    dqn = RainbowDQN(input_shape=(input_channels, 10, 10), output_size=400, num_atoms=51).to(device)

    # Create synthetic input matching the expected shape
    batch_size = 4
    x = torch.randn(batch_size, input_channels, 10, 10, device=device)

    try:
        dqn.eval()
        with torch.no_grad():
            log_probs = dqn(x)

        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {log_probs.shape}")
        print(f"  Expected:     ({batch_size}, 400, 51)")

        shape_ok = (log_probs.shape == (batch_size, 400, 51))

        # Check log_probs are valid (no NaN/Inf, and log-softmax produces values <= 0)
        no_nan = not torch.isnan(log_probs).any().item()
        no_inf = not torch.isinf(log_probs).any().item()
        all_negative = (log_probs.max().item() <= 1e-5)  # log-softmax output should be <= 0

        # Check distributions sum to ~1 (exp of log-softmax)
        probs = log_probs.exp()
        sum_check = probs.sum(dim=2)  # Should be ~1 for each action
        sums_ok = torch.allclose(sum_check, torch.ones_like(sum_check), atol=1e-3)

        print(f"  No NaN: {no_nan} | No Inf: {no_inf} | Valid log-probs: {all_negative}")
        print(f"  Distributions sum to ~1: {sums_ok} (range: {sum_check.min().item():.4f} - {sum_check.max().item():.4f})")

        if shape_ok and no_nan and no_inf and sums_ok:
            print("  [PASS] DQN forward pass produces valid output.")
            return True
        else:
            print("  [FAIL] DQN forward pass has issues.")
            return False

    except RuntimeError as e:
        print(f"  [FAIL] DQN forward pass crashed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Test 5: Gradient Flow
# ─────────────────────────────────────────────────────────────
def test_gradient_flow(device):
    """Verify backward pass produces gradients on conv_in (which processes AAREN channels)."""
    print("\n── Test 5: Gradient Flow ──")

    input_channels = 15 + HISTORY_EMBEDDING_SIZE
    dqn = RainbowDQN(input_shape=(input_channels, 10, 10), output_size=400, num_atoms=51).to(device)
    dqn.train()

    # Create input where ONLY the AAREN channels are non-zero
    # This isolates whether gradients flow through AAREN channel processing
    x = torch.zeros(2, input_channels, 10, 10, device=device, requires_grad=True)

    # Fill only AAREN channels (15+) with signal
    with torch.no_grad():
        x[:, 15:, :, :] = torch.randn(2, HISTORY_EMBEDDING_SIZE, 10, 10, device=device)

    # Forward + backward
    log_probs = dqn(x)
    # Use a simple loss (mean of all log-probs)
    loss = log_probs.mean()
    loss.backward()

    # Check gradients on conv_in weights
    conv_in_grad = dqn.conv_in.weight.grad
    has_grad = conv_in_grad is not None
    print(f"  conv_in has gradients: {has_grad}")

    if has_grad:
        # Check specifically the AAREN channel gradients (channels 15+)
        aaren_channel_grads = conv_in_grad[:, 15:, :, :]
        board_channel_grads = conv_in_grad[:, :15, :, :]

        aaren_grad_norm = aaren_channel_grads.norm().item()
        board_grad_norm = board_channel_grads.norm().item()

        aaren_grad_nonzero = (aaren_grad_norm > 1e-10)

        print(f"  AAREN channel grad norm:  {aaren_grad_norm:.6f}")
        print(f"  Board channel grad norm:  {board_grad_norm:.6f}")
        print(f"  AAREN channels receive gradients: {aaren_grad_nonzero}")

        if aaren_grad_nonzero:
            print("  [PASS] Gradients flow through AAREN channels in conv_in.")
            return True
        else:
            print("  [FAIL] AAREN channels in conv_in receive ZERO gradients.")
            print("    → The DQN cannot learn from AAREN embeddings.")
            return False
    else:
        print("  [FAIL] conv_in has no gradients at all.")
        return False


# ─────────────────────────────────────────────────────────────
# Test 6: Sparse Death Detection
# ─────────────────────────────────────────────────────────────
def test_sparse_death(device):
    """Verify AAREN doesn't collapse to all-zeros after multiple updates."""
    print("\n── Test 6: Sparse Death Detection ──")

    ha = HistoryAggregator(player_id=1, device=device, hidden_size=AAREN_HIDDEN_SIZE, num_layers=AAREN_NUM_LAYERS)
    game_state, _ = make_dummy_game_state(device)

    # Simulate many opponent actions across different positions
    actions = [
        ((6, 0), (5, 0)),
        ((6, 1), (5, 1)),
        ((6, 2), (5, 2)),
        ((7, 3), (6, 3)),
        ((6, 4), (5, 4)),
    ]

    for action in actions:
        ha.update(action, game_state, acting_player=-1)

    emb = ha.get_embedding_tensor()

    # Count active positions (non-zero embeddings)
    active_mask = (emb.abs().sum(dim=0) > 1e-6)
    active_count = active_mask.sum().item()

    print(f"  Actions simulated: {len(actions)}")
    print(f"  Active positions:  {active_count}")
    print(f"  Embedding mean:    {emb.mean().item():.6f}")
    print(f"  Embedding std:     {emb.std().item():.6f}")
    print(f"  Embedding max:     {emb.abs().max().item():.6f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(emb).any().item()
    has_inf = torch.isinf(emb).any().item()
    print(f"  Has NaN: {has_nan} | Has Inf: {has_inf}")

    # Each action should leave at least 1 active position
    # (piece moves from source to dest, so dest gets the embedding)
    min_expected_active = 1  # At least some positions should be active
    active_ok = (active_count >= min_expected_active)
    stable = not has_nan and not has_inf

    # Check variance across positions (if all identical = collapsed)
    if active_count > 1:
        active_embeddings = emb[:, active_mask]  # (hidden_size, num_active)
        # Check if all active embeddings are identical (bad sign)
        pairwise_diff = (active_embeddings[:, 0:1] - active_embeddings[:, 1:]).abs().sum().item()
        diverse = (pairwise_diff > 1e-6)
        print(f"  Embeddings diverse: {diverse} (pairwise_diff={pairwise_diff:.6f})")
    else:
        diverse = True  # Can't check diversity with < 2 active positions
        print(f"  Embeddings diverse: N/A (only {active_count} active positions)")

    if active_ok and stable and diverse:
        print("  [PASS] No sparse death detected. AAREN produces healthy embeddings.")
        return True
    else:
        if not active_ok:
            print(f"  [FAIL] Sparse death: only {active_count} active positions after {len(actions)} updates.")
        if not stable:
            print("  [FAIL] NaN or Inf detected in embeddings (numerical instability).")
        if not diverse:
            print("  [FAIL] All embeddings are identical (collapsed representation).")
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AAREN → Rainbow DQN Integration Check")
    print("=" * 60)

    device = make_device()
    print(f"\nDevice: {device}")

    results = {}
    tests = [
        ("Dimension Match", test_dimension_match),
        ("Embedding Generation", test_embedding_generation),
        ("Channel Concatenation", test_channel_concatenation),
        ("DQN Forward Pass", test_dqn_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("Sparse Death", test_sparse_death),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn(device)
        except Exception as e:
            print(f"  [FAIL] Test crashed with exception: {e}")
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
        print("  [OK] AAREN is correctly integrated with Rainbow DQN.")
        print("  [OK] The agent CAN learn from AAREN embeddings.")
    else:
        print("  [!!] AAREN integration has issues. The agent may NOT learn correctly.")
        print("  [!!] Fix the failing tests above before training.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
