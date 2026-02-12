# Stratego Modular: Project Status & Agent Guide

**Last Updated:** 2026-02-12
**Objective:** Achieve human-like or better strategic play in Stratego (Partial Observability).

## Recent Progress
- **AAREN Integration Verified:** All 6 diagnostic tests passed (`check_aaren_integration.py`). AAREN correctly feeds embeddings into the Rainbow DQN — dimensions match, embeddings are non-zero, channel concatenation works, gradients flow correctly, and no sparse death detected.
- **PBS Code Fully Removed:** All legacy PBS code has been deleted from the codebase. The `pbs/` directory, `pbs_evaluator.py`, and `pbs_visualizer.py` are gone. All runtime references (`get_uncertainty_map()`, `ProbabilisticBeliefState`, `train_pbs_evaluator()`) have been replaced with AAREN equivalents via `HistoryAggregator`. Final stale PBS variable names (`opponent_uses_pbs`, `lane_opponent_uses_pbs`, `opp_uses_pbs`) in `train_dqn.py` have been renamed to their `_history` equivalents. Backward-compatibility aliases (`self.pbs`, `use_pbs`) remain as thin wrappers to the AAREN history system.
- **PBS Ablation:** Explicit Probabilistic Belief States (PBS) fully replaced by implicit AAREN memory. The `HistoryAggregator` wraps AAREN and produces per-position embeddings. Uncertainty is handled implicitly via learned embeddings rather than explicit `get_uncertainty_map()` calls.
- **Sparse Death Fix:** Learnable default embedding implemented. Verified stable — after 5 simulated updates, all 5 positions produce diverse, non-zero embeddings.
- **Exploration:** Epsilon-Greedy disabled (`EXPLORATION_EPSILON_START = 0.0`). Noisy Networks handle state-dependent exploration.
- **Reward Consolidation:** All rewards centralized in `distributional_reward.py` via `UnifiedRewardShaper`.
- **Computational Optimization:** AMP (Mixed-Precision) and `torch.compile` support for faster throughput.

## AAREN Data Flow (Verified)

The AAREN → DQN pipeline operates as follows:

1. **Feature Extraction** — `HistoryAggregator._extract_action_features()` converts each opponent action into a 24-dimensional feature vector (direction, distance, position, attack flag, scout indicator, activity, etc.).
2. **Sequential Encoding** — `PieceActionAaren.forward_sequential()` processes each action feature through a 3-layer AAREN stack (input_size=24, hidden_size=64) producing per-position hidden states. This runs under `torch.no_grad()` during inference.
3. **Embedding Assembly** — `HistoryAggregator.get_embedding_tensor()` packs all per-position hidden states into a `(64, 10, 10)` tensor. Positions with no history remain zero.
4. **Channel Concatenation** — `RainbowAgent.get_state_representation()` concatenates:
   - Board features: `(15, 10, 10)` — 12 own-piece channels + enemy presence + obstacles + empty
   - AAREN embeddings: `(64, 10, 10)` — implicit belief state from action history
   - **Total input: `(79, 10, 10)`**
5. **DQN Consumption** — `RainbowDQN.forward()` processes the 79-channel tensor through `conv_in` → 6 ResBlocks → SpatialAttention → Dueling Heads → C51 distribution over 400 actions × 51 atoms.

AAREN trains separately via reveal data (supervised, predicting piece types from action sequences). The DQN learns to interpret AAREN embeddings as implicit belief features through end-to-end gradient flow on the conv_in layer.

## ResNet Backbone Analysis

| Component | Parameters | Share |
|-----------|-----------|-------|
| conv_in + bn_in | 45,568 | 0.2% |
| **ResBlocks (6×)** | **443,136** | **2.0%** |
| SpatialAttention | 33,472 | 0.2% |
| Value Head | 77,991 | 0.4% |
| **Advantage Head** | **21,136,354** | **97.2%** |
| **Total** | **21,736,521** | 100% |

**Key observations:**
- **Receptive field: 27x27** — 13 conv layers (1 initial + 2 per block x 6) fully cover the 10x10 board.
- **Feature maps healthy:** After 6 ResBlocks: mean=1.60, std=1.60, max=9.99. After SpatialAttention: normalized to mean=0.00, std=1.00 (LayerNorm working).
- **Bottleneck is NOT the ResNet.** The backbone is only 2% of total parameters. The advantage head (NoisyLinear 200->512->20,400) dominates at 97.2%.
- **Increasing ResNet blocks is not recommended at this stage.** The 6 blocks already provide full receptive field coverage and the current bottleneck is in exploration/reward signal, not representation capacity. Adding blocks would increase training cost with marginal benefit. The SpatialAttention layer already captures long-range relationships that deeper ResNets would otherwise require.

## SpatialAttention Status (Verified)

Diagnostic (`check_spatial_attention.py`) — **5/5 tests passed:**

| Test | Result | Key Values |
|------|--------|------------|
| Non-Identity Transform | PASS | Cosine sim=0.97; attention actively modifies features, not a pass-through |
| Attention Weight Spread | PASS | Not collapsed or degenerate; near-uniform at init (expected for untrained weights) |
| Positional Sensitivity | PASS | Each board position produces a distinct representation (mean cosine sim=0.01) |
| Gradient Flow | PASS | 12/12 attention parameters receive gradients; in_proj_weight grad norm=5130 in full DQN |
| Semantic Responsiveness | PASS | Output diff=0.54 between uniform vs piece-clustered input |

The attention layer operates as a standard transformer block: multi-head self-attention (4 heads, 64-dim) with LayerNorm and feed-forward network. It allows each board position to attend to all other positions, enabling the network to reason about long-range piece relationships (e.g., attacker-defender pairings, flag proximity) without requiring additional ResNet depth.

## Current Results
- **Phase 1 (Full Obs):** High win rates (>90%) against random and basic heuristic opponents.
- **Phase 2 (Partial Obs):** Agents struggle with high draw rates and passive shuffling. Being addressed via anti-stall rewards and material advantage incentives.
- **Inference:** AAREN generates piece-identity embeddings that correlate with true piece types; ResNet interpretation is still maturing.

## Known Struggles & Fixes
| Struggle | Status | Resolution |
| :--- | :--- | :--- |
| **Sparse Death** | Fixed | Learnable default embedding; verified stable via diagnostics. |
| **High Draw Rate** | Ongoing | Material advantage reward for draws; step penalties. |
| **Gradient Flow** | Fixed | Multi-step returns (n=3) and centralized scaling. |
| **Action Entropy** | High | Noisy Networks help; strategic focus remains a challenge. |

## Tech Stack
- **Core:** Rainbow DQN (C51, Dueling, Noisy, Multi-step, PER).
- **History:** AAREN (Action-Augmented Recurrent Encoder) — 3-layer, 64-dim, parallel training / O(1) inference.
- **Vision:** 6-block ResNet (64ch) + Spatial Self-Attention (4-head). Total 21.7M params.
- **Training:** Curriculum-based (5 Phases), League-based opponent pool.

## Future Directions
1. **Strategic Intent:** Moving pieces with purpose towards high-value targets.
2. **Belief Interpretation:** Improving how the agent uses AAREN embeddings to bluff or deduce.
3. **Decisiveness:** Reducing repetitive moves in the mid-game.

---
> **Tip:** Check `training_config.py` for current hyperparameters. Run `check_aaren_integration.py` to verify AAREN-DQN pipeline health. Run `visual_train_dqn.py` to observe agent behavior in real-time.
