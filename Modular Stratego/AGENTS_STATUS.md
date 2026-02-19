# Stratego Modular: Project Status & Agent Guide

**Last Updated:** 2026-02-19
**Current Checkpoint:** `agent1_rainbow_episode_11000.pth` (Deployed in GUI)

## Recent Progress
- **Episode 11000 Deployment (Feb 19, 2026):** Transitioned the GUI production bot from Episode 1000 to Episode 11000. This model features significantly more stable Q-values and better piece-value recognition.
- **Expectamax-Style Test-Time Search:** Integrated `UpdateEquivalenceSearch` (Ataraxos-style) into the GUI. This replaces shallow 2-ply minimax with a more robust world-sampling approach. By evaluating candidate piece moves across 50 plausible AAREN-generated worlds, the bot mitigates "hallucination" (over-optimistic attacks on hidden pieces).
- **History Tracking in GUI:** Re-enabled AAREN history aggregation in the Pygame GUI. The bot now 'remembers' human moves, allowing for increasingly accurate piece-identity inference as the game progresses.
- **Episode-Level Replay (Trajectory Segment Sampling):** Dual-buffer architecture introduced alongside PER. An `EpisodicReplayBuffer` stores up to 500 complete episodes and samples contiguous 16-step trajectory segments during training. 25% of each training batch is drawn from episode segments (prioritized by game outcome), 75% from PER. This enables temporal credit assignment across multi-move strategies and episode-level prioritization of decisive games.
- **Git Repository Maintenance (Feb 19, 2026):** Resolved a push failure caused by large `.pth` model binaries (up to 1.2GB) exceeding GitHub's 100MB limit. Cleaned the Git history of the `UpdateMemory` branch to remove binary tracking while preserving all LaTeX and Python logic updates. Updated `.gitignore` to prevent accidental staging of `.pth` files.
- **Long-Horizon Tuning:** Architecture adjusted for 200–500 move games. Per-phase turn limits increased (Phase 1: 300, Phase 2: 800, Phase 3: 1000, Phase 4: 1500; `DEFAULT_MAX_TURNS=1500`). Turn limits consolidated into a single `PHASE_MAX_TURNS` dict in `training_config.py` and carried as `PhaseConfig.max_turns` in `curriculum.py` — the duplicated inline dicts in `train_dqn.py` are removed. `MAX_HISTORY_LENGTH` remains 50. Turn normalization in `history_aggregator.py` now dynamically reads `DEFAULT_MAX_TURNS` instead of using a hardcoded `1000.0`. `N_STEPS` set to 5 for deeper multi-step credit assignment (`γ^5 = 0.975`).
- **AAREN Integration Verified:** All 6 diagnostic tests passed (`check_aaren_integration.py`). AAREN correctly feeds embeddings into the Rainbow DQN — dimensions match, embeddings are non-zero, channel concatenation works, gradients flow correctly, and no sparse death detected.
- **AAREN Implicit Memory System:** The MARQ framework now fully relies on AAREN (Attention-as-a-Recurrent-Neural-Network) for history-based piece inference. This replaces legacy explicit belief systems with a DeepNash-inspired implicit representation. All associated modules (`HistoryAggregator`, `PieceActionAaren`) have been verified for gradient flow and output stability.
- **Inferred Piece Identity:** Piece identities are handled implicitly via learned 64-dimensional embeddings. The `HistoryAggregator` processes action sequences to produce these embeddings, which are then interpreted end-to-end by the Rainbow DQN's convolutional backbone. This eliminates the need for explicit probability distributions or manual belief updates.
- **Sparse Death Fix:** Learnable default embedding implemented. Verified stable — after 5 simulated updates, all 5 positions produce diverse, non-zero embeddings.
- **Exploration:** Epsilon-Greedy disabled (`EXPLORATION_EPSILON_START = 0.0`). Noisy Networks handle state-dependent exploration.
- **Reward System Overhaul (v2):** All rewards rescaled for C51 atom visibility. Terminal: ±1.0 (was ±15–25). C51 support tightened to [-10,+10] (was [-30,+30], now 0.4/atom resolution). Piece-loss penalties are rank-weighted. Forward-movement reward gated on piece rank ≤ Captain. Scout bonus tightened to enemy back rank (2 rows). Pending transition double-counting removed. Curiosity tracker resets per episode. Dead info-gain code removed.
- **Literature Review Expanded:** Integrated modern research on DQN variants (Rainbow 2021), ResNets in board games (AlphaZero 2023, Ataraxos 2025), and Transformer-based self-attention (2025). Adhered to a strict 5-year reference rule (2021 or newer) for all new theoretical integrations.
- **Refined Paper Style (Feb 18, 2026)**: Conducted a comprehensive stylistic overhaul of the methodology and technical background chapters. Eliminated informal summaries, em-dashes, and colons in favor of a mature academic tone. Verified all core RL formulas for accuracy and added technical figures for architectural clarity.
- **Computational Optimization:** AMP (Mixed-Precision) and `torch.compile` support for faster throughput.
- **Interpretability Dashboard (Discrete Thinking):** A real-time Matplotlib-based visualizer for frozen Rainbow DQN agents. It features a $2 \times 3$ grid visualizing:
    1. **Main Board:** 10x10 perception grid with AAREN-inferred piece identities.
    2. **C51 PDF:** Real-time atom distribution for selected actions.
    3. **Q-Heatmap:** Expected return intensity across the board.
    4. **Spatial Attention:** Dynamic heatmaps showing inter-square dependency (attending to own flag, enemy marshals, etc.).
    5. **AAREN Latent Space:** PCA projection of the 64-dimensional piece-inference embeddings.

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

AAREN trains via supervised learning on reveal data: when battles occur, piece types are revealed and fed to `update_from_reveal()`. The `train_history()` method periodically trains AAREN using cross-entropy loss on these (action_sequence → piece_type) pairs. The DQN additionally learns to interpret AAREN embeddings through end-to-end gradient flow on the `conv_in` layer.

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

## Current Results (First 1000 Episodes — Phase 1, 300 steps/episode)

| Metric | Value |
|--------|-------|
| **Flag-capture wins** | 30 (3.0%) |
| **Flag-capture losses** | 1 (0.1%) |
| **Depletion losses** | 33 (3.3%) |
| **Draws (timeout)** | 936 (93.6%) |
| **Flag efficiency** | 30:1 (wins:losses) |
| **Avg P1 reward** | 1.74 ± 0.39 |
| **AAREN accuracy** | ~9.7% (vs 8.3% random baseline) |
| **AAREN buffer** | 10,000 (full) |

**Interpretation:** The agent is in pure exploration phase. All 30 wins are flag captures (not depletion), indicating the agent discovered the primary win condition. The 33 depletion losses stem from indiscriminate piece usage — the agent does not yet understand piece-value hierarchy or bomb avoidance. The 93.6% timeout rate reflects extensive exploration without strategic intent to close games. AAREN accuracy is marginally above random, consistent with the reveal-data buffer only recently filling. Q-values are stabilizing (0.26 ± 0.02) as the C51 distribution calibrates. DQN gradient flow is healthy (grad norm 0.012).

## Known Struggles & Fixes
| Struggle | Status | Resolution |
| :--- | :--- | :--- |
| **Sparse Death** | Fixed | Learnable default embedding; verified stable via diagnostics. |
| **Reward Scale vs C51** | Fixed | Terminal ±1.0, shaping 0.01–0.3, support [-10,+10]. Atoms now resolve intermediate rewards. |
| **Double-Counting Rewards** | Fixed | Removed ghost reward from P2's action being credited to P1's pending transition. |
| **Suicidal Forward Bias** | Fixed | Forward reward gated on rank ≤ Captain; high-value pieces don't rush. |
| **Flat Piece-Loss Penalty** | Fixed | Loss penalty now rank-weighted (Marshal 10× Scout). |
| **High Draw Rate** | Ongoing | Material-advantage draws, step penalties, rank-weighted combat incentives. |
| **Gradient Flow** | Fixed | Multi-step returns (n=5) and centralized scaling. |
| **Action Entropy** | High | Noisy Networks help; strategic focus remains a challenge. |
| **Advantage Filtering Slowdown** | Fixed | Disabled for early phases (4× oversampled batch + 2 extra forward passes = 5× slowdown). Re-enable at Phase 3+. |

## Tech Stack
- **Core:** Rainbow DQN (C51, Dueling, Noisy, Multi-step, PER).
- **History:** AAREN (Attention-as-a-Recurrent-Neural-Network) — 3-layer, 64-dim, parallel training / O(1) inference.
- **Vision:** 6-block ResNet (64ch) + Spatial Self-Attention (4-head). Total 21.7M params.
- **Replay:** Dual-buffer — `PrioritizedReplayBuffer` (150K flat) + `EpisodicReplayBuffer` (500 episodes, deque-backed O(1) eviction, 16-step segments, 25% mix ratio).
- **Training:** Curriculum-based (5 Phases), League-based opponent pool.
- **GUI Inference:** `DQNBotLogic` adapter (`Python Stratego Game/dqn_bot_logic.py`) bridges the Pygame GUI to the trained Rainbow DQN. Translates GUI `Board` (Piece objects) → 79-channel tensor → C51 Q-values → Formal Expectamax refinement → legal move selection. AAREN history tracking is strictly enforced turn-by-turn. Compatible checkpoint: `agent1_rainbow_episode_11000.pth`.


The MARQ framework utilizes a formal **Expectamax** search for move refinement at test-time. This replaces standard minimax in partially observable environments by incorporating chance nodes for hidden information.

1. **Max Nodes (Bot Moves)** — The bot evaluates all legal moves.
2. **Chance Nodes (Piece Identities)** — For each potential attack, the system calculates the expected utility by performing an exact summation over piece-rank probabilities provided by AAREN:
   $$\mathbb{E}[V] = \sum_{r=1}^{12} P(\text{target is rank } r | \mathcal{H}) \cdot \text{CombatUtility}(\text{attacker}, r)$$
3. **Utility Weighting** — Final move scores are a weighted mixture of the DQN's positional prior (Q-values) and the search-derived combat expectation. This ensures the bot avoids "hallucinating" successful attacks on squares with high probabilities of being Bombs or Marshals.

**Key constraints:**
- **Inference only** — never used during training to preserve iterations/sec.
- Uses the agent's full 79-channel AAREN state representation (`agent.get_state_representation()`).
- C51 support range reads from agent (`[-10, 10]`, 51 atoms, 0.4/atom).
- Computational cost: ~5 forward passes per decision.

**Integration:** `PolicyRefinedSearch(agent)` accepts a `RainbowAgent` instance. Used in `visual_train_dqn.py` and `dqn_bot_logic.py` for test-time evaluation.

## Episode-Level Replay Architecture

The dual-buffer replay system addresses four limitations of flat PER for long-horizon Stratego:

1. **Temporal credit assignment** — contiguous 16-step segments allow the agent to learn from multi-move strategic maneuvers rather than isolated transitions.
2. **Episode-level prioritization** — winning and losing episodes are sampled at higher rates than draws, ensuring the agent focuses on decisive game patterns.
3. **N-step extension** — within each segment, the reward is accumulated across all 16 steps, extending the effective credit horizon beyond the standard 5-step returns.
4. **Zero-risk integration** — the `EpisodicReplayBuffer` runs alongside PER without modifying the existing training pipeline; disabling it restores the original behavior.

**Episode lifecycle:** `start_episode(lane_id)` initializes tracking when a lane resets. Each step appends `(s, a, r, s', done)` to the current episode. `end_episode(lane_id, outcome, total_reward)` finalizes and stores with metadata. During `replay()`, 25% of the batch is drawn from the episode buffer (contiguous segments), 75% from PER.

## Architectural Evolutions and Ablations

The current MARQ architecture resulted from extensive ablation studies and performance benchmarking. Several key transitions defined the evolution of the system.

- **Value-Based Evolution.** The agent transitioned from a vanilla DQN with $\epsilon$-greedy exploration to a Rainbow DQN that utilizes Noisy Networks. Additionally, the Double-DQN (DDQN) configuration was replaced with Dueling Heads to improve multi-action utility resolution.
- **Cognitive Shift.** The framework replaced explicit Probabilistic Belief States (PBS) with implicit AAREN embeddings. This change addressed the $O(N^2)$ computational complexity and significantly increased training iterations per second.
- **Memory Optimization.** LSTMs were replaced with AAREN. The attention-as-a-recurrent-network architecture provides superior gradient stability for the 100-move horizons common in Stratego and handles non-Markovian history with O(1) inference.
- **Vision Backbone.** After testing deeper configurations, the system settled on 6 ResNet blocks with 64 channels. This arrangement provides the $27 \times 27$ receptive field required for a $10 \times 10$ board without excessive parameterization.
- **Setup Efficiency.** The architecture transitioned from an agentic SetupAgent trained via distributional RL to a fixed Heuristic Setup Agent. This move reduced exploitability because random flag and bomb positioning prevents opponent memorization of setup patterns. Furthermore, it reduced setup time from approximately 2 seconds to less than 100 milliseconds. This optimization halved the total training compute requirement by eliminating the need to train a separate model for the placement phase.
- **Optimizer Tuning.** The project migrated from ADAM to ADAMW to ensure more stable weight decay behaviors in policy loss.

## Future Directions
1. **Strategic Intent:** Moving pieces with purpose towards high-value targets.
2. **Belief Interpretation:** Improving how the agent uses AAREN embeddings to bluff or deduce.
3. **Decisiveness:** Reducing repetitive moves in the mid-game.
4. **R2D2-style AAREN Burn-in:** Potential future upgrade to recompute AAREN embeddings at training time, addressing representational drift in the episode buffer.

---
> **Tip:** Check `training_config.py` for current hyperparameters. Run `check_aaren_integration.py` to verify AAREN-DQN pipeline health. Run `visual_train_dqn.py` to observe agent behavior in real-time.
