# Stratego Modular: Project Status & Agent Guide

### Recent Progress & Checkpoints
- **Curriculum Rapid Transition Bug (Feb 22, 2026):**
    - **Issue:** The curriculum incorrectly transitioned through Phase 1 and Phase 2 in under 500 episodes. The transition logic checked for a minimum number of *games played* (which included a massive number of draws) while calculating a 100% win rate from a tiny sample of actual decisive wins (e.g., 2 wins, 0 losses, 53 draws).
    - **Resolution:** Modified `check_phase_transition` in `curriculum.py` to evaluate thresholds based strictly on actual wins (`wins_vs_random >= 30` and `wins_vs_heuristic >= 30` for Phase 1; `total_wins >= 60` for Phase 2), preventing premature progression with statistically insignificant win samples. Added dynamic thresholds imported from `training_config.py`.
- **Reward Annealing & Boltzmann Exploration (Feb 21, 2026):**
    - **Reward Annealing:** Integrated curriculum-based shaping weight scaling. Dense shaping rewards (material, epistemic, positional) fade out gracefully (Phase 3: 0.5x, Phase 4+: 0.0x), enabling the agent to prioritize strict terminal Win/Loss rewards in later phases.
    - **Boltzmann Exploration (Temperature Scaling):** Replaced hard `argmax` action selection with soft temperature-scaled exploration during self-play and evaluation for opponents. This minimizes deterministic loops and produces a more unpredictable curriculum roster.
- **Curriculum Progression Fixes (Feb 21, 2026):**
    - **Phase 1 Early Transition Bug:** Fixed an issue where the agent could skip Phase 1 without facing the heuristic opponent if its initial games against random were successful enough. The transition now strictly requires competence against *both* `random` (70%) and `heuristic` (50%).
    - **Phase 3 Self-Play Lock:** Corrected the mathematical logic for the Phase 3 transition. Previously, the system attempted to calculate the variance of binary (win/loss) outcomes against a hardcoded mean of 0.5, which always evaluated to 0.25 and stalled the curriculum. The phase now relies on a consistent rolling win-rate metric (≥55%) against the agent's past self.
    - **Unseen Opponent Prioritization:** Modified the adaptive opponent scheduler to allocate maximum priority (1.0 weight) to uncharted opponents instead of an assumed equal priority (0.50). This guarantees rapid exploration against new opponent types.
    - **Phase 4 Roster Alignment:** Updated the League Training opponent distribution configurations to accurately map to the opponents effectively supported by the training loop (`["league", "smart_heuristic", "greedy", "self"]`).
- **PBS Field Removal & PFSP Curriculum Adaptive Weighting (Feb 21, 2026):**
    - **Dead PBS Fields Removed:** `pbs_accuracy_sum`, `pbs_accuracy_count`, and `avg_pbs_accuracy` are fully excised from `PhaseMetrics` in `curriculum.py`. These were leftover after the PBS→AAREN migration and accumulated silently without effect. `from_dict()` has a backward-compat `.pop()` guard so old `.json` saves still load cleanly. `curriculum_state.json` has been regenerated without these fields.
    - **Phase 2 Transition Fixed:** The Memory Gap phase transition previously required `pbs_accuracy >= 0.70`, which could never be satisfied. It now relies purely on `overall_win_rate >= 0.55` over at least 100 decisive games.
    - **PFSP Adaptive Scheduler Introduced:** `get_opponent_distribution()` now uses a `_pfsp_weights()` scheduler inspired by the Prioritized Fictitious Self-Play (PFSP) scheme from AlphaStar/OpenAI Five. Per-opponent win/loss tracking (`opponent_stats` dict) is recorded via `record_opponent_result()` in `PhaseMetrics`. The scheduler computes `weight = (1 - win_rate)` per opponent, floored at 5%, so opponents the agent has **mastered (≥ 90% win rate) receive a minimum 5% share** while opponents the agent **struggles against receive proportionally more matchup time**. A `min_games=30` guard prevents premature reweighting; new opponents default to a 50% assumed win rate (equal priority) until enough data accumulates.
    - **Candidate Pool Gating (Phase 1):** The Phase 1 PFSP pool expands progressively: only `random` until the aggregate rate vs. random exceeds 70%, then `random + heuristic`, then the full `random + heuristic + smart_heuristic` pool once competence against heuristic reaches 30%. This prevents the agent from being exposed to opponents it cannot yet meaningfully learn from.
- **"Happy Wanderer" Stagnation Fix v3 — PBRS Architecture (Feb 22, 2026):**
    - **Issue:** Around episode 4000, training exhibited paradoxically lowering loss and increasing Q-values alongside a stagnating win rate and massive draw rate (93.6%). The agent fell into a local optimum where it learned to endlessly shuffle pieces to farm small, dense shaping rewards (material, positional) rather than capturing the flag.
    - **Root Cause**: Additive shaping rewards were time-proportional — every step gave small positive signals. Over 200 steps these dwarfed the sparse +1.0 win reward. No mechanism existed to detect or penalize piece oscillation (A->B->A patterns), and the per-step penalty (-0.005 x 200 = -1.0) made slow wins worthless.
    - **Resolution — Potential-Based Reward Shaping (PBRS):** Replaced ALL additive shaping (material, positional, epistemic, curiosity) with a single potential function $\Phi(s)$ and the shaping reward $F = \gamma\Phi(s') - \Phi(s)$. This guarantees: (a) zero reward for stationary behavior ($s' \approx s \Rightarrow F \approx 0$), (b) positive reward for progress (captures, territory), (c) optimal policy invariance (Ng et al., 1999; extended by Potential-Based Intrinsic Motivation, AAMAS 2024). $\Phi(s)$ combines material advantage (40%), offensive proximity to enemy flag (35%), territorial penetration (15%), and information gain (10%).
    - **Resolution — Piece Oscillation Penalty:** Added per-piece position history tracking. When a piece returns to a previously visited position (A->B->A pattern), escalating penalties apply: -0.02 per oscillation beyond threshold 2. Moves 1-3 of shuffling are tolerated (could be tactical); moves 4+ are increasingly punished. This directly targets the shuffle mechanism.
    - **Resolution — Terminal Game-Length Modifier:** Replaced the per-step penalty with a terminal-only modifier: (a) quick wins receive a speed bonus of up to +0.3, (b) long draws receive an additional penalty of up to -0.3. This preserves the incentive to finish games quickly without distorting intermediate Q-values. Slow wins are still positive.
    - **Resolution — Move Diversity Penalty:** Added a 20-move lookback window. If fewer than 3 unique (source, destination) pairs appear in the window, a -0.05 penalty applies. This catches "3 pieces taking turns shuffling" patterns that the old stalemate check (`num_valid_moves < 5`) missed.
    - **Legacy stalemate penalty retained** for near-immobility (`num_valid_moves < 5`); attack bonus (+0.02) preserved outside PBRS to always encourage combat engagement.
    - **Phase annealing unchanged**: Phase 1 (1.0x), Phase 2 (0.5x), Phase 3 (0.2x), Phase 4+ (0.0x) — applied to PBRS multiplier.
    - **All 14 reward system tests pass** (`test_reward_system.py`).
- **Current Main Checkpoint**: `agent1_rainbow_episode_*.pt` (Periodic), `agent1_rainbow_episode_*.tar.gz` (Stop-Save)
- **Hybrid Serialization Strategy**:
    - **Periodic Checkpoints (`.pt`)**: Weights-only saves for league inference and performance monitoring.
    - **Stop-Save Archives (`.tar.gz`)**: Full state (optimizers, buffers, AAREN history) saved upon manual interruption (`Ctrl+C`).
    - **Seamless Resumption**: The checkpointer prioritizes archive loads to ensure zero data loss and mathematical continuity (preserving AdamW momentum and experience replay distributions).
    - **AAREN Training Buffer**: Verified that unsupervised reveal-data buffers are preserved in the full-state archives.
- **Episode 11000 Deployment (Feb 19, 2026):** Transitioned the GUI production bot from Episode 1000 to Episode 11000. This model features significantly more stable Q-values and better piece-value recognition.
- **Full-State Serialization for Continuous Training (Feb 19, 2026):** Migrated model saving from `.pth` to a dual-format system:
    - **`.tar.gz` (Archive):** Saves the complete agent state, including network weights, optimizer state, AAREN training buffer, and all replay buffers (PER + Episodic). This ensures that resuming training is mathematically equivalent to uninterrupted execution, preventing the "forgetting" or distribution shift common after checkpoint restarts.
    - **`.pt` (Inference):** Saves optimized, weights-only models for the league and GUI. These are stripped of training overhead (optimizer, buffers) to enable fast loading and secure `weights_only=True` unpickling.
    - **Enforcement:** The `Checkpointer` and `LeagueManager` now strictly enforce these extensions, and `train_dqn.py` automatically handles archive extraction during session resumption.
    - **Global Step Preservation:** Fixed a bug where `global_step` metrics were resetting to zero upon resumption after a KeyboardInterrupt. The `metrics_tracker` is now explicitly updated with the current `global_step` before any periodic save or manual interruption full-state save, ensuring perfect step continuity.
    - **Curriculum & MARL Activation (Feb 20, 2026):**
    - **Curriculum Progress Fix:** Refactored `curriculum.py` to calculate win rates based on *decisive games* (wins vs. losses) rather than total games. This prevents the high volume of early-stage draws from artificially stalling the curriculum advancement.
    - **True MARL for Agent 2:** Transitioned Agent 2 from a static opponent to an independent learning agent.
    - **MARL Phase Gating:** Implemented a learning gate where Agent 2's weight updates (`replay` and target updates) are only active from Phase 4 (League Training) onwards. Agent 2 continues to collect experience during Phase 1-3 to ensure a robust buffer is ready for the MARL transition.
    - **AAREN Reveal Integration:** Agent 2's history aggregator now correctly processes piece-reveal data for parallel supervised learning.
- **Robust Training Display (Feb 20, 2026):** Optimized `train_dqn.py` progress bar to include fixed-width padding for all metrics (`R1`, `W1`, `W2`, `Phase`, `Step`). This prevents terminal flickering and ensures that late-positioned metrics like Player 2 wins and global step counts are not truncated by terminal wrapping or neighboring output. Enforced `tqdm.write` for all asynchronous logs to maintain UI stability. Applied `set_postfix_str` specifically to enforce trailing padding spaces, eliminating leftover carriage return artifacts.
- **Atomic Checkpoints:** All checkpoints (`.pt` and `.tar.gz`) are written to `.tmp` files first, then immediately renamed to their target filename. This cleanly solves issues with corrupted checkpoint files stemming from processes suddenly terminating during execution.
- **Improved AAREN Metrics Plotting:** Restored missing metrics (gradient norm and gradient ratio) in `aaren_progress_episode_*.png`. Because `PieceActionAaren` inside the `HistoryAggregator` is trained primarily via supervised learning loops, its gradients are completely distinct from the end-to-end DQN `backward()` pass. Tracking and logging the `clip_grad_norm_` locally within `train_history()` properly exposes these metrics to the visualizer to show if the AAREN is actively learning. 
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
- **Architectural Diagrams (Feb 21, 2026):**
    - Updated `marq_framework_arch.drawio.xml` to reflect the removal of data augmentation symmetry operations since it was found to cause issues and was disabled. 
    - Created `marq_curriculum_league.drawio.xml` to visually explain the MARQ League Training and Curriculum Pipeline, showing phase progressions, PFSP sampling, the evaluation loop, and hybrid checkpointing.
    - Created `marq_data_flow.drawio.xml` to diagram the 5-stage feature extraction, sequential encoding, and tensor assembly pipeline bridging AAREN inference into the Rainbow DQN.
    - Updated the theoretical background in Latex to include the Temperature Scaling (Boltzmann Options) stochastic action selection logic. This provides the agent with an exploration mechanism to prevent it from collapsing into easily exploitable loop patterns.

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


## Research Methodology: Answering the Questions

To convert the current technical progress into a formal thesis, the Research Questions (RQs) should be answered using the following data streams:

*   **RQ1 (Convergence & Win-Rates)**: 
    *   **Data Source**: `metrics_tracker` logs and TensorBoard curves.
    *   **Answer Strategy**: Identify the episode where the reward stabilizes (convergence) and calculate the final win/loss ratio against the current baseline.
*   **RQ2 (Relative Superiority)**:
    *   **Data Source**: Head-to-head match logs from the League.
    *   **Answer Strategy**: Run 100-500 games between the MARQ agent and a "Standard DQN" baseline. A t-test or chi-square test on the win rates will provide the "statistically significant" answer required by the RQ.
*   **RQ3 (AAREN Impact)**:
    *   **Data Source**: `check_aaren_integration.py` results and `history_aggregator` accuracy metrics.
    *   **Answer Strategy**: Compare historical piece prediction accuracy. If AAREN achieves ~15-20% accuracy (well above the ~8.3% random chance), it proves the framework is successfully inferring hidden piece configurations without needing explicit belief states.

## Objective Alignment: "Does this affect the scope?"

The implementation of Elo systems or advanced metrics **does not change your objectives**; it validates them.

*   **Objective 3 ("Build & Implement...")**: Achieved by the code itself.
*   **Objective 4 ("Train & Optimize...")**: Can *only* be proven if you have a measurement like Elo or win-rates. Optimization is a comparative process; adding these metrics is the only way to confirm you have satisfied the objective.
*   **Verdict**: The research is **holistic** because you aren't just building a bot (Engineering), you are measuring its cognitive performance (Science).

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
| **Advantage Filtering Slowdown** | Fixed | Disabled for early phases (4× oversampled batch + 2 extra forward passes = 5× slowdown). Re-enable at Phase 3+. |

## Tech Stack
- **Core:** Rainbow DQN (C51, Dueling, Noisy, Multi-step, PER).
- **History:** AAREN (Attention-as-a-Recurrent-Neural-Network) — 3-layer, 64-dim, parallel training / O(1) inference.
- **Vision:** 6-block ResNet (64ch) + Spatial Self-Attention (4-head). Total 21.7M params.
- **Replay:** Dual-buffer — `PrioritizedReplayBuffer` (150K flat) + `EpisodicReplayBuffer` (500 episodes, deque-backed O(1) eviction, 16-step segments, 25% mix ratio).
- **Inference Optimization:** `RainbowAgent` supports an `inference_only` mode which aggressively culls training overhead (disabling AdamW, PER, EpisodicBuffer, and N-Step caching components). In modular self-play, Agent 2 is initialized with full buffers to support the transition to active MARL learning in Phase 4+. Prior to Phase 4, Agent 2 operates in a "quasi-inference" state where it collects experiences but does not perform model updates.
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


## Research Alignment & Holistic Evaluation (Feb 20, 2026)

The current research trajectory has been evaluated against the core objectives defined in `introduction.tex`:

1.  **Holistic Framework**: The MARQ framework is deemed **holistic** as it addresses the entire decision-making pipeline in imperfect information games:
    *   **Perception & Memory**: AAREN provides a non-Markovian history embedding, replacing explicit belief states with implicit latent representations.
    *   **Decision Foundation**: Rainbow DQN integrates six proven enhancements (C51, Noisy Nets, etc.) to handle value distribution and exploration.
    *   **Strategic Depth**: Test-time Expectamax/UE Search mitigates the "short-sightedness" of model-free RL.
    *   **Training Robustness**: Curriculum and League-based self-play prevent strategy cycling and ensure generalization.
2.  **Objective Connectivity**: The implementation phase is directly synchronized with the research goals. The transition from Phase 1 (exploration) to Phase 4 (MARL) mirrors the academic intent to study emergent strategic reasoning.
3.  **Performance Metrics (Elo vs. Reward)**: In the context of the League Manager, an **Elo system** is identified as a critical future metric. While episode rewards and AAREN accuracy quantify "learning," Elo quantifies "skill" relative to the population, which is essential for determining if MARQ's learning translates into competitive superiority.

---
> **Tip:** Check `training_config.py` for current hyperparameters. Run `check_aaren_integration.py` to verify AAREN-DQN pipeline health. Run `visual_train_dqn.py` to observe agent behavior in real-time.
