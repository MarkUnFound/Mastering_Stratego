# Stratego Modular: Project Status & Agent Guide

## Core Architecture & Tech Stack
- **Core:** Rainbow DQN (C51, Dueling, Noisy, Multi-step, PER). Total 21.7M params.
- **Vision Backbone:** 6-block ResNet (64ch) + Spatial Self-Attention (4-head, learnable 2D positional encoding). Provides 27x27 receptive field without excessive depth.
- **History Representation:** AAREN (Attention-as-a-RNN) — 3-layer, 64-dim, parallel training / O(1) inference. Implicitly infers piece identities. Trained end-to-end with DQN (gradient flow from C51 loss through AAREN) plus supervised cross-entropy on revealed pieces.
- **Replay Memory:** Dual-buffer — `PrioritizedReplayBuffer` (150K) + `EpisodicReplayBuffer` (500 episodes, 16-step contiguous segments, 25% mix ratio). Stores 15ch board-only tensors + compact AAREN history snapshots; embeddings reconstructed at replay time using current AAREN model.
- **GUI Inference:** Formal Expectamax search mitigates "hallucination" by weighting DQN Q-values with AAREN's rank probabilities.
- **AI Guidance (Updated):** Real-time "AI Coach" in `guided_pygame_test.py` featuring automated Top-3 move recommendations with Chess.com-style classifications (Best, Excellent, Good, Inaccuracy). Utilizes direct extraction of the Dueling Network's Value Head for stable, global board evaluation.
- **Serialization:** Hybrid strategy. Full-state `.tar.gz` for training pause/resume (preserves buffers & optimizers). Inference-only `.pt` for GUI and league checkout. The checkpoint loader automatically falls back to the latest `.pt` weights (prioritizing `league` files to avoid optimizer state dict mismatches) if the `.tar.gz` archive is older than the latest `.pt` checkpoints.
- **Animations:** Manim visualization scenes (ResNet & Attention, Dueling C51, Softmax Exploration) successfully completed. Reside in `animations/` or root with their own `media/` output. 
## Training Pipeline & Curriculum
- **5-Phase Curriculum:** Prevents strategy cycling and orchestrates opponent transitions. Start phase can be configured. Currently set to **Phase 4 (League Training)** directly to accelerate MARL.
- **Pure MARL (League Training):** Heuristic "cheating" opponents (`smart_heuristic` and `greedy`) removed from Phase 4 to ensure pure RL self-play against historical snapshots (`league`) and current versions (`self`). 
- **PFSP Adaptive Opponent Scheduler:** Candidate pools expand progressively. Opponents the agent masters receive minimum 5% priority weight.
- **Boltzmann Exploration (Temperature Scaling):** `argmax` action selection replaced with soft stochastic exploration.
- **Agent 2 MARL:** Agent 2 serves as an independent learning agent. AAREN history is enabled immediately in Phase 4+. 
- **Performance Optimizations:** PyTorch 2.0+ Compilation (`torch.compile`) is currently disabled on Windows due to missing native Triton support, but code infrastructure is ready for Linux/WSL.

## Reward System (PBRS)
Replaced additive dense rewards with Potential-Based Reward Shaping (PBRS) to fix the "Happy Wanderer" stagnation (high Q-value shuffling, zero flag captures).
- **Potential Function $\Phi(s)$:** Material advantage (40%), flag proximity (35%), penetration (15%), info gain (10%).
- **Anti-Stall Penalties:** 
  - Severe piece oscillation penalty (A->B->A patterns) scaling exponentially beyond 3 moves.
  - Move diversity penalty if <3 unique source/destination pairs exist in a 20-move window.
  - Terminal-only speed bonus (+0.3 for quick win) and stall penalty (-0.3 for long draw). Rank-weighted piece loss penalties restrict high-level sacrifices.
- **Phase Annealing:** Shaping multipliers fade to enforce pure objective focus: Phase 1 (1.0x), Phase 2 (0.5x), Phase 3 (0.2x), Phase 4+ (0.0x).

## AAREN Data Flow
1. **Feature Extraction:** Converters parse each opponent action into a 24-dim feature vector.
2. **Sequential Encoding:** 3-layer AAREN stack produces per-position hidden states `(input=24, hidden=64)`.
3. **Embedding Assembly:** Formats into a `(64, 10, 10)` tensor.
4. **DQN Consumption:** `(79, 10, 10)` total spatial tensor (15 board + 64 AAREN) enters the Rainbow backbone. AAREN is never bypassed — even under full observability the DQN sees AAREN embeddings, not ground-truth one-hot piece types. This enforces co-evolution of AAREN and DQN from Phase 1.
- **Dual Training Regime:** AAREN receives gradients from two sources: (1) supervised cross-entropy on `revealed_in_step` data (piece identity prediction), and (2) C51 distributional DQN loss via replay-time embedding reconstruction (end-to-end). AAREN parameters are included in the DQN optimizer for joint updates.

## Component Verification Status
- **SpatialAttention:** Passes non-identity, weight spread, positional sensitivity, grad flow, and semantic responsiveness checks.
- **AAREN Integration:** AAREN gradient norms and accuracy metrics plot successfully on progress charts. Integrates correctly into DQN. Sparse death fixed via learnable default embeddings.

## Current Results (First 1000 Episodes - Phase 1, 300 steps/episode)
| Metric | Value |
|--------|-------|
| **Flag-capture wins** | 3.0% (30 captures) |
| **Depletion losses** | 3.3% (33 depletions) |
| **Draws (timeout)** | 93.6% (936 timeouts) |
| **Avg P1 reward** | 1.74 ± 0.39 |
| **AAREN accuracy** | ~9.7% (vs 8.3% random baseline) |

## Known Struggles & Fixes
| Struggle | Status | Resolution |
| :--- | :--- | :--- |
| **Phase 4 MARL Undeclared Variable Crash** | Fixed | Removed legacy PBS batch arguments passing into `agent2.history_aggregator` in `train_dqn.py`. |
| **Sparse Death** | Fixed | Supported via learnable default embedding in AAREN. |
| **Reward Scale vs C51** | Fixed | Discretization tightened: Terminal ±1.0, support [-10,+10], atoms 51. |
| **Double-Counting Rewards** | Fixed | Pending transition double-counting averted on P2 response frames. |
| **Suicidal Forward Bias** | Fixed | Forward reward strictly gated on rank ≤ Captain. |
| **High Draw Rate (93%)** | Ongoing | Addressed via newly active PBRS diversity and oscillation penalties. |
| **Win/Loss Tracking Inaccuracy (Overwritten by Timeouts)** | Fixed | Verified `win_type` tracking (depletion vs flag) and fixed `environment.py` allowing timeouts/stalls to incorrectly overwrite valid decisive game ends across the turn limit. |
| **AAREN Accuracy Plotting** | Fixed | Extracted prediction accuracy from History Aggregator and plotted it along with embedding std deviations to track learning over time. |
| **AAREN Checkpointing & Plotting** | Fixed | Supervised `train_history` gradient norms were missing (0.0) and shared models weren't saving their `state_dict`. Fixed by assigning a standalone optimizer to the first shared instance, restoring AAREN supervised training and plot values. |
| **Phase 2 Stagnation (Distribution Shift \& Missing Presence Bug)** | Fixed | Two root causes identified and resolved: (1) `HIDDEN_PIECE` (-20) was accidentally filtered out of the enemy presence mask by the `> LAKE_SQUARE` (-13) condition, blinding the agent to hidden enemies — fixed by explicitly including `HIDDEN_PIECE`. (2) `get_state_representation` bypassed AAREN with ground-truth one-hot piece encodings under full observability, causing a distribution shift when Phase 2 switched to AAREN embeddings — removed the bypass so AAREN is never circumvented and co-evolves with the DQN backbone from Phase 1. |
| **Stale AAREN Embeddings in Replay Buffer** | Fixed (Tradeoff) | AAREN embeddings were "baked" into 79ch state tensors at storage time. Fixed by storing 15ch board-only tensors + compact AAREN snapshots. While full sequential reconstruction was aborted for performance (60s/it vs 15s/it), the **Gradient Bridge** enables end-to-end gradient flow from the DQN loss back to AAREN's `input_proj` linear layer by adding a differentiable epsilon-scaled zero-vector to the stored snapshots. This preserves the 4x speedup of pre-computed embeddings while ensuring the DQN can still influence AAREN's early feature projection. Stale value noise is mitigated by AAREN's **Dual Training Regime**, where fresh sequences are used for supervised reveal training. |
| **Missing Positional Encoding in SpatialAttention** | Fixed | `SpatialAttention` processed board positions without positional encoding, making it unable to distinguish locations (e.g., flag row vs midfield). Added learnable 2D positional encoding `(1, 100, 64)` applied before self-attention. |
| **Duplicated Code in board.py** | Fixed | `board.py` contained the entire `Board` class defined twice (128 duplicate lines). Python used only the second copy. Removed the first duplicate. |
| **PBS Heuristic Move Blending** | Fixed | `bot_logic.py` previously combined a rule-based simplified PBS heuristic (70%) with a dummy StrategoNet (30%) for move selection. Replaced this entire module with a proxy to `DQNBotLogic` (which houses `ExpectamaxSearch`), fully transitioning test-time GUI inference to the genuine trained 21.7M parameter `RainbowAgent` and its AAREN uncertainty logic. |
| **GitHub Large File Push Rejection** | Fixed | `agent1_league_episode_1000.pt` (247.25 MB) exceeded GitHub's 100 MB limit, blocking the push. Fixed by removing the file from git tracking (`git rm --cached`), adding `*.pt` to `.gitignore` (alongside existing `*.pth` rule), and amending the commit to purge the blob. `.pt` model checkpoints are now excluded from version control. |
| **MARL Anti-Stationarity & Target Smoothing** | Fixed | In pure self-play, agents adapting simultaneously caused gradients to explode and evaluation to oscillate. Addressed via (1) `MARL_REWARD_SCALE` dampening global scalar rewards, and (2) `MARL_TARGET_UPDATE_INTERVAL` slowing down target network polyak averaging to provide a stationary anchor. |
| **Multiprocessing Connection Crash on Exit** | Fixed | `train_dqn.py` often crashed with `EOFError` during arbitrary exit or interruptions due to zombie parallel environments. Added a robust `finally: parallel_env.close()` block. |
| **Missing Game Legends for Beginners** | Fixed | Integrated a "Legends & Rules" UI panel to the right of the Stratego board in `test strategoo.py`. The panel clarifies piece precedence (e.g. Miners defuse bombs, Spies beat Marshals) and color codes (e.g. green tiles for valid moves) without obstructing the move history. |
| **Legacy Import Path Errors** | Fixed | Root project reorganization caused `ModuleNotFoundError` for `dqn_bot_logic.py` and `dqn_dashboard.py`. Replicated `train_dqn.py`'s `sys.path.append` logic for `environment` and `network` directories resolving the module resolution breakages immediately. |

## Research Methodology & Alignment
- **RQ1 (Convergence & Win-Rates):** Evaluated strictly via TensorBoard reward stability and benchmark win ratios over decisive games.
- **RQ2 (Relative Superiority):** Quantified later via statistical tests against standard baseline DQNs.
- **RQ3 (AAREN Impact):** Assessed through historical piece prediction accuracy significantly topping random chance (8.3%).
- **Alignment (Holistic Nature):** Engineering the bot establishes the platform; gathering data (e.g., Elo, explicit AAREN impact) satisfies the scientific objective of modeling AI strategic deduction under uncertainty.

## Future Directions
1. **Strategic Intent:** Reinforce moving pieces with long-term purpose past tactical engagements.
2. **Belief Interpretation:** Improve how AAREN embeddings translate into explicit bluffing dynamics.
3. **Tournament / Elo Integration:** Complete a statistical matchmaking ladder to benchmark relative agent skill against historical iterations.

## Reorganized Folder Structure (New)
- `train_dqn.py`: Main training script (root).
- `environment/`: Core game logic (`board.py`, `piece.py`, `game_state.py`), environment wrappers (`environment.py`, `parallel_environment.py`), and curriculum management.
- `network/`: Rainbow DQN agent (`drqn_agent.py`), AAREN (`history_aggregator.py`), replay buffers, and training infrastructure (`training/`).
- `settings/`: Project configurations (`training_config.py`).
- `test/`: Verification scripts (`preflight_checks.py`), unit tests, and profiling tools.
- `visualizers/`: Training visualizers, dashboards, and live demos.
- `docs/`: Project documentation and status reports.
- `utils/`: Helper scripts and utility functions.
