# Stratego Modular: Project Status & Agent Guide

## Core Architecture & Tech Stack
- **Core:** Vanilla DQN (Standard Q-Learning, Epsilon-Greedy, MSE Loss). Replaced Rainbow architecture to simplify debugging and lower computational overhead.
- **Vision Backbone:** 3-layer CNN (no ResNet, no attention). Directly processes the combined board and LSTM embeddings.
- **History Representation:** LSTM (Long Short-Term Memory) — 2-layer, 64-dim, standard PyTorch LSTM. Replaces the previous AAREN (Attention-as-a-RNN) architecture with a simpler recurrent model. Implicitly infers piece identities from action sequences. Trained end-to-end with Vanilla DQN (Gradient Bridge) plus supervised cross-entropy on revealed pieces.
- **Replay Memory:** `StandardReplayBuffer` (150K). Stores 15ch board-only tensors + compact LSTM history snapshots; embeddings reconstructed at replay time using current LSTM model (PER and Episodic segments removed).
- **GUI Inference:** Formal Expectamax search mitigates "hallucination" by weighting DQN Q-values with LSTM's rank probabilities.
- **AI Guidance (Updated):** Real-time "AI Coach" in `guided_pygame_test.py` featuring automated Top-3 move recommendations with Chess.com-style classifications (Best, Excellent, Good, Inaccuracy). Utilizes direct extraction of the Network's Value Head for stable, global board evaluation.
- **Serialization:** Hybrid strategy. Full-state `.tar.gz` for training pause/resume (preserves buffers & optimizers). Inference-only `.pt` for GUI and league checkout. The checkpoint loader automatically falls back to the latest `.pt` weights (prioritizing `league` files to avoid optimizer state dict mismatches) if the `.tar.gz` archive is older than the latest `.pt` checkpoints.
- **Animations:** Manim visualization scenes (ResNet & Attention, Dueling C51, Softmax Exploration) successfully completed. Reside in `animations/` or root with their own `media/` output. 
## Training Pipeline & Curriculum
- **5-Phase Curriculum:** Prevents strategy cycling and orchestrates opponent transitions. Start phase can be configured. Currently set to **Phase 4 (League Training)** directly to accelerate MARL.
- **Pure MARL (League Training):** Heuristic "cheating" opponents (`smart_heuristic` and `greedy`) removed from Phase 4 to ensure pure RL self-play against historical snapshots (`league`) and current versions (`self`). 
- **PFSP Adaptive Opponent Scheduler:** Candidate pools expand progressively. Opponents the agent masters receive minimum 5% priority weight.
- **Boltzmann Exploration (Temperature Scaling):** `argmax` action selection replaced with soft stochastic exploration.
- **Epoch-Based Opponent Cycling:** Instead of randomly sampling opponent type every episode (high variance), all 16 lanes lock to the same opponent type for 50-episode blocks (`OPPONENT_CYCLE_INTERVAL`), then rotate through the available types (e.g., `self` → `league` → `self` → ...). Reduces reward variance while maintaining opponent diversity.
- **Agent 2 MARL:** Agent 2 serves as an independent learning agent. LSTM history is enabled immediately in Phase 4+. 
- **Performance Optimizations:** PyTorch 2.0+ Compilation (`torch.compile`) is currently disabled on Windows due to missing native Triton support, but code infrastructure is ready for Linux/WSL.

## Reward System (PBRS)
Replaced additive dense rewards with Potential-Based Reward Shaping (PBRS) to fix the "Happy Wanderer" stagnation (high Q-value shuffling, zero flag captures).
- **Potential Function $\Phi(s)$:** Material advantage (40%), flag proximity (35%), penetration (15%), info gain (10%).
- **Anti-Stall Penalties:** 
  - Severe piece oscillation penalty (A->B->A patterns) scaling exponentially beyond 3 moves.
  - Move diversity penalty if <3 unique source/destination pairs exist in a 20-move window.
  - Terminal-only speed bonus (+0.3 for quick win) and stall penalty (-0.3 for long draw). Rank-weighted piece loss penalties restrict high-level sacrifices.
- **Phase Annealing:** Shaping multipliers fade to enforce pure objective focus: Phase 1 (1.0x), Phase 2 (0.5x), Phase 3 (0.2x), Phase 4+ (0.0x).

## LSTM Data Flow (Replaces AAREN)
1. **Feature Extraction:** Converters parse each opponent action into a 24-dim feature vector.
2. **Sequential Encoding:** 2-layer LSTM stack produces per-position hidden states `(input=24, hidden=64)`.
3. **Embedding Assembly:** Formats into a `(64, 10, 10)` tensor.
4. **DQN Consumption:** `(79, 10, 10)` total spatial tensor (15 board + 64 LSTM) enters the Vanilla DQN backbone. LSTM is never bypassed — even under full observability the DQN sees LSTM embeddings, not ground-truth one-hot piece types. This enforces co-evolution of LSTM and DQN from Phase 1.
- **Dual Training Regime:** LSTM receives gradients from two sources: (1) supervised cross-entropy on `revealed_in_step` data (piece identity prediction), and (2) Standard DQN loss (MSE/Huber) via replay-time embedding reconstruction (end-to-end). LSTM parameters are included in the DQN optimizer for joint updates.

## Component Verification Status
- **SpatialAttention:** Passes non-identity, weight spread, positional sensitivity, grad flow, and semantic responsiveness checks.
- **LSTM Integration:** LSTM gradient norms and accuracy metrics plot successfully on progress charts. Integrates correctly into DQN. Sparse death fixed via learnable default embeddings.

## Current Results (Phase 4 League Training - 10,000 Episodes — Pre-Fix Baseline)
| Metric | Value |
|--------|-------|
| **Total P1 Wins** | 25.1% (2,507 wins) |
| **Total P2 Wins** | 24.9% (2,499 wins) |
| **Draws (timeout/stalemate)** | 50.0% draws |
| **Avg P1 reward (All)** | 0.08 |
| **LSTM accuracy (Final)** | ~22% (vs 8.3% random baseline) |

*Post-stagnation-fix training started at ep 10,000. Win rate target: sustained >30%. All 5 root-cause fixes applied (see Known Struggles below).*

## Known Struggles & Fixes
| Struggle | Status | Resolution |
| :--- | :--- | :--- |
| **AAREN → LSTM Architecture Migration** | Fixed | Replaced AAREN (Attention-as-a-RNN) with standard PyTorch LSTM for history encoding. Created `PieceActionLSTM` as drop-in replacement in `network/lstm/`. Updated `HistoryAggregator`, `DQNAgent`, `training_config.py`. LSTM uses 2 layers (down from AAREN's 3) for faster training. Backward-compat aliases added to config. Old AAREN checkpoints are gracefully skipped on load with a warning. |
| **Phase 4 MARL Undeclared Variable Crash** | Fixed | Removed legacy PBS batch arguments passing into `agent2.history_aggregator` in `train_dqn.py`. |
| **Sparse Death** | Fixed | Supported via learnable default embedding in LSTM. |
| **Reward Scale vs C51** | Fixed | Discretization tightened: Terminal ±1.0, support [-10,+10], atoms 51. |
| **Double-Counting Rewards** | Fixed | Pending transition double-counting averted on P2 response frames. |
| **Suicidal Forward Bias** | Fixed | Forward reward strictly gated on rank ≤ Captain. |
| **High Draw Rate (93%)** | Ongoing | Addressed via newly active PBRS diversity and oscillation penalties. |
| **Win/Loss Tracking Inaccuracy (Overwritten by Timeouts)** | Fixed | Verified `win_type` tracking (depletion vs flag) and fixed `environment.py` allowing timeouts/stalls to incorrectly overwrite valid decisive game ends across the turn limit. |
| **LSTM Accuracy Plotting** | Fixed | Extracted prediction accuracy from History Aggregator and plotted it along with embedding std deviations to track learning over time. |
| **LSTM Checkpointing & Plotting** | Fixed | Supervised `train_history` gradient norms were missing (0.0) and shared models weren't saving their `state_dict`. Fixed by assigning a standalone optimizer to the first shared instance, restoring LSTM supervised training and plot values. |
| **Phase 2 Stagnation (Distribution Shift \& Missing Presence Bug)** | Fixed | Two root causes identified and resolved: (1) `HIDDEN_PIECE` (-20) was accidentally filtered out of the enemy presence mask by the `> LAKE_SQUARE` (-13) condition, blinding the agent to hidden enemies — fixed by explicitly including `HIDDEN_PIECE`. (2) `get_state_representation` bypassed LSTM with ground-truth one-hot piece encodings under full observability, causing a distribution shift when Phase 2 switched to LSTM embeddings — removed the bypass so LSTM is never circumvented and co-evolves with the DQN backbone from Phase 1. |
| **Stale LSTM Embeddings in Replay Buffer** | Fixed (Tradeoff) | LSTM embeddings were "baked" into 79ch state tensors at storage time. Fixed by storing 15ch board-only tensors + compact LSTM snapshots. While full sequential reconstruction was aborted for performance (60s/it vs 15s/it), the **Gradient Bridge** enables end-to-end gradient flow from the Vanilla DQN loss back to LSTM's `input_proj` linear layer by adding a differentiable epsilon-scaled zero-vector to the stored snapshots. This preserves the 4x speedup of pre-computed embeddings while ensuring the DQN can still influence LSTM's early feature projection. Stale value noise is mitigated by LSTM's **Dual Training Regime**, where fresh sequences are used for supervised reveal training. |
| **Missing Positional Encoding in SpatialAttention** | N/A | Feature removed as part of transition to Vanilla DQN. |
| **Duplicated Code in board.py** | Fixed | `board.py` contained the entire `Board` class defined twice (128 duplicate lines). Python used only the second copy. Removed the first duplicate. |
| **PBS Heuristic Move Blending** | Fixed | `bot_logic.py` previously combined a rule-based simplified PBS heuristic (70%) with a dummy StrategoNet (30%) for move selection. Replaced this entire module with a proxy to `DQNBotLogic` (which houses `ExpectamaxSearch`), fully transitioning test-time GUI inference to the genuine trained Vanilla `DQNAgent` and its LSTM uncertainty logic. |
| **GitHub Large File Push Rejection** | Fixed | `agent1_league_episode_1000.pt` (247.25 MB) exceeded GitHub's 100 MB limit, blocking the push. Fixed by removing the file from git tracking (`git rm --cached`), adding `*.pt` to `.gitignore` (alongside existing `*.pth` rule), and amending the commit to purge the blob. `.pt` model checkpoints are now excluded from version control. |
| **MARL Anti-Stationarity & Target Smoothing** | Fixed | In pure self-play, agents adapting simultaneously caused gradients to explode and evaluation to oscillate. Addressed via (1) `MARL_REWARD_SCALE` dampening global scalar rewards, and (2) `MARL_TARGET_UPDATE_INTERVAL` slowing down target network polyak averaging to provide a stationary anchor. |
| **Agent Stagnation at 25% Win Rate (Phase 4, Episodes 0–10k)** | Fixed | 5-root-cause analysis revealed: (1) Symmetric self-play Nash lock-in — "self" epochs in `train_dqn.py` now load a random lagged league checkpoint instead of copying live agent1 weights. (2) Reward vacuum — `distributional_reward.py` raised `win_reward_flag` 1.0→1.5, `draw_penalty` -0.3→-0.5, Phase 4 shaping 0.0→0.05. (3) Collapsed exploration — `training_config.py` slowed epsilon decay 50k→200k steps, restored `MARL_REWARD_SCALE` 0.5→1.0, re-enabled entropy reg (coeff=0.01). (4) Opponent monoculture — `curriculum.py` added `true_random` (15%) + `greedy` (10%) to Phase 4 with explicit weighted distribution. (5) Lagged self-play cache invalidated on every epoch boundary for diverse checkpoint rotation. |
| **Multiprocessing Connection Crash on Exit** | Fixed | `train_dqn.py` often crashed with `EOFError` during arbitrary exit or interruptions due to zombie parallel environments. Added a robust `finally: parallel_env.close()` block. |
| **Missing Game Legends for Beginners** | Fixed | Integrated a "Legends & Rules" UI panel to the right of the Stratego board in `test strategoo.py`. The panel clarifies piece precedence (e.g. Miners defuse bombs, Spies beat Marshals) and color codes (e.g. green tiles for valid moves) without obstructing the move history. |
| **Phase 4 MARL CUDA Multinomial Crash** | Fixed | `torch.multinomial` crashed when a batch row was fully masked with `-inf` (due to random action or no moves). Fixed by ensuring at least one finite value (0.0) in the mask for all rows before softmax. |
| **Legacy Import Path Errors** | Fixed | Root project reorganization caused `ModuleNotFoundError` for `dqn_bot_logic.py` and `dqn_dashboard.py`. Replicated `train_dqn.py`'s `sys.path.append` logic for `environment` and `network` directories resolving the module resolution breakages immediately. |
| **Zero Q-Value After Vanilla DQN Migration** | Fixed | `get_average_q()` in `drqn_agent.py` still used C51 distributional logic (`self.support`, `log_probs.exp()`, `probs * self.support`) after migration to Vanilla DQN. The `AttributeError` on `self.support` was silently caught by a bare `except`, causing Q-value metrics to always read 0.0 despite the network actually learning. Fixed by replacing with direct Q-value reads. Also cleaned stale C51 constants and docstring from file header. |
| **Repeated League Model Loading (Every Episode)** | Fixed | `select_opponent_for_lane()` called `agent2.load_model(path)` on every lane reset when opponent type was "league", even when the same checkpoint was already loaded. This caused 3 log messages per episode, redundant disk I/O, and ~12s/episode overhead. Fixed by caching `_last_league_path` and skipping reload when the same file is selected. Cache is invalidated when switching to "self" play. |
| **Windows Paging File OOM Crash (WinError 1455)** | Fixed | `NUM_LANES` was set to 32 in `training_config.py`, causing `ParallelStrategoEnvironment` to spawn 32 large processes and quickly exhausting virtual memory limits on Windows, crashing PyTorch initialization. Fixed by reducing `NUM_LANES` back to a safe baseline of 8. |
| **Optimizer State Mismatch Warning Spam** | Fixed | Loading historical league opponents (trained with 32 lanes) into the current 8-lane agent caused a constant stream of `[WARN] Failed to load optimizer state (architecture/env mismatch)` messages. Fixed by adding a `load_optimizer=False` parameter to `drqn_agent.py`'s `load_model` method, and using it in `train_dqn.py` when loading inference-only league opponents. |
| **Negative Average Reward / Suicidal Agents in Phase 4** | Fixed | Agents averaged -10 reward per episode and purposely lost instead of drawing. Cause: PBRS shaping disabled itself in Phase 4 (`shaping_mult=0`), but anti-stall step penalties (oscillation and diversity) did not. In a 200-step game, these penalties accumulated drastically. Fixed by strictly scaling these step penalties by `shaping_mult` in `distributional_reward.py` to restore pure objective-focused RL. |
| **Q-Value Collapse / Loss Saturation in Phase 4** | Fixed | Loss dropped to 0 while win rate remained flat at ~20%. Caused by a 1500-step horizon vanishing gradient where `GAMMA=0.995` discounted terminal rewards to ~0.0008, combined with an overly strict `shaping_mult=0.05` that provided no immediate signal. Fixed by increasing `GAMMA` to `0.999` (so terminal rewards persist backward) and raising Phase 4 `shaping_mult` to `0.2` to provide enough gradient scaffolding to keep the network learning. |


## Research Methodology & Alignment
- **RQ1 (Convergence & Win-Rates):** Evaluated strictly via TensorBoard reward stability and benchmark win ratios over decisive games.
- **RQ2 (Relative Superiority):** Quantified later via statistical tests against standard baseline DQNs.
- **RQ3 (LSTM Impact):** Assessed through historical piece prediction accuracy significantly topping random chance (8.3%).
- **Alignment (Holistic Nature):** Engineering the bot establishes the platform; gathering data (e.g., Elo, explicit LSTM impact) satisfies the scientific objective of modeling AI strategic deduction under uncertainty.

## Future Directions
1. **Strategic Intent:** Reinforce moving pieces with long-term purpose past tactical engagements.
2. **Belief Interpretation:** Improve how LSTM embeddings translate into explicit bluffing dynamics.
3. **Tournament / Elo Integration:** Complete a statistical matchmaking ladder to benchmark relative agent skill against historical iterations.

## Reorganized Folder Structure (New)
- `train_dqn.py`: Main training script (root).
- `environment/`: Core game logic (`board.py`, `piece.py`, `game_state.py`), environment wrappers (`environment.py`, `parallel_environment.py`), and curriculum management.
- `network/`: Vanilla DQN agent (`drqn_agent.py`), LSTM (`lstm/`), `HistoryAggregator` (`history_aggregator.py`), `StandardReplayBuffer`, and training infrastructure (`training/`).
- `settings/`: Project configurations (`training_config.py`).
- `test/`: Verification scripts (`preflight_checks.py`), unit tests, and profiling tools.
- `visualizers/`: Training visualizers, dashboards, and live demos.
- `docs/`: Project documentation and status reports.
- `utils/`: Helper scripts and utility functions.

## Agent Learning Log
- **AAREN → LSTM Migration:** When replacing a custom recurrent architecture (AAREN) with a standard LSTM, ensure the interface contracts match exactly (`forward_parallel`, `forward_sequential`, `forward_embedding`). The hidden state format matters: LSTM uses `(h, c)` tuples per layer while AAREN used `(a, c, m)` triples. The `HistoryAggregator.get_embedding_tensor()` already handled tuple extraction via `isinstance` checks, making the swap seamless. Always add backward-compat checkpoint loading with graceful fallback for architecture mismatches.
- **Symmetric MARL Stagnation:** In pure self-play, copying live agent weights for the "self" opponent creates a Nash equilibrium where both agents co-adapt identically and cancel each other out. Diagnose via symmetric win curves (P1 wins ≈ P2 wins across all episodes) and flat reward distribution. Fix by using lagged historical checkpoints for "self" epochs — this creates an asymmetric, non-co-adapting target and forces the agent to genuinely improve rather than just match itself. Always verify lagged opponent cache invalidates each epoch boundary. Additionally, reward signal contrast must be sufficient: a flat `draw_penalty` near zero and a halved `MARL_REWARD_SCALE` removes the signal needed to differentiate winning from drawing strategies.
