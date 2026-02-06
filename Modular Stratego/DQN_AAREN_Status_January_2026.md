# DQN with AAREN: Status Report & Stack Overview
**Date:** January 30, 2026
**Subject:** Current Progress, Architecture, and Challenges

The current architecture implements **Rainbow DQN** augmented with **AAREN (Attention-based Action-Relational Episodic Network)**. This stack replaces explicit Probabilistic Belief States (PBS) with *implicit* latent embeddings generated from historical action sequences.

**Current Status:** The system is in the **diagnostics and verification phase**. Empirical evidence indicates a convergence failure, with win rates remaining stochastic (~0-1%) against random baselines. Technical focus is prioritized on resolving gradient vanishing/explosion ("dead gradients") within the end-to-end recurrent-attention stack.

---

## 2. The Stack: Architecture Overview

### Core Agent: Rainbow DQN
The agent is a modification of the **Rainbow DQN** architecture, incorporating:
-   **Distributional RL (C51):** Learns a categorical distribution of returns (51 atoms) instead of a single scalar Q-value. The support range is set to `[-30, +30]` to accommodate boosted rewards.
-   **Noisy Networks:** Replaces $\epsilon$-greedy exploration with learnable parametric noise in the linear layers, allowing the network to self-regulate exploration.
-   **Dueling Architecture:** Separates the estimation of State Value $V(s)$ and Advantage $A(s, a)$.
-   **N-Step Returns:** Uses 3-step returns (`N_STEPS=3`) for better credit assignment.
-   **Prioritized Experience Replay (PER):** Prioritizes training on transitions with high TD-error.

### Memory & Context: AAREN (Implicit PBS)
AAREN serves as the **History Aggregator**, replacing explicit belief state computations with learned contextual embeddings.
-   **Architecture:** AAREN is formulated as **Attention as an RNN**, utilizing a parallel prefix-scan kernel for training and recurrent updates for inference.
-   **LSTM Replacement:** This architecture replaces traditional LSTM units, providing O(1) inference complexity and $O(\log N)$ parallel training paths, which facilitates modeling of long-horizon dependencies in Stratego episodes.
-   **Implicit PBS:** The Public Belief State is no longer an explicit probability distribution; instead, it is an implicit latent representation ($z \in \mathbb{R}^{64}$) concatenated to the spatial feature map.
-   **Optimization:** AAREN is trained **end-to-end** via the DQN loss. Shared gradient flow ensures that the history representation is optimized specifically for Q-value estimation.

### State Representation (79 Channels)
The ResNet input consists of:
-   **15 Channels:** Board state (Player pieces, Enemy presence, Obstacles, Empty spaces).
-   **64 Channels:** AAREN History Embeddings (Contextual information about unrevealed enemy pieces).

---

## 3. Process Flow: Environment & Rewards

### Environment Processing
1.  **Observation:** The agent receives a partial-information board.
2.  **History Aggregation:** The `HistoryAggregator` updates its internal state with the latest move/outcome.
3.  **Feature Fusion:** The 15-channel board + 64-channel history embedding are stacked.
4.  **Action Selection:**
    -   **Heuristic Filter:** Top-100 moves are pre-selected using a heuristic filter (attack priority, forward movement) to prune the action space.
    -   **Q-Network:** The stack processes the features to output Q-value distributions for these moves.
    -   **Decision:** The action with the highest expected value is chosen (Noisy Nets handle exploration).

### Reward Structure (Unified)
The reward signal is heavily "shaped" to guide the agent through the difficulty of Stratego.
-   **Terminal Rewards (Boosted):**
    -   **Flag Capture:** `+25.0` (Primary Objective)
    -   **Depletion Win:** `+10.0`
    -   **Loss:** `-15.0`
-   **Shaping Rewards:**
    -   **Flag Distance:** `+0.15` per step for progression toward the enemy back rank.
    -   **Material:** Incentives for piece captures, scaled by rank (e.g., Spy-Marshal capture bonus).
    -   **Epistemic:** `+0.01 - 0.02` for information revelation (hidden piece types).
    -   **Curiosity:** Intrinsic reward based on state novelty to mitigate local optima in exploration.

---

## 4. Current Results

### Quantitative Metrics
-   **Win Rate:** ~0-1% (Vs Random/Heuristic). The agent is effectively not winning.
-   **Training Speed:** ~34k steps (slow).
-   **Bugs:** "Dead Gradients" detected in diagnostics, meaning parts of the network (or the entire network) are receiving near-zero updates, preventing learning.

### Qualitative Observations
-   **Stagnation:** The agent appears stuck in a mode of random or ineffective play.
-   **Learning Failure:** Despite the rich reward signal (flag distance, etc.), the agent has not yet correlated actions with the "Flag Capture" objective.

---

## 5. Challenges & Roadblocks

### 1. Gradient Flow & Dead Neurons
**Issue:** Diagnostics reveal that gradient norms often drop to near zero or vanish.
**Hypothesis:** The connection between the AAREN module (history) and the DQN head might be a bottleneck, or the "Noisy Net" noise parameters might be collapsing early.
**Action:** Currently implementing Phase 1-3 Diagnostics to trace signal propagation.

### 2. End-to-End Training Difficulty
**Issue:** Training AAREN implicitly (via Q-learning signal only) is difficult. The Reward signal must propagate all the way back through the DQN, the ResNet, and into the History Aggregator to teach AAREN *how* to represent memory.
**Challenge:** Without an auxiliary loss (e.g., "predict the enemy piece type"), AAREN might not be learning meaningful representations of the hidden state.

### 3. Sparse Positive Outcomes
**Issue:** Even with "Flag Distance" rewards, actually capturing the Flag is a rare event in early training.
**Challenge:** The agent struggles to bridge the gap between "moving forward" (which it gets paid for) and "winning" (which is rare).

### 4. Computational Bottlenecks
**Issue:** The heavy stack (Attention + ResNet + Distributional Heads) is computationally expensive.
**Mitigation:**
-   **Mixed Precision (AMP):** Enabled.
-   **Heuristic Action Filter:** Reduces search space from ~400 to 100 moves.
-   **PyTorch Compile:** Investigated but seemingly disabled (`False`) in current config.
