<div align="center">

# MARQ: Mastering Stratego with Multi-Agent Rainbow Deep Q-Networks & AAREN

**An Imperfect-Information Game AI Framework Combining Distributional Rainbow DQN, Attention as a Recurrent Neural Network (AAREN), Potential-Based Reward Shaping, Graduated Curriculum Learning, and Test-Time Expectamax Search.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/GUI-Pygame-green.svg)](https://www.pygame.org/)
[![Architecture](https://img.shields.io/badge/Model-Rainbow%20DQN%20%2B%20AAREN-purple.svg)](#core-architectural-pillars)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-lightgrey.svg)](#license)

</div>

---

## 👥 Co-Ownership & Academic Attribution

This research and software repository represents joint work developed for the Senior Thesis at **Ateneo de Naga University, Department of Computer Science**:

* **Co-Owner / Co-Author:** **Mark Lawrence M. Quibot** ([@MarkUnFound](https://github.com/MarkUnFound))
* **Co-Owner / Co-Author:** **James Gabriel P. Mabagos**
* **Thesis Advisor:** **Marianne A. Tolentino**
* **Institution:** Department of Computer Science, Ateneo de Naga University
* **Thesis Title:** *MARQ: A Multi-Agent Rainbow Deep Q-Networks Framework using AAREN for Mastering Imperfect Information Games*
* **Repository:** [https://github.com/MarkUnFound/Mastering_Stratego.git](https://github.com/MarkUnFound/Mastering_Stratego.git)

---

## 📌 Overview

**Stratego** is a two-player, imperfect-information board game of deceptive depth played on a $10 \times 10$ grid with two $2 \times 2$ impassable lakes. Each side commands 40 pieces:
* Ranks 1–9 (Marshal down to Scout)
* Specialized tactical units: **Spy** (defeats the Marshal if attacking), **Miner** (defuses Bombs)
* Immobile hazards & targets: **Bombs** (defeat any attacker except Miners) and the **Flag** (capturing it wins the game).

### The Challenge
Unlike perfect-information games such as Chess ($10^{123}$ game tree complexity) or Go ($10^{360}$), Stratego features **hidden piece identities** with a combinatorial game-tree complexity exceeding **$10^{535}$**. At the start of a match, all 40 opponent pieces are face-down. Standard minimax search is computationally intractable and fundamentally flawed because pieces are stochastic chance variables rather than deterministic states.

Furthermore, traditional Deep Q-Networks (DQN) applied to Stratego suffer from:
1. **The "Happy Wanderer" Problem**: Agents learn passive, cyclical piece shuffling to farm heuristic rewards, resulting in catastrophic draw rates (>50%).
2. **The Memory Bottleneck**: LSTMs struggle with long-horizon dependencies (500–1,500 turn games) and slow sequential backpropagation through time.
3. **Tactical Hallucination / Overconfidence**: Single-pass neural networks often gamble high-value pieces (e.g., committing a Marshal into a hidden square that has a high probability of being a Bomb).

**MARQ** (Multi-Agent Rainbow Deep Q-Networks) solves these challenges through an end-to-end framework integrating **Distributional Rainbow DQN**, a novel **AAREN** recurrence mechanism, **Potential-Based Reward Shaping (PBRS)**, a **5-phase graduated curriculum**, and test-time **Expectamax Search**.

---

## 🧠 Core Architectural Pillars

```
+-------------------------------------------------------------------------------+
|                             MARQ AGENT PIPELINE                               |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [Opponent Move History] ------------> [ AAREN Network ]                      |
|                                        - Associative Scan Training (Parallel) |
|                                        - O(1) Recurrent Update (Inference)    |
|                                        - 64-dim Latent Belief Embedding       |
|                                                   |                           |
|  [15-Channel Board Observation] ------------------+                           |
|                                                   |                           |
|                                                   v                           |
|                                      [ Concatenated State: 79-Ch ]            |
|                                                   |                           |
|                                                   v                           |
|                                         [ ResNet-6 Backbone ]                 |
|                                         - 6 Residual Blocks (64 ch)           |
|                                         - 4-Head Spatial Self-Attention       |
|                                         - 27x27 Receptive Field               |
|                                                   |                           |
|                                                   v                           |
|                                        [ Rainbow Dueling Heads ]              |
|                                        - Value V(s) / Advantage A(s,a)        |
|                                        - C51 Categorical Distribution (51)    |
|                                        - Noisy Exploration (No Epsilon Decay) |
|                                                   |                           |
|                                                   v                           |
|                           [ Raw Action Q-Values & Posterior Beliefs ]         |
|                                                   |                           |
|                               [ Test-Time Expectamax Search ]                 |
|                               - Depth-4 Lookahead (4.0s Timeout)              |
|                               - Chance Nodes weighted by AAREN Predictions    |
|                               - Prunes Tactical Blunders / Bombs              |
|                                                   |                           |
|                                                   v                           |
|                                         [ Executed Board Move ]               |
+-------------------------------------------------------------------------------+
```

---

### 1. Rainbow DQN Backbone
The decision engine features a full **21.7M parameter Rainbow DQN**:
* **Categorical Distributional RL (C51)**: Instead of estimating a scalar expected reward, the network outputs a discrete probability distribution over 51 atoms representing return quantiles. This models value uncertainty directly stemming from hidden opponent ranks.
* **Dueling Network Architecture**: Decouples the state value stream $V(s)$ from action advantages $A(s, a)$, stabilizing training in large state spaces where many actions have equivalent non-terminal values.
* **Multi-Step Returns ($N=5$)**: Bootstraps credit assignment over 5-step trajectories, propagating sparse terminal rewards (flag capture) significantly faster across 1,500-step games.
* **Noisy Linear Layers**: Injects parametric Gaussian noise into network weights for state-dependent exploration, eliminating fragile hand-tuned $\epsilon$-greedy decay schedules.
* **Prioritized Experience Replay (PER)**: Samples transitions proportional to TD-error magnitudes using sum-tree data structures, focusing updates on high-leverage tactical transitions.
* **Vision Backbone**: A 6-block ResNet (64 channels) augmented with a 4-head Spatial Self-Attention layer, creating an effective receptive field of $27 \times 27$ across the $10 \times 10$ board.

---

### 2. AAREN: Attention as a Recurrent Neural Network
A core contribution of this project is **AAREN (Attention as a Recurrent Neural Network)**, designed to overcome the limitations of both LSTMs and Transformers in real-time game inference:

* **The Problem with LSTMs**: Sequential gating causes vanishing gradients over 1,000+ turn games; piece identity accuracy plateaued at only 17.74%.
* **The Problem with Standard Transformers**: Quadratic $O(T^2)$ memory scaling and high per-step latency make live interactive inference unfeasible.

#### How AAREN Works
AAREN reformulates the attention mechanism into an **associative prefix operator** maintaining a recurrent state tuple $(a_t, c_t, m_t)$:
* $a_t$: Numerator accumulating exponentially weighted value projections
* $c_t$: Denominator tracking partition normalization
* $m_t$: Running cumulative maximum score $\max(m_{\text{prev}}, s_t)$ preventing numerical overflow:

$$s_t = q^\top k_t$$

$$m_t = \max(m_{t-1}, s_t)$$

$$a_t = a_{t-1} e^{m_{t-1} - m_t} + v_t e^{s_t - m_t}$$

$$c_t = c_{t-1} e^{m_{t-1} - m_t} + e^{s_t - m_t}$$

$$\text{Output}_t = \frac{a_t}{c_t + \epsilon}$$

#### Dual-Mode Advantage
1. **Parallel Training**: Trained across full action trajectories using a custom parallel prefix scan (JIT-compiled kernel), bypassing sequential recurrent bottlenecks.
2. **$O(1)$ Inference**: Executes in constant time per move during live play without recalculating attention over prior history.

The resulting **64-dimensional latent embedding** is concatenated with the 15-channel raw board observation, providing the Rainbow DQN with a rich **79-channel state representation**.

---

### 3. Potential-Based Reward Shaping (PBRS) & Anti-Stall
Early models converged into the **"Happy Wanderer"** local optimum: repeatedly shuffling pieces back and forth ($A \leftrightarrow B$) to farm dense heuristic rewards while avoiding combat risks, dragging 50.04% of games into stalemate draws.

To eliminate this without altering the optimal policy, MARQ replaces additive rewards with formal **Potential-Based Reward Shaping (PBRS)**:

$$R_{\text{shaped}}(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

Where the potential function $\Phi(s)$ balances:
* **Material Advantage (40%)**: Evaluates surviving friendly pieces vs. revealed opponent losses.
* **Flag Proximity (35%)**: Inverted Manhattan distance from combat pieces to suspected flag positions.
* **Board Penetration (15%)**: Rewarding advancement into the opponent's defensive half.
* **Information Gain (10%)**: Rewarding actions that force opponent piece revelations.

Strict anti-oscillation penalties punish repetitive moves, compelling the agent to seek decisive terminal victories.

---

### 4. Graduated Curriculum & League Training (PFSP)
Training is structured into a 5-phase progressive curriculum gated by strict win-rate thresholds (>55% decisive wins required to graduate):

| Phase | Focus | Environment & Rules | Opponents | Max Turns |
|---|---|---|---|---|
| **Phase 1** | Spatial Rules & Flag Navigation | Full visibility, forward pathing | Random Walkers | 200 |
| **Phase 2** | Combat & Material Preservation | Partial visibility, basic combat | Greedy & Rule-Based | 400 |
| **Phase 3** | Bluffing & Hidden Piece Defense | Hidden ranks, scout probing | Smart Heuristics | 1,000 |
| **Phase 4** | Tactical Diversity & Exploiters | Full Stratego rules | Rusher, Turtle, Flanking bots | 1,500 |
| **Phase 5** | League Play & Self-Improvement | Unrestricted competitive play | Prioritized Fictitious Self-Play (PFSP) | 1,500 |

Under **Prioritized Fictitious Self-Play (PFSP)**, opponent matchmaking is dynamically weighted: opponents that defeat the agent are sampled more frequently, while mastered opponents maintain a 5% floor to guard against catastrophic forgetting.

---

### 5. Test-Time Expectamax Search ("The Final Move on the Piece")

> *"Not really sure how this worked, but it did. Probably just hallucinated half way of the tech."*

In reinforcement learning under partial observability, deep neural networks often suffer from **tactical hallucination**—the network's single-pass inference may evaluate an aggressive forward attack as high-value, blind to the catastrophic risk that the concealed opponent square harbors a lethal Bomb or higher rank.

At test time, MARQ resolves this through a **Formal Expectamax Search Engine** ([`dqn_bot_logic.py`](file:///c:/Users/Mark%20Lawrence%20Quibot/repo/Research/Python%20Stratego%20Game/dqn_bot_logic.py)) operating over a 4-ply depth bounded by a safety timeout:

1. **Max Nodes (Agent Turn)**: Evaluates candidate legal moves prioritizing branches with high prior Q-values from the Rainbow network.
2. **Chance Nodes (Attacking Hidden Pieces)**: When a move initiates combat against a face-down enemy piece, standard minimax cannot resolve the outcome. Expectamax queries the **AAREN posterior rank distribution** $P(\text{rank}_j \mid \mathcal{H})$, falling back to the 40-piece setup prior if the piece has not moved yet:

$$\mathbb{E}[U(\text{move})] = \sum_{r \in \text{Ranks}} P(\text{defender} = r \mid \mathcal{H}) \cdot \mathcal{U}(\text{attacker}, r)$$

3. **Pruning & Grounding**: Low-probability chance branches are pruned (evaluating the top-3 most likely ranks). The expected combat utility is dynamically combined with the Rainbow DQN's positional Q-value:
   * A Marshal attacking a square with high Bomb probability receives a sharp negative penalty, overriding optimistic raw Q-values.
   * A Miner attacking that same square receives positive utility.

This hybrid integration grounds deep neural value estimates in rigorous Bayesian tactical verification, preventing blunders on crucial pieces.

---

## 📊 Empirical Benchmarks

Extensive data mining across historical training logs demonstrates the superiority of the Rainbow DQN + AAREN architecture over the baseline Vanilla DQN + LSTM:

| Metric | Vanilla DQN + LSTM (15k Baseline) | Rainbow DQN + AAREN (9.5k Run) | Rainbow DQN + AAREN (75k Extended) |
|---|---|---|---|
| **Episodes Trained** | 15,302 | 9,778 | 75,047 |
| **Win Rate** | 24.97% | **49.88%** | **Continuous Scaling** |
| **Draw Rate (Stalemates)** | 50.04% | **6.15%** (8× reduction) | < 5% |
| **Average Steps / Episode** | 673.5 | **523.0** | Task-directed |
| **Steps Required Per Win** | 2,697 | **1,062** (2.5× faster) | Optimized |
| **Piece Identity Accuracy** | 17.74% | **24.23%** | Superior inference |
| **Loss vs. Win-Rate Correlation** | +0.051 (Decoupled) | **-0.787** (Strong coupling) | Stable convergence |
| **Model Size** | ~39 MB | ~433 MB (259 MB checkpoint) | Rich representations |

---

## 📂 Repository Structure

```plaintext
Mastering_Stratego/
├── Modular Stratego/                   # Core RL & Training Framework
│   ├── train_dqn.py                    # Multi-lane training loop with league & PFSP
│   ├── analyze_results_for_paper.py    # Evaluation and metric calculation
│   ├── analyze_training_data.py        # Historical trajectory data mining
│   ├── network/                        # Neural architectures
│   │   ├── aaren/                      # AAREN cell, associative kernel, and network
│   │   │   ├── cell.py                 # O(1) recurrent update step
│   │   │   ├── kernel.py               # Parallel associative scan operator
│   │   │   └── network.py              # PieceActionAaren embedding module
│   │   ├── dqn_models/                 # Checkpoints, metric plots, and evaluations
│   │   ├── drqn_agent.py               # Rainbow DQN agent implementation
│   │   ├── distributional_reward.py    # C51 distributional RL helpers
│   │   ├── prioritized_memory.py       # Prioritized Experience Replay (PER) buffer
│   │   └── opponents.py                # Opponent pool (Heuristic, Greedy, Exploiters)
│   ├── environment/                    # Stratego game engine & vectorization
│   │   ├── curriculum.py               # 5-Phase curriculum manager
│   │   └── league_manager.py           # PFSP league matchmaking
│   └── settings/                       # Hyperparameter configurations
│       └── training_config.py          # Unified training hyperparameters
│
├── Python Stratego Game/               # Interactive Pygame GUI & Deployment Engine
│   ├── guided_pygame_test.py           # AI Coach GUI with eval bar, rank probabilities, & hints
│   ├── pygame test.py                  # Standard interactive GUI
│   ├── test strategoo.py               # Interactive game with Legends & Rules side-panel
│   ├── dqn_bot_logic.py                # Expectamax Search & Rainbow DQN bridge
│   ├── bot_logic.py                    # Bot interface adapter for Pygame
│   ├── verify_expectamax.py            # Automated test suite for Expectamax combat utilities
│   ├── agent1_league_episode_1000.pt   # Pre-trained Rainbow DQN + AAREN checkpoint (Step 389,530)
│   └── pieces/                         # Graphical board and piece assets
│
├── Latex/                              # Complete Undergraduate Thesis Manuscript
│   ├── sp.tex                          # Primary LaTeX thesis document
│   ├── chapters/                       # Individual chapters (Methodology, Results, etc.)
│   └── figures/                        # Architectural diagrams, charts, and gameplay plots
│
└── README.md                           # Repository documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU recommended (CPU execution supported)

Install dependencies:
```bash
pip install torch numpy pygame matplotlib tqdm scipy
```

### 2. Play Against the Trained Agent (AI Coach GUI)
Launch the interactive AI Coach interface featuring the trained Rainbow DQN agent integrated with Expectamax test-time search and real-time rank predictions:

```bash
cd "Python Stratego Game"
python guided_pygame_test.py
```

Or launch the standard graphical game:
```bash
cd "Python Stratego Game"
python "pygame test.py"
```

### 3. Verify Expectamax Tactical Logic
Run the verification suite testing that combat outcomes (including Flag capture and Bomb defusal) evaluate to accurate positive/negative utilities across both GUI and canonical schemas:

```bash
cd "Python Stratego Game"
python verify_expectamax.py
```

### 4. Run System Diagnostics & Unit Tests
Run preflight checks and the 5-point Spatial Self-Attention diagnostic:

```bash
python "Modular Stratego/test/preflight_checks.py"
python "Modular Stratego/test/check_spatial_attention.py"
```

### 5. Train the MARQ Agent
To launch multi-lane parallel training with the 5-phase curriculum and PFSP league:

```bash
cd "Modular Stratego"
python train_dqn.py
```

Hyperparameters can be adjusted in [`Modular Stratego/settings/training_config.py`](Modular%20Stratego/settings/training_config.py).

---

## 📖 Citation

If you use this codebase or build upon the MARQ framework in your research, please cite:

```bibtex
@thesis{mabagos_quibot_2025_marq,
  author    = {James Gabriel P. Mabagos and Mark Lawrence M. Quibot},
  title     = {MARQ: A Multi-Agent Rainbow Deep Q-Networks Framework using AAREN for Mastering Imperfect Information Games},
  school    = {Department of Computer Science, Ateneo de Naga University},
  year      = {2025},
  month     = {September},
  note      = {Undergraduate Senior Thesis. Advised by Marianne A. Tolentino},
  url       = {https://github.com/MarkUnFound/Mastering_Stratego}
}
```

---

## 📜 License

This project is licensed under the **MIT License** for academic, research, and non-commercial open-source use. See individual file headers for specific component licensing.
