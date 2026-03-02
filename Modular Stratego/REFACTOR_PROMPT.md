# Modular Stratego - AI Model Prompt

**Instructions for the User:** 
Copy the text below the line and paste it as the initial prompt to your agent (or start a new session with this prompt) to begin building the model.

---

You are an expert Principal AI Software Engineer tasked with building the "Modular Stratego" game environment and AI model from scratch.

## Project Context
Your objective is to build a clean, highly modular, efficient, and readable Python codebase for the Modular Stratego game environment and AI agent. I will be handling the git repository, version control, and surrounding infrastructure; your sole focus is generating the codebase, the working environment playground, and the architecture of the model itself from scratch.

## Strict Architectural Constraints
You must implement the following core components with mathematical correctness and optimal design:
1. **The Game Environment (Playground)**: A fully functional Python implementation of Stratego. It must handle all game rules, piece mechanics, battle resolutions, and win/loss states. It should be structured as an RL environment (e.g., following an OpenAI Gymnasium-like interface) and serve as a playable playground for evaluating the agent.
2. **Vision Backbone**: AAREN 6-block ResNet (64ch) with Spatial Self-Attention (4-head, learnable 2D positional encoding).
3. **History Representation**: AAREN (Attention-as-a-RNN) — 3-layer, 64-dim, parallel training.
4. **Core RL Algorithm**: Rainbow DQN integration (C51 Distributional, Dueling Network, Noisy Nets, Multi-step returns, and Prioritized Experience Replay).
5. **GUI / Inference**: Expectamax test-time search.
6. **Action Selection**: Softmax-based move selection (Boltzmann Exploration).
7. **Metrics & Visualization**: Matplotlib-based graphing and tracking for training metrics, loss, and agent improvement over time.

## Development Goals
1. **Extreme Modularity**: Separate concerns strictly from day one:
   - Create a dedicated `networks/` or `models/` package for `ResNet`, `SpatialAttention`, and `AAREN`.
   - Separate the RL agent logic (memory, action selection, loss calculation) from the training loop.
   - Isolate environment logic, reward shaping (PBRS), and curriculum progression into their own modules.
2. **Configuration Management**: Create a centralized hyperparameter and configuration management system to avoid passing dozens of kwargs across functions.
3. **Code Clarity**: Add clear type hinting, modularize utility functions, and write concise, highly readable docstrings.

## Task Tracking & Token Efficiency (CRITICAL INSTRUCTION)
Because this is a massive undertaking, **DO NOT** attempt to write the entire codebase in a single massive output. You will quickly run out of tokens and lose context. Follow this strict scaffolding process:

**Phase 1: Planning (PLANNING MODE)**
- Focus on designing the architecture of the model and its modules.
- Create an `implementation_plan.md` that outlines the target directory structure, expected file delineations, and dependency graphs.
- **Create a `task.md` file** containing an exhaustive checklist of all tasks required to complete the model. This `task.md` will be used to track the current completion state of the model. 
- Use `notify_user` to pause and get output approval on the architecture plan and tasks before writing any python code.

**Phase 2: Execution - Phased Approach (EXECUTION MODE)**
Execute the build in bite-sized, independent phases using `task_boundary`. For example:
- **Task 1**: Scaffold the directory structure and implement configuration management.
- **Task 2**: Implement the Stratego Game Environment (Board mechanics, rules, playable playground GUI, and RL wrapper).
- **Task 3**: Implement Neural Network modules (`networks/`).
- **Task 4**: Implement Replay Memory & AAREN history aggregation.
- **Task 5**: Implement the Rainbow DQN Agent class.
- **Task 6**: Implement the Training Curriculum, Metric Tracking (Matplotlib), & Main Loop.
After completing each task, mark it as complete in `task.md` and verify everything is clean and logical before moving to the next task. Keep `task.md` updated at all times!

**Phase 3: Verification (VERIFICATION MODE)**
- Create unit tests (`test_*.py`) to ensure the mathematical integrity of the models.
- Document the resulting architecture and your final understanding of the model features in `AGENTS_STATUS.md`.

Proceed with Phase 1: Planning. Start by generating the `implementation_plan.md` and `task.md` files.
