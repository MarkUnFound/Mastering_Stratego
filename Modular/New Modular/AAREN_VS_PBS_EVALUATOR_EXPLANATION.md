# AAREN vs PBS Evaluator: Understanding the Relationship

## Overview

There is **NO direct relationship** between AAREN-RNN and the PBS Evaluator. They are **separate, independent components** that work together in the PBS (Probabilistic Belief State) system.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PBS (Probabilistic Belief State)          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │   AAREN-RNN      │         │   PBS Evaluator      │     │
│  │                  │         │                      │     │
│  │  Generates       │         │  Evaluates           │     │
│  │  Predictions     │────────▶│  Predictions         │     │
│  │                  │         │                      │     │
│  │  Input: Actions  │         │  Input: Belief Dist. │     │
│  │  Output: Beliefs │         │  Output: Quality     │     │
│  └──────────────────┘         └──────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. AAREN-RNN (Attention as a Recurrent Neural Network)

### Purpose
**Generates PBS predictions** (belief distributions) from action sequences.

### How It Works

1. **Input**: Action sequences (8 features per action)
   - Move distance
   - Is attack
   - Direction
   - Distance from center
   - Forward/backward/lateral
   - Aggressiveness score

2. **Processing**: 
   - Uses **AarenCell** for attention-based recurrent computation
   - Maintains constant memory O(1) per position
   - Processes action history to infer piece types

3. **Output**: **Belief distribution** (probability over all piece types)
   ```python
   {
       PieceType.FLAG: 0.15,
       PieceType.MARSHAL: 0.10,
       PieceType.GENERAL: 0.08,
       ...
   }
   ```

### Key Features
- ✅ **Parallel training**: Can train on multiple sequences simultaneously
- ✅ **O(1) inference**: Constant memory per position
- ✅ **Stream processing**: Updates beliefs incrementally
- ✅ **No vanishing gradients**: Better than LSTM for long sequences

### Location
- **File**: `probabilistic_belief_state.py`
- **Classes**: `AarenCell`, `PieceActionAaren`
- **Used by**: `ProbabilisticBeliefState._apply_aaren_inference()`

---

## 2. PBS Evaluator

### Purpose
**Evaluates the quality** of PBS predictions (assesses how good the predictions are).

### How It Works

1. **Input**: **Belief distribution** (from AAREN or rule-based inference)
   ```python
   {
       PieceType.FLAG: 0.15,
       PieceType.MARSHAL: 0.10,
       ...
   }
   ```

2. **Processing**:
   - Uses **PBSEvaluatorNetwork** (4-layer MLP)
   - Takes belief distribution as input (NUM_PIECE_TYPES values)
   - Outputs a **quality score** (single value)

3. **Output**: **Quality score** (higher = better prediction)
   - Positive score = good prediction
   - Negative score = poor prediction

### Training Process

1. **Data Collection**:
   - When pieces are revealed, stores:
     - PBS prediction (before reveal)
     - Ground truth (actual piece type)
   - Only collects from middle/end game

2. **Training**:
   - Computes ground truth rewards based on:
     - Confidence in correct piece type
     - Distance from actual value
     - Piece value (higher value = more important)
   - Trains network to predict these rewards
   - Uses supervised learning (MSE loss)

3. **Improvement**:
   - Learns which belief patterns indicate good predictions
   - Better at identifying overconfident wrong predictions
   - Provides feedback to improve PBS inference

### Key Features
- ✅ **Supervised learning**: Learns from ground truth
- ✅ **Value-aware**: Understands high-value pieces matter more
- ✅ **Confidence calibration**: Distinguishes justified vs. overconfident predictions
- ✅ **Experience replay**: Breaks correlation between experiences

### Location
- **File**: `pbs_evaluator.py`
- **Classes**: `PBSEvaluatorNetwork`, `PBSEvaluator`
- **Used by**: `ProbabilisticBeliefState.get_evaluator_feedback()`

---

## 3. How They Work Together

### Flow Diagram

```
Game Action
    │
    ▼
┌─────────────────┐
│  AAREN-RNN      │  ← Processes action sequence
│  (Generates)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Belief Dist.   │  ← Output: {PieceType: probability}
│  {FLAG: 0.15,   │
│   MARSHAL: 0.10}│
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│  PBS Evaluator  │  │  DQN Agent      │
│  (Evaluates)    │  │  (Uses beliefs) │
└────────┬────────┘  └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Quality Score  │  ← Output: How good is the prediction?
│  (e.g., 5.2)    │
└─────────────────┘
```

### Step-by-Step Process

1. **Action Occurs**:
   - Agent makes a move
   - Action features extracted

2. **AAREN Generates Prediction**:
   - AAREN processes action sequence
   - Updates belief distribution for that position
   - Output: `{PieceType: probability}`

3. **PBS Evaluator Evaluates**:
   - Takes belief distribution as input
   - Outputs quality score
   - Can provide feedback on prediction quality

4. **Piece Revealed** (after battle):
   - Ground truth becomes known
   - PBS evaluator stores experience:
     - Prediction (before reveal)
     - Ground truth
   - Trains on this data to improve evaluation

---

## 4. Key Differences

| Aspect | AAREN-RNN | PBS Evaluator |
|--------|-----------|---------------|
| **Purpose** | Generate predictions | Evaluate predictions |
| **Input** | Action sequences | Belief distributions |
| **Output** | Belief distribution | Quality score |
| **Architecture** | Attention-based RNN | Feedforward MLP |
| **Training** | Supervised (piece types) | Supervised (rewards) |
| **Memory** | O(1) per position | Experience buffer |
| **When Used** | During gameplay | After piece reveal |

---

## 5. Why Both Are Needed

### AAREN-RNN
- **Generates** predictions from incomplete information
- Learns patterns from action sequences
- Provides belief distributions for decision-making

### PBS Evaluator
- **Evaluates** prediction quality
- Provides feedback on confidence
- Helps identify when predictions are reliable
- Can guide PBS improvement

### Together
- AAREN generates predictions
- PBS Evaluator assesses their quality
- Both improve over time through training
- Better predictions → better decisions → better gameplay

---

## 6. Common Misconception

**❌ WRONG**: "AAREN is used by the PBS Evaluator"

**✅ CORRECT**: 
- AAREN generates predictions
- PBS Evaluator evaluates those predictions
- They are **separate, independent components**
- PBS Evaluator **uses** AAREN's output (belief distribution), but doesn't **contain** AAREN

---

## Summary

- **AAREN-RNN**: Generates belief distributions from actions
- **PBS Evaluator**: Evaluates quality of belief distributions
- **Relationship**: PBS Evaluator uses AAREN's output (belief distribution) as input
- **Independence**: They are separate neural networks with different purposes
- **Current State**: **NO direct feedback loop** - PBS Evaluator does NOT improve AAREN's beliefs
- **Potential**: Infrastructure exists (`get_evaluator_feedback()`) but is not currently used

---

## 7. Does PBS Evaluator Improve AAREN? (Current Implementation)

### ❌ **NO - Not Currently Implemented**

**Current State:**
- AAREN trains independently using action sequences + ground truth piece types
- PBS Evaluator trains independently using belief distributions + ground truth
- **No feedback loop** from evaluator to AAREN

**Evidence:**
1. `get_evaluator_feedback()` method exists but is **never called** in the codebase
2. AAREN training (`train_aaren()`) only uses:
   - Action sequences
   - Ground truth piece types
   - No evaluator feedback
3. Evaluator feedback is available but not integrated into AAREN training

### How AAREN Currently Improves:

1. **Direct Supervision**: 
   - Trains on action sequences with known piece types
   - Uses cross-entropy loss against ground truth
   - Learns patterns from action → piece type mappings

2. **Rule-Based Updates**:
   - Uses game rules (e.g., "can't move = bomb or flag")
   - Updates beliefs based on constraints

3. **No Evaluator Feedback**:
   - Does NOT use evaluator quality scores
   - Does NOT weight training by evaluator confidence
   - Does NOT adjust beliefs based on evaluator assessment

### ✅ **IMPLEMENTED: Evaluator Feedback Integration**

The evaluator feedback loop has been implemented! Here's how it works:

1. **Weighted AAREN Training**:
   ```python
   # When pieces are revealed, evaluator feedback is stored
   feedback = pbs.get_evaluator_feedback(pos, ground_truth)
   # Quality score and piece value are combined into training weight
   # High-quality predictions for high-value pieces get more weight
   ```

2. **Dynamic Belief Updates**:
   ```python
   # AAREN inference uses evaluator confidence to adjust alpha
   # High quality score -> trust AAREN more (alpha = 0.1 to 0.5)
   # Low quality score -> trust AAREN less (alpha = 0.1)
   ```

3. **Focus on High-Value Pieces**:
   - Piece value (Marshal=11, General=10, etc.) is factored into training weights
   - High-value pieces with good predictions get prioritized in training

**Implementation Details:**
- `train_aaren()` now accepts `evaluator_weights` and `positions` parameters
- `_apply_aaren_inference()` uses evaluator feedback to adjust belief update confidence
- `update_from_reveal()` stores evaluator feedback for later AAREN training
- `train_aaren_with_evaluator_feedback()` automatically collects and trains with feedback

---

## Final Summary

- **AAREN-RNN**: Generates belief distributions from actions
- **PBS Evaluator**: Evaluates quality of belief distributions
- **Relationship**: PBS Evaluator uses AAREN's output as input
- **Feedback Loop**: **✅ IMPLEMENTED** - Evaluator feedback now improves AAREN:
  - **Weighted Training**: High-quality predictions get more weight in AAREN training
  - **Dynamic Belief Updates**: Evaluator confidence adjusts how much AAREN predictions are trusted
  - **High-Value Focus**: Training prioritizes high-value pieces (Marshal, General, etc.)
- **Synergy**: 
  - Better AAREN predictions → Better evaluator training data → Better quality assessment
  - Better evaluator assessment → Better AAREN training weights → Better predictions
- **Implementation**: 
  - `train_aaren()` supports evaluator-weighted training
  - `_apply_aaren_inference()` uses evaluator confidence for dynamic alpha adjustment
  - `train_aaren_with_evaluator_feedback()` automatically trains with stored feedback

Both components now work together in a **feedback loop** where the evaluator improves AAREN's beliefs through weighted training and dynamic confidence adjustment.

