# PBS Evaluator and Setup Agent Analysis

## 1. How PBS Evaluator Improves

### Training Process

The PBS Evaluator learns through **supervised learning** with experience replay:

1. **Data Collection** (during gameplay):
   - When a piece is revealed (after battle), `update_from_reveal()` is called
   - The PBS prediction (before reveal) is stored along with ground truth
   - Only collects data from middle/end game (skips early game)
   - Stored in experience replay buffer: `(pbs_prediction, ground_truth, position, game_phase, turn_count)`

2. **Reward Computation**:
   - Ground truth rewards are computed using `compute_reward()`:
     - **Base reward**: Confidence in correct piece type × 10.0
     - **Distance penalty**: Penalizes predictions far from actual value (-5.0 × normalized_distance)
     - **Value multiplier**: Higher value pieces (Marshal=11, General=10) get more weight (0.5-1.5×)
     - **Bonus**: High confidence (>0.7) in correct prediction gets +2.0
     - **Penalty**: High confidence (>0.5) in wrong piece gets -3.0
   - Example: Correctly predicting Marshal with 0.8 confidence = (0.8×10 - 0×5) × 1.5 + 2.0 = 14.0 reward

3. **Training** (every 20 episodes):
   - Uses experience replay buffer (random sampling)
   - **Supervised learning**: Network learns to predict ground truth rewards
   - **Hybrid approach**: 80% ground truth + 20% target network (for stability)
   - Loss: MSE between predicted quality score and target reward
   - Network architecture: 4-layer MLP with batch normalization

4. **Improvement Over Time**:
   - **More training data**: More pieces revealed = more diverse examples
   - **Better pattern recognition**: Learns which belief patterns indicate good predictions
   - **Value awareness**: Understands that correct predictions for high-value pieces matter more
   - **Confidence calibration**: Learns to distinguish between justified and overconfident predictions
   - **Stable learning**: Target network prevents overfitting to recent experiences

### Why It Improves

- **Supervised learning**: Directly learns from ground truth (actual piece types)
- **Experience replay**: Breaks correlation between consecutive experiences
- **Target network**: Provides stable targets (80% ground truth + 20% target)
- **Value-weighted**: Higher value pieces get more weight in training
- **Game phase awareness**: Only trains on middle/end game (more meaningful predictions)

---

## 2. Setup Agent Loss = 0 Issue

### What Loss = 0 Actually Means

**Loss = 0 does NOT mean optimal setup!**

The setup agent uses **DQN loss** (MSE between Q-values and target Q-values):
```python
loss = F.mse_loss(current_q_values, target_q_values)
```

**Loss = 0** means:
- ✅ The Q-network **perfectly predicts** the target Q-values
- ✅ The network has **learned the reward structure**
- ✅ It can accurately estimate rewards for different setups
- ✅ The network has **converged** to the reward function

**But it does NOT mean:**
- ❌ The setup is optimal
- ❌ The reward function is correct
- ❌ The agent is making good decisions
- ❌ The flag is in a safe position

### Why Flag in Front Still Gets Rewards

The **original reward function** had multiple components that could compensate for poor flag placement:

1. **Flag Protection (0-5.0 points)**:
   - Only checked **adjacent** positions
   - **Did NOT penalize front-row flags**
   - Could still get high score if flag had bombs/pieces adjacent

2. **Other Rewards Could Compensate**:
   - Win/loss (+10.0 for win) - could win despite bad setup
   - Scout placement (+1.5)
   - Bomb placement (+2.0)
   - Defensive formation (+1.5)
   - Piece coordination (+1.0)
   - Early survival (+2.0)
   - **Total possible: ~15+ points even with flag in front**

3. **The Problem**:
   - Flag in front row (row 9 for Player 1, row 0 for Player 2) is **very vulnerable**
   - Original `evaluate_flag_protection()` didn't check row position
   - No explicit penalty for front-row flags
   - Agent learned to predict rewards accurately, but rewards didn't penalize bad flag placement enough

### Fix Applied

**1. Explicit Front-Row Penalty** (in `calculate_setup_agent_reward()`):
```python
# Front row: -10.0 penalty (very vulnerable)
# Second row: -5.0 penalty (still vulnerable)
# Back rows: +2.0 bonus (safer)
```

**2. Improved Flag Protection Evaluation** (in `evaluate_flag_protection()`):
```python
# Row-based vulnerability penalty
# Front-row flags need MUCH more protection to get same score
# Protection score reduced by up to 50% for front-row flags
```

**Result**:
- Front-row flags now get **-10.0 penalty** immediately
- Even with perfect protection, front-row flags get reduced protection score
- Back-row flags get **+2.0 bonus**
- Setup agent should now learn to avoid front-row flags

---

## Summary

### PBS Evaluator
- ✅ **Improves through supervised learning** on revealed pieces
- ✅ **Learns to assess prediction quality** (confidence + accuracy)
- ✅ **Gets better with more training data** (more pieces revealed)
- ✅ **Value-aware**: Understands high-value pieces matter more
- ✅ **Stable training**: Target network prevents overfitting

### Setup Agent Loss = 0
- ⚠️ **Loss = 0 means accurate reward prediction, NOT optimal setup**
- ⚠️ **Original reward function didn't penalize front-row flags enough**
- ✅ **FIXED**: Added explicit -10.0 penalty for front-row flags
- ✅ **FIXED**: Added row-based vulnerability penalty to protection score
- ✅ **FIXED**: Added +2.0 bonus for back-row flags
- 📊 **Monitor**: Track flag positions over time to verify improvement

### Expected Behavior After Fix

1. **Setup Agent**:
   - Should learn to place flags in back rows (rows 6-9 for Player 1, rows 0-3 for Player 2)
   - Loss may increase initially (as it learns new reward structure)
   - Should eventually converge to better setups

2. **PBS Evaluator**:
   - Continues to improve as more pieces are revealed
   - Better at assessing prediction quality over time
   - Provides feedback to improve PBS inference

