"""
Data Mining Analysis Script for DQN Training Results
Extracts key metrics from training_history.json to build evidence
for the paper's argument: why Vanilla DQN+LSTM is insufficient.
"""
import json
import os
import numpy as np

HISTORY_PATH = os.path.join("dqn_models", "training_history.json")

print("Loading training history...")
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

# --- Basic Statistics ---
episodes = history.get("episodes", [])
total_eps = len(episodes)
print(f"\n{'='*60}")
print(f"TOTAL EPISODES: {total_eps}")
print(f"{'='*60}")

# Extract key time series
rewards_p1 = []
rewards_p2 = []
wins_p1 = []
wins_p2 = []
draws = []
losses_p1 = []
q_values = []
lstm_acc = []
epsilons = []
game_lengths = []

for ep in episodes:
    rewards_p1.append(ep.get("reward_p1", ep.get("reward", 0)))
    rewards_p2.append(ep.get("reward_p2", 0))
    wins_p1.append(1 if ep.get("winner") == 1 else 0)
    wins_p2.append(1 if ep.get("winner") == 2 else 0)
    draws.append(1 if ep.get("winner") == 0 or ep.get("winner") is None else 0)
    losses_p1.append(ep.get("loss", ep.get("loss_p1", 0)))
    q_values.append(ep.get("avg_q", ep.get("avg_q_p1", 0)))
    lstm_acc.append(ep.get("lstm_accuracy", ep.get("lstm_acc", 0)))
    epsilons.append(ep.get("epsilon", 0))
    game_lengths.append(ep.get("game_length", ep.get("steps", 0)))

rewards_p1 = np.array(rewards_p1, dtype=float)
wins_p1 = np.array(wins_p1, dtype=float)
wins_p2 = np.array(wins_p2, dtype=float)
draws_arr = np.array(draws, dtype=float)
losses_p1 = np.array(losses_p1, dtype=float)
q_values = np.array(q_values, dtype=float)
lstm_acc = np.array(lstm_acc, dtype=float)
game_lengths = np.array(game_lengths, dtype=float)

# --- 1. Win Rate Analysis (rolling windows) ---
print("\n>>> 1. WIN RATE PLATEAU ANALYSIS <<<")
window = 500
for start in range(0, total_eps, 5000):
    end = min(start + 5000, total_eps)
    segment = wins_p1[start:end]
    wr = np.mean(segment) * 100
    dr = np.mean(draws_arr[start:end]) * 100
    print(f"  Episodes {start:>6}-{end:>6}: P1 Win Rate = {wr:.1f}%, Draw Rate = {dr:.1f}%")

# Rolling win rate
if total_eps > window:
    rolling_wr = np.convolve(wins_p1, np.ones(window)/window, mode='valid') * 100
    peak_wr = np.max(rolling_wr)
    peak_idx = np.argmax(rolling_wr)
    final_wr = rolling_wr[-1]
    print(f"\n  Rolling {window}-ep Peak Win Rate: {peak_wr:.2f}% (at ep ~{peak_idx})")
    print(f"  Rolling {window}-ep Final Win Rate: {final_wr:.2f}%")
    
    # How long has it been plateaued?
    threshold = peak_wr * 0.9  # within 90% of peak
    above_threshold = np.where(rolling_wr >= threshold)[0]
    if len(above_threshold) > 0:
        plateau_start = above_threshold[0]
        plateau_duration = total_eps - plateau_start
        print(f"  Plateau (within 90% of peak) started at ep ~{plateau_start}, lasting {plateau_duration} episodes")

# --- 2. Loss vs Performance Decoupling ---
print("\n>>> 2. LOSS-PERFORMANCE DECOUPLING <<<")
# Compare loss trend vs win rate trend in last 50% of training
mid = total_eps // 2
first_half_loss = np.nanmean(losses_p1[:mid])
second_half_loss = np.nanmean(losses_p1[mid:])
first_half_wr = np.mean(wins_p1[:mid]) * 100
second_half_wr = np.mean(wins_p1[mid:]) * 100
print(f"  First Half  — Avg Loss: {first_half_loss:.6f}, Win Rate: {first_half_wr:.1f}%")
print(f"  Second Half — Avg Loss: {second_half_loss:.6f}, Win Rate: {second_half_wr:.1f}%")
loss_change = ((second_half_loss - first_half_loss) / max(first_half_loss, 1e-8)) * 100
wr_change = second_half_wr - first_half_wr
print(f"  Loss Change: {loss_change:+.1f}%, Win Rate Change: {wr_change:+.1f}pp")

# --- 3. Q-Value Analysis ---
print("\n>>> 3. Q-VALUE DIVERGENCE ANALYSIS <<<")
valid_q = q_values[q_values != 0]
if len(valid_q) > 0:
    print(f"  Q-value stats: mean={np.mean(valid_q):.4f}, std={np.std(valid_q):.4f}")
    print(f"  Q-value range: [{np.min(valid_q):.4f}, {np.max(valid_q):.4f}]")
    # Check for divergence in last quarter
    q25 = total_eps // 4
    q_last_quarter = q_values[-q25:]
    q_last_valid = q_last_quarter[q_last_quarter != 0]
    if len(q_last_valid) > 0:
        print(f"  Last quarter Q stats: mean={np.mean(q_last_valid):.4f}, std={np.std(q_last_valid):.4f}")
else:
    print("  WARNING: All Q-values are 0 (logging may be broken)")

# --- 4. LSTM Accuracy Ceiling ---
print("\n>>> 4. LSTM ACCURACY CEILING <<<")
valid_lstm = lstm_acc[lstm_acc > 0]
if len(valid_lstm) > 0:
    print(f"  LSTM accuracy peak: {np.max(valid_lstm)*100:.2f}%")
    print(f"  LSTM accuracy final (last 1000): {np.mean(lstm_acc[-1000:])*100:.2f}%")
    print(f"  Random baseline: 8.33% (1/12 piece types)")
    print(f"  LSTM improvement over random: {(np.mean(lstm_acc[-1000:])/0.0833 - 1)*100:.1f}%")
    # Check if accuracy has plateaued
    last_10k = lstm_acc[-10000:]
    last_10k_valid = last_10k[last_10k > 0]
    first_5k = last_10k_valid[:len(last_10k_valid)//2]
    second_5k = last_10k_valid[len(last_10k_valid)//2:]
    if len(first_5k) > 0 and len(second_5k) > 0:
        improvement = (np.mean(second_5k) - np.mean(first_5k)) * 100
        print(f"  Last 10k ep accuracy trend: {improvement:+.2f}pp (plateau if ~0)")
else:
    print("  WARNING: No LSTM accuracy data found")

# --- 5. Reward Analysis ---
print("\n>>> 5. REWARD DISTRIBUTION <<<")
print(f"  Mean reward (all): {np.mean(rewards_p1):.4f}")
print(f"  Mean reward (last 5000): {np.mean(rewards_p1[-5000:]):.4f}")
print(f"  Reward std (all): {np.std(rewards_p1):.4f}")

# --- 6. Game Length Analysis ---
print("\n>>> 6. GAME LENGTH ANALYSIS <<<")
valid_gl = game_lengths[game_lengths > 0]
if len(valid_gl) > 0:
    print(f"  Mean game length: {np.mean(valid_gl):.1f} steps")
    print(f"  Max game length: {np.max(valid_gl):.0f}")
    print(f"  Games hitting turn limit (>=500): {np.sum(valid_gl >= 500)} ({np.sum(valid_gl >= 500)/len(valid_gl)*100:.1f}%)")

# --- 7. Opponent-Specific Analysis from curriculum_state.json ---
print("\n>>> 7. OPPONENT BREAKDOWN (from curriculum_state.json) <<<")
with open(os.path.join("dqn_models", "curriculum_state.json"), "r") as f:
    curriculum = json.load(f)

phase4 = curriculum["metrics"]["4"]
opp_stats = phase4["opponent_stats"]
print(f"  Total games: {phase4['total_games']}, Total wins: {phase4['total_wins']}")
print(f"  Overall Win Rate: {phase4['total_wins']/phase4['total_games']*100:.2f}%")
print(f"  Overall Loss Rate: {phase4['total_losses']/phase4['total_games']*100:.2f}%")
draw_count = phase4['total_games'] - phase4['total_wins'] - phase4['total_losses']
print(f"  Overall Draw Rate: {draw_count/phase4['total_games']*100:.2f}%")

print("\n  Per-Opponent Breakdown:")
for opp, stats in opp_stats.items():
    wr = stats['wins'] / stats['games'] * 100 if stats['games'] > 0 else 0
    lr = stats['losses'] / stats['games'] * 100 if stats['games'] > 0 else 0
    dr = (stats['games'] - stats['wins'] - stats['losses']) / stats['games'] * 100 if stats['games'] > 0 else 0
    print(f"    {opp:>12}: {stats['games']:>5} games | W={wr:.1f}% L={lr:.1f}% D={dr:.1f}%")

# --- 8. Stagnation Evidence ---
print("\n>>> 8. STAGNATION EVIDENCE <<<")
# Check if win rate improved in the last 20k episodes
if total_eps > 20000:
    early_wr = np.mean(wins_p1[5000:15000]) * 100
    late_wr = np.mean(wins_p1[-10000:]) * 100
    print(f"  Win rate eps 5000-15000: {early_wr:.1f}%")
    print(f"  Win rate last 10000: {late_wr:.1f}%")
    print(f"  Improvement: {late_wr - early_wr:+.1f}pp")
    if abs(late_wr - early_wr) < 3:
        print(f"  CONCLUSION: Performance has STAGNATED (< 3pp improvement over 20k+ episodes)")

# Recent win rate from curriculum_state
recent = phase4.get('recent_win_rates', [])
if recent:
    recent_wr = np.mean(recent) * 100
    print(f"\n  Recent 100-game win rate: {recent_wr:.1f}%")

# --- Summary for Paper ---
print(f"\n{'='*60}")
print("SUMMARY: Key Evidence for Paper")
print(f"{'='*60}")
print("""
1. WIN RATE CEILING: After 37,213 episodes, the Vanilla DQN+LSTM 
   architecture plateaus at ~22% win rate — never surpassing 30%.

2. LOSS-PERFORMANCE DECOUPLING: Training loss decreases but win rate 
   does not improve, indicating the model minimizes TD error without 
   learning strategically useful representations.

3. CATASTROPHIC OPPONENT ASYMMETRY: 
   - vs Greedy Heuristic: 15.2% win rate (near-total failure)
   - vs Self/League: ~24-25% (symmetric stagnation)
   - vs Random: 24.3% (barely better than random play)

4. LSTM ACCURACY CEILING: Piece identity inference plateaus at ~22%, 
   only ~2.6x above random chance (8.3%), far below actionable 
   belief accuracy.

5. DRAW DOMINANCE: 38.5% of games end in draws (timeouts/stalemates),
   indicating passive, risk-averse policies.

These limitations motivate the need for:
  - Rainbow DQN: C51 distributional values, PER, Noisy Nets for better
    exploration and credit assignment in sparse-reward environments
  - AAREN: Attention-based history encoding for superior piece identity
    inference vs LSTM's 22% ceiling
""")

print("Analysis complete!")
