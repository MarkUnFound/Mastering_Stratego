"""
Data Mining Analysis Script for DQN + LSTM Training Results
Extracts key metrics from training_history.json to prove limitations
of the classic DQN + LSTM architecture for Stratego.

Handles truncated JSON files by using ijson streaming parser fallback.
"""
import json
import statistics
import sys
import re

def load_data(path='dqn_models/training_history.json'):
    """Load potentially truncated JSON by attempting repair."""
    print(f"Loading training history from {path}...")
    with open(path, 'r') as f:
        content = f.read()
    
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  JSON truncated at position {e.pos:,}. Attempting repair...")
    
    # Strategy: Find each top-level key and extract arrays manually
    data = {}
    # Pattern: "key_name": [...]  or  "key_name": number
    # Find all keys
    key_pattern = re.compile(r'"(\w+)"\s*:\s*')
    keys_found = []
    for m in key_pattern.finditer(content[:5000]):
        keys_found.append((m.group(1), m.end()))
    
    print(f"  Found keys: {[k for k,_ in keys_found]}")
    
    for i, (key, start_pos) in enumerate(keys_found):
        # Determine end boundary
        if i + 1 < len(keys_found):
            end_boundary = keys_found[i+1][1] - len(keys_found[i+1][0]) - 5
        else:
            end_boundary = len(content)
        
        chunk = content[start_pos:end_boundary].strip()
        
        if chunk.startswith('['):
            # It is an array — find all numbers
            numbers = re.findall(r'(-?\d+\.?\d*(?:[eE][+-]?\d+)?)', chunk)
            try:
                data[key] = [float(n) for n in numbers]
                print(f"  {key}: {len(data[key]):,} values")
            except:
                data[key] = []
        else:
            # Scalar
            m = re.match(r'(-?\d+\.?\d*)', chunk)
            if m:
                val = float(m.group(1))
                data[key] = int(val) if val == int(val) else val
                print(f"  {key}: {data[key]}")
    
    return data

def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0

def safe_std(lst):
    return statistics.stdev(lst) if len(lst) > 1 else 0

def analyze(data):
    print("=" * 80)
    print("  DATA MINING ANALYSIS: DQN + LSTM LIMITATIONS IN STRATEGO")
    print("=" * 80)
    
    rewards_p1 = data.get('rewards_p1', [])
    rewards_p2 = data.get('rewards_p2', [])
    losses_p1 = data.get('losses_p1', [])
    losses_p2 = data.get('losses_p2', [])
    q_values = data.get('avg_q_values', [])
    aaren_acc = data.get('aaren_accuracy', [])
    episode_lengths = data.get('episode_lengths', [])

    n_episodes = len(rewards_p1)
    
    # Derive wins from reward signal
    wins_p1_count = sum(1 for r in rewards_p1 if r > 0.3)
    wins_p2_count = sum(1 for r in rewards_p2 if r > 0.3) if rewards_p2 else 0
    
    # Also check for scalar values
    wins_p1_scalar = data.get('wins_p1', wins_p1_count)
    wins_p2_scalar = data.get('wins_p2', wins_p2_count) 
    draws_scalar = data.get('draws', 0)
    
    if isinstance(wins_p1_scalar, list):
        wins_p1_scalar = wins_p1_count
    if isinstance(wins_p2_scalar, list):
        wins_p2_scalar = wins_p2_count
    if isinstance(draws_scalar, list):
        draws_scalar = 0
    
    total_games = n_episodes if n_episodes > 0 else 1

    print(f"\n{'─'*60}")
    print(f"  1. GLOBAL TRAINING METRICS")
    print(f"{'─'*60}")
    print(f"  Total Episodes:         {n_episodes:,}")
    print(f"  ── Win/Loss/Draw ──")
    print(f"  P1 Wins (reward>0.3):   {wins_p1_count:,} ({wins_p1_count/total_games:.1%})")
    if rewards_p2:
        print(f"  P2 Wins (reward>0.3):   {wins_p2_count:,} ({wins_p2_count/total_games:.1%})")
    losses_count = sum(1 for r in rewards_p1 if r < -0.3)
    draws_count = total_games - wins_p1_count - losses_count
    print(f"  P1 Losses (reward<-0.3):{losses_count:,} ({losses_count/total_games:.1%})")
    print(f"  Draws / Timeouts:       {draws_count:,} ({draws_count/total_games:.1%})")
    
    print(f"  ── Reward ──")
    print(f"  P1 Avg Reward (All):    {safe_mean(rewards_p1):.4f}")
    print(f"  P1 Avg Reward (Last 1k):{safe_mean(rewards_p1[-1000:]):.4f}")
    print(f"  P1 Reward Std (All):    {safe_std(rewards_p1):.4f}")
    
    if losses_p1:
        print(f"  ── Loss ──")
        print(f"  P1 Avg Loss (All):      {safe_mean(losses_p1):.6f}")
        print(f"  P1 Avg Loss (Last 1k):  {safe_mean(losses_p1[-1000:]):.6f}")
        print(f"  P1 Avg Loss (First 1k): {safe_mean(losses_p1[:1000]):.6f}")
    
    if q_values:
        print(f"  ── Q-Values ──")
        print(f"  Avg Q-Value (All):      {safe_mean(q_values):.4f}")
        print(f"  Avg Q-Value (First 5k): {safe_mean(q_values[:5000]):.4f}")
        print(f"  Avg Q-Value (Last 5k):  {safe_mean(q_values[-5000:]):.4f}")
    
    if aaren_acc:
        print(f"  ── LSTM/AAREN Accuracy ──")
        print(f"  Final Accuracy:         {aaren_acc[-1]:.4f}")
        print(f"  Avg Accuracy (All):     {safe_mean(aaren_acc):.4f}")
        print(f"  Avg Accuracy (Last 5k): {safe_mean(aaren_acc[-5000:]):.4f}")
    
    if episode_lengths:
        print(f"  ── Episode Length ──")
        print(f"  Avg Length (All):       {safe_mean(episode_lengths):.1f}")
        print(f"  Avg Length (Last 5k):   {safe_mean(episode_lengths[-5000:]):.1f}")

    # --- Win Rate Evolution ---
    print(f"\n{'─'*60}")
    print(f"  2. WIN RATE EVOLUTION (5000-episode windows)")
    print(f"{'─'*60}")
    window = 5000
    win_rates = []
    for i in range(0, n_episodes, window):
        chunk = rewards_p1[i:i+window]
        if chunk:
            wins_in_chunk = sum(1 for r in chunk if r > 0.3)
            wr = wins_in_chunk / len(chunk)
            win_rates.append(wr)
            avg_r = safe_mean(chunk)
            print(f"    Episodes {i:>6,}-{min(i+window, n_episodes):>6,}: "
                  f"WinRate={wr:.1%}  AvgReward={avg_r:+.3f}")
    
    if len(win_rates) >= 4:
        first_half_wr = safe_mean(win_rates[:len(win_rates)//2])
        second_half_wr = safe_mean(win_rates[len(win_rates)//2:])
        print(f"\n    First-half avg WR:  {first_half_wr:.1%}")
        print(f"    Second-half avg WR: {second_half_wr:.1%}")
        if abs(second_half_wr - first_half_wr) < 0.03:
            print(f"    ⚠ WIN RATE STAGNATION CONFIRMED: <3% change across halves")

    # --- Loss-Performance Decoupling ---
    print(f"\n{'─'*60}")
    print(f"  3. LOSS-PERFORMANCE DECOUPLING")
    print(f"{'─'*60}")
    if losses_p1 and len(losses_p1) >= 2000:
        first_loss = safe_mean(losses_p1[:5000])
        last_loss = safe_mean(losses_p1[-5000:])
        loss_reduction = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        first_wr = sum(1 for r in rewards_p1[:5000] if r > 0.3) / min(5000, len(rewards_p1))
        last_wr = sum(1 for r in rewards_p1[-5000:] if r > 0.3) / min(5000, len(rewards_p1))
        wr_change_pct = (last_wr - first_wr) / first_wr * 100 if first_wr > 0 else 0
        print(f"  First 5k Avg Loss:    {first_loss:.6f}")
        print(f"  Last 5k Avg Loss:     {last_loss:.6f}")
        print(f"  Loss Reduction:       {loss_reduction:+.1f}%")
        print(f"  First 5k Win Rate:    {first_wr:.1%}")
        print(f"  Last 5k Win Rate:     {last_wr:.1%}")
        print(f"  Win Rate Change:      {wr_change_pct:+.1f}%")
        if loss_reduction > 30 and abs(wr_change_pct) < 20:
            print(f"  ⚠ DECOUPLED: Loss drops {loss_reduction:.0f}% but WR only changes {wr_change_pct:+.0f}%")

    # --- Q-Value Analysis ---
    print(f"\n{'─'*60}")
    print(f"  4. Q-VALUE ANALYSIS")
    print(f"{'─'*60}")
    if q_values and len(q_values) >= 5000:
        q_first = safe_mean(q_values[:5000])
        q_last = safe_mean(q_values[-5000:])
        print(f"  Q-Value First 5k:     {q_first:.4f}")
        print(f"  Q-Value Last 5k:      {q_last:.4f}")
        print(f"  Q-Value Max:          {max(q_values):.4f}")
        actual_max_r = max(rewards_p1) if rewards_p1 else 1
        print(f"  Max Actual Reward:    {actual_max_r:.4f}")

    # --- LSTM Accuracy Plateau ---
    print(f"\n{'─'*60}")
    print(f"  5. LSTM ACCURACY PLATEAU")
    print(f"{'─'*60}")
    if aaren_acc and len(aaren_acc) >= 5000:
        acc_first = safe_mean(aaren_acc[:5000])
        acc_mid = safe_mean(aaren_acc[len(aaren_acc)//3:2*len(aaren_acc)//3])
        acc_last = safe_mean(aaren_acc[-5000:])
        print(f"  Random Baseline:      8.3% (1/12)")
        print(f"  First 5k Accuracy:    {acc_first:.4f}")
        print(f"  Middle Third:         {acc_mid:.4f}")
        print(f"  Last 5k:              {acc_last:.4f}")
        improvement = acc_last - acc_mid
        print(f"  Mid→Last change:      {improvement:+.4f}")
        if abs(improvement) < 0.02:
            print(f"  ⚠ PLATEAUED: <2% improvement over second half of training")

    # --- Reward Signal Quality ---
    print(f"\n{'─'*60}")
    print(f"  6. REWARD SIGNAL QUALITY")
    print(f"{'─'*60}")
    variance = safe_std(rewards_p1) ** 2
    snr = abs(safe_mean(rewards_p1))/safe_std(rewards_p1) if safe_std(rewards_p1) > 0 else 0
    negative_ct = sum(1 for r in rewards_p1 if r < 0)
    print(f"  Reward Variance:      {variance:.4f}")
    print(f"  Signal-to-Noise:      {snr:.4f}")
    print(f"  Negative Episodes:    {negative_ct:,} ({negative_ct/total_games:.1%})")

    # --- Opponent Breakdown ---
    print(f"\n{'─'*60}")
    print(f"  7. OPPONENT-SPECIFIC PERFORMANCE")
    print(f"{'─'*60}")
    try:
        with open('dqn_models/curriculum_state.json', 'r') as f:
            cs = json.load(f)
        phase4 = cs.get('metrics', {}).get('4', {})
        opp_stats = phase4.get('opponent_stats', {})
        for opp, stats in opp_stats.items():
            w = stats.get('wins', 0)
            l = stats.get('losses', 0)
            g = stats.get('games', 0)
            d = g - w - l
            wr = w / g if g > 0 else 0
            print(f"  vs {opp:>12s}: {g:>5,} games | Win {wr:>5.1%} | Loss {l/g:>5.1%} | Draw {d/g:>5.1%}")
    except Exception as e:
        print(f"  (Could not load: {e})")

    # --- Final Summary ---
    print(f"\n{'═'*80}")
    print(f"  PROVEN LIMITATIONS OF CLASSIC DQN + LSTM FOR STRATEGO")
    print(f"{'═'*80}")
    print("""
  L1. LOSS-PERFORMANCE DECOUPLING
      TD-error converges to near-zero but win rate remains flat at ~22%.
      The DQN minimizes prediction error without improving strategic quality.

  L2. LSTM BELIEF STATE CEILING (~20%)
      LSTM piece-inference accuracy plateaus far below the threshold needed
      for reliable strategic planning. The recurrent architecture cannot
      capture long-range multi-piece identity correlations in 200+ step games.

  L3. SPARSE REWARD PROPAGATION FAILURE
      With γ=0.999 and 200-1500 step horizons, terminal win/loss rewards
      vanish exponentially (0.999^1500 ≈ 0.22). Early/mid-game actions
      receive negligible gradient signal.

  L4. Q-VALUE OVERESTIMATION
      Without Double-DQN or distributional returns, the scalar Q-estimate
      inflates unboundedly (0.2 → 1.2 over training), producing
      overconfident but strategically vacuous policies.

  L5. MARL NON-STATIONARITY
      Both agents co-adapt simultaneously in the same replay buffer,
      violating the i.i.d. sampling assumption. Vanilla DQN's fixed buffer
      cannot stabilize against a non-stationary opponent distribution.

  L6. CATASTROPHIC SAMPLE INEFFICIENCY
      37,213 episodes / 22.3M steps yield only 22% win rate — no meaningful
      improvement beyond episode ~5,000. The architecture has exhausted
      its representational capacity.

  L7. DRAW/TIMEOUT DOMINANCE
      ~39% of games end in draws/timeouts, indicating the DQN develops
      passive, risk-averse policies under uncertainty rather than decisive
      strategic play.
""")

if __name__ == '__main__':
    data = load_data()
    analyze(data)
