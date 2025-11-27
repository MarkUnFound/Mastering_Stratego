import json
import numpy as np

file_path = r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\training_history.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    print(f"Total episodes: {len(data.get('episode_history', []))}")
    
    def print_metric(name, values):
        if not values:
            print(f"{name}: No data")
            return
        recent = values[-100:]
        avg = np.mean(recent)
        print(f"{name}: Avg={avg:.4f}, Min={np.min(recent):.4f}, Max={np.max(recent):.4f}, Last={recent[-1]}")

    print_metric("Agent 1 Rewards", data.get("rewards_history", {}).get("agent1", []))
    print_metric("Agent 2 Rewards", data.get("rewards_history", {}).get("agent2", []))
    print_metric("Agent 1 Loss", data.get("policy_loss_history", {}).get("agent1", []))
    print_metric("Agent 2 Loss", data.get("policy_loss_history", {}).get("agent2", []))
    print_metric("Agent 1 Epsilon", data.get("epsilon_history", {}).get("agent1", []))
    print_metric("Agent 1 Avg Q", data.get("avg_q_history", {}).get("agent1", []))
    
    # Wins
    wins1 = data.get("wins_history", {}).get("agent1", [])
    wins2 = data.get("wins_history", {}).get("agent2", [])
    if wins1 and wins2:
        # Calculate win rate over last 100 episodes
        # wins_history stores CUMULATIVE wins
        w1_start = wins1[-101] if len(wins1) > 100 else wins1[0]
        w1_end = wins1[-1]
        w2_start = wins2[-101] if len(wins2) > 100 else wins2[0]
        w2_end = wins2[-1]
        
        episodes = min(100, len(wins1))
        win_rate1 = (w1_end - w1_start) / episodes
        win_rate2 = (w2_end - w2_start) / episodes
        print(f"Win Rate (Last {episodes}): Agent1={win_rate1:.2%}, Agent2={win_rate2:.2%}")

except Exception as e:
    print(f"Error: {e}")
