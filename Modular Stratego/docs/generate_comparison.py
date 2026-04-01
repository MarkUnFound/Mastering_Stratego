import json
import numpy as np

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        wins_hist = data.get('wins_p1_history', [])
        rewards_p1 = data.get('rewards_p1', [])
        avg_loss_p1_history = data.get('avg_loss_p1_history', [])
        avg_q_values_p1 = data.get('avg_q_values_p1', [])
        draws = data.get('draws', 0)
        wins_p1 = data.get('wins_p1', 0)
        wins_by_flag = data.get('wins_by_flag', 0)
        wins_by_depletion = data.get('wins_by_depletion', 0)
        global_step = data.get('global_step', 0)
        
        episodes = len(rewards_p1)
        if episodes == 0:
            return {"Error": "No episodes"}
            
        win_rates = []
        window = min(episodes, 100)
        for i in range(len(wins_hist)):
            start = max(0, i - window + 1)
            win_rates.append(sum(wins_hist[start:i+1]) / (i - start + 1))
            
        max_win_rate = max(win_rates) if win_rates else 0
        final_win_rate = win_rates[-1] if win_rates else 0
        
        valid_losses = [l for l in avg_loss_p1_history[-100:] if l is not None] if avg_loss_p1_history else []
        valid_qs = [q for q in avg_q_values_p1[-100:] if q is not None] if avg_q_values_p1 else []
        
        avg_loss = np.mean(valid_losses) if valid_losses else 0
        avg_q = np.mean(valid_qs) if valid_qs else 0
        avg_reward = np.mean(rewards_p1[-100:]) if rewards_p1 else 0
        
        return {
            "Total Episodes": episodes,
            "Total Steps": global_step,
            "Total Wins": wins_p1,
            "Draws": draws,
            "Wins by Flag / Depletion": f"{wins_by_flag} / {wins_by_depletion}",
            "Overall Win Rate": f"{(wins_p1/episodes*100):.2f}%",
            "Max Moving Win Rate (100-ep)": f"{(max_win_rate*100):.2f}%",
            "Final Moving Win Rate (100-ep)": f"{(final_win_rate*100):.2f}%",
            "Average Loss (last 100-ep)": f"{avg_loss:.4f}",
            "Average Q-Value (last 100-ep)": f"{avg_q:.4f}",
            "Average Reward (last 100-ep)": f"{avg_reward:.4f}"
        }
    except Exception as e:
        return {"Error": str(e)}

f1 = r"c:\Users\Mark Lawrence Quibot\repo\Research\History\LSTMDQN RUN\training_history.json"
f2 = r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\training_history.json"

res1 = process_file(f1)
res2 = process_file(f2)

output = "## Comparison: LSTM DQN vs Rainbow DQN (AAREN)\n\n"
output += "| Metric | LSTM DQN | Rainbow DQN (AAREN) |\n"
output += "|---|---|---|\n"

if 'Error' not in res1 and 'Error' not in res2:
    for key in res1.keys():
        output += f"| {key} | {res1[key]} | {res2[key]} |\n"
else:
    output += f"LSTM DQN Error: {res1}\n"
    output += f"Rainbow DQN Error: {res2}\n"

with open(r'c:\Users\Mark Lawrence Quibot\repo\Research\comparison.md', 'w') as f:
    f.write(output)
print("done")
