import json
import numpy as np

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        rewards_p1 = data.get('rewards_p1', [])
        episodes = len(rewards_p1)
        if episodes == 0:
            return {"Error": "No episodes"}
            
        wins_p1 = data.get('wins_p1', sum(data.get('wins_p1_history', [])))
        if isinstance(wins_p1, list): wins_p1 = sum(wins_p1)
        
        draws = data.get('draws', 0)
        wins_by_flag = data.get('wins_by_flag', 0)
        wins_by_depletion = data.get('wins_by_depletion', 0)
        global_step = data.get('global_step', 0)
        
        avg_loss = np.mean([l for l in data.get('avg_loss_p1_history', [])[-100:] if l is not None] or [0])
        avg_q = np.mean([q for q in data.get('avg_q_values_p1', [])[-100:] if q is not None] or [0])
        avg_reward = np.mean(rewards_p1[-100:] or [0])
        
        avg_entropy = np.mean([e for e in data.get('avg_entropy_p1', [])[-100:] if e is not None] or [0])
        dqn_grad_norm = np.mean([g for g in data.get('dqn_grad_norm', [])[-100:] if g is not None] or [0])
        
        aaren_acc = np.mean([a for a in data.get('aaren_accuracy', [])[-100:] if a is not None] or [0])
        aaren_loss = np.mean([l for l in data.get('aaren_loss', [])[-100:] if l is not None] or [0])
        aaren_embed_std = np.mean([s for s in data.get('aaren_embedding_std', [])[-100:] if s is not None] or [0])
        
        early_reward = np.mean(rewards_p1[:1000] or [0])
        late_reward = np.mean(rewards_p1[-1000:] or [0])
        
        losses_by_flag = data.get('losses_by_flag', 0)
        losses_by_depletion = data.get('losses_by_depletion', 0)
        total_losses = losses_by_flag + losses_by_depletion
        
        return {
            "Total Episodes": episodes,
            "Total Steps": global_step,
            "Average Steps/Episode": f"{(global_step / episodes):.1f}" if episodes else "0",
            "Total Wins": wins_p1,
            "Total Losses": total_losses,
            "Draws": draws,
            "Overall Win Rate": f"{(wins_p1/episodes*100):.2f}%",
            "Overall Draw Rate": f"{(draws/episodes*100):.2f}%",
            "Wins by Flag / Depletion": f"{wins_by_flag} / {wins_by_depletion}",
            "Losses by Flag / Depletion": f"{losses_by_flag} / {losses_by_depletion}",
            "Average Loss (last 100)": f"{avg_loss:.4f}",
            "Average Q-Value (last 100)": f"{avg_q:.4f}",
            "Average Reward (last 100)": f"{avg_reward:.4f}",
            "Early Avg Reward (first 1K)": f"{early_reward:.4f}",
            "Late Avg Reward (last 1K)": f"{late_reward:.4f}",
            "Avg Entropy (last 100)": f"{avg_entropy:.4f}",
            "DQN Grad Norm (last 100)": f"{dqn_grad_norm:.4f}",
            "AAREN Accuracy (last 100)": f"{aaren_acc:.4f}",
            "AAREN Loss (last 100)": f"{aaren_loss:.4f}",
            "AAREN Embed Std (last 100)": f"{aaren_embed_std:.4f}",
        }
    except Exception as e:
        return {"Error": str(e)}

f1 = r"c:\Users\Mark Lawrence Quibot\repo\Research\History\LSTMDQN RUN\training_history.json"
f2 = r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\training_history.json"

res1 = process_file(f1)
res2 = process_file(f2)

output = "## In-Depth Architectural Comparison: LSTM vs Rainbow+AAREN\n\n"
output += "| Metric | Vanilla DQN + LSTM | Rainbow DQN + AAREN |\n"
output += "|---|---|---|\n"

if 'Error' not in res1 and 'Error' not in res2:
    for key in res1.keys():
        output += f"| **{key}** | {res1[key]} | {res2[key]} |\n"
else:
    output += f"LSTM Error: {res1}\n"
    output += f"Rainbow Error: {res2}\n"

with open(r'c:\Users\Mark Lawrence Quibot\repo\Research\deep_comparison.md', 'w') as f:
    f.write(output)
print("done deep comparison")
