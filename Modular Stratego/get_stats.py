import json

try:
    with open('dqn_models/training_history.json', 'r') as f:
        data = json.load(f)
        
    num_episodes = len(data.get('rewards_p1', [])) or 1
    print(f"Total Episodes Logged: {num_episodes}")
    
    rewards_p1 = data.get('rewards_p1', [])
    aaren_acc = data.get('aaren_accuracy', [])
    print(f"Average P1 Reward (Full): {sum(rewards_p1)/num_episodes:.2f}")
    
    r1k = rewards_p1[-1000:]
    if r1k:
        print(f"Average P1 Reward (Last 1k): {sum(r1k)/len(r1k):.2f}")
    if aaren_acc:
        print(f"Final AAREN Accuracy: {aaren_acc[-1]:.4f}")
    
    w1 = data.get('wins_p1', 0)
    w2 = data.get('wins_p2', 0)
    dr = data.get('draws', 0)
    
    print(f"Total P1 Wins: {w1} ({w1/num_episodes:.1%})")
    print(f"Total P2 Wins: {w2} ({w2/num_episodes:.1%})")
    print(f"Total Draws: {dr} ({dr/num_episodes:.1%})")
    
except Exception as e:
    print(f"Error: {e}")
