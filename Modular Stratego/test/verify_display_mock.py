import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

def verify_display():
    num_episodes = 50
    metrics = {'wins_p1': 0, 'wins_p2': 0, 'rewards_p1': []}
    global_step = 2643000
    
    pbar = tqdm(total=num_episodes, desc="Verification", dynamic_ncols=True)
    
    for i in range(num_episodes):
        time.sleep(0.1)
        metrics['wins_p1'] += np.random.randint(0, 2)
        metrics['wins_p2'] += np.random.randint(0, 2)
        metrics['rewards_p1'].append(np.random.normal(1.8, 0.2))
        global_step += np.random.randint(500, 1500)
        
        recent_reward = np.mean(metrics['rewards_p1'][-10:])
        phase_val = 1
        
        pbar.set_postfix(OrderedDict([
            ('R1', f"{recent_reward:5.2f}"),
            ('W1', f"{metrics['wins_p1']:4d}"),
            ('W2', f"{metrics['wins_p2']:4d}"),
            ('P', f"{phase_val:1d}"),
            ('Step', f"{global_step/1000:6.1f}k")
        ]), refresh=False)
        pbar.update(1)
        
        if i % 10 == 0:
            pbar.write(f"[INFO] Mock log message at step {global_step}")
            
    pbar.close()
    print("Verification complete.")

if __name__ == "__main__":
    verify_display()
