
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())

from training_visualizer import plot_additional_metrics

def verify_plotting():
    print("🧪 Verifying Plotting Logic...")
    
    # Mock Data
    episodes = list(range(1, 11))
    
    metrics = {
        'lengths': [100.0 + i*5 + np.random.randn()*10 for i in episodes],
        'pbs_eval1_buffer_sizes': [1000 * i for i in episodes],
        'avg_q_values_p1': [-0.5 + 0.01 * i for i in episodes],
        'avg_entropy_p1': [1.0 - 0.05 * i for i in episodes]
    }
    
    save_path = "test_metrics_plot.png"
    
    try:
        plot_additional_metrics(
            episode_history=episodes,
            episode_lengths={'agent1': metrics['lengths']},
            pbs_buffer_sizes={'agent1': metrics['pbs_eval1_buffer_sizes']},
            avg_q_history={'agent1': metrics['avg_q_values_p1']},
            entropy_history={'agent1': metrics['avg_entropy_p1']},
            save_path=save_path
        )
        
        if os.path.exists(save_path):
            print(f"✅ Plot successfully created at {save_path}")
        else:
            print("❌ Plot execution finished but file not found.")
            
    except Exception as e:
        print(f"❌ Plotting Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_plotting()
