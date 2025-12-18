
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

from train_dqn import train_dqn_agents

if __name__ == "__main__":
    print("🚀 Starting Validation Run (5 episodes)...")
    try:
        train_dqn_agents(num_episodes=5, save_interval=5, plot_interval=1, generate_gifs=False, model_save_path="test_dqn_models")
        print("✅ Validation Run Completed Successfully")
    except Exception as e:
        print(f"❌ Validation Run Failed: {e}")
        import traceback
        traceback.print_exc()
