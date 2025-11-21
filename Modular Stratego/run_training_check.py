
from train_dqn import train_dqn_agents
import os

def run_check():
    print("Running training check for 1 episode...")
    try:
        # Create a temporary directory for models to avoid messing up real training if any
        model_save_path = "dqn_models_check"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            
        train_dqn_agents(num_episodes=1, save_interval=1, model_save_path=model_save_path, generate_gifs=False)
        print("Training check completed successfully.")
        
        # Check if files were created
        files = os.listdir(model_save_path)
        print(f"Files created in {model_save_path}: {files}")
        
    except Exception as e:
        print(f"Error during training check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_check()
