import torch
from drqn_agent import RainbowAgent
from environment import StrategoEnvironment

def test_agent():
    print("Testing RainbowAgent with Environment...")
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    try:
        print("Initializing Environment...")
        env = StrategoEnvironment(device=device)
        print("Environment initialized.")
        
        print("Initializing RainbowAgent...")
        agent = RainbowAgent(player_id=1, device=device, num_envs=1)
        print("RainbowAgent instantiated successfully.")
        
        print(f"Agent input channels: {agent.input_channels}")
        print(f"Q-Network input shape: {agent.q_network.input_shape}")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
