import torch
import sys

def inspect_checkpoint(filepath):
    try:
        print(f"Inspecting: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"Keys: {list(checkpoint.keys())}")
        
        q_dict = checkpoint['q_network_state_dict']
        print("\nQ-Network Shapes:")
        for key in ['conv_in.weight', 'value_conv.weight', 'advantage_conv.weight', 'advantage_out.weight_mu']:
            if key in q_dict:
                print(f"  {key}: {q_dict[key].shape}")
            else:
                print(f"  {key}: NOT FOUND")
        
        if 'history_state_dict' in checkpoint:
            h_dict = checkpoint['history_state_dict']
            print("\nHistory Aggregator Keys:")
            print(f"  {list(h_dict.keys())}")
            
            # Check if it's the aaren parameters directly
            if 'aaren_cells.0.q' in h_dict or 'aaren.aaren_cells.0.q' in h_dict:
                print("  Looks like AAREN parameters are directly in history_state_dict or nested.")
                layer_keys = [k for k in h_dict.keys() if 'aaren_cells' in k]
                if layer_keys:
                    num_cells = len(set([k.split('aaren_cells.')[1].split('.')[0] for k in layer_keys if 'aaren_cells.' in k]))
                    print(f"  Detected AAREN layers: {num_cells}")
            
            for key in h_dict.keys():
                if isinstance(h_dict[key], torch.Tensor):
                    print(f"  {key}: {h_dict[key].shape}")
                else:
                    print(f"  {key}: {type(h_dict[key])}")
        
        if 'step_count' in checkpoint:
            print(f"\nStep Count: {checkpoint['step_count']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        inspect_checkpoint(r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\agent1_rainbow_episode_1000.pth")
