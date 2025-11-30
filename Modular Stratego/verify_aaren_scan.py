
import torch
import torch.nn as nn
import numpy as np
from probabilistic_belief_state import PieceActionAaren, AarenCell

def verify_aaren_scan():
    print("🧪 Verifying AAREN Parallel Prefix Scan...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    batch_size = 4
    seq_len = 16
    input_size = 8
    hidden_size = 16
    output_size = 5
    device = torch.device("cpu") # Test on CPU for simplicity
    
    # Initialize model
    model = PieceActionAaren(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1, # Test single layer first
        output_size=output_size,
        device=device
    )
    model.eval() # Disable dropout
    
    # Create random input
    x = torch.randn(batch_size, seq_len, input_size, device=device)
    
    # 1. Run Parallel Forward Pass
    print("Running Parallel Forward Pass...")
    with torch.no_grad():
        # We need to access the intermediate output before the final linear layer to compare with sequential
        # But forward_parallel returns the final logits.
        # Let's check if we can access the hidden states or just compare final outputs.
        # Comparing final outputs is fine as long as the rest of the network is deterministic.
        parallel_logits = model.forward_parallel(x)
        
    # 2. Run Sequential Forward Pass
    print("Running Sequential Forward Pass...")
    with torch.no_grad():
        # forward_sequential returns (probs, new_states) and takes x_t one by one
        # We need to loop through the sequence
        
        # Initialize states
        prev_states = None
        sequential_logits_list = []
        
        # We need to replicate what forward_parallel does:
        # It projects input to hidden size first: h = self.input_proj(x)
        # Then passes h to the layers.
        # forward_sequential takes x_t (input_size) and projects it inside.
        
        for t in range(seq_len):
            x_t = x[:, t, :] # (batch, input_size)
            
            # forward_sequential returns probs (softmaxed), but forward_parallel returns logits (before softmax)
            # Wait, forward_sequential returns probs = F.softmax(x, dim=1)
            # forward_parallel returns x (logits)
            # This is a discrepancy in the API I should probably fix or account for.
            # Let's look at forward_sequential again.
            
            # In forward_sequential:
            # x = F.relu(self.fc1(h))
            # x = self.dropout(x)
            # x = self.fc2(x)
            # probs = F.softmax(x, dim=1)
            # return probs, new_states
            
            # In forward_parallel:
            # x = F.relu(self.fc1(last_hidden))
            # x = self.dropout(x)
            # x = self.fc2(x)
            # return x
            
            # Ah, forward_parallel returns the sequence of logits?
            # No, forward_parallel returns:
            # return x (which is from last_hidden = h[:, -1, :])
            # So forward_parallel only returns the output for the LAST timestep?
            # Let's check the code I just read.
            pass

    # Re-reading code to confirm behavior
    # forward_parallel:
    # h = self._parallel_prefix_scan(...) -> returns sequence h (batch, seq_len, hidden)
    # last_hidden = h[:, -1, :]
    # ...
    # return x (batch, output_size)
    
    # So forward_parallel returns the logits for the LAST timestep.
    
    # forward_sequential:
    # returns probs, new_states
    
    # So to compare, I should run sequential for the whole sequence, take the logits (before softmax) of the last step
    # But forward_sequential applies softmax.
    # I can modify the test to check the hidden states instead, or just compare the probabilities.
    
    # Let's compare the probabilities.
    parallel_probs = torch.softmax(parallel_logits, dim=1)
    
    # Run sequential
    prev_states = None
    final_sequential_probs = None
    
    for t in range(seq_len):
        x_t = x[:, t, :]
        probs, new_states = model.forward_sequential(x_t, prev_states)
        prev_states = new_states
        final_sequential_probs = probs
        
    # Compare
    print("\nComparing Results...")
    diff = torch.abs(parallel_probs - final_sequential_probs)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Difference: {max_diff:.8f}")
    print(f"Mean Difference: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ Verification PASSED: Parallel and Sequential outputs match!")
    else:
        print("❌ Verification FAILED: Outputs do not match.")
        print("Parallel Probs (First sample):", parallel_probs[0])
        print("Sequential Probs (First sample):", final_sequential_probs[0])

if __name__ == "__main__":
    try:
        verify_aaren_scan()
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
