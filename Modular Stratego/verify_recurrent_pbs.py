import torch
import numpy as np
from aaren import PieceActionAaren

def verify_recurrent_inference():
    print("🧪 Verifying Recurrent Inference (O(1) vs O(N))...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. Initialize Model
    input_size = 24
    hidden_size = 64
    num_layers = 3
    output_size = 12
    
    model = PieceActionAaren(input_size, hidden_size, num_layers, output_size, device=device).to(device)
    model.eval()
    
    # 2. Generate Random Sequence
    seq_len = 50
    batch_size = 4
    
    # (Batch, Seq, Features)
    sequence = torch.randn(batch_size, seq_len, input_size, device=device)
    
    # 3. Run Parallel Inference (Ground Truth)
    print("   Running Parallel Inference (Full History)...")
    with torch.no_grad():
        parallel_logits = model.forward_parallel(sequence)
        # forward_parallel returns (batch, output_size) - the prediction for the FULL sequence
        final_parallel_probs = torch.softmax(parallel_logits, dim=1)
        
    # 4. Run Sequential Inference (Step-by-Step)
    print("   Running Sequential Inference (Step-by-Step)...")
    
    # Initialize states for each batch element
    # List of [List of Tuples (one per layer)]
    # But forward_sequential expects a list of states where each state is a batch tensor
    # Let's see how I implemented it in drqn_agent.py...
    
    # In drqn_agent.py, I passed `inference_batch_states` which was a list of `hidden_state`.
    # `hidden_state` comes from `prepare_recurrent_update`, which gets it from `piece_hidden_states`.
    # `piece_hidden_states` stores `List[Tuple]`.
    
    # Wait, `forward_sequential` signature:
    # def forward_sequential(self, x_t: torch.Tensor, prev_states: Optional[List[Tuple]] = None):
    
    # If I pass a batch of inputs, I need to pass a batch of states.
    # But `prev_states` in `forward_sequential` seems to expect a list of layer states, 
    # where each layer state is a tuple of Tensors (a, c, m).
    # These tensors should have batch dimension.
    
    # Let's check `forward_sequential` implementation again.
    # It calls `cell.forward(x, prev_state=prev_states[i] if prev_states else None)`
    
    # So `prev_states` is indeed a list of tuples (one per layer).
    # And the tensors inside the tuple must have shape (batch, ...).
    
    # So I need to initialize the state as None for the first step, 
    # and then feed the output state back in.
    
    current_states = None 
    
    with torch.no_grad():
        for t in range(seq_len):
            # Get current step input: (Batch, Input_Size)
            x_t = sequence[:, t, :]
            
            # Run step
            step_probs, new_states = model.forward_sequential(x_t, current_states)
            
            # Update state for next step
            current_states = new_states
            
            # Verify intermediate steps (optional, but good for debugging)
            # parallel_step_probs = parallel_probs[:, t, :]
            # diff = torch.max(torch.abs(step_probs - parallel_step_probs)).item()
            # print(f"   Step {t}: Max Diff = {diff:.6f}")
            
    final_sequential_probs = step_probs
    
    # 5. Compare Results
    print("\n   Comparing Final Outputs...")
    max_diff = torch.max(torch.abs(final_parallel_probs - final_sequential_probs)).item()
    print(f"   Maximum Difference: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("   ✅ SUCCESS: Sequential inference matches Parallel inference!")
    else:
        print("   ❌ FAILURE: Outputs diverge!")
        exit(1)

    # 6. Verify Mixed Batch State Logic (Simulating drqn_agent.py fix)
    print("\n   Verifying Mixed Batch State Logic...")
    
    # Create a batch where:
    # Sample 0: Has state (continuation)
    # Sample 1: New sequence (state is None)
    
    # Run one step to get a state for Sample 0
    x_0 = torch.randn(1, input_size, device=device)
    _, state_0 = model.forward_sequential(x_0, None)
    
    # Sample 1 is new
    x_1 = torch.randn(1, input_size, device=device)
    state_1 = None
    
    # Prepare batch input
    batch_x = torch.cat([x_0, x_1], dim=0) # (2, input_size)
    
    # Prepare batched state manually (logic from drqn_agent.py)
    inference_batch_states = [state_0, state_1]
    
    num_layers = model.num_layers
    batched_states = []
    
    for layer_idx in range(num_layers):
        a_list, c_list, m_list = [], [], []
        for i, sample_state in enumerate(inference_batch_states):
            if sample_state is not None:
                a, c, m = sample_state[layer_idx]
                a_list.append(a)
                c_list.append(c)
                m_list.append(m)
            else:
                # Identity initialization
                a_list.append(torch.zeros(1, hidden_size, device=device))
                c_list.append(torch.zeros(1, 1, device=device))
                m_list.append(torch.full((1, 1), -1e9, device=device))
        
        a_batch = torch.cat(a_list, dim=0)
        c_batch = torch.cat(c_list, dim=0)
        m_batch = torch.cat(m_list, dim=0)
        batched_states.append((a_batch, c_batch, m_batch))
        
    # Run batched inference
    probs_batch, new_states_batch = model.forward_sequential(batch_x, batched_states)
    
    print("   ✅ Mixed batch inference ran without error.")

if __name__ == "__main__":
    verify_recurrent_inference()
