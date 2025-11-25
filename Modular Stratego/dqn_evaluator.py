import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Try to import AarenCell from probabilistic_belief_state, or define a simple version if not found
try:
    from probabilistic_belief_state import AarenCell
except ImportError:
    # Minimal AarenCell implementation if import fails (fallback)
    class AarenCell(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.q = nn.Parameter(torch.randn(hidden_size))
            self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        def forward(self, x_t, prev_state=None):
            k_t = self.W_k(x_t)
            v_t = self.W_v(x_t)
            s_t = torch.sum(self.q.unsqueeze(0) * k_t, dim=1, keepdim=True)
            if prev_state is None:
                a_t = v_t * torch.exp(s_t)
                c_t = torch.exp(s_t)
                m_t = s_t
            else:
                a_prev, c_prev, m_prev = prev_state
                m_t = torch.maximum(m_prev, s_t)
                exp_prev = torch.exp(m_prev - m_t)
                exp_curr = torch.exp(s_t - m_t)
                a_t = a_prev * exp_prev + v_t * exp_curr
                c_t = c_prev * exp_prev + exp_curr
            output = a_t / (c_t + 1e-8)
            return output, (a_t, c_t, m_t)

class StrategoEvaluatorNetwork(nn.Module):
    """
    Deep Q-Network for evaluating Stratego states.
    
    Input Shape: (Batch, 41, 10, 10)
    """
    def __init__(self):
        super(StrategoEvaluatorNetwork, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(41, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Feature vector size after flattening (128 * 10 * 10 = 12800)
        self.feature_size = 128 * 10 * 10
        
        # Value Head (Standard)
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 10 * 10, 256)
        self.value_fc2 = nn.Linear(256, 1) # Scalar Value V(s)

    def forward(self, x):
        # x: (Batch, 41, 10, 10)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def get_value(self, x):
        """Standard forward pass for single state"""
        features = self.forward(x)
        v = F.relu(self.value_bn(self.value_conv(features)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return v

class HistoryAwareDQNEvaluator(nn.Module):
    """
    Evaluator that uses AAREN to process game history.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.cnn = StrategoEvaluatorNetwork()
        
        # AAREN for history
        # Input to AAREN is the flattened feature vector from CNN
        self.feature_dim = 128 * 10 * 10 # 12800
        # Project to smaller dimension for AAREN to save memory/compute
        self.projection = nn.Linear(self.feature_dim, 512)
        self.aaren = AarenCell(hidden_size=512)
        
        # Value head from AAREN output
        self.value_fc = nn.Linear(512, 1)
        
        self.to(device)

    def forward(self, state_sequence: torch.Tensor, prev_aaren_state=None):
        """
        Args:
            state_sequence: (Batch, Seq_Len, 41, 10, 10)
            prev_aaren_state: Previous AAREN state tuple
        """
        batch_size, seq_len, c, h, w = state_sequence.size()
        
        # Process each timestep through CNN
        # Flatten batch and seq dimensions: (Batch*Seq, 41, 10, 10)
        flat_input = state_sequence.view(-1, c, h, w)
        cnn_features = self.cnn(flat_input) # (Batch*Seq, 128, 10, 10)
        
        # Flatten features
        flat_features = cnn_features.view(batch_size * seq_len, -1) # (Batch*Seq, 12800)
        
        # Project
        projected = F.relu(self.projection(flat_features)) # (Batch*Seq, 512)
        
        # Reshape back to sequence
        sequence_features = projected.view(batch_size, seq_len, 512)
        
        # Run AAREN over sequence
        # For simplicity in this forward pass, we just loop or use parallel scan if implemented
        # Here we do sequential for clarity/compatibility with the cell
        
        current_aaren_state = prev_aaren_state
        outputs = []
        
        for t in range(seq_len):
            x_t = sequence_features[:, t, :]
            output, current_aaren_state = self.aaren(x_t, current_aaren_state)
            outputs.append(output)
            
        # Use last output for value
        last_output = outputs[-1]
        value = torch.tanh(self.value_fc(last_output))
        
        return value, current_aaren_state

class DQNEvaluator:
    def __init__(self, model_path: Optional[str] = None, device='cpu', use_history=False):
        self.device = device
        self.use_history = use_history
        
        if use_history:
            self.network = HistoryAwareDQNEvaluator(device=device)
        else:
            self.network = StrategoEvaluatorNetwork().to(device)
            
        if model_path:
            self.network.load_state_dict(torch.load(model_path, map_location=device))
        self.network.eval()

    def evaluate(self, state_tensor: torch.Tensor, history: Optional[torch.Tensor] = None) -> float:
        """
        Evaluate a single state tensor.
        Returns V(s) in range [-1, 1].
        """
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            
            if self.use_history and history is not None:
                # history should be (Seq_Len, 41, 10, 10)
                # Add batch dim
                if history.dim() == 4:
                    history = history.unsqueeze(0)
                # Append current state to history
                if state_tensor.dim() == 3:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(1) # (1, 1, 41, 10, 10)
                elif state_tensor.dim() == 4:
                    state_tensor = state_tensor.unsqueeze(1)
                    
                # Combine history and current state? 
                # Or just use history if it includes current state.
                # Assuming history includes up to t-1, and state_tensor is t
                if history.size(0) == state_tensor.size(0):
                    full_seq = torch.cat([history, state_tensor], dim=1)
                else:
                    full_seq = state_tensor # Fallback
                
                value, _ = self.network(full_seq)
                return value.item()
            else:
                # Standard evaluation
                if isinstance(self.network, HistoryAwareDQNEvaluator):
                    # Treat as seq len 1
                    if state_tensor.dim() == 3:
                        state_tensor = state_tensor.unsqueeze(0).unsqueeze(1)
                    value, _ = self.network(state_tensor)
                    return value.item()
                else:
                    if state_tensor.dim() == 3:
                        state_tensor = state_tensor.unsqueeze(0)
                    value = self.network.get_value(state_tensor)
                    return value.item()

    def evaluate_batch(self, state_tensors: torch.Tensor) -> List[float]:
        """
        Evaluate a batch of states.
        """
        with torch.no_grad():
            state_tensors = state_tensors.to(self.device)
            values = self.network(state_tensors)
            return values.squeeze().tolist()
