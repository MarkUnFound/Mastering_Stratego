# stratego_modular/pbs_evaluator.py

"""
PBS Evaluator RL Network
Compares PBS predictions to ground truth and learns to evaluate prediction quality.
Rewards are relative to piece value - closer predictions get more reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple
from .piece import PieceType, PIECE_RANKS

# Number of piece types
NUM_PIECE_TYPES = len(PieceType)

# Experience tuple for PBS evaluation
PBSEvaluationExperience = namedtuple('PBSEvaluationExperience', [
    'pbs_prediction',  # Belief distribution tensor
    'ground_truth',    # Actual piece type
    'position',        # Position tuple
    'game_phase',     # 'middle' or 'end'
    'turn_count'      # Turn number
])


class PBSEvaluatorNetwork(nn.Module):
    """
    Neural network that evaluates PBS prediction quality.
    Takes PBS belief distribution and outputs a quality score.
    """
    
    def __init__(self, input_size: int = NUM_PIECE_TYPES, hidden_size: int = 128):
        """
        Initialize PBS evaluator network.
        
        Args:
            input_size: Size of PBS belief distribution (number of piece types)
            hidden_size: Hidden layer size
        """
        super(PBSEvaluatorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)  # Output: quality score
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, pbs_beliefs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through evaluator network.
        
        Args:
            pbs_beliefs: Tensor of shape (batch_size, NUM_PIECE_TYPES) with belief probabilities
            
        Returns:
            Quality score tensor of shape (batch_size, 1)
        """
        x = F.relu(self.bn1(self.fc1(pbs_beliefs)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # Quality score (can be negative or positive)
        
        return x


class PBSEvaluator:
    """
    RL-based PBS evaluator that learns to assess prediction quality.
    Rewards are computed based on:
    1. Prediction accuracy (how close predicted value is to actual)
    2. Piece value (higher value pieces = more important)
    3. Confidence (higher confidence in correct prediction = more reward)
    """
    
    def __init__(self, device, buffer_size: int = 10000, batch_size: int = 64, lr: float = 0.001):
        """
        Initialize PBS evaluator.
        
        Args:
            device: PyTorch device
            buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            lr: Learning rate
        """
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Evaluator network
        self.evaluator_network = PBSEvaluatorNetwork().to(device)
        self.target_network = PBSEvaluatorNetwork().to(device)
        self.optimizer = torch.optim.AdamW(self.evaluator_network.parameters(), lr=lr, weight_decay=0.01)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Update target network
        self.update_target_network()
        
        # Training statistics
        self.training_losses = []
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.evaluator_network.state_dict())
    
    def compute_reward(self, pbs_prediction: Dict[PieceType, float], 
                      ground_truth: PieceType, piece_value: Optional[int] = None) -> float:
        """
        Compute reward for a PBS prediction based on ground truth.
        
        Reward formula:
        - Base reward: confidence in correct piece type
        - Value multiplier: higher value pieces get more reward/penalty
        - Distance penalty: predictions far from actual value get penalized
        
        Args:
            pbs_prediction: Dictionary mapping PieceType to confidence
            ground_truth: Actual piece type
            piece_value: Optional piece value (rank) for weighting
            
        Returns:
            Reward value (positive for good predictions, negative for bad)
        """
        if piece_value is None:
            piece_value = PIECE_RANKS.get(ground_truth, 1)
        
        # Get confidence in correct piece type
        correct_confidence = pbs_prediction.get(ground_truth, 0.0)
        
        # Compute expected value from prediction
        predicted_value = sum(
            PIECE_RANKS.get(pt, 0) * conf 
            for pt, conf in pbs_prediction.items()
        )
        
        # Actual value
        actual_value = PIECE_RANKS.get(ground_truth, 1)
        
        # Distance between predicted and actual value
        value_distance = abs(predicted_value - actual_value)
        max_distance = 11  # Maximum possible distance (1 to 12)
        normalized_distance = value_distance / max_distance
        
        # Base reward: confidence in correct piece
        base_reward = correct_confidence * 10.0
        
        # Distance penalty: penalize predictions far from actual value
        distance_penalty = -normalized_distance * 5.0
        
        # Value multiplier: higher value pieces are more important
        # Scale by piece value (1-12 range, normalize to 0.5-1.5)
        value_multiplier = 0.5 + (piece_value / 12.0)
        
        # Combined reward
        reward = (base_reward + distance_penalty) * value_multiplier
        
        # Bonus for high confidence in correct prediction
        if correct_confidence > 0.7:
            reward += 2.0 * value_multiplier
        
        # Penalty for very wrong predictions (high confidence in wrong piece)
        max_wrong_confidence = max(
            (conf for pt, conf in pbs_prediction.items() if pt != ground_truth),
            default=0.0
        )
        if max_wrong_confidence > 0.5:
            reward -= 3.0 * value_multiplier
        
        return reward
    
    def remember(self, pbs_prediction: Dict[PieceType, float], ground_truth: PieceType,
                position: Tuple[int, int], game_phase: str, turn_count: int):
        """
        Store PBS evaluation experience.
        
        Args:
            pbs_prediction: PBS belief distribution
            ground_truth: Actual piece type
            position: Position of the piece
            game_phase: 'middle' or 'end'
            turn_count: Current turn number
        """
        # Convert prediction dict to tensor
        piece_types = list(PieceType)
        prediction_tensor = torch.zeros(NUM_PIECE_TYPES, device=self.device, dtype=torch.float32)
        for i, pt in enumerate(piece_types):
            # Convert to Python float first (in case it's a numpy type)
            value = float(pbs_prediction.get(pt, 0.0))
            prediction_tensor[i] = value
        
        experience = PBSEvaluationExperience(
            pbs_prediction=prediction_tensor,
            ground_truth=ground_truth,
            position=position,
            game_phase=game_phase,
            turn_count=turn_count
        )
        
        self.memory.append(experience)
    
    def train(self, epochs: int = 1, use_target_network: bool = True) -> Optional[float]:
        """
        Train the evaluator network on collected experiences using experience replay.
        
        Uses a hybrid approach:
        1. Supervised learning: Compute ground truth rewards from actual piece types
        2. RL-style stability: Use target network for stable target computation (optional)
        
        Args:
            epochs: Number of training epochs
            use_target_network: If True, use target network for more stable training (RL-style)
            
        Returns:
            Average loss value or None if not enough data
        """
        if len(self.memory) < self.batch_size:
            return None
        
        self.evaluator_network.train()
        if use_target_network:
            self.target_network.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Compute number of batches per epoch
        num_samples = len(self.memory)
        num_batches_per_epoch = max(1, num_samples // self.batch_size)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx in range(num_batches_per_epoch):
                # Sample random batch from experience replay buffer
                # This breaks correlation between consecutive experiences
                batch_indices = np.random.choice(num_samples, 
                                                size=min(self.batch_size, num_samples),
                                                replace=False)
                batch = [self.memory[i] for i in batch_indices]
                
                # OPTIMIZATION: Prepare batch tensors more efficiently
                # Stack predictions directly (they should already be tensors on correct device)
                batch_predictions = torch.stack([e.pbs_prediction for e in batch])
                
                # OPTIMIZATION: Compute target rewards using ground truth
                # Batch process rewards to reduce overhead
                ground_truth_rewards = []
                for exp in batch:
                    # OPTIMIZATION: Avoid converting to dict if prediction is already a tensor
                    # Use tensor operations directly when possible
                    if isinstance(exp.pbs_prediction, torch.Tensor):
                        # Extract values more efficiently
                        prediction_dict = {
                            pt: float(exp.pbs_prediction[i].item()) 
                            for i, pt in enumerate(PieceType)
                        }
                    else:
                        prediction_dict = exp.pbs_prediction
                    reward = self.compute_reward(prediction_dict, exp.ground_truth)
                    ground_truth_rewards.append(reward)
                
                # OPTIMIZATION: Create tensor once with proper dtype
                ground_truth_rewards_tensor = torch.tensor(ground_truth_rewards, 
                                                          dtype=torch.float32, 
                                                          device=self.device).unsqueeze(1)
                
                # Forward pass through main network
                predicted_scores = self.evaluator_network(batch_predictions)
                
                # If using target network, blend ground truth with target network predictions
                # This provides more stable training (RL-style)
                if use_target_network:
                    with torch.no_grad():
                        target_scores = self.target_network(batch_predictions)
                    # Blend: 80% ground truth (supervised), 20% target network (RL-style)
                    # This provides stability while still learning from ground truth
                    blended_targets = 0.8 * ground_truth_rewards_tensor + 0.2 * target_scores
                    target_rewards_tensor = blended_targets
                else:
                    # Pure supervised learning: use only ground truth
                    target_rewards_tensor = ground_truth_rewards_tensor
                
                # Compute loss
                loss = F.mse_loss(predicted_scores, target_rewards_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.evaluator_network.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / num_batches_per_epoch
        
        self.evaluator_network.eval()
        
        avg_loss = total_loss / epochs if epochs > 0 else None
        if avg_loss is not None:
            self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate_prediction(self, pbs_prediction: Dict[PieceType, float]) -> float:
        """
        Evaluate a PBS prediction using the trained network.
        
        Args:
            pbs_prediction: PBS belief distribution dictionary
            
        Returns:
            Quality score (higher = better prediction)
        """
        self.evaluator_network.eval()
        
        # Convert prediction dict to tensor
        piece_types = list(PieceType)
        prediction_tensor = torch.zeros(1, NUM_PIECE_TYPES, device=self.device, dtype=torch.float32)
        for i, pt in enumerate(piece_types):
            # Convert to Python float first (in case it's a numpy type)
            value = float(pbs_prediction.get(pt, 0.0))
            prediction_tensor[0, i] = value
        
        with torch.no_grad():
            quality_score = self.evaluator_network(prediction_tensor)
        
        return quality_score.item()
    
    def get_feedback(self, pbs_prediction: Dict[PieceType, float], 
                    ground_truth: Optional[PieceType] = None) -> Dict[str, float]:
        """
        Get feedback on PBS prediction quality.
        
        Args:
            pbs_prediction: PBS belief distribution
            ground_truth: Optional ground truth for computing actual reward
            
        Returns:
            Dictionary with feedback metrics
        """
        # Get network evaluation
        quality_score = self.evaluate_prediction(pbs_prediction)
        
        feedback = {
            'quality_score': quality_score,
            'predicted_confidence': max(pbs_prediction.values()),
            'predicted_piece': max(pbs_prediction.items(), key=lambda x: x[1])[0]
        }
        
        # If ground truth available, compute actual reward
        if ground_truth is not None:
            actual_reward = self.compute_reward(pbs_prediction, ground_truth)
            feedback['actual_reward'] = actual_reward
            feedback['is_correct'] = (feedback['predicted_piece'] == ground_truth)
        
        return feedback
    
    def save_model(self, filepath: str):
        """Save the evaluator model including experience buffer"""
        # Convert experience buffer to serializable format
        # Note: PBSEvaluationExperience contains tensors, need to convert to CPU and detach
        memory_data = []
        for exp in self.memory:
            # Convert tensor to CPU and detach for saving
            pbs_pred_cpu = exp.pbs_prediction.cpu().detach() if isinstance(exp.pbs_prediction, torch.Tensor) else exp.pbs_prediction
            memory_data.append({
                'pbs_prediction': pbs_pred_cpu,
                'ground_truth': exp.ground_truth.value,  # Save enum value
                'position': exp.position,
                'game_phase': exp.game_phase,
                'turn_count': exp.turn_count
            })
        
        torch.save({
            'evaluator_state_dict': self.evaluator_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': memory_data,  # Save experience buffer
            'training_losses': self.training_losses,  # Save training history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the evaluator model including experience buffer"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.evaluator_network.load_state_dict(checkpoint['evaluator_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_target_network()
        
        # Load experience buffer if available
        if 'memory' in checkpoint:
            self.memory.clear()
            for mem_data in checkpoint['memory']:
                # Convert back to PBSEvaluationExperience
                pbs_pred = mem_data['pbs_prediction']
                if isinstance(pbs_pred, torch.Tensor):
                    pbs_pred = pbs_pred.to(self.device)
                else:
                    pbs_pred = torch.tensor(pbs_pred, device=self.device)
                
                # Convert ground truth value back to PieceType
                ground_truth = PieceType(mem_data['ground_truth'])
                
                experience = PBSEvaluationExperience(
                    pbs_prediction=pbs_pred,
                    ground_truth=ground_truth,
                    position=tuple(mem_data['position']),
                    game_phase=mem_data['game_phase'],
                    turn_count=mem_data['turn_count']
                )
                self.memory.append(experience)
            print(f"✅ Loaded {len(self.memory)} experiences into PBS evaluator buffer")
        
        # Load training losses if available
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
    
    def get_average_loss(self, window: int = 100) -> float:
        """Get average training loss over the last N steps"""
        if not self.training_losses:
            return 0.0
        recent_losses = self.training_losses[-window:]
        return sum(recent_losses) / len(recent_losses)

