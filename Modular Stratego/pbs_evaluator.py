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
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple, defaultdict
from piece import PieceType, PIECE_RANKS

# Number of piece types
NUM_PIECE_TYPES = len(PieceType)

# Experience tuple for PBS evaluation
PBSEvaluationExperience = namedtuple('PBSEvaluationExperience', [
    'pbs_prediction',  # Belief distribution tensor
    'ground_truth',    # Actual piece type
    'position',        # Position tuple
    'game_phase',     # 'middle' or 'end'
    'turn_count',     # Turn number
    'q_value'         # Q-value of the action leading to this state (optional)
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


class FeatureImportanceNetwork(nn.Module):
    """
    Learns which action features are most predictive for PBS inference.
    Outputs importance weights for each feature.
    """
    
    def __init__(self, num_features: int = 24, hidden_size: int = 64):
        """
        Initialize feature importance network.
        
        Args:
            num_features: Number of action features (24 in current implementation)
            hidden_size: Hidden layer size
        """
        super(FeatureImportanceNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_features)  # Output: importance weight per feature
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature importance network.
        
        Args:
            features: Tensor of shape (batch_size, num_features) with action features
            
        Returns:
            Importance weights tensor of shape (batch_size, num_features) in [0, 1]
        """
        x = F.relu(self.bn1(self.fc1(features)))
        weights = torch.sigmoid(self.fc2(x))  # [0, 1] per feature
        return weights


class BiasTracker:
    """
    Tracks systematic PBS biases and provides correction factors.
    Identifies when PBS consistently over/under-predicts certain piece types.
    """
    
    def __init__(self):
        """Initialize bias tracker."""
        # Confusion matrix: predicted -> actual -> count
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        # Track overconfidence: predicted_type -> list of confidence values when wrong
        self.overconfidence_by_type = defaultdict(list)
        # Track underconfidence: predicted_type -> list of confidence values when correct but low
        self.underconfidence_by_type = defaultdict(list)
        # Total predictions per type
        self.prediction_counts = defaultdict(int)
        # Total actual occurrences per type
        self.actual_counts = defaultdict(int)
        
    def update(self, predicted: PieceType, actual: PieceType, confidence: float):
        """
        Update bias tracking with a prediction-accuracy pair.
        
        Args:
            predicted: Predicted piece type
            actual: Actual piece type
            confidence: Confidence in prediction
        """
        self.confusion_matrix[predicted][actual] += 1
        self.prediction_counts[predicted] += 1
        self.actual_counts[actual] += 1
        
        if predicted == actual:
            # Correct prediction
            if confidence < 0.5:
                # Underconfident: correct but low confidence
                self.underconfidence_by_type[predicted].append(confidence)
                if len(self.underconfidence_by_type[predicted]) > 1000:
                    self.underconfidence_by_type[predicted].pop(0)
        else:
            # Wrong prediction
            if confidence > 0.5:
                # Overconfident: wrong but high confidence
                self.overconfidence_by_type[predicted].append(confidence)
                if len(self.overconfidence_by_type[predicted]) > 1000:
                    self.overconfidence_by_type[predicted].pop(0)

    
    def get_correction_factor(self, piece_type: PieceType, min_samples: int = 10) -> float:
        """
        Get multiplicative correction factor for a piece type.
        
        Returns:
            Correction factor:
            - < 1.0 if PBS over-predicts this type (reduce probability)
            - > 1.0 if PBS under-predicts this type (increase probability)
            - 1.0 if no bias detected or insufficient data
        """
        # Use .get() for safety in case defaultdict behavior is lost after loading
        pred_count = self.prediction_counts.get(piece_type, 0)
        if pred_count < min_samples:
            return 1.0  # No correction if insufficient data
        
        # Calculate prediction rate vs actual rate
        total_predictions = sum(self.prediction_counts.values())
        total_actuals = sum(self.actual_counts.values())
        
        if total_predictions == 0 or total_actuals == 0:
            return 1.0
        
        predicted_rate = pred_count / total_predictions
        actual_rate = self.actual_counts.get(piece_type, 0) / total_actuals
        
        # Calculate accuracy for this type
        correct = self.confusion_matrix[piece_type][piece_type]
        total_predicted = pred_count
        accuracy = correct / total_predicted if total_predicted > 0 else 0.0
        
        # Correction factor based on:
        # 1. Prediction rate vs actual rate (over/under-prediction)
        # 2. Accuracy (if low accuracy, reduce confidence)
        if actual_rate > 0:
            rate_ratio = predicted_rate / actual_rate
        else:
            rate_ratio = 1.0
        
        # If over-predicting (rate_ratio > 1.2) or low accuracy, reduce
        # If under-predicting (rate_ratio < 0.8) and good accuracy, increase
        if rate_ratio > 1.2 or accuracy < 0.3:
            correction = 0.7 + 0.3 * accuracy  # Reduce: 0.7-1.0 range
        elif rate_ratio < 0.8 and accuracy > 0.5:
            correction = 1.0 + 0.3 * (1.0 - rate_ratio)  # Increase: 1.0-1.3 range
        else:
            correction = 1.0
        
        # Clamp to reasonable range
        return max(0.5, min(1.5, correction))
    
    def get_bias_summary(self) -> Dict[str, float]:
        """
        Get summary of detected biases.
        
        Returns:
            Dictionary mapping piece type names to correction factors
        """
        summary = {}
        for piece_type in PieceType:
            correction = self.get_correction_factor(piece_type)
            if correction != 1.0:
                summary[piece_type.name] = correction
        return summary


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
        
        # Bias tracking for systematic error detection
        self.bias_tracker = BiasTracker()
        
        # Feature importance network for adaptive feature weighting
        self.feature_importance_network = FeatureImportanceNetwork(num_features=24).to(device)
        self.feature_optimizer = torch.optim.AdamW(
            self.feature_importance_network.parameters(), 
            lr=lr * 0.5,  # Slightly lower LR for feature network
            weight_decay=0.01
        )
        
        # Feature importance training data
        self.feature_training_data = deque(maxlen=5000)  # Store (features, quality_score) pairs
        
        # Update target network
        self.update_target_network()
        
        # Training statistics (bounded to prevent memory growth)
        self.training_losses = deque(maxlen=1000)
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.evaluator_network.state_dict())
    
    def compute_reward(self, pbs_prediction: Dict[PieceType, float], 
                      ground_truth: PieceType, piece_value: Optional[int] = None,
                      q_value: Optional[float] = None) -> float:
        """
        Compute reward for a PBS prediction based on ground truth.
        
        Reward formula:
        - Base reward: confidence in correct piece type
        - Value multiplier: higher value pieces get more reward/penalty
        - Distance penalty: predictions far from actual value get penalized
        - Q-value bonus: if action had high Q-value (important move), amplify reward
        
        Args:
            pbs_prediction: Dictionary mapping PieceType to confidence
            ground_truth: Actual piece type
            piece_value: Optional piece value (rank) for weighting
            q_value: Optional Q-value of the action
            
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
            
        # Q-value adjustment: important moves (high Q-value) should have higher stakes
        if q_value is not None:
            # Normalize Q-value (assuming typical range -10 to 10)
            # Sigmoid to get 0-1 importance factor
            importance = 1.0 / (1.0 + math.exp(-abs(q_value) / 5.0))
            # Amplify reward/penalty based on importance
            reward *= (0.5 + importance)
        
        return reward
    
    def remember(self, pbs_prediction: Dict[PieceType, float], ground_truth: PieceType,
                position: Tuple[int, int], game_phase: str, turn_count: int,
                action_features: Optional[np.ndarray] = None,
                q_value: Optional[float] = None):
        """
        Store PBS evaluation experience.
        
        Args:
            pbs_prediction: PBS belief distribution
            ground_truth: Actual piece type
            position: Position of the piece
            game_phase: 'middle' or 'end'
            turn_count: Current turn number
            action_features: Optional action features for feature importance learning
            q_value: Optional Q-value of the action
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
            turn_count=turn_count,
            q_value=q_value
        )
        
        self.memory.append(experience)
        
        # Update bias tracker
        predicted_type = max(pbs_prediction.items(), key=lambda x: x[1])[0]
        confidence = pbs_prediction.get(predicted_type, 0.0)
        self.bias_tracker.update(predicted_type, ground_truth, confidence)
        
        # Store feature data for importance learning
        if action_features is not None:
            # Compute quality score for this prediction
            quality_score = self.compute_reward(pbs_prediction, ground_truth, q_value=q_value)
            self.feature_training_data.append((action_features.copy(), quality_score))
    
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
                
                # Stack predictions directly (they should already be tensors on correct device)
                batch_predictions = torch.stack([e.pbs_prediction for e in batch])
                
                # Batch process rewards to reduce overhead
                ground_truth_rewards = []
                for exp in batch:
                    # Use tensor operations directly when possible
                    if isinstance(exp.pbs_prediction, torch.Tensor):
                        # Extract values more efficiently
                        prediction_dict = {
                            pt: float(exp.pbs_prediction[i].item()) 
                            for i, pt in enumerate(PieceType)
                        }
                    else:
                        prediction_dict = exp.pbs_prediction
                    reward = self.compute_reward(prediction_dict, exp.ground_truth, q_value=exp.q_value)
                    ground_truth_rewards.append(reward)
                
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
        # PBSEvaluationExperience contains tensors, need to convert to CPU and detach
        memory_data = []
        for exp in self.memory:
            # Convert tensor to CPU and detach for saving
            pbs_pred_cpu = exp.pbs_prediction.cpu().detach() if isinstance(exp.pbs_prediction, torch.Tensor) else exp.pbs_prediction
            memory_data.append({
                'pbs_prediction': pbs_pred_cpu,
                'ground_truth': exp.ground_truth.value,  # Save enum value
                'position': exp.position,
                'game_phase': exp.game_phase,
                'turn_count': exp.turn_count,
                'q_value': exp.q_value
            })
        
        # Convert bias tracker to serializable format
        bias_data = {
            'confusion_matrix': {pred.name: {act.name: count for act, count in actuals.items()} 
                                for pred, actuals in self.bias_tracker.confusion_matrix.items()},
            'prediction_counts': {pt.name: count for pt, count in self.bias_tracker.prediction_counts.items()},
            'actual_counts': {pt.name: count for pt, count in self.bias_tracker.actual_counts.items()},
            'overconfidence_by_type': {pt.name: confidences for pt, confidences in self.bias_tracker.overconfidence_by_type.items()},
            'underconfidence_by_type': {pt.name: confidences for pt, confidences in self.bias_tracker.underconfidence_by_type.items()}
        }
        
        # Convert feature training data to CPU
        feature_data = []
        for features, quality in self.feature_training_data:
            if isinstance(features, np.ndarray):
                feature_data.append((features.tolist(), float(quality)))
            else:
                feature_data.append((features, float(quality)))
        
        torch.save({
            'evaluator_state_dict': self.evaluator_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_importance_state_dict': self.feature_importance_network.state_dict(),
            'feature_optimizer_state_dict': self.feature_optimizer.state_dict(),
            'memory': memory_data,  # Save experience buffer
            'training_losses': self.training_losses,  # Save training history
            'bias_tracker': bias_data,  # Save bias tracking data
            'feature_training_data': feature_data,  # Save feature importance training data
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the evaluator model including experience buffer"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.evaluator_network.load_state_dict(checkpoint['evaluator_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load feature importance network if available
        if 'feature_importance_state_dict' in checkpoint:
            self.feature_importance_network.load_state_dict(checkpoint['feature_importance_state_dict'])
        if 'feature_optimizer_state_dict' in checkpoint:
            self.feature_optimizer.load_state_dict(checkpoint['feature_optimizer_state_dict'])
        
        # Load bias tracker if available
        if 'bias_tracker' in checkpoint:
            bias_data = checkpoint['bias_tracker']
            self.bias_tracker.confusion_matrix = defaultdict(lambda: defaultdict(int))
            for pred_name, actuals in bias_data.get('confusion_matrix', {}).items():
                pred_type = PieceType[pred_name]
                for act_name, count in actuals.items():
                    act_type = PieceType[act_name]
                    self.bias_tracker.confusion_matrix[pred_type][act_type] = count
            
            self.bias_tracker.prediction_counts = defaultdict(int, {PieceType[name]: count 
                                                   for name, count in bias_data.get('prediction_counts', {}).items()})
            self.bias_tracker.actual_counts = defaultdict(int, {PieceType[name]: count 
                                              for name, count in bias_data.get('actual_counts', {}).items()})
            self.bias_tracker.overconfidence_by_type = defaultdict(list, {PieceType[name]: confidences 
                                                        for name, confidences in bias_data.get('overconfidence_by_type', {}).items()})
            self.bias_tracker.underconfidence_by_type = defaultdict(list, {PieceType[name]: confidences 
                                                         for name, confidences in bias_data.get('underconfidence_by_type', {}).items()})
        
        # Load feature training data if available
        if 'feature_training_data' in checkpoint:
            self.feature_training_data.clear()
            for features, quality in checkpoint['feature_training_data']:
                if isinstance(features, list):
                    features = np.array(features)
                self.feature_training_data.append((features, quality))
        
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
                    turn_count=mem_data['turn_count'],
                    q_value=mem_data.get('q_value')  # Handle backward compatibility
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
    
    def get_bias_correction(self, piece_type: PieceType) -> float:
        """
        Get bias correction factor for a piece type.
        
        Args:
            piece_type: Piece type to get correction for
            
        Returns:
            Correction factor (multiply belief by this value)
        """
        return self.bias_tracker.get_correction_factor(piece_type)
    
    def get_feature_importance(self, action_features: np.ndarray) -> np.ndarray:
        """
        Get importance weights for action features.
        
        Args:
            action_features: Array of 24 action features
            
        Returns:
            Array of importance weights (same shape as input)
        """
        self.feature_importance_network.eval()
        
        # Convert to tensor
        features_tensor = torch.tensor(
            action_features.reshape(1, -1), 
            device=self.device, 
            dtype=torch.float32
        )
        
        with torch.no_grad():
            importance_weights = self.feature_importance_network(features_tensor)
        
        # Convert back to numpy
        return importance_weights.cpu().numpy().flatten()
    
    def train_feature_importance(self, epochs: int = 1):
        """
        Train feature importance network on collected feature-quality pairs.
        
        Args:
            epochs: Number of training epochs
        """
        if len(self.feature_training_data) < self.batch_size:
            return
        
        self.feature_importance_network.train()
        
        # Prepare training data
        features_list = [data[0] for data in self.feature_training_data]
        quality_scores = [data[1] for data in self.feature_training_data]
        
        # Normalize quality scores to [0, 1] for training
        quality_array = np.array(quality_scores)
        if quality_array.max() > quality_array.min():
            quality_normalized = (quality_array - quality_array.min()) / (quality_array.max() - quality_array.min())
        else:
            quality_normalized = np.ones_like(quality_array) * 0.5
        
        num_batches = max(1, len(self.feature_training_data) // self.batch_size)
        
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                # Sample random batch
                batch_indices = np.random.choice(
                    len(self.feature_training_data),
                    size=min(self.batch_size, len(self.feature_training_data)),
                    replace=False
                )
                
                batch_features = torch.tensor(
                    np.array([features_list[i] for i in batch_indices]),
                    device=self.device,
                    dtype=torch.float32
                )
                batch_quality = torch.tensor(
                    quality_normalized[batch_indices],
                    device=self.device,
                    dtype=torch.float32
                )
                
                # Forward pass
                importance_weights = self.feature_importance_network(batch_features)
                
                # Loss: features with higher quality should have higher importance
                # Use weighted features to predict quality (simplified approach)
                weighted_features = batch_features * importance_weights
                # Simple prediction: sum of weighted features should correlate with quality
                # REMOVED SIGMOID: Sigmoid clamps to [0.5, 1.0] for positive inputs
                predicted_quality = weighted_features.sum(dim=1)
                
                # MSE loss between predicted and actual quality
                loss = F.mse_loss(predicted_quality, batch_quality)
                
                # Backward pass
                self.feature_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_importance_network.parameters(), max_norm=1.0)
                self.feature_optimizer.step()
        
        self.feature_importance_network.eval()
    
    def should_gather_more_info(self, pbs_prediction: Dict[PieceType, float], 
                                uncertainty_threshold: float = 0.7,
                                quality_threshold: float = -5.0) -> bool:
        """
        Determine if more information should be gathered for this prediction (active learning).
        
        Args:
            pbs_prediction: PBS belief distribution
            uncertainty_threshold: High uncertainty threshold (0-1)
            quality_threshold: Low quality score threshold
            
        Returns:
            True if more information gathering is recommended
        """
        # Calculate entropy (uncertainty)
        probs = [p for p in pbs_prediction.values() if p > 0]
        if not probs:
            return True  # No prediction, need info
        
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(pbs_prediction))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Get quality score
        quality_score = self.evaluate_prediction(pbs_prediction)
        
        # Request more info if quality is poor AND uncertainty is high
        return quality_score < quality_threshold and normalized_entropy > uncertainty_threshold

