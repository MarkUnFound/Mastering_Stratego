import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, player_id, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 memory_size=100000, batch_size=64, target_update=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent {player_id} using device: {self.device}")
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training stats
        self.steps_done = 0
        self.losses = []
        self.rewards = []
        self.episode_rewards = []
        
    def select_action(self, state, valid_actions, training=True):
        """Select action using epsilon-greedy policy"""
        if not valid_actions:
            return None
            
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()
        
        # Mask invalid actions
        masked_q_values = torch.full((self.action_size,), float('-inf'), device=self.device)
        for action in valid_actions:
            masked_q_values[action] = q_values[action]
        
        return masked_q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'losses': self.losses,
            'rewards': self.rewards,
            'episode_rewards': self.episode_rewards
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.losses = checkpoint.get('losses', [])
            self.rewards = checkpoint.get('rewards', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_training_mode(self, training=True):
        """Set training mode"""
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'player_id': self.player_id,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'total_episodes': len(self.episode_rewards)
        }
    
    def reset_episode(self):
        """Reset episode statistics"""
        if hasattr(self, 'current_episode_reward'):
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
    
    def add_reward(self, reward):
        """Add reward to current episode"""
        if not hasattr(self, 'current_episode_reward'):
            self.current_episode_reward = 0
        self.current_episode_reward += reward
        self.rewards.append(reward)

class RandomAgent:
    """Random agent for baseline comparison"""
    def __init__(self, player_id):
        self.player_id = player_id
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def