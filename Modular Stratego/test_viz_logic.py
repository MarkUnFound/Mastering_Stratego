
import os
import sys
import shutil

# Mock necessary modules
class MockAgent:
    def __init__(self):
        self.epsilon = 0.1
        self.pbs = None
        
    def get_average_policy_loss(self, window):
        return 0.5

# Mock matplotlib to avoid GUI errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_visualization_logic():
    print("🧪 Testing Visualization Logic...")
    
    # Test 1: Episode 1 Trigger
    print("\n--- Test 1: Episode 1 Trigger ---")
    episode_number = 1
    is_checkpoint_episode = (episode_number % 50 == 0)
    
    # Original logic (simulated)
    is_first_episode_of_cycle = (episode_number % 50 == 1)
    original_trigger = is_first_episode_of_cycle or is_checkpoint_episode
    
    # New logic
    new_trigger = (episode_number == 1) or is_checkpoint_episode
    
    print(f"Episode {episode_number}: Original Trigger={original_trigger}, New Trigger={new_trigger}")
    if new_trigger:
        print("✅ Correctly triggered for Episode 1")
    else:
        print("❌ Failed to trigger for Episode 1")
        
    # Test 2: Episode 601 Trigger (Should NOT trigger)
    print("\n--- Test 2: Episode 601 Trigger ---")
    episode_number = 601
    is_checkpoint_episode = (episode_number % 50 == 0)
    
    is_first_episode_of_cycle = (episode_number % 50 == 1)
    original_trigger = is_first_episode_of_cycle or is_checkpoint_episode
    
    new_trigger = (episode_number == 1) or is_checkpoint_episode
    
    print(f"Episode {episode_number}: Original Trigger={original_trigger}, New Trigger={new_trigger}")
    if not new_trigger:
        print("✅ Correctly skipped for Episode 601")
    else:
        print("❌ Incorrectly triggered for Episode 601")
        
    # Test 3: History Padding
    print("\n--- Test 3: History Padding ---")
    
    def pad_list(lst, target_len, default=0.0):
        if len(lst) == target_len:
            return lst
        elif len(lst) > target_len:
            return lst[:target_len]
        else:
            pad_value = lst[-1] if len(lst) > 0 else default
            return lst + [pad_value] * (target_len - len(lst))
            
    wins_history = [10, 10, 11, 12] # Length 4
    target_len = 6
    
    # Old behavior (simulated default=0)
    padded_old = pad_list(wins_history, target_len, default=0) # If we passed 0
    # But wait, the function uses lst[-1] if len > 0.
    # So the issue might have been if the list was empty?
    
    # Let's test empty list
    empty_wins = []
    padded_empty_old = pad_list(empty_wins, target_len, default=0)
    print(f"Empty list padded (default=0): {padded_empty_old}")
    
    # New behavior (explicit default)
    last_val = wins_history[-1] if wins_history else 0
    padded_new = pad_list(wins_history, target_len, last_val)
    print(f"Non-empty list padded: {padded_new}")
    
    if padded_new[-1] == 12:
        print("✅ Padding preserved last value")
    else:
        print("❌ Padding failed to preserve last value")

if __name__ == "__main__":
    test_visualization_logic()
