import torch
from live_environment import LiveStrategoEnvironment

# Create a test environment
device = torch.device('cpu')
env = LiveStrategoEnvironment(device, show_live_view=False, show_agent_views=False)

# Reset the environment to start a new game
env.reset()

# Print initial game state
print(f"Initial game state:")
print(f"  Game over: {env.game_over}")
print(f"  Winner: {env.winner}")
print(f"  Current player: {env.current_player}")
print(f"  P1 flag position: {env.p1_flag_position}")
print(f"  P2 flag position: {env.p2_flag_position}")

# Try to make a move that captures a flag (this would require setting up a specific board state)
# For now, let's just verify that the environment is working
current_player = env.current_player
valid_moves = env.get_valid_moves()

if valid_moves:
    # Make a random valid move
    action = valid_moves[0]
    print(f"\nMaking move: {action}")
    
    # Execute the move
    state, reward, done, info = env.step(action)
    
    print(f"\nAfter move:")
    print(f"  Game over: {done}")
    print(f"  Winner: {env.winner}")
    print(f"  Current player: {env.current_player}")
    print(f"  P1 flag position: {env.p1_flag_position}")
    print(f"  P2 flag position: {env.p2_flag_position}")
    print(f"  Reward: {reward}")
else:
    print("\nNo valid moves available.")

# Clean up
env.close_viewers()
