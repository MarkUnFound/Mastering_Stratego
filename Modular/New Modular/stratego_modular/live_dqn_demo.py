# stratego_modular/live_dqn_demo.py

import torch
import random
import numpy as np
from typing import List, Tuple
from .live_environment import LiveStrategoEnvironment
from .piece import PieceType
from .dqn_agent import DQNAgent
import time

# MockDQNAgent class has been replaced with real DQN agents
# See dqn_agent.py for the DQNAgent implementation


def run_live_dqn_game(num_games: int = 1, move_delay: float = 1.0, 
                      show_agent_views: bool = False):
    """Run live DQN games with visualization."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agents
    agent1 = DQNAgent(1, device)
    agent2 = DQNAgent(-1, device)
    
    # Load trained models if available
    try:
        agent1.load_model("dqn_models/agent1_final.pth")
        agent2.load_model("dqn_models/agent2_final.pth")
        print("✅ Loaded trained DQN models")
    except FileNotFoundError:
        print("⚠️  No trained models found, using untrained DQN agents")
    except Exception as e:
        print(f"⚠️  Error loading models: {e}, using untrained DQN agents")
    
    print(f"\n🎮 Starting {num_games} live Stratego game{'s' if num_games > 1 else ''}")
    print(f"🤖 {agent1.name} vs {agent2.name}")
    print(f"⏱️  Move delay: {move_delay} seconds")
    print("🔍 Live viewer shows full board information")
    if show_agent_views:
        print("👁️  Agent views show restricted information")
    print("\n" + "="*50)
    
    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}
    
    for game_num in range(num_games):
        print(f"\n🎯 Game {game_num + 1}/{num_games}")
        
        # Create environment with live visualization
        env = LiveStrategoEnvironment(
            device=device, 
            show_live_view=True, 
            show_agent_views=show_agent_views
        )
        
        try:
            game_state = env.reset()
            move_count = 0
            max_moves = 500  # Prevent infinite games
            
            print(f"🚀 Game started! Watch the live visualization window.")
            print(f"🏁 Flag positions are marked with 'F' and colored distinctly")
            
            # Game loop
            while not game_state.game_over and move_count < max_moves:
                current_agent = agent1 if env.current_player == 1 else agent2
                valid_moves = env.get_valid_moves()
                
                if not valid_moves:
                    print(f"❌ No valid moves for {current_agent.name}")
                    break
                
                # Agent selects action
                action = current_agent.act(game_state, valid_moves)
                
                if action is None:
                    print(f"❌ {current_agent.name} returned invalid action")
                    break
                
                # Execute move
                (r_from, c_from), (r_to, c_to) = action
                print(f"🎯 Turn {move_count + 1}: {current_agent.name} moves from ({r_from},{c_from}) to ({r_to},{c_to})")
                
                game_state, reward, done, info = env.step(action)
                move_count += 1
                
                # Pause for visualization
                env.pause_for_viewing(move_delay)
                
                if done:
                    break
            
            # Game finished
            if game_state.winner == 1:
                print(f"🏆 {agent1.name} wins!")
                results["agent1_wins"] += 1
            elif game_state.winner == -1:
                print(f"🏆 {agent2.name} wins!")
                results["agent2_wins"] += 1
            else:
                print(f"🤝 Game ended in a draw!")
                results["draws"] += 1
                
            print(f"📊 Game lasted {move_count} moves")
            
            if game_num < num_games - 1:
                print(f"\n⏳ Starting next game in 3 seconds...")
                time.sleep(3)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Game interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error during game: {e}")
        finally:
            env.close_viewers()
    
    # Final results
    print(f"\n" + "="*50)
    print(f"📈 FINAL RESULTS ({num_games} games)")
    print(f"🤖 {agent1.name}: {results['agent1_wins']} wins")
    print(f"🤖 {agent2.name}: {results['agent2_wins']} wins")
    print(f"🤝 Draws: {results['draws']}")
    print(f"="*50)


def test_flag_deployment():
    """Test that flags are properly deployed and labeled."""
    print("🧪 Testing flag deployment and labeling...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LiveStrategoEnvironment(device=device, show_live_view=True, show_agent_views=False)
    
    game_state = env.reset()
    
    # Check that flags exist using tracked positions
    flag_positions = []
    
    # Check Player 1 flag
    if env.p1_flag_position is not None:
        r, c = env.p1_flag_position
        # Verify the position is valid
        if env.board.actual_board[r, c].item() == PieceType.FLAG.value:
            flag_positions.append((r, c, PieceType.FLAG.value))
        else:
            # Invalid flag position, clear it
            env.p1_flag_position = None
    
    # Check Player 2 flag
    if env.p2_flag_position is not None:
        r, c = env.p2_flag_position
        # Verify the position is valid
        if env.board.actual_board[r, c].item() == -PieceType.FLAG.value:
            flag_positions.append((r, c, -PieceType.FLAG.value))
        else:
            # Invalid flag position, clear it
            env.p2_flag_position = None
    
    print(f"🏁 Found {len(flag_positions)} flags on the board:")
    for r, c, value in flag_positions:
        player = "Player 1" if value > 0 else "Player 2"
        print(f"   - {player} flag at position ({r}, {c})")
    
    if len(flag_positions) == 2:
        print("✅ Flag deployment test PASSED - Both flags are deployed")
    else:
        print("❌ Flag deployment test FAILED - Expected 2 flags")
    
    # Keep visualization open briefly so user can see
    print("🔍 Check the live visualization window - flags should be labeled 'F' and colored distinctly")
    time.sleep(10)  # Keep window open for 10 seconds
    print("✅ Flag deployment test completed")
    
    env.close_viewers()
    print("✅ Flag deployment test completed")


def test_bomb_movement():
    """Test that bombs cannot be moved."""
    print("🧪 Testing bomb movement prevention...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LiveStrategoEnvironment(device=device, show_live_view=False, show_agent_views=False)
    
    game_state = env.reset()
    
    # Find valid moves and check if any involve bombs
    valid_moves = env.get_valid_moves()
    bomb_moves = []
    
    for move in valid_moves:
        (r_from, c_from), (r_to, c_to) = move
        piece_value = env.board.actual_board[r_from, c_from].item()
        if abs(piece_value) == PieceType.BOMB.value:
            bomb_moves.append(move)
    
    if len(bomb_moves) == 0:
        print("✅ Bomb movement test PASSED - No bomb moves found in valid moves")
    else:
        print(f"❌ Bomb movement test FAILED - Found {len(bomb_moves)} bomb moves:")
        for move in bomb_moves:
            print(f"   - Bomb move: {move}")
    
    print("✅ Bomb movement test completed")


if __name__ == "__main__":
    print("🎮 Stratego Live DQN Demo")
    print("=" * 50)
    
    # Run tests first
    test_flag_deployment()
    print()
    test_bomb_movement()
    print()
    
    # Run live game
    try:
        run_live_dqn_game(
            num_games=1, 
            move_delay=2.0,  # 2 second delay between moves
            show_agent_views=True  # Show both full and restricted views
        )
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
