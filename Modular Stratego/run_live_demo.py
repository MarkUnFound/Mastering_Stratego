# run_live_demo.py

import sys
import os

# Add the current directory to the Python path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from live_dqn_demo import run_live_dqn_game, test_flag_deployment, test_bomb_movement

def main():
    """Main function to run the live Stratego demo."""
    print("🎮 Stratego Live Game Demonstration")
    print("=" * 60)
    print("This demo will:")
    print("1. Test flag deployment and labeling")
    print("2. Test bomb movement prevention")
    print("3. Run a live game between two DQN agents")
    print("4. Show full board information to viewers")
    print("5. Restrict agent views to only their pieces")
    print("=" * 60)
    
    try:
        
        print("\n🎯 STARTING LIVE GAME...")
        print("📺 Watch the visualization windows that will open!")
        print("🎮 The game will play automatically with 2-second delays between moves")
        print("⏹️  Press Ctrl+C to stop the demo at any time")
        
        # Run live game with visualization
        run_live_dqn_game(
            num_games=1,
            move_delay=2.0,  # 2 seconds between moves for easy viewing
            show_agent_views=True  # Show both full and restricted views
        )
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
