# run_live_demo.py

import sys
import os

# Add the stratego_modular directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stratego_modular'))

from stratego_modular.live_dqn_demo import run_live_dqn_game, test_flag_deployment, test_bomb_movement

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
        # Run comprehensive tests
        print("\n🧪 RUNNING TESTS...")
        test_flag_deployment()
        print()
        test_bomb_movement()
        
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
