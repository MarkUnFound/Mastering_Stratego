import torch
import numpy as np
import time
import sys
import os
from typing import List, Tuple, Dict, Optional

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drqn_agent import RainbowAgent
from environment import StrategoEnvironment
from piece import PieceType
from board import HIDDEN_PIECE, LAKE_SQUARE

def get_symbol(piece_value: int) -> str:
    """Get a symbol for the piece value."""
    if piece_value == 0: return "."
    if piece_value == LAKE_SQUARE: return "~"
    if piece_value == HIDDEN_PIECE: return "?"
    
    abs_val = abs(piece_value)
    if abs_val == PieceType.FLAG.value: return "F"
    if abs_val == PieceType.BOMB.value: return "B"
    if abs_val == PieceType.SPY.value: return "S"
    if abs_val == PieceType.MARSHAL.value: return "10"
    if abs_val == PieceType.GENERAL.value: return "9"
    if abs_val == PieceType.COLONEL.value: return "8"
    if abs_val == PieceType.MAJOR.value: return "7"
    if abs_val == PieceType.CAPTAIN.value: return "6"
    if abs_val == PieceType.LIEUTENANT.value: return "5"
    if abs_val == PieceType.SERGEANT.value: return "4"
    if abs_val == PieceType.MINER.value: return "3"
    if abs_val == PieceType.SCOUT.value: return "2"
    return str(abs_val)

def print_board(board_tensor, player_id):
    """Print the board state."""
    # Convert tensor to numpy if needed
    if isinstance(board_tensor, torch.Tensor):
        board = board_tensor.cpu().numpy()
    else:
        board = board_tensor
        
    print("   " + " ".join([str(i) for i in range(10)]))
    print("  +" + "-"*20 + "+")
    
    for r in range(10):
        row_str = f"{r} |"
        for c in range(10):
            val = board[r, c]
            # If looking from player -1 perspective, flip signs for display logic if needed
            # But usually board representation is standardized.
            # Let's just print symbols.
            
            # Colorize
            symbol = get_symbol(val)
            if val > 0: # Player 1
                row_str += f"\033[94m{symbol:>2}\033[0m" # Blue
            elif val < 0 and val != LAKE_SQUARE: # Player -1
                row_str += f"\033[91m{symbol:>2}\033[0m" # Red
            else:
                row_str += f"{symbol:>2}"
        row_str += "|"
        print(row_str)
    print("  +" + "-"*20 + "+")

def analyze_moves(agent: RainbowAgent, game_state, valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Dict]:
    """
    Analyze valid moves and return detailed Q-value components.
    Replicates logic from RainbowAgent.act()
    """
    if not valid_moves:
        return []

    # 1. Get State
    state = agent.get_state_representation(game_state)
    
    # 2. Get Uncertainty Map (now implicitly handled via AAREN)
    uncertainty_map = {}

    # 3. Base Q-values (Expected Value for Rainbow)
    # Ensure state is correct shape/device
    if state.dim() == 1:
        state = state.unsqueeze(0)
    elif state.dim() == 3:
        state = state.unsqueeze(0)
        
    agent.q_network.eval()
    with torch.no_grad():
        log_probs = agent.q_network(state)
        probs = log_probs.exp()
        expected_q_values = (probs * agent.support).sum(dim=2) # (1, actions)
        base_q_values = expected_q_values.squeeze(0) # (actions)
    agent.q_network.train()

    # 4. Calculate Scores
    analysis_results = []
    
    for move in valid_moves:
        action_idx = agent._move_to_action_index(move)
        
        # Base Q (from network)
        base_q = base_q_values[action_idx].item()
        
        # Exploration Bonus (Uncertainty)
        uncertainty = agent.get_move_uncertainty(move, uncertainty_map)
        exploration_bonus = uncertainty * agent.uncertainty_exploration_multiplier
        
        # Final Score used for selection
        final_score = base_q + exploration_bonus
        
        analysis_results.append({
            'move': move,
            'base_q': base_q,
            'uncertainty': uncertainty,
            'exploration_bonus': exploration_bonus,
            'final_score': final_score
        })
        
    # Sort by final score descending
    analysis_results.sort(key=lambda x: x['final_score'], reverse=True)
    return analysis_results

def run_visualization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Environment
    env = StrategoEnvironment(device=device)
    
    # Initialize Agents
    agent1 = RainbowAgent(player_id=1, device=device)
    agent2 = RainbowAgent(player_id=-1, device=device)

    # Load Models (Try to load if available)
    model_path = "dqn_models" # Assuming default path
    
    # Try loading Agent 1
    try:
        # Find latest rainbow model
        import glob
        files = glob.glob(os.path.join(model_path, "agent1_rainbow_episode_*.pth"))
        if files:
            latest = max(files, key=os.path.getctime)
            agent1.load_model(latest)
            print(f"Loaded Agent 1 Model: {latest}")
    except Exception as e:
        print(f"Could not load Agent 1 Model: {e}, using random initialization")

    # Try loading Agent 2
    try:
        files = glob.glob(os.path.join(model_path, "agent2_rainbow_episode_*.pth"))
        if files:
            latest = max(files, key=os.path.getctime)
            agent2.load_model(latest)
            print(f"Loaded Agent 2 Model: {latest}")
    except Exception as e:
        print(f"Could not load Agent 2 Model: {e}, using random initialization")

    # Reset Game
    game_state = env.reset()
    done = False
    move_count = 0
    
    print("\n" + "="*50)
    print("Starting Live Q-Value Visualization")
    print("="*50)

    while not done:
        current_player = env.current_player
        current_agent = agent1 if current_player == 1 else agent2
        opponent_agent = agent2 if current_player == 1 else agent1
        
        print(f"\nTurn {move_count + 1}: Player {current_player} ({current_agent.name})")
        
        # Display Board (Visible to current player)
        # Note: env.board.get_visible_board(player_id) returns tensor
        visible_board = env.board.get_visible_board(current_player)
        print_board(visible_board, current_player)
        
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("No valid moves available. Game Over.")
            break

        # Analyze Moves
        print("\nAnalyzing Moves...")
        analysis = analyze_moves(current_agent, game_state, valid_moves)
        
        # Display Top 5 Moves
        print(f"{'From':<10} {'To':<10} {'Base Q':<10} {'Uncert.':<12} {'Expl. Bonus':<12} {'Final Score':<12}")
        print("-" * 70)
        
        for i, item in enumerate(analysis[:5]):
            move = item['move']
            (r1, c1), (r2, c2) = move
            move_str_from = f"({r1},{c1})"
            move_str_to = f"({r2},{c2})"
            
            # Highlight best move
            prefix = ">> " if i == 0 else "   "
            
            print(f"{prefix}{move_str_from:<8} {move_str_to:<8} {item['base_q']:<10.4f} {item['uncertainty']:<12.4f} {item['exploration_bonus']:<12.4f} {item['final_score']:<12.4f}")

        # Agent Action
        # We use the agent's act method to ensure we follow its policy
        
        # Get state for act()
        action = current_agent.act(game_state.board, valid_moves, game_state)
        
        if action is None:
            print("Agent returned None action.")
            break
            
        # Check if action matches top analyzed move
        top_move = analysis[0]['move']
        is_best = (action == top_move)
        
        (r_from, c_from), (r_to, c_to) = action
        print(f"\nSelected Move: ({r_from},{c_from}) -> ({r_to},{c_to})")
        if is_best:
            print("Reason: Highest Value (Exploitation)")
        else:
            print("Reason: Exploration (Noisy Nets / Uncertainty)")

        # Execute Step
        game_state, reward, done, info = env.step(action)
        
        # Update AAREN history for BOTH agents (tracking opponent actions)
        if hasattr(opponent_agent, 'update_history_batch'):
            opponent_agent.update_history_batch([action], [game_state], acting_player=current_player)
        
        move_count += 1
        
        # Wait
        time.sleep(2.0)

    print("\nGame Over!")
    if game_state.winner:
        print(f"Winner: Player {game_state.winner}")
    else:
        print("Draw")

if __name__ == "__main__":
    run_visualization()
