"""
DQN Agent Move Visualizer for Stratego Game
"""

import torch
# Set matplotlib backend to non-interactive (thread-safe)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from .game_state import GameState

# Try to import imageio for GIF creation
try:
    import imageio
    GIF_AVAILABLE = True
except ImportError:
    GIF_AVAILABLE = False
    print("imageio not available. GIF creation will be disabled.")


class DQNMoveVisualizer:
    """Visualizer for DQN agent moves in Stratego game."""
    
    def __init__(self):
        self.move_history = []
        self.game_states = []
        self.move_counts = {}  # Track move frequencies for penalty calculation
        
    def record_move(self, action: Tuple[Tuple[int, int], Tuple[int, int]], 
                   game_state: GameState, player: int):
        """Record a move made by a DQN agent."""
        move_key = (action, player)
        if move_key in self.move_counts:
            self.move_counts[move_key] += 1
        else:
            self.move_counts[move_key] = 1
            
        # For visualization, we want to show the actual board state with all pieces revealed
        # We'll store both the player view and the actual board
        self.move_history.append({
            'action': action,
            'player': player,
            'turn': game_state.turn_count,
            'board_state': game_state.board.clone(),
            'actual_board': getattr(game_state, 'actual_board', None)  # Will be set by environment
        })
        
    def record_game_state(self, game_state: GameState):
        """Record the current game state."""
        self.game_states.append(game_state)
        
    def get_move_penalty(self, action: Tuple[Tuple[int, int], Tuple[int, int]], player: int) -> float:
        """Calculate penalty for repeating moves."""
        move_key = (action, player)
        count = self.move_counts.get(move_key, 0)
        # Exponential penalty for repeated moves
        return -0.05 if count > 1 else 0.0
        
    def visualize_move(self, move_index: int, save_path: Optional[str] = None):
        """Visualize a specific move from the recorded history."""
        if not self.move_history or move_index >= len(self.move_history):
            print(f"Move index {move_index} not found in history.")
            return
            
        move_data = self.move_history[move_index]
        board = move_data['board_state']
        action = move_data['action']
        player = move_data['player']
        turn = move_data['turn']
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Display board with piece values
        board_np = board.cpu().numpy()
        im = ax.imshow(board_np, cmap='RdYlBu', vmin=-12, vmax=12)
        
        # Add piece value numbers to each square
        for r in range(10):
            for c in range(10):
                piece_value = int(board_np[r, c])
                if piece_value != 0:  # Not empty square
                    # Determine text color based on background
                    text_color = 'white' if abs(piece_value) > 6 else 'black'
                    ax.text(c, r, str(abs(piece_value)), 
                           ha='center', va='center', 
                           fontsize=14, fontweight='bold',
                           color=text_color)
        
        # Highlight move
        from_pos, to_pos = action
        from_r, from_c = from_pos
        to_r, to_c = to_pos
        
        # Mark start position
        ax.scatter(from_c, from_r, c='green', s=300, marker='s', alpha=0.7, label='Start', edgecolors='black', linewidth=2)
        
        # Mark end position
        ax.scatter(to_c, to_r, c='red', s=300, marker='s', alpha=0.7, label='End', edgecolors='black', linewidth=2)
        
        # Draw arrow showing move direction
        ax.arrow(from_c, from_r, to_c-from_c, to_r-from_r, 
                head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=3)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 10, 1))
        ax.set_yticks(np.arange(-0.5, 10, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linewidth=2)
        
        # Add title
        ax.set_title(f'Player {player} Move at Turn {turn}\nFrom {from_pos} to {to_pos}', fontsize=16, fontweight='bold')
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Piece Values', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Move visualization saved to {save_path}")
        # Note: plt.show() is disabled when using 'Agg' backend (non-interactive, thread-safe)
        # Plots are saved to files instead
        
        plt.close()
        
    def create_move_gif(self, save_path: str = "game_moves.gif", duration: float = 0.5):
        """Create a GIF animation of all recorded moves."""
        if not GIF_AVAILABLE:
            print("GIF creation is not available. Please install imageio.")
            return
            
        if not self.move_history:
            print("No moves recorded yet.")
            return
            
        # Create temporary directory for frames
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        
        try:
            frame_paths = []
            
            for i, move_data in enumerate(self.move_history):
                board = move_data['board_state']
                action = move_data['action']
                player = move_data['player']
                turn = move_data['turn']
                
                # Create visualization
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                
                # Display board with piece values
                # Use actual board if available, otherwise use the player view
                if 'actual_board' in move_data and move_data['actual_board'] is not None:
                    board_to_show = move_data['actual_board']
                else:
                    board_to_show = board
                    
                board_np = board_to_show.cpu().numpy()
                im = ax.imshow(board_np, cmap='RdYlBu', vmin=-12, vmax=12)
                
                # Add piece value numbers to each square with proper labeling
                for r in range(10):
                    for c in range(10):
                        piece_value = int(board_np[r, c])
                        if piece_value != 0:  # Not empty square
                            # Special handling for lakes
                            if piece_value == -2:  # Lake square
                                ax.text(c, r, '~~', ha='center', va='center', 
                                       fontsize=10, fontweight='bold', color='blue')
                            else:
                                # Determine ownership and piece type
                                owner = 1 if piece_value > 0 else -1
                                piece_type_value = abs(piece_value)
                                
                                # Get proper piece label
                                from .piece import PieceType, PIECE_NAMES
                                try:
                                    piece_type = PieceType(piece_type_value)
                                    piece_label = PIECE_NAMES.get(piece_type, str(piece_type_value))
                                except (ValueError, KeyError):
                                    piece_label = str(piece_type_value)
                                
                                # Determine text color based on owner
                                # Player -1 (blue) pieces get white text, Player 1 (red) pieces get appropriate text color
                                text_color = 'white' if owner == -1 else ('white' if abs(piece_value) > 6 else 'black')
                                
                                # Add owner indicator for clarity
                                display_text = f'{piece_label}' if piece_type_value != 0 else ''
                                
                                ax.text(c, r, display_text, 
                                       ha='center', va='center', 
                                       fontsize=12, fontweight='bold',
                                       color=text_color)
                
                # Highlight move
                from_pos, to_pos = action
                from_r, from_c = from_pos
                to_r, to_c = to_pos
                
                # Mark start position
                ax.scatter(from_c, from_r, c='green', s=300, marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                
                # Mark end position
                ax.scatter(to_c, to_r, c='red', s=300, marker='s', alpha=0.7, edgecolors='black', linewidth=2)
                
                # Draw arrow showing move direction
                ax.arrow(from_c, from_r, to_c-from_c, to_r-from_r, 
                        head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=3)
                
                # Add grid
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.grid(True, color='black', linewidth=1)
                
                # Set title
                ax.set_title(f"Move {i+1}: Player {player} Turn {turn}", fontsize=14, pad=20)
                
                # Save frame as temporary image
                frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                frame_paths.append(frame_path)
                
                plt.close(fig)
                
            # Create GIF from frames
            if frame_paths:
                images = [imageio.imread(path) for path in frame_paths]
                imageio.mimsave(save_path, images, duration=duration)
                print(f"GIF saved to {save_path}")
            else:
                print("No frames to create GIF.")
                
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
        
    def visualize_game_sequence(self, start_index: int = 0, end_index: Optional[int] = None,
                              save_dir: Optional[str] = None):
        """Visualize a sequence of moves from the game."""
        if end_index is None:
            end_index = len(self.move_history)
            
        if start_index >= len(self.move_history) or start_index >= end_index:
            print("Invalid indices for visualization sequence.")
            return
            
        for i in range(start_index, min(end_index, len(self.move_history))):
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/move_{i}.png"
            self.visualize_move(i, save_path)
            
    def print_move_history(self):
        """Print a summary of recorded moves with improved formatting."""
        if not self.move_history:
            print("No moves recorded yet.")
            return
            
        print(f"Total moves recorded: {len(self.move_history)}")
        print("Move History:")
        print("-" * 50)
        for i, move in enumerate(self.move_history):
            action = move['action']
            player = move['player']
            turn = move['turn']
            move_key = (action, player)
            count = self.move_counts.get(move_key, 1)
            repeat_indicator = f" (x{count})" if count > 1 else ""
            print(f"Move {i:2d}: Player {player} at turn {turn:3d} - {action[0]} to {action[1]}{repeat_indicator}")
        print("-" * 50)
            
    def clear_history(self):
        """Clear the recorded move history."""
        self.move_history.clear()
        self.game_states.clear()
        self.move_counts.clear()
