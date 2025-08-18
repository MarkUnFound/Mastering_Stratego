# stratego_modular/live_game_viewer.py

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple, Optional, Dict
from .piece import PieceType, PIECE_NAMES
from .board import BOARD_SIZE, EMPTY_SQUARE, LAKE_SQUARE, HIDDEN_PIECE
import threading
import time

class LiveGameViewer:
    """Live viewer for Stratego games showing full board information to spectators."""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.current_board = None
        self.current_player = 1
        self.turn_count = 0
        self.last_move = None
        self.game_over = False
        self.winner = None
        self.move_history = []
        
        # Colors for visualization
        self.colors = {
            'empty': 'white',
            'lake': 'lightblue',
            'p1_piece': 'lightcoral',
            'p2_piece': 'lightgreen',
            'p1_flag': 'red',
            'p2_flag': 'darkgreen',
            'p1_bomb': 'orange',
            'p2_bomb': 'darkorange',
            'highlight': 'yellow',
            'grid': 'black'
        }
        
        self.setup_board_display()
        
    def setup_board_display(self):
        """Setup the initial board display."""
        self.ax.clear()
        self.ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left
        
        # Add grid
        for i in range(BOARD_SIZE + 1):
            self.ax.axhline(i - 0.5, color=self.colors['grid'], linewidth=1)
            self.ax.axvline(i - 0.5, color=self.colors['grid'], linewidth=1)
        
        # Add coordinate labels
        self.ax.set_xticks(range(BOARD_SIZE))
        self.ax.set_yticks(range(BOARD_SIZE))
        self.ax.set_xlabel('Column')
        self.ax.set_ylabel('Row')
        
        # Add title
        self.ax.set_title('Stratego Live Game - Full Information View', fontsize=16, fontweight='bold')
        
    def update_display(self, actual_board: torch.Tensor, current_player: int, 
                      turn_count: int, last_move: Optional[Tuple] = None,
                      game_over: bool = False, winner: Optional[int] = None,
                      p1_flag_pos: Optional[Tuple[int, int]] = None, 
                      p2_flag_pos: Optional[Tuple[int, int]] = None):
        """Update the display with new game state."""
        self.current_board = actual_board.clone()
        self.current_player = current_player
        self.turn_count = turn_count
        self.last_move = last_move
        self.game_over = game_over
        self.winner = winner
        self.p1_flag_pos = p1_flag_pos
        self.p2_flag_pos = p2_flag_pos
        
        if last_move:
            self.move_history.append(last_move)
        
        self.render_board()
        
    def render_board(self):
        """Render the current board state."""
        self.ax.clear()
        self.setup_board_display()
        
        if self.current_board is None:
            return
            
        # Draw lake squares
        lake_positions = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
        for r, c in lake_positions:
            rect = patches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                   facecolor=self.colors['lake'], 
                                   edgecolor=self.colors['grid'], linewidth=2)
            self.ax.add_patch(rect)
            self.ax.text(c, r, 'LAKE', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece_value = self.current_board[r, c].item()
                
                if piece_value == EMPTY_SQUARE or piece_value == LAKE_SQUARE:
                    continue
                    
                # Determine piece type and owner
                # Special handling for flags using tracked positions
                is_player1 = piece_value > 0
                
                # Check if this position is a tracked flag position
                if (r, c) == self.p1_flag_pos or (r, c) == self.p2_flag_pos:
                    piece_type = PieceType.FLAG
                elif piece_value != EMPTY_SQUARE and piece_value != LAKE_SQUARE:
                    # Handle other pieces based on their values
                    if piece_value < 0:
                        piece_type = PieceType(abs(piece_value))
                    elif piece_value > 0:
                        piece_type = PieceType(piece_value)
                    else:
                        # This shouldn't happen for actual pieces
                        continue
                else:
                    # Skip empty squares and lakes
                    continue
                
                # Choose color based on piece type and owner
                if piece_type == PieceType.FLAG:
                    color = self.colors['p1_flag'] if is_player1 else self.colors['p2_flag']
                elif piece_type == PieceType.BOMB:
                    color = self.colors['p1_bomb'] if is_player1 else self.colors['p2_bomb']
                else:
                    color = self.colors['p1_piece'] if is_player1 else self.colors['p2_piece']
                
                # Highlight last move
                if (self.last_move and 
                    ((r, c) == self.last_move[1] or (r, c) == self.last_move[0])):
                    edge_color = self.colors['highlight']
                    edge_width = 3
                else:
                    edge_color = self.colors['grid']
                    edge_width = 1
                
                # Draw piece
                circle = patches.Circle((c, r), 0.35, facecolor=color, 
                                      edgecolor=edge_color, linewidth=edge_width)
                self.ax.add_patch(circle)
                
                # Add piece label
                piece_label = PIECE_NAMES[piece_type]
                player_prefix = '1' if is_player1 else '2'
                
                # Use different text color for better visibility
                text_color = 'white' if piece_type in [PieceType.FLAG, PieceType.BOMB] else 'black'
                
                self.ax.text(c, r - 0.1, piece_label, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color=text_color)
                self.ax.text(c, r + 0.15, f'P{player_prefix}', ha='center', va='center', 
                           fontsize=6, fontweight='bold', color=text_color)
        
        # Update title with game status
        if self.game_over:
            if self.winner == 0:
                status = "Game Over - Draw"
            else:
                status = f"Game Over - Player {1 if self.winner == 1 else 2} Wins!"
            title_color = 'red'
        else:
            status = f"Turn {self.turn_count} - Player {1 if self.current_player == 1 else 2}'s Turn"
            title_color = 'blue' if self.current_player == 1 else 'green'
        
        self.ax.set_title(f'Stratego Live Game - Full Information View\n{status}', 
                         fontsize=14, fontweight='bold', color=title_color)
        
        # Add legend
        self.add_legend()
        
        plt.draw()
        plt.pause(0.01)  # Small pause to allow display update
        
    def add_legend(self):
        """Add a legend explaining the colors and symbols."""
        legend_elements = [
            patches.Patch(color=self.colors['p1_piece'], label='Player 1 Pieces'),
            patches.Patch(color=self.colors['p2_piece'], label='Player 2 Pieces'),
            patches.Patch(color=self.colors['p1_flag'], label='Player 1 Flag'),
            patches.Patch(color=self.colors['p2_flag'], label='Player 2 Flag'),
            patches.Patch(color=self.colors['p1_bomb'], label='Player 1 Bombs'),
            patches.Patch(color=self.colors['p2_bomb'], label='Player 2 Bombs'),
            patches.Patch(color=self.colors['lake'], label='Lakes'),
            patches.Patch(color=self.colors['highlight'], label='Last Move')
        ]
        
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
    def show(self):
        """Show the live game viewer window."""
        plt.tight_layout()
        plt.show(block=False)
        
    def close(self):
        """Close the viewer window."""
        plt.close(self.fig)


class RestrictedGameViewer:
    """Viewer that shows only what each agent can see (for debugging agent perspective)."""
    
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.current_visible_p1 = None
        self.current_visible_p2 = None
        self.current_player = 1
        self.turn_count = 0
        
        self.colors = {
            'empty': 'white',
            'lake': 'lightblue',
            'hidden': 'gray',
            'own_piece': 'lightgreen',
            'enemy_piece': 'lightcoral',
            'grid': 'black'
        }
        
    def update_display(self, visible_p1: torch.Tensor, visible_p2: torch.Tensor, 
                      current_player: int, turn_count: int):
        """Update both player views."""
        self.current_visible_p1 = visible_p1.clone()
        self.current_visible_p2 = visible_p2.clone()
        self.current_player = current_player
        self.turn_count = turn_count
        
        self.render_player_view(self.ax1, visible_p1, 1, "Player 1 View")
        self.render_player_view(self.ax2, visible_p2, 2, "Player 2 View")
        
        plt.draw()
        plt.pause(0.01)
        
    def render_player_view(self, ax, visible_board: torch.Tensor, player: int, title: str):
        """Render a single player's view."""
        ax.clear()
        ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
        ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Add grid
        for i in range(BOARD_SIZE + 1):
            ax.axhline(i - 0.5, color=self.colors['grid'], linewidth=1)
            ax.axvline(i - 0.5, color=self.colors['grid'], linewidth=1)
        
        # Draw squares
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece_value = visible_board[r, c].item()
                
                if piece_value == LAKE_SQUARE:
                    rect = patches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                           facecolor=self.colors['lake'], 
                                           edgecolor=self.colors['grid'])
                    ax.add_patch(rect)
                    ax.text(c, r, 'LAKE', ha='center', va='center', fontsize=6)
                elif piece_value == HIDDEN_PIECE:
                    rect = patches.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                           facecolor=self.colors['hidden'], 
                                           edgecolor=self.colors['grid'])
                    ax.add_patch(rect)
                    ax.text(c, r, '?', ha='center', va='center', fontsize=12, fontweight='bold')
                elif piece_value != EMPTY_SQUARE:
                    # Determine if it's own piece or enemy piece
                    is_own_piece = (piece_value * player) > 0
                    color = self.colors['own_piece'] if is_own_piece else self.colors['enemy_piece']
                    
                    circle = patches.Circle((c, r), 0.35, facecolor=color, 
                                          edgecolor=self.colors['grid'])
                    ax.add_patch(circle)
                    
                    # Add piece label if known
                    if abs(piece_value) <= 11:  # Valid piece type
                        piece_type = PieceType(abs(piece_value))
                        piece_label = PIECE_NAMES[piece_type]
                        ax.text(c, r, piece_label, ha='center', va='center', 
                               fontsize=8, fontweight='bold')
        
        # Highlight current player
        title_color = 'blue' if self.current_player == player else 'gray'
        ax.set_title(f'{title}\nTurn {self.turn_count}', fontweight='bold', color=title_color)
        ax.set_xticks(range(BOARD_SIZE))
        ax.set_yticks(range(BOARD_SIZE))
        
    def show(self):
        """Show the restricted viewer window."""
        plt.tight_layout()
        plt.show(block=False)
        
    def close(self):
        """Close the viewer window."""
        plt.close(self.fig)
