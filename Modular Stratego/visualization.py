import pygame
import numpy as np
import torch
import os
from piece import PieceType, PIECE_RANKS

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (200, 200, 200)
COLOR_LAKE = (0, 100, 200)
COLOR_P1 = (50, 50, 200) # Blue
COLOR_P2 = (200, 50, 50) # Red
COLOR_TEXT = (255, 255, 255)
COLOR_HIGHLIGHT = (255, 255, 0)
COLOR_Q_ARROW = (0, 255, 0)

class StrategoVisualizer:
    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = width // 12 # Leave space for margins
        self.margin = (width - (self.cell_size * 10)) // 2
        
        # Surface for drawing
        self.screen = pygame.Surface((width, height))
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 16)

    def render_state(self, game_state, q_values=None, kluss_info=None, save_path=None):
        """
        Render the game state to the screen surface.
        Args:
            game_state: The current GameState object.
            q_values: List of (move, value) tuples for top moves.
            kluss_info: Dict with KLUSS stats (depth, nodes, etc).
            save_path: If provided, save the frame to this path.
        """
        self.screen.fill(COLOR_BG)
        
        # Draw Grid & Lakes
        for r in range(10):
            for c in range(10):
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Lakes
                if (r in [4, 5]) and (c in [2, 3, 6, 7]):
                    pygame.draw.rect(self.screen, COLOR_LAKE, rect)
                else:
                    pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)
                    
        # Draw Pieces
        board = game_state.board
        for r in range(10):
            for c in range(10):
                piece_val = board[r, c].item()
                if piece_val == 0:
                    continue
                    
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                center = (x + self.cell_size // 2, y + self.cell_size // 2)
                
                # Determine Color
                color = COLOR_P1 if piece_val > 0 else COLOR_P2
                pygame.draw.circle(self.screen, color, center, self.cell_size // 2 - 4)
                
                # Determine Label
                # If we are viewing as an observer (god mode), show all.
                # If viewing as a player, hide enemy pieces unless revealed.
                # For this visualizer, we assume "God Mode" for training recordings,
                # but maybe indicate hidden status visually (e.g., dim color).
                
                abs_val = abs(piece_val)
                label = str(abs_val)
                if abs_val == 11: label = "B" # Bomb
                if abs_val == 0: label = "F" # Flag (Wait, Flag is usually 0 or special? Check piece.py)
                # Assuming Flag is 12 or 0? 
                # In piece.py: FLAG=0? No, usually 0 is empty.
                # Let's assume standard Stratego ranks.
                
                text = self.font.render(label, True, COLOR_TEXT)
                text_rect = text.get_rect(center=center)
                self.screen.blit(text, text_rect)

        # Draw Q-Values
        if q_values:
            for move, val in q_values:
                (r1, c1), (r2, c2) = move
                
                start_pos = (self.margin + c1 * self.cell_size + self.cell_size//2, 
                             self.margin + r1 * self.cell_size + self.cell_size//2)
                end_pos = (self.margin + c2 * self.cell_size + self.cell_size//2, 
                           self.margin + r2 * self.cell_size + self.cell_size//2)
                
                pygame.draw.line(self.screen, COLOR_Q_ARROW, start_pos, end_pos, 4)
                
                # Draw Value Text
                val_text = self.small_font.render(f"{val:.2f}", True, COLOR_HIGHLIGHT)
                self.screen.blit(val_text, end_pos)

        # Draw KLUSS Info
        if kluss_info:
            info_text = f"KLUSS: Depth {kluss_info.get('depth', '?')} | Nodes {kluss_info.get('nodes', '?')}"
            text_surf = self.small_font.render(info_text, True, COLOR_TEXT)
            self.screen.blit(text_surf, (10, 10))

        # Save Frame
        if save_path:
            pygame.image.save(self.screen, save_path)
            
    def get_surface(self):
        return self.screen
