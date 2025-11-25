import pygame
import sys
import torch
import time
from environment import StrategoEnvironment
from visualization import StrategoVisualizer
from hybrid_agent import HybridAgent
from piece import PieceType

class StrategoGUI:
    def __init__(self):
        self.env = StrategoEnvironment(device='cpu')
        self.state = self.env.reset()
        self.visualizer = StrategoVisualizer()
        
        # Initialize AI Agent (Player -1)
        self.ai_agent = HybridAgent(player_id=-1, device='cpu')
        # Load model if available
        # self.ai_agent.evaluator.load_model("models/best_model.pt")
        
        self.selected_pos = None
        self.running = True
        self.clock = pygame.time.Clock()
        
        # Human Player is 1 (Blue/Positive)
        # AI Player is -1 (Red/Negative)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.state.current_player == 1: # Human Turn
                    x, y = pygame.mouse.get_pos()
                    c = (x - self.visualizer.margin) // self.visualizer.cell_size
                    r = (y - self.visualizer.margin) // self.visualizer.cell_size
                    
                    if 0 <= r < 10 and 0 <= c < 10:
                        self.handle_click(r, c)

    def handle_click(self, r, c):
        # If no piece selected, select one
        if self.selected_pos is None:
            piece = self.state.board[r, c].item()
            if piece > 0: # My piece
                self.selected_pos = (r, c)
                print(f"Selected {r}, {c}")
        else:
            # Try to move
            r_from, c_from = self.selected_pos
            action = ((r_from, c_from), (r, c))
            
            # Validate
            valid_moves = self.env.get_valid_moves()
            if action in valid_moves:
                self.state, reward, game_over, info = self.env.step(action)
                self.selected_pos = None
                if game_over:
                    print(f"Game Over! Winner: {info['winner']}")
                    self.running = False
            else:
                # Deselect or select new
                piece = self.state.board[r, c].item()
                if piece > 0:
                    self.selected_pos = (r, c)
                else:
                    self.selected_pos = None

    def run(self):
        while self.running:
            self.handle_input()
            
            # AI Turn
            if self.state.current_player == -1 and self.running:
                print("AI Thinking...")
                # Visualize thinking?
                # We can get Q-values from the agent if we modify act() to return them
                # For now, just act
                valid_moves = self.env.get_valid_moves()
                if not valid_moves:
                    print("AI has no moves. You win!")
                    self.running = False
                    continue
                    
                action = self.ai_agent.act(self.state, valid_moves)
                self.state, reward, game_over, info = self.env.step(action)
                
                if game_over:
                    print(f"Game Over! Winner: {info['winner']}")
                    self.running = False
            
            # Render
            self.visualizer.render_state(self.state)
            
            # Highlight selection
            if self.selected_pos:
                r, c = self.selected_pos
                x = self.visualizer.margin + c * self.visualizer.cell_size
                y = self.visualizer.margin + r * self.visualizer.cell_size
                rect = pygame.Rect(x, y, self.visualizer.cell_size, self.visualizer.cell_size)
                pygame.draw.rect(self.visualizer.screen, (255, 255, 0), rect, 3)
                
            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()

if __name__ == "__main__":
    gui = StrategoGUI()
    gui.run()
