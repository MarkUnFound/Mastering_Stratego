import pygame
import torch
import numpy as np
import sys
import os
import time
from typing import Dict, Tuple, Optional, List

# Add repository root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import StrategoEnvironment
from drqn_agent import RainbowAgent
from piece import PieceType, PIECE_NAMES, PIECE_RANKS
from board import BOARD_SIZE, LAKE_SQUARE

# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50
CELL_SIZE = 70
BOARD_PIXEL_SIZE = CELL_SIZE * BOARD_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 200, 50)
YELLOW = (200, 200, 50)
CYAN = (50, 200, 200)
DARK_GREEN = (0, 100, 0)
LAKE_COLOR = (0, 100, 200)
HIGHLIGHT_COLOR = (100, 255, 100, 128)
INVALID_COLOR = (255, 100, 100, 128)
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (240, 240, 245)
PANEL_BG = (220, 220, 230)

class StrategoVisualizer:
    def __init__(self, model_path_p1=None, model_path_p2=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Stratego Rainbow DQN Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Environment
        self.env = StrategoEnvironment(self.device)
        
        # Agents
        print("Initializing Agents...")
        self.agent1 = RainbowAgent(player_id=1, device=self.device)
        self.agent2 = RainbowAgent(player_id=-1, device=self.device)
        
        # Load Models (use load_model method)
        if model_path_p1 and os.path.exists(model_path_p1):
            print(f"Loading P1 Model: {model_path_p1}")
            self.agent1.load_model(model_path_p1)
        else:
            print("Warning: P1 Model not found, using random initialization.")

        if model_path_p2 and os.path.exists(model_path_p2):
            print(f"Loading P2 Model: {model_path_p2}")
            self.agent2.load_model(model_path_p2)
        else:
            print("Warning: P2 Model not found, using random initialization.")

        # Game State
        self.game_state = None
        self.running = True
        self.paused = True
        self.auto_step_interval = 500
        self.last_step_time = 0
        self.selected_pos = None
        self.hover_pos = None
        self.valid_moves_cache = []
        self.last_action_log = "Game Start"
        self.episode_reward = {1: 0.0, -1: 0.0}
        
        # Q-value visualization
        self.qvalue_cache = {}
        self.show_qvalues = True
        
        self.reset_game()

    def reset_game(self):
        print("Resetting Game...")
        self.game_state = self.env.reset()
        self.agent1.reset_pbs()
        self.agent2.reset_pbs()
        self.episode_reward = {1: 0.0, -1: 0.0}
        self.last_action_log = "Game Reset"
        self.selected_pos = None
        self.valid_moves_cache = []
        self.qvalue_cache = {}
        self.paused = True
        self._update_qvalue_cache()

    def get_board_pos(self, mouse_pos):
        x, y = mouse_pos
        if BOARD_OFFSET_X <= x < BOARD_OFFSET_X + BOARD_PIXEL_SIZE and \
           BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + BOARD_PIXEL_SIZE:
            c = (x - BOARD_OFFSET_X) // CELL_SIZE
            r = (y - BOARD_OFFSET_Y) // CELL_SIZE
            return int(r), int(c)
        return None

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self.step_game()
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_q:
                    self.show_qvalues = not self.show_qvalues
                    print(f"Q-value display: {'ON' if self.show_qvalues else 'OFF'}")
            
            elif event.type == pygame.MOUSEMOTION:
                self.hover_pos = self.get_board_pos(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked_pos = self.get_board_pos(event.pos)
                    if clicked_pos:
                        self.selected_pos = clicked_pos
                        self.update_valid_moves_visualization()

    def update_valid_moves_visualization(self):
        self.valid_moves_cache = []
        if not self.selected_pos:
            return

        all_moves = self.env.get_valid_moves()
        r_sel, c_sel = self.selected_pos
        for move in all_moves:
            (r1, c1), (r2, c2) = move
            if r1 == r_sel and c1 == c_sel:
                self.valid_moves_cache.append((r2, c2))

    def _update_qvalue_cache(self):
        """Compute Q-values for all valid moves of current player."""
        self.qvalue_cache = {}
        if self.env.game_over:
            return

        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return

        state_tensor = agent.get_state_representation(self.game_state.board, pbs_instance=agent.pbs)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        agent.q_network.eval()
        with torch.no_grad():
            log_probs = agent.q_network(state_tensor)
            probs = log_probs.exp()
            expected_q_values = (probs * agent.support).sum(dim=2).squeeze(0)
        agent.q_network.train()
        
        for move in valid_moves:
            (r_from, c_from), (r_to, c_to) = move
            action_idx = agent._move_to_action_index(move)
            q_val = expected_q_values[action_idx].item()
            
            if (r_to, c_to) not in self.qvalue_cache or q_val > self.qvalue_cache[(r_to, c_to)]:
                self.qvalue_cache[(r_to, c_to)] = q_val

    def step_game(self):
        if self.env.game_over:
            return

        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        opponent = self.agent2 if current_player == 1 else self.agent1
        
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            print("No valid moves!")
            self.env.game_over = True
            return

        action = agent.act_batch([self.game_state.board], [valid_moves], [self.game_state])[0]
        
        if action is None:
            print("Agent returned None action")
            return

        next_state, reward, done, info = self.env.step(action)
        
        # Handle revealed pieces
        revealed = info.get('revealed_in_step', [])
        for pos, piece_type in revealed:
            game_phase = info.get('game_phase', 'middle')
            turn_count = info.get('turn_count', 0)
            
            self.agent1.pbs.update_from_reveal(pos, piece_type, 
                                               game_phase=game_phase,
                                               turn_count=turn_count)
            self.agent2.pbs.update_from_reveal(pos, piece_type,
                                               game_phase=game_phase,
                                               turn_count=turn_count)
        
        opponent.update_pbs_batch([action], [self.game_state], acting_player=current_player)
        
        self.game_state = next_state
        self.episode_reward[current_player] += reward
        
        self._update_qvalue_cache()
        
        (r1, c1), (r2, c2) = action
        piece_val = self.env.board.actual_board[r2, c2].item()
        piece_type = PieceType(abs(int(piece_val))) if abs(int(piece_val)) > 0 else "Unknown"
        p_name = "Blue (P1)" if current_player == 1 else "Red (P2)"
        self.last_action_log = f"{p_name} moved {PIECE_NAMES.get(piece_type, '?')} from ({r1},{c1}) to ({r2},{c2}). R={reward:.2f}"
        print(self.last_action_log)
        
        if done:
            winner = info.get('winner', 0)
            w_text = "Draw" if winner == 0 else ("Blue Wins!" if winner == 1 else "Red Wins!")
            self.last_action_log = f"GAME OVER: {w_text}"
            print(self.last_action_log)
            self.paused = True

    def update(self):
        if not self.paused and not self.env.game_over:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time > self.auto_step_interval:
                self.step_game()
                self.last_step_time = current_time

    def draw_piece(self, r, c, piece_value, x, y):
        piece_type = PieceType(abs(piece_value))
        is_p1 = piece_value > 0
        
        color = BLUE if is_p1 else RED
        
        pygame.draw.rect(self.screen, color, (x + 5, y + 5, CELL_SIZE - 10, CELL_SIZE - 10), border_radius=8)
        
        text = PIECE_NAMES.get(piece_type, "?")
        text_surf = self.font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=(x + CELL_SIZE//2, y + CELL_SIZE//2))
        self.screen.blit(text_surf, text_rect)
        
        rank = PIECE_RANKS.get(piece_type, 0)
        rank_surf = self.small_font.render(str(rank), True, (200, 200, 200))
        self.screen.blit(rank_surf, (x + 8, y + 5))

    def draw_pbs_overlay(self, r, c, x, y):
        piece_val = self.env.board.actual_board[r, c].item()
        if piece_val >= 0:
            return
            
        beliefs = self.agent1.pbs.get_belief_distribution((r, c))
        if not beliefs:
            return
            
        most_likely = max(beliefs.items(), key=lambda x: x[1])
        prob = most_likely[1]
        
        if prob < 0.4:
            txt = "?"
            col = YELLOW
        else:
            txt = PIECE_NAMES.get(most_likely[0], "?")
            col = CYAN
            
        pygame.draw.rect(self.screen, BLACK, (x + CELL_SIZE - 25, y + 5, 20, 20))
        pygame.draw.rect(self.screen, col, (x + CELL_SIZE - 24, y + 6, 18, 18))
        
        t_surf = self.small_font.render(txt, True, BLACK)
        t_rect = t_surf.get_rect(center=(x + CELL_SIZE - 15, y + 15))
        self.screen.blit(t_surf, t_rect)

    def draw_qvalue_heatmap(self, r, c, x, y):
        """Draw Q-value heatmap on valid move destinations."""
        if (r, c) not in self.qvalue_cache:
            return
            
        q_value = self.qvalue_cache[(r, c)]
        all_q = list(self.qvalue_cache.values())
        
        if not all_q:
            return
            
        min_q, max_q = min(all_q), max(all_q)
        
        if max_q == min_q:
            normalized = 0.5
        else:
            normalized = (q_value - min_q) / (max_q - min_q)
        
        # Red (bad) -> Yellow (neutral) -> Green (good)
        if normalized < 0.5:
            r_col = 255
            g_col = int(255 * normalized * 2)
        else:
            r_col = int(255 * (1 - normalized) * 2)
            g_col = 255
        
        heatmap_color = (r_col, g_col, 0, 100)
        
        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        s.fill(heatmap_color)
        self.screen.blit(s, (x, y))
        
        q_text = self.small_font.render(f"{q_value:.2f}", True, BLACK)
        self.screen.blit(q_text, (x + 5, y + CELL_SIZE - 18))

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = BOARD_OFFSET_X + c * CELL_SIZE
                y = BOARD_OFFSET_Y + r * CELL_SIZE
                
                pygame.draw.rect(self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
                
                if self.env.board.actual_board[r, c].item() == LAKE_SQUARE:
                    pygame.draw.rect(self.screen, LAKE_COLOR, (x+1, y+1, CELL_SIZE-2, CELL_SIZE-2))
                    continue
                
                if self.selected_pos == (r, c):
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((255, 255, 0, 100))
                    self.screen.blit(s, (x, y))
                
                if (r, c) in self.valid_moves_cache:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill(HIGHLIGHT_COLOR)
                    self.screen.blit(s, (x, y))
                
                if self.show_qvalues:
                    self.draw_qvalue_heatmap(r, c, x, y)
                    
                piece_val = self.env.board.actual_board[r, c].item()
                if piece_val != 0:
                    self.draw_piece(r, c, int(piece_val), x, y)
                    self.draw_pbs_overlay(r, c, x, y)
                    
        # --- HUD ---
        hud_x = BOARD_OFFSET_X + BOARD_PIXEL_SIZE + 20
        pygame.draw.rect(self.screen, PANEL_BG, (hud_x, BOARD_OFFSET_Y, 350, SCREEN_HEIGHT - 100))
        
        y_off = BOARD_OFFSET_Y + 20
        
        title = self.title_font.render("Stratego Visualizer", True, BLACK)
        self.screen.blit(title, (hud_x + 20, y_off))
        y_off += 40
        
        stats = [
            f"Turn: {self.env.turn_count}",
            f"Player: {'Blue (1)' if self.env.current_player == 1 else 'Red (-1)'}",
            f"State: {'Paused' if self.paused else 'Running'}",
            f"Q-Values: {'ON' if self.show_qvalues else 'OFF'} (Press Q)",
            f"P1 Reward: {self.episode_reward[1]:.2f}",
            f"P2 Reward: {self.episode_reward[-1]:.2f}"
        ]
        
        for s in stats:
            surf = self.font.render(s, True, TEXT_COLOR)
            self.screen.blit(surf, (hud_x + 20, y_off))
            y_off += 30
            
        y_off += 20
        pygame.draw.line(self.screen, GRAY, (hud_x + 10, y_off), (hud_x + 340, y_off), 2)
        y_off += 20
        
        if self.hover_pos:
            r, c = self.hover_pos
            piece_val = self.env.board.actual_board[r, c].item()
            
            info_title = self.font.render(f"Square ({r}, {c})", True, BLACK)
            self.screen.blit(info_title, (hud_x + 20, y_off))
            y_off += 25
            
            if (r, c) in self.qvalue_cache:
                q_text = self.font.render(f"Q-Value: {self.qvalue_cache[(r, c)]:.3f}", True, DARK_GREEN)
                self.screen.blit(q_text, (hud_x + 20, y_off))
                y_off += 25
            
            if piece_val == 0:
                self.screen.blit(self.small_font.render("Empty", True, GRAY), (hud_x + 20, y_off))
                y_off += 20
            elif piece_val == LAKE_SQUARE:
                self.screen.blit(self.small_font.render("Lake", True, BLUE), (hud_x + 20, y_off))
                y_off += 20
            else:
                true_type = PieceType(abs(int(piece_val)))
                owner = "P1" if piece_val > 0 else "P2"
                self.screen.blit(self.font.render(f"{owner}: {true_type.name}", True, BLACK), (hud_x + 20, y_off))
                y_off += 30
                
                if piece_val < 0:
                    self.screen.blit(self.font.render("P1 Beliefs:", True, DARK_GREEN), (hud_x + 20, y_off))
                    y_off += 25
                    
                    beliefs = self.agent1.pbs.get_belief_distribution((r, c))
                    sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)
                    
                    for pt, prob in sorted_beliefs[:5]:
                        if prob < 0.01: continue
                        bar_w = int(prob * 100)
                        pygame.draw.rect(self.screen, BLUE, (hud_x + 120, y_off + 2, bar_w, 12))
                        txt = f"{pt.name[:8]}: {prob:.2f}"
                        self.screen.blit(self.small_font.render(txt, True, BLACK), (hud_x + 20, y_off))
                        y_off += 20
        
        # Log Console
        log_y = SCREEN_HEIGHT - 60
        pygame.draw.rect(self.screen, BLACK, (0, log_y, SCREEN_WIDTH, 60))
        log_surf = self.font.render(self.last_action_log, True, GREEN)
        self.screen.blit(log_surf, (20, log_y + 15))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    # Find model files in dqn_models directory
    model_dir = "dqn_models"
    p1_model = None
    p2_model = None
    
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        p1_candidates = [f for f in files if ('agent1' in f or 'best_model_p1' in f) and f.endswith(('.pt', '.pth'))]
        if p1_candidates:
            p1_model = os.path.join(model_dir, sorted(p1_candidates)[-1])
        
        p2_candidates = [f for f in files if ('agent2' in f or 'best_model_p2' in f) and f.endswith(('.pt', '.pth'))]
        if p2_candidates:
            p2_model = os.path.join(model_dir, sorted(p2_candidates)[-1])
    
    print(f"P1 Model: {p1_model}")
    print(f"P2 Model: {p2_model}")
    
    viz = StrategoVisualizer(p1_model, p2_model)
    viz.run()