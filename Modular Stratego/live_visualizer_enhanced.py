import pygame
import torch
import numpy as np
import sys
import os
import time
import re
from typing import List, Tuple, Dict, Optional

# Import existing modules
try:
    from drqn_agent import RainbowAgent
    from setup_agent import SetupAgent
    from environment import StrategoEnvironment
    from piece import PieceType
    from board import HIDDEN_PIECE, LAKE_SQUARE, BOARD_SIZE, EMPTY_SQUARE
except ImportError as e:
    print(f"Error importing game modules: {e}")
    print("Please ensure live_visualizer.py is in the same directory as your game files.")
    sys.exit(1)

# --- Constants & Config ---
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50
TILE_SIZE = 80
SIDEBAR_X = BOARD_OFFSET_X + (BOARD_SIZE * TILE_SIZE) + 50
SIDEBAR_WIDTH = 450

# Colors
COLOR_BG = (30, 30, 30)
COLOR_BOARD_LIGHT = (240, 217, 181)
COLOR_BOARD_DARK = (181, 136, 99)
COLOR_LAKE = (64, 164, 223)
COLOR_HIGHLIGHT = (255, 255, 0, 100)  # Yellow, transparent
COLOR_TEXT = (255, 255, 255)
COLOR_P1 = (70, 130, 180)   # Steel Blue
COLOR_P2 = (205, 92, 92)    # Indian Red
COLOR_BUTTON = (100, 100, 100)
COLOR_BUTTON_HOVER = (150, 150, 150)
COLOR_GRID = (200, 200, 200) # Light gray for grid lines
COLOR_P2_UNKNOWN = (100, 100, 100) # Gray for unknown enemy pieces

# Heatmap Gradient (Low -> High)
HEATMAP_COLORS = [
    (255, 0, 0, 128),    # Red (Bad)
    (255, 165, 0, 128),  # Orange
    (255, 255, 0, 128),  # Yellow
    (0, 255, 0, 128)     # Green (Good)
]

class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hovered = False

    def draw(self, surface, font):
        color = COLOR_BUTTON_HOVER if self.hovered else COLOR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=5)
        
        text_surf = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered and event.button == 1:
                self.callback()

class LiveVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Stratego DQN Live Visualizer")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_med = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        # Init Environment and Agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.env = StrategoEnvironment(device=self.device)
        self.agent1 = RainbowAgent(player_id=1, device=self.device)
        self.agent2 = RainbowAgent(player_id=-1, device=self.device)
        
        # Find latest episode to load by default
        self.episode_num = self._find_best_model_episode()
        self.agent_status = "Not Loaded"
        self._load_models(self.episode_num)
        
        self.setup_agent1 = SetupAgent(1, self.device)
        self.setup_agent2 = SetupAgent(-1, self.device)
        self.setup_status = "Not Loaded"
        self._load_setup_models(self.episode_num)
        
        self.pending_battle = None # Store battle info: {action, attacker, defender, time}
        self.use_pbs = True
        
        # UI Elements
        self.buttons = [
            Button(SIDEBAR_X, WINDOW_HEIGHT - 100, 100, 40, "Step (>)", self.step_game),
            Button(SIDEBAR_X + 120, WINDOW_HEIGHT - 100, 100, 40, "Play/Pause", self.toggle_pause),
            Button(SIDEBAR_X + 240, WINDOW_HEIGHT - 100, 100, 40, "Reset", self.reset_game),
            Button(SIDEBAR_X + 360, WINDOW_HEIGHT - 100, 100, 40, "PBS: ON", self.toggle_pbs)
        ]
        
        # Cache lake positions for drawing
        self.lakes = set((int(r.item()), int(c.item())) for r, c in self.env.board.lakes)
        
        # Initial Setup
        self.reset_game()
        
        self.running = True
        self.paused = True
        self.use_pbs = not self.use_pbs
        # Update button text
        self.buttons[3].text = f"PBS: {'ON' if self.use_pbs else 'OFF'}"
        # Clear analysis to force refresh
        self.current_analysis = None

        self.last_move_time = time.time()
        self.auto_play_speed = 0.5 # Seconds per move
        
        # Move History Log
        self.move_history = [] # List of strings or dicts

    def _find_best_model_episode(self):
        models_dir = "dqn_models"
        if not os.path.exists(models_dir):
            return 0
        
        max_episode = 0
        # Look for agent1_rainbow_episode_{num}.pth
        for filename in os.listdir(models_dir):
            if filename.startswith("agent1_rainbow_episode_") and filename.endswith(".pth"):
                try:
                    # Format: agent1_rainbow_episode_{episode}.pth
                    parts = filename.split("_")
                    # parts: ['agent1', 'rainbow', 'episode', '{num}.pth']
                    ep_part = parts[-1].split(".")[0]
                    episode = int(ep_part)
                    if episode > max_episode:
                        max_episode = episode
                except ValueError:
                    continue
        return max_episode

    def _load_models(self, episode_num):
        if episode_num == 0:
            self.agent_status = "Random Init"
            print("\n⚠️  No trained models found. Using Random Initialization.")
            return

        print(f"\n🔄 Loading models for Episode {episode_num}...")
        models_dir = "dqn_models"
        
        # Load Agent 1
        try:
            # Rainbow
            dqn_path = f"{models_dir}/agent1_rainbow_episode_{episode_num}.pth"
            if os.path.exists(dqn_path):
                self.agent1.load_model(dqn_path)
                print(f"  ✅ Agent 1 Rainbow loaded")
            else:
                print(f"  ❌ Agent 1 Rainbow file not found: {dqn_path}")

        except Exception as e:
            print(f"  ❌ Failed to load Agent 1: {e}")

        # Load Agent 2
        try:
            # Rainbow
            dqn_path = f"{models_dir}/agent2_rainbow_episode_{episode_num}.pth"
            if os.path.exists(dqn_path):
                self.agent2.load_model(dqn_path)
                print(f"  ✅ Agent 2 Rainbow loaded")
            else:
                print(f"  ❌ Agent 2 Rainbow file not found: {dqn_path}")

        except Exception as e:
            print(f"  ❌ Failed to load Agent 2: {e}")
            
        self.agent_status = f"Loaded Ep {episode_num}"

    def _load_setup_models(self, episode_num):
        if episode_num == 0:
            self.setup_status = "Random Init"
            print("⚠️  No setup models found. Using Random/Heuristic Placement.")
            return

        print(f"🔄 Loading Setup Agents for Episode {episode_num}...")
        models_dir = "dqn_models"
        
        try:
            path1 = f"{models_dir}/setup_agent1_episode_{episode_num}.pth"
            if os.path.exists(path1):
                self.setup_agent1.load_model(path1)
                print(f"  ✅ Setup Agent 1 loaded")
            else:
                print(f"  ❌ Setup Agent 1 file not found: {path1}")
                
            path2 = f"{models_dir}/setup_agent2_episode_{episode_num}.pth"
            if os.path.exists(path2):
                self.setup_agent2.load_model(path2)
                print(f"  ✅ Setup Agent 2 loaded")
            else:
                print(f"  ❌ Setup Agent 2 file not found: {path2}")
                
            self.setup_status = f"Loaded Ep {episode_num}"
        except Exception as e:
            print(f"  ❌ Failed to load setup models: {e}")
            self.setup_status = "Load Failed"

    def reset_game(self):
        # Generate placements using setup agents
        try:
            # Player 1
            p1_pieces = self.env._generate_pieces()
            p1_positions = self.env.get_valid_placement_positions(1)
            p1_placement = self.setup_agent1.place_pieces(p1_pieces, p1_positions)
            
            # Player 2
            p2_pieces = self.env._generate_pieces()
            p2_positions = self.env.get_valid_placement_positions(-1)
            p2_placement = self.setup_agent2.place_pieces(p2_pieces, p2_positions)
            
        except Exception as e:
            print(f"Error generating placement: {e}")
            # Fallback to random if setup agents fail
            p1_placement = None
            p2_placement = None
        
        self.game_state = self.env.reset(p1_placement, p2_placement)
        self.current_analysis = None
        self.pending_battle = None
        self.last_move_time = time.time()
        self.selected_tile = None
        self.hovered_tile = None
        self.move_history = []
        
        # Reset PBS beliefs
        if self.agent1.pbs:
            self.agent1.pbs.reset()
        if self.agent2.pbs:
            self.agent2.pbs.reset()

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_pbs(self):
        self.use_pbs = not self.use_pbs
        self.buttons[3].text = f"PBS: {'ON' if self.use_pbs else 'OFF'}"
        self.current_analysis = None # Force re-analysis

    def get_color_from_gradient(self, value):
        # Value 0.0 -> Red, 1.0 -> Green
        idx = int(value * (len(HEATMAP_COLORS) - 1))
        idx = max(0, min(idx, len(HEATMAP_COLORS) - 1))
        return HEATMAP_COLORS[idx]

    def analyze_current_state(self):
        # Get Q-values for all valid moves of current player
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        
        state_rep = agent.get_state_representation(self.game_state)
        valid_moves = self.env.get_valid_moves()
        
        if not valid_moves:
            return []
            
        # Get Uncertainty Map
        uncertainty_map = {}
        if agent.pbs and self.game_state:
            uncertainty_map = agent.pbs.get_uncertainty_map(self.game_state)
            
        # Base Q-values
        if state_rep.dim() == 1:
            state_rep = state_rep.unsqueeze(0)
        elif state_rep.dim() == 3:
            state_rep = state_rep.unsqueeze(0)
            
        agent.q_network.eval()
        with torch.no_grad():
            log_probs = agent.q_network(state_rep)
            probs = log_probs.exp()
            expected_q_values = (probs * agent.support).sum(dim=2)
            base_q_values = expected_q_values.squeeze(0)
        agent.q_network.train()
        
        analysis_results = []
        for move in valid_moves:
            action_idx = agent._move_to_action_index(move)
            base_q = base_q_values[action_idx].item()
            
            uncertainty = agent.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * agent.uncertainty_exploration_multiplier
            final_score = base_q + exploration_bonus
            
            analysis_results.append({
                'move': move,
                'base_q': base_q,
                'uncertainty': uncertainty,
                'final_score': final_score
            })
            
        analysis_results.sort(key=lambda x: x['final_score'], reverse=True)
        return analysis_results

    def step_game(self):
        if self.game_state.game_over:
            return

        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        opponent = self.agent2 if current_player == 1 else self.agent1
        
        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            print(f"Player {current_player} has no moves!")
            self.game_state.game_over = True
            return

        # Get Action
        # We use act() which uses Noisy Nets
        action = agent.act(self.game_state.board, valid_moves, self.game_state)
        
        if action is None:
            print("Agent returned None action")
            return

        # Analyze for visualization (after action selection to match state)
        self.current_analysis = self.analyze_current_state()
        
        # Log move
        top_move = self.current_analysis[0]
        is_best = (action == top_move['move'])
        q_val = next((item['final_score'] for item in self.current_analysis if item['move'] == action), 0.0)
        max_q = top_move['final_score']
        
        self.move_history.append({
            'player': current_player,
            'move': action,
            'q_val': q_val,
            'max_q': max_q,
            'is_best': is_best
        })
        if len(self.move_history) > 10:
            self.move_history.pop(0)

        # Execute Step
        # Check for battle first to visualize
        (r_from, c_from), (r_to, c_to) = action
        target_piece = self.env.board.grid[r_to, c_to].item()
        
        if target_piece != EMPTY_SQUARE and target_piece != LAKE_SQUARE:
            # Battle!
            attacker = self.env.board.grid[r_from, c_from].item()
            defender = target_piece
            self.pending_battle = {
                'action': action,
                'attacker': attacker,
                'defender': defender,
                'time': time.time()
            }
            # Pause briefly to show battle?
            # For now, just proceed.
            
        next_state, reward, done, info = self.env.step(action)
        self.game_state = next_state
        
        # Update PBS
        opponent.update_pbs_from_action(action, self.game_state, acting_player=current_player)
        
        if done:
            print(f"Game Over! Winner: {self.game_state.winner}")

    def draw_board(self):
        self.screen.fill(COLOR_BG)
        
        # Draw Board Grid
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = BOARD_OFFSET_X + c * TILE_SIZE
                y = BOARD_OFFSET_Y + r * TILE_SIZE
                rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                
                # Background
                if (r, c) in self.lakes:
                    color = COLOR_LAKE
                elif (r + c) % 2 == 0:
                    color = COLOR_BOARD_LIGHT
                else:
                    color = COLOR_BOARD_DARK
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected/hovered
                if self.selected_tile == (r, c):
                    pygame.draw.rect(self.screen, (0, 255, 0), rect, 3)
                elif self.hovered_tile == (r, c):
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
                    
                # Draw Piece
                piece_val = self.env.board.grid[r, c].item()
                if piece_val != 0 and piece_val != LAKE_SQUARE:
                    self.draw_piece(x, y, piece_val, r, c)
                    
                # Draw PBS Overlay (if enabled)
                if self.use_pbs and piece_val == HIDDEN_PIECE: # Only for hidden pieces? Or all enemy pieces?
                    # Actually, we want to show PBS for pieces unknown to the viewer (usually P1 perspective)
                    # Let's assume viewer is P1.
                    if piece_val < 0: # Enemy
                        self.draw_pbs_overlay(x, y, r, c)

    def draw_piece(self, x, y, piece_val, r, c):
        # Determine color and text
        if piece_val > 0:
            color = COLOR_P1
            text = self.get_piece_text(piece_val)
        elif piece_val < 0:
            color = COLOR_P2
            if piece_val == HIDDEN_PIECE: # Hidden
                color = COLOR_P2_UNKNOWN
                text = "?"
            else:
                text = self.get_piece_text(piece_val)
        else:
            return

        center = (x + TILE_SIZE // 2, y + TILE_SIZE // 2)
        radius = TILE_SIZE // 2 - 5
        
        pygame.draw.circle(self.screen, color, center, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)
        
        text_surf = self.font_med.render(text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=center)
        self.screen.blit(text_surf, text_rect)

    def get_piece_text(self, val):
        abs_val = abs(val)
        if abs_val == PieceType.FLAG.value: return "F"
        if abs_val == PieceType.BOMB.value: return "B"
        if abs_val == PieceType.SPY.value: return "S"
        if abs_val == 10: return "10" # Marshal
        if abs_val == 9: return "9"
        if abs_val == 8: return "8"
        if abs_val == 7: return "7"
        if abs_val == 6: return "6"
        if abs_val == 5: return "5"
        if abs_val == 4: return "4"
        if abs_val == 3: return "3"
        if abs_val == 2: return "2"
        return str(abs_val)

    def draw_pbs_overlay(self, x, y, r, c):
        # Get PBS belief for this square
        # Assuming viewer is P1, we want PBS for P2 pieces
        if not self.agent1.pbs: return
        
        # We need to get the belief distribution for this specific square
        # The PBS module stores beliefs in a tensor (1, 12, 10, 10) usually
        # Let's access it directly if possible, or via a method
        
        # For now, let's just draw a small indicator if high confidence of a strong piece
        # This requires exposing PBS internals or adding a method to PBS
        # Let's skip detailed overlay for now to avoid complexity, 
        # just draw a small dot if we have info.
        pass

    def draw_sidebar(self):
        # Draw background
        rect = pygame.Rect(SIDEBAR_X, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), rect)
        
        # Title
        title = self.font_large.render("Stratego AI", True, COLOR_TEXT)
        self.screen.blit(title, (SIDEBAR_X + 20, 20))
        
        # Status
        status_y = 70
        p1_text = self.font_med.render(f"P1 (Blue): {self.agent_status}", True, COLOR_P1)
        self.screen.blit(p1_text, (SIDEBAR_X + 20, status_y))
        
        p2_text = self.font_med.render(f"P2 (Red): {self.agent_status}", True, COLOR_P2)
        self.screen.blit(p2_text, (SIDEBAR_X + 20, status_y + 30))
        
        turn_text = self.font_med.render(f"Turn: {self.env.turn_count}", True, COLOR_TEXT)
        self.screen.blit(turn_text, (SIDEBAR_X + 20, status_y + 70))
        
        # Move History
        hist_y = 200
        hist_title = self.font_med.render("Move History:", True, COLOR_TEXT)
        self.screen.blit(hist_title, (SIDEBAR_X + 20, hist_y))
        
        for i, move in enumerate(reversed(self.move_history)):
            y = hist_y + 30 + (i * 25)
            if y > WINDOW_HEIGHT - 150: break
            
            p_color = COLOR_P1 if move['player'] == 1 else COLOR_P2
            (r1, c1), (r2, c2) = move['move']
            text = f"P{1 if move['player']==1 else 2}: ({r1},{c1})->({r2},{c2})"
            
            # Add Q-value info
            q_text = f" Q:{move['q_val']:.2f}"
            if move['is_best']:
                q_text += " (*)"
            
            surf = self.font_small.render(text + q_text, True, p_color)
            self.screen.blit(surf, (SIDEBAR_X + 20, y))

    def run(self):
        while self.running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                for btn in self.buttons:
                    btn.handle_event(event)
                    
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if BOARD_OFFSET_X <= mx < BOARD_OFFSET_X + BOARD_SIZE * TILE_SIZE and \
                       BOARD_OFFSET_Y <= my < BOARD_OFFSET_Y + BOARD_SIZE * TILE_SIZE:
                        c = (mx - BOARD_OFFSET_X) // TILE_SIZE
                        r = (my - BOARD_OFFSET_Y) // TILE_SIZE
                        self.hovered_tile = (r, c)
                    else:
                        self.hovered_tile = None
                        
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.hovered_tile:
                        self.selected_tile = self.hovered_tile
                        print(f"Clicked: {self.selected_tile}")

            # Auto-Play
            if not self.paused and not self.game_state.game_over:
                if time.time() - self.last_move_time > self.auto_play_speed:
                    self.step_game()
                    self.last_move_time = time.time()

            # Drawing
            self.draw_board()
            self.draw_sidebar()
            
            for btn in self.buttons:
                btn.draw(self.screen, self.font_med)
                
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    viz = LiveVisualizer()
    viz.run()