import pygame
import torch
import numpy as np
import sys
import os
import time
from typing import List, Tuple, Dict, Optional

# Import existing modules
try:
    from dqn_agent import DQNAgent
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
        self.agent1 = DQNAgent(player_id=1, device=self.device, use_pbs=True)
        self.agent2 = DQNAgent(player_id=-1, device=self.device, use_pbs=True)
        
        # State
        self.game_state = self.env.reset()
        self.running = True
        self.paused = True
        self.auto_play_speed = 1.0  # Seconds per move
        self.last_move_time = time.time()
        self.selected_tile = None
        self.hovered_tile = None
        self.current_analysis = None
        self.use_pbs = True  # Toggle for PBS usage
        self.agent_status = "Initializing..."
        self.pending_battle = None # Store battle info: {action, attacker, defender, time}
        
        # Load models if possible
        self._load_models()
        
        # UI Elements
        self.buttons = [
            Button(SIDEBAR_X, WINDOW_HEIGHT - 100, 100, 40, "Step (>)", self.step_game),
            Button(SIDEBAR_X + 120, WINDOW_HEIGHT - 100, 100, 40, "Play/Pause", self.toggle_pause),
            Button(SIDEBAR_X + 240, WINDOW_HEIGHT - 100, 100, 40, "Reset", self.reset_game),
            Button(SIDEBAR_X + 360, WINDOW_HEIGHT - 100, 100, 40, "PBS: ON", self.toggle_pbs)
        ]
        
        # Cache lake positions for drawing
        self.lakes = set((int(r.item()), int(c.item())) for r, c in self.env.board.lakes)

    def _load_models(self):
        try:
            # Load models from parent directory dqn_models, relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "..", "dqn_models")
            episode = 3100
            
            # Load Agent 1
            self.agent1.load_model(os.path.join(model_dir, f"agent1_dqn_episode_{episode}.pth"))
            if self.agent1.pbs:
                self.agent1.load_aaren_model(os.path.join(model_dir, f"agent1_aaren_episode_{episode}.pth"))
                self.agent1.load_pbs_evaluator(os.path.join(model_dir, f"agent1_pbs_evaluator_episode_{episode}.pth"))
                
            # Load Agent 2
            self.agent2.load_model(os.path.join(model_dir, f"agent2_dqn_episode_{episode}.pth"))
            if self.agent2.pbs:
                self.agent2.load_aaren_model(os.path.join(model_dir, f"agent2_aaren_episode_{episode}.pth"))
                self.agent2.load_pbs_evaluator(os.path.join(model_dir, f"agent2_pbs_evaluator_episode_{episode}.pth"))
                
            # Force exploitation for visualization
            self.agent1.epsilon = 0.0
            self.agent2.epsilon = 0.0
            
            print(f"Loaded agent models for episode {episode}.")
            self.agent_status = f"Loaded (Ep {episode})"
        except Exception as e:
            print(f"Could not load models: {e}")
            print("Using random agents.")
            self.agent_status = "Random (Untrained)"

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_pbs(self):
        self.use_pbs = not self.use_pbs
        # Update button text
        self.buttons[3].text = f"PBS: {'ON' if self.use_pbs else 'OFF'}"
        # Clear analysis to force refresh
        self.current_analysis = None

    def reset_game(self):
        self.game_state = self.env.reset()
        self.current_analysis = None
        self.pending_battle = None
        print("Game Reset")

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
            
        # Set to eval mode to avoid BatchNorm errors with batch size 1
        agent.q_network.eval()
        
        # Get base Q-values (without bonus/penalty)
        with torch.no_grad():
            base_q_values = agent.q_network(state_rep.unsqueeze(0))
        
        # Get uncertainty if available
        uncertainty_map = {}
        
        # Handle PBS Toggle
        original_pbs = agent.pbs
        if not self.use_pbs:
            agent.pbs = None # Temporarily disable for calculation
            
        try:
            if agent.pbs:
                uncertainty_map = agent.pbs.get_uncertainty_map(self.env.board.actual_board)
                
            # Calculate uncertainty-aware Q-values (Adjusted Q)
            # This subtracts the penalty
            q_values = agent.calculate_uncertainty_aware_q_values(
                base_q_values, valid_moves, uncertainty_map
            )
        finally:
            if not self.use_pbs:
                agent.pbs = original_pbs # Restore
            
        results = []
        all_scores = []
        
        for move in valid_moves:
            action_idx = agent._move_to_action_index(move)
            
            base_q = base_q_values[0, action_idx].item()
            adjusted_q = q_values[0, action_idx].item()
            
            # Calculate penalty (Base - Adjusted)
            uncertainty_penalty = base_q - adjusted_q
            
            uncertainty = agent.get_move_uncertainty(move, uncertainty_map)
            exploration_bonus = uncertainty * agent.uncertainty_exploration_multiplier
            
            final_score = adjusted_q + exploration_bonus
            
            all_scores.append(final_score)
            
            results.append({
                'move': move,
                'base_q': base_q,
                'adjusted_q': adjusted_q,
                'uncertainty_penalty': uncertainty_penalty,
                'uncertainty': uncertainty,
                'bonus': exploration_bonus,
                'final_score': final_score
            })
            
        # Normalize scores for heatmap colors
        if all_scores:
            min_s = min(all_scores)
            max_s = max(all_scores)
            rng = max_s - min_s if max_s != min_s else 1.0
            
            for res in results:
                res['normalized'] = (res['final_score'] - min_s) / rng
        
        # Sort by score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results

    def step_game(self):
        if self.env.game_over or self.pending_battle:
            return

        # 1. Analyze before moving
        self.current_analysis = self.analyze_current_state()
        
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        
        # 2. Act
        
        # Handle PBS Toggle for Action
        original_pbs = agent.pbs
        if not self.use_pbs:
            agent.pbs = None
            
        try:
            state_rep = agent.get_state_representation(self.game_state)
            valid_moves = self.env.get_valid_moves()
            
            if not valid_moves:
                return

            action = agent.act(state_rep, valid_moves, self.game_state)
        finally:
            if not self.use_pbs:
                agent.pbs = original_pbs
        
        # Check for Battle
        (r_from, c_from), (r_to, c_to) = action
        target_val = int(self.env.board.actual_board[r_to, c_to].item())
        
        is_attack = False
        if target_val != 0 and target_val != LAKE_SQUARE:
            # Check if enemy
            if (current_player == 1 and target_val < 0) or \
               (current_player == -1 and target_val > 0):
                is_attack = True
                
        if is_attack:
            self.pending_battle = {
                'action': action,
                'attacker_pos': (r_from, c_from),
                'defender_pos': (r_to, c_to),
                'attacker_val': int(self.env.board.actual_board[r_from, c_from].item()),
                'defender_val': target_val,
                'start_time': time.time()
            }
            return # Wait for animation
        
        # 3. Step Environment (Immediate if no battle)
        self.game_state, reward, done, info = self.env.step(action)
        
        # Clear analysis cache for next state
        self.current_analysis = None

    def draw_board(self):
        # Draw Grid
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(
                    BOARD_OFFSET_X + c * TILE_SIZE, 
                    BOARD_OFFSET_Y + r * TILE_SIZE, 
                    TILE_SIZE, TILE_SIZE
                )
                
                # Check background color (Lake vs Land)
                color = COLOR_LAKE if (r, c) in self.lakes else (COLOR_BOARD_LIGHT if (r+c)%2==0 else COLOR_BOARD_DARK)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selection
                if self.selected_tile == (r, c):
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 3)
                elif self.hovered_tile == (r, c):
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 2)

        # Draw Coordinates
        for i in range(BOARD_SIZE):
            # Rows
            text = self.font_small.render(str(i), True, COLOR_TEXT)
            self.screen.blit(text, (BOARD_OFFSET_X - 25, BOARD_OFFSET_Y + i * TILE_SIZE + TILE_SIZE//2 - 10))
            # Cols
            text = self.font_small.render(str(i), True, COLOR_TEXT)
            self.screen.blit(text, (BOARD_OFFSET_X + i * TILE_SIZE + TILE_SIZE//2 - 5, BOARD_OFFSET_Y - 25))

    def get_piece_text(self, piece_val):
        # Abbreviations
        piece_map = {
            1: "F",   # FLAG
            2: "Spy", # SPY
            3: "Sc",  # SCOUT  
            4: "Mn",  # MINER
            5: "Sgt", # SERGEANT
            6: "Lt",  # LIEUTENANT
            7: "Cpt", # CAPTAIN
            8: "Maj", # MAJOR
            9: "Col", # COLONEL
            10: "Gen",# GENERAL
            11: "Mar",# MARSHAL
            12: "B"   # BOMB
        }
        return piece_map.get(abs(piece_val), "?")

    def draw_pieces(self):
        board = self.env.board.actual_board
        current_player = self.env.current_player
        agent = self.agent1 if current_player == 1 else self.agent2
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = int(board[r, c].item())
                if val == 0 or val == LAKE_SQUARE:
                    continue
                
                x = BOARD_OFFSET_X + c * TILE_SIZE + TILE_SIZE // 2
                y = BOARD_OFFSET_Y + r * TILE_SIZE + TILE_SIZE // 2
                
                # Color
                color = COLOR_P1 if val > 0 else COLOR_P2
                
                # Draw Piece Circle
                pygame.draw.circle(self.screen, color, (x, y), TILE_SIZE // 2 - 5)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), TILE_SIZE // 2 - 5, 2)
                
                # Draw Text
                txt = self.get_piece_text(val)
                text_surf = self.font_med.render(txt, True, (255, 255, 255))
                text_rect = text_surf.get_rect(center=(x, y))
                self.screen.blit(text_surf, text_rect)
                
                # PBS Overlay for Enemy Pieces
                is_enemy = (current_player == 1 and val < 0) or (current_player == -1 and val > 0)
                if is_enemy and self.use_pbs and agent.pbs:
                    # Get predicted rank
                    beliefs = agent.pbs.get_belief_distribution((r, c))
                    if beliefs:
                        # Get max probability rank
                        best_rank, conf = max(beliefs.items(), key=lambda x: x[1])
                        if conf > 0.3: # Only show if somewhat confident
                            # Draw predicted rank below piece
                            pred_txt = self.get_piece_text(best_rank.value)
                            pred_surf = self.font_small.render(f"[{pred_txt}]", True, (255, 255, 0))
                            pred_rect = pred_surf.get_rect(center=(x, y + 20))
                            # Add black background for readability
                            bg_rect = pred_rect.inflate(4, 4)
                            pygame.draw.rect(self.screen, (0,0,0, 180), bg_rect)
                            self.screen.blit(pred_surf, pred_rect)

    def draw_heatmap(self):
        if not self.current_analysis:
            # If not cached, analyze
            self.current_analysis = self.analyze_current_state()
            
        if not self.current_analysis:
            return

        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # Draw heatmap on destination squares
        for item in self.current_analysis:
            move = item['move']
            score = item['normalized']
            (r_from, c_from), (r_to, c_to) = move
            
            should_draw = False
            if self.hovered_tile == (r_from, c_from):
                should_draw = True
            elif self.hovered_tile is None and item['normalized'] > 0.8: # Show top moves lightly
                 should_draw = True
                 
            if should_draw:
                rect_x = BOARD_OFFSET_X + c_to * TILE_SIZE
                rect_y = BOARD_OFFSET_Y + r_to * TILE_SIZE
                
                color = self.get_color_from_gradient(score)
                pygame.draw.rect(surface, color, (rect_x, rect_y, TILE_SIZE, TILE_SIZE))
                
                # Draw Q-value text small
                q_txt = f"{item['final_score']:.2f}"
                txt_surf = self.font_small.render(q_txt, True, (0,0,0))
                surface.blit(txt_surf, (rect_x + 5, rect_y + 5))
                
                # Draw Arrow
                start_pos = (BOARD_OFFSET_X + c_from * TILE_SIZE + TILE_SIZE//2, 
                             BOARD_OFFSET_Y + r_from * TILE_SIZE + TILE_SIZE//2)
                end_pos = (BOARD_OFFSET_X + c_to * TILE_SIZE + TILE_SIZE//2, 
                           BOARD_OFFSET_Y + r_to * TILE_SIZE + TILE_SIZE//2)
                pygame.draw.line(surface, (255, 255, 255, 200), start_pos, end_pos, 3)
                pygame.draw.circle(surface, color, end_pos, 5)

        self.screen.blit(surface, (0, 0))

    def draw_sidebar(self):
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), (SIDEBAR_X, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        
        # Title
        title = self.font_large.render("Game Info", True, COLOR_HIGHLIGHT)
        self.screen.blit(title, (SIDEBAR_X + 20, 20))
        
        # Game State Stats
        y = 80
        turn_txt = self.font_med.render(f"Turn: {self.env.turn_count}", True, COLOR_TEXT)
        self.screen.blit(turn_txt, (SIDEBAR_X + 20, y))
        
        y += 40
        p_color = COLOR_P1 if self.env.current_player == 1 else COLOR_P2
        p_name = f"Player {self.env.current_player} (Agent {'1' if self.env.current_player==1 else '2'})"
        player_txt = self.font_med.render(p_name, True, p_color)
        self.screen.blit(player_txt, (SIDEBAR_X + 20, y))
        
        y += 30
        status_txt = self.font_small.render(f"Agent Status: {self.agent_status}", True, (200, 200, 200))
        self.screen.blit(status_txt, (SIDEBAR_X + 20, y))
        
        y += 20
        pbs_status = "Enabled" if self.use_pbs else "Disabled"
        pbs_color = (100, 255, 100) if self.use_pbs else (255, 100, 100)
        pbs_txt = self.font_small.render(f"PBS Input: {pbs_status}", True, pbs_color)
        self.screen.blit(pbs_txt, (SIDEBAR_X + 20, y))
        
        y += 40
        pygame.draw.line(self.screen, (100,100,100), (SIDEBAR_X, y), (WINDOW_WIDTH, y), 1)
        y += 20
        
        # Selected/Hovered Tile Info
        target = self.hovered_tile if self.hovered_tile else self.selected_tile
        if target:
            r, c = target
            val = int(self.env.board.actual_board[r, c].item())
            agent = self.agent1 if self.env.current_player == 1 else self.agent2

            # Note: Agent 1 tracks beliefs about Player -1 (values < 0)
            # So if we hover a P2 piece while it's P1's turn, show P1's beliefs
            is_enemy_piece = (self.env.current_player == 1 and val < 0) or \
                             (self.env.current_player == -1 and val > 0)
            
            if is_enemy_piece and agent.pbs:
                y += 10
                pbs_title = self.font_med.render("Agent Beliefs (PBS):", True, COLOR_HIGHLIGHT)
                self.screen.blit(pbs_title, (SIDEBAR_X + 20, y))
                y += 30
                
                beliefs = agent.pbs.get_belief_distribution((r, c))
                # Sort by confidence
                sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for pt, conf in sorted_beliefs:
                    if conf > 0.01:
                        pt_name = str(pt).split('.')[1]
                        bar_w = int(conf * 200)
                        # Draw Bar
                        pygame.draw.rect(self.screen, (100, 100, 200), (SIDEBAR_X + 150, y+5, bar_w, 10))
                        # Draw Text
                        conf_txt = self.font_small.render(f"{pt_name}: {conf:.2%}", True, COLOR_TEXT)
                        self.screen.blit(conf_txt, (SIDEBAR_X + 20, y))
                        y += 20

        # Move Analysis (if hovering a piece that can move)
        y += 20
        pygame.draw.line(self.screen, (100,100,100), (SIDEBAR_X, y), (WINDOW_WIDTH, y), 1)
        y += 20
        
        ana_title = self.font_med.render("Move Analysis (Top 3)", True, COLOR_HIGHLIGHT)
        self.screen.blit(ana_title, (SIDEBAR_X + 20, y))
        y += 30
        
        # Filter analysis for hovered piece if valid, else global top 3
        display_analysis = []
        if target:
            if self.current_analysis:
                display_analysis = [item for item in self.current_analysis if item['move'][0] == target]
        
        if not display_analysis and self.current_analysis:
            display_analysis = self.current_analysis
            
        # Sort and take top 3
        if display_analysis:
            display_analysis.sort(key=lambda x: x['final_score'], reverse=True)
            
            for i, item in enumerate(display_analysis[:3]):
                m = item['move']
                score = item['final_score']
                base = item['base_q']
                unc = item['uncertainty']
                pen = item.get('uncertainty_penalty', 0.0)
                
                txt_str = f"{m[0]}->{m[1]}: Q={score:.2f} (P={pen:.2f}, U={unc:.2f})"
                txt = self.font_small.render(txt_str, True, COLOR_TEXT)
                self.screen.blit(txt, (SIDEBAR_X + 20, y))
                y += 20
                
                # Mini bar for score
                norm = max(0, min(1, item.get('normalized', 0.5)))
                color = self.get_color_from_gradient(norm)
                pygame.draw.rect(self.screen, color, (SIDEBAR_X + 20, y, int(norm*300), 5))
                y += 15

    def draw_battle_overlay(self):
        if not self.pending_battle:
            return
            
        # Semi-transparent background
        s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        # Battle Info
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        
        # Title
        title = self.font_large.render("BATTLE!", True, (255, 50, 50))
        title_rect = title.get_rect(center=(cx, cy - 150))
        self.screen.blit(title, title_rect)
        
        # Attacker
        att_val = self.pending_battle['attacker_val']
        att_txt = self.get_piece_text(att_val)
        att_color = COLOR_P1 if att_val > 0 else COLOR_P2
        
        pygame.draw.circle(self.screen, att_color, (cx - 150, cy), 60)
        att_surf = self.font_large.render(att_txt, True, (255, 255, 255))
        att_rect = att_surf.get_rect(center=(cx - 150, cy))
        self.screen.blit(att_surf, att_rect)
        
        # VS
        vs_txt = self.font_large.render("VS", True, (255, 255, 255))
        vs_rect = vs_txt.get_rect(center=(cx, cy))
        self.screen.blit(vs_txt, vs_rect)
        
        # Defender
        def_val = self.pending_battle['defender_val']
        def_txt = self.get_piece_text(def_val)
        def_color = COLOR_P1 if def_val > 0 else COLOR_P2
        
        pygame.draw.circle(self.screen, def_color, (cx + 150, cy), 60)
        def_surf = self.font_large.render(def_txt, True, (255, 255, 255))
        def_rect = def_surf.get_rect(center=(cx + 150, cy))
        self.screen.blit(def_surf, def_rect)
        
        # Result Prediction (Who wins?)
        att_rank = abs(att_val)
        def_rank = abs(def_val)
        
        result_text = "???"
        res_color = (200, 200, 200)
        
        if def_rank == 12: # Bomb
            if att_rank == 4: # Miner
                result_text = "Miner Defuses!"
                res_color = (100, 255, 100)
            else:
                result_text = "Boom!"
                res_color = (255, 100, 100)
        elif att_rank == 2 and def_rank == 11: # Spy vs Marshal
            result_text = "Spy Assassinates!"
            res_color = (100, 255, 100)
        elif att_rank > def_rank:
            result_text = "Attacker Wins!"
            res_color = (100, 255, 100)
        elif att_rank < def_rank:
            result_text = "Defender Wins!"
            res_color = (255, 100, 100)
        else:
            result_text = "Draw!"
            res_color = (255, 255, 100)
            
        res_surf = self.font_large.render(result_text, True, res_color)
        res_rect = res_surf.get_rect(center=(cx, cy + 150))
        self.screen.blit(res_surf, res_rect)

    def update_input(self):
        mouse_pos = pygame.mouse.get_pos()
        
        # Calculate hovered tile
        if (BOARD_OFFSET_X <= mouse_pos[0] < BOARD_OFFSET_X + BOARD_SIZE * TILE_SIZE and
            BOARD_OFFSET_Y <= mouse_pos[1] < BOARD_OFFSET_Y + BOARD_SIZE * TILE_SIZE):
            c = (mouse_pos[0] - BOARD_OFFSET_X) // TILE_SIZE
            r = (mouse_pos[1] - BOARD_OFFSET_Y) // TILE_SIZE
            self.hovered_tile = (r, c)
        else:
            self.hovered_tile = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            for btn in self.buttons:
                btn.handle_event(event)
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    self.selected_tile = self.hovered_tile
                elif event.button == 3: # Right click
                    self.selected_tile = None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.toggle_pause()
                elif event.key == pygame.K_RIGHT:
                    self.step_game()
                elif event.key == pygame.K_r:
                    self.reset_game()

    def run(self):
        while self.running:
            self.clock.tick(60) # 60 FPS UI
            self.update_input()
            
            # Auto-play logic
            if not self.paused and not self.pending_battle:
                current_time = time.time()
                if current_time - self.last_move_time > self.auto_play_speed:
                    self.step_game()
                    self.last_move_time = current_time
            
            # Battle Animation Logic
            if self.pending_battle:
                if time.time() - self.pending_battle['start_time'] > 2.0: # 2 seconds
                    # Execute move
                    action = self.pending_battle['action']
                    self.game_state, reward, done, info = self.env.step(action)
                    self.current_analysis = None
                    self.pending_battle = None
                    self.last_move_time = time.time()

            # Drawing
            self.screen.fill(COLOR_BG)
            
            self.draw_board()
            self.draw_pieces()
            self.draw_heatmap() # Overlay
            self.draw_sidebar()
            self.draw_battle_overlay() # Top layer
            
            # Draw Buttons
            for btn in self.buttons:
                btn.draw(self.screen, self.font_med)
            
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    vis = LiveVisualizer()
    vis.run()