import pygame
import time
import random
import csv
import os
import sys
from typing import List, Dict, Tuple, Optional
from stratego import Board, auto_setup, side_name, FILES, Piece

# Access the MARQ framework from the sibling directory
MODULAR_STRATEGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modular Stratego"))
if MODULAR_STRATEGO_PATH not in sys.path:
    sys.path.append(MODULAR_STRATEGO_PATH)

from dqn_bot_logic import DQNBotLogic
from setup_agent_integration import StrategicSetupAgent

pygame.init()

# Get screen info for dynamic scaling
display_info = pygame.display.Info()
SCREEN_WIDTH = display_info.current_w
SCREEN_HEIGHT = display_info.current_h

# Base dimensions
BOARD_SIZE = 10
MIN_TILE_SIZE = 30
MAX_TILE_SIZE = 75
BASE_WINDOW_WIDTH = 1100  # Wider for eval bar and coach panel
BASE_WINDOW_HEIGHT = 800

# Colors - Modern Slate/Neon Palette
BACKGROUND = (15, 23, 42)     # Dark Slate
CARD_BG = (30, 41, 59)        # Slate 800
CARD_LIGHT = (51, 65, 85)     # Slate 700
WHITE = (248, 250, 252)       # Slate 50
GREY_100 = (241, 245, 249)
GREY_300 = (203, 213, 225)
GREY_500 = (100, 116, 139)
GREY_700 = (51, 65, 85)
GREY_800 = (30, 41, 59)

PLAYER1_PRIMARY = (59, 130, 246)  # Blue 500
PLAYER1_LIGHT = (96, 165, 250)    # Blue 400
PLAYER1_GLOW = (147, 197, 253)
PLAYER2_PRIMARY = (239, 68, 68)   # Red 500
PLAYER2_LIGHT = (248, 113, 113)   # Red 400
PLAYER2_GLOW = (252, 165, 165)
DANGER = (220, 38, 38)            # Red 600
WARNING = (245, 158, 11)          # Amber 500

TILE_LIGHT = (226, 232, 240)
TILE_DARK = (148, 163, 184)
TILE_LAKE = (14, 165, 233)       # Sky 500
TILE_SELECTED = (234, 179, 8)    # Yellow 500
TILE_HOVER = (250, 204, 21)
TILE_VALID = (34, 197, 94)       # Green 500

# Game State
current_player = 1
selected = None
running = True
game_state = "menu"
message = "Welcome to Guided Stratego"
human_side = 1
vs_bot = False
opponent_bot = None # AI playing as Red (Player 2)
coach_bot = None    # AI providing hints for Blue (Player 1)
setup_agent = None
move_history = []
fullscreen = False

# AI Guidance State
current_evaluation = 0.0
recommendations = []
show_hints = True  # AUTOMATIC RECOMMENDATIONS
hovered_sq = None
BATTLE_POPUP_DURATION = 2.0
battle_popup = None

# Button/Animation State
button_states = {}

# Setup board
board = Board()
auto_setup(board, 1)
auto_setup(board, 2)

# Determine model paths
model_dir = os.path.dirname(os.path.abspath(__file__))
agent_model_path = os.path.join(model_dir, 'agent1_league_episode_1000.pt')
setup_model_path = os.path.join(model_dir, 'setup_agent_final.pth')

# PIECE IMAGE LOADER
piece_images = {}
def load_piece_images():
    global piece_images
    # Search in multiple possible locations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates_dirs = [
        os.path.join(base_dir, 'pieces'),
        os.path.join(base_dir, 'assets', 'pieces'),
        os.path.join(os.path.dirname(base_dir), 'Python Stratego Game', 'pieces')
    ]
    
    pieces_dir = None
    for d in candidates_dirs:
        if os.path.exists(d):
            pieces_dir = d
            break
            
    if not pieces_dir:
        print("⚠️ [GUI] Warning: Pieces directory not found.")
        return
        
    print(f"🎨 [GUI] Loading assets from: {pieces_dir}")
    piece_types = ['F', 'B', '10', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for ptype in piece_types:
        for player in [1, 2]:
            color_name = 'blue' if player == 1 else 'red'
            # Try multiple naming conventions
            candidates = [
                f"{ptype}_player{player}.png",
                f"{ptype}_{color_name}.png",
                f"{ptype}.png" if player == 1 else None # Fallback
            ]
            for c in candidates:
                if not c: continue
                path = os.path.join(pieces_dir, c)
                if os.path.exists(path):
                    try:
                        img = pygame.image.load(path).convert_alpha()
                        piece_images[(ptype, player)] = img
                        break
                    except Exception as e:
                        print(f"❌ [GUI] Error loading {ptype}: {e}")
    print(f"✅ [GUI] Loaded {len(piece_images)} piece images.")

# Helper Functions
def get_font(size, bold=False):
    font = pygame.font.SysFont("Inter, Segoe UI, Roboto, sans-serif", size)
    if bold: font.set_bold(True)
    return font

def draw_text(surface, text, x, y, size, color, bold=False, center=True):
    font = get_font(size, bold)
    text_surf = font.render(str(text), True, color)
    rect = text_surf.get_rect()
    if center: rect.center = (x, y)
    else: rect.topleft = (x, y)
    surface.blit(text_surf, rect)

def draw_rounded_rect(surface, color, rect, radius=10, alpha=255):
    if alpha < 255:
        temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(temp, (*color, alpha), temp.get_rect(), border_radius=radius)
        surface.blit(temp, (rect.x, rect.y))
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_shadow(surface, rect, offset=4, alpha=60, radius=10):
    shadow_rect = rect.copy()
    shadow_rect.x += offset
    shadow_rect.y += offset
    draw_rounded_rect(surface, (0, 0, 0), shadow_rect, radius=radius, alpha=alpha)

def draw_button(text, x, y, width, height, action=None, primary=False, danger=False, disabled=False):
    global button_states
    if action not in button_states:
        button_states[action] = {'hover': False, 'pressed': False}
    mouse_pos = pygame.mouse.get_pos()
    button_rect = pygame.Rect(x, y, width, height)
    is_hover = button_rect.collidepoint(mouse_pos) and not disabled
    button_states[action]['hover'] = is_hover
    
    if disabled:
        bg_color = GREY_700
        text_color = GREY_500
    elif primary:
        bg_color = PLAYER1_PRIMARY if not is_hover else PLAYER1_LIGHT
        text_color = WHITE
    elif danger:
        bg_color = DANGER if not is_hover else PLAYER2_LIGHT
        text_color = WHITE
    else:
        bg_color = CARD_BG if not is_hover else CARD_LIGHT
        text_color = WHITE
        
    if not disabled:
        draw_shadow(screen, button_rect, offset=3, alpha=40)
    draw_rounded_rect(screen, bg_color, button_rect, radius=8)
    draw_text(screen, text, button_rect.centerx, button_rect.centery, 18, text_color, bold=True)
    
    if is_hover and not disabled:
        mouse_pressed = pygame.mouse.get_pressed()[0]
        if mouse_pressed and not button_states[action]['pressed']:
            button_states[action]['pressed'] = True
            return action
        elif not mouse_pressed:
            button_states[action]['pressed'] = False
    return None

def get_board_dimensions():
    cur_w, cur_h = screen.get_size()
    scale = min(cur_w / BASE_WINDOW_WIDTH, cur_h / BASE_WINDOW_HEIGHT)
    tile_size = int(max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, 60 * scale)))
    border = int(tile_size * 0.15)
    bw, bh = tile_size * 10, tile_size * 10
    
    # Center board but leave room for Eval Bar (left) and Coach (right)
    start_x = (cur_w - bw) // 2
    start_y = (cur_h - bh) // 2 + 30
    return {'tile': tile_size, 'border': border, 'x': start_x, 'y': start_y, 'w': bw, 'h': bh}

def draw_piece(surface, piece, x, y, size, highlight=False, hidden=False, revealed=False):
    if not piece: return
    rect = pygame.Rect(x + 4, y + 4, size-8, size-8)
    player = piece.owner
    ptype = piece.short()
    
    # Shadow
    draw_shadow(surface, rect, offset=2, alpha=60, radius=6)
    
    if hidden and not revealed:
        draw_rounded_rect(surface, GREY_700, rect, 6, 200)
        draw_text(surface, "??", rect.centerx, rect.centery, int(size*0.4), GREY_300, True)
    else:
        if (ptype, player) in piece_images:
            img = pygame.transform.smoothscale(piece_images[(ptype, player)], (rect.width, rect.height))
            surface.blit(img, rect.topleft)
        else:
            color = PLAYER1_PRIMARY if player == 1 else PLAYER2_PRIMARY
            draw_rounded_rect(surface, color, rect, 6)
            draw_text(surface, ptype, rect.centerx, rect.centery, int(size*0.4), WHITE, True)
            
    if highlight:
        pygame.draw.rect(surface, TILE_SELECTED, rect, 3, border_radius=6)

def draw_board():
    global hovered_sq
    dims = get_board_dimensions()
    ts, bx, by = dims['tile'], dims['x'], dims['y']
    
    # Background Panel
    bg_rect = pygame.Rect(bx - 10, by - 10, dims['w'] + 20, dims['h'] + 20)
    draw_rounded_rect(screen, CARD_BG, bg_rect, 12)
    pygame.draw.rect(screen, GREY_700, bg_rect, 2, border_radius=12)
    
    mouse_pos = pygame.mouse.get_pos()
    hovered_sq = None
    
    for r in range(10):
        for f in range(10):
            tx, ty = bx + f * ts, by + r * ts
            tile_rect = pygame.Rect(tx, ty, ts, ts)
            
            # Base color
            if board.is_lake(r, f): color = TILE_LAKE
            else: color = TILE_LIGHT if (r + f) % 2 == 0 else TILE_DARK
            
            # Coach Hints
            if show_hints:
                for rec in recommendations:
                    m = rec['move']
                    if (r, f) == m[0]: color = (rec['color'][0], rec['color'][1], rec['color'][2], 120)
                    elif (r, f) == m[1]: color = rec['color']
            
            if tile_rect.collidepoint(mouse_pos) and not board.is_lake(r, f):
                color = TILE_HOVER
                hovered_sq = (r, f)
            
            if selected == (r, f): color = TILE_SELECTED
            
            pygame.draw.rect(screen, color[:3] if len(color)>3 else color, tile_rect)
            if len(color) > 3: # Handle alpha for hints
                temp = pygame.Surface((ts, ts), pygame.SRCALPHA)
                temp.fill(color)
                screen.blit(temp, tile_rect.topleft)
                
            if selected and (r, f) in board.legal_moves_from(selected):
                draw_rounded_rect(screen, TILE_VALID, tile_rect, 0, 80)
            
            pygame.draw.rect(screen, GREY_800, tile_rect, 1)
            
            piece = board.get((r, f))
            if piece:
                visible = (piece.owner == human_side) or piece.revealed or game_state == "setup"
                draw_piece(screen, piece, tx, ty, ts, highlight=(selected == (r, f)), hidden=not visible, revealed=piece.revealed)

def draw_eval_bar():
    dims = get_board_dimensions()
    bar_w, bar_h = 24, dims['h']
    bar_x, bar_y = dims['x'] - 60, dims['y']
    
    # Tracks
    pygame.draw.rect(screen, GREY_800, (bar_x, bar_y, bar_w, bar_h), border_radius=6)
    
    # Value calculation (Dueling Head expected value from COACH BOT perspective)
    val = max(-10.0, min(10.0, current_evaluation))
    fill_pivot = bar_y + bar_h // 2
    pixel_per_unit = (bar_h / 2) / 10.0
    
    if val > 0:
        h = int(val * pixel_per_unit)
        pygame.draw.rect(screen, PLAYER1_PRIMARY, (bar_x, fill_pivot - h, bar_w, h), border_radius=4)
    else:
        h = int(abs(val) * pixel_per_unit)
        pygame.draw.rect(screen, PLAYER2_PRIMARY, (bar_x, fill_pivot, bar_w, h), border_radius=4)
        
    pygame.draw.line(screen, WHITE, (bar_x, fill_pivot), (bar_x + bar_w, fill_pivot), 2)
    draw_text(screen, f"{val:+.1f}", bar_x + bar_w//2, bar_y - 20, 16, WHITE, True)

def draw_coach_panel():
    if not show_hints or not recommendations: return
    dims = get_board_dimensions()
    panel_x = dims['x'] + dims['w'] + 40
    panel_y = dims['y']
    panel_w = 220
    
    draw_rounded_rect(screen, CARD_BG, pygame.Rect(panel_x, panel_y, panel_w, 320), 12)
    pygame.draw.rect(screen, GREY_700, (panel_x, panel_y, panel_w, 320), 2, border_radius=12)
    draw_text(screen, " COACH ANALYSIS", panel_x + panel_w//2, panel_y + 25, 18, PLAYER1_GLOW, True)
    
    for i, rec in enumerate(recommendations):
        ry = panel_y + 60 + i * 80
        # Classification Label
        draw_rounded_rect(screen, rec['color'], pygame.Rect(panel_x + 15, ry, 85, 24), 12)
        draw_text(screen, rec['label'], panel_x + 15 + 42, ry + 12, 13, BACKGROUND, True)
        
        # Move Text
        m = rec['move']
        move_str = f"{FILES[m[0][1]]}{m[0][0]+1} -> {FILES[m[1][1]]}{m[1][0]+1}"
        draw_text(screen, move_str, panel_x + 110, ry + 12, 15, WHITE, False, center=False)
        draw_text(screen, f"Val: {rec['score']:+.2f}", panel_x + 15, ry + 45, 13, GREY_300, False, center=False)

def draw_prediction_tooltip():
    if not hovered_sq or game_state != "play": return
    piece = board.get(hovered_sq)
    if not piece or piece.owner == human_side or piece.revealed: return
    
    # Use COACH BOT to get probabilities (it tracks opponent side)
    probs = coach_bot.get_rank_probabilities(hovered_sq) if coach_bot else None
    if not probs: return
    
    mx, my = pygame.mouse.get_pos()
    tw, th = 170, 120
    tx, ty = mx + 20, my + 20
    if tx + tw > screen.get_width(): tx -= (tw + 40)
    if ty + th > screen.get_height(): ty -= (th + 40)
    
    draw_rounded_rect(screen, (0,0,0), pygame.Rect(tx+4, ty+4, tw, th), 8, 100)
    draw_rounded_rect(screen, CARD_BG, pygame.Rect(tx, ty, tw, th), 8)
    pygame.draw.rect(screen, PLAYER2_PRIMARY, (tx, ty, tw, th), 2, border_radius=8)
    
    draw_text(screen, "AI Piece Prediction", tx + tw//2, ty + 15, 14, PLAYER2_GLOW, True)
    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (name, p) in enumerate(sorted_p):
        draw_text(screen, f"{name}: {p*100:.1f}%", tx + tw//2, ty + 45 + i*22, 13, GREY_100 if i==0 else GREY_300)

def draw_battle_popup():
    global battle_popup
    if not battle_popup: return
    elapsed = time.time() - battle_popup['start_time']
    if elapsed > BATTLE_POPUP_DURATION:
        battle_popup = None; return
    
    alpha = 255
    if elapsed > BATTLE_POPUP_DURATION - 0.5:
        alpha = int(255 * (BATTLE_POPUP_DURATION - elapsed) / 0.5)
        
    pw, ph = 340, 180
    px = (screen.get_width() - pw) // 2
    py = (screen.get_height() - ph) // 2
    
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    draw_rounded_rect(surf, (*CARD_BG, alpha), pygame.Rect(0, 0, pw, ph), 16)
    pygame.draw.rect(surf, (*GREY_700, alpha), (0, 0, pw, ph), 3, border_radius=16)
    
    # Header
    res = battle_popup['result']
    h_color = PLAYER1_PRIMARY if res == 'attacker_wins' else (PLAYER2_PRIMARY if res == 'defender_wins' else WARNING)
    draw_rounded_rect(surf, (*h_color, alpha), pygame.Rect(0, 0, pw, 40), 16)
    
    screen.blit(surf, (px, py))
    draw_text(screen, "BATTLE RESULT", px + pw//2, py + 20, 20, WHITE, True)
    
    # Pieces
    atk_pt, def_pt = battle_popup['attacker_type'], battle_popup['defender_type']
    atk_o, def_o = battle_popup['attacker_owner'], battle_popup['defender_owner']
    
    def draw_pop_piece(pt, owner, cx, cy):
        rect = pygame.Rect(cx-30, cy-30, 60, 60)
        col = PLAYER1_PRIMARY if owner == 1 else PLAYER2_PRIMARY
        draw_rounded_rect(screen, col, rect, 8, alpha)
        draw_text(screen, pt, cx, cy, 24, WHITE, True)
        
    draw_pop_piece(atk_pt, atk_o, px + pw//4, py + ph//2)
    draw_pop_piece(def_pt, def_o, px + 3*pw//4, py + ph//2)
    draw_text(screen, "VS", px + pw//2, py + ph//2, 28, GREY_300, True)
    
    res_text = "VICTORY" if res == 'attacker_wins' else ("DEFEAT" if res == 'defender_wins' else "TIE")
    draw_text(screen, res_text, px + pw//2, py + ph - 25, 24, h_color, True)

def update_analysis():
    global current_evaluation, recommendations
    if coach_bot:
        current_evaluation = coach_bot.get_state_evaluation(board, human_side)
        recommendations = coach_bot.get_multi_recommendations(board, human_side)

def handle_click(pos):
    global selected, current_player, game_state, show_hints, battle_popup
    dims = get_board_dimensions()
    f = (pos[0] - dims['x']) // dims['tile']
    r = (pos[1] - dims['y']) // dims['tile']
    if not (0 <= r < 10 and 0 <= f < 10): return
    sq = (r, f)
    
    if game_state == "setup":
        p = board.get(sq)
        if p and p.owner == human_side:
            if selected is None: selected = sq
            else:
                p1, p2 = board.get(selected), board.get(sq)
                board.set(selected, p2); board.set(sq, p1)
                selected = None
    elif game_state == "play" and current_player == human_side:
        if selected is None:
            p = board.get(sq)
            if p and p.owner == human_side: selected = sq
        else:
            if sq in board.legal_moves_from(selected):
                atk = board.get(selected)
                dfn = board.get(sq)
                
                msg, winner = board.move_and_resolve(selected, sq, human_side)
                
                # Update both bots
                if opponent_bot: opponent_bot.update_from_opponent_move(board, selected, sq)
                if coach_bot: coach_bot.update_from_opponent_move(board, selected, sq) # In setup, both track moves
                
                if dfn:
                    p_after = board.get(sq)
                    if p_after and p_after.owner == atk.owner: res = 'attacker_wins'
                    elif not p_after and not board.get(selected): res = 'tie'
                    else: res = 'defender_wins'
                    battle_popup = {'attacker_type': atk.short(), 'attacker_owner': atk.owner, 'defender_type': dfn.short(), 'defender_owner': dfn.owner, 'result': res, 'start_time': time.time()}
                
                if winner is not None: game_state = "game_over"
                else:
                    current_player = 3 - human_side
                    update_analysis()
                selected = None
            else:
                p = board.get(sq)
                if p and p.owner == human_side: selected = sq
                else: selected = None

def bot_step():
    global current_player, game_state, battle_popup
    time.sleep(0.6)
    move = opponent_bot.choose_move(board, current_player)
    if move:
        src, dst = move
        atk = board.get(src); dfn = board.get(dst)
        msg, winner = board.move_and_resolve(src, dst, human_side)
        
        # Update Coach AI on opponent's move
        if coach_bot: coach_bot.update_from_opponent_move(board, src, dst)
        
        if dfn:
            p_after = board.get(dst)
            if p_after and p_after.owner == atk.owner: res = 'attacker_wins'
            elif not p_after and not board.get(src): res = 'tie'
            else: res = 'defender_wins'
            battle_popup = {'attacker_type': atk.short(), 'attacker_owner': atk.owner, 'defender_type': dfn.short(), 'defender_owner': dfn.owner, 'result': res, 'start_time': time.time()}
        if winner is not None: game_state = "game_over"
    current_player = human_side
    update_analysis()

# Main Loop
screen = pygame.display.set_mode((BASE_WINDOW_WIDTH, BASE_WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("GUIDED STRATEGO: AI COACH")
load_piece_images() # LOAD AFTER SCREEN INIT
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                fullscreen = not fullscreen
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if game_state in ["menu", "setup", "play"]: handle_click(event.pos)
            
    screen.fill(BACKGROUND)
    
    if game_state == "menu":
        cx = screen.get_width() // 2
        draw_text(screen, "GUIDED STRATEGO", cx, 180, 72, PLAYER1_PRIMARY, True)
        draw_text(screen, "POWERED BY MARQ FRAMEWORK", cx, 240, 24, PLAYER1_GLOW, True)
        if draw_button("BATTLE AI COACH", cx - 150, 360, 300, 50, action="play", primary=True) == "play":
            board = Board(); auto_setup(board, 2)
            try:
                setup_agent = StrategicSetupAgent(load_path=setup_model_path)
                board = setup_agent.setup_side(board, 1)
            except: auto_setup(board, 1)
            vs_bot = True; human_side = 1; current_player = 1; game_state = "setup"
            
            # INITIALIZE TWO BOTS
            print("🤖 [MARQ] Initializing Dual-Bot Architecture...")
            opponent_bot = DQNBotLogic(agent_model_path, player_id=2)
            coach_bot = DQNBotLogic(agent_model_path, player_id=1)
            update_analysis()
            
        if draw_button("EXIT GAME", cx - 150, 430, 300, 50, action="quit", danger=True) == "quit": running = False
        
    elif game_state == "setup":
        draw_board()
        dims = get_board_dimensions()
        draw_text(screen, "STRATEGIC SETUP", dims['x'] + dims['w']//2, dims['y'] - 40, 32, PLAYER1_GLOW, True)
        if draw_button("CONFIRM SETUP", dims['x'] + dims['w']//2 - 100, dims['y'] + dims['h'] + 40, 200, 40, action="start", primary=True) == "start":
            game_state = "play"
            
    elif game_state == "play":
        draw_board()
        draw_eval_bar()
        draw_coach_panel()
        draw_prediction_tooltip()
        draw_battle_popup()
        if vs_bot and current_player != human_side: bot_step()
    
    elif game_state == "game_over":
        draw_board()
        draw_rounded_rect(screen, (0,0,0), pygame.Rect(0,0,screen.get_width(), screen.get_height()), 0, 150)
        draw_text(screen, "GAME OVER", screen.get_width()//2, screen.get_height()//2 - 50, 100, WHITE, True)
        if draw_button("RETURN TO MENU", screen.get_width()//2 - 120, screen.get_height()//2 + 50, 240, 50, action="menu", primary=True) == "menu":
            game_state = "menu"

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
