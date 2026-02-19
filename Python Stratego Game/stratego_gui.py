"""
Stratego with MARQ - Modern Visual Interface
Features:
- Strategic Setup Agent for intelligent piece placement
- Modern, clean UI design with blue/red color scheme
- Enhanced visual feedback and animations
- Responsive design optimized for playability
"""

import pygame
import time
import random
import csv
import os
from stratego import Board, auto_setup, side_name, FILES, Piece
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
BASE_BORDER_SIZE = 10
MIN_HISTORY_WIDTH = 80
PREFERRED_HISTORY_WIDTH = 100
BASE_WINDOW_WIDTH = 900
BASE_WINDOW_HEIGHT = 750

# Game state
current_player = 1
selected = None
running = True
game_state = "menu"
message = "Welcome to Enhanced Stratego!"
human_side = 1
vs_bot = False
bot_logic = None
setup_agent = None
move_history = []
fullscreen = False
show_history_panel = True
lost_pieces_player1 = []
lost_pieces_player2 = []

# Button animation state
button_states = {}

# Setup board
board = Board()
auto_setup(board, 1)
auto_setup(board, 2)

# Determine model paths
model_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(model_dir, 'agent1_rainbow_episode_11000.pth'),
    os.path.join(model_dir, 'dqn_agent_final.pth'),
    os.path.join(model_dir, 'user_input_files', 'dqn_agent_final.pth'),
    'dqn_models/agent1_rainbow_episode_11000.pth',
    os.path.join(model_dir, '..', 'dqn_models', 'dqn_agent_final.pth'),
]

agent_model_path = None
for path in possible_paths:
    if os.path.exists(path):
        agent_model_path = os.path.abspath(path)
        print(f"Found agent model at: {agent_model_path}")
        break

if agent_model_path is None:
    agent_model_path = os.path.join(model_dir, 'agent1_rainbow_episode_11000.pth')
    print(f"⚠️  Warning: agent1_rainbow_episode_11000.pth not found in any expected location, using: {agent_model_path}")

# Setup model paths
possible_setup_paths = [
    os.path.join(model_dir, 'setup_agent_final.pth'),
    os.path.join(model_dir, 'user_input_files', 'setup_agent_final.pth'),
    'dqn_models/setup_agent_final.pth',
    os.path.join(model_dir, '..', 'dqn_models', 'setup_agent_final.pth'),
]

setup_model_path = None
for path in possible_setup_paths:
    if os.path.exists(path):
        setup_model_path = os.path.abspath(path)
        print(f"Found setup model at: {setup_model_path}")
        break

if setup_model_path is None:
    setup_model_path = os.path.join(model_dir, 'setup_agent_final.pth')
    print(f"⚠️  Warning: setup_agent_final.pth not found in any expected location, using: {setup_model_path}")

# Initialize window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
fullscreen = False
pygame.display.set_caption("Stratego - with MARQ (F11 for Fullscreen)")

# Modern color palette - Blue vs Red theme
BACKGROUND = (15, 23, 42)  # Dark slate background
CARD_BG = (30, 41, 59)  # Slate 800
CARD_LIGHT = (51, 65, 85)  # Slate 700

# Player 1 - Blue Team
PLAYER1_PRIMARY = (59, 130, 246)  # Blue 500
PLAYER1_LIGHT = (96, 165, 250)  # Blue 400
PLAYER1_DARK = (37, 99, 235)  # Blue 600
PLAYER1_GLOW = (147, 197, 253)  # Blue 300

# Player 2 - Red Team
PLAYER2_PRIMARY = (239, 68, 68)  # Red 500
PLAYER2_LIGHT = (248, 113, 113)  # Red 400
PLAYER2_DARK = (220, 38, 38)  # Red 600
PLAYER2_GLOW = (252, 165, 165)  # Red 300

# Neutral colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY_100 = (243, 244, 246)
GREY_300 = (209, 213, 219)
GREY_500 = (107, 114, 128)
GREY_700 = (55, 65, 81)
GREY_900 = (17, 24, 39)

# Board colors
TILE_LIGHT = (226, 232, 240)  # Slate 200
TILE_DARK = (148, 163, 184)  # Slate 400
TILE_LAKE = (56, 189, 248)  # Sky 400
TILE_SELECTED = (250, 204, 21)  # Yellow 400
TILE_HOVER = (253, 224, 71)  # Yellow 300
TILE_VALID = (134, 239, 172)  # Green 300

# UI colors
ACCENT = (99, 102, 241)  # Indigo 500
SUCCESS = (34, 197, 94)  # Green 500
WARNING = (234, 179, 8)  # Yellow 500
DANGER = (239, 68, 68)  # Red 500

def get_player_color(player, variant='primary'):
    """Get color for a player with variant support"""
    if player == 1:
        if variant == 'light':
            return PLAYER1_LIGHT
        elif variant == 'dark':
            return PLAYER1_DARK
        elif variant == 'glow':
            return PLAYER1_GLOW
        return PLAYER1_PRIMARY
    else:
        if variant == 'light':
            return PLAYER2_LIGHT
        elif variant == 'dark':
            return PLAYER2_DARK
        elif variant == 'glow':
            return PLAYER2_GLOW
        return PLAYER2_PRIMARY

# Calculate optimal tile size based on available space
def calculate_optimal_tile_size(current_width, current_height, show_history=True):
    """Calculate optimal tile size to maximize board size while maintaining aspect ratio"""
    base_vertical_padding = 20
    base_horizontal_padding = 10
    
    scale_factor = min(current_width / BASE_WINDOW_WIDTH, current_height / BASE_WINDOW_HEIGHT)
    scale_factor = max(0.9, min(scale_factor, 3.0))
    
    vertical_padding = int(base_vertical_padding * scale_factor)
    horizontal_padding = int(base_horizontal_padding * scale_factor)
    
    vertical_padding = max(20, vertical_padding)
    horizontal_padding = max(10, horizontal_padding)
    
    available_height = current_height - vertical_padding
    if show_history:
        history_reserve = 60
        available_width = current_width - horizontal_padding - history_reserve
    else:
        available_width = current_width - horizontal_padding
    
    tile_from_height = available_height // BOARD_SIZE
    tile_from_width = available_width // BOARD_SIZE
    
    optimal_tile = min(tile_from_height, tile_from_width)
    optimal_tile = max(MIN_TILE_SIZE, min(optimal_tile, MAX_TILE_SIZE))
    
    return optimal_tile

def calculate_history_width(current_width, board_width_with_borders, game_state=None):
    """Calculate history panel width"""
    if game_state == "setup":
        return 0
    
    remaining_space = current_width - board_width_with_borders
    if remaining_space > MIN_HISTORY_WIDTH:
        return min(remaining_space - 10, 200)
    return 0

def toggle_fullscreen():
    """Toggle fullscreen mode"""
    global fullscreen, screen
    fullscreen = not fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Stratego - with MARQ (F11 for Windowed)")
    else:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Stratego - with MARQ (F11 for Fullscreen)")

def get_font_size(base_size):
    """Scale font size based on window dimensions"""
    current_width, current_height = screen.get_size()
    scale_factor = min(current_width / BASE_WINDOW_WIDTH, current_height / BASE_WINDOW_HEIGHT)
    scale_factor = max(0.7, min(scale_factor, 2.5))
    return int(base_size * scale_factor)

def get_board_dimensions():
    """Calculate all board-related dimensions"""
    current_width, current_height = screen.get_size()
    tile_size = calculate_optimal_tile_size(current_width, current_height, show_history_panel)
    border_size = max(5, int(tile_size * 0.15))
    board_width = tile_size * BOARD_SIZE
    board_height = tile_size * BOARD_SIZE
    board_width_with_borders = board_width + border_size * 2
    board_height_with_borders = board_height + border_size * 2
    
    history_width = 0
    if show_history_panel and game_state != "setup":
        history_width = calculate_history_width(current_width, board_width_with_borders, game_state)
    
    available_board_width = current_width - history_width
    board_start_x = (available_board_width - board_width_with_borders) // 2
    board_start_y = (current_height - board_height_with_borders) // 2
    board_start_y = max(60, board_start_y)
    
    return {
        'tile_size': tile_size,
        'border_size': border_size,
        'board_width': board_width,
        'board_height': board_height,
        'board_width_with_borders': board_width_with_borders,
        'board_height_with_borders': board_height_with_borders,
        'board_start_x': board_start_x,
        'board_start_y': board_start_y,
        'history_width': history_width,
        'available_board_width': available_board_width
    }

def draw_rounded_rect(surface, color, rect, radius=10, alpha=255):
    """Draw a rounded rectangle with optional transparency"""
    if alpha < 255:
        temp_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, (*color, alpha), temp_surface.get_rect(), border_radius=radius)
        surface.blit(temp_surface, (rect.x, rect.y))
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_text(text, x, y, size, color, bold=False, center=True):
    """Draw text with modern styling"""
    font = pygame.font.Font(None, size)
    if bold:
        font.set_bold(True)
    text_surf = font.render(str(text), True, color)
    text_rect = text_surf.get_rect()
    if center:
        text_rect.center = (x, y)
    else:
        text_rect.topleft = (x, y)
    screen.blit(text_surf, text_rect)

def draw_shadow(rect, offset=4, alpha=60):
    """Draw a subtle shadow under an element"""
    shadow_rect = rect.copy()
    shadow_rect.x += offset
    shadow_rect.y += offset
    draw_rounded_rect(screen, (0, 0, 0), shadow_rect, radius=10, alpha=alpha)

def draw_button(text, x, y, width, height, action=None, primary=False, danger=False, disabled=False):
    """Draw a modern button with hover and press effects"""
    global button_states
    
    if action not in button_states:
        button_states[action] = {'hover': False, 'pressed': False}
    
    mouse_pos = pygame.mouse.get_pos()
    button_rect = pygame.Rect(x, y, width, height)
    is_hover = button_rect.collidepoint(mouse_pos) and not disabled
    
    # Update hover state
    button_states[action]['hover'] = is_hover
    
    # Determine colors
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
    
    # Draw shadow
    if not disabled:
        draw_shadow(button_rect, offset=3, alpha=40)
    
    # Draw button background
    draw_rounded_rect(screen, bg_color, button_rect, radius=8)
    
    # Draw border with glow effect on hover
    if is_hover and not disabled:
        border_color = PLAYER1_GLOW if primary else (PLAYER2_GLOW if danger else GREY_300)
        pygame.draw.rect(screen, border_color, button_rect, 2, border_radius=8)
    
    # Draw text
    draw_text(text, button_rect.centerx, button_rect.centery, get_font_size(18), text_color, bold=True)
    
    # Check for click
    if is_hover and not disabled:
        mouse_pressed = pygame.mouse.get_pressed()[0]
        if mouse_pressed and not button_states[action]['pressed']:
            button_states[action]['pressed'] = True
            return action
        elif not mouse_pressed:
            button_states[action]['pressed'] = False
    
    return None

def get_piece_symbol(piece):
    """Get Unicode symbol for piece"""
    symbols = {
        'F': '🚩', 'B': '💣', 'S': '🔍', '1': '⚔️', '2': '⚔️', '3': '⚔️',
        '4': '⚔️', '5': '⚔️', '6': '⚔️', '7': '⚔️', '8': '⚔️', '9': '⚔️'
    }
    return symbols.get(piece.symbol, '⚔️') if piece else ''

def draw_piece(piece, x, y, size, highlight=False, semi_transparent=False):
    """Draw a modern piece with team colors"""
    if not piece:
        return
    
    player = piece.owner
    
    # Determine colors based on player
    primary_color = get_player_color(player, 'primary')
    light_color = get_player_color(player, 'light')
    dark_color = get_player_color(player, 'dark')
    
    piece_rect = pygame.Rect(x + size//8, y + size//8, size - size//4, size - size//4)
    
    # Draw shadow
    if not semi_transparent:
        draw_shadow(piece_rect, offset=2, alpha=50)
    
    # Draw piece background
    alpha = 180 if semi_transparent else 255
    draw_rounded_rect(screen, primary_color, piece_rect, radius=6, alpha=alpha)
    
    # Draw highlight border
    if highlight:
        pygame.draw.rect(screen, TILE_SELECTED, piece_rect, 3, border_radius=6)
    else:
        # Draw subtle border
        pygame.draw.rect(screen, dark_color, piece_rect, 2, border_radius=6)
    
    # Draw piece rank/symbol
    font_size = int(size * 0.4)
    piece_text = piece.short() if not semi_transparent else "??"
    draw_text(piece_text, piece_rect.centerx, piece_rect.centery, font_size, WHITE, bold=True)
    
    # Draw small team indicator dot in corner
    dot_radius = max(3, size // 12)
    dot_x = piece_rect.right - dot_radius - 3
    dot_y = piece_rect.top + dot_radius + 3
    pygame.draw.circle(screen, light_color, (dot_x, dot_y), dot_radius)

def draw_board():
    """Draw the game board with modern styling"""
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    border_size = dims['border_size']
    board_start_x = dims['board_start_x']
    board_start_y = dims['board_start_y']
    
    # Draw board shadow
    board_rect = pygame.Rect(board_start_x, board_start_y, 
                             dims['board_width_with_borders'], 
                             dims['board_height_with_borders'])
    draw_shadow(board_rect, offset=6, alpha=80)
    
    # Draw board border with gradient effect
    draw_rounded_rect(screen, CARD_BG, board_rect, radius=12)
    pygame.draw.rect(screen, GREY_700, board_rect, 4, border_radius=12)
    
    # Draw tiles
    mouse_pos = pygame.mouse.get_pos()
    
    for r in range(BOARD_SIZE):
        for f in range(BOARD_SIZE):
            x = board_start_x + border_size + f * tile_size
            y = board_start_y + border_size + r * tile_size
            tile_rect = pygame.Rect(x, y, tile_size, tile_size)
            
            # Determine tile color
            is_lake = board.is_lake(r, f)
            is_light = (r + f) % 2 == 0
            
            if is_lake:
                tile_color = TILE_LAKE
            else:
                tile_color = TILE_LIGHT if is_light else TILE_DARK
            
            # Check for hover
            if tile_rect.collidepoint(mouse_pos) and not is_lake:
                tile_color = TILE_HOVER
            
            # Check for selection
            if selected == (r, f):
                tile_color = TILE_SELECTED
            
            # Check for valid move
            if selected and (r, f) in board.legal_moves_from(selected):
                # Draw subtle highlight for valid moves
                overlay_color = TILE_VALID
                pygame.draw.rect(screen, tile_color, tile_rect)
                draw_rounded_rect(screen, overlay_color, tile_rect, radius=4, alpha=100)
            else:
                pygame.draw.rect(screen, tile_color, tile_rect)
            
            # Draw grid lines
            pygame.draw.rect(screen, GREY_500, tile_rect, 1)
            
            # Draw piece if present
            piece = board.get((r, f))
            if piece:
                is_selected = selected == (r, f)
                # Determine visibility
                if game_state == "setup":
                    visible = (piece.owner == human_side)
                else:
                    visible = (piece.owner == human_side) or piece.revealed
                
                if visible:
                    draw_piece(piece, x, y, tile_size, highlight=is_selected)
                else:
                    # Draw hidden piece
                    draw_piece(piece, x, y, tile_size, highlight=is_selected, semi_transparent=True)
    
    # Draw coordinates
    coord_size = get_font_size(12)
    coord_color = GREY_300
    
    for i in range(BOARD_SIZE):
        # Files (a-j)
        file_x = board_start_x + border_size + i * tile_size + tile_size // 2
        file_y_top = board_start_y + border_size - 18
        file_y_bottom = board_start_y + border_size + BOARD_SIZE * tile_size + 15
        draw_text(FILES[i], file_x, file_y_top, coord_size, coord_color, bold=True)
        draw_text(FILES[i], file_x, file_y_bottom, coord_size, coord_color, bold=True)
        
        # Ranks (1-10)
        rank_y = board_start_y + border_size + i * tile_size + tile_size // 2
        rank_x_left = board_start_x + border_size - 18
        rank_x_right = board_start_x + border_size + BOARD_SIZE * tile_size + 18
        draw_text(str(10 - i), rank_x_left, rank_y, coord_size, coord_color, bold=True)
        draw_text(str(10 - i), rank_x_right, rank_y, coord_size, coord_color, bold=True)

def draw_move_history():
    """Draw move history panel"""
    dims = get_board_dimensions()
    history_width = dims['history_width']
    
    if history_width == 0 or not show_history_panel:
        return
    
    current_width, current_height = screen.get_size()
    history_x = dims['board_start_x'] + dims['board_width_with_borders'] + 40
    history_y = 60
    history_height = current_height - 200
    
    # Draw history panel background
    history_rect = pygame.Rect(history_x, history_y, history_width - 10, history_height)
    draw_rounded_rect(screen, CARD_BG, history_rect, radius=10)
    pygame.draw.rect(screen, GREY_700, history_rect, 2, border_radius=10)
    
    # Draw header
    header_height = 40
    header_rect = pygame.Rect(history_x, history_y, history_width - 10, header_height)
    draw_rounded_rect(screen, CARD_LIGHT, header_rect, radius=10)
    draw_text("Move History", history_rect.centerx, history_y + header_height // 2, 
              get_font_size(14), WHITE, bold=True)
    
    # Draw moves
    move_y = history_y + header_height + 10
    move_height = get_font_size(14) + 4
    visible_moves = int((history_height - header_height - 20) / move_height)
    
    start_index = max(0, len(move_history) - visible_moves)
    for i, move in enumerate(move_history[start_index:], start=start_index + 1):
        move_color = get_player_color(1 if i % 2 == 1 else 2, 'light')
        draw_text(f"{i}. {move}", history_rect.centerx, move_y, 
                  get_font_size(12), move_color, center=True)
        move_y += move_height

def draw_lost_pieces_tracker():
    """Draw captured pieces tracker"""
    dims = get_board_dimensions()
    
    # Position at top of screen
    tracker_width = dims['board_width_with_borders']
    tracker_x = dims['board_start_x']
    tracker_y = 10
    tracker_height = 40
    
    # Player 1 captured pieces (left side)
    p1_rect = pygame.Rect(tracker_x, tracker_y, tracker_width // 2 - 5, tracker_height)
    draw_rounded_rect(screen, CARD_BG, p1_rect, radius=8)
    pygame.draw.rect(screen, get_player_color(1, 'dark'), p1_rect, 2, border_radius=8)
    
    p1_text = f"Blue: {len(lost_pieces_player1)} lost"
    draw_text(p1_text, p1_rect.centerx, p1_rect.centery, get_font_size(12), 
              get_player_color(1, 'light'), bold=True)
    
    # Player 2 captured pieces (right side)
    p2_rect = pygame.Rect(tracker_x + tracker_width // 2 + 5, tracker_y, 
                          tracker_width // 2 - 5, tracker_height)
    draw_rounded_rect(screen, CARD_BG, p2_rect, radius=8)
    pygame.draw.rect(screen, get_player_color(2, 'dark'), p2_rect, 2, border_radius=8)
    
    p2_text = f"Red: {len(lost_pieces_player2)} lost"
    draw_text(p2_text, p2_rect.centerx, p2_rect.centery, get_font_size(12), 
              get_player_color(2, 'light'), bold=True)

def handle_click(pos):
    """Handle mouse click on board"""
    global selected, current_player, message, game_state
    
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    border_size = dims['border_size']
    board_start_x = dims['board_start_x']
    board_start_y = dims['board_start_y']
    
    x, y = pos
    f = (x - board_start_x - border_size) // tile_size
    r = (y - board_start_y - border_size) // tile_size
    
    if not (0 <= r < BOARD_SIZE and 0 <= f < BOARD_SIZE):
        return
    
    sq = (r, f)
    
    if game_state == "setup":
        piece = board.get(sq)
        if piece and piece.owner == human_side:
            if selected is None:
                selected = sq
            else:
                if board.get(selected) and board.get(selected).owner == human_side:
                    # Swap pieces
                    p1 = board.get(selected)
                    p2 = board.get(sq)
                    board.set(selected, p2)
                    board.set(sq, p1)
                    selected = None
        else:
            selected = None
    
    elif game_state == "play":
        if selected is None:
            piece = board.get(sq)
            if piece and piece.owner == current_player:
                if vs_bot and current_player != human_side:
                    return
                selected = sq
        else:
            if sq in board.legal_moves_from(selected):
                # Execute move
                src = selected
                dst = sq
                
                attacker = board.get(src)
                defender = board.get(dst)
                
                move_notation = f"{FILES[src[1]]}{10-src[0]} to {FILES[dst[1]]}{10-dst[0]}"
                
                if defender:
                    move_notation += f" ({attacker.short()} vs {defender.short()})"
                
                msg, winner = board.move_and_resolve(src, dst, human_side if vs_bot else None)
                
                # Update AAREN history for DQN piece inference
                if vs_bot and bot_logic is not None:
                    bot_logic.update_from_opponent_move(board, src, dst)
                
                # Track lost pieces
                piece_after_dst = board.get(dst)
                piece_after_src = board.get(src)
                
                if defender and not piece_after_dst:
                    # Defender was eliminated
                    if defender.owner == 1:
                        lost_pieces_player1.append(defender)
                    else:
                        lost_pieces_player2.append(defender)
                elif defender and piece_after_dst and piece_after_dst.owner != defender.owner:
                    # Defender was eliminated
                    if defender.owner == 1:
                        lost_pieces_player1.append(defender)
                    else:
                        lost_pieces_player2.append(defender)
                
                if attacker and not piece_after_src and not piece_after_dst:
                    # Attacker was eliminated (both eliminated scenario)
                    if attacker.owner == 1:
                        lost_pieces_player1.append(attacker)
                    else:
                        lost_pieces_player2.append(attacker)
                
                move_history.append(move_notation)
                
                if winner is not None:
                    game_state = "game_over"
                    if vs_bot:
                        if winner == human_side:
                            message = "You Win! "
                        else:
                            message = "Bot Wins!"
                    else:
                        message = f"Player {winner} Wins! "
                else:
                    current_player = 2 if current_player == 1 else 1
                    if vs_bot:
                        if current_player == human_side:
                            message = "Your turn!"
                        else:
                            message = "Bot is thinking..."
                    else:
                        message = f"Player {current_player}'s turn"
                
                selected = None
            else:
                piece = board.get(sq)
                if piece and piece.owner == current_player:
                    selected = sq
                else:
                    selected = None

def bot_turn():
    """Execute bot's turn"""
    global current_player, message, game_state, selected
    
    time.sleep(0.5)
    
    if bot_logic is None:
        return
    
    move = bot_logic.choose_move(board, current_player)
    
    if move is None:
        message = "Bot has no valid moves!"
        current_player = human_side
        return
    
    src, dst = move
    
    attacker = board.get(src)
    defender = board.get(dst)
    
    move_notation = f"{FILES[src[1]]}{10-src[0]} to {FILES[dst[1]]}{10-dst[0]}"
    
    if defender:
        move_notation += f" ({attacker.short()} vs {defender.short()})"
    
    msg, winner = board.move_and_resolve(src, dst, human_side)
    
    # Track lost pieces
    piece_after_dst = board.get(dst)
    piece_after_src = board.get(src)
    
    if defender and not piece_after_dst:
        # Defender was eliminated
        if defender.owner == 1:
            lost_pieces_player1.append(defender)
        else:
            lost_pieces_player2.append(defender)
    elif defender and piece_after_dst and piece_after_dst.owner != defender.owner:
        # Defender was eliminated
        if defender.owner == 1:
            lost_pieces_player1.append(defender)
        else:
            lost_pieces_player2.append(defender)
    
    if attacker and not piece_after_src and not piece_after_dst:
        # Attacker was eliminated
        if attacker.owner == 1:
            lost_pieces_player1.append(attacker)
        else:
            lost_pieces_player2.append(attacker)
    
    move_history.append(move_notation)
    
    if winner is not None:
        game_state = "game_over"
        message = "Bot Wins!"
    else:
        current_player = human_side
        message = "Your turn!"

# Main game loop
clock = pygame.time.Clock()

while running:
    current_width, current_height = screen.get_size()
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    border_size = dims['border_size']
    history_width = dims['history_width']
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                toggle_fullscreen()
            elif event.key == pygame.K_h:
                show_history_panel = not show_history_panel
            elif event.key == pygame.K_ESCAPE:
                if game_state == "play":
                    game_state = "pause"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if game_state in ["setup", "play"]:
                handle_click(event.pos)
    
    # Clear screen with dark background
    screen.fill(BACKGROUND)
    
    if game_state == "menu":
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2
        
        # Title 
        title_y = get_font_size(150)
        draw_text("STRATEGO", center_x, title_y, get_font_size(72), PLAYER1_PRIMARY, bold=True)
        draw_text("with MARQ", center_x, title_y + get_font_size(60), 
                  get_font_size(24), PLAYER2_PRIMARY, bold=True)
        
        # Buttons
        button_width = int(menu_area_width * 0.4)
        button_width = max(250, min(button_width, 400))
        button_height = get_font_size(35) + 20
        button_spacing = button_height + 25
        start_y = get_font_size(220)
        
        if draw_button("Play vs Bot", center_x - button_width//2, start_y, 
                      button_width, button_height, action="bot", primary=True) == "bot":
            board = Board()
            auto_setup(board, 2)
            try:
                setup_agent = StrategicSetupAgent(load_path=setup_model_path)
                board = setup_agent.setup_side(board, 1)
            except:
                auto_setup(board, 1)
            
            vs_bot = True
            human_side = 1
            current_player = 1
            message = "Arrange your pieces"
            game_state = "setup"
            move_history.clear()
            lost_pieces_player1.clear()
            lost_pieces_player2.clear()
            
            try:
                bot_logic = DQNBotLogic(agent_model_path, player_id=2)
                bot_logic.reset()
            except Exception as e:
                print(f"Failed to load MARQ Bot: {e}")
                bot_logic = None
            
        if draw_button("2-Player Mode", center_x - button_width//2, start_y + button_spacing, 
                      button_width, button_height, action="2p") == "2p":
            board = Board()
            auto_setup(board, 1)
            auto_setup(board, 2)
            vs_bot = False
            human_side = None
            current_player = 1
            message = "Player 1's turn"
            game_state = "play"
            move_history.clear()
            lost_pieces_player1.clear()
            lost_pieces_player2.clear()
            
        if draw_button("Download History", center_x - button_width//2, start_y + button_spacing * 2, 
                      button_width, button_height, action="download") == "download":
            with open('stratego_history.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Move Number', 'Notation'])
                for i, move in enumerate(move_history):
                    writer.writerow([i + 1, move])
            message = "History saved to stratego_history.csv"
            
        if draw_button("Quit", center_x - button_width//2, start_y + button_spacing * 3, 
                      button_width, button_height, action="quit", danger=True) == "quit":
            running = False
        
        # Hint text
        hint_y = current_height - 30
        draw_text("Press F11 for fullscreen mode", center_x, hint_y, get_font_size(14), GREY_500)

    elif game_state == "setup":
        draw_board()
        
        dims = get_board_dimensions()
        board_width_with_borders = dims['board_width_with_borders']
        board_start_x = dims['board_start_x']
        board_center_x = board_start_x + board_width_with_borders // 2
        
        # Start button
        button_width = 200
        button_height = get_font_size(30) + 16
        button_x = board_center_x - button_width // 2
        button_y = dims['board_start_y'] + dims['board_height'] + 50
        
        action = draw_button("START GAME", button_x, button_y, 
                            button_width, button_height, action="start", primary=True)
        
        # Instruction text
        #instruction_y = button_y - 10
        #draw_text("Arrange pieces, then click START!", board_center_x, instruction_y, 
                  #get_font_size(15), GREY_300)
        
        # Setup banner
        #setup_info_y = dims['board_start_y'] - 100
        setup_banner_height = 30
        setup_banner_width = board_width_with_borders
        setup_banner_x = dims['board_start_x']
        setup_banner_center_x = setup_banner_x + setup_banner_width // 2
        setup_banner = pygame.Rect(setup_banner_x, 10,   # Put it at top of screen (y=10)
                           setup_banner_width, setup_banner_height)
        draw_rounded_rect(screen, ACCENT, setup_banner, 10)
        pygame.draw.rect(screen, PLAYER1_DARK, setup_banner, 2, border_radius=10)
        
        if vs_bot:
            setup_text = "Setup Phase - Arrange your pieces | Bot is ready"
        else:
            setup_text = f"Setup Phase - Arrange Player {human_side}'s pieces"
        
        draw_text(setup_text, setup_banner_center_x, 1050 + setup_banner_height//2, 
          get_font_size(14), WHITE, bold=True)
        if action == "start":
            game_state = "play"
            if vs_bot:
                message = "Your turn! Click on your piece, then click where to move."
            else:
                message = f"Player {current_player}'s turn"

    elif game_state == "play":
        draw_board()
        draw_lost_pieces_tracker()
        draw_move_history()
        
        if vs_bot and current_player != human_side:
            bot_turn()
        
        # Message bar
        msg_bar_y = dims['board_start_y'] + dims['board_height'] + border_size + 40
        msg_bar_height = get_font_size(25) + 15
        msg_bar_width = dims['board_width_with_borders']
        msg_bar_x = dims['board_start_x']
        msg_bar_center_x = msg_bar_x + msg_bar_width // 2
        
        msg_bar = pygame.Rect(msg_bar_x, msg_bar_y, msg_bar_width, msg_bar_height)
        draw_rounded_rect(screen, CARD_BG, msg_bar, 10)
        
        # Colored top border based on current player
        player_color = get_player_color(current_player, 'primary')
        highlight_rect = pygame.Rect(msg_bar_x, msg_bar_y, msg_bar_width, 4)
        draw_rounded_rect(screen, player_color, highlight_rect, 10)
        
        pygame.draw.rect(screen, GREY_700, msg_bar, 2, border_radius=10)
        
        draw_text(message, msg_bar_center_x, msg_bar_y + msg_bar_height // 2, 
                  get_font_size(14), WHITE, bold=True)
        
        # History toggle hint
        if dims['history_width'] > 0:
            hint_text = f"Press H to {'show' if not show_history_panel else 'hide'} history"
            draw_text(hint_text, msg_bar_center_x, msg_bar_y + msg_bar_height + 20, 
                      get_font_size(12), GREY_500)

    elif game_state == "pause":
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2



        
        draw_text("PAUSED", center_x, get_font_size(100), get_font_size(72), ACCENT, bold=True)
        
        button_width = int(menu_area_width * 0.35)
        button_width = max(200, min(button_width, 300))
        button_height = get_font_size(35) + 20
        button_spacing = button_height + 25
        button_y = get_font_size(200)
        
        if draw_button("Resume", center_x - button_width//2, button_y, 
                      button_width, button_height, action="resume", primary=True) == "resume":
            game_state = "play"
        if draw_button("Quit to Menu", center_x - button_width//2, button_y + button_spacing, 
                      button_width, button_height, action="menu") == "menu":
            game_state = "menu"
        if draw_button("Exit", center_x - button_width//2, button_y + button_spacing * 2, 
                      button_width, button_height, action="exit", danger=True) == "exit":
            running = False
    
    elif game_state == "game_over":
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2
        
        draw_text("GAME OVER", center_x, get_font_size(100), get_font_size(72), PLAYER2_PRIMARY, bold=True)
        draw_text(message, center_x, get_font_size(180), get_font_size(36), PLAYER1_PRIMARY, bold=True)
        
        button_width = int(menu_area_width * 0.35)
        button_width = max(200, min(button_width, 300))
        button_height = get_font_size(35) + 20
        button_y = get_font_size(260)
        
        if draw_button("Back to Menu", center_x - button_width//2, button_y, 
                      button_width, button_height, action="menu", primary=True) == "menu":
            board = Board()
            auto_setup(board, 1)
            auto_setup(board, 2)
            current_player = 1
            selected = None
            message = "Welcome to Stratego - with MARQ!"
            move_history.clear()
            game_state = "menu"
            vs_bot = False
            human_side = 1
            bot_logic = None
            setup_agent = None
            button_states.clear()
            show_history_panel = True
            lost_pieces_player1.clear()
            lost_pieces_player2.clear()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()