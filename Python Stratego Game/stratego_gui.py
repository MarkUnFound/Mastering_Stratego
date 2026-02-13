"""
Stratego with MARQ - Maximally Optimized Visual Interface
Features:
- Strategic Setup Agent for intelligent piece placement
- RADICAL space utilization - board takes maximum screen real estate
- Ultra-compact UI with minimal padding and margins
- Dynamic scaling up to 3x for very large screens
- History panel with toggle functionality for maximum board space
- Responsive design optimized for playability and visual clarity
"""

import pygame
import time
import random
import csv
import os
from stratego import Board, auto_setup, side_name, FILES, Piece
from dqn_bot_logic import DQNBotLogic
from bot_logic import EnhancedBotLogic  # Fallback
from setup_agent_integration import StrategicSetupAgent

pygame.init()

# Get screen info for dynamic scaling
display_info = pygame.display.Info()
SCREEN_WIDTH = display_info.current_w
SCREEN_HEIGHT = display_info.current_h

# Base dimensions - RADICALLY optimized for maximum board space utilization
BOARD_SIZE = 10
MIN_TILE_SIZE = 30  # Further reduced to allow maximum boards
MAX_TILE_SIZE = 75  # Further reduced to ensure top row pieces are fully visible
BASE_BORDER_SIZE = 10  # Minimal borders
MIN_HISTORY_WIDTH = 150  # Very narrow history when needed
PREFERRED_HISTORY_WIDTH = 200  # Still narrow
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
fullscreen = False  # Fullscreen mode flag
show_history_panel = True  # Toggle for history panel visibility
lost_pieces_player1 = []  # Track lost pieces for player 1
lost_pieces_player2 = []  # Track lost pieces for player 2

# Button animation state
button_states = {}  # Track button press states for animations

# Setup board
board = Board()
auto_setup(board, 1)
auto_setup(board, 2)

# Determine model paths - check multiple possible locations
# Priority: Rainbow DQN checkpoints (79-channel) > old StrategoNet (dqn_agent_final.pth)
model_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    # Rainbow DQN checkpoints (compatible with DQNBotLogic)
    os.path.join(model_dir, '..', 'History', '12', 'agent1_rainbow_episode_8000.pth'),
    os.path.join(model_dir, '..', 'Modular Stratego', 'dqn_models', 'agent1_rainbow_latest.pth'),
    # Legacy StrategoNet checkpoint (fallback for EnhancedBotLogic)
    os.path.join(model_dir, 'dqn_agent_final.pth'),
    os.path.join(model_dir, 'user_input_files', 'dqn_agent_final.pth'),
    'dqn_models/dqn_agent_final.pth',
    os.path.join(model_dir, '..', 'dqn_models', 'dqn_agent_final.pth'),
]

agent_model_path = None
for path in possible_paths:
    if os.path.exists(path):
        agent_model_path = os.path.abspath(path)
        print(f"Found agent model at: {agent_model_path}")
        break

if agent_model_path is None:
    agent_model_path = os.path.join(model_dir, 'dqn_agent_final.pth')
    print(f"⚠️  Warning: dqn_agent_final.pth not found in any expected location, using: {agent_model_path}")

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

# Initialize window - default to windowed mode for maximum board space
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
fullscreen = False  # Set default to windowed
pygame.display.set_caption("Stratego with MARQ (F11 for Fullscreen Mode)")

# Enhanced color palette
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
BLUE = (30, 144, 255)
LIGHT_BLUE = (100, 180, 255)
DARK_BLUE = (20, 90, 170)
RED = (220, 50, 50)
DARK_RED = (150, 30, 30)
GREY = (200, 200, 200)
LIGHT_GREY = (240, 240, 240)
DARK_GREY = (80, 80, 80)
GREEN = (50, 180, 50)
LIGHT_GREEN = (100, 220, 100)
DARK_GREEN = (30, 120, 30)
YELLOW = (255, 215, 0)
ORANGE = (255, 165, 0)
LAKE_BLUE = (100, 150, 255)
BACKGROUND = (245, 245, 250)
ACCENT = (70, 130, 180)
BUTTON_SHADOW = (0, 0, 0, 80)

# Calculate optimal tile size based on available space
def calculate_optimal_tile_size(current_width, current_height, show_history=True):
    """Calculate optimal tile size to maximize board size while maintaining aspect ratio"""
    # MINIMAL padding for maximum board space - ULTRA-AGGRESSIVE space utilization
    base_vertical_padding = 20  # Even more reduced
    base_horizontal_padding = 10  # Even more reduced
    
    # Calculate scale based on screen size - allow much larger scaling
    scale_factor = min(current_width / BASE_WINDOW_WIDTH, current_height / BASE_WINDOW_HEIGHT)
    scale_factor = max(0.9, min(scale_factor, 3.0))  # Allow 3x scaling for very large screens
    
    vertical_padding = int(base_vertical_padding * scale_factor)
    horizontal_padding = int(base_horizontal_padding * scale_factor)
    
    # Ensure minimal padding - ultra-minimal
    vertical_padding = max(20, vertical_padding)
    horizontal_padding = max(10, horizontal_padding)
    
    # Available space for the board - calculate based on whether history is shown
    available_height = current_height - vertical_padding
    if show_history:
        # Reserve space for history panel but make it ULTRA narrow
        history_reserve = 120  # Even smaller than before
        available_width = current_width - horizontal_padding - history_reserve
    else:
        # Use almost full width when history is hidden
        available_width = current_width - horizontal_padding
    
    # Calculate tile size based on available space - be very aggressive
    tile_from_height = available_height // BOARD_SIZE
    tile_from_width = available_width // BOARD_SIZE
    
    # Use the smaller dimension to maintain square tiles, but within limits
    optimal_tile = min(tile_from_height, tile_from_width)
    optimal_tile = max(MIN_TILE_SIZE, min(optimal_tile, MAX_TILE_SIZE))
    
    return optimal_tile

def calculate_history_width(current_width, board_width_with_borders, game_state=None):
    """Calculate history panel width, using remaining horizontal space"""
    # During setup, completely hide history panel for maximum board space
    if game_state == "setup":
        return 0  # Hide history during setup
    
    remaining_space = current_width - board_width_with_borders
    # Use VERY narrow width for history panel to maximize board space
    if remaining_space > MIN_HISTORY_WIDTH:
        # Use minimal portion of remaining space to prioritize board size
        return min(remaining_space - 10, 180)  # Reduced from 250 to 180
    return 0  # Don't show history if not enough space

def toggle_fullscreen():
    """Toggle between fullscreen and windowed mode"""
    global fullscreen, screen
    fullscreen = not fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Stratego with MARQ (F11 for Windowed Mode)")
    else:
        # Use full screen dimensions in windowed mode (maximized window)
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Stratego with MARQ (F11 for Fullscreen Mode)")

# --- Helper functions ---
def get_font_size(base_size):
    """Get scaled font size with improved fullscreen scaling"""
    current_width, current_height = screen.get_size()
    base_width = BASE_WINDOW_WIDTH
    base_height = BASE_WINDOW_HEIGHT
    
    # Calculate scale based on available space, allowing more flexibility in fullscreen
    width_scale = current_width / base_width
    height_scale = current_height / base_height
    
    # Use the smaller scale but allow larger scaling in fullscreen for better visibility
    scale = min(width_scale, height_scale)
    
    # Allow up to 1.8x scaling for very large screens, minimum 0.8x for small screens
    scale = max(0.8, min(scale, 1.8))
    
    # Ensure minimum readable size and reasonable maximum
    scaled_size = int(base_size * scale)
    return max(10, min(scaled_size, base_size * 2))  # Cap at 2x original size

def draw_text(text, x, y, size=20, color=BLACK, center=True, bold=False):
    """Draw text with dynamic scaling"""
    font_size = get_font_size(size)
    font = pygame.font.SysFont('Arial', font_size, bold=bold)
    surf = font.render(str(text), True, color)
    rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
    screen.blit(surf, rect)
    return rect

def draw_rounded_rect(surface, color, rect, radius, border=0):
    """Draw a rounded rectangle with tapered/curved borders"""
    if radius <= 0:
        pygame.draw.rect(surface, color, rect, border)
        return
    
    # Create a surface with per-pixel alpha for smooth curves
    temp_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    
    if border > 0:
        # Draw border as rounded rect
        pygame.draw.rect(temp_surface, color, temp_surface.get_rect(), border, border_radius=radius)
    else:
        # Draw filled rounded rect
        pygame.draw.rect(temp_surface, color, temp_surface.get_rect(), border_radius=radius)
    
    surface.blit(temp_surface, rect.topleft)

def get_board_dimensions():
    """Get current board dimensions for consistent layout with MAXIMUM space utilization"""
    current_width, current_height = screen.get_size()
    
    # Determine if we should show history based on game state and user preference
    show_history = game_state not in ["setup", "menu"] and show_history_panel
    
    tile_size = calculate_optimal_tile_size(current_width, current_height, show_history)
    border_size = max(8, int(tile_size * 0.2))  # Minimal borders to maximize board space
    board_width = tile_size * BOARD_SIZE
    board_width_with_borders = board_width + border_size * 2
    total_board_height = board_width + border_size * 2
    
    # MAXIMIZE board positioning - ensure top row is visible with adequate top margin
    board_start_y = (current_height - total_board_height) // 2
    # Ensure top margin is sufficient to see top pieces (account for labels and piece rendering)
    label_space = get_font_size(18) + 10
    piece_height = int(tile_size * 0.35) * 2  # Account for piece circle radius
    top_margin_needed = border_size + label_space + piece_height + 20  # Extra space for pieces at top
    board_start_y = max(top_margin_needed, board_start_y)  # Ensure top pieces are fully visible
    board_start_y = min(board_start_y, current_height - total_board_height - border_size - 80)  # Bottom margin for message bar
    
    # Center board in the window - left panel (piece tracker) larger than right panel
    spacing = 20  # Increased spacing to prevent overlap
    
    if game_state == "play" and show_history:
        # Calculate panel widths - left panel larger than right
        # Use more of the available horizontal space
        board_space_needed = board_width_with_borders + spacing * 3  # Board + spacing on both sides + extra buffer
        available_for_panels = current_width - board_space_needed
        
        # Give 60% of space to left panel (piece tracker), 40% to right panel (history)
        # Minimum widths: left 280px, right 200px
        left_panel_base = max(280, int(available_for_panels * 0.6))
        right_panel_base = max(200, int(available_for_panels * 0.4))
        
        # Ensure we don't exceed available space
        total_panels = left_panel_base + right_panel_base
        if total_panels > available_for_panels:
            # Scale down proportionally
            scale = available_for_panels / total_panels
            left_panel_base = int(left_panel_base * scale)
            right_panel_base = int(right_panel_base * scale)
        
        lost_tracker_width = left_panel_base
        history_width = right_panel_base
    elif game_state == "play" and not show_history:
        # Only lost pieces tracker shown - use even more space
        lost_tracker_width = max(280, int((current_width - board_width_with_borders - spacing * 2) * 0.4))
        history_width = 0
    else:
        # Setup or menu - no panels
        lost_tracker_width = 0
        history_width = 0
    
    # Calculate total width used by side panels with margins
    left_panel_width = lost_tracker_width + 20 if lost_tracker_width > 0 else 0  # 20px left margin
    right_panel_width = history_width + 15 if history_width > 0 else 0  # 15px right margin
    
    # Available width for board and spacing
    available_width = current_width - left_panel_width - right_panel_width
    
    # Center the board in available space with proper spacing
    board_start_x = left_panel_width + spacing + (available_width - board_width_with_borders) // 2
    
    # Ensure panels don't overlap - verify spacing
    left_panel_end = left_panel_width
    right_panel_start = current_width - right_panel_width
    
    # Additional check: ensure minimum gap between left panel and board
    min_gap = spacing
    if board_start_x - left_panel_end < min_gap:
        # Adjust board position to maintain minimum gap
        board_start_x = left_panel_end + min_gap
    
    # Ensure minimum gap between board and right panel
    board_end = board_start_x + board_width_with_borders
    if right_panel_start - board_end < min_gap:
        # Adjust right panel to maintain minimum gap
        excess = min_gap - (right_panel_start - board_end)
        history_width = max(200, history_width - excess) if history_width > 0 else 0
        right_panel_width = history_width + 15 if history_width > 0 else 0
        right_panel_start = current_width - right_panel_width
    
    return {
        'tile_size': tile_size,
        'border_size': border_size,
        'board_width': board_width,
        'board_width_with_borders': board_width_with_borders,
        'board_start_y': board_start_y,
        'board_start_x': board_start_x,
        'board_height': tile_size * BOARD_SIZE,
        'total_board_height': total_board_height,
        'history_width': history_width
    }

def draw_button(text, x, y, w, h, action=None, primary=False):
    """Draw an enhanced button with smooth animations and curved borders"""
    global button_states
    
    mouse = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]
    rect = pygame.Rect(x, y, w, h)
    is_hovered = rect.collidepoint(mouse)
    
    # Get button state for animation
    button_key = f"{x},{y}"
    is_pressed = button_key in button_states and button_states[button_key] > 0
    
    # Update button press state
    if is_hovered and mouse_pressed and action:
        button_states[button_key] = 5  # Press animation frames
    elif button_key in button_states:
        button_states[button_key] = max(0, button_states[button_key] - 1)
    
    # Enhanced color scheme with smooth transitions
    if action == "start" and game_state == "setup":
        base_color = DARK_GREEN
        hover_color = GREEN
        highlight_color = LIGHT_GREEN
    elif primary:
        base_color = DARK_BLUE
        hover_color = BLUE
        highlight_color = LIGHT_BLUE
    else:
        base_color = DARK_GREY
        hover_color = GREY
        highlight_color = LIGHT_GREY
    
    # Animated color based on state
    if is_pressed:
        current_color = tuple(max(0, c - 30) for c in base_color)  # Darker when pressed
        offset_y = 2  # Push down effect
    elif is_hovered:
        current_color = hover_color
        offset_y = -1  # Lift up effect on hover
    else:
        current_color = base_color
        offset_y = 0
    
    # Shadow (moves with button)
    shadow_offset = 4 if not is_pressed else 2
    shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset + offset_y, w, h)
    draw_rounded_rect(screen, (0, 0, 0, 60), shadow_rect, 15)
    
    # Main button (with offset for press/hover animation)
    button_rect = pygame.Rect(x, y + offset_y, w, h)
    
    # Draw button fill
    draw_rounded_rect(screen, current_color, button_rect, 15)
    
    # Gradient highlight effect on hover
    if is_hovered and not is_pressed:
        highlight_rect = pygame.Rect(button_rect.x + 2, button_rect.y + 2, 
                                     button_rect.w - 4, int(button_rect.h * 0.4))
        highlight_surface = pygame.Surface((highlight_rect.w, highlight_rect.h), pygame.SRCALPHA)
        for i in range(highlight_rect.h):
            alpha = int(40 * (1 - i / highlight_rect.h))
            color_with_alpha = (*highlight_color, alpha)
            pygame.draw.line(highlight_surface, color_with_alpha, 
                           (0, i), (highlight_rect.w, i))
        screen.blit(highlight_surface, highlight_rect.topleft)
    
    # Border with curved edges
    border_color = tuple(min(255, c + 60) for c in current_color)
    draw_rounded_rect(screen, border_color, button_rect, 15, border=2)
    
    # Inner shadow for depth (subtle)
    if not is_pressed:
        inner_shadow = pygame.Rect(button_rect.x + 3, button_rect.y + 3, 
                                   button_rect.w - 6, 3)
        draw_rounded_rect(screen, (0, 0, 0, 30), inner_shadow, 8)
    
    # Text (slightly offset when pressed for press effect)
    text_offset = 1 if is_pressed else 0
    font_size = get_font_size(18)
    font = pygame.font.SysFont('Arial', font_size, bold=True)
    text_surf = font.render(str(text), True, WHITE)
    text_rect = text_surf.get_rect(center=(button_rect.centerx, button_rect.centery + text_offset))
    
    # Text shadow for readability
    shadow_text = font.render(str(text), True, (0, 0, 0, 100))
    screen.blit(shadow_text, (text_rect.x + 1, text_rect.y + 1))
    screen.blit(text_surf, text_rect)
    
    if is_hovered and mouse_pressed and action and not is_pressed:
        return action
    return None

def draw_board():
    """Draw board with optimized sizing"""
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    border_size = dims['border_size']
    board_start_x = dims['board_start_x']
    board_start_y = dims['board_start_y']
    
    # Draw labels with enhanced styling
    label_font_size = get_font_size(18)
    label_font = pygame.font.SysFont('Arial', label_font_size, bold=True)
    
    # Column labels (letters) - above board with MINIMAL spacing
    for i in range(BOARD_SIZE):
        label_x = board_start_x + i * tile_size + tile_size // 2
        label_y = board_start_y - border_size // 2 - 10 # Minimal spacing from board
        text_surf = label_font.render(FILES[i], True, DARK_BLUE)
        text_rect = text_surf.get_rect(center=(label_x, label_y))
        screen.blit(text_surf, text_rect)
    
    # Row labels (numbers) - left of board with MINIMAL spacing
    for i in range(BOARD_SIZE):
        label_x = board_start_x - border_size // 2 - 10  # Minimal spacing from board
        label_y = board_start_y + i * tile_size + tile_size // 2
        text_surf = label_font.render(str(i + 1), True, DARK_BLUE)
        text_rect = text_surf.get_rect(center=(label_x, label_y))
        screen.blit(text_surf, text_rect)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(board_start_x + c*tile_size, board_start_y + r*tile_size, tile_size, tile_size)
            
            # Lake tiles with gradient effect
            if board.is_lake(r, c):
                draw_rounded_rect(screen, LAKE_BLUE, rect, 3)
                pygame.draw.rect(screen, DARK_BLUE, rect, 1)
                # Wave pattern hint
                for offset in [3, 6]:
                    pygame.draw.line(screen, tuple(min(255, c + 30) for c in LAKE_BLUE), 
                                   (rect.left, rect.top + offset), (rect.right, rect.top + offset), 1)
                continue

            # Alternate square colors for better visibility
            square_color = LIGHT_GREY if (r + c) % 2 == 0 else WHITE
            pygame.draw.rect(screen, square_color, rect)
            pygame.draw.rect(screen, DARK_GREY, rect, 1)
            
            piece = board.get((r, c))
            
            if not piece:
                continue

            color = RED if piece.owner == 1 else BLACK
            owner_color_dark = DARK_RED if piece.owner == 1 else (40, 40, 40)

            # Visibility rules
            if game_state == "setup":
                visible = (piece.owner == human_side)
            else:
                visible = (piece.owner == human_side) or piece.revealed

            # Selection highlight with glow
            if selected == (r, c):
                pygame.draw.rect(screen, GREEN, rect, 4)
                # Glow effect
                glow_rect = pygame.Rect(rect.x - 3, rect.y - 3, rect.w + 6, rect.h + 6)
                glow_surface = pygame.Surface((glow_rect.w, glow_rect.h), pygame.SRCALPHA)
                for i in range(6):
                    alpha = int(30 * (1 - i / 6))
                    pygame.draw.rect(glow_surface, (*GREEN, alpha), 
                                   pygame.Rect(i, i, glow_rect.w - 2*i, glow_rect.h - 2*i), 2)
                screen.blit(glow_surface, glow_rect.topleft)

            # Draw piece with shadow
            circle_radius = int(tile_size * 0.35)
            shadow_offset = 2
            # Shadow
            pygame.draw.circle(screen, (0, 0, 0, 120), 
                             (rect.centerx + shadow_offset, rect.centery + shadow_offset), 
                             circle_radius)
            # Main piece circle
            pygame.draw.circle(screen, owner_color_dark, rect.center, circle_radius)
            pygame.draw.circle(screen, color, rect.center, circle_radius - 2)
            
            # Piece text
            piece_font_size = get_font_size(int(tile_size * 0.3))
            piece_font = pygame.font.SysFont('Arial', piece_font_size, bold=True)
            text = piece_font.render(str(piece.short()) if visible else "??", True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

def get_square_from_mouse(pos):
    """Get board square from mouse position - uses same calculation as draw_board"""
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    board_start_x = dims['board_start_x']
    board_start_y = dims['board_start_y']
    
    x, y = pos
    if x < board_start_x or y < board_start_y or x > board_start_x + tile_size * BOARD_SIZE or y > board_start_y + tile_size * BOARD_SIZE:
        return None
    c = int((x - board_start_x) // tile_size)
    r = int((y - board_start_y) // tile_size)
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        return (r, c)
    return None

def draw_lost_pieces_tracker():
    """Draw lost pieces counter on the left side of the screen with enhanced visuals"""
    dims = get_board_dimensions()
    board_start_y = dims['board_start_y']
    board_height = dims['board_height']
    
    # Panel width - adjust to fill available space
    panel_width = dims['lost_tracker_width'] if 'lost_tracker_width' in dims else 140
    panel_x = 10
    
    # Only show during play, not setup or menu
    if game_state != "play":
        return
    
    # Calculate panel height and position
    panel_height = board_height
    panel_y = board_start_y
    
    # Enhanced background with gradient effect
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    # Main background
    pygame.draw.rect(screen, (45, 45, 50), panel_rect)
    # Outer border with rounded corners effect
    draw_rounded_rect(screen, DARK_BLUE, panel_rect, 8, border=2)
    
    # Inner shadow for depth
    inner_shadow = pygame.Rect(panel_x + 2, panel_y + 2, panel_width - 4, 10)
    pygame.draw.rect(screen, (0, 0, 0, 60), inner_shadow)
    
    # Enhanced header with gradient
    header_height = get_font_size(28) + 12
    header_rect = pygame.Rect(panel_x, panel_y, panel_width, header_height)
    # Gradient background
    pygame.draw.rect(screen, ACCENT, header_rect)
    draw_rounded_rect(screen, ACCENT, header_rect, 8)
    # Header border
    pygame.draw.rect(screen, DARK_BLUE, header_rect, 2, border_radius=8)
    
    # Header text with shadow
    header_text = "Casualties"
    header_font_size = get_font_size(12)
    shadow_offset = 1
    draw_text(header_text, panel_x + panel_width // 2 + shadow_offset, panel_y + header_height // 2 + shadow_offset, 
              header_font_size, (0, 0, 0, 100), bold=True)
    draw_text(header_text, panel_x + panel_width // 2, panel_y + header_height // 2, 
              header_font_size, WHITE, bold=True)
    
    # Content area with padding - larger for better readability
    content_y = panel_y + header_height + 15
    content_height = panel_height - header_height - 30
    font_size = get_font_size(10)  # Increased from 12 to 14
    line_height = font_size + 8  # Increased from 6 to 8
    item_padding = 6  # Increased from 4 to 6
    
    # Player 1 section (human if vs_bot, or just player 1)
    player1_label = "You" if vs_bot and human_side == 1 else "Player 1"
    player1_color = RED
    player1_bg = (60, 20, 20)  # Darker red background for better contrast
    
    # Player 1 section background
    p1_section_height = content_height // 2 - 8
    p1_section = pygame.Rect(panel_x + 8, content_y, panel_width - 16, p1_section_height)
    pygame.draw.rect(screen, player1_bg, p1_section, border_radius=5)
    pygame.draw.rect(screen, player1_color, p1_section, 2, border_radius=5)
    
    y_pos = content_y + 10
    
    # Player 1 label with icon - better contrast
    label_bg = pygame.Rect(panel_x + 12, y_pos - 2, panel_width - 24, font_size + 6)
    pygame.draw.rect(screen, (80, 30, 30), label_bg, border_radius=3)
    draw_text(player1_label, panel_x + panel_width // 2, y_pos + 2, font_size + 2, WHITE, bold=True)  # White text instead of red
    y_pos += line_height + 6
    
    # Count lost pieces by rank for player 1
    if lost_pieces_player1:
        piece_counts = {}
        for piece in lost_pieces_player1:
            rank = piece.rank
            piece_counts[rank] = piece_counts.get(rank, 0) + 1
        
        # Display counts (sorted by rank, highest first)
        for rank in sorted(piece_counts.keys(), reverse=True):
            count = piece_counts[rank]
            piece_name = Piece(owner=1, rank=rank).short()
            
            # Item background - darker for better contrast
            item_rect = pygame.Rect(panel_x + 14, y_pos - 2, panel_width - 28, font_size + item_padding)
            pygame.draw.rect(screen, (50, 15, 15), item_rect, border_radius=3)
            
            # Piece display with icon-like indicator
            text_x = panel_x + 22
            # Draw small colored circle for piece indicator
            circle_y = y_pos + font_size // 2
            circle_radius = 5  # Slightly larger
            pygame.draw.circle(screen, player1_color, (text_x - 10, circle_y), circle_radius)
            pygame.draw.circle(screen, (255, 150, 150), (text_x - 10, circle_y), circle_radius - 1)
            
            text = f"{piece_name} ×{count}"
            draw_text(text, text_x, y_pos, font_size, WHITE, center=False)  # White text instead of LIGHT_GREY
            y_pos += line_height + 3
            if y_pos > panel_y + p1_section_height + content_y - 5:
                break
    else:
        # No losses indicator - better contrast
        no_loss_rect = pygame.Rect(panel_x + 14, y_pos - 2, panel_width - 28, font_size + item_padding)
        pygame.draw.rect(screen, (40, 20, 20), no_loss_rect, border_radius=3)
        draw_text("No losses", panel_x + panel_width // 2, y_pos + item_padding // 2, font_size, WHITE, center=True)
    
    # Enhanced divider
    divider_y = content_y + content_height // 2
    divider_line = pygame.Rect(panel_x + 10, divider_y - 1, panel_width - 20, 3)
    pygame.draw.rect(screen, DARK_BLUE, divider_line, border_radius=2)
    # Divider accent
    pygame.draw.line(screen, ACCENT, (panel_x + 12, divider_y), (panel_x + panel_width - 12, divider_y), 1)
    
    # Player 2 section (bot if vs_bot) - better contrast
    player2_label = "Bot" if vs_bot else "Player 2"
    player2_color = (200, 200, 200)  # Light grey instead of black
    player2_bg = (25, 25, 30)  # Dark background
    
    # Player 2 section background
    p2_section_y = divider_y + 6
    p2_section_height = content_height // 2 - 8
    p2_section = pygame.Rect(panel_x + 8, p2_section_y, panel_width - 16, p2_section_height)
    pygame.draw.rect(screen, player2_bg, p2_section, border_radius=5)
    pygame.draw.rect(screen, (100, 100, 100), p2_section, 2, border_radius=5)  # Light border instead of black
    
    y_pos = p2_section_y + 10
    
    # Player 2 label with icon - better contrast
    label_bg = pygame.Rect(panel_x + 12, y_pos - 2, panel_width - 24, font_size + 6)
    pygame.draw.rect(screen, (35, 35, 40), label_bg, border_radius=3)
    draw_text(player2_label, panel_x + panel_width // 2, y_pos + 2, font_size + 2, WHITE, bold=True)  # White text instead of black
    y_pos += line_height + 6
    
    # Count lost pieces by rank for player 2
    if lost_pieces_player2:
        piece_counts = {}
        for piece in lost_pieces_player2:
            rank = piece.rank
            piece_counts[rank] = piece_counts.get(rank, 0) + 1
        
        # Display counts (sorted by rank, highest first)
        for rank in sorted(piece_counts.keys(), reverse=True):
            count = piece_counts[rank]
            piece_name = Piece(owner=2, rank=rank).short()
            
            # Item background - darker for better contrast
            item_rect = pygame.Rect(panel_x + 14, y_pos - 2, panel_width - 28, font_size + item_padding)
            pygame.draw.rect(screen, (20, 20, 25), item_rect, border_radius=3)
            
            # Piece display with icon-like indicator
            text_x = panel_x + 22
            # Draw small colored circle for piece indicator
            circle_y = y_pos + font_size // 2
            circle_radius = 5  # Slightly larger
            pygame.draw.circle(screen, (150, 150, 150), (text_x - 10, circle_y), circle_radius)
            pygame.draw.circle(screen, (200, 200, 200), (text_x - 10, circle_y), circle_radius - 1)
            
            text = f"{piece_name} ×{count}"
            draw_text(text, text_x, y_pos, font_size, WHITE, center=False)  # White text instead of LIGHT_GREY
            y_pos += line_height + 3
            if y_pos > panel_y + p2_section_height + p2_section_y - 5:
                break
    else:
        # No losses indicator - better contrast
        no_loss_rect = pygame.Rect(panel_x + 14, y_pos - 2, panel_width - 28, font_size + item_padding)
        pygame.draw.rect(screen, (30, 30, 35), no_loss_rect, border_radius=3)
        draw_text("No losses", panel_x + panel_width // 2, y_pos + item_padding // 2, font_size, WHITE, center=True)

def draw_move_history():
    """Draw the move history panel with dynamic width based on available space"""
    current_width, current_height = screen.get_size()
    dims = get_board_dimensions()
    history_width = dims['history_width']
    
    # Don't draw if no space allocated for history
    if history_width <= 0:
        return
        
    history_x = current_width - history_width
    
    # Background with gradient effect and rounded corners
    history_area = pygame.Rect(history_x, 0, history_width, current_height)
    pygame.draw.rect(screen, (45, 45, 50), history_area)  # Slightly lighter background
    # Add rounded corners to the entire panel (top corners)
    pygame.draw.rect(screen, (45, 45, 50), pygame.Rect(history_x, 0, history_width, 15), border_radius=8)
    
    # Header with accent - rounded top corners
    header_height = get_font_size(30) + 15
    header_rect = pygame.Rect(history_x, 0, history_width, header_height)
    draw_rounded_rect(screen, ACCENT, header_rect, 8)  # Rounded top corners
    pygame.draw.rect(screen, DARK_BLUE, header_rect, 2, border_radius=8)
    
    draw_text("History", history_x + history_width // 2, header_height // 2, 
              get_font_size(16), WHITE, bold=True)
    
    # Move history list - compact and efficient
    y_offset = header_height + 8
    history_font_size = get_font_size(10)  # Smaller font
    padding = 6
    
    if not move_history:
        # Show placeholder when no moves yet
        draw_text("No moves yet", history_x + history_width // 2, current_height // 2, 
                  get_font_size(12), GREY)
        return
    
    for i, move_text in enumerate(reversed(move_history)):
        if y_offset > current_height - 20:  # Leave minimal space at bottom
            break
        
        # Alternate background for readability - more subtle
        if i % 2 == 0:
            bg_rect = pygame.Rect(history_x + padding, y_offset - 1, 
                                history_width - padding * 2, history_font_size + 2)
            pygame.draw.rect(screen, (60, 60, 60), bg_rect, border_radius=1)
        
        move_num = len(move_history) - i
        text = f"{move_num}. {move_text}"
        # Display full move text - no truncation
        # Text wrapping will happen automatically if needed
        draw_text(text, history_x + padding * 2, y_offset, history_font_size, WHITE, center=False)
        y_offset += history_font_size + 8  # Increased spacing to prevent overlap (was 2, now 8)

# --- Bot Logic ---
def choose_bot_move(board, owner):
    """Use enhanced bot logic to choose move"""
    if bot_logic:
        move = bot_logic.choose_move(board, owner)
        return move
    return None

# --- Bot Turn ---
def bot_turn():
    global message, current_player, game_state
    draw_move_history()
    time.sleep(0.5)

    move = choose_bot_move(board, current_player)
    if not move:
        message = f"{side_name(current_player)} (Bot) has no moves! {side_name(3 - current_player)} wins!"
        game_state = "menu"
        return

    src, dst = move
    
    # Track move for PBS (internal to bot)
    piece_before = board.get(dst)
    attacker_piece = board.get(src)
    
    msg, winner = board.move_and_resolve(src, dst, human_side if vs_bot else None)
    move_history.append(msg)
    message = msg

    # Track lost pieces - correctly identify which pieces were actually lost
    defender_copy = None
    attacker_copy = None
    
    # Check pieces after move resolution
    piece_after_dst = board.get(dst)
    piece_after_src = board.get(src)
    
    # Determine who won/lost based on final board state
    if piece_before:  # There was a defender (battle occurred)
        if piece_after_dst and piece_after_dst.owner == attacker_piece.owner:
            # Attacker won - defender was lost, attacker survived
            defender_copy = Piece(owner=piece_before.owner, rank=piece_before.rank, revealed=piece_before.revealed)
        elif piece_after_dst and piece_after_dst.owner == piece_before.owner:
            # Defender won - attacker was lost, defender survived
            attacker_copy = Piece(owner=attacker_piece.owner, rank=attacker_piece.rank, revealed=attacker_piece.revealed)
        elif piece_after_dst is None:
            # Both lost (equal trade or bomb scenario) - both pieces were lost
            attacker_copy = Piece(owner=attacker_piece.owner, rank=attacker_piece.rank, revealed=attacker_piece.revealed)
            defender_copy = Piece(owner=piece_before.owner, rank=piece_before.rank, revealed=piece_before.revealed)
    
    # Add lost pieces to tracking lists
    if defender_copy:
        if defender_copy.owner == 1:
            lost_pieces_player1.append(defender_copy)
        else:
            lost_pieces_player2.append(defender_copy)
    if attacker_copy:
        if attacker_copy.owner == 1:
            lost_pieces_player1.append(attacker_copy)
        else:
            lost_pieces_player2.append(attacker_copy)

    # Update bot's PBS with opponent's revealed pieces (internal, not displayed)
    if piece_before and piece_before.owner == human_side and bot_logic:
        if piece_before.revealed:
            bot_logic.pbs.update_from_reveal(dst, piece_before.rank)

    time.sleep(0.3)

    if winner:
        winner_name = side_name(winner)
        if vs_bot:
            if winner == human_side:
                message = "You win! Congratulations!"
            else:
                message = "You lost! Better luck next time!"
        else:
            message = f"{winner_name} wins!"
        game_state = "game_over"
    else:
        current_player = 3 - current_player

def human_move(src, dst):
    """Process human move and update bot's PBS (internal)"""
    global message, current_player, game_state
    
    piece_at_src = board.get(src)
    piece_at_dst = board.get(dst)
    
    msg, winner = board.move_and_resolve(src, dst, human_side if vs_bot else None)
    move_history.append(msg)
    
    # Track lost pieces - correctly identify which pieces were actually lost
    defender_copy = None
    attacker_copy = None
    
    # Check pieces after move resolution
    piece_after_dst = board.get(dst)
    piece_after_src = board.get(src)
    
    # Determine who won/lost based on final board state
    if piece_at_dst:  # There was a defender (battle occurred)
        if piece_after_dst and piece_after_dst.owner == piece_at_src.owner:
            # Attacker won - defender was lost, attacker survived
            defender_copy = Piece(owner=piece_at_dst.owner, rank=piece_at_dst.rank, revealed=piece_at_dst.revealed)
        elif piece_after_dst and piece_after_dst.owner == piece_at_dst.owner:
            # Defender won - attacker was lost, defender survived
            attacker_copy = Piece(owner=piece_at_src.owner, rank=piece_at_src.rank, revealed=piece_at_src.revealed)
        elif piece_after_dst is None:
            # Both lost (equal trade or bomb scenario) - both pieces were lost
            attacker_copy = Piece(owner=piece_at_src.owner, rank=piece_at_src.rank, revealed=piece_at_src.revealed)
            defender_copy = Piece(owner=piece_at_dst.owner, rank=piece_at_dst.rank, revealed=piece_at_dst.revealed)
    
    # Add lost pieces to tracking lists
    if defender_copy:
        if defender_copy.owner == 1:
            lost_pieces_player1.append(defender_copy)
        else:
            lost_pieces_player2.append(defender_copy)
    if attacker_copy:
        if attacker_copy.owner == 1:
            lost_pieces_player1.append(attacker_copy)
        else:
            lost_pieces_player2.append(attacker_copy)
    
    # Update bot's PBS about human's move (internal, not displayed)
    if vs_bot and bot_logic and piece_at_src and piece_at_src.owner == human_side:
        if piece_at_src.revealed and current_player == human_side:
            bot_logic.pbs.update_from_reveal(src, piece_at_src.rank)
    
    if winner:
        if vs_bot:
            if winner == human_side:
                message = "You win! Congratulations!"
            else:
                message = "You lost! Better luck next time!"
        else:
            message = f"{side_name(winner)} wins!"
        game_state = "game_over"
    else:
        message = msg
        current_player = 3 - current_player

# --- Main loop ---
clock = pygame.time.Clock()

while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds for smooth animations
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle window resize
        if event.type == pygame.VIDEORESIZE and not fullscreen:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        # Fullscreen toggle (F11) and history panel toggle (H)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                toggle_fullscreen()
            elif game_state == "play" or game_state == "setup":
                if event.key == pygame.K_h:
                    show_history_panel = not show_history_panel
                    # Force screen refresh to apply new layout immediately
                    if game_state != "setup":  # Don't trigger during setup
                        pygame.event.clear()  # Clear pending events to prevent issues
                elif event.key == pygame.K_ESCAPE:
                    game_state = "pause"

        if game_state == "setup" and event.type == pygame.MOUSEBUTTONDOWN:
            sq = get_square_from_mouse(event.pos)
            if not sq:
                continue

            if selected is None:
                piece = board.get(sq)
                if piece and piece.owner == human_side:
                    selected = sq
            else:
                if (human_side == 1 and sq[0] >= 6) or (human_side == 2 and sq[0] <= 3):
                    p1 = board.get(selected)
                    p2 = board.get(sq)
                    board.set(selected, p2)
                    board.set(sq, p1)
                    selected = None
                else:
                    message = "Invalid move. Place on your side."
                    selected = None

        elif game_state == "play" and (not vs_bot or current_player == human_side):
            if event.type == pygame.MOUSEBUTTONDOWN:
                sq = get_square_from_mouse(event.pos)
                if not sq:
                    continue
                if selected is None:
                    piece = board.get(sq)
                    if piece and piece.owner == current_player and piece.is_movable():
                        selected = sq
                else:
                    src, dst = selected, sq
                    if dst in board.legal_moves_from(src):
                        human_move(src, dst)
                        selected = None
                    else:
                        selected = None

    # --- Screen Rendering ---
    screen.fill(BACKGROUND)

    current_width, current_height = screen.get_size()
    dims = get_board_dimensions()
    tile_size = dims['tile_size']
    border_size = dims['border_size']
    board_width = dims['board_width']
    board_width_with_borders = dims['board_width_with_borders']
    history_width = dims['history_width']

    if game_state == "menu":
        # Center menu content in the board area (not including history panel)
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2
        
        # Title with shadow effect - fixed spacing
        title_y = get_font_size(60) + 20
        title_font_size = get_font_size(64)
        draw_text("Stratego with MARQ", center_x, title_y, title_font_size, DARK_BLUE, bold=True)
        draw_text("Stratego with MARQ", center_x + 3, title_y + 3, title_font_size, BLUE, bold=True)
        
        # Subtitle with proper spacing below title
        subtitle_y = title_y + title_font_size + 20  # Increased spacing from 10 to 20, and account for full title height
        #draw_text("Intelligent Strategic Gameplay", center_x, subtitle_y, get_font_size(32), DARK_GREY)
        
        # Menu buttons with spacing - ensure adequate separation
        button_width = int(menu_area_width * 0.35)
        button_width = max(200, min(button_width, 250))
        button_height = get_font_size(35) + 20
        # Use larger spacing to prevent button overlap, especially in fullscreen
        button_spacing = max(button_height + 20, get_font_size(60))
        start_y = subtitle_y + get_font_size(40) + 30
        
        if draw_button("Play vs Bot", center_x - button_width//2, start_y, 
                      button_width, button_height, action="bot", primary=True) == "bot":
            board = Board()
            vs_bot = True
            human_side = 1
            current_player = 1
            
            try:
                setup_agent = StrategicSetupAgent(player_id=2, model_path=setup_model_path)
                setup_agent.apply_setup_to_board(board, Piece)
                print("✓ Bot setup completed using Setup Agent")
            except Exception as e:
                print(f"Setup agent failed: {e}, using auto_setup")
                auto_setup(board, 2)
            
            auto_setup(board, human_side)
            
            try:
                bot_logic = DQNBotLogic(agent_model_path, player_id=2)
                bot_logic.reset()
                print("✓ DQN bot initialized with Rainbow DQN + AAREN")
            except Exception as e:
                print(f"DQN bot initialization failed: {e}")
                print("Falling back to EnhancedBotLogic...")
                try:
                    bot_logic = EnhancedBotLogic(agent_model_path, player_id=2)
                    bot_logic.reset()
                    print("✓ Fallback: Enhanced bot initialized with PBS")
                except Exception as e2:
                    print(f"Enhanced bot also failed: {e2}")
                    if not os.path.exists(agent_model_path):
                        print(f"❌ Error: Model file not found: {agent_model_path}")
                        raise FileNotFoundError(f"Model file not found: {agent_model_path}")
                    from bot_logic import BotLogic
                    bot_logic = BotLogic(agent_model_path)
            
            game_state = "setup"
            message = "Arrange your pieces. Click to select and move."
            move_history.clear()
            lost_pieces_player1.clear()
            lost_pieces_player2.clear()
            
        if draw_button("2-Player Mode", center_x - button_width//2, start_y + button_spacing, 
                      button_width, button_height, action="2p", primary=True) == "2p":
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
                      button_width, button_height, action="quit") == "quit":
            running = False
        
        # Fullscreen hint
        hint_y = current_height - get_font_size(16)
        draw_text("Press F11 for fullscreen mode", center_x, hint_y, get_font_size(14), BLACK)

    elif game_state == "setup":
        draw_board()
        # Don't draw history during setup phase to maximize board space
        
        # Get dimensions after ensuring history is hidden
        dims = get_board_dimensions()
        board_width_with_borders = dims['board_width_with_borders']
        board_start_x = dims['board_start_x']
        board_center_x = board_start_x + board_width_with_borders // 2
        
        # Position Start Game button centered below the board
        button_width = 180
        button_height = get_font_size(30) + 16
        
        # Center button horizontally with board
        button_x = board_center_x - button_width // 2
        button_y = dims['board_start_y'] + dims['board_height'] + 35
        
        # Draw the Start button with rounded corners (already has rounded corners in draw_button)
        action = draw_button("START GAME", button_x, button_y, 
                            button_width, button_height, action="start", primary=True)
        
        # Add instruction text centered above button
        instruction_y = button_y - get_font_size(16) + 5
        draw_text("Rearrange pieces, then click START!", board_center_x, instruction_y, 
                  get_font_size(15), DARK_GREY)
        
        # No backup button needed - single centered button
        final_action = action
        
        # Add a prominent setup instruction banner with rounded corners
        setup_info_y = dims['board_start_y'] - 50
        setup_banner_height = 40
        setup_banner_width = board_width_with_borders
        setup_banner_x = dims['board_start_x']
        setup_banner_center_x = setup_banner_x + setup_banner_width // 2
        setup_banner = pygame.Rect(setup_banner_x, setup_info_y - setup_banner_height//2, 
                                   setup_banner_width, setup_banner_height)
        draw_rounded_rect(screen, ACCENT, setup_banner, 8)
        pygame.draw.rect(screen, DARK_BLUE, setup_banner, 2, border_radius=8)
        
        # Setup instruction text
        if vs_bot:
            setup_text = f"Setup Phase - Arrange your pieces | Bot has pre-placed"
        else:
            setup_text = f"Setup Phase - Arrange Player {human_side}'s pieces"
        
        draw_text(setup_text, setup_banner_center_x, setup_info_y, 
                  get_font_size(12), WHITE, bold=True)
        
        if final_action == "start":
            print(f"Starting game from setup - Player: {human_side}, Bot: {vs_bot}")
            game_state = "play"
            if vs_bot:
                message = "Your turn! Click on your piece, then click where to move."
            else:
                message = f"Player {current_player}'s turn"
            print(f"Game state changed to: {game_state}, message: {message}")

    elif game_state == "play":
        draw_board()
        draw_lost_pieces_tracker()
        draw_move_history()
        
        if vs_bot and current_player != human_side:
            bot_turn()
        
        # Enhanced message bar - centered with rounded corners
        msg_bar_y = dims['board_start_y'] + dims['board_height'] + border_size + 10
        msg_bar_height = get_font_size(25) + 15
        msg_bar_width = board_width_with_borders
        msg_bar_x = dims['board_start_x']
        msg_bar_center_x = msg_bar_x + msg_bar_width // 2
        
        # Message bar with rounded corners
        msg_bar = pygame.Rect(msg_bar_x, msg_bar_y, msg_bar_width, msg_bar_height)
        draw_rounded_rect(screen, DARK_GREY, msg_bar, 8)
        # Highlight at top
        highlight_rect = pygame.Rect(msg_bar_x, msg_bar_y, msg_bar_width, 3)
        pygame.draw.rect(screen, ACCENT, highlight_rect, border_radius=8)
        # Border
        pygame.draw.rect(screen, DARK_BLUE, msg_bar, 2, border_radius=8)
        
        draw_text(message, msg_bar_center_x, msg_bar_y + msg_bar_height // 2, 
                  get_font_size(12), WHITE, bold=True)
        
        # Show history toggle hint - more compact
        hint_text = f"Press H to {'show' if not show_history_panel else 'hide'} history" if dims['history_width'] > 0 else ""
        if hint_text:
            draw_text(hint_text, msg_bar_center_x, msg_bar_y + msg_bar_height + 20, 
                      get_font_size(12), BLACK)

    elif game_state == "pause":
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2
        
        draw_text("Paused", center_x, get_font_size(80), get_font_size(72), BLUE, bold=True)
        
        button_width = int(menu_area_width * 0.35)
        button_width = max(200, min(button_width, 250))
        button_height = get_font_size(35) + 20
        # Use larger spacing to prevent button overlap, especially in fullscreen
        button_spacing = max(button_height + 20, get_font_size(60))
        button_y = get_font_size(180)
        
        if draw_button("Resume", center_x - button_width//2, button_y, 
                      button_width, button_height, action="resume", primary=True) == "resume":
            game_state = "play"
        if draw_button("Quit to Menu", center_x - button_width//2, button_y + button_spacing, 
                      button_width, button_height, action="menu") == "menu":
            game_state = "menu"
        if draw_button("Exit", center_x - button_width//2, button_y + button_spacing * 2, 
                      button_width, button_height, action="exit") == "exit":
            running = False
    
    elif game_state == "game_over":
        menu_area_width = current_width - history_width if history_width > 0 else current_width
        center_x = menu_area_width // 2
        
        draw_text("Game Over", center_x, get_font_size(80), get_font_size(72), BLUE, bold=True)
        draw_text(message, center_x, get_font_size(160), get_font_size(32), DARK_BLUE, bold=True)
        
        button_width = int(menu_area_width * 0.35)
        button_width = max(200, min(button_width, 250))
        button_height = get_font_size(35) + 20
        button_y = get_font_size(230)
        
        if draw_button("Back to Menu", center_x - button_width//2, button_y, 
                      button_width, button_height, action="menu", primary=True) == "menu":
            # Reset ALL game state variables to clean slate
            board = Board()
            auto_setup(board, 1)
            auto_setup(board, 2)
            current_player = 1
            selected = None
            message = "Welcome to Stratego with MARQ!"
            move_history.clear()
            game_state = "menu"
            
            # Properly reset bot-related variables
            vs_bot = False
            human_side = 1
            bot_logic = None
            setup_agent = None
            
            # Reset other state
            button_states.clear()
            show_history_panel = True
            lost_pieces_player1.clear()
            lost_pieces_player2.clear()

    pygame.display.flip()

pygame.quit()
