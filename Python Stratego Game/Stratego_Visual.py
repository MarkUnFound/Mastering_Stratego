import pygame
import time
import random
from stratego import Board, auto_setup, side_name
from bot_logic import BotLogic

pygame.init()
TILE_SIZE = 60
BOARD_SIZE = 10
WIDTH, HEIGHT = TILE_SIZE * BOARD_SIZE, TILE_SIZE * BOARD_SIZE + 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stratego Visual")

# Colors
WHITE = (240, 240, 240)
BLACK = (30, 30, 30)
BLUE = (50, 150, 255)
RED = (220, 50, 50)
GREY = (180, 180, 180)
DARK_GREY = (100, 100, 100)
GREEN = (60, 180, 60)

font = pygame.font.SysFont(None, 28)

# Game state
current_player = 1
selected = None
running = True
game_state = "menu"
message = "Welcome to Stratego!"
human_side = 1
vs_bot = False
bot_logic = None

# Track revealed pieces (that have moved or fought)
revealed_positions = set()

# Setup board
board = Board()
auto_setup(board, 1)
auto_setup(board, 2)

# --- Helper functions ---
def draw_text(text, x, y, size=28, color=BLACK, center=True):
    f = pygame.font.SysFont(None, size)
    surf = f.render(text, True, color)
    rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
    screen.blit(surf, rect)
    return rect

def draw_button(text, x, y, w, h, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    rect = pygame.Rect(x, y, w, h)
    color = DARK_GREY
    if rect.collidepoint(mouse):
        color = GREY
        if click[0] and action:
            return action
    pygame.draw.rect(screen, color, rect)
    draw_text(text, x + w // 2, y + h // 2, 32, WHITE)
    return None

def draw_board(bot_move=None):
    """Draw board with fog of war — enemy pieces hidden unless revealed or moved."""
    screen.fill(WHITE)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if board.is_lake(r, c):
                pygame.draw.rect(screen, BLUE, rect)
                continue

            pygame.draw.rect(screen, GREY, rect, 1)
            piece = board.get((r, c))
            if not piece:
                continue

            color = RED if piece.owner == 1 else BLACK

            # Determine if the piece should be visible
            visible = False
            if piece.owner == human_side:
                visible = True
            elif (r, c) in revealed_positions:
                visible = True
            elif bot_move and (r, c) in bot_move:
                visible = True
            elif piece.revealed:
                visible = True

            # Draw selection highlight
            if selected == (r, c):
                pygame.draw.rect(screen, GREEN, rect, 3)

            pygame.draw.circle(screen, color, rect.center, TILE_SIZE // 3)
            text = font.render(str(piece.short()) if visible else "??", True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

    pygame.draw.rect(screen, DARK_GREY, (0, HEIGHT-100, WIDTH, 100))
    draw_text(message, WIDTH//2, HEIGHT-50, 28, WHITE)
    pygame.display.flip()

def get_square_from_mouse(pos):
    x, y = pos
    if y > TILE_SIZE * BOARD_SIZE:
        return None
    c, r = x // TILE_SIZE, y // TILE_SIZE
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        return (r, c)
    return None

# --- Bot Logic ---
def choose_bot_move(board, owner):
    if bot_logic:
        return bot_logic.choose_move(board, owner)
    return None  # Should not be reached if bot_logic is initialized

# --- Bot Turn ---
def bot_turn():
    global message, current_player, game_state
    draw_board()
    #draw_text("Bot is thinking...", WIDTH//2, HEIGHT-50, 28, WHITE)
    pygame.display.flip()
    time.sleep(0.8)

    move = choose_bot_move(board, current_player)
    if not move:
        message = f"{side_name(current_player)} (Bot) has no moves! {side_name(3 - current_player)} wins!"
        game_state = "menu"
        return

    src, dst = move
    msg, winner = board.move_and_resolve(src, dst)
    message = msg

    # Mark bot's moved piece as revealed
    revealed_positions.add(dst)

    draw_board(bot_move=[src, dst])
    pygame.display.flip()
    time.sleep(0.5)

    if winner:
        message = f"{side_name(winner)} wins!"
        game_state = "menu"
    else:
        current_player = 3 - current_player

# --- Main loop ---
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if game_state == "play" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_state = "pause"

        if game_state == "play" and (not vs_bot or current_player == human_side):
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
                        msg, winner = board.move_and_resolve(src, dst)
                        message = msg
                        revealed_positions.add(dst)  # reveal moved piece
                        if winner:
                            message = f"{side_name(winner)} wins!"
                            game_state = "menu"
                        current_player = 3 - current_player
                        selected = None
                    else:
                        selected = None

    # --- Screen Rendering ---
    if game_state == "menu":
        screen.fill(WHITE)
        draw_text("STRATEGO", WIDTH//2, 100, 72, BLUE)
        if draw_button("Play vs Bot", WIDTH//2-100, 200, 200, 60, "bot") == "bot":
            board = Board()
            auto_setup(board, 1)
            auto_setup(board, 2)
            revealed_positions = set()
            vs_bot = True
            human_side = 1
            current_player = 1
            message = "Player 1's turn"
            bot_logic = BotLogic('d:\\Research\\Python Stratego Game\\agent2_final.pth')
            game_state = "play"
        if draw_button("2-Player Mode", WIDTH//2-100, 280, 200, 60, "2p") == "2p":
            board = Board()
            auto_setup(board, 1)
            auto_setup(board, 2)
            revealed_positions = set()
            vs_bot = False
            human_side = None
            current_player = 1
            message = "Player 1's turn"
            game_state = "play"
        if draw_button("Quit", WIDTH//2-100, 380, 200, 60, "quit") == "quit":
            running = False
        pygame.display.flip()

    elif game_state == "play":
        if vs_bot and current_player != human_side and game_state == "play":
            bot_turn()
        else:
            draw_board()

    elif game_state == "pause":
        screen.fill(WHITE)
        draw_text("Paused", WIDTH//2, 120, 72, BLUE)
        if draw_button("Resume", WIDTH//2-100, 200, 200, 60, "resume") == "resume":
            game_state = "play"
        if draw_button("Quit to Menu", WIDTH//2-100, 300, 200, 60, "menu") == "menu":
            game_state = "menu"
        if draw_button("Exit", WIDTH//2-100, 400, 200, 60, "exit") == "exit":
            running = False
        pygame.display.flip()

pygame.quit()
