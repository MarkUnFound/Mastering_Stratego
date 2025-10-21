#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal Stratego (supports 1-player vs bot and 2-player hot-seat)
Based on the user's original strategoo.py with a simple bot added.
Author: ChatGPT (modified)
License: MIT
Python 3.8+
"""
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Board constants ---
BOARD_SIZE = 10

LAKES = {
    (4, 2), (4, 3), (5, 2), (5, 3),
    (4, 6), (4, 7), (5, 6), (5, 7)
}

RANK_NAMES = {
    10: "Marshal",
    9: "General",
    8: "Colonel",
    7: "Major",
    6: "Captain",
    5: "Lieutenant",
    4: "Sergeant",
    3: "Miner",
    2: "Scout",
    1: "Spy",
    0: "Bomb",
    -1: "Flag",
}

PIECE_COUNTS = {
    10: 1,  # Marshal
    9: 1,   # General
    8: 2,   # Colonel
    7: 3,   # Major
    6: 4,   # Captain
    5: 4,   # Lieutenant
    4: 4,   # Sergeant
    3: 5,   # Miner
    2: 8,   # Scout
    1: 1,   # Spy
    0: 6,   # Bomb
    -1: 1,  # Flag
}

START_ROWS = {
    1: [6, 7, 8, 9],
    2: [0, 1, 2, 3],
}

FILES = "ABCDEFGHIJ"
RANKS = "12345678910"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause(msg: str = "Press Enter to continue..."):
    try:
        input(msg)
    except EOFError:
        pass

def coord_to_str(pos: Tuple[int, int]) -> str:
    r, c = pos
    return f"{FILES[c]}{r+1}"

def parse_square(s: str) -> Optional[Tuple[int, int]]:
    s = s.strip().upper()
    if len(s) < 2:
        return None
    file_char = s[0]
    if file_char not in FILES:
        return None
    try:
        rank_num = int(s[1:])
    except ValueError:
        return None
    if not (1 <= rank_num <= BOARD_SIZE):
        return None
    r = rank_num - 1
    c = FILES.index(file_char)
    return (r, c)

@dataclass
class Piece:
    owner: int
    rank: int
    revealed: bool = False

    def display(self, viewer: int) -> str:
        if self.owner == viewer or self.revealed:
            if self.rank >= 1:
                return f"{self.rank:>2}"
            elif self.rank == 0:
                return " B"
            else:
                return " F"
        else:
            return "??"

    def short(self) -> str:
        if self.rank >= 1:
            return str(self.rank)
        elif self.rank == 0:
            return "B"
        else:
            return "F"

    def name(self) -> str:
        return RANK_NAMES[self.rank]

    def is_movable(self) -> bool:
        return self.rank not in (0, -1)

@dataclass
class Board:
    grid: List[List[Optional[Piece]]] = field(default_factory=lambda: [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def is_lake(self, r: int, c: int) -> bool:
        return (r, c) in LAKES

    def get(self, pos: Tuple[int, int]) -> Optional[Piece]:
        r, c = pos
        return self.grid[r][c]

    def set(self, pos: Tuple[int, int], piece: Optional[Piece]):
        r, c = pos
        self.grid[r][c] = piece

    def empty(self, pos: Tuple[int, int]) -> bool:
        return self.get(pos) is None

    def all_positions(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]

    def owner_positions(self, owner: int) -> List[Tuple[int, int]]:
        out = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.grid[r][c]
                if p and p.owner == owner:
                    out.append((r, c))
        return out

    def has_movable(self, owner: int) -> bool:
        for r, c in self.owner_positions(owner):
            p = self.grid[r][c]
            if p and p.is_movable():
                if any(self.legal_moves_from((r, c))):
                    return True
        return False

    def path_clear(self, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        sr, sc = src
        dr, dc = dst
        if sr == dr:
            step = 1 if dc > sc else -1
            for c in range(sc + step, dc, step):
                if self.grid[sr][c] is not None:
                    return False
            return True
        elif sc == dc:
            step = 1 if dr > sr else -1
            for r in range(sr + step, dr, step):
                if self.grid[r][sc] is not None:
                    return False
            return True
        return False

    def legal_moves_from(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        sr, sc = pos
        p = self.get(pos)
        if not p or not p.is_movable():
            return []

        moves = []
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]

        def can_land(r, c):
            if not self.in_bounds(r, c): return False
            if self.is_lake(r, c): return False
            dest = self.grid[r][c]
            if dest and dest.owner == p.owner: return False
            return True

        if p.rank == 2:  # Scout
            for dr, dc in deltas:
                r, c = sr + dr, sc + dc
                while self.in_bounds(r, c) and not self.is_lake(r, c):
                    if self.grid[r][c] is None:
                        moves.append((r, c))
                    else:
                        if self.grid[r][c].owner != p.owner:
                            moves.append((r, c))
                        break
                    r += dr
                    c += dc
            return moves

        for dr, dc in deltas:
            r, c = sr + dr, sc + dc
            if can_land(r, c):
                moves.append((r, c))
        return moves

    def move_and_resolve(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[str, Optional[int]]:
        attacker = self.get(src)
        defender = self.get(dst)
        assert attacker is not None
        if defender is None:
            self.set(dst, attacker)
            self.set(src, None)
            return (f"{side_name(attacker.owner)} moved {attacker.name()} from {coord_to_str(src)} to {coord_to_str(dst)}.", None)

        a, d = attacker, defender
        if d.rank == -1:
            self.set(dst, attacker)
            self.set(src, None)
            return (f"{side_name(a.owner)} captured the enemy Flag at {coord_to_str(dst)} with {a.name()}!", a.owner)

        outcome = resolve_combat(a, d)
        a.revealed = True
        d.revealed = True

        if outcome == "attacker":
            self.set(dst, attacker)
            self.set(src, None)
            return (f"Combat at {coord_to_str(dst)}: {a.name()} (attacker) defeated {d.name()} (defender).", None)
        elif outcome == "defender":
            self.set(src, None)
            return (f"Combat at {coord_to_str(dst)}: {d.name()} (defender) defeated {a.name()} (attacker).", None)
        else:
            self.set(src, None)
            self.set(dst, None)
            return (f"Combat at {coord_to_str(dst)}: {a.name()} and {d.name()} eliminated each other.", None)

def resolve_combat(attacker: Piece, defender: Piece) -> str:
    if defender.rank == 0:
        if attacker.rank == 3:
            return "attacker"
        else:
            return "defender"
    if attacker.rank == 0:
        return "defender"
    if attacker.rank == 1 and defender.rank == 10:
        return "attacker"
    if attacker.rank > defender.rank:
        return "attacker"
    elif attacker.rank < defender.rank:
        return "defender"
    else:
        return "both"

def side_name(owner: int) -> str:
    return "Player 1" if owner == 1 else "Player 2"

def print_board(board: Board, viewer: int):
    print("   " + " ".join([f"{f:>2}" for f in FILES]))
    for r in range(BOARD_SIZE-1, -1, -1):
        row_label = f"{r+1:>2} "
        line = []
        for c in range(BOARD_SIZE):
            if (r, c) in LAKES:
                cell = "~~"
            else:
                p = board.grid[r][c]
                if p is None:
                    cell = " . "
                else:
                    token = p.display(viewer)
                    if p.owner == viewer:
                        cell = f"[{token.strip():>2}]"
                    else:
                        cell = f" {token} "
            line.append(cell)
        print(row_label + "".join(line) + f" {r+1:>2}")
    print("   " + " ".join([f"{f:>2}" for f in FILES]))

def help_text():
    return """\
Commands
- Move: enter FROM TO (e.g., "E3 E4" or "b7 b5").
- help: show this help.
- reveal: show full board (debug/learning aid).
- resign: resign the game.
- quit: quit the program.

Rules (highlights)
- Bombs (B) and Flag (F) cannot move.
- Scouts (2) move any number of squares in a straight line (no jumping).
- Miner (3) defuses Bombs.
- Spy (1) defeats Marshal (10) only when the Spy attacks.
- Lakes ~~ are impassable.
- Win by capturing the enemy Flag or when your opponent has no legal moves.
"""

def auto_setup(board: Board, owner: int):
    start_cells = [(r, c) for r in START_ROWS[owner] for c in range(BOARD_SIZE) if not board.is_lake(r, c)]
    random.shuffle(start_cells)
    pieces = []
    for rank, count in PIECE_COUNTS.items():
        for _ in range(count):
            pieces.append(Piece(owner=owner, rank=rank))
    for pos, piece in zip(start_cells, pieces):
        board.set(pos, piece)

def any_moves_for_owner(board: Board, owner: int) -> bool:
    for pos in board.owner_positions(owner):
        if any(board.legal_moves_from(pos)):
            return True
    return False

def prompt_move(board: Board, current: int) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return None

        if raw.lower() in ("help", "?"):
            print(help_text()); continue
        if raw.lower() in ("reveal", "debug"):
            print_full_board(board); continue
        if raw.lower() in ("resign", "gg"):
            return ("RESIGN", None)
        if raw.lower() in ("quit", "exit"):
            confirm = input("Quit the program? (y/n): ").strip().lower()
            if confirm == "y":
                sys.exit(0)
            else:
                continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Enter a move like 'E3 E4' (FROM TO). Type 'help' for options.")
            continue
        src = parse_square(parts[0])
        dst = parse_square(parts[1])
        if src is None or dst is None:
            print("Invalid coordinates. Use file+rank like A1..J10."); continue

        sr, sc = src
        dr, dc = dst
        if (sr, sc) == (dr, dc):
            print("Source and destination are the same."); continue

        piece = board.get(src)
        if piece is None:
            print("No piece at the source square."); continue
        if piece.owner != current:
            print("That's not your piece."); continue
        if not piece.is_movable():
            print(f"{piece.name()} cannot move."); continue
        if board.is_lake(dr, dc):
            print("Cannot move into a lake."); continue

        legal = board.legal_moves_from(src)
        if dst not in legal:
            print("That move is not legal for this piece."); continue

        return (src, dst)

def print_full_board(board: Board):
    print("   " + " ".join([f"{f:>2}" for f in FILES]))
    for r in range(BOARD_SIZE-1, -1, -1):
        row_label = f"{r+1:>2} "
        line = []
        for c in range(BOARD_SIZE):
            if (r, c) in LAKES:
                cell = "~~"
            else:
                p = board.grid[r][c]
                if p is None:
                    cell = " . "
                else:
                    token = p.display(p.owner)
                    token = f"{token}{'*' if p.owner == 2 else ' '}".rstrip()
                    cell = f"{token:>3}"
            line.append(cell)
        print(row_label + "".join(line) + f" {r+1:>2}")
    print("   " + " ".join([f"{f:>2}" for f in FILES]))

def banner():
    print("="*64)
    print("                STRATEGO — Terminal Edition")
    print("="*64)
    print("Play 1-player (vs bot) or 2-player hot-seat. Random legal setup for both sides.")
    print("On your turn you'll see your pieces; opponent's are '??' unless revealed.")
    print("Win by capturing the Flag or when the opponent has no legal moves.")
    print("Type 'help' any time for commands.\n")

def turn_header(current: int, board: Board, human_side: Optional[int]):
    print("-"*64)
    actor = side_name(current)
    if human_side is not None and current != human_side:
        actor = f"{actor} (BOT)"
    print(f"{actor} to move.")
    print("-"*64)
    viewer = current if (human_side is None or current == human_side) else human_side
    # If viewer is None (two-player and both human), we still show current view
    print_board(board, viewer=current if human_side is None else viewer)

def pass_device_to(player: int):
    print(f"\nPass the device to {side_name(player)}.")
    pause("Press Enter when ready (screen will clear)...")
    clear_screen()

# ---- Simple Bot AI ----
def choose_bot_move(board: Board, owner: int) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Simple bot:
    - Gather all legal moves for bot.
    - Prefer capturing moves (dest occupied by enemy).
    - Otherwise pick a random legal move.
    """
    moves = []
    capture_moves = []
    for src in board.owner_positions(owner):
        piece = board.get(src)
        if piece is None or not piece.is_movable():
            continue
        legal = board.legal_moves_from(src)
        for dst in legal:
            dest_piece = board.get(dst)
            if dest_piece is not None and dest_piece.owner != owner:
                capture_moves.append((src, dst))
            else:
                moves.append((src, dst))
    if capture_moves:
        return random.choice(capture_moves)
    if moves:
        return random.choice(moves)
    return None

# ---- Game loop ----
def play_game():
    clear_screen()
    banner()
    # Mode selection
    mode = None
    while mode not in ("1", "2"):
        mode = input("Choose mode: [1] One-player vs Bot, [2] Two-player hot-seat: ").strip()
    human_side = None
    if mode == "1":
        # pick which side human plays
        hs = None
        while hs not in ("1", "2"):
            hs = input("Play as Player 1 (bottom) or Player 2 (top)? Enter 1 or 2: ").strip()
        human_side = int(hs)
        print(f"You will play as {side_name(human_side)}.")
    else:
        print("Two-player hot-seat selected.")

    board = Board()
    auto_setup(board, owner=1)
    auto_setup(board, owner=2)

    current = 1
    other = 2

    # If two-player, do initial pass; if one-player and human is player 1, show pass; else no pass needed.
    if human_side is None or current == human_side:
        pass_device_to(current)
    else:
        clear_screen()

    while True:
        turn_header(current, board, human_side)

        if not any_moves_for_owner(board, current):
            print(f"{side_name(current)} has no legal moves remaining! {side_name(other)} wins!")
            break

        # Bot move if single-player and current != human_side
        if human_side is not None and current != human_side:
            print("Bot is thinking...")
            time.sleep(0.5)
            mv = choose_bot_move(board, current)
            if mv is None:
                print(f"{side_name(current)} (BOT) has no moves! {side_name(other)} wins!")
                break
            src, dst = mv
            msg, winner = board.move_and_resolve(src, dst)
            print(msg)
        else:
            mv = prompt_move(board, current)
            if mv is None:
                print("Input stream ended. Exiting.")
                break
            if mv == ("RESIGN", None):
                print(f"{side_name(current)} resigns. {side_name(other)} wins!")
                break
            src, dst = mv
            msg, winner = board.move_and_resolve(src, dst)
            print(msg)

        if winner is not None:
            print(f"{side_name(winner)} wins by capturing the Flag!")
            break

        if not any_moves_for_owner(board, other):
            print(f"{side_name(other)} has no legal moves remaining! {side_name(current)} wins!")
            break

        # swap turns
        pause()
        clear_screen()
        current, other = other, current

        # If human needs to receive the device for their turn (in one-player, we don't pass for bot)
        if human_side is None:
            pass_device_to(current)
        else:
            # if the next player is human, give prompt to continue; otherwise don't require passing
            if current == human_side:
                pause("Your turn. Press Enter to view the board...")
                clear_screen()

if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\nGame interrupted. Bye!")
