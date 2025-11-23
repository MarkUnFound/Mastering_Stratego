# stratego_modular/game_constants.py

"""
Centralized constants for the Stratego game.
Defines the board representation values to avoid negative value confusion.
"""

BOARD_SIZE = 10

# Board values
EMPTY_SQUARE = 0
LAKE_SQUARE = -13  # Matches board.py, distinct from all piece values
HIDDEN_PIECE = -20  # Changed from -9 to -20 to avoid ambiguity with Scout (3/-3)

# Agent 2 piece value offset
# Agent 1 pieces: 1 to 12 (PieceType.value)
# Agent 2 pieces: 101 to 112 (PieceType.value + AGENT2_OFFSET)
#
# NOTE: We use "Agent 2" terminology in constants/functions, but player_id values are 1 and 2
# - Player 1 (player_id = 1): Also called "Agent 1", uses piece values 1-12
# - Player 2 (player_id = 2): Also called "Agent 2", uses piece values 101-112
AGENT2_OFFSET = 100

def get_piece_owner(piece_value: int) -> int:
    """
    Return the owner of the piece based on its value.
    Returns:
        1: Player 1
        2: Player 2
        0: Empty, Lake, or Hidden
    """
    if piece_value == EMPTY_SQUARE or piece_value == LAKE_SQUARE or piece_value == HIDDEN_PIECE:
        return 0
    if piece_value > AGENT2_OFFSET:
        return 2  # Changed from -1 to 2
    if piece_value > 0:
        return 1
    return 0

def get_piece_type_int(piece_value: int) -> int:
    """
    Return the raw PieceType integer value (1-12).
    Handles both Agent 1 and Agent 2 encodings.
    """
    if piece_value > AGENT2_OFFSET:
        return piece_value - AGENT2_OFFSET
    return abs(piece_value) # Fallback for legacy or safety, though shouldn't be negative for pieces anymore

def get_agent2_value(piece_type_value: int) -> int:
    """Convert a raw piece type value (1-12) to Agent 2's encoded value."""
    return piece_type_value + AGENT2_OFFSET
