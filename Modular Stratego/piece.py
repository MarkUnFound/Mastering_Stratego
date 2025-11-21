# stratego_modular/piece.py

from enum import Enum

class PieceType(Enum):
    FLAG = 1
    SPY = 2
    SCOUT = 3
    MINER = 4
    SERGEANT = 5
    LIEUTENANT = 6
    CAPTAIN = 7
    MAJOR = 8
    COLONEL = 9
    GENERAL = 10
    MARSHAL = 11
    BOMB = 12

# Total number of piece types
NUM_PIECE_TYPES = 12

PIECE_NAMES = {
    PieceType.FLAG: 'F', PieceType.SPY: '1', PieceType.SCOUT: '2', PieceType.MINER: '3', PieceType.SERGEANT: '4',
    PieceType.LIEUTENANT: '5', PieceType.CAPTAIN: '6', PieceType.MAJOR: '7', PieceType.COLONEL: '8',
    PieceType.GENERAL: '9', PieceType.MARSHAL: 'M', PieceType.BOMB: 'B'
}

# Piece ranks for battle resolution
PIECE_RANKS = {
    PieceType.FLAG: 1,
    PieceType.SPY: 2,
    PieceType.SCOUT: 3,
    PieceType.MINER: 4,
    PieceType.SERGEANT: 5,
    PieceType.LIEUTENANT: 6,
    PieceType.CAPTAIN: 7,
    PieceType.MAJOR: 8,
    PieceType.COLONEL: 9,
    PieceType.GENERAL: 10,
    PieceType.MARSHAL: 11,
    PieceType.BOMB: 12
}

# Special battle rules
SPECIAL_BATTLES = {
    (PieceType.SPY, PieceType.MARSHAL): True,  # Spy wins against Marshal
    (PieceType.MINER, PieceType.BOMB): True    # Miner wins against Bomb
}