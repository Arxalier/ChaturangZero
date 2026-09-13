"""
chaturanga.pieces
-----------------
Defines the numerical piece representations, Sanskrit aliases, player constants,
and display symbols for the ancient Indian board game Chaturanga (Murray/Golombek ruleset).

Piece Encodings:
    Positive integers represent White pieces.
    Negative integers represent Black pieces.
    0 represents an empty square.
"""

# Player turn indicators
WHITE = 1
BLACK = -1

# Piece Constants (Integer representations)
EMPTY = 0
PAWN = 1
HORSE = 2
ELEPHANT = 3
CHARIOT = 4
MINISTER = 5
KING = 6

# Traditional Sanskrit Aliases
PADATI = PAWN       # Foot Soldier
ASHWA = HORSE       # Horse / Steed
GAJA = ELEPHANT     # Elephant (also known as Hasti)
HASTI = ELEPHANT
RATHA = CHARIOT     # Chariot (ancestor of Rook)
MANTRI = MINISTER   # Minister / Counselor (ancestor of Queen)
RAJA = KING         # King

# ASCII/Unicode display symbols used for board rendering
PIECE_SYMBOLS = {
    EMPTY: '.',
    PAWN: 'P',
    HORSE: 'H',
    ELEPHANT: 'E',
    CHARIOT: 'R',
    MINISTER: 'M',
    KING: 'K',
    -PAWN: 'p',
    -HORSE: 'h',
    -ELEPHANT: 'e',
    -CHARIOT: 'r',
    -MINISTER: 'm',
    -KING: 'k'
}

PIECE_NAMES = {
    EMPTY: "Empty",
    PAWN: "Pawn / Padati",
    HORSE: "Horse / Ashwa",
    ELEPHANT: "Elephant / Gaja",
    CHARIOT: "Chariot / Ratha",
    MINISTER: "Minister / Mantri",
    KING: "King / Raja"
}
