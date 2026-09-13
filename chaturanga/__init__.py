"""
ChaturangZero: Core Game Engine
================================
A modular rule engine and environment for Chaturanga (ancient Indian chess)
implementing the Murray/Golombek ruleset.

Submodules:
    - pieces: Piece and player constants, Sanskrit aliases, display symbols.
    - moves: Pseudo-legal move generators for each piece type.
    - board: Board setup, make_move state transition, pawn promotion, rendering.
    - rules: King checks, legal move filtering, game outcomes, and perft.
    - env: High-level ChaturangaEnv environment class.
"""

from .pieces import (
    EMPTY, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING,
    WHITE, BLACK,
    PADATI, ASHWA, GAJA, HASTI, RATHA, MANTRI, RAJA,
    PIECE_SYMBOLS, PIECE_NAMES
)
from .moves import (
    is_on_board,
    get_chariot_moves,
    get_horse_moves,
    get_elephant_moves,
    get_minister_moves,
    get_king_moves,
    get_pawn_moves
)
from .board import (
    create_starting_board,
    make_move,
    render_board
)
from .rules import (
    Move,
    find_king,
    is_in_check,
    get_all_legal_moves,
    get_game_outcome,
    perft
)
from .env import ChaturangaEnv

__all__ = [
    # Environment
    "ChaturangaEnv",
    "Move",
    # Pieces
    "EMPTY", "PAWN", "HORSE", "ELEPHANT", "CHARIOT", "MINISTER", "KING",
    "WHITE", "BLACK",
    "PADATI", "ASHWA", "GAJA", "HASTI", "RATHA", "MANTRI", "RAJA",
    "PIECE_SYMBOLS", "PIECE_NAMES",
    # Board & Moves
    "create_starting_board",
    "make_move",
    "render_board",
    "is_on_board",
    "get_chariot_moves",
    "get_horse_moves",
    "get_elephant_moves",
    "get_minister_moves",
    "get_king_moves",
    "get_pawn_moves",
    # Rules
    "find_king",
    "is_in_check",
    "get_all_legal_moves",
    "get_game_outcome",
    "perft",
]
