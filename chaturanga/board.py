"""
chaturanga.board
----------------
Board setup, state transitions, pawn promotion, and board visualization.

In Chaturanga (Murray/Golombek ruleset):
    - The board is an 8x8 grid (the Ashtapada).
    - Setup is symmetrical: King on e-file (col 4), Minister on d-file (col 3).
    - Pawns automatically promote exclusively to Ministers upon reaching the opposite back rank.
    - All state transitions via make_move return a fresh copy of the board to facilitate
      tree exploration (e.g., MCTS) without state corruption.
"""

from typing import Optional, Tuple
import numpy as np
from .pieces import (
    EMPTY, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING,
    WHITE, BLACK, PIECE_SYMBOLS
)


def create_starting_board() -> np.ndarray:
    """
    Sets up the initial 8x8 Chaturanga board using Murray/Golombek symmetry.
    
    Back rank order (files 0 to 7):
        Chariot, Horse, Elephant, Minister, King, Elephant, Horse, Chariot
        
    Row 0: Black major pieces
    Row 1: Black pawns
    Row 6: White pawns
    Row 7: White major pieces
    """
    board = np.zeros((8, 8), dtype=np.int8)

    # --- BLACK PIECES (Ranks 0 and 1) ---
    board[0] = [
        -CHARIOT, -HORSE, -ELEPHANT, -MINISTER,
        -KING, -ELEPHANT, -HORSE, -CHARIOT
    ]
    board[1] = [-PAWN] * 8

    # --- WHITE PIECES (Ranks 6 and 7) ---
    board[6] = [PAWN] * 8
    board[7] = [
        CHARIOT, HORSE, ELEPHANT, MINISTER,
        KING, ELEPHANT, HORSE, CHARIOT
    ]

    return board


def make_move(
    board: np.ndarray,
    start_pos: Tuple[int, int],
    end_pos: Tuple[int, int]
) -> np.ndarray:
    """
    Applies a move and returns a NEW board state.
    
    Returning an independent copy is essential for search algorithms (like MCTS
    and alpha-beta) to explore future tree nodes without corrupting current game states.
    
    Pawn Promotion:
        Pawns automatically promote to Ministers when reaching the opponent's back rank
        (row 0 for White, row 7 for Black).
    """
    new_board = board.copy()

    r, c = start_pos
    nr, nc = end_pos

    piece = new_board[r, c]
    player = WHITE if piece > 0 else BLACK

    # Execute move (capturing by overwriting target square)
    new_board[r, c] = EMPTY
    new_board[nr, nc] = piece

    # --- Pawn Promotion Logic ---
    if abs(piece) == PAWN:
        if (player == WHITE and nr == 0) or (player == BLACK and nr == 7):
            new_board[nr, nc] = MINISTER * player

    return new_board


def render_board(board: np.ndarray, current_player: Optional[int] = None) -> str:
    """
    Generates a human-readable text string of the board layout.
    """
    lines = []
    if current_player is not None:
        player_name = "White" if current_player == WHITE else "Black"
        lines.append(f"Current Player: {player_name}")

    lines.append("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = f"{r} "
        for c in range(8):
            row_str += PIECE_SYMBOLS.get(board[r, c], '?') + " "
        lines.append(row_str)

    return "\n".join(lines)
