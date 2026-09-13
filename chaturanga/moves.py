"""
chaturanga.moves
----------------
Pseudo-legal move generators for all six Chaturanga piece types:
    - Chariot (Ratha): Orthogonal sliding moves along ranks and files.
    - Horse (Ashwa): L-shaped 2x3 leaps over intervening pieces.
    - Elephant (Gaja): Strict 2-square diagonal leaps over intervening pieces.
    - Minister (Mantri): 1-square diagonal steps in four directions.
    - King (Raja): 1-square steps in any orthogonal or diagonal direction.
    - Foot Soldier (Padati / Pawn): 1-square forward move, 1-square diagonal forward capture.

Note:
    These functions generate pseudo-legal moves; they do not filter out moves that
    leave the king in check. Check filtering is handled in chaturanga.rules.
"""

from typing import List, Tuple
import numpy as np


def is_on_board(r: int, c: int) -> bool:
    """Checks whether the given row and column indices lie within the 8x8 board."""
    return 0 <= r < 8 and 0 <= c < 8


def get_chariot_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates orthogonal sliding moves for the Chariot (Ratha).
    Slides along ranks and files until blocked or capturing an enemy piece.
    """
    moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while is_on_board(nr, nc):
            target = board[nr, nc]
            if target == 0:
                moves.append((nr, nc))  # Empty square, keep sliding
            elif target * player < 0:
                moves.append((nr, nc))  # Enemy piece, capture and stop
                break
            else:
                break  # Blocked by own piece, stop
            nr += dr
            nc += dc

    return moves


def get_horse_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates L-shaped leaps for the Horse (Ashwa).
    Jumps over any intervening pieces to empty or enemy squares.
    """
    moves = []
    jumps = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    for dr, dc in jumps:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            target = board[nr, nc]
            # Square is empty or occupied by enemy piece
            if target * player <= 0:
                moves.append((nr, nc))

    return moves


def get_elephant_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates strict 2-square diagonal leaps for the Elephant (Gaja / Hasti).
    Leaps over any intervening piece to an empty or enemy square.
    """
    moves = []
    leaps = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

    for dr, dc in leaps:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            target = board[nr, nc]
            # Square is empty or occupied by enemy piece
            if target * player <= 0:
                moves.append((nr, nc))

    return moves


def get_minister_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates strictly 1-square diagonal moves for the Minister (Mantri).
    Moves to adjacent diagonal squares if empty or occupied by an enemy.
    """
    moves = []
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 4 diagonals

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            target = board[nr, nc]
            if target * player <= 0:
                moves.append((nr, nc))

    return moves


def get_king_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates 1-square moves in all 8 directions for the King (Raja).
    Note: Check filtering (suicidal moves) is handled downstream by rules.get_all_legal_moves.
    """
    moves = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),    # Orthogonal
        (-1, -1), (-1, 1), (1, -1), (1, 1)   # Diagonal
    ]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            target = board[nr, nc]
            if target * player <= 0:
                moves.append((nr, nc))

    return moves


def get_pawn_moves(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """
    Generates 1-square forward steps and diagonal captures for the Pawn / Foot Soldier (Padati).
    White moves 'up' (row decreases: -1).
    Black moves 'down' (row increases: +1).
    There is no initial double-step or en passant in Chaturanga.
    """
    moves = []
    forward_dir = -1 if player == 1 else 1

    # 1. Normal forward move (must be an empty square)
    nr, nc = r + forward_dir, c
    if is_on_board(nr, nc) and board[nr, nc] == 0:
        moves.append((nr, nc))

    # 2. Diagonal captures (must contain an enemy piece)
    capture_directions = [(forward_dir, -1), (forward_dir, 1)]
    for dr, dc in capture_directions:
        nr, nc = r + dr, c + dc
        if is_on_board(nr, nc):
            target = board[nr, nc]
            if target * player < 0:  # Enemy piece
                moves.append((nr, nc))

    return moves
