"""
chaturanga.rules
----------------
Game rules, check detection, legal move filtering, game outcome determination,
and perft move generator correctness verification.

Rules implemented (Murray/Golombek ruleset):
    1. Check:
       A King is in check if any enemy piece can capture it on the next turn.
    2. Legal Moves:
       Only moves that do not leave or place one's own King in check are legal.
    3. Game Over Conditions & Rewards:
       - Bare King: Baring the opponent's King (leaving the opponent with only their
         King on the board) awards an immediate win (+1 for baring player, -1 for bared).
       - Checkmate: If a player is in check and has no legal moves, they lose (-1 for
         player in check, +1 for attacking player).
       - Stalemate: If a player is NOT in check but has no legal moves, that player
         WINS (+1 for stalemated player) per traditional Murray/Golombek rules.
"""

from typing import List, Optional, Tuple
import numpy as np
from .pieces import (
    PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING
)
from .moves import (
    get_chariot_moves,
    get_horse_moves,
    get_elephant_moves,
    get_minister_moves,
    get_king_moves,
    get_pawn_moves
)
from .board import make_move

# Type alias for a move: ((start_row, start_col), (end_row, end_col))
Move = Tuple[Tuple[int, int], Tuple[int, int]]


def find_king(board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
    """
    Locates the King's coordinates (row, col) for the given player.
    Returns None if the King is not present on the board.
    """
    target_king = KING * player
    result = np.where(board == target_king)
    if len(result[0]) > 0:
        return (int(result[0][0]), int(result[1][0]))
    return None


def is_in_check(board: np.ndarray, player: int) -> bool:
    """
    Determines if the specified player's King is under attack by any enemy piece.
    """
    king_pos = find_king(board, player)
    if not king_pos:
        return False

    enemy = -player

    # Iterate through the entire board looking for enemy pieces
    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece * enemy > 0:
                enemy_moves = []
                abs_piece = abs(piece)

                if abs_piece == PAWN:
                    enemy_moves = get_pawn_moves(board, r, c, enemy)
                elif abs_piece == HORSE:
                    enemy_moves = get_horse_moves(board, r, c, enemy)
                elif abs_piece == ELEPHANT:
                    enemy_moves = get_elephant_moves(board, r, c, enemy)
                elif abs_piece == CHARIOT:
                    enemy_moves = get_chariot_moves(board, r, c, enemy)
                elif abs_piece == MINISTER:
                    enemy_moves = get_minister_moves(board, r, c, enemy)
                elif abs_piece == KING:
                    enemy_moves = get_king_moves(board, r, c, enemy)

                if king_pos in enemy_moves:
                    return True

    return False


def get_all_legal_moves(board: np.ndarray, player: int) -> List[Move]:
    """
    Generates all STRICTLY LEGAL moves for the given player.
    Filters out any move that leaves or places the player's King in check.
    
    Returns:
        List of tuples: [((start_r, start_c), (end_r, end_c)), ...]
    """
    legal_moves: List[Move] = []

    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece * player > 0:  # Friendly piece
                abs_piece = abs(piece)
                pseudo_moves: List[Tuple[int, int]] = []

                if abs_piece == PAWN:
                    pseudo_moves = get_pawn_moves(board, r, c, player)
                elif abs_piece == HORSE:
                    pseudo_moves = get_horse_moves(board, r, c, player)
                elif abs_piece == ELEPHANT:
                    pseudo_moves = get_elephant_moves(board, r, c, player)
                elif abs_piece == CHARIOT:
                    pseudo_moves = get_chariot_moves(board, r, c, player)
                elif abs_piece == MINISTER:
                    pseudo_moves = get_minister_moves(board, r, c, player)
                elif abs_piece == KING:
                    pseudo_moves = get_king_moves(board, r, c, player)

                # Filter out moves that leave own King in check
                for end_pos in pseudo_moves:
                    simulated_board = make_move(board, (r, c), end_pos)
                    if not is_in_check(simulated_board, player):
                        legal_moves.append(((r, c), end_pos))

    return legal_moves


def get_game_outcome(board: np.ndarray, player: int) -> Optional[int]:
    """
    Evaluates the board to see if the game has ended.
    
    Returns:
         1 if 'player' wins
        -1 if 'player' loses
         0 for a draw (reserved for turn limits / repetition)
        None if the game is still ongoing
    """
    enemy = -player

    # 1. Bare King Rule (Insta-win if opponent has only 1 piece left: their King)
    enemy_pieces = np.count_nonzero(board[board * enemy > 0])
    if enemy_pieces == 1:
        return 1  # Player wins! Opponent's King is bared.

    # Check if the player themselves was bared
    player_pieces = np.count_nonzero(board[board * player > 0])
    if player_pieces == 1:
        return -1  # Player loses.

    # 2. Checkmate and Stalemate
    legal_moves = get_all_legal_moves(board, player)

    if len(legal_moves) == 0:
        if is_in_check(board, player):
            return -1  # Checkmate: Player has no legal moves and is in check (Loss)
        else:
            return 1   # Stalemate: Player has no legal moves but is safe (Win per Murray/Golombek!)

    return None  # Game continues


def perft(board: np.ndarray, player: int, depth: int) -> int:
    """
    Recursively counts all leaf nodes in the game tree up to a given depth.
    Used for verifying move generator correctness and consistency.
    """
    if depth == 0:
        return 1

    nodes = 0
    legal_moves = get_all_legal_moves(board, player)

    for move in legal_moves:
        next_board = make_move(board, move[0], move[1])

        # If the game is over, this branch of the tree ends here
        outcome = get_game_outcome(next_board, -player)
        if outcome is not None:
            nodes += 1
        else:
            nodes += perft(next_board, -player, depth - 1)

    return nodes
