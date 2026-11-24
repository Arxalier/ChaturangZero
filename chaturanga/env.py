"""
chaturanga.env
--------------
Environment wrapper for the Chaturanga game engine.

Provides the ChaturangaEnv class which encapsulates the board state,
turn tracking, move execution, legal move queries, and terminal evaluation.
"""

from typing import List, Optional, Tuple
import numpy as np

from .pieces import (
    EMPTY, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING,
    WHITE, BLACK, PIECE_SYMBOLS
)
from .board import create_starting_board, make_move, render_board
from .rules import get_all_legal_moves, get_game_outcome, is_in_check, Move


class ChaturangaEnv:
    """
    Game environment for 2-player Chaturanga.
    
    Attributes:
        board (np.ndarray): 8x8 numpy array of int8 representing the board state.
        current_player (int): 1 for White, -1 for Black.
        move_history (List[Move]): Sequence of moves played in this session.
    """

    # Piece encodings preserved as class/instance attributes for backwards compatibility
    EMPTY = EMPTY
    PAWN = PAWN
    HORSE = HORSE
    ELEPHANT = ELEPHANT
    CHARIOT = CHARIOT
    MINISTER = MINISTER
    KING = KING

    def __init__(self):
        self.current_player: int = WHITE
        self.board: np.ndarray = self._create_starting_board()
        self.move_history: List[Move] = []
        self._cached_legal_moves: Optional[List[Move]] = None
        self._cached_outcome: Optional[int] = None

    def _create_starting_board(self) -> np.ndarray:
        """Sets up the initial Chaturanga board (Murray/Golombek symmetry)."""
        return create_starting_board()

    def reset(self) -> np.ndarray:
        """Resets the game to the starting position and returns the board."""
        self.current_player = WHITE
        self.board = self._create_starting_board()
        self.move_history.clear()
        self._cached_legal_moves = None
        self._cached_outcome = None
        return self.board

    def set_state(self, board: np.ndarray, player: int) -> None:
        """Sets the board and current player, clearing any cached state."""
        self.board = board.copy()
        self.current_player = player
        self._cached_legal_moves = None
        self._cached_outcome = None

    def get_legal_moves(self) -> List[Move]:
        """Returns all strictly legal moves for the current player."""
        if self._cached_legal_moves is None:
            self._cached_legal_moves = get_all_legal_moves(self.board, self.current_player)
        return self._cached_legal_moves

    def is_in_check(self, player: Optional[int] = None) -> bool:
        """Checks if the given player (or current player) is in check."""
        target_player = self.current_player if player is None else player
        return is_in_check(self.board, target_player)

    def step(self, move: Move) -> Tuple[np.ndarray, Optional[int], bool]:
        """
        Executes a move in the environment.
        
        Args:
            move: Tuple of ((start_row, start_col), (end_row, end_col))
            
        Returns:
            board (np.ndarray): New board state.
            outcome (Optional[int]): Game outcome for the player who just moved (1, -1, 0, or None).
            done (bool): True if the game is over, False otherwise.
        """
        start_pos, end_pos = move
        self.board = make_move(self.board, start_pos, end_pos)
        self.move_history.append(move)

        # Transition to next player and evaluate outcome
        next_player = -self.current_player
        self.current_player = next_player
        self._cached_legal_moves = get_all_legal_moves(self.board, self.current_player)
        self._cached_outcome = get_game_outcome(self.board, self.current_player, self._cached_legal_moves)

        done = self._cached_outcome is not None
        if done:
            outcome = -self._cached_outcome if self._cached_outcome != 0 else 0
        else:
            outcome = None

        return self.board, outcome, done

    def get_outcome(self) -> Optional[int]:
        """Returns the game outcome for the current player, or None if ongoing."""
        if self._cached_outcome is None:
            self._cached_outcome = get_game_outcome(self.board, self.current_player, self.get_legal_moves())
        return self._cached_outcome

    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        return self.get_outcome() is not None

    def render(self) -> None:
        """Prints a human-readable text representation of the board."""
        print(render_board(self.board, self.current_player))
        print("\n")

    def copy(self) -> 'ChaturangaEnv':
        """Creates an independent deep copy of the environment."""
        new_env = ChaturangaEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.move_history = list(self.move_history)
        new_env._cached_legal_moves = list(self._cached_legal_moves) if self._cached_legal_moves is not None else None
        new_env._cached_outcome = self._cached_outcome
        return new_env
