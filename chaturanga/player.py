"""
chaturanga.player
-----------------
Abstract BasePlayer interface shared by all Chaturanga bots and neural agents.
"""

import numpy as np
from typing import Tuple

Move = Tuple[Tuple[int, int], Tuple[int, int]]

class BasePlayer:
    """Universal interface for all Chaturanga bots and AI agents."""
    def get_move(self, board: np.ndarray, player: int) -> Move:
        """
        Takes the current 8x8 numpy board state and active player (1 for White, -1 for Black).
        Returns a legal move formatted as ((start_row, start_col), (end_row, end_col)).
        """
        raise NotImplementedError
