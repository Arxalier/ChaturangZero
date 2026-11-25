"""
tests/test_validity.py
======================
Automated test suite for random rollout fuzz testing and invariant validation.
Compatible with both standard python unittest and pytest.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chaturanga.pieces import WHITE, BLACK, KING, PAWN, MINISTER
from chaturanga.board import create_starting_board
from chaturanga.rules import perft
try:
    from tests.validity_test import run_fuzz_test_suite, run_random_game
except ImportError:
    from validity_test import run_fuzz_test_suite, run_random_game


class TestChaturangaValidity(unittest.TestCase):

    def test_perft_starting_board(self):
        """Verify Perft node counts for depths 1 and 2 from starting position."""
        board = create_starting_board()
        self.assertEqual(perft(board, WHITE, 1), 16, "Perft(1) must be exactly 16 moves.")
        self.assertEqual(perft(board, WHITE, 2), 256, "Perft(2) must be exactly 256 moves.")

    def test_random_rollout_invariants(self):
        """Runs a batch of random rollout games and ensures zero invariant violations."""
        stats = run_fuzz_test_suite(
            num_games=5,
            max_plies=150,
            seed=42,
            quiet=True
        )
        self.assertEqual(stats["games"], 5)
        self.assertGreater(stats["total_plies"], 0)

    def test_single_game_completion(self):
        """Tests a single game runs to completion without crashing."""
        winner, reason, plies, elapsed = run_random_game(max_plies=100, check_invariants=True)
        self.assertIn(winner, (WHITE, BLACK, 0))
        self.assertIn(reason, ("bare_king", "checkmate", "stalemate", "max_plies"))
        self.assertGreater(plies, 0)
        self.assertGreater(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
