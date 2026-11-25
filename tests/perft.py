"""
perft.py
--------
Revamped performance and validation testing utility for ChaturangZero:
  1. Perft (Performance Test): Recursively counts leaf nodes with move-by-move breakdown ("divide").
  2. Playout Fuzz Testing: Fast random rollout validation with invariant checking.

Usage:
    python perft.py --depth 3
    python perft.py --depth 2 --divide
    python perft.py --fuzz 50 --max-moves 200
"""

import argparse
import os
import sys
import time
from typing import Optional

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from chaturanga import (
    create_starting_board,
    make_move,
    get_all_legal_moves,
    get_game_outcome,
    perft,
    WHITE
)
from validity_test import run_fuzz_test_suite


def perft_divide(board, player: int, depth: int) -> int:
    """
    Perft divide: Prints child node counts for every legal root move.
    Essential for identifying exactly which moves cause branching discrepancies.
    """
    print(f"\n--- Perft Divide (Depth {depth}) ---")
    legal_moves = get_all_legal_moves(board, player)
    total_nodes = 0

    for move in legal_moves:
        next_board = make_move(board, move[0], move[1])
        outcome = get_game_outcome(next_board, -player)
        if outcome is not None:
            subnodes = 1
        else:
            subnodes = perft(next_board, -player, depth - 1)

        total_nodes += subnodes
        start_str = f"{move[0][0]},{move[0][1]}"
        end_str = f"{move[1][0]},{move[1][1]}"
        print(f"  {start_str} -> {end_str}: {subnodes:,} nodes")

    print(f"Total: {total_nodes:,} nodes across {len(legal_moves)} root moves.\n")
    return total_nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance, Perft, and Playout Validity Testing for ChaturangZero."
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Perft depth to search (default: 2)"
    )
    parser.add_argument(
        "--divide",
        action="store_true",
        help="Print perft node count breakdown for each root move ('perft divide')"
    )
    parser.add_argument(
        "--fuzz", "-f",
        type=int,
        default=0,
        metavar="GAMES",
        help="Also run random rollout fuzz testing for N games (e.g. --fuzz 100)"
    )
    parser.add_argument(
        "--max-moves", "-m",
        type=int,
        default=300,
        help="Max plies per fuzz test game (default: 300)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducible fuzz testing"
    )
    args = parser.parse_args()

    board = create_starting_board()

    # 1. Perft Benchmark
    if args.depth > 0 and args.fuzz == 0 or (args.depth > 0 and not args.divide):
        print(f"Starting Perft calculation at depth {args.depth}...")
        start_time = time.perf_counter()
        if args.divide:
            nodes = perft_divide(board, WHITE, args.depth)
        else:
            nodes = perft(board, WHITE, args.depth)
        elapsed = time.perf_counter() - start_time

        print(f"Perft({args.depth}) result: {nodes:,} nodes visited in {elapsed:.3f} seconds.")
        if elapsed > 0:
            print(f"Speed: {nodes / elapsed:,.0f} nodes/sec")
    elif args.divide:
        perft_divide(board, WHITE, args.depth)

    # 2. Playout Fuzz Testing (if requested)
    if args.fuzz > 0:
        print("\nLaunching Random Playout Fuzz Validation...")
        run_fuzz_test_suite(
            num_games=args.fuzz,
            max_plies=args.max_moves,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
