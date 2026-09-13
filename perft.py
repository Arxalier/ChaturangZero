"""
perft.py
--------
Command-line utility to run Perft (Performance Test) for Chaturanga.

Recursively counts leaf nodes to verify move generator correctness and speed.

Usage:
    python perft.py [--depth DEPTH]
    Example: python perft.py --depth 3
"""

import argparse
import time
from chaturanga import create_starting_board, perft, WHITE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Perft validation on starting board.")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Depth to search (default: 2)"
    )
    args = parser.parse_args()

    board = create_starting_board()
    print(f"Starting Perft calculation at depth {args.depth}...")
    start_time = time.time()
    nodes = perft(board, WHITE, args.depth)
    elapsed = time.time() - start_time

    print(f"Perft({args.depth}) result: {nodes} nodes visited in {elapsed:.3f} seconds.")
    if elapsed > 0:
        print(f"Speed: {nodes / elapsed:.1f} nodes/sec")


if __name__ == "__main__":
    main()
