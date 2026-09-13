"""
play.py
-------
Interactive command-line interface for human-to-human play and perft benchmarking
in Chaturanga (Murray/Golombek ruleset).

Usage:
    python play.py

Commands during gameplay:
    <r1>,<c1> <r2>,<c2>  - Make a move (e.g. '6,4 5,4')
    perft <depth>        - Run perft tree validation at given depth
    q                    - Quit the game
"""

import time
from chaturanga import (
    ChaturangaEnv,
    get_game_outcome,
    get_all_legal_moves,
    make_move,
    perft,
    WHITE
)


def play_game() -> None:
    """An interactive console loop to play Chaturanga manually."""
    env = ChaturangaEnv()
    board = env.board
    player = WHITE

    print("=" * 45)
    print("        Welcome to ChaturangZero!         ")
    print("=" * 45)
    print("Enter moves in format: start_row,start_col end_row,end_col")
    print("Example: '6,4 5,4'")
    print("Type 'perft <depth>' to run a correctness check.")
    print("Type 'q' to quit.\n")

    while True:
        env.board = board
        env.current_player = player
        env.render()

        # Check for game over
        outcome = get_game_outcome(board, player)
        if outcome is not None:
            player_str = "White" if player == WHITE else "Black"
            if outcome == 1:
                print(f"Game Over! Current player ({player_str}) WINS!")
            elif outcome == -1:
                print(f"Game Over! Current player ({player_str}) LOSES!")
            else:
                print("Game Over! DRAW.")
            break

        # Get legal moves
        legal_moves = get_all_legal_moves(board, player)
        print(f"Number of legal moves available: {len(legal_moves)}")

        try:
            user_input = input("Enter move: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting game.")
            break

        if user_input == 'q':
            print("Game aborted.")
            break

        if user_input.startswith("perft"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                depth = int(parts[1])
                print(f"Running perft at depth {depth}... this might take a moment.")
                start_time = time.time()
                nodes = perft(board, player, depth)
                elapsed = time.time() - start_time
                print(f"Perft({depth}) completed: {nodes} nodes in {elapsed:.2f} seconds.\n")
            else:
                print("Invalid perft command. Use: 'perft 1', 'perft 2', etc.")
            continue

        # Parse move input
        try:
            start_str, end_str = user_input.split()
            r1, c1 = map(int, start_str.split(','))
            r2, c2 = map(int, end_str.split(','))
            attempted_move = ((r1, c1), (r2, c2))

            if attempted_move in legal_moves:
                board = make_move(board, attempted_move[0], attempted_move[1])
                player = -player
            else:
                print("\n[!] Illegal move. Try again.\n")
        except ValueError:
            print("\n[!] Invalid format. Use: row,col row,col (e.g. 6,4 5,4)\n")


if __name__ == "__main__":
    play_game()
