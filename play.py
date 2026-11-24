"""
play.py
-------
Interactive command-line interface for human play against the AI
or human-to-human play in Chaturanga (Murray/Golombek ruleset).

Usage:
    python play.py --ai              # Play as White vs Trained AI (Black)
    python play.py --ai --side black # Play as Black vs Trained AI (White)
    python play.py                   # Two-player manual game

Commands during gameplay:
    <r1>,<c1> <r2>,<c2>  - Make a move (e.g. '6,4 5,4')
    perft <depth>        - Run perft tree validation at given depth
    q                    - Quit the game
"""

import sys
import time
import argparse
import numpy as np

from chaturanga import (
    ChaturangaEnv,
    get_game_outcome,
    get_all_legal_moves,
    make_move,
    perft,
    WHITE, BLACK
)


def play_game(vs_ai: bool = False, ai_side: int = BLACK, model_path: str = "checkpoints/gen_final.pt", sims: int = 40) -> None:
    """An interactive console loop to play Chaturanga manually or against the AI."""
    env = ChaturangaEnv()
    board = env.board
    player = WHITE
    ai_player = None

    if vs_ai:
        try:
            import torch
            from neural.model import ChaturangaNet
            from training.mcts_agent import ChaturangPlayer
            device = torch.device("cpu")
            model = ChaturangaNet().to(device)
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            ai_player = ChaturangPlayer(model=model, num_sims=sims, temperature=0.0)
            print(f"✓ Loaded Neural AI from {model_path} ({sims} MCTS sims)\n")
        except Exception as e:
            print(f"Warning: Could not load AI model ({e}). Playing 2-player manual mode.\n")
            vs_ai = False

    print("=" * 45)
    print("        Welcome to ChaturangZero!         ")
    print("=" * 45)
    if vs_ai:
        human_str = "White" if ai_side == BLACK else "Black"
        ai_str = "Black" if ai_side == BLACK else "White"
        print(f"Match: Human ({human_str}) vs ChaturangZero AI ({ai_str})")
    print("Enter moves in format: start_row,start_col end_row,end_col")
    print("Example: '6,4 5,4'")
    print("Type 'q' to quit.\n")

    pos_counts = {}

    while True:
        env.board = board
        env.current_player = player
        env.render()

        # Check for game over
        outcome = get_game_outcome(board, player)
        if outcome is not None:
            player_str = "White" if player == WHITE else "Black"
            if outcome == 1:
                print(f"Game Over! {player_str} WINS!")
            elif outcome == -1:
                print(f"Game Over! {player_str} LOSES!")
            else:
                print("Game Over! DRAW.")
            break

        pos_key = board.tobytes() + bytes([1 if player == 1 else 0])
        pos_counts[pos_key] = pos_counts.get(pos_key, 0) + 1
        if pos_counts[pos_key] >= 3:
            print("Game Over! DRAW by 3-fold repetition.")
            break

        legal_moves = get_all_legal_moves(board, player)
        print(f"Turn: {'White' if player == WHITE else 'Black'} | Legal moves: {len(legal_moves)}")

        # AI Turn
        if vs_ai and player == ai_side:
            print("ChaturangZero AI is thinking...", flush=True)
            t0 = time.time()
            ai_move = ai_player.get_move(board, player, past_position_counts=pos_counts)
            dt = time.time() - t0
            (r1, c1), (r2, c2) = ai_move
            print(f"AI plays: {r1},{c1} -> {r2},{c2} ({dt:.2f}s)\n")
            board = make_move(board, ai_move[0], ai_move[1])
            player = -player
            continue

        # Human Turn
        try:
            user_input = input("Enter your move (e.g. '6,4 5,4'): ").strip().lower()
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
                print(f"Running perft at depth {depth}...")
                start_time = time.time()
                nodes = perft(board, player, depth)
                elapsed = time.time() - start_time
                print(f"Perft({depth}): {nodes} nodes in {elapsed:.2f}s\n")
            continue

        try:
            start_str, end_str = user_input.split()
            r1, c1 = map(int, start_str.split(','))
            r2, c2 = map(int, end_str.split(','))
            attempted_move = ((r1, c1), (r2, c2))

            if attempted_move in legal_moves:
                board = make_move(board, attempted_move[0], attempted_move[1])
                player = -player
                print("")
            else:
                print("\n[!] Illegal move! That piece cannot move there or leaves King in check.\n")
        except ValueError:
            print("\n[!] Invalid format. Use: row,col row,col (e.g. 6,4 5,4)\n")


def main():
    parser = argparse.ArgumentParser(description="Play Chaturanga interactively against AI or Human")
    parser.add_argument("--ai", action="store_true", help="Play against ChaturangZero Neural AI")
    parser.add_argument("--side", choices=["white", "black"], default="white", help="Human side (default: white)")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulations for AI (default: 40)")
    parser.add_argument("--model", type=str, default="checkpoints/gen_final.pt", help="Model checkpoint path")
    args = parser.parse_args()

    ai_side = BLACK if args.side == "white" else WHITE
    play_game(vs_ai=args.ai, ai_side=ai_side, model_path=args.model, sims=args.sims)


if __name__ == "__main__":
    main()
