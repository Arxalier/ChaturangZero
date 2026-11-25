#!/usr/bin/env python3
"""
misc.test_vs_random
-------------------
Runs configurable N-game benchmarks pitting ChaturangZero (Neural MCTS)
against a uniform random opponent, alternating White and Black sides.
"""

import os
import sys
import argparse
import random
import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chaturanga import (
    ChaturangaEnv, get_game_outcome,
    WHITE, BLACK
)
from neural.model import ChaturangaNet
from training.mcts_agent import ChaturangPlayer


def run_benchmark(
    model_path: str = "checkpoints/gen_final.pt",
    num_games: int = 10,
    sims: int = 30,
    max_turns: int = 300
):
    print("=" * 70)
    print(f"CHATURANGZERO BENCHMARK: Neural MCTS vs Random Bot ({num_games} Games)")
    print(f"Model: {model_path} | MCTS Sims: {sims} | Max Plies (Capacity): {max_turns}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChaturangaNet().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✓ Successfully loaded weights from {model_path} on {device}\n")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return

    model.eval()
    ai_player = ChaturangPlayer(model=model, num_sims=sims, temperature=0.0)

    ai_wins = 0
    random_wins = 0
    draws = 0

    for game_idx in range(1, num_games + 1):
        # Alternate sides: odd games AI is White, even games AI is Black
        ai_color = WHITE if (game_idx % 2 != 0) else BLACK
        ai_name = f"AI ({'White' if ai_color == WHITE else 'Black'})"
        opp_name = f"Random ({'Black' if ai_color == WHITE else 'White'})"

        env = ChaturangaEnv()
        ai_player.reset()
        position_counts = {}
        turn = 0
        ended_by = "max_turns"

        while turn < max_turns:
            board = env.board
            curr = env.current_player

            # 3-fold repetition check
            pos_key = board.tobytes() + bytes([1 if curr == 1 else 0])
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
            if position_counts[pos_key] >= 3:
                ended_by = "3-fold repetition"
                break

            legal_moves = env.get_legal_moves()
            outcome = get_game_outcome(board, curr, legal_moves=legal_moves)
            if outcome is not None:
                ended_by = "terminal"
                break

            if curr == ai_color:
                move = ai_player.get_move(board, curr, past_position_counts=position_counts)
            else:
                move = random.choice(legal_moves)

            env.step(move)
            turn += 1

        final_outcome = get_game_outcome(env.board, env.current_player)
        w_pcs = int(np.count_nonzero(env.board > 0))
        b_pcs = int(np.count_nonzero(env.board < 0))

        if final_outcome is None:
            result_str = f"DRAW ({ended_by}) [Pcs: W={w_pcs}, B={b_pcs}]"
            draws += 1
        else:
            winner = env.current_player if final_outcome == 1 else -env.current_player
            if winner == ai_color:
                result_str = f"AI WIN ({ended_by}) [Pcs: W={w_pcs}, B={b_pcs}]"
                ai_wins += 1
            else:
                result_str = f"RANDOM WIN ({ended_by}) [Pcs: W={w_pcs}, B={b_pcs}]"
                random_wins += 1

        print(f"Game {game_idx:02d}/{num_games:02d} | {ai_name:<10} vs {opp_name:<12} | {turn:3d} plies | {result_str}")

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"ChaturangZero (AI) Wins : {ai_wins} / {num_games} ({ai_wins / num_games:.1%})")
    print(f"Random Bot Wins         : {random_wins} / {num_games} ({random_wins / num_games:.1%})")
    print(f"Draws                   : {draws} / {num_games} ({draws / num_games:.1%})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ChaturangZero vs Random Bot")
    parser.add_argument("sims_pos", nargs="?", type=int, default=None, help="MCTS sims (positional)")
    parser.add_argument("games_pos", nargs="?", type=int, default=None, help="Num games (positional)")
    parser.add_argument("turns_pos", nargs="?", type=int, default=None, help="Max plies (positional)")
    parser.add_argument("--sims", type=int, default=30, help="MCTS simulations per move (default: 30)")
    parser.add_argument("--games", type=int, default=10, help="Number of benchmark games (default: 10)")
    parser.add_argument("--max-turns", type=int, default=300, help="Max plies before draw (default: 300)")
    parser.add_argument("--model", type=str, default="checkpoints/gen_final.pt", help="Path to checkpoint")
    args = parser.parse_args()

    # Allow positional arguments to override flags for backward compatibility
    sims = args.sims_pos if args.sims_pos is not None else args.sims
    games = args.games_pos if args.games_pos is not None else args.games
    max_turns = args.turns_pos if args.turns_pos is not None else args.max_turns

    run_benchmark(model_path=args.model, num_games=games, sims=sims, max_turns=max_turns)


if __name__ == "__main__":
    main()
