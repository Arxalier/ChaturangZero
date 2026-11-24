#!/usr/bin/env python3
"""
test_vs_random.py
-----------------
Runs a 10-game benchmark match of ChaturangZero (Neural MCTS) vs Random Bot.
Alternates colors: 5 games as White, 5 games as Black.
"""

import sys
import random
import torch
import numpy as np

from chaturanga import (
    ChaturangaEnv, get_game_outcome,
    WHITE, BLACK
)
from neural.model import ChaturangaNet
from training.mcts_agent import ChaturangPlayer


def run_benchmark(model_path: str = "checkpoints/gen_final.pt", num_games: int = 10, sims: int = 25):
    print("=" * 65)
    print(f"CHATURANGZERO BENCHMARK: Neural MCTS vs Random Bot ({num_games} Games)")
    print(f"Model Checkpoint: {model_path} | MCTS Sims: {sims}")
    print("=" * 65)

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
        max_turns = 120
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
        if final_outcome is None:
            result_str = f"DRAW ({ended_by})"
            draws += 1
        else:
            winner = env.current_player if final_outcome == 1 else -env.current_player
            if winner == ai_color:
                result_str = "AI WIN"
                ai_wins += 1
            else:
                result_str = "RANDOM WIN"
                random_wins += 1

        print(f"Game {game_idx:02d}/{num_games:02d} | {ai_name} vs {opp_name} | {turn:3d} turns | {result_str}")

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 65)
    print(f"ChaturangZero (AI) Wins : {ai_wins} / {num_games} ({ai_wins / num_games:.1%})")
    print(f"Random Bot Wins         : {random_wins} / {num_games} ({random_wins / num_games:.1%})")
    print(f"Draws                   : {draws} / {num_games} ({draws / num_games:.1%})")
    print("=" * 65)


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_benchmark(num_games=games, sims=sims)
