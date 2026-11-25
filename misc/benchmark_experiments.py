#!/usr/bin/env python3
"""
misc.benchmark_experiments
--------------------------
Runs a 10-game benchmark of ChaturangZero (Neural MCTS @ 50/200 sims) vs. Random Bot,
compares it against the pre-bootstrap baseline, and logs summaries to report.txt.
"""

import os
import sys
import time
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


def run_experiment(num_games: int = 10, sims: int = 50, max_turns: int = 250):
    model_path = "checkpoints/gen_final.pt"
    print("=" * 72)
    print(f"EXPERIMENT: Neural MCTS @ {sims} Sims vs Random Bot ({num_games} Games)")
    print(f"Model Checkpoint: {model_path} | Max Plies: {max_turns}")
    print("=" * 72)

    device = torch.device("cpu")
    model = ChaturangaNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    ai_player = ChaturangPlayer(model=model, num_sims=sims, temperature=0.0)

    results = []
    ai_wins = 0
    random_wins = 0
    draws = 0

    t_start = time.time()

    for game_idx in range(1, num_games + 1):
        g_start = time.time()
        ai_color = WHITE if (game_idx % 2 != 0) else BLACK
        ai_side_str = "White" if ai_color == WHITE else "Black"
        opp_side_str = "Black" if ai_color == WHITE else "White"

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
        g_duration = time.time() - g_start

        if final_outcome is None:
            res_type = "DRAW"
            draws += 1
        else:
            winner = env.current_player if final_outcome == 1 else -env.current_player
            if winner == ai_color:
                res_type = "AI WIN"
                ai_wins += 1
            else:
                res_type = "RANDOM WIN"
                random_wins += 1

        game_summary = {
            "game": game_idx,
            "ai_color": ai_side_str,
            "plies": turn,
            "result": res_type,
            "ended_by": ended_by,
            "pieces": f"W={w_pcs}, B={b_pcs}",
            "time_sec": g_duration
        }
        results.append(game_summary)

        print(f"Game {game_idx:02d}/{num_games:02d} | AI ({ai_side_str:<5}) vs Random ({opp_side_str:<5}) | "
              f"{turn:3d} plies ({g_duration:4.1f}s) | {res_type} ({ended_by}) [Pcs: W={w_pcs}, B={b_pcs}]", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"EXPERIMENT COMPLETE in {total_time:.1f}s")
    print(f"AI Wins: {ai_wins}/{num_games} ({ai_wins/num_games:.0%}) | Random Wins: {random_wins}/{num_games} | Draws: {draws}/{num_games}")
    print("=" * 72)

    # Append to report.txt
    report_entry = []
    report_entry.append("\n" + "=" * 75)
    report_entry.append(f"ABLATION BENCHMARK 2: FINE-TUNED MODEL (@ {sims} Sims)")
    report_entry.append("=" * 75)
    report_entry.append(f"Configuration:")
    report_entry.append(f"  * Model Checkpoint: {model_path} (AlphaGo-Style Teacher Bootstrapping)")
    report_entry.append(f"  * MCTS Depth      : {sims} simulations per move")
    report_entry.append(f"  * Max Plies Cap   : {max_turns} plies")
    report_entry.append(f"  * Opponent        : Uniform Random Bot")
    report_entry.append(f"  * Total Test Time : {total_time:.1f} seconds")
    report_entry.append("-" * 75)
    report_entry.append(f"Summary:")
    report_entry.append(f"  * ChaturangZero (AI) Wins : {ai_wins} / {num_games} ({ai_wins/num_games:.1%})")
    report_entry.append(f"  * Random Bot Wins         : {random_wins} / {num_games} ({random_wins/num_games:.1%})")
    report_entry.append(f"  * Draws                   : {draws} / {num_games} ({draws/num_games:.1%})")
    report_entry.append("-" * 75)
    report_entry.append(f"{'Game':<6} | {'AI Color':<10} | {'Plies':<8} | {'Result':<12} | {'Termination':<18} | {'Pieces':<12}")
    report_entry.append("-" * 75)
    for r in results:
        report_entry.append(f"{r['game']:<6} | {r['ai_color']:<10} | {r['plies']:<8} | {r['result']:<12} | {r['ended_by']:<18} | {r['pieces']:<12}")
    report_entry.append("=" * 75)

    entry_str = "\n".join(report_entry) + "\n"
    with open("report.txt", "a") as f:
        f.write(entry_str)
    print("\n✓ Appended full comparative results to report.txt successfully!")


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_experiment(num_games=games, sims=sims)
