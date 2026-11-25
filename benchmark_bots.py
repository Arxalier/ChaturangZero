#!/usr/bin/env python3
"""
benchmark_bots.py
-----------------
Benchmark tournament: Pits ChaturangZero Neural MCTS agent against all 4 Scholar Bots:
  1. NoviceBot (Random baseline)
  2. ChanakyaBot (Greedy material)
  3. AryabhataBot (Positional centrality)
  4. PaniniBot (Minimax Alpha-Beta)

Runs sequentially to prevent OpenMP deadlocks, printing live progress.
"""

import os
import sys
import time
import argparse
from typing import Dict, List
import numpy as np
import torch

# Limit CPU threads to prevent thread contention
torch.set_num_threads(4)

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chaturanga import (
    ChaturangaEnv, WHITE, BLACK, make_move, get_all_legal_moves, get_game_outcome
)
from bots import NoviceBot, ChanakyaBot, AryabhataBot, PaniniBot
from neural.model import ChaturangaNet
from training.mcts_agent import ChaturangPlayer


def create_opponent(name: str, seed: int):
    if name == "NoviceBot":
        return NoviceBot(seed=seed)
    elif name == "ChanakyaBot":
        return ChanakyaBot(seed=seed)
    elif name == "AryabhataBot":
        return AryabhataBot(seed=seed)
    elif name == "PaniniBot":
        return PaniniBot(seed=seed, depth=2)
    else:
        raise ValueError(f"Unknown bot: {name}")


def play_game(match_id: int, opp_name: str, ai_color: int, ai_player: ChaturangPlayer, max_plies: int) -> Dict:
    opp_player = create_opponent(opp_name, seed=42 + match_id * 13)
    env = ChaturangaEnv()
    pos_counts: Dict[bytes, int] = {}
    winner_str = "Draw"
    reason_str = "max_plies"

    color_name = "White" if ai_color == WHITE else "Black"
    print(f"\n[Game {match_id:02d}] AI ({color_name}) vs {opp_name} ...", end="", flush=True)

    t0 = time.time()
    ai_player.reset()

    for ply in range(max_plies):
        curr = env.current_player
        b = env.board.copy()

        key = b.tobytes() + bytes([1 if curr == 1 else 0])
        pos_counts[key] = pos_counts.get(key, 0) + 1
        if pos_counts[key] >= 3:
            winner_str = "Draw"
            reason_str = "3-fold repetition"
            break

        legal = get_all_legal_moves(b, curr)
        outcome = get_game_outcome(b, curr, legal_moves=legal)
        if outcome is not None:
            if outcome == 1:
                winning_side = curr
            elif outcome == -1:
                winning_side = -curr
            else:
                winning_side = 0

            if winning_side == ai_color:
                winner_str = "AI Win"
            elif winning_side == -ai_color:
                winner_str = "Bot Win"
            else:
                winner_str = "Draw"
            reason_str = "terminal"
            break

        if not legal:
            winner_str = "Draw"
            reason_str = "no_legal_moves"
            break

        if curr == ai_color:
            move = ai_player.get_move(b, curr, past_position_counts=pos_counts)
        else:
            move = opp_player.get_move(b, curr)

        env.step(move)
        if (ply + 1) % 20 == 0:
            print(f" p{ply+1}", end="", flush=True)

    elapsed = time.time() - t0
    plies = ply + 1
    w_pcs = int(np.count_nonzero(env.board > 0))
    b_pcs = int(np.count_nonzero(env.board < 0))

    print(f" {winner_str} in {plies} plies ({elapsed:.1f}s) [{reason_str}] (W={w_pcs}, B={b_pcs})", flush=True)

    return {
        "match_id": match_id,
        "opp_name": opp_name,
        "ai_color": color_name,
        "winner": winner_str,
        "reason": reason_str,
        "plies": plies,
        "elapsed": elapsed,
        "w_pcs": w_pcs,
        "b_pcs": b_pcs,
    }


def main():
    parser = argparse.ArgumentParser(description="Tournament vs Scholar Bots")
    parser.add_argument("--sims", type=int, default=40, help="MCTS sims per move for AI (default: 40)")
    parser.add_argument("--games-per-bot", type=int, default=2, help="Games per bot (default: 2, 1 White/1 Black)")
    parser.add_argument("--max-plies", type=int, default=180, help="Max plies per game (default: 180)")
    args = parser.parse_args()

    print("=" * 80)
    print("CHATURANGZERO TOURNAMENT: Neural MCTS vs. 4 Scholar Bots")
    print(f"MCTS Sims: {args.sims} | Games per bot: {args.games_per_bot} | Max Plies: {args.max_plies}")
    print("=" * 80)

    # 1. Load model once
    model = ChaturangaNet()
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "gen_final.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Successfully loaded checkpoints/gen_final.pt")

    ai_player = ChaturangPlayer(model=model, num_sims=args.sims, temperature=0.0, draw_penalty=0.30)

    bots = ["NoviceBot", "ChanakyaBot", "AryabhataBot", "PaniniBot"]
    results = []
    match_id = 1
    t_start = time.time()

    for bot in bots:
        for g in range(args.games_per_bot):
            ai_color = WHITE if g % 2 == 0 else BLACK
            r = play_game(match_id, bot, ai_color, ai_player, args.max_plies)
            results.append(r)
            match_id += 1

    total_time = time.time() - t_start
    total_games = len(results)

    # Summary
    header = f"\n{'Game':<5} | {'Opponent':<14} | {'AI Color':<8} | {'Result':<10} | {'Plies':<6} | {'Time (s)':<8} | {'Termination':<18} | {'Pieces'}"
    divider = "-" * 88
    lines = [header, divider]

    ai_wins = 0
    bot_wins = 0
    draws = 0
    per_bot = {b: {"wins": 0, "losses": 0, "draws": 0} for b in bots}

    for r in results:
        res = r["winner"]
        bot = r["opp_name"]
        if res == "AI Win":
            ai_wins += 1
            per_bot[bot]["wins"] += 1
        elif res == "Bot Win":
            bot_wins += 1
            per_bot[bot]["losses"] += 1
        else:
            draws += 1
            per_bot[bot]["draws"] += 1

        pcs = f"W={r['w_pcs']}, B={r['b_pcs']}"
        lines.append(
            f"#{r['match_id']:<4} | {r['opp_name']:<14} | {r['ai_color']:<8} | {r['winner']:<10} | {r['plies']:<6} | {r['elapsed']:<8.1f} | {r['reason']:<18} | {pcs}"
        )

    summary_lines = [
        "",
        "=" * 80,
        "TOURNAMENT SUMMARY: CHATURANGZERO NEURAL MCTS",
        "=" * 80,
        f"Total Games Run : {total_games} in {total_time:.1f}s (Avg {total_time/total_games:.1f}s/game)",
        f"ChaturangZero AI: {ai_wins} Wins ({ai_wins/total_games*100:.1f}%)",
        f"Scholar Bots    : {bot_wins} Wins ({bot_wins/total_games*100:.1f}%)",
        f"Draws           : {draws} ({draws/total_games*100:.1f}%)",
        "-" * 80,
        f"{'Bot Name':<16} | {'AI Wins':<8} | {'Bot Wins':<8} | {'Draws':<8} | {'AI Win Rate'}",
        "-" * 80,
    ]

    for b in bots:
        w = per_bot[b]["wins"]
        l = per_bot[b]["losses"]
        d = per_bot[b]["draws"]
        tot = w + l + d
        wr = (w / tot * 100) if tot > 0 else 0.0
        summary_lines.append(f"{b:<16} | {w:<8} | {l:<8} | {d:<8} | {wr:.1f}%")

    summary_lines.append("=" * 80)
    full_output = "\n".join(lines + summary_lines)
    print(full_output)

    # Append to report.txt
    report_path = os.path.join(os.path.dirname(__file__), "report.txt")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("TOURNAMENT BENCHMARK: CHATURANGZERO VS SCHOLAR BOTS\n")
        f.write("=" * 80 + "\n")
        f.write(full_output + "\n")

    print(f"\n✓ Successfully logged tournament results to {report_path}")


if __name__ == "__main__":
    main()
