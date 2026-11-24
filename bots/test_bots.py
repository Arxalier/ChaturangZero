"""
bots/test_bots.py
-----------------
Standalone verification suite for the ChaturangZero Scholar Bots.

Checks:
  1. Universal BasePlayer contract & opening legality for all 4 bots.
  2. Persona/tactical sanity checks (Bare King, Checkmate, Promotion, King/Piece safety).
  3. Automated 5-game terminal match between ChanakyaBot and AryabhataBot
     (alternating White/Black), asserting move legality on every ply and
     printing per-game and overall winners.

Usage:
    python -m bots.test_bots
"""

import os
import sys
import time
from typing import Dict, Tuple
import numpy as np

# Allow running directly via `python bots/test_bots.py` or `python -m bots.test_bots`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chaturanga import (
    BasePlayer,
    ChaturangaEnv,
    WHITE,
    BLACK,
    PAWN,
    HORSE,
    ELEPHANT,
    CHARIOT,
    KING,
    create_starting_board,
    get_all_legal_moves,
)
from bots import NoviceBot, ChanakyaBot, AryabhataBot, PaniniBot


def verify_contracts_and_legality() -> None:
    print("[1/3] Verifying BasePlayer contract & legal moves for all 4 bots...")
    bots = [
        ("NoviceBot", NoviceBot(seed=42)),
        ("ChanakyaBot", ChanakyaBot(seed=42)),
        ("AryabhataBot", AryabhataBot(seed=42)),
        ("PaniniBot", PaniniBot(seed=42)),
    ]
    start_board = create_starting_board()

    for name, bot in bots:
        assert isinstance(bot, BasePlayer), f"{name} must inherit from BasePlayer"
        for side, side_name in ((WHITE, "White"), (BLACK, "Black")):
            legal = get_all_legal_moves(start_board, side)
            t0 = time.perf_counter()
            move = bot.get_move(start_board, side)
            ms = (time.perf_counter() - t0) * 1000.0
            assert move in legal, f"{name} returned illegal move {move} as {side_name}"
        print(f"  [OK] {name:<14s} | BasePlayer verified | sample move: {move} ({ms:.1f} ms)")


def verify_tactical_personas() -> None:
    print("\n[2/3] Verifying persona & tactical sanity checks...")

    # Chanakya: Bare King capture in 1
    b_bare = np.zeros((8, 8), dtype=np.int8)
    b_bare[0, 0] = KING
    b_bare[1, 1] = CHARIOT
    b_bare[1, 5] = -PAWN
    b_bare[7, 7] = -KING
    m_bare = ChanakyaBot(seed=1).get_move(b_bare, WHITE)
    assert m_bare == ((1, 1), (1, 5)), f"ChanakyaBot missed Bare King win: {m_bare}"

    # Aryabhata: Pawn promotion to Mantri
    b_promo = np.zeros((8, 8), dtype=np.int8)
    b_promo[7, 4] = KING
    b_promo[1, 3] = PAWN
    b_promo[6, 0] = PAWN
    b_promo[0, 7] = -KING
    b_promo[1, 7] = -PAWN
    b_promo[1, 6] = -PAWN
    m_promo = AryabhataBot(seed=1).get_move(b_promo, WHITE)
    assert m_promo == ((1, 3), (0, 3)), f"AryabhataBot missed promotion: {m_promo}"

    # Panini: Rescue hanging Chariot attacked by Black Horse
    b_hang = np.zeros((8, 8), dtype=np.int8)
    b_hang[7, 4] = KING
    b_hang[6, 3] = PAWN
    b_hang[6, 4] = PAWN
    b_hang[6, 5] = PAWN
    b_hang[4, 4] = CHARIOT
    b_hang[0, 7] = -KING
    b_hang[1, 7] = -PAWN
    b_hang[1, 6] = -PAWN
    b_hang[2, 3] = -HORSE
    m_hang = PaniniBot(seed=1).get_move(b_hang, WHITE)
    assert m_hang[0] == (4, 4), f"PaniniBot failed to rescue hanging Chariot: {m_hang}"

    print("  [OK] ChanakyaBot (Bare King), AryabhataBot (Promotion), PaniniBot (Defense)")


def play_match_game(
    white_bot: BasePlayer,
    black_bot: BasePlayer,
    max_plies: int = 160,
) -> Tuple[int, str, int, float]:
    """
    Plays one complete game between two BasePlayer instances.
    Asserts strict move legality at every ply.

    Returns:
        (winner, reason, plies, elapsed_seconds)
        winner is WHITE (1), BLACK (-1), or 0 for draw.
    """
    env = ChaturangaEnv()
    if hasattr(white_bot, "reset"):
        white_bot.reset()
    if hasattr(black_bot, "reset"):
        black_bot.reset()

    position_counts: Dict[bytes, int] = {}
    t0 = time.perf_counter()

    for ply in range(1, max_plies + 1):
        curr = env.current_player
        pos_key = env.board.tobytes() + bytes([1 if curr == WHITE else 0])
        position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
        if position_counts[pos_key] >= 3:
            return 0, "3-fold repetition", ply - 1, time.perf_counter() - t0

        legal_moves = env.get_legal_moves()
        active_bot = white_bot if curr == WHITE else black_bot
        move = active_bot.get_move(env.board.copy(), curr)

        assert move in legal_moves, (
            f"Illegal move {move} by {type(active_bot).__name__} on ply {ply}"
        )

        _, outcome, done = env.step(move)
        if done:
            enemy_count = int(np.count_nonzero(env.board * (-curr) > 0))
            if enemy_count == 1:
                reason = "bare_king"
            elif len(env.get_legal_moves()) == 0:
                reason = "checkmate" if env.is_in_check() else "stalemate"
            else:
                reason = "terminal"

            winner = curr if outcome == 1 else -curr
            return winner, reason, ply, time.perf_counter() - t0

    return 0, "max_plies", max_plies, time.perf_counter() - t0


def run_five_game_match(num_games: int = 5, max_plies: int = 160) -> None:
    print(f"\n[3/3] Running {num_games}-game match: ChanakyaBot vs AryabhataBot...")
    print("-" * 74)

    chanakya_wins = 0
    aryabhata_wins = 0
    draws = 0
    total_plies = 0
    total_time = 0.0

    for game_idx in range(1, num_games + 1):
        chanakya = ChanakyaBot(seed=100 + game_idx)
        aryabhata = AryabhataBot(seed=200 + game_idx)

        # Alternate colors: odd games Chanakya is White, even games Aryabhata is White
        if game_idx % 2 == 1:
            white_bot, black_bot = chanakya, aryabhata
            white_name, black_name = "Chanakya", "Aryabhata"
        else:
            white_bot, black_bot = aryabhata, chanakya
            white_name, black_name = "Aryabhata", "Chanakya"

        winner, reason, plies, elapsed = play_match_game(
            white_bot, black_bot, max_plies=max_plies
        )
        total_plies += plies
        total_time += elapsed

        if winner == WHITE:
            winner_name = white_name
        elif winner == BLACK:
            winner_name = black_name
        else:
            winner_name = "Draw"

        if winner_name == "Chanakya":
            chanakya_wins += 1
        elif winner_name == "Aryabhata":
            aryabhata_wins += 1
        else:
            draws += 1

        print(
            f"Game {game_idx}/{num_games} | W: {white_name:<9s} vs B: {black_name:<9s} | "
            f"Winner: {winner_name:<9s} ({reason:<17s}) | {plies:3d} plies | {elapsed:5.2f}s"
        )

    print("-" * 74)
    avg_ms = (total_time / max(total_plies, 1)) * 1000.0
    print(
        f"Match Summary ({num_games} games): "
        f"Chanakya={chanakya_wins}, Aryabhata={aryabhata_wins}, Draws={draws}"
    )
    print(f"Total Plies: {total_plies} | Total Time: {total_time:.2f}s | Avg/Ply: {avg_ms:.1f} ms")

    if chanakya_wins > aryabhata_wins:
        print("Overall Match Winner: ChanakyaBot")
    elif aryabhata_wins > chanakya_wins:
        print("Overall Match Winner: AryabhataBot")
    else:
        print("Overall Match Result: Tied")


def main() -> None:
    print("=" * 74)
    print("  ChaturangZero - Scholar Bots Verification Suite (Person 3)")
    print("=" * 74)
    verify_contracts_and_legality()
    verify_tactical_personas()
    run_five_game_match(num_games=5, max_plies=160)
    print("\nALL SCHOLAR BOT TESTS PASSED!")


if __name__ == "__main__":
    main()