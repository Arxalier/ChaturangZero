"""
Heuristic analysis tool: Evaluates move generators to estimate mobility and board centrality.
"""

import numpy as np
from chaturanga import (
    get_chariot_moves, get_horse_moves, get_elephant_moves, get_minister_moves,
    PAWN, ELEPHANT, MINISTER, HORSE, CHARIOT, KING, WHITE, PIECE_NAMES
)


def discover_values_from_rules():
    """Derives piece weights and square centrality purely from chaturanga.moves."""
    gens = {
        ELEPHANT: get_elephant_moves,
        MINISTER: get_minister_moves,
        HORSE: get_horse_moves,
        CHARIOT: get_chariot_moves,
    }
    raw_mob = {PAWN: 0.0}
    square_mob = np.zeros((8, 8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            # Pawn reach: 1 forward + diagonal capture squares (weighted 0.5 each)
            pm = (1.0 + (0.5 if c > 0 else 0.0) + (0.5 if c < 7 else 0.0)) if r > 0 else 0.0
            raw_mob[PAWN] += pm
            square_mob[r, c] += pm

            b = np.zeros((8, 8), dtype=np.int8)
            for p, fn in gens.items():
                b[r, c] = p
                m = len(fn(b, r, c, WHITE))
                raw_mob[p] = raw_mob.get(p, 0.0) + m
                if p != CHARIOT:  # Chariot has constant 14 everywhere on empty board
                    square_mob[r, c] += m
                b[r, c] = 0

    base = raw_mob[PAWN] / 64.0
    vals = {KING: 0.0, PAWN: 1.0}
    for p in gens:
        # Sublinear dampening (avg_mob^0.85) accounts for board-occupancy blocking on sliders
        vals[p] = round(((raw_mob[p] / 64.0) / base) ** 0.85, 2)

    center_bonus = 0.20 * (square_mob - square_mob.min()) / max(square_mob.max() - square_mob.min(), 1e-6)
    return vals, center_bonus


if __name__ == "__main__":
    vals, center_bonus = discover_values_from_rules()
    print("Derived Piece Evaluations:")
    print("-" * 35)
    order = [PAWN, ELEPHANT, MINISTER, HORSE, CHARIOT, KING]
    for p in order:
        name = PIECE_NAMES.get(p, f"Piece {p}")
        print(f"  {name:<20}: {vals[p]:.2f}")
    print("-" * 35)
    print("\nPositional Centrality Bonus Matrix (Square Mobility Heatmap):")
    print(np.round(center_bonus, 2))
