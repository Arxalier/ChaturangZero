"""
Standalone verification for the arena (Person 4).

    python -m arena.test_arena        # 2-game match + sample plot, prints the result
    python -m pytest arena -q         # same checks plus unit tests

The required check plays NoviceBot vs AryabhataBot (Person 3's bots). Until
bots/scholar_bots.py exists, it falls back to Random vs Greedy so the arena can
still be verified on its own.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from arena.elo import compute_elo, elo_diff_from_score, expected_score
from arena.match import (
    BARE_KING,
    CHECKMATE,
    ILLEGAL_MOVE,
    MAX_PLIES,
    REPETITION,
    STALEMATE,
    classify_terminal,
    play_game,
)
from arena.players import (
    GreedyPlayer,
    MinimaxPlayer,
    RandomPlayer,
    build_player,
    parse_spec,
    scholar_bots_available,
)
from chaturanga import (
    BasePlayer,
    BLACK,
    CHARIOT,
    KING,
    PAWN,
    WHITE,
    get_all_legal_moves,
)


def _empty():
    return np.zeros((8, 8), dtype=np.int8)


# ---------------------------------------------------------------------------
# Required standalone verification
# ---------------------------------------------------------------------------
def run_verification(out_dir: str) -> dict:
    if scholar_bots_available():
        specs = ["novice", "aryabhata"]
    else:
        print("bots/scholar_bots.py not found yet: verifying with Random vs Greedy instead.")
        specs = ["random", "greedy"]
    from arena.tournament import run_tournament

    res = run_tournament(specs, games_per_pair=2, workers=1, out_dir=out_dir, resume=False,
                         bootstrap=50, title="Arena verification", verbose=False)
    for g in res.games:
        assert g["reason"] not in (ILLEGAL_MOVE, "crash"), g.get("error")
        assert g["plies"] > 0
    plot = os.path.join(out_dir, "winrate_heatmap.png")
    assert os.path.exists(plot) and os.path.getsize(plot) > 1000
    return {"players": res.labels, "games": res.games, "plot": plot}


def test_verification_match_and_plot(tmp_path):
    info = run_verification(str(tmp_path))
    assert len(info["games"]) == 2
    whites = {g["white"] for g in info["games"]}
    assert whites == set(info["players"]), "colours must alternate"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_players_return_legal_moves():
    from chaturanga import create_starting_board

    board = create_starting_board()
    for p in (RandomPlayer(), GreedyPlayer(), MinimaxPlayer(2)):
        for side in (WHITE, BLACK):
            assert p.get_move(board.copy(), side) in get_all_legal_moves(board.copy(), side)


def test_greedy_and_minimax_take_bare_king_win():
    b = _empty()
    b[7, 0] = KING
    b[6, 1] = CHARIOT
    b[6, 5] = -PAWN
    b[0, 7] = -KING
    win = ((6, 1), (6, 5))
    assert GreedyPlayer().get_move(b, WHITE) == win
    assert MinimaxPlayer(2).get_move(b, WHITE) == win


def test_classify_terminal_reasons():
    # Bare king: Black has only its King left.
    b = _empty()
    b[7, 0], b[6, 0], b[0, 7] = KING, PAWN, -KING
    assert classify_terminal(b, BLACK, get_all_legal_moves(b, BLACK)) == (WHITE, BARE_KING)
    # Checkmate: Black king a8 (0,0), White chariots on rows 0 and 1, Black pawn keeps it un-bared.
    b = _empty()
    b[0, 0], b[7, 7], b[0, 7], b[1, 6], b[4, 4] = -KING, KING, CHARIOT, CHARIOT, -PAWN
    assert classify_terminal(b, BLACK, get_all_legal_moves(b, BLACK)) == (WHITE, CHECKMATE)


def test_stalemated_side_wins():
    # Black king in the corner, not in check, every square covered; its pawn is blocked.
    b = _empty()
    b[0, 0] = -KING
    b[7, 1] = CHARIOT        # covers column 1, including (0, 1)
    b[1, 7] = CHARIOT        # covers row 1, including (1, 0) and (1, 1)
    b[7, 7] = KING
    b[4, 4] = -PAWN
    b[5, 4] = PAWN           # blocks the black pawn
    legal = get_all_legal_moves(b, BLACK)
    assert legal == []
    assert classify_terminal(b, BLACK, legal) == (BLACK, STALEMATE)


class _Shuffler(BasePlayer):
    """Moves the King back and forth to force a repetition."""

    def get_move(self, board, player):
        moves = [m for m in get_all_legal_moves(board.copy(), player) if abs(int(board[m[0]])) == KING]
        return sorted(moves)[0]


def test_repetition_and_ply_cap_are_draws():
    b = _empty()
    b[7, 0], b[7, 7], b[0, 0], b[0, 7] = KING, PAWN, -KING, -PAWN
    b[6, 7], b[1, 7] = -PAWN, PAWN  # pawns blocked head to head
    r = play_game(_Shuffler(), _Shuffler(), "a", "b", start_board=b, max_plies=200)
    assert r.winner == 0 and r.reason == REPETITION
    r = play_game(_Shuffler(), _Shuffler(), "a", "b", start_board=b, max_plies=3)
    assert r.winner == 0 and r.reason == MAX_PLIES and r.plies == 3


class _Cheater(BasePlayer):
    def get_move(self, board, player):
        return ((0, 0), (5, 5))


def test_illegal_move_loses():
    r = play_game(_Cheater(), RandomPlayer(), "cheat", "rand", seed=1)
    assert r.winner == BLACK and r.reason == ILLEGAL_MOVE
    assert r.score_for("rand") == 1.0 and r.score_for("cheat") == 0.0


def test_spec_parsing():
    s = parse_spec("Master=az-master:ckpt=a/b.pt,sims=80")
    assert (s.label, s.name, s.params["sims"], s.params["ckpt"]) == ("Master", "az-master", 80, "a/b.pt")
    s = parse_spec("az:ckpt=x.pt,temp=0.05")
    assert s.name == "az" and s.params["temp"] == 0.05 and s.label.startswith("AZ(")
    assert parse_spec("minimax:depth=3").label == "Minimax-d3"
    assert build_player("random")[0] == "Random"


def test_elo_math():
    assert abs(expected_score(400, 0) - 10 / 11) < 1e-9
    assert abs(elo_diff_from_score(0.5)) < 1e-9
    # A beats B 75% of the time => about +191 Elo.
    games = [("A", "B", 1.0)] * 750 + [("A", "B", 0.0)] * 250
    t = compute_elo(["A", "B"], games, anchor="B", prior_draws=0, bootstrap=20)
    assert abs(t.get("A") - 400 * math.log10(3)) < 1.0
    assert t.get("B") == 0.0
    # Perfect scores stay finite thanks to the prior.
    t = compute_elo(["A", "B"], [("A", "B", 1.0)] * 10, anchor="B", bootstrap=0)
    assert 0 < t.get("A") < 1000


def test_az_player_untrained_plays_legal_move():
    from chaturanga import create_starting_board

    _, p = build_player("az-untrained:sims=4")
    b = create_starting_board()
    assert p.get_move(b.copy(), WHITE) in get_all_legal_moves(b.copy(), WHITE)


if __name__ == "__main__":
    out = os.path.join(tempfile.gettempdir(), "chaturangzero_arena_check")
    info = run_verification(out)
    for g in info["games"]:
        res = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[g["winner"]]
        print(f"{g['white']} (White) vs {g['black']} (Black): {res} by {g['reason']} after {g['plies']} plies")
    print(f"Sample plot: {info['plot']}")
    print("Arena verification passed.")


class _HistoryAware(BasePlayer):
    """Records what play_game hands a repetition-aware player."""

    def __init__(self):
        self.resets = 0
        self.seen_counts = []

    def reset(self):
        self.resets += 1

    def get_move(self, board, player, past_position_counts=None):
        self.seen_counts.append(dict(past_position_counts or {}))
        return random.choice(get_all_legal_moves(board, player))


def test_history_aware_players_get_reset_and_history():
    a = _HistoryAware()
    play_game(a, RandomPlayer(), "h", "r", max_plies=6, seed=3)
    play_game(RandomPlayer(), a, "r", "h", max_plies=6, seed=4)
    assert a.resets == 2
    first = a.seen_counts[0]
    assert sum(first.values()) == 1  # the start position, counted once before White moves
    assert all(sum(c.values()) >= 1 for c in a.seen_counts)
