"""
ui.engine_api
-------------
Thin adapter layer between the HTTP layer and the `chaturanga` package.

Every value exposed to the browser is produced by calling the real engine
functions (`chaturanga.moves`, `chaturanga.board`, `chaturanga.rules`,
`chaturanga.env`). Nothing about the rules is re-implemented here or in the
frontend: this module only reshapes engine output into JSON and records which
engine functions were called so the UI can display the call trace.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chaturanga import (
    ChaturangaEnv,
    EMPTY, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING,
    WHITE, BLACK,
    PIECE_SYMBOLS, PIECE_NAMES,
    create_starting_board,
    make_move,
    render_board,
    get_chariot_moves,
    get_horse_moves,
    get_elephant_moves,
    get_minister_moves,
    get_king_moves,
    get_pawn_moves,
    find_king,
    is_in_check,
    get_all_legal_moves,
    get_game_outcome,
    perft,
)

# Maps a piece type to the exact generator function in chaturanga.moves that
# produces its pseudo-legal moves (mirrors the dispatch in rules.py).
GENERATORS = {
    PAWN: ("get_pawn_moves", get_pawn_moves),
    HORSE: ("get_horse_moves", get_horse_moves),
    ELEPHANT: ("get_elephant_moves", get_elephant_moves),
    CHARIOT: ("get_chariot_moves", get_chariot_moves),
    MINISTER: ("get_minister_moves", get_minister_moves),
    KING: ("get_king_moves", get_king_moves),
}

SANSKRIT = {
    PAWN: "Padati",
    HORSE: "Ashwa",
    ELEPHANT: "Gaja",
    CHARIOT: "Ratha",
    MINISTER: "Mantri",
    KING: "Raja",
}

ENGLISH = {
    PAWN: "Foot Soldier",
    HORSE: "Horse",
    ELEPHANT: "Elephant",
    CHARIOT: "Chariot",
    MINISTER: "Minister",
    KING: "King",
}

MODERN = {
    PAWN: "Pawn (no double-step / en passant)",
    HORSE: "Knight",
    ELEPHANT: "Alfil (2-square diagonal leaper)",
    CHARIOT: "Rook",
    MINISTER: "Queen, restricted to 1 diagonal step",
    KING: "King (no castling)",
}

MECHANICS = {
    PAWN: "1 square straight forward into an empty square; captures 1 square diagonally forward.",
    HORSE: "Leaps to the opposite corner of a 2x3 rectangle, jumping over anything in between.",
    ELEPHANT: "Leaps exactly 2 squares diagonally, jumping over the intervening square.",
    CHARIOT: "Slides any distance along a rank or file until blocked, capturing, or the edge.",
    MINISTER: "Exactly 1 square diagonally, in any of the 4 directions.",
    KING: "Exactly 1 square in any of the 8 directions; may not move into check.",
}

# Decorative Ashtapada markings (quadrant corners). Purely cosmetic: the rules
# documents note the markings have no effect on play.
MARKED_SQUARES = [
    (r, c) for r in (0, 3, 4, 7) for c in (0, 3, 4, 7)
]


def algebraic(pos: Tuple[int, int]) -> str:
    """Converts (row, col) to chess-style algebraic notation, e.g. (7, 4) -> 'e1'."""
    r, c = pos
    return f"{'abcdefgh'[c]}{8 - r}"


class Session:
    """
    Holds one live game for the UI: a ChaturangaEnv plus an undo stack and a
    record of which engine functions were called while serving the last request.
    """

    def __init__(self) -> None:
        self.env = ChaturangaEnv()
        self.undo_stack: List[Tuple[np.ndarray, int]] = []
        self.captured: List[int] = []
        self.capture_stack: List[int] = []
        self.calls: List[str] = []
        self.scenario: str = "Standard opening position"
        self.note: str = ""
        self.starting_player: int = WHITE

    # ---------------------------------------------------------------- helpers

    def _log(self, name: str) -> None:
        if name not in self.calls:
            self.calls.append(name)

    def reset_calls(self) -> None:
        self.calls = []

    # ------------------------------------------------------------- mutations

    def new_game(self) -> None:
        self.env.reset()
        self._log("ChaturangaEnv.reset")
        self._log("create_starting_board")
        self.undo_stack.clear()
        self.captured.clear()
        self.capture_stack.clear()
        self.scenario = "Standard opening position"
        self.note = ""
        self.starting_player = WHITE

    def load_position(self, board: np.ndarray, player: int, scenario: str, note: str = "") -> None:
        self.env.set_state(board, player)
        self.env.move_history = []
        self.undo_stack.clear()
        self.captured.clear()
        self.capture_stack.clear()
        self.scenario = scenario
        self.note = note
        self.starting_player = player

    def play(self, start: Tuple[int, int], end: Tuple[int, int]) -> Dict[str, Any]:
        """Validates a move against get_all_legal_moves, then applies it via env.step."""
        legal = get_all_legal_moves(self.env.board, self.env.current_player)
        self._log("get_all_legal_moves")
        self._log("is_in_check")
        self._log("make_move")
        if (start, end) not in legal:
            return {"ok": False, "error": f"Illegal move {algebraic(start)}->{algebraic(end)} "
                                          f"(rejected by rules.get_all_legal_moves)"}

        self.undo_stack.append((self.env.board.copy(), self.env.current_player))
        captured = int(self.env.board[end])
        mover = int(self.env.board[start])
        promoted = abs(mover) == PAWN and end[0] in (0, 7)

        _, outcome, done = self.env.step((start, end))
        self._log("ChaturangaEnv.step")
        self._log("get_game_outcome")

        self.capture_stack.append(captured)
        if captured != EMPTY:
            self.captured.append(captured)

        return {
            "ok": True,
            "outcome_for_mover": outcome,
            "done": done,
            "captured": captured or None,
            "promoted": promoted,
        }

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        board, player = self.undo_stack.pop()
        self.env.set_state(board, player)
        if self.env.move_history:
            self.env.move_history.pop()
        if self.capture_stack:
            taken = self.capture_stack.pop()
            if taken != EMPTY and self.captured:
                self.captured.pop()
        return True

    # ----------------------------------------------------------------- state

    def piece_info(self, code: int) -> Optional[Dict[str, Any]]:
        if code == EMPTY:
            return None
        kind = abs(int(code))
        return {
            "code": int(code),
            "kind": kind,
            "player": WHITE if code > 0 else BLACK,
            "symbol": PIECE_SYMBOLS.get(int(code), "?"),
            "name": PIECE_NAMES.get(kind, "?"),
            "sanskrit": SANSKRIT[kind],
            "english": ENGLISH[kind],
        }

    def outcome_report(self) -> Dict[str, Any]:
        """
        Calls rules.get_game_outcome for the player on turn and reconstructs WHICH
        terminal rule fired, using the same engine primitives rules.py uses.
        """
        board, player = self.env.board, self.env.current_player
        outcome = get_game_outcome(board, player)
        self._log("get_game_outcome")
        if outcome is None:
            return {"over": False, "outcome": None, "rule": None, "headline": None, "detail": None}

        side = "White" if player == WHITE else "Black"
        other = "Black" if player == WHITE else "White"
        enemy_pieces = int(np.count_nonzero(board[board * -player > 0]))
        own_pieces = int(np.count_nonzero(board[board * player > 0]))

        if enemy_pieces == 1:
            rule, headline = "bare_king", f"{side} wins - Bare King"
            detail = f"{other} has only their Raja left, so {side} scores +1 immediately."
        elif own_pieces == 1:
            rule, headline = "bare_king", f"{side} loses - Bare King"
            detail = f"{side} has only their Raja left, so {other} scores +1 immediately."
        elif is_in_check(board, player):
            rule, headline = "checkmate", f"{side} is checkmated"
            detail = f"{side} is in check with zero legal moves: {side} -1, {other} +1."
        else:
            rule, headline = "stalemate", f"{side} wins - Stalemate"
            detail = (f"{side} is NOT in check but has zero legal moves. Under Murray/Golombek "
                      f"the stalemated player WINS (+1), unlike modern chess.")
        return {"over": True, "outcome": outcome, "rule": rule,
                "headline": headline, "detail": detail}

    def state(self) -> Dict[str, Any]:
        board, player = self.env.board, self.env.current_player

        legal = get_all_legal_moves(board, player)
        self._log("get_all_legal_moves")

        # Group legal moves by origin square so the UI can list them per piece.
        by_origin: Dict[str, Dict[str, Any]] = {}
        for (sr, sc), (er, ec) in legal:
            key = f"{sr},{sc}"
            info = by_origin.setdefault(key, {
                "from": [sr, sc],
                "from_alg": algebraic((sr, sc)),
                "piece": self.piece_info(int(board[sr, sc])),
                "targets": [],
            })
            info["targets"].append({
                "to": [er, ec],
                "to_alg": algebraic((er, ec)),
                "capture": int(board[er, ec]) != EMPTY,
                "promotion": abs(int(board[sr, sc])) == PAWN and er in (0, 7),
            })

        squares = []
        for r in range(8):
            for c in range(8):
                squares.append({
                    "r": r, "c": c,
                    "alg": algebraic((r, c)),
                    "piece": self.piece_info(int(board[r, c])),
                    "marked": (r, c) in MARKED_SQUARES,
                })

        white_king = find_king(board, WHITE)
        black_king = find_king(board, BLACK)
        self._log("find_king")
        white_check = is_in_check(board, WHITE)
        black_check = is_in_check(board, BLACK)
        self._log("is_in_check")

        history = []
        for i, ((sr, sc), (er, ec)) in enumerate(self.env.move_history):
            p = self.starting_player if i % 2 == 0 else -self.starting_player
            history.append({
                "index": i + 1,
                "player": "White" if p == WHITE else "Black",
                "from": [sr, sc], "to": [er, ec],
                "text": f"{algebraic((sr, sc))}-{algebraic((er, ec))}",
                "coords": f"({sr},{sc})->({er},{ec})",
            })

        counts = {}
        for side, sign in (("white", WHITE), ("black", BLACK)):
            per_kind = {}
            for kind in (PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING):
                n = int(np.count_nonzero(board == kind * sign))
                if n:
                    per_kind[SANSKRIT[kind]] = n
            counts[side] = {
                "total": int(np.count_nonzero(board[board * sign > 0])),
                "non_king": int(np.count_nonzero(board[board * sign > 0])) - (1 if find_king(board, sign) else 0),
                "per_kind": per_kind,
            }

        return {
            "squares": squares,
            "matrix": [[int(v) for v in row] for row in board],
            "ascii": render_board(board, player),
            "current_player": int(player),
            "current_player_name": "White" if player == WHITE else "Black",
            "legal_moves": [[[sr, sc], [er, ec]] for (sr, sc), (er, ec) in legal],
            "legal_count": len(legal),
            "legal_by_origin": list(by_origin.values()),
            "white_king": list(white_king) if white_king else None,
            "black_king": list(black_king) if black_king else None,
            "white_in_check": bool(white_check),
            "black_in_check": bool(black_check),
            "history": history,
            "captured": [self.piece_info(p) for p in self.captured],
            "counts": counts,
            "outcome": self.outcome_report(),
            "scenario": self.scenario,
            "note": self.note,
            "can_undo": bool(self.undo_stack),
            "engine_calls": self.calls,
        }

    # ------------------------------------------------------------ inspection

    def inspect_square(self, r: int, c: int) -> Dict[str, Any]:
        """
        Shows the two-stage pipeline for one piece: the raw pseudo-legal squares
        straight out of chaturanga.moves, and the subset that survives the
        check filter in chaturanga.rules.get_all_legal_moves.
        """
        board, player = self.env.board, self.env.current_player
        code = int(board[r, c])
        if code == EMPTY:
            return {"empty": True}

        kind = abs(code)
        owner = WHITE if code > 0 else BLACK
        gen_name, gen = GENERATORS[kind]
        pseudo = gen(board, r, c, owner)
        self._log(gen_name)

        legal: List[Tuple[int, int]] = []
        if owner == player:
            for end in pseudo:
                if not is_in_check(make_move(board, (r, c), end), player):
                    legal.append(end)
            self._log("make_move")
            self._log("is_in_check")

        filtered = [p for p in pseudo if p not in legal] if owner == player else []

        return {
            "empty": False,
            "square": {"r": r, "c": c, "alg": algebraic((r, c))},
            "piece": self.piece_info(code),
            "generator": f"chaturanga.moves.{gen_name}",
            "mechanics": MECHANICS[kind],
            "modern": MODERN[kind],
            "is_own_turn": owner == player,
            "pseudo": [{"to": [pr, pc], "alg": algebraic((pr, pc)),
                        "capture": int(board[pr, pc]) != EMPTY} for pr, pc in pseudo],
            "legal": [{"to": [pr, pc], "alg": algebraic((pr, pc)),
                       "capture": int(board[pr, pc]) != EMPTY,
                       "promotion": kind == PAWN and pr in (0, 7)} for pr, pc in legal],
            "filtered": [{"to": [pr, pc], "alg": algebraic((pr, pc))} for pr, pc in filtered],
            "engine_calls": self.calls,
        }

    def threat_map(self, player: int) -> Dict[str, Any]:
        """
        The squares on which `player`'s Raja would stand in check.

        Rather than re-deriving attack patterns in the UI, each square is probed
        the way the engine itself decides legality: drop the Raja there and ask
        rules.is_in_check. That keeps pawn capture squares, pins and king
        adjacency exactly consistent with get_all_legal_moves, which does the
        same thing one move at a time.
        """
        board = self.env.board
        base = board.copy()
        king = find_king(board, player)
        if king is not None:
            base[king] = EMPTY  # move the real Raja out of the way while probing

        unsafe: Dict[str, bool] = {}
        for r in range(8):
            for c in range(8):
                if int(base[r, c]) * player > 0:
                    continue  # own piece stands here; the Raja could not go there anyway
                probe = base.copy()
                probe[r, c] = KING * player
                if is_in_check(probe, player):
                    unsafe[f"{r},{c}"] = True
        self._log("is_in_check")
        self._log("find_king")
        return {"player": player, "attacked": unsafe, "engine_calls": self.calls}

    def run_perft(self, depth: int) -> Dict[str, Any]:
        """Runs rules.perft from the current position, with per-move breakdown at the root."""
        import time
        board, player = self.env.board, self.env.current_player
        t0 = time.time()
        nodes = perft(board, player, depth)
        elapsed = time.time() - t0
        self._log("perft")

        divide = []
        if depth >= 1:
            for start, end in get_all_legal_moves(board, player):
                child = make_move(board, start, end)
                if depth == 1:
                    sub = 1
                elif get_game_outcome(child, -player) is not None:
                    sub = 1
                else:
                    sub = perft(child, -player, depth - 1)
                divide.append({"move": f"{algebraic(start)}-{algebraic(end)}", "nodes": sub})
            divide.sort(key=lambda d: (-d["nodes"], d["move"]))

        return {
            "depth": depth,
            "nodes": nodes,
            "seconds": round(elapsed, 3),
            "nps": int(nodes / elapsed) if elapsed > 0 else None,
            "divide": divide,
            "engine_calls": self.calls,
        }


def codex() -> List[Dict[str, Any]]:
    """
    Builds the piece reference by calling each generator from chaturanga.moves on
    an otherwise empty board, so the diagrams are the engine's actual output.
    """
    out = []
    for kind in (KING, MINISTER, ELEPHANT, HORSE, CHARIOT, PAWN):
        gen_name, gen = GENERATORS[kind]
        board = np.zeros((8, 8), dtype=np.int8)
        r, c = 4, 4
        board[r, c] = kind * WHITE
        plain = gen(board, r, c, WHITE)

        # Second diagram: a ring of blockers, to show leaping vs blocking.
        blocked = np.zeros((8, 8), dtype=np.int8)
        blocked[r, c] = kind * WHITE
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0):
                    blocked[r + dr, c + dc] = PAWN * BLACK
        leaps = gen(blocked, r, c, WHITE)

        out.append({
            "kind": kind,
            "symbol": PIECE_SYMBOLS[kind],
            "name": PIECE_NAMES[kind],
            "sanskrit": SANSKRIT[kind],
            "english": ENGLISH[kind],
            "modern": MODERN[kind],
            "mechanics": MECHANICS[kind],
            "generator": f"chaturanga.moves.{gen_name}",
            "origin": [r, c],
            "free_moves": [[a, b] for a, b in plain],
            "surrounded_moves": [[a, b] for a, b in leaps],
            "free_count": len(plain),
            "surrounded_count": len(leaps),
        })
    return out

# ---------------------------------------------------------------------------
# Bot Registry & Opponent Dispatch (Person 3 — Scholar Bots + AlphaZero)
# ---------------------------------------------------------------------------

BOT_SPECS: List[Dict[str, str]] = [
    {
        "id": "alphazero",
        "name": "ChaturangZero",
        "subtitle": "AlphaZero Neural MCTS",
        "family": "neural",
    },
    {
        "id": "chanakya",
        "name": "Chanakya",
        "subtitle": "Aggressive Tactician · Depth-2 Alpha-Beta",
        "family": "scholar",
    },
    {
        "id": "aryabhata",
        "name": "Aryabhata",
        "subtitle": "Positional Sage · Piece-Square Tables",
        "family": "scholar",
    },
    {
        "id": "panini",
        "name": "Panini",
        "subtitle": "Defensive Strategist · King & Piece Safety",
        "family": "scholar",
    },
    {
        "id": "novice",
        "name": "Novice",
        "subtitle": "Baseline · 80% Random / 20% Capture",
        "family": "scholar",
    },
]

_BOT_BY_ID: Dict[str, Dict[str, str]] = {spec["id"]: spec for spec in BOT_SPECS}


def bot_catalogue() -> List[Dict[str, str]]:
    """Returns metadata for all selectable UI opponent bots."""
    return list(BOT_SPECS)


def create_bot(bot_id: str, model_loader=None, num_sims: int = 40):
    """Instantiates a BasePlayer matching `bot_id`."""
    from bots import NoviceBot, ChanakyaBot, AryabhataBot, PaniniBot

    if bot_id == "chanakya":
        return ChanakyaBot()
    if bot_id == "aryabhata":
        return AryabhataBot()
    if bot_id == "panini":
        return PaniniBot()
    if bot_id == "novice":
        return NoviceBot()
    if bot_id in ("alphazero", "chaturangzero"):
        from training.mcts_agent import ChaturangPlayer
        model = model_loader() if callable(model_loader) else None
        return ChaturangPlayer(model=model, num_sims=num_sims, temperature=0.1)

    raise ValueError(f"Unknown bot id: {bot_id!r}")


def play_bot_move(
    session: Session,
    bot_id: str = "alphazero",
    model_loader=None,
    num_sims: int = 40,
) -> Dict[str, Any]:
    """
    Queries the selected BasePlayer for a move via get_move() and applies it
    to `session` through the rules engine.
    """
    import time

    if bot_id not in _BOT_BY_ID:
        return {"ok": False, "error": f"Unknown opponent '{bot_id}'"}

    board = session.env.board
    player = session.env.current_player

    if get_game_outcome(board, player) is not None:
        return {"ok": False, "error": "Game is already over"}

    legal = get_all_legal_moves(board, player)
    if not legal:
        return {"ok": False, "error": "No legal moves available"}

    t0 = time.perf_counter()
    bot = create_bot(bot_id, model_loader=model_loader, num_sims=num_sims)
    bot_cls = type(bot).__name__

    if bot_id == "alphazero":
        pos_counts: Dict[bytes, int] = {}
        for b, p in session.undo_stack:
            k = b.tobytes() + bytes([1 if p == WHITE else 0])
            pos_counts[k] = pos_counts.get(k, 0) + 1
        move = bot.get_move(board.copy(), player, past_position_counts=pos_counts)
    else:
        move = bot.get_move(board.copy(), player)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    session._log(f"{bot_cls}.get_move")

    start, end = move
    result = session.play(start, end)
    result["move"] = [list(start), list(end)]
    result["bot"] = bot_id
    result["bot_name"] = _BOT_BY_ID[bot_id]["name"]
    result["elapsed_ms"] = round(elapsed_ms, 1)
    return result