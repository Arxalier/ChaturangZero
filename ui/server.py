"""
ui/server.py
------------
Zero-dependency web UI for the ChaturangZero engine (Milestone 1 demo).

Serves a single-page board explorer and a small JSON API. Every rule decision
behind the API - legal moves, check, promotion, terminal states, perft - is
delegated to the `chaturanga` package, so what the browser shows is exactly what
the engine computes. The HTTP layer uses only the Python standard library;
`numpy` is required because the engine itself uses it.

Usage:
    python ui/server.py                 # opens http://127.0.0.1:8000
    python ui/server.py --port 9000     # custom port
    python ui/server.py --no-browser    # do not auto-open a browser
"""

import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Allow `python ui/server.py` from the repository root or from inside ui/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - startup guard
    sys.exit(
        "ChaturangZero UI: numpy is not installed for this interpreter "
        f"({sys.executable}).\nInstall it with:  pip install -r requirements.txt"
    )

from chaturanga import WHITE, BLACK, PIECE_SYMBOLS, PIECE_NAMES  # noqa: E402
from ui.engine_api import (  # noqa: E402
    Session, codex, bot_catalogue, play_bot_move, SANSKRIT, ENGLISH,
)
from ui import scenarios as scen  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# One shared game session: the UI is a single-user demo tool.
SESSION = Session()
LOCK = threading.Lock()

# Cached AI Model for fast UI responses
CACHED_AI_MODEL = None
CACHED_CKPT_MTIME = 0
AI_NUM_SIMS = 40

def get_ai_model():
    global CACHED_AI_MODEL, CACHED_CKPT_MTIME
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "gen_final.pt")
    if os.path.exists(ckpt_path):
        mtime = os.path.getmtime(ckpt_path)
        if CACHED_AI_MODEL is None or mtime != CACHED_CKPT_MTIME:
            try:
                import torch
                try:
                    from neural.model import ChaturangaNet
                except ImportError:
                    from training.mcts_agent import DummyNeuralNet as ChaturangaNet
                model = ChaturangaNet()
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                model.eval()
                CACHED_AI_MODEL = model
                CACHED_CKPT_MTIME = mtime
            except Exception as e:
                print(f"Warning loading checkpoint: {e}")
    return CACHED_AI_MODEL

# Honest status board for the "engine components" panel in the UI.
COMPONENTS = [
    {"module": "chaturanga/pieces.py", "status": "done",
     "summary": "Piece encodings (+White / -Black), Sanskrit aliases, display symbols.",
     "items": ["WHITE/BLACK turn constants", "PAWN..KING integer codes",
               "PADATI/ASHWA/GAJA/HASTI/RATHA/MANTRI/RAJA aliases",
               "PIECE_SYMBOLS, PIECE_NAMES"]},
    {"module": "chaturanga/moves.py", "status": "done",
     "summary": "Pseudo-legal generators, one per piece type. No check filtering here.",
     "items": ["get_chariot_moves (slider)", "get_horse_moves (leaper)",
               "get_elephant_moves (2-square diagonal leaper)",
               "get_minister_moves", "get_king_moves", "get_pawn_moves", "is_on_board"]},
    {"module": "chaturanga/board.py", "status": "done",
     "summary": "Setup, immutable state transition, automatic promotion, ASCII render.",
     "items": ["create_starting_board", "make_move returns a fresh copy (MCTS-safe)",
               "Padati -> Mantri promotion on rank 0 / 7", "render_board"]},
    {"module": "chaturanga/rules.py", "status": "done",
     "summary": "Check detection, legality filter, terminal states, perft.",
     "items": ["find_king", "is_in_check", "get_all_legal_moves (filters pins & self-check)",
               "get_game_outcome (bare king, checkmate, stalemate-wins)", "perft"]},
    {"module": "chaturanga/env.py", "status": "done",
     "summary": "RL-style wrapper: reset / step / legal moves / outcome / copy.",
     "items": ["ChaturangaEnv.reset", "step -> (board, outcome, done)",
               "get_legal_moves", "is_game_over", "copy for tree search"]},
    {"module": "tests/", "status": "done",
     "summary": "Full invariant test suite: perft, validity fuzzing, move legality.",
     "items": ["tests/test_validity.py", "perft.py", "validity_test.py"]},
    {"module": "Phase 2 (Neural, MCTS, Training)", "status": "done",
     "summary": "Dual-headed ResNet architecture, PUCT MCTS, self-play training loop.",
     "items": ["neural/model.py (ChaturangaNet)", "neural/encoder.py (13x8x8, 4096-dim)",
               "training/mcts_agent.py (PUCT)", "training/self_play.py", "training/train.py"]},
    {"module": "bots/scholar_bots.py", "status": "done",
     "summary": "Themed heuristic opponents implementing the universal BasePlayer interface.",
     "items": ["ChanakyaBot (Aggressive Tactician, Depth-2 Alpha-Beta)",
               "AryabhataBot (Positional Sage, Piece-Square Tables)",
               "PaniniBot (Defensive Strategist, King & Piece Safety)",
               "NoviceBot (80% Random / 20% Greedy Capture)"]},
]

RULES_CARD = {
    "terminal": [
        {"name": "Checkmate", "reward": "mated player -1 / opponent +1",
         "detail": "King attacked and no legal move escapes, blocks, or captures."},
        {"name": "Baring the King", "reward": "+1 to the player who bares",
         "detail": "Opponent's last non-King piece is captured; checked before mate in "
                   "rules.get_game_outcome."},
        {"name": "Stalemate", "reward": "+1 to the stalemated player",
         "detail": "Not in check but zero legal moves - the stalemated player WINS under "
                   "Murray/Golombek, not a draw."},
        {"name": "Turn limit / draw", "reward": "reserved (0)",
         "detail": "get_game_outcome documents 0 for draws, but no turn cap is implemented yet."},
    ],
    "notes": [
        "8x8 uncheckered Ashtapada; board markings are decorative only.",
        "Kings face each other on the e-file, Ministers on the d-file.",
        "Padati promotes exclusively and automatically to Mantri.",
        "No castling, no pawn double-step, no en passant.",
    ],
}


def piece_legend():
    """Legend rows tying board glyphs to the engine's integer encoding."""
    rows = []
    for kind in (6, 5, 4, 3, 2, 1):  # King .. Pawn
        rows.append({
            "kind": kind,
            "white_code": kind, "black_code": -kind,
            "white_symbol": PIECE_SYMBOLS[kind], "black_symbol": PIECE_SYMBOLS[-kind],
            "name": PIECE_NAMES[kind], "sanskrit": SANSKRIT[kind], "english": ENGLISH[kind],
        })
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "ChaturangZeroUI/1.0"

    # ------------------------------------------------------------- utilities

    def log_message(self, fmt, *args):  # keep the console readable during a demo
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"  [api] {self.command} {self.path}\n")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _serve_static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------- routing

    def do_GET(self):
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)

        if not route.startswith("/api/"):
            self._serve_static(route)
            return

        with LOCK:
            SESSION.reset_calls()
            if route == "/api/bootstrap":
                self._send_json({
                    "codex": codex(),
                    "scenarios": scen.catalogue(),
                    "bots": bot_catalogue(),
                    "components": COMPONENTS,
                    "rules": RULES_CARD,
                    "legend": piece_legend(),
                    "default_sims": AI_NUM_SIMS,
                    "state": SESSION.state(),
                })
            elif route == "/api/state":
                self._send_json(SESSION.state())
            elif route == "/api/inspect":
                r, c = int(query["r"][0]), int(query["c"][0])
                self._send_json(SESSION.inspect_square(r, c))
            elif route == "/api/threats":
                player = int(query.get("player", ["1"])[0])
                self._send_json(SESSION.threat_map(WHITE if player > 0 else BLACK))
            else:
                self.send_error(404, "Unknown API route")

    def do_POST(self):
        route = urlparse(self.path).path
        body = self._read_body()

        with LOCK:
            SESSION.reset_calls()
            try:
                if route == "/api/new":
                    SESSION.new_game()
                    self._send_json({"result": {"ok": True}, "state": SESSION.state()})

                elif route == "/api/scenario":
                    board, player, label, note = scen.build(body["id"])
                    SESSION.load_position(board, player, label, note)
                    self._send_json({"result": {"ok": True}, "state": SESSION.state()})

                elif route == "/api/move":
                    start = tuple(body["from"])
                    end = tuple(body["to"])
                    result = SESSION.play(start, end)
                    self._send_json({"result": result, "state": SESSION.state()})

                elif route == "/api/undo":
                    ok = SESSION.undo()
                    self._send_json({"result": {"ok": ok}, "state": SESSION.state()})

                elif route == "/api/random":
                    from chaturanga import get_all_legal_moves
                    import random
                    moves = get_all_legal_moves(SESSION.env.board, SESSION.env.current_player)
                    if not moves:
                        self._send_json({"result": {"ok": False, "error": "No legal moves"},
                                         "state": SESSION.state()})
                        return
                    start, end = random.choice(moves)
                    result = SESSION.play(start, end)
                    result["move"] = [list(start), list(end)]
                    self._send_json({"result": result, "state": SESSION.state()})

                elif route in ("/api/ai_move", "/api/chaturang_move", "/api/bot_move"):
                    bot_id = body.get("bot", "alphazero")
                    sims = body.get("sims", AI_NUM_SIMS)
                    sims = max(1, min(1000, int(sims)))
                    result = play_bot_move(SESSION, bot_id=bot_id, model_loader=get_ai_model, num_sims=sims)
                    self._send_json({"result": result, "state": SESSION.state()})

                elif route == "/api/perft":
                    depth = max(1, min(4, int(body.get("depth", 2))))
                    self._send_json({"result": SESSION.run_perft(depth),
                                     "state": SESSION.state()})
                else:
                    self.send_error(404, "Unknown API route")
            except (KeyError, ValueError, TypeError) as exc:
                self._send_json({"result": {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                                 "state": SESSION.state()}, status=400)


def main():
    parser = argparse.ArgumentParser(description="Run the ChaturangZero engine explorer UI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulations per AI move (default: 40)")
    args = parser.parse_args()

    global AI_NUM_SIMS
    AI_NUM_SIMS = args.sims

    port = args.port
    for attempt in range(10):  # step forward if the port is busy
        try:
            httpd = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            port += 1
    else:
        sys.exit(f"Could not bind a port in {args.port}..{args.port + 9}")

    url = f"http://{args.host}:{port}"
    print("=" * 62)
    print("  ChaturangZero - Engine Explorer (Milestone 1)")
    print("=" * 62)
    print(f"  Serving  : {url}")
    print(f"  Engine   : chaturanga package, numpy {np.__version__}")
    print(f"  AI MCTS  : {AI_NUM_SIMS} simulations per move (temperature=0.0)")
    print(f"  Python   : {sys.version.split()[0]} ({sys.executable})")
    print("  Stop with Ctrl+C")
    print("=" * 62)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
