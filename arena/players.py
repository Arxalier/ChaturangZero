"""
arena.players
-------------
Baseline players for benchmarking, plus a registry that builds any player
(baseline, scholar bot or AlphaZero tier) from a short string spec.

Every player implements ``chaturanga.BasePlayer``::

    get_move(board: np.ndarray, player: int) -> ((sr, sc), (er, ec))

Spec grammar (used by every CLI in this package)::

    [LABEL=]name[:key=value,key=value,...]

    random
    greedy
    minimax:depth=2
    az-master:ckpt=checkpoints/gen_final.pt
    Gen0=az:ckpt=checkpoints/gen_0.pt,sims=25,temp=0.05
    chanakya                      (needs bots/scholar_bots.py from Person 3)

The LABEL is what appears in tables and plots. Without one, a readable
label is derived from the spec.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from chaturanga import (
    BasePlayer,
    KING,
    get_all_legal_moves,
    get_game_outcome,
    make_move,
)

Move = Tuple[Tuple[int, int], Tuple[int, int]]

# ---------------------------------------------------------------------------
# Material values used by the arena's own baselines.
#
# These follow the usual historical assessment of Chaturanga (the Chariot is
# by far the strongest piece; the Minister and Elephant are weak short-range
# pieces). They are deliberately independent of Person 3's scholar bots so the
# baselines stay fixed even if the scholar bots are retuned.
# ---------------------------------------------------------------------------
PIECE_VALUES = np.array([0.0, 1.0, 3.0, 1.5, 5.0, 2.0, 0.0], dtype=np.float64)
#                       EMPTY PAWN HORSE ELEPH CHAR MINIS KING
WIN_SCORE = 10_000.0


def material_balance(board: np.ndarray, player: int) -> float:
    """Material of ``player`` minus material of the opponent."""
    vals = PIECE_VALUES[np.abs(board).astype(np.int64)]
    signed = np.where(board > 0, vals, -vals)
    return float(signed.sum()) * player


def piece_count(board: np.ndarray, player: int) -> int:
    """Number of pieces (including the King) belonging to ``player``."""
    return int(np.count_nonzero(board * player > 0))


# ---------------------------------------------------------------------------
# Baseline players
# ---------------------------------------------------------------------------
class RandomPlayer(BasePlayer):
    """Uniformly random legal move. The Elo anchor of every tournament."""

    def get_move(self, board: np.ndarray, player: int) -> Move:
        moves = get_all_legal_moves(board.copy(), player)
        if not moves:
            raise ValueError("No legal moves available.")
        return random.choice(moves)


class GreedyPlayer(BasePlayer):
    """
    1-ply greedy: plays an immediate win if one exists, otherwise the move with
    the best material balance *after the move*, ties broken at random.

    It never looks at the opponent's reply, so it happily captures a pawn with
    a Chariot that is then taken by an Elephant leap. Ablation 3 measures this.
    """

    def get_move(self, board: np.ndarray, player: int) -> Move:
        moves = get_all_legal_moves(board.copy(), player)
        if not moves:
            raise ValueError("No legal moves available.")
        random.shuffle(moves)
        best_move, best_score = moves[0], -math.inf
        for move in moves:
            child = make_move(board, move[0], move[1])
            outcome = get_game_outcome(child, -player)  # from opponent's view
            if outcome is not None:
                score = -outcome * WIN_SCORE
            else:
                score = material_balance(child, player)
            if score > best_score:
                best_move, best_score = move, score
        return best_move


class MinimaxPlayer(BasePlayer):
    """
    Fixed-depth negamax with alpha-beta pruning over material balance.

    Terminal states (checkmate, stalemate, bare king) are scored exactly at
    every interior node. Leaves only check the cheap bare-king condition, so
    depth=2 sees every win or loss that happens within its own move and the
    opponent's reply.
    """

    def __init__(self, depth: int = 2):
        self.depth = depth

    def get_move(self, board: np.ndarray, player: int) -> Move:
        board = board.copy()
        moves = get_all_legal_moves(board, player)
        if not moves:
            raise ValueError("No legal moves available.")
        random.shuffle(moves)
        moves.sort(key=lambda m: -abs(int(board[m[1]])))  # captures first
        best_move, best_score = moves[0], -math.inf
        alpha, beta = -math.inf, math.inf
        for move in moves:
            child = make_move(board, move[0], move[1])
            score = -self._negamax(child, -player, self.depth - 1, -beta, -alpha, ply=1)
            if score > best_score:
                best_move, best_score = move, score
            alpha = max(alpha, score)
        return best_move

    def _negamax(self, board, player, depth, alpha, beta, ply) -> float:
        if depth == 0:
            if piece_count(board, -player) == 1:
                return WIN_SCORE - ply
            if piece_count(board, player) == 1:
                return -(WIN_SCORE - ply)
            return material_balance(board, player)

        moves = get_all_legal_moves(board, player)
        outcome = get_game_outcome(board, player, moves)
        if outcome is not None:
            # Prefer faster wins and slower losses.
            return outcome * (WIN_SCORE - ply)

        moves.sort(key=lambda m: -abs(int(board[m[1]])))
        best = -math.inf
        for move in moves:
            child = make_move(board, move[0], move[1])
            score = -self._negamax(child, -player, depth - 1, -beta, -alpha, ply + 1)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best


# ---------------------------------------------------------------------------
# AlphaZero player construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier:
    """Inference-time settings that turn one network into a skill tier."""

    name: str
    sims: int
    temperature: float
    blunder_rate: float = 0.0
    target_elo: Optional[float] = None
    note: str = ""


# Tiers from the project breakdown. The Novice tier uses sims=0 because that is
# the code path in ChaturangPlayer that samples the raw policy head; sims=1
# would instead run a one-node search and play the most-visited child.
TIERS: Dict[str, Tier] = {
    "master": Tier("Master", sims=80, temperature=0.05, target_elo=1800),
    "intermediate": Tier("Intermediate", sims=15, temperature=0.60, target_elo=1300),
    "novice": Tier("Novice", sims=0, temperature=1.0, blunder_rate=0.15, target_elo=700,
                   note="raw policy head (sims=0) with 15% random moves"),
}

_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def resolve_checkpoint(path: Optional[str]) -> str:
    """Find a checkpoint, trying the common locations used by the team."""
    candidates = []
    if path:
        candidates.append(path)
    env = os.environ.get("CHATURANGZERO_CKPT")
    if env:
        candidates.append(env)
    candidates += [
        "checkpoints/gen_final.pt",
        "training/gen_final.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No checkpoint found. Pass ckpt=<path> in the player spec, set "
        "CHATURANGZERO_CKPT, or place gen_final.pt in checkpoints/. Tried: "
        + ", ".join(candidates)
    )


def load_model(path: str, device: str = "cpu"):
    """Load a ChaturangaNet from a raw state_dict or a training checkpoint dict."""
    import torch
    from neural.model import ChaturangaNet

    key = (os.path.abspath(path), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = ChaturangaNet()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def make_alphazero(ckpt: Optional[str] = None, sims: int = 80, temperature: float = 0.05,
                   blunder_rate: float = 0.0, top_k: Optional[int] = None,
                   untrained: bool = False):
    """Build Person 2's AlphaZero player around a checkpoint."""
    from training.mcts_agent import AlphaZeroPlayer

    if untrained:
        import torch
        from neural.model import ChaturangaNet

        torch.manual_seed(0)
        model = ChaturangaNet()
        model.eval()
    else:
        model = load_model(resolve_checkpoint(ckpt))
    return AlphaZeroPlayer(model, num_sims=sims, temperature=temperature,
                           top_k=top_k, blunder_rate=blunder_rate)


# ---------------------------------------------------------------------------
# Scholar bots (Person 3). Imported lazily so the arena works before they land.
# ---------------------------------------------------------------------------
SCHOLAR_BOTS = {
    "novice": "NoviceBot",
    "chanakya": "ChanakyaBot",
    "aryabhata": "AryabhataBot",
    "panini": "PaniniBot",
}


def scholar_bots_available() -> bool:
    try:
        import bots.scholar_bots  # noqa: F401
        return True
    except ImportError:
        return False


def _make_scholar(name: str, **kwargs):
    try:
        import bots.scholar_bots as sb
    except ImportError as exc:
        raise ImportError(
            f"'{name}' needs bots/scholar_bots.py (Person 3), which is not in the repo yet."
        ) from exc
    cls = getattr(sb, SCHOLAR_BOTS[name])
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Spec parsing and registry
# ---------------------------------------------------------------------------
def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


@dataclass
class PlayerSpec:
    label: str
    name: str
    params: Dict[str, object] = field(default_factory=dict)
    raw: str = ""


def parse_spec(spec: str) -> PlayerSpec:
    """Parse ``[LABEL=]name[:k=v,...]``."""
    raw = spec.strip()
    label = None
    colon = raw.find(":")
    eq = raw.find("=")
    if eq != -1 and (colon == -1 or eq < colon):
        label, raw_body = raw[:eq].strip(), raw[eq + 1:].strip()
    else:
        raw_body = raw
    name, _, arg_str = raw_body.partition(":")
    name = name.strip().lower()
    params: Dict[str, object] = {}
    if arg_str:
        for part in arg_str.split(","):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            params[k.strip().lower()] = _coerce(v.strip())
    if label is None:
        label = _default_label(name, params)
    return PlayerSpec(label=label, name=name, params=params, raw=spec)


def _default_label(name: str, params: Dict[str, object]) -> str:
    pretty = {
        "random": "Random", "greedy": "Greedy-1ply", "minimax": "Minimax",
        "az": "AZ", "az-untrained": "AZ-untrained",
        "az-master": "AZ-Master", "az-intermediate": "AZ-Intermediate",
        "az-novice": "AZ-Novice",
        "novice": "NoviceBot", "chanakya": "ChanakyaBot",
        "aryabhata": "AryabhataBot", "panini": "PaniniBot",
    }.get(name, name)
    if name == "minimax":
        return f"Minimax-d{params.get('depth', 2)}"
    extras = []
    if "ckpt" in params:
        extras.append(os.path.splitext(os.path.basename(str(params["ckpt"])))[0])
    if name == "az":
        extras.append(f"s{params.get('sims', 80)}")
        extras.append(f"t{params.get('temp', 0.05)}")
        if params.get("blunder"):
            extras.append(f"b{params['blunder']}")
    return pretty + ("(" + ",".join(extras) + ")" if extras else "")


def _build_az(params: Dict[str, object], tier: Optional[str] = None, untrained: bool = False):
    base = TIERS[tier] if tier else Tier("custom", sims=80, temperature=0.05)
    return make_alphazero(
        ckpt=params.get("ckpt"),
        sims=int(params.get("sims", base.sims)),
        temperature=float(params.get("temp", base.temperature)),
        blunder_rate=float(params.get("blunder", base.blunder_rate)),
        top_k=params.get("top_k"),
        untrained=untrained,
    )


_REGISTRY: Dict[str, Callable[[Dict[str, object]], BasePlayer]] = {
    "random": lambda p: RandomPlayer(),
    "greedy": lambda p: GreedyPlayer(),
    "minimax": lambda p: MinimaxPlayer(depth=int(p.get("depth", 2))),
    "az": lambda p: _build_az(p),
    "az-untrained": lambda p: _build_az(p, untrained=True),
    "az-master": lambda p: _build_az(p, "master"),
    "az-intermediate": lambda p: _build_az(p, "intermediate"),
    "az-novice": lambda p: _build_az(p, "novice"),
}
for _key in SCHOLAR_BOTS:
    _REGISTRY[_key] = (lambda k: (lambda p: _make_scholar(k, **p)))(_key)


def available_players() -> List[str]:
    return sorted(_REGISTRY)


def build_player(spec: str) -> Tuple[str, BasePlayer]:
    """Return ``(label, player)`` for a spec string."""
    ps = parse_spec(spec)
    if ps.name not in _REGISTRY:
        raise ValueError(f"Unknown player '{ps.name}'. Known: {', '.join(available_players())}")
    return ps.label, _REGISTRY[ps.name](ps.params)
