"""
arena.tournament
----------------
Round-robin (or gauntlet) tournaments between any players, run in parallel.

Fairness:
    * Every pair plays an even number of games. Games come in pairs that share
      the same short random opening, once with each colour, so neither colour
      advantage nor a lucky opening favours either side.
    * Draw rules are identical to self-play (threefold repetition, ply cap).

Outputs (in --out, default arena_results/<name>/):
    games.jsonl        one line per game (moves included), used for --resume
    results.json       everything below, machine readable
    summary.md         Markdown tables ready for the report
    wld_matrix.csv     "W-L-D" of row player vs column player
    score_matrix.csv   row player's score rate vs column player
    length_matrix.csv  average game length (plies)
    elo.csv            Elo with 95% bootstrap CI (anchor: Random = 0)
    *.png              heatmap and Elo chart (via arena.plots)

Examples:
    python -m arena.tournament --games 20
    python -m arena.tournament --players random greedy minimax:depth=2 \\
        "Master=az-master:ckpt=checkpoints/gen_final.pt" --games 40 --workers 8
    python -m arena.tournament --progression checkpoints/gen_0.pt \\
        checkpoints/gen_mid.pt checkpoints/gen_final.pt --games 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .elo import EloTable, compute_elo
from .match import DEFAULT_MAX_PLIES, GameResult, play_game, random_opening
from .players import build_player, parse_spec, resolve_checkpoint, scholar_bots_available


# ---------------------------------------------------------------------------
# Scheduling and parallel execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameTask:
    pair: Tuple[int, int]
    k: int                    # game index within the pair
    white_spec: str
    black_spec: str
    white_label: str
    black_label: str
    seed: int
    opening: Tuple
    max_plies: int


def _stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little")


def schedule(specs: Sequence[str], labels: Sequence[str], pairs: Sequence[Tuple[int, int]],
             games_per_pair: int, seed: int, opening_plies: int, max_plies: int) -> List[GameTask]:
    if games_per_pair % 2:
        games_per_pair += 1  # keep colours balanced
    tasks = []
    for (i, j) in pairs:
        for k in range(games_per_pair):
            opening_seed = _stable_seed(seed, labels[i], labels[j], "opening", k // 2)
            opening = tuple(random_opening(opening_plies, random.Random(opening_seed)))
            if k % 2 == 0:
                w, b = i, j
            else:
                w, b = j, i
            tasks.append(GameTask(
                pair=(i, j), k=k,
                white_spec=specs[w], black_spec=specs[b],
                white_label=labels[w], black_label=labels[b],
                seed=_stable_seed(seed, labels[i], labels[j], "game", k),
                opening=opening, max_plies=max_plies,
            ))
    return tasks


_WORKER_PLAYERS: Dict[str, object] = {}


def _worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _get_player(spec: str):
    if spec not in _WORKER_PLAYERS:
        _WORKER_PLAYERS[spec] = build_player(spec)[1]
    return _WORKER_PLAYERS[spec]


def run_task(task: GameTask) -> dict:
    white = _get_player(task.white_spec)
    black = _get_player(task.black_spec)
    result = play_game(white, black, task.white_label, task.black_label,
                       max_plies=task.max_plies, opening=task.opening, seed=task.seed)
    d = result.to_dict()
    d["pair"] = list(task.pair)
    d["k"] = task.k
    return d


def execute(tasks: Sequence[GameTask], workers: int, log_path: Optional[str] = None,
            done: Optional[Dict[Tuple, dict]] = None, verbose: bool = True) -> List[dict]:
    """Run tasks (in parallel if workers > 1), appending each game to log_path."""
    done = dict(done or {})
    todo = [t for t in tasks if (tuple(t.pair), t.k) not in done]
    results = [done[(tuple(t.pair), t.k)] for t in tasks if (tuple(t.pair), t.k) in done]
    if verbose and done:
        print(f"Resuming: {len(results)} games already played, {len(todo)} to go.")
    log = open(log_path, "a") if log_path else None
    start = time.time()

    def record(d: dict, n: int) -> None:
        results.append(d)
        if log:
            log.write(json.dumps(d) + "\n")
            log.flush()
        if verbose:
            res = {1: "1-0", -1: "0-1", 0: "1/2"}[d["winner"]]
            eta = (time.time() - start) / n * (len(todo) - n)
            print(f"[{n:>4}/{len(todo)}] {d['white']:>22} vs {d['black']:<22} {res:>4} "
                  f"{d['reason']:<12} {d['plies']:>3} plies   eta {eta/60:5.1f} min", flush=True)

    try:
        if workers <= 1:
            _worker_init()
            for n, t in enumerate(todo, 1):
                record(run_task(t), n)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_worker_init) as pool:
                futures = [pool.submit(run_task, t) for t in todo]
                for n, fut in enumerate(as_completed(futures), 1):
                    record(fut.result(), n)
    finally:
        if log:
            log.close()
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class TournamentResult:
    labels: List[str]
    specs: List[str]
    games: List[dict]
    elo: EloTable
    wins: np.ndarray        # wins[i, j]: games i won against j
    draws: np.ndarray
    plies: np.ndarray       # summed plies between i and j
    count: np.ndarray       # games between i and j
    reasons: Dict[str, Counter] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def losses(self) -> np.ndarray:
        return self.wins.T

    def score_rate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, (self.wins + 0.5 * self.draws) / self.count, np.nan)

    def avg_length(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.plies / self.count, np.nan)

    def player_rows(self) -> List[dict]:
        rows = []
        for i, p in enumerate(self.labels):
            w, l, d = self.wins[i].sum(), self.wins[:, i].sum(), self.draws[i].sum()
            n = w + l + d
            rows.append({
                "player": p, "games": int(n), "wins": int(w), "losses": int(l), "draws": int(d),
                "score": float((w + 0.5 * d) / n) if n else float("nan"),
                "avg_plies": float(self.plies[i].sum() / n) if n else float("nan"),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "players": [{"label": l, "spec": s} for l, s in zip(self.labels, self.specs)],
            "elo": self.elo.as_rows(),
            "elo_anchor": {"player": self.elo.anchor, "elo": self.elo.anchor_elo},
            "standings": self.player_rows(),
            "wins": self.wins.tolist(),
            "draws": self.draws.tolist(),
            "count": self.count.tolist(),
            "score_rate": _nan_to_none(self.score_rate()),
            "avg_length": _nan_to_none(self.avg_length()),
            "end_reasons": {p: dict(c) for p, c in self.reasons.items()},
            "num_games": len(self.games),
        }


def _nan_to_none(a: np.ndarray):
    return [[None if np.isnan(x) else float(x) for x in row] for row in a]


def aggregate(labels: Sequence[str], specs: Sequence[str], games: Sequence[dict],
              anchor: Optional[str] = None, anchor_elo: float = 0.0,
              bootstrap: int = 500, config: Optional[dict] = None) -> TournamentResult:
    labels = list(labels)
    idx = {p: k for k, p in enumerate(labels)}
    n = len(labels)
    wins, draws, plies, count = (np.zeros((n, n)) for _ in range(4))
    reasons: Dict[str, Counter] = defaultdict(Counter)
    triples = []
    for g in games:
        w, b = idx[g["white"]], idx[g["black"]]
        count[w, b] += 1
        count[b, w] += 1
        plies[w, b] += g["plies"]
        plies[b, w] += g["plies"]
        if g["winner"] == 1:
            wins[w, b] += 1
            reasons[g["white"]]["won_" + g["reason"]] += 1
            reasons[g["black"]]["lost_" + g["reason"]] += 1
        elif g["winner"] == -1:
            wins[b, w] += 1
            reasons[g["black"]]["won_" + g["reason"]] += 1
            reasons[g["white"]]["lost_" + g["reason"]] += 1
        else:
            draws[w, b] += 1
            draws[b, w] += 1
            reasons[g["white"]]["draw_" + g["reason"]] += 1
            reasons[g["black"]]["draw_" + g["reason"]] += 1
        score_white = {1: 1.0, -1: 0.0, 0: 0.5}[g["winner"]]
        triples.append((g["white"], g["black"], score_white))
    if anchor is None:
        anchor = "Random" if "Random" in labels else None
    elo = compute_elo(labels, triples, anchor=anchor, anchor_elo=anchor_elo, bootstrap=bootstrap)
    return TournamentResult(labels, list(specs), list(games), elo, wins, draws, plies, count,
                            dict(reasons), dict(config or {}))


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------
def _write_matrix(path: str, labels: Sequence[str], fn) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(labels))
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + [fn(i, j) for j in range(len(labels))])


def markdown_summary(res: TournamentResult, title: str = "Tournament") -> str:
    L = res.labels
    out = [f"# {title}", ""]
    cfg = res.config
    if cfg:
        out.append(f"{len(res.games)} games, {cfg.get('games_per_pair')} per pair, colours alternated, "
                   f"{cfg.get('opening_plies')} random opening plies shared by each colour-swapped pair, "
                   f"draw at threefold repetition or {cfg.get('max_plies')} plies.")
        out.append("")
    out += ["## Elo (anchor: %s = %d, 95%% bootstrap CI)" % (res.elo.anchor, res.elo.anchor_elo), "",
            "| Rank | Player | Elo | 95% CI | Games |", "|---:|---|---:|---|---:|"]
    for r, row in enumerate(res.elo.as_rows(), 1):
        out.append(f"| {r} | {row['player']} | {row['elo']:.0f} | "
                   f"[{row['ci_low']:.0f}, {row['ci_high']:.0f}] | {row['games']} |")
    out += ["", "## Standings", "", "| Player | W | L | D | Score | Avg plies |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(res.player_rows(), key=lambda r: -r["score"]):
        out.append(f"| {row['player']} | {row['wins']} | {row['losses']} | {row['draws']} | "
                   f"{row['score']:.1%} | {row['avg_plies']:.1f} |")
    out += ["", "## Head to head (row W-L-D vs column)", "",
            "| | " + " | ".join(L) + " |", "|---|" + "---|" * len(L)]
    for i, a in enumerate(L):
        cells = []
        for j in range(len(L)):
            if i == j or res.count[i, j] == 0:
                cells.append("–")
            else:
                cells.append(f"{int(res.wins[i, j])}-{int(res.wins[j, i])}-{int(res.draws[i, j])}")
        out.append(f"| **{a}** | " + " | ".join(cells) + " |")
    out += ["", "## How games ended (from each player's side)", ""]
    keys = sorted({k for c in res.reasons.values() for k in c})
    out += ["| Player | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for p in L:
        c = res.reasons.get(p, Counter())
        out.append(f"| {p} | " + " | ".join(str(c.get(k, 0)) for k in keys) + " |")
    errors = [g for g in res.games if g.get("error")]
    if errors:
        out += ["", f"## Errors ({len(errors)})", ""]
        for g in errors[:10]:
            out.append(f"- {g['white']} vs {g['black']} ({g['reason']}): `{g['error'].splitlines()[-1]}`")
    return "\n".join(out) + "\n"


def save_results(res: TournamentResult, out_dir: str, title: str = "Tournament",
                 make_plots: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    L = res.labels
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res.to_dict(), f, indent=2)
    _write_matrix(os.path.join(out_dir, "wld_matrix.csv"), L,
                  lambda i, j: "" if i == j or res.count[i, j] == 0 else
                  f"{int(res.wins[i, j])}-{int(res.wins[j, i])}-{int(res.draws[i, j])}")
    sr, al = res.score_rate(), res.avg_length()
    _write_matrix(os.path.join(out_dir, "score_matrix.csv"), L,
                  lambda i, j: "" if np.isnan(sr[i, j]) or i == j else f"{sr[i, j]:.3f}")
    _write_matrix(os.path.join(out_dir, "length_matrix.csv"), L,
                  lambda i, j: "" if np.isnan(al[i, j]) or i == j else f"{al[i, j]:.1f}")
    with open(os.path.join(out_dir, "elo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["player", "elo", "ci_low", "ci_high", "games"])
        w.writeheader()
        for row in res.elo.as_rows():
            w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(markdown_summary(res, title))
    if make_plots:
        try:
            from . import plots

            plots.plot_winrate_heatmap(res, os.path.join(out_dir, "winrate_heatmap.png"))
            plots.plot_elo_bars(res.elo, os.path.join(out_dir, "elo_ratings.png"))
        except Exception as exc:  # plotting must never lose tournament data
            print(f"Warning: plotting failed: {exc}", file=sys.stderr)


def load_games(path: str) -> Dict[Tuple, dict]:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done[(tuple(d["pair"]), d["k"])] = d
    return done


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def run_tournament(specs: Sequence[str], games_per_pair: int = 10, fmt: str = "round-robin",
                   hero: Optional[Sequence[str]] = None, max_plies: int = DEFAULT_MAX_PLIES,
                   opening_plies: int = 2, workers: int = 1, seed: int = 0,
                   out_dir: Optional[str] = None, resume: bool = True,
                   anchor: Optional[str] = None, anchor_elo: float = 0.0,
                   bootstrap: int = 500, title: str = "Tournament",
                   make_plots: bool = True, verbose: bool = True) -> TournamentResult:
    """
    Play a tournament and (optionally) write all outputs to ``out_dir``.

    fmt: "round-robin" (every pair) or "gauntlet" (each hero spec vs every
    non-hero spec; heroes do not play each other).
    """
    specs = list(specs)
    labels = [parse_spec(s).label for s in specs]
    dupes = [l for l, c in Counter(labels).items() if c > 1]
    if dupes:
        raise ValueError(f"Duplicate player labels {dupes}; name them with LABEL=spec.")
    for s in specs:  # fail fast on bad specs before spawning workers
        build_player(s)

    n = len(specs)
    if fmt == "round-robin":
        pairs = list(itertools.combinations(range(n), 2))
    elif fmt == "gauntlet":
        hero_set = {labels.index(parse_spec(h).label) for h in (hero or [])}
        if not hero_set:
            raise ValueError("gauntlet needs at least one hero")
        pairs = [(h, j) for h in sorted(hero_set) for j in range(n) if j not in hero_set]
    else:
        raise ValueError(f"unknown format {fmt}")

    config = dict(format=fmt, games_per_pair=games_per_pair + games_per_pair % 2,
                  max_plies=max_plies, opening_plies=opening_plies, seed=seed)
    tasks = schedule(specs, labels, pairs, games_per_pair, seed, opening_plies, max_plies)

    log_path = done = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "games.jsonl")
        cfg_path = os.path.join(out_dir, "config.json")
        current = {**config, "players": dict(zip(labels, specs))}
        if resume and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                previous = json.load(f)
            if previous != current:
                if verbose:
                    print("Config changed since the last run in this folder; starting fresh.")
                resume = False
        if not resume and os.path.exists(log_path):
            os.remove(log_path)
        with open(cfg_path, "w") as f:
            json.dump(current, f, indent=2)
        done = load_games(log_path) if resume else None
        if done:  # drop games whose labels no longer match the schedule
            valid = {(tuple(t.pair), t.k): (t.white_label, t.black_label) for t in tasks}
            done = {k: v for k, v in done.items() if valid.get(k) == (v["white"], v["black"])}

    games = execute(tasks, workers, log_path, done, verbose)
    res = aggregate(labels, specs, games, anchor=anchor, anchor_elo=anchor_elo,
                    bootstrap=bootstrap, config=config)
    if out_dir:
        save_results(res, out_dir, title, make_plots)
    return res


def default_players(ckpt: Optional[str]) -> List[str]:
    specs = ["random", "greedy", "minimax:depth=2"]
    if scholar_bots_available():
        specs += ["novice", "aryabhata", "chanakya", "panini"]
    else:
        print("Note: bots/scholar_bots.py not found; scholar bots skipped.")
    try:
        path = resolve_checkpoint(ckpt)
        specs += [f"AZ-Master=az-master:ckpt={path}",
                  f"AZ-Intermediate=az-intermediate:ckpt={path}",
                  f"AZ-Novice=az-novice:ckpt={path}"]
    except FileNotFoundError:
        print("Note: no trained checkpoint found; AlphaZero tiers skipped (use --checkpoint).")
    return specs


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="ChaturangZero round-robin tournament",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--players", nargs="+", help="player specs (see arena/players.py)")
    p.add_argument("--checkpoint", help="checkpoint for the default AlphaZero tiers")
    p.add_argument("--progression", nargs="+", metavar="[LABEL=]CKPT",
                   help="checkpoints to compare (one AlphaZero player each); "
                        "'untrained' means a fresh network")
    p.add_argument("--progression-sims", type=int, default=25,
                   help="MCTS simulations for --progression players (default 25)")
    p.add_argument("--games", type=int, default=20, help="games per pair (rounded up to even)")
    p.add_argument("--format", choices=["round-robin", "gauntlet"], default="round-robin")
    p.add_argument("--hero", nargs="+", help="hero specs for --format gauntlet")
    p.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    p.add_argument("--opening-plies", type=int, default=2,
                   help="random opening half-moves shared by each colour-swapped pair")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor", help="player fixed at --anchor-elo (default: Random)")
    p.add_argument("--anchor-elo", type=float, default=0.0)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--name", default="round_robin")
    p.add_argument("--out", help="output folder (default arena_results/<name>)")
    p.add_argument("--fresh", action="store_true", help="ignore previously played games")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    specs = list(args.players) if args.players else []
    if args.progression:
        if not specs:
            specs = ["random", "greedy", "minimax:depth=2"]
        for item in args.progression:
            label, _, ck = item.rpartition("=")
            if ck == "untrained":  # a freshly initialised network (seed 0) stands in for gen_0
                specs.append(f"{label or 'untrained'}=az-untrained:sims={args.progression_sims},temp=0.05")
                continue
            label = label or os.path.splitext(os.path.basename(ck))[0]
            specs.append(f"{label}=az:ckpt={ck},sims={args.progression_sims},temp=0.05")
    if not specs:
        specs = default_players(args.checkpoint)

    out = args.out or os.path.join("arena_results", args.name)
    res = run_tournament(specs, games_per_pair=args.games, fmt=args.format, hero=args.hero,
                         max_plies=args.max_plies, opening_plies=args.opening_plies,
                         workers=args.workers, seed=args.seed, out_dir=out,
                         resume=not args.fresh, anchor=args.anchor, anchor_elo=args.anchor_elo,
                         bootstrap=args.bootstrap, title=f"Tournament: {args.name}",
                         make_plots=not args.no_plots)
    print()
    print(markdown_summary(res, f"Tournament: {args.name}"))
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
