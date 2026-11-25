"""
arena.plots
-----------
Publication figures for the report and presentation (static PNG + PDF-ready).

    python -m arena.plots --training-log checkpoints/training_log.json \\
        --tournament arena_results/round_robin/results.json --out figures/

Figures:
    training_losses.png     total / value / policy loss per iteration
    training_games.png      average self-play game length and how games ended
    winrate_heatmap.png     head-to-head score rate (row vs column)
    elo_ratings.png         Elo with 95% bootstrap CI
    elo_progression.png     Elo of successive checkpoints
    tier_calibration.png    measured Elo of each skill tier vs its target
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Reference palette (validated categorical order; light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL_MID = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "score", ["#e34948", "#f19c9b", NEUTRAL_MID, "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def _direct_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=2, zorder=5)


# ---------------------------------------------------------------------------
# Training curves (from training/train.py's training_log.json)
# ---------------------------------------------------------------------------
def load_training_log(path: str) -> List[dict]:
    with open(path) as f:
        log = json.load(f)
    return [r for r in log if r.get("total_loss", 0) or r.get("pol_loss", 0)] or log


def has_game_stats(log: Sequence[dict]) -> bool:
    """Self-play logs record game lengths and outcomes; teacher fine-tuning logs do not."""
    return any("outcomes" in r or "avg_game_length" in r for r in log)


def plot_training_losses(log: Sequence[dict], path: str, smooth: int = 3,
                         title: str = "Self-play training loss") -> str:
    it = np.array([r["iteration"] for r in log])
    series = [("Total loss", "total_loss"), ("Policy loss (cross-entropy)", "pol_loss"),
              ("Value loss (MSE)", "val_loss")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, (panel, key), color in zip(axes, series, SERIES):
        y = np.array([r[key] for r in log], dtype=float)
        ax.plot(it, y, color=color, alpha=0.25, lw=1.5)
        ys = _smooth(y, smooth)
        ax.plot(it, ys, color=color)
        _direct_label(ax, it[-1], ys[-1], f"{ys[-1]:.2f}", color)
        ax.set_title(panel)
        ax.set_xlabel("Training iteration")
        ax.margins(x=0.08)
    axes[0].set_ylabel("Loss")
    fig.suptitle("%s (faint: per iteration, solid: %d-iteration mean)" % (title, smooth),
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_games(log: Sequence[dict], path: str) -> str:
    it = np.array([r["iteration"] for r in log])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    length = np.array([r.get("avg_game_length", np.nan) for r in log], dtype=float)
    a1.plot(it, length, color=SERIES[0])
    a1.set_title("Average self-play game length")
    a1.set_xlabel("Training iteration")
    a1.set_ylabel("Plies")
    a1.set_ylim(bottom=0)

    keys = [("white", "White win"), ("black", "Black win"),
            ("repetition", "Draw: repetition"), ("max_turns", "Draw: ply cap")]
    counts = np.array([[r.get("outcomes", {}).get(k, 0) for k, _ in keys] for r in log], dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    bottom = np.zeros(len(it))
    for (k, name), color, col in zip(keys, SERIES, share.T):
        a2.bar(it, col, bottom=bottom, color=color, width=0.85, label=name,
               edgecolor=SURFACE, linewidth=1)
        bottom += col
    a2.set_title("How self-play games ended")
    a2.set_xlabel("Training iteration")
    a2.set_ylabel("Share of games")
    a2.set_ylim(0, 1)
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _save(fig, path)


def plot_value_probe(logs: Dict[str, Sequence[dict]], path: str) -> str:
    """The value head on three fixed positions over training, one panel per run.

    train.py's probe: the start position, White up a Chariot, White down a
    Chariot. A value head that has learned material separates the three lines.
    """
    logs = {k: [r for r in v if r.get("probe")] for k, v in logs.items()}
    logs = {k: v for k, v in logs.items() if v}
    keys = [("up_chariot", "White up a Chariot"), ("start", "Start position"),
            ("down_chariot", "White down a Chariot")]
    colors = [SERIES[2], INK_2, SERIES[1]]
    fig, axes = plt.subplots(1, len(logs), figsize=(5.2 * len(logs) + 1.6, 3.8), sharey=True, squeeze=False)
    for ax, (name, log) in zip(axes[0], logs.items()):
        it = np.array([r["iteration"] for r in log])
        for (k, label), color in zip(keys, colors):
            y = np.array([r["probe"][k] for r in log], dtype=float)
            ax.plot(it, y, color=color, marker="o", markersize=5, label=label)
        spread = log[-1]["probe"]["up_chariot"] - log[-1]["probe"]["down_chariot"]
        ax.set_title(f"{name}\nfinal spread (up - down): {spread:+.2f}")
        ax.set_xlabel("Training iteration")
        ax.axhline(0, color=INK_2, lw=0.8, ls="--", zorder=0)
        ax.margins(x=0.08)
    axes[0][0].set_ylabel("Value head output (White to move)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Tournament figures
# ---------------------------------------------------------------------------
def _result_arrays(res):
    """Accept a TournamentResult or a loaded results.json dict."""
    if isinstance(res, dict):
        labels = [p["label"] for p in res["players"]]
        sr = np.array([[np.nan if v is None else v for v in row] for row in res["score_rate"]])
        count = np.array(res["count"])
        elo = {r["player"]: r for r in res["elo"]}
        return labels, sr, count, elo
    labels = res.labels
    elo = {r["player"]: r for r in res.elo.as_rows()}
    return labels, res.score_rate(), res.count, elo


def plot_winrate_heatmap(res, path: str) -> str:
    labels, sr, count, elo = _result_arrays(res)
    order = sorted(range(len(labels)), key=lambda i: -elo[labels[i]]["elo"])
    L = [labels[i] for i in order]
    M = sr[np.ix_(order, order)]
    C = count[np.ix_(order, order)]
    n = len(L)
    size = max(4.5, 0.75 * n + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    ax.grid(False)
    shown = np.where(C > 0, M, np.nan)
    im = ax.imshow(shown, cmap=DIVERGING, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#e4e3df", lw=0))
            elif C[i, j] > 0:
                v = M[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color=SURFACE if abs(v - 0.5) > 0.3 else INK)
    ax.set_xticks(range(n), L, rotation=35, ha="right")
    ax.set_yticks(range(n), L)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Player")
    ax.set_title("Head-to-head score rate (wins + ½ draws), sorted by Elo")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    fig.tight_layout()
    return _save(fig, path)


def plot_elo_bars(elo_table, path: str, title: str = "Elo rating with 95% bootstrap CI",
                  highlight: Optional[Sequence[str]] = None) -> str:
    rows = elo_table.as_rows() if hasattr(elo_table, "as_rows") else list(elo_table)
    rows = sorted(rows, key=lambda r: r["elo"])
    names = [r["player"] for r in rows]
    e = np.array([r["elo"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(rows) + 1)))
    colors = [SERIES[1] if highlight and n in highlight else SERIES[0] for n in names]
    y = np.arange(len(rows))
    ax.hlines(y, lo, hi, color=INK_2, lw=1.5, zorder=2)
    ax.plot(e, y, "o", ms=9, color="none", zorder=3)
    for yi, ei, c in zip(y, e, colors):
        ax.plot([ei], [yi], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=4)
    for yi, ei, h in zip(y, e, hi):
        ax.annotate(f"{ei:.0f}", (h, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    anchor = getattr(elo_table, "anchor", None)
    if anchor is not None:
        ax.axvline(getattr(elo_table, "anchor_elo", 0), color=GRID, lw=1, zorder=1)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Elo" + (f" (relative: {anchor} = {getattr(elo_table, 'anchor_elo', 0):.0f})" if anchor else ""))
    ax.set_title(title)
    ax.margins(x=0.12)
    fig.tight_layout()
    return _save(fig, path)


def plot_elo_progression(points: Sequence[dict], path: str,
                         baselines: Optional[Sequence[dict]] = None,
                         title: str = "Elo of successive checkpoints") -> str:
    """points: [{label, elo, ci_low, ci_high}] in training order. baselines: same shape."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(points))
    e = np.array([p["elo"] for p in points])
    lo = np.array([p["ci_low"] for p in points])
    hi = np.array([p["ci_high"] for p in points])
    ax.fill_between(x, lo, hi, color=SERIES[0], alpha=0.15, lw=0)
    ax.plot(x, e, color=SERIES[0], marker="o", ms=8, mec=SURFACE, mew=2)
    for xi, ei in zip(x, e):
        ax.annotate(f"{ei:.0f}", (xi, ei), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2)
    for b in baselines or []:
        ax.axhline(b["elo"], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"{b['player']} ({b['elo']:.0f})", (0.0, b["elo"]),
                    xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points",
                    va="bottom", ha="left", fontsize=9, color=INK_2)
    ax.set_xticks(x, [p["label"] for p in points])
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Elo")
    ax.set_title(title)
    ax.margins(x=0.15)
    fig.tight_layout()
    return _save(fig, path)


def plot_tier_calibration(rows: Sequence[dict], path: str) -> str:
    """rows: [{tier, target, elo, ci_low, ci_high}]."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(rows))
    tgt = np.array([r["target"] for r in rows], dtype=float)
    e = np.array([r["elo"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    ax.vlines(x + 0.12, lo, hi, color=INK_2, lw=1.5)
    ax.plot(x - 0.12, tgt, "D", ms=8, color=SERIES[1], mec=SURFACE, mew=2, label="Target Elo")
    ax.plot(x + 0.12, e, "o", ms=9, color=SERIES[0], mec=SURFACE, mew=2, label="Measured Elo (95% CI)")
    for xi, ei in zip(x, e):
        ax.annotate(f"{ei:.0f}", (xi + 0.12, ei), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, [r["tier"] for r in rows])
    ax.set_ylabel("Elo (relative to Random = 0)")
    ax.set_title("Skill tiers of one network: target vs measured")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.margins(x=0.2)
    fig.tight_layout()
    return _save(fig, path)


def plot_bars(categories: Sequence[str], values: Sequence[float], path: str, title: str,
              ylabel: str, fmt: str = "{:.0%}", ylim: Optional[tuple] = None,
              errors: Optional[Sequence[float]] = None) -> str:
    """Single-series bar chart used by the ablation experiments."""
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(categories) + 1.5), 3.6))
    x = np.arange(len(categories))
    ax.bar(x, values, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
           yerr=errors, ecolor=INK_2, capsize=4 if errors is not None else 0)
    for xi, v in zip(x, values):
        ax.annotate(fmt.format(v), (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if "%" in fmt:
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    return _save(fig, path)


def plot_lines(x, series: Dict[str, Sequence[float]], path: str, title: str, xlabel: str,
               ylabel: str, fmt: str = "{:.2f}", logx: bool = False,
               bands: Optional[Dict[str, tuple]] = None,
               refs: Optional[Dict[str, float]] = None,
               ylim: Optional[tuple] = None) -> str:
    """
    Multi-series line chart with a legend and direct end labels.
    ``refs`` are constant reference levels (drawn dashed, labelled at the right).
    """
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = iter(SERIES)
    labels = []  # (y, text) for the right-hand labels
    for name, y in series.items():
        color = next(colors)
        y = np.asarray(y, dtype=float)
        if bands and name in bands:
            ax.fill_between(x, bands[name][0], bands[name][1], color=color, alpha=0.15, lw=0)
        ax.plot(x, y, color=color, label=name, marker="o", ms=5, mec=SURFACE, mew=1.5)
        labels.append((y[-1], fmt.format(y[-1])))
    for name, level in (refs or {}).items():
        color = next(colors)
        ax.axhline(level, color=color, lw=2, ls=(0, (5, 3)), label=name)
        labels.append((level, fmt.format(level)))
    if logx:
        from matplotlib.ticker import NullFormatter

        ax.set_xscale("log")
        ax.set_xticks(list(x), [str(v) for v in x])
        ax.xaxis.set_minor_formatter(NullFormatter())
    if "%" in fmt:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.margins(x=0.1)
    if ylim:
        ax.set_ylim(*ylim)
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)
    placed = []
    for yv, text in sorted(labels):
        ypos = yv if not placed or yv - placed[-1] >= gap else placed[-1] + gap
        placed.append(ypos)
        ax.annotate(text, (1.0, ypos), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2,
                    annotation_clip=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    n_items = len(series) + len(refs or {})
    if n_items > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(n_items, 3))
    fig.tight_layout()
    return _save(fig, path)


def plot_histograms(data: Dict[str, Sequence[float]], path: str, title: str, xlabel: str,
                    bins=30) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for (name, vals), color in zip(data.items(), SERIES):
        ax.hist(vals, bins=bins, color=color, alpha=0.55, label=name,
                edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate ChaturangZero report figures")
    p.add_argument("--training-log", nargs="*", default=[],
                   help="training_log.json files (train.py or bootstrap_train.py); NAME=PATH sets a label")
    p.add_argument("--tournament", nargs="*", default=[], help="results.json from arena.tournament")
    p.add_argument("--progression", help="results.json of a --progression tournament")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)

    made = []
    probe_logs = {}
    for i, item in enumerate(args.training_log):
        name, _, path = item.rpartition("=")
        name = name or os.path.splitext(os.path.basename(path))[0]
        log = load_training_log(path)
        suffix = "" if len(args.training_log) == 1 or i == 0 else "_" + name.lower().replace(" ", "_")
        made.append(plot_training_losses(log, os.path.join(args.out, f"training_losses{suffix}.png"),
                                         args.smooth, title=f"{name}: training loss"))
        if has_game_stats(log):
            made.append(plot_training_games(log, os.path.join(args.out, f"training_games{suffix}.png")))
        probe_logs[name] = log
    if any(r.get("probe") for log in probe_logs.values() for r in log):
        made.append(plot_value_probe(probe_logs, os.path.join(args.out, "value_probe.png")))
    for i, path in enumerate(args.tournament):
        with open(path) as f:
            res = json.load(f)
        suffix = "" if len(args.tournament) == 1 else f"_{i}"
        made.append(plot_winrate_heatmap(res, os.path.join(args.out, f"winrate_heatmap{suffix}.png")))
        made.append(plot_elo_bars(res["elo"], os.path.join(args.out, f"elo_ratings{suffix}.png")))
    if args.progression:
        with open(args.progression) as f:
            res = json.load(f)
        elo = {r["player"]: r for r in res["elo"]}
        from .players import parse_spec

        ckpts = [pl["label"] for pl in res["players"] if parse_spec(pl["spec"]).name.startswith("az")]
        base = [elo[pl["label"]] for pl in res["players"] if pl["label"] not in ckpts]
        pts = [dict(elo[c], label=c) for c in ckpts]
        made.append(plot_elo_progression(pts, os.path.join(args.out, "elo_progression.png"), base))
    for m in made:
        print("wrote", m)
    if not made:
        p.print_help()


if __name__ == "__main__":
    main()
