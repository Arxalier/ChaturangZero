"""
arena.elo_tuning
----------------
Turns ONE trained network into several skill tiers by changing only
inference-time settings (MCTS simulations, move temperature, blunder rate),
then *measures* each tier's Elo against fixed baseline players.

Why measure instead of assume: Elo is relative. A label like "1800" means
nothing until it is tied to an anchor. Here every rating is relative to the
uniformly random player (Random = 0 by default, change with --anchor-elo), and
targets are compared against measured ratings with 95% confidence intervals.

Modes:
    measure    (default) play the three tiers from the project breakdown
               (Master 80 sims / tau 0.05, Intermediate 15 / 0.60,
               Novice raw policy + 15% blunders) against the baseline pool.
    calibrate  play a grid of candidate settings against the same pool and
               pick, for each target, the setting whose measured Elo is
               closest. Use this when the fixed tiers miss their targets.

    python -m arena.elo_tuning --checkpoint checkpoints/gen_final.pt --games 20
    python -m arena.elo_tuning --checkpoint checkpoints/gen_final.pt --mode calibrate
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from .match import DEFAULT_MAX_PLIES
from .players import TIERS, parse_spec, resolve_checkpoint, scholar_bots_available
from .tournament import aggregate, run_tournament, save_results

# Settings explored by --mode calibrate: (label, sims, temperature, blunder_rate)
CANDIDATES = [
    ("raw-b50", 0, 1.0, 0.50),
    ("raw-b30", 0, 1.0, 0.30),
    ("raw-b15", 0, 1.0, 0.15),
    ("raw-t1", 0, 1.0, 0.0),
    ("raw-greedy", 0, 0.01, 0.0),
    ("s5-t1", 5, 1.0, 0.0),
    ("s15-t0.6", 15, 0.6, 0.0),
    ("s15-t0.05", 15, 0.05, 0.0),
    ("s40-t0.05", 40, 0.05, 0.0),
    ("s80-t0.05", 80, 0.05, 0.0),
    ("s160-t0.05", 160, 0.05, 0.0),
]


def anchor_pool(include_scholars: bool = True) -> List[str]:
    pool = ["random", "greedy", "minimax:depth=2"]
    if include_scholars and scholar_bots_available():
        pool += ["novice", "aryabhata", "chanakya", "panini"]
    return pool


def tier_specs(ckpt: str) -> List[str]:
    return [f"AZ-{t.name}=az-{key}:ckpt={ckpt}" for key, t in TIERS.items()]


def candidate_specs(ckpt: str, candidates=CANDIDATES) -> List[str]:
    return [f"AZ[{lab}]=az:ckpt={ckpt},sims={s},temp={t},blunder={b}"
            for lab, s, t, b in candidates]


def _tier_rows(res, specs: Sequence[str]) -> List[dict]:
    rows = []
    elo = {r["player"]: r for r in res.elo.as_rows()}
    for key, tier in TIERS.items():
        label = f"AZ-{tier.name}"
        if label not in elo:
            continue
        r = elo[label]
        rows.append({
            "tier": tier.name, "target": tier.target_elo, "elo": r["elo"],
            "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "gap": r["elo"] - tier.target_elo,
            "within_ci": r["ci_low"] <= tier.target_elo <= r["ci_high"],
            "settings": f"sims={tier.sims}, tau={tier.temperature}, blunder={tier.blunder_rate}",
        })
    return rows


def measure(ckpt: str, games: int, workers: int, out: str, max_plies: int, seed: int,
            anchor_elo: float, include_scholars: bool) -> dict:
    specs = anchor_pool(include_scholars) + tier_specs(ckpt)
    res = run_tournament(specs, games_per_pair=games, workers=workers, out_dir=out,
                         max_plies=max_plies, seed=seed, anchor_elo=anchor_elo,
                         title="Elo tiers of one network")
    rows = _tier_rows(res, specs)
    return {"mode": "measure", "checkpoint": ckpt, "anchor": res.elo.anchor,
            "anchor_elo": anchor_elo, "tiers": rows, "elo": res.elo.as_rows()}


def calibrate(ckpt: str, games: int, workers: int, out: str, max_plies: int, seed: int,
              anchor_elo: float, include_scholars: bool) -> dict:
    pool = anchor_pool(include_scholars)
    cands = candidate_specs(ckpt)
    rr = run_tournament(pool, games_per_pair=games, workers=workers,
                        out_dir=os.path.join(out, "anchors"), max_plies=max_plies, seed=seed,
                        anchor_elo=anchor_elo, title="Anchor pool", make_plots=False)
    gl = run_tournament(pool + cands, games_per_pair=games, fmt="gauntlet", hero=cands,
                        workers=workers, out_dir=os.path.join(out, "candidates"),
                        max_plies=max_plies, seed=seed + 1, anchor_elo=anchor_elo,
                        title="Candidate settings vs anchor pool", make_plots=False)
    specs = pool + cands
    labels = [parse_spec(s).label for s in specs]
    res = aggregate(labels, specs, rr.games + gl.games, anchor_elo=anchor_elo,
                    config={**gl.config, "format": "anchors round-robin + candidate gauntlet"})
    save_results(res, out, "Elo calibration of inference settings")

    elo = {r["player"]: r for r in res.elo.as_rows()}
    cand_rows = []
    for (lab, s, t, b), label in zip(CANDIDATES, labels[len(pool):]):
        r = elo[label]
        cand_rows.append({"candidate": lab, "sims": s, "temperature": t, "blunder_rate": b,
                          "elo": r["elo"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    picks = []
    for key, tier in TIERS.items():
        best = min(cand_rows, key=lambda c: abs(c["elo"] - tier.target_elo))
        picks.append({"tier": tier.name, "target": tier.target_elo, **best,
                      "gap": best["elo"] - tier.target_elo,
                      "within_ci": best["ci_low"] <= tier.target_elo <= best["ci_high"]})
    return {"mode": "calibrate", "checkpoint": ckpt, "anchor": res.elo.anchor,
            "anchor_elo": anchor_elo, "candidates": cand_rows, "tiers": picks,
            "elo": res.elo.as_rows()}


def report_markdown(report: dict) -> str:
    out = [f"# Elo tiers ({report['mode']})", "",
           f"Checkpoint: `{report['checkpoint']}`. Ratings are relative to "
           f"{report['anchor']} = {report['anchor_elo']:.0f}.", ""]
    if report["mode"] == "calibrate":
        out += ["## Candidate settings", "",
                "| Setting | Sims | Temp | Blunder | Elo | 95% CI |", "|---|---:|---:|---:|---:|---|"]
        for c in sorted(report["candidates"], key=lambda c: c["elo"]):
            out.append(f"| {c['candidate']} | {c['sims']} | {c['temperature']} | {c['blunder_rate']} | "
                       f"{c['elo']:.0f} | [{c['ci_low']:.0f}, {c['ci_high']:.0f}] |")
        out.append("")
    out += ["## Tiers: target vs measured", "",
            "| Tier | Target | Measured | 95% CI | Gap | Target inside CI? | Settings |",
            "|---|---:|---:|---|---:|---|---|"]
    for t in report["tiers"]:
        settings = t.get("settings") or f"sims={t['sims']}, tau={t['temperature']}, blunder={t['blunder_rate']}"
        out.append(f"| {t['tier']} | {t['target']:.0f} | {t['elo']:.0f} | "
                   f"[{t['ci_low']:.0f}, {t['ci_high']:.0f}] | {t['gap']:+.0f} | "
                   f"{'yes' if t['within_ci'] else 'no'} | {settings} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="trained model (default: checkpoints/gen_final.pt)")
    p.add_argument("--mode", choices=["measure", "calibrate"], default="measure")
    p.add_argument("--games", type=int, default=20, help="games per pair")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor-elo", type=float, default=0.0, help="rating given to Random")
    p.add_argument("--no-scholars", action="store_true", help="leave Person 3's bots out of the pool")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    ckpt = resolve_checkpoint(args.checkpoint)
    out = args.out or os.path.join("arena_results", f"elo_{args.mode}")
    fn = measure if args.mode == "measure" else calibrate
    report = fn(ckpt, args.games, args.workers, out, args.max_plies, args.seed,
                args.anchor_elo, not args.no_scholars)

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tiers.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = report_markdown(report)
    with open(os.path.join(out, "tiers.md"), "w") as f:
        f.write(md)
    from . import plots

    plots.plot_tier_calibration(report["tiers"], os.path.join(out, "tier_calibration.png"))
    print(md)
    print(f"Written to {out}/")


if __name__ == "__main__":
    main()
