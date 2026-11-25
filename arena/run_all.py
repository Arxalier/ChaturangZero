"""
arena.run_all
-------------
One command for the whole Person 4 evaluation once training has produced
checkpoints:

    python -m arena.run_all --ckpt-dir checkpoints --out arena_results/final

Steps (each is resumable, so rerunning after an interruption is cheap):
    1. Final tournament: baselines + scholar bots (if present) + the 3 AZ tiers
    2. Learning curve: gen_0 / gen_mid / gen_final vs baselines
    3. Elo tiers: measured against the baseline pool
    4. Ablations (all three experiments, with the trained network)
    5. Figures: training curves, heatmaps, Elo charts, progression
"""

from __future__ import annotations

import argparse
import json
import os
import time

from . import ablation_experiments, elo_tuning, plots, tournament


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="folder with gen_0.pt, gen_mid.pt, gen_final.pt and training_log.json")
    p.add_argument("--out", default=os.path.join("arena_results", "final"))
    p.add_argument("--games", type=int, default=20, help="games per pair")
    p.add_argument("--max-plies", type=int, default=300,
                   help="draw after this many half-moves (default 300, as in test_vs_random.py; "
                        "self-play trains with 120, which is too short for a trained model to convert)")
    p.add_argument("--extra-ckpt", nargs="*", default=[], metavar="LABEL=PATH",
                   help="more checkpoints for the progression tournament, e.g. an earlier run")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["tournament", "progression", "tiers", "ablations", "figures"])
    p.add_argument("--quick", action="store_true", help="tiny sizes, for checking the pipeline")
    args = p.parse_args(argv)

    final = os.path.join(args.ckpt_dir, "gen_final.pt")
    if not os.path.exists(final):
        raise SystemExit(f"{final} not found")
    games = 2 if args.quick else args.games
    out = args.out
    t0 = time.time()

    def step(name):
        print(f"\n######## {name} ({(time.time() - t0) / 60:.1f} min elapsed) ########", flush=True)

    if "tournament" not in args.skip:
        step("Final tournament")
        tournament.main(["--checkpoint", final, "--games", str(games), "--workers", str(args.workers),
                         "--max-plies", str(args.max_plies),
                         "--out", os.path.join(out, "tournament"), "--name", "final"])
    if "progression" not in args.skip:
        step("Checkpoint progression")
        gen0 = os.path.join(args.ckpt_dir, "gen_0.pt")
        ckpts = [gen0 if os.path.exists(gen0) else "gen_0=untrained"]  # gen_0 is an untrained net anyway
        ckpts += list(args.extra_ckpt)
        ckpts += [c for c in (os.path.join(args.ckpt_dir, f"gen_{g}.pt") for g in ("mid", "final"))
                  if os.path.exists(c)]
        tournament.main(["--progression", *ckpts, "--games", str(games), "--workers", str(args.workers),
                         "--max-plies", str(args.max_plies),
                         "--out", os.path.join(out, "progression"), "--name", "progression"])
    if "tiers" not in args.skip:
        step("Elo tiers")
        elo_tuning.main(["--checkpoint", final, "--games", str(games), "--workers", str(args.workers),
                         "--max-plies", str(args.max_plies), "--out", os.path.join(out, "elo_tiers")])
    if "ablations" not in args.skip:
        step("Ablations")
        abl = ["--checkpoint", final, "--workers", str(args.workers), "--out", os.path.join(out, "ablations")]
        ablation_experiments.main(abl + (["--quick"] if args.quick else []))
    if "figures" not in args.skip:
        step("Figures")
        fig_args = ["--out", os.path.join(out, "figures")]
        names = {"training_log.json": "Final training run",
                 "training_log_original.json": "Original pure self-play run"}
        logs = [f"{name}={os.path.join(args.ckpt_dir, f)}" for f, name in names.items()
                if os.path.exists(os.path.join(args.ckpt_dir, f))]
        if logs:
            fig_args += ["--training-log", *logs]
        tr = os.path.join(out, "tournament", "results.json")
        if os.path.exists(tr):
            fig_args += ["--tournament", tr]
        pr = os.path.join(out, "progression", "results.json")
        if os.path.exists(pr):
            fig_args += ["--progression", pr]
        plots.main(fig_args)

    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump({"ckpt_dir": args.ckpt_dir, "games_per_pair": games, "max_plies": args.max_plies,
                   "extra_ckpt": args.extra_ckpt, "quick": args.quick,
                   "minutes": (time.time() - t0) / 60}, f, indent=2)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results in {out}/")


if __name__ == "__main__":
    main()
