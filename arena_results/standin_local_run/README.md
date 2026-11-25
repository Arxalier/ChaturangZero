# Stand-in results (not the final model)

These results come from a local training run (`python -m training.train
--save-dir checkpoints/local_run`, all defaults, commit 6178ff3). That run was
made only to exercise the arena before the team's trained `gen_final.pt` was
available.

**Model-independent results.** The parts of `ablations/` that do not use the
network are final and reproducible with seed 0:

- Experiment 1a, 1b (rollouts and material), 1c
- Experiment 2a (untrained network) and 2b
- Experiment 3 (baseline bots)

**Stand-in results.** Everything involving the AlphaZero player only reflects
this stand-in model, which suffered self-play draw collapse:

- `tournament/`, `progression/` and `elo_tiers/`
- 1b (trained value net), 1d and 2a (trained network)
- The AZ rows of 3a

To redo everything with the real checkpoints:

```bash
python -m arena.run_all --ckpt-dir <folder with gen_0/gen_mid/gen_final.pt + training_log.json> \
    --out arena_results/final --games 20
```

The real results, from the team's `checkpoints/gen_final.pt`, are in `arena_results/final/`.
