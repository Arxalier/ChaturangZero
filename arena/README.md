# arena/: benchmarking, Elo tiers and failure analysis

This folder is Person 4's part of ChaturangZero. It plays games between any
players, turns the results into Elo ratings, splits one trained network into
skill tiers, runs the "what didn't work" experiments and draws the report
figures.

Every command below runs from the repository root.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m arena.test_arena          # standalone verification (2 games + a plot)
.venv/bin/python -m pytest arena -q           # unit tests
```

## One command for the final evaluation

Once `training/train.py` has written `gen_0.pt`, `gen_mid.pt`, `gen_final.pt`
and `training_log.json` into a folder:

```bash
python -m arena.run_all --ckpt-dir checkpoints --out arena_results/final --games 20
python -m arena.run_all --ckpt-dir checkpoints --out /tmp/check --quick   # pipeline check, ~10 min
```

This runs, in order, the final tournament, the checkpoint progression, the Elo
tiers, all three ablations and every figure. Each step is resumable.

- `--max-plies` (default **300**, as in `test_vs_random.py`). Self-play uses
  120, but a trained model needs 100 to 280 plies to finish off even a random
  player, so a 120-ply cap would turn most real wins into draws.
- `gen_0.pt` is optional. Without it the progression uses a freshly
  initialised network, which is what `gen_0` is.
- `--extra-ckpt LABEL=PATH` adds more checkpoints to the progression, for
  example an earlier pure self-play run.
- Both `training_log.json` and `training_log_original.json` are plotted when
  present, along with the value-head probe that `train.py` logs.

## Players

All players implement `chaturanga.BasePlayer`. Every command names players
with a short spec, `[LABEL=]name[:key=value,...]`:

| Spec | Player |
|---|---|
| `random` | Uniformly random legal move. Anchors the Elo scale at 0. |
| `greedy` | 1-ply greedy: takes an immediate win, otherwise the move with the best material balance afterwards. Never looks at the reply. |
| `minimax:depth=2` | Alpha-beta negamax over material, exact win/loss scoring. |
| `az:ckpt=PATH,sims=N,temp=T,blunder=B` | Person 2's `AlphaZeroPlayer` around a checkpoint. `sims=0` samples the raw policy head. |
| `az-master`, `az-intermediate`, `az-novice` | The three tiers below (add `:ckpt=PATH`). |
| `az-untrained` | A freshly initialised network, for testing. |
| `novice`, `chanakya`, `aryabhata`, `panini` | Person 3's scholar bots (enabled automatically once `bots/scholar_bots.py` exists). |

The arena baselines value pieces at Pawn 1, Elephant 1.5, Minister 2, Horse 3
and Chariot 5. They are kept separate from the scholar bots on purpose, so the
baselines stay fixed if those bots are retuned.

If no `ckpt=` is given, the AlphaZero players look for `$CHATURANGZERO_CKPT`,
then `checkpoints/gen_final.pt` (where `training/train.py` saves it), then
`training/gen_final.pt`.

## Game rules used by every match

`arena/match.py` stops a game under exactly the same rules as self-play
(`training/self_play.py`):

- **Checkmate** and **bare king** are wins. **Stalemate** is a win for the
  stalemated side.
- **Threefold repetition** (same board, same side to move) is a draw.
- **Reaching the ply cap** is a draw. The cap is 120 by default (the
  `train.py` default) and 300 in `run_all`; set it with `--max-plies`.
- **Repetition-aware players:** players with a `reset()` method are reset
  before every game. Players whose `get_move` accepts `past_position_counts`
  (the current `ChaturangPlayer`) get the game's position counts, the same
  history self-play gives MCTS.
- **Illegal moves and crashes:** a player that returns an illegal move or raises an exception loses. The reason is recorded, so a long tournament never dies on one bot bug.

## Tournament: `python -m arena.tournament`

```bash
# Default pool: baselines + scholar bots (if present) + the three AZ tiers (if a checkpoint exists)
python -m arena.tournament --games 20

# Explicit players
python -m arena.tournament --players random greedy minimax:depth=2 \
    "Master=az-master:ckpt=checkpoints/gen_final.pt" --games 40 --name final

# Learning curve: one AlphaZero player per checkpoint, plus the baselines
python -m arena.tournament --progression checkpoints/gen_0.pt checkpoints/gen_mid.pt \
    checkpoints/gen_final.pt --games 20 --name progression
```

- **Fairness:** each pair plays an even number of games. Games come in pairs that share one short random opening (`--opening-plies`, default 2), played once with each colour. Neither the colour nor a lucky opening favours either side, and deterministic bots still produce varied games.
- **Elo:** a Bradley-Terry maximum-likelihood fit with draws counted as half points, anchored at Random = 0 (`--anchor`, `--anchor-elo`). The 95% intervals come from bootstrapping the games within each pair. One virtual draw per pair keeps ratings finite when a player wins every game.
- **Speed and resuming:** games run in parallel (`--workers`, default all cores but one). Each game is appended to `games.jsonl`, so an interrupted run resumes where it stopped (`--fresh` starts over).
- **Outputs** (in `arena_results/<name>/`): `summary.md` (report-ready tables), `results.json`, `wld_matrix.csv`, `score_matrix.csv`, `length_matrix.csv`, `elo.csv`, `winrate_heatmap.png` and `elo_ratings.png`.

## Elo tiers from one network: `python -m arena.elo_tuning`

| Tier | Settings | Target |
|---|---|---|
| Master | 80 simulations, τ = 0.05 | ~1800 |
| Intermediate | 15 simulations, τ = 0.60 | ~1300 |
| Novice | raw policy head (`sims=0`), 15% random moves | ~700 |

Elo has no absolute zero, so the targets only mean something once tied to an
anchor. Here every rating is relative to Random = 0, and the script **measures**
each tier against the baseline pool instead of assuming the label.

The Novice tier uses `sims=0` rather than the "1 simulation" in the project
breakdown. In `ChaturangPlayer`, `sims=0` is the path that samples the raw
policy; `sims=1` would run a one-node search.

```bash
python -m arena.elo_tuning --checkpoint checkpoints/gen_final.pt              # measure the 3 tiers
python -m arena.elo_tuning --checkpoint checkpoints/gen_final.pt --mode calibrate
```

`--mode calibrate` plays 11 candidate settings (from 50% blunders up to 160
simulations) against the pool. For each target it picks the setting whose
measured Elo is closest and reports the gap, and whether the target falls
inside the confidence interval. Outputs: `tiers.md`, `tiers.json` and
`tier_calibration.png`.

## Ablations ("what didn't work"): `python -m arena.ablation_experiments`

Each experiment measures something and reports whatever comes out; none of them
assumes its conclusion.

| Experiment | What is measured |
|---|---|
| **1. Random rollouts vs value net** | Length and endings of random playouts from the start. How well k random rollouts, material count and the value network predict the result Minimax-d2 actually reaches from the same position. Rollout-MCTS cost per simulation and its move stability across seeds. Rollout-MCTS vs network-MCTS head to head (needs a checkpoint). |
| **2. Unmasked vs masked policy** | Share of an unmasked softmax that lands on illegal moves (untrained and trained network). A controlled imitation run: two identical networks learn the same Minimax-labelled data with the same batches, one masked and one unmasked. Compares held-out accuracy, steps to converge and the share of the policy gradient that reaches legal moves. |
| **3. Greedy 1-ply vs lookahead** | An automatically built suite of Elephant-trap positions: a capture that a 2-square Elephant leap punishes with a net material loss, while a safe move exists. Measures how often each player takes the poisoned capture, plus Elephant-leap blunders per 100 moves in real games. |

```bash
python -m arena.ablation_experiments --quick                     # ~5 min smoke run
python -m arena.ablation_experiments --checkpoint checkpoints/gen_final.pt
python -m arena.ablation_experiments --exp 2                     # one experiment
```

Outputs go to `arena_results/ablations/`: `ablation_report.md`,
`ablation_results.json` and the `exp*.png` figures.

## Figures: `python -m arena.plots`

```bash
python -m arena.plots --training-log checkpoints/training_log.json \
    --tournament arena_results/final/results.json \
    --progression arena_results/progression/results.json --out figures/
```

Writes:

- `training_losses.png`
- `training_games.png`: game length and how self-play games ended
- `value_probe.png`: the value head on the start position and ±1 Chariot during training
- `winrate_heatmap.png`
- `elo_ratings.png`
- `elo_progression.png`

`--training-log` takes several files (`NAME=PATH` sets a label).

## Files

| File | Purpose |
|---|---|
| `players.py` | Baselines, tier definitions, checkpoint loading, spec parser |
| `match.py` | One game, with termination identical to self-play |
| `elo.py` | Bradley-Terry Elo with bootstrap confidence intervals |
| `tournament.py` | Parallel round-robin or gauntlet, resumable, report outputs |
| `elo_tuning.py` | Skill tiers: measure or calibrate |
| `ablation_experiments.py` | The three experiments |
| `plots.py` | All figures |
| `test_arena.py` | Standalone verification and unit tests |
