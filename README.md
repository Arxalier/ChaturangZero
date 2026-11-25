# ChaturangZero

**CS F407 (Artificial Intelligence), AY 2026–27 Semester I — BITS Pilani, K.K. Birla Goa Campus**  
**Team Syllogism:** Aaryan Pandey, Vraj Makwana, Harshal Lahoti, Sahaj Jhamb

ChaturangZero is a reinforcement learning and game-engine project for **Chaturanga**, the ancient Indian predecessor to chess, shogi, and makruk, played on an 8×8 uncheckered grid (the *Ashtapada*).

This repository covers both halves of the project: the historical **Murray/Golombek ruleset** engine with Perft and invariant testing, an interactive local browser UI, four depth-2 heuristic "Scholar Bots", a dual-headed PyTorch ResNet paired with PUCT Monte Carlo Tree Search (MCTS), and our post-midsem experiments with pure self-play collapse and mobility-based heuristic bootstrapping.

---

## Table of Contents
1. [The Rules of Chaturanga (Murray/Golombek)](#the-rules-of-chaturanga-murraygolombek)
2. [Repository Layout & Module Guide](#repository-layout--module-guide)
3. [Quickstart & Usage](#quickstart--usage)
   - [1. Installation](#1-installation)
   - [2. Browser UI (Engine Explorer & Bot Arena)](#2-browser-ui-engine-explorer--bot-arena)
   - [3. Interactive Terminal Play](#3-interactive-terminal-play)
   - [4. Benchmarking & Value Discovery](#4-benchmarking--value-discovery)
   - [5. Automated Tests, Invariant Fuzzing & Perft](#5-automated-tests-invariant-fuzzing--perft)
   - [6. Python Package Usage](#6-python-package-usage)
4. [System Design: How Everything Fits Together](#system-design-how-everything-fits-together)
   - [A. Board Representations (2D Matrix vs. 3D Tensor)](#a-board-representations-2d-matrix-vs-3d-tensor)
   - [B. Action Space & Illegal Move Masking (4,096 Dimensions)](#b-action-space--illegal-move-masking-4096-dimensions)
   - [C. Move Generation & Raycast Check Detection](#c-move-generation--raycast-check-detection)
   - [D. Testing Strategy: Perft vs. Random Playout Fuzzing](#d-testing-strategy-perft-vs-random-playout-fuzzing)
5. [Post-Midsem Progress (Milestones 2–4)](#post-midsem-progress-milestones-24)
   - [Pure Self-Play Failure & Draw Collapse](#pure-self-play-failure--draw-collapse)
   - [Mobility-Derived Piece Values & Supervised Bootstrapping](#mobility-derived-piece-values--supervised-bootstrapping)
   - [Scholar Heuristic Bots & Tournament Benchmarks](#scholar-heuristic-bots--tournament-benchmarks)
   - [Observed Results & Current Limitations](#observed-results--current-limitations)

---

## The Rules of Chaturanga (Murray/Golombek)

> *Historical Reference: [ChessVariants Historical Rules: Chaturanga](https://www.chessvariants.com/historic.dir/chaturanga.html)*

### 1. Board & Starting Symmetry
* **Ashtapada:** An 8×8 uncheckered grid (`WHITE = 1` on ranks 6–7, `BLACK = -1` on ranks 0–1). Corner and center board markings are decorative only.
* **Alignment:** Each side starts with 16 pieces mirroring each other across the center:
  * **Kings (*Rāja*)** face each other on the e-file (col 4).
  * **Ministers (*Mantri*)** face each other on the d-file (col 3).

### 2. Piece Movements & Sanskrit Names

| English Name | Sanskrit Name | Movement Mechanics | Modern Chess Equivalent |
|---|---|---|---|
| **King** | *Rāja* | 1 square in any orthogonal or diagonal direction. Cannot move into check. No castling. | King (no castling) |
| **Minister** | *Mantri* | Exactly 1 square diagonally in any of the 4 directions. | Queen (heavily restricted) |
| **Elephant** | *Gaja / Hasti* | Exactly 2 squares diagonally, leaping over any intervening piece. | Alfil |
| **Horse** | *Ashwa* | L-shaped leap (2×3), jumping over intervening pieces. | Knight |
| **Chariot** | *Ratha* | Orthogonal slider along ranks and files over open squares. | Rook |
| **Foot Soldier** | *Padāti* | 1 square straight forward to move; 1 square diagonally forward to capture. No double-step, no en passant. | Pawn (restricted) |

### 3. Automatic Promotion
When a Foot Soldier (*Padāti*) reaches the opponent's back rank (row 0 for White, row 7 for Black), it **automatically promotes to a Minister (*Mantri*)**. There is no underpromotion choice.

### 4. Terminal States
Games end right away when any of these three conditions occur:
* **Checkmate (`+1` Win / `-1` Loss):** The opponent's King is under attack and has no legal move to escape, block, or capture the attacker.
* **Baring the King (`+1` Win / `-1` Loss):** A player captures the opponent's last non-King piece, leaving the enemy King alone on the board. The capturing player wins immediately.
* **Stalemate (`+1` Win / `-1` Loss):** A player is **not** in check and has zero legal moves. Under Murray/Golombek rules, **the stalemated player wins (`+1`)**, rather than drawing.

---

## Repository Layout & Module Guide

```
ChaturangZero/
├── chaturanga/              # Core game engine package
│   ├── __init__.py          # Public API exports
│   ├── pieces.py            # Signed int8 piece codes, Sanskrit aliases, display symbols
│   ├── moves.py             # Pseudo-legal move generators per piece type
│   ├── board.py             # Starting board, copy-on-move transitions, ASCII render
│   ├── rules.py             # Raycast check detector, legal move filter, outcomes, perft
│   └── env.py               # ChaturangaEnv RL environment with move/outcome caching
├── neural/                  # PyTorch tensor encoding & dual-headed ResNet
│   ├── encoder.py           # Board -> (13, 8, 8) tensor & move <-> 4,096 action index
│   └── model.py             # 4-block ResNet (ChaturangaNet) with Policy & Value heads
├── training/                # MCTS search, self-play, and supervised bootstrapping
│   ├── mcts_agent.py        # NeuralMCTS (PUCT) and ChaturangPlayer (BasePlayer)
│   ├── self_play.py         # Self-play rollouts, curriculum generator, ReplayBuffer
│   ├── train.py             # Masked multi-task loss, flip augmentation, diagnostics
│   ├── bootstrap_train.py   # 2-ply heuristic game generation & transfer fine-tuning
│   └── test_pipeline.py     # CPU end-to-end MCTS + training step sanity check
├── bots/                    # Heuristic opponents (Scholar Bots)
│   ├── __init__.py
│   ├── scholar_bots.py      # NoviceBot, ChanakyaBot, AryabhataBot, PaniniBot
│   └── test_bots.py         # Automated verification matches between Scholar Bots
├── ui/                      # Local stdlib web UI + JSON API
│   ├── server.py            # ThreadingHTTPServer + static asset hosting + --sims flag
│   ├── engine_api.py        # Adapter connecting engine, Scholar Bots, and ChaturangZero
│   ├── scenarios.py         # 7 demo positions (mate, stalemate-win, bare king, pin, etc.)
│   ├── README.md            # Notes specific to the browser UI
│   └── static/              # Frontend HTML/CSS/JS (dark & light themes)
├── checkpoints/             # Saved PyTorch weights (gen_final.pt) and training JSON logs
├── tests/                   # Automated unittest suite, Perft, and fuzz validation
│   ├── test_validity.py     # unittest suite for Perft(1)/Perft(2) and game invariants
│   ├── perft.py             # Perft leaf counter with root divide & fuzz option
│   └── validity_test.py     # Random playout fuzz tester checking 7 game invariants
├── misc/                    # Analysis & ablation scripts (discover_values, test_vs_random)
├── report/                  # Final 4-page LaTeX report & Markdown draft
├── docs/                    # Milestone specs & early reference notes
├── benchmark_bots.py        # 8-game tournament script vs. all 4 Scholar Bots
├── play.py                  # Interactive terminal play (human vs. human or AI)
├── report.txt               # Saved diagnostic probes and benchmark tournament logs
└── requirements.txt         # Core dependencies (numpy, torch)
```

### Key Package Details

* **`chaturanga/pieces.py`**: Stores pieces as signed `int8` constants (`EMPTY = 0`, White `+1..+6` for `PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING`, Black `-1..-6`), along with Sanskrit names (`PADATI`, `ASHWA`, `GAJA`, `RATHA`, `MANTRI`, `RAJA`) and ASCII/Unicode symbols.
* **`chaturanga/board.py`**: `create_starting_board()` returns the initial `(8, 8) np.int8` array. `make_move(board, start_pos, end_pos)` copies the board before applying moves (safe for MCTS rollouts) and promotes any Pawn reaching row `0` (White) or row `7` (Black) to a Minister.
* **`chaturanga/rules.py`**: Holds `find_king`, raycast `is_in_check`, `get_all_legal_moves`, `get_game_outcome` (checking Bare King, Checkmate, and Stalemate), and `perft`.
* **`chaturanga/env.py`**: Wraps `board`, `current_player`, and `move_history` in `ChaturangaEnv` with `reset()`, `step(move)`, `copy()`, and cached legal moves/outcomes per ply.
* **`ui/`**: Built with Python's standard library `http.server` so it runs offline with no JS build step. Exposes legal move highlighting, pseudo-legal rejection overlays, `is_in_check` threat maps, Perft divide, `int8` board views, and an opponent dropdown with a live MCTS simulation slider for `ChaturangZero`.

---

## Quickstart & Usage

### 1. Installation
Requires Python 3.8+ with `numpy` and `torch`:
```bash
pip install -r requirements.txt
```

### 2. Browser UI (Engine Explorer & Bot Arena)
Start the local web server:
```bash
# Default launch (http://127.0.0.1:8000, default 40 MCTS sims for ChaturangZero)
python3 ui/server.py

# Custom port and initial simulation count
python3 ui/server.py --port 8000 --sims 100
```
Once open in your browser:
* **Play vs Bot / ChaturangZero:** Choose `ChaturangZero` (selected by default) or one of the Scholar Bots (`Chanakya`, `Aryabhata`, `Panini`, `Novice`). Use the **Sims** slider to adjust MCTS lookahead on the fly and toggle **Play vs Bot (Black)** or press <kbd>Z</kbd> for single bot moves.
* **Demo Positions:** Press <kbd>1</kbd>–<kbd>7</kbd> to jump to preset states (Starting Ashtapada, Mate in 1, Stalemate-Win, Bare King, Pin, Pawn Promotion, Leapers in a Crowd).
* Press <kbd>?</kbd> for the full keyboard shortcut modal. See [`ui/README.md`](ui/README.md) for more UI notes.

### 3. Interactive Terminal Play
Play human vs. human or human vs. `ChaturangZero` in the terminal:
```bash
# Two-player local game
python3 play.py

# Play as White against trained AI (adjust lookahead with --sims)
python3 play.py --ai --sims 100
```
* Enter moves as `start_row,start_col end_row,end_col` (for example, `6,4 5,4`).
* Type `perft <depth>` mid-game to inspect leaf node counts, or `q` to quit.

### 4. Benchmarking & Value Discovery
Run the evaluation scripts from the repo root:
```bash
# 8-game Scholar Bots tournament (appends summary to report.txt)
python3 benchmark_bots.py --games-per-bot 2 --sims 120 --max-plies 220

# Benchmark against uniform random play (e.g., 40 sims over 10 games)
python3 misc/test_vs_random.py 40 10

# Compute empty-board mobility piece values and centrality matrix
python3 misc/discover_values.py
```

### 5. Automated Tests, Invariant Fuzzing & Perft
Run the `unittest` suite:
```bash
python3 -m unittest discover -v tests
```

Run standalone Perft divide or random playout invariant checks:
```bash
# Perft depth 2 with move-by-move divide
python3 tests/perft.py --depth 2 --divide

# Fuzz 100 random games up to 300 plies each
python3 tests/validity_test.py -n 100 -m 300
```

`validity_test.py` checks 7 invariants after every single ply:
1. **King Integrity:** Exactly 1 White King (`+6`) and 1 Black King (`-6`) exist at all times.
2. **Move Legality:** Executed moves come directly from `get_all_legal_moves()`.
3. **Check Safety:** No move leaves or places the moving player's King in check.
4. **Pawn Promotion:** No Pawns remain on back ranks (`0` for White, `7` for Black).
5. **Material Monotonicity:** Total pieces never exceed 32 and only decrease on captures.
6. **Board Geometry:** Board remains `(8, 8)` with values in `[-6, 6]`.
7. **Terminal Soundness:** Bare King, Checkmate (`-1`), and Stalemate (`+1`) terminate properly.

### 6. Python Package Usage
```python
from chaturanga import ChaturangaEnv

env = ChaturangaEnv()
legal_moves = env.get_legal_moves()
print(f"Available legal moves: {len(legal_moves)}")

board, outcome, done = env.step(legal_moves[0])
env.render()
```

---

## System Design: How Everything Fits Together

### A. Board Representations (2D Matrix vs. 3D Tensor)
The codebase separates fast engine simulation from neural network input:
1. **2D Engine Matrix (`(8, 8) np.int8`):** Used by `chaturanga/` and the minimax bots for cheap array copies and integer comparisons (`+1..+6` White, `-1..-6` Black, `0` empty).
2. **3D Neural Tensor (`(13, 8, 8) torch.float32`):** Built in `neural/encoder.py`.
   * **Channels 0–5:** One-hot binary planes for White `P, H, E, R, M, K`.
   * **Channels 6–11:** One-hot binary planes for Black `P, H, E, R, M, K`.
   * **Channel 12:** Constant turn plane (`1.0` if White to move, `0.0` if Black).
   * Using one-hot planes avoids feeding numeric encodings (`1..6`) into conv filters where arithmetic would treat a Minister (`5`) as `Pawn (1) + Chariot (4)`, and tiling the turn flag spatially lets early `3×3` conv layers condition on turn everywhere.

### B. Action Space & Illegal Move Masking (4,096 Dimensions)
Moves are flattened into a fixed `64 × 64 = 4,096` vector:
$$\text{action} = (r_{\text{start}} \times 8 + c_{\text{start}}) \times 64 + (r_{\text{end}} \times 8 + c_{\text{end}})$$
Only ~15–35 of those 4,096 actions are legal in a typical Chaturanga position. Both inference (`ChaturangaNet.forward`) and loss computation (`ChaturangZeroLoss`) mask illegal indices to `-1e9` before `log_softmax` so illegal moves never receive probability mass or noisy gradients.

### C. Move Generation & Raycast Check Detection
Move generation filters pseudo-legal piece moves (`chaturanga.moves`) by simulating each candidate with `make_move` and discarding any that leave the King in check.

Because `is_in_check` is called for every candidate move in search and training, scanning all 64 enemy squares and allocating move lists per ply would dominate runtime. Instead, `is_in_check` probes outward directly from the King's square:
1. 4 orthogonal rays until first blocker (checking for enemy Chariots),
2. 8 Knight jump offsets (enemy Horses),
3. 4 2-square diagonal leap offsets (enemy Elephants),
4. 4 adjacent diagonals (enemy Ministers),
5. 8 adjacent squares (enemy King), and
6. 2 forward-attack squares (enemy Pawns).

### D. Testing Strategy: Perft vs. Random Playout Fuzzing

| Feature | Perft | Random Playout Fuzzing |
|---|---|---|
| **Traversal** | Exhaustive DFS over every legal branch | Single random legal trajectory per game |
| **Depth** | Shallow (depths 1–4; `16`, `256` at depths 1–2) | Long (100–300+ plies to termination) |
| **Catches** | Missing/extra moves, leap bounds, pin bugs | Promotion bugs, Bare King / Stalemate triggers, crashes |

---

## Post-Midsem Progress (Milestones 2–4)

### Pure Self-Play Failure & Draw Collapse
After finishing the Milestone 1 engine and Milestone 2 network/MCTS setup (`ChaturangaNet` with `4` residual blocks at `64` channels + `NeuralMCTS` using `c_puct = 1.25`), our initial goal was pure self-play reinforcement learning with no heuristics.

In practice, Chaturanga moves much slower than modern chess: Pawns lack an initial double-step, Ministers step 1 diagonal square, and Elephants jump 2 diagonals. Even uniform random games have a median length of 411 plies, and armies take 30–50 plies just to reach each other. Over 30 self-play iterations (15 games/iteration, 450 games total at 20 MCTS sims per move), almost every game (>90%) ended in a 3-fold repetition or turn-cap draw with target reward `z = 0`.

Another bottleneck was network size: `ChaturangaNet` has **8,702,692 parameters**, of which **8,392,704 (96.4%)** come from the dense `Linear(2048 -> 4096)` policy projection. Sequential batch-1 forward passes through an 8.70M-parameter model made game generation slow on CPU and Colab, limiting us to a few hundred games—nowhere near enough data for a model that size to learn piece values from sparse terminal rewards. Its value head flattened around `+0.1330` regardless of material balance, and it scored `0W-1L-9D` against random play at 20 sims.

### Mobility-Derived Piece Values & Supervised Bootstrapping
To see if the network and MCTS could still play coherent Chaturanga with a working value signal, we derived initial piece weights from the move rules rather than hand-picking chess values:
* [`misc/discover_values.py`](misc/discover_values.py) measures average empty-board legal moves (`Pawn: 1.64`, `Elephant: 2.25`, `Minister: 3.06`, `Horse: 5.25`, `Chariot: 14.00`), applies sublinear dampening (`(avg / base) ** 0.85`) to adjust for slider crowding (`1.00, 1.31, 1.70, 2.69, 6.19`), and normalizes a `[0.00, 0.20]` central square bonus.
* After slight manual calibration (`[P: 1.0, E: 1.5, M: 2.0, H: 3.0, C: 5.0]`), [`training/bootstrap_train.py`](training/bootstrap_train.py) generated 150 curriculum games using a 2-ply lookahead teacher (`eps = 0.15`, ~10,000 moves) and fine-tuned the CNN for 12 epochs (`4.0 * MSE(v, z) + masked CrossEntropy(p, pi)`). Value loss fell from `0.1752` to `0.0099` and policy loss from `2.7178` to `0.2218`.

### Scholar Heuristic Bots & Tournament Benchmarks
To test the bootstrapped model against more structured opponents than random moves, [`bots/scholar_bots.py`](bots/scholar_bots.py) implements four `BasePlayer` baselines:
* **NoviceBot:** 80% random legal moves, 20% greedy captures.
* **ChanakyaBot:** Depth-2 alpha-beta minimax focused on material, checks, and pushing toward Bare King.
* **AryabhataBot:** Depth-2 alpha-beta minimax using piece-square tables, Pawn advancement, and Gaja mobility.
* **PaniniBot:** Depth-2 alpha-beta minimax using single-pass attack/defense maps to penalize hanging pieces and exposed Kings.

### Observed Results & Current Limitations
* **Material Value Probes ([`report.txt`](report.txt)):** From the starting position, net value deltas (`White Up 1 piece - White Down 1 piece`) became positive and monotonic across every piece type: Pawn (`+0.0803`), Minister (`+0.0939`), Elephant (`+0.1383`), Horse (`+0.4687`), Chariot (`+0.4750`), and Major `+Ratha+Ashwa` (`+0.9598`).
* **vs. Uniform Random (10 games):** Jumped from `0W-1L-9D` (pure self-play, 20 sims) to **90% win rate (`9W-0L-1D`)** at both 40 sims and 200 sims.
* **vs. Scholar Bots (8 games, 1 White / 1 Black per bot):**

| Opponent | AI Wins | Bot Wins | Draws | AI Win Rate |
|---|:---:|:---:|:---:|:---:|
| NoviceBot | 2 | 0 | 0 | 100.0% |
| ChanakyaBot | 0 | 0 | 2 | 0.0% |
| AryabhataBot | 0 | 0 | 2 | 0.0% |
| PaniniBot | 0 | 1 | 1 | 0.0% |
| **Total** | **2 (25.0%)** | **1 (12.5%)** | **5 (62.5%)** | **25.0% (87.5% unbeaten)** |

* **Limitations & Next Steps:**
  * **Repetition Draws (62.5%):** Scholar Bots have no move-history state and oscillate between equal-scoring moves, while neutral `V = 0.0` draw evaluations give MCTS no incentive to refuse repetitions.
  * **Simulation Compute Cost:** On CPU, 300 sims per move requires roughly 30,000 forward passes per game (~4–5 minutes/game), while low simulation budgets (20–30 sims) search too shallowly and fall into depth-2 minimax traps.
  * **Model Size vs. Data:** Replacing the 8.70M-parameter network (especially the 8.39M dense policy head) with a compact CNN on the order of a few thousand parameters would make forward passes much faster and enable training across thousands of self-play games.
