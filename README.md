# ChaturangZero: Self-Play AlphaZero Agent for Chaturanga

ChaturangZero is an artificial intelligence project aiming to train an AlphaZero-style reinforcement learning agent for **Chaturanga**, the ancient Indian ancestor of Chess, Shogi, and Makruk, played on an 8x8 uncheckered grid (the *Ashtapada*).

This repository implements the historical **Murray/Golombek ruleset** with high-performance move generation, strict mathematical invariant checking, automated unit testing, an interactive browser explorer, and full support for future Monte Carlo Tree Search (MCTS) and deep neural network integration.

---

## Table of Contents
1. [The Rules of Chaturanga](#the-rules-of-chaturanga-murraygolombek-ruleset)
2. [Repository Layout & File Breakdown](#repository-layout--file-breakdown)
3. [Quickstart & Usage Instructions](#quickstart--usage-instructions)
   - [1. Installation](#1-installation)
   - [2. Visual Engine Explorer (Browser UI)](#2-visual-engine-explorer-browser-ui)
   - [3. Interactive Terminal Play](#3-interactive-terminal-play)
   - [4. AlphaZero MCTS Agent (Milestone 2 Smoke Test)](#4-alphazero-mcts-agent-milestone-2-smoke-test)
   - [5. Automated Unit Testing](#5-automated-unit-testing)
   - [6. Random Playout Fuzz Testing & Invariant Validation](#6-random-playout-fuzz-testing--invariant-validation)
   - [7. Perft Benchmarks & Move Divide](#7-perft-benchmarks--move-divide)
   - [8. Using as a Python Package](#8-using-as-a-python-package)
4. [Comprehensive Study Guide: How the System Works](#comprehensive-study-guide-how-the-system-works)
   - [A. Board Representations (2D Matrix vs. 3D Tensor)](#a-board-representations-2d-matrix-vs-3d-tensor)
   - [B. Action Space Representation (4,096 Dimensions)](#b-action-space-representation-4096-dimensions)
   - [C. Move Generation & Check Detection Pipeline](#c-move-generation--check-detection-pipeline)
   - [D. Dual-Layer Testing Philosophy (Perft vs. Fuzzing)](#d-dual-layer-testing-philosophy-perft-vs-fuzzing)
   - [E. Roadmap to Milestone 2 & AlphaZero](#e-roadmap-to-milestone-2--alphazero)

---

## The Rules of Chaturanga (Murray/Golombek Ruleset)

> *Historical Source: [ChessVariants Historical Rules: Chaturanga](https://www.chessvariants.com/historic.dir/chaturanga.html)*

### 1. The Board & Setup
- **Ashtapada**: An 8x8 uncheckered grid.
- **Symmetry**: 16 pieces per player mirroring each other across ranks 0–1 (Black) and 6–7 (White).
  - The **Kings (*Rāja*)** face each other on the e-file (col 4).
  - The **Ministers (*Mantri*)** face each other on the d-file (col 3).

### 2. Piece Movements & Sanskrit Aliases

| English Name | Sanskrit Name | Movement Mechanics | Modern Chess Equivalent |
|---|---|---|---|
| **King** | *Rāja* | 1 square in any direction (orthogonal or diagonal). Cannot move into check. No castling. | King (no castling) |
| **Minister** | *Mantri* | Exactly 1 square diagonally in any of the 4 directions. | Queen (heavily restricted) |
| **Elephant** | *Gaja / Hasti* | Exactly 2 squares diagonally, leaping over intervening pieces. | Alfil (ancient diagonal leaper) |
| **Horse** | *Ashwa* | L-shaped leap (2x3 rectangle), jumping over intervening pieces. | Knight |
| **Chariot** | *Ratha* | Orthogonal slider along ranks and files over unobstructed squares. | Rook |
| **Foot Soldier** | *Padāti* | Moves 1 square straight forward; captures 1 square diagonally forward. No initial double-step, no en passant. | Pawn (no double-step/en passant) |

### 3. Automatic Pawn Promotion
When a Foot Soldier (*Padāti*) reaches the furthest rank on the opponent's side (rank 0 for White, rank 7 for Black), it **promotes exclusively and automatically to a Minister (*Mantri*)**. Unlike modern chess, there is no choice of piece.

### 4. Terminal States & Reward Triggers
The game ends immediately upon meeting any of the following three conditions:
- **Checkmate (+1 Win / -1 Loss)**: The opponent's King is under attack (in check) and has zero legal moves to escape, block, or capture the attacker.
- **Baring the King (+1 Win / -1 Loss)**: A player captures the opponent's last remaining non-King piece, leaving the opponent's King completely isolated on the board. The baring player instantly wins.
- **Stalemate (+1 Win / -1 Loss)**: A player is **not** in check but has zero legal moves. Per Murray/Golombek rules, **the stalemated player wins the game (+1)**! (Unlike modern chess where it is a draw).

---

## Repository Layout & File Breakdown

```
ChaturangZero/
├── chaturanga/              # Core game engine package
│   ├── __init__.py          # Public API exports
│   ├── pieces.py            # Numerical piece representations, Sanskrit constants, display glyphs
│   ├── moves.py             # Pseudo-legal move generators for each piece type
│   ├── board.py             # Board initialization, immutable state transitions, text rendering
│   ├── rules.py             # Optimized check detection, legal move pruning, outcomes, perft
│   └── env.py               # ChaturangaEnv RL/Gym environment wrapper with state caching
├── ui/                      # Browser UI for presentations (stdlib HTTP server + JSON API)
│   ├── server.py            # HTTP server routing API requests and serving static assets
│   ├── engine_api.py        # Adapter converting engine outputs into JSON with call tracing
│   ├── scenarios.py         # 7 preset tactical demo positions (mate, stalemate, bare king, etc.)
│   ├── README.md            # Dedicated documentation for the visual browser UI
│   └── static/              # Frontend web assets (index.html, style.css, app.js)
├── tests/                   # Automated unit testing suite
│   ├── __init__.py          # Package initialization
│   └── test_validity.py     # Unittest suite verifying Perft counts and rollout invariants
├── mcts.py                  # PUCT-guided Monte Carlo Tree Search (AlphaZero tree search)
├── test_mcts.py             # MCTS smoke test with dummy policy/value evaluator
├── validity_test.py         # Standalone random rollout fuzz tester & invariant validator
├── perft.py                 # Perft tool with root-move breakdown ("divide") and fuzzing
├── play.py                  # Interactive CLI play loop (2-player human terminal interface)
├── requirements.txt         # Minimal project dependencies (numpy)
├── info.txt                 # Quick reference: piece encodings, board logic, 3D tensor map
├── info_unit_test.txt       # Technical documentation: testing methodologies & optimizations
├── milestone1.txt           # Milestone 1 specification and rules reference
└── projectDetails.txt       # 5-phase AlphaZero architecture roadmap & team division
```

### File-by-File Details

#### `chaturanga/pieces.py`
Defines piece constants as signed 8-bit integers:
* `EMPTY = 0`
* Positive integers for White: `PAWN=1`, `HORSE=2`, `ELEPHANT=3`, `CHARIOT=4`, `MINISTER=5`, `KING=6`
* Negative integers for Black: `-1` to `-6`
* Exports Sanskrit aliases (`PADATI`, `ASHWA`, `GAJA`, `RATHA`, `MANTRI`, `RAJA`) and ASCII/Unicode display symbols.

#### `chaturanga/moves.py`
Contains pseudo-legal move generators for each piece type:
* `get_chariot_moves()`: Orthogonal raycaster terminating on board edges, obstacles, or enemy captures.
* `get_horse_moves()`: 8 L-shaped leaps jumping over intervening squares.
* `get_elephant_moves()`: 4 strict 2-square diagonal leaps jumping over intervening squares.
* `get_minister_moves()`: 4 strict 1-square diagonal steps.
* `get_king_moves()`: 8 adjacent orthogonal and diagonal steps.
* `get_pawn_moves()`: 1 square straight forward into empty space, 1 square diagonal forward capture.

#### `chaturanga/board.py`
* `create_starting_board()`: Creates an $8 \times 8$ `np.int8` array initialized with Murray/Golombek symmetry.
* `make_move(board, start_pos, end_pos)`: Applies a move and returns an **independent copy** of the board, automatically promoting any pawn reaching the opposite back rank (row 0 for White, row 7 for Black) strictly to a Minister.
* `render_board()`: Produces a human-readable ASCII layout of the board.

#### `chaturanga/rules.py`
* `find_king(board, player)`: Fast linear scan locating the King's coordinates.
* `is_in_check(board, player)`: High-performance raycast and leap check detector evaluating direct attack lines from the King's square.
* `get_all_legal_moves(board, player)`: Filters pseudo-legal moves by simulating them and discarding any move that leaves the moving player's King in check.
* `get_game_outcome(board, player, legal_moves)`: Evaluates terminal game states (Bare King, Checkmate, Stalemate).
* `perft(board, player, depth)`: Recursive depth-first tree traversal counting leaf nodes.

#### `chaturanga/env.py`
Encapsulates game state into a standard RL environment:
* `reset()`: Resets to starting board and clears history.
* `step(move)`: Executes a move, switches turn, updates history, and returns `(board, outcome, done)`.
* Caches legal moves and outcomes per ply to eliminate redundant computations.

#### `ui/` (Browser Engine Explorer)
A standalone presentation tool built with Python's standard library `http.server`:
* `ui/server.py`: Listens on `http://127.0.0.1:8000`.
* `ui/engine_api.py`: Dispatches HTTP calls directly to `chaturanga` functions and logs an execution trace.
* `ui/scenarios.py`: Provides 7 curated board states demonstrating complex rule scenarios.
* `ui/static/`: Interactive dark-mode frontend featuring square highlighting, move history, threat maps, and live board tensor inspection.

#### `mcts.py`
Monte Carlo Tree Search algorithm specialized for AlphaZero:
* `Node`: Tracks visit counts ($N$), accumulated value ($W$), mean value ($Q = W/N$), prior probability ($P$), parent, and child edges.
* `get_puct_value()`: Implements the Polynomial Upper Confidence Tree (PUCT) formula:
  $$\text{PUCT} = Q + c_{\text{puct}} \cdot P \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N}$$
* `search()`: 4-step search loop (Selection $\to$ Expansion $\to$ Evaluation $\to$ Backpropagation).
* `get_best_move()`: Selects the most robust action according to highest visit count ($N$).
* `dummy_evaluate()`: Random-uniform prior and scalar placeholder enabling full tree exploration before PyTorch model training.

#### `test_mcts.py`
Smoke test verifying tree node expansion, PUCT selection, leaf evaluation, and action selection over 50 simulations from the starting board.

#### `validity_test.py`
High-speed random playout fuzz testing script. Plays full games to completion with random moves and strictly validates 7 categories of game invariants after every single ply.

#### `perft.py`
Performance verification tool supporting standard depth search, root-move breakdown (`--divide`), and integrated playout fuzz testing (`--fuzz`).

---

## Quickstart & Usage Instructions

### 1. Installation
Ensure Python 3.8+ is installed. Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Visual Engine Explorer (Browser UI)
Launch the browser UI that visualizes every part of the engine — board, legal moves, pseudo-legal vs legal filtering, check/threat maps, terminal states, perft, and raw board tensors:
```bash
python3 ui/server.py
```
Open **`http://127.0.0.1:8000`** in your browser.
* Press <kbd>1</kbd>–<kbd>7</kbd> to load preset demo scenarios:
  1. Starting Ashtapada
  2. Checkmate in 1 (Chariot + Minister battery)
  3. Stalemate (demonstrates stalemated player WINNING)
  4. Bare King (capturing second-to-last piece triggers instant win)
  5. Absolute Pin (piece blocked from exposing King)
  6. Pawn Promotion (pawn reaches 8th rank $\to$ Minister)
  7. Leapers in a Crowd (Elephants and Horses jumping over blocked lines)
* Press <kbd>?</kbd> for the keyboard shortcut guide.
* See [ui/README.md](ui/README.md) for deeper UI documentation.

### 3. Interactive Terminal Play
Play manually from the terminal (2-player human hot-seat):
```bash
python3 play.py
```
* Enter moves as: `start_row,start_col end_row,end_col` (e.g., `6,4 5,4`).
* Type `perft <depth>` during play to run an instant node count from the current board.
* Type `q` to quit.

### 4. AlphaZero MCTS Agent (Milestone 2 Smoke Test)
Run the PUCT-guided Monte Carlo Tree Search agent:
```bash
python3 test_mcts.py
```
Performs 50 MCTS lookahead simulations guided by PUCT node values and selects the best move based on visit counts ($N$).

### 5. Automated Unit Testing
Run the automated test suite verifying Perft node counts and invariant rollouts:
```bash
# Standard library unittest (zero dependencies required)
python3 -m unittest discover -v tests

# Or with pytest (if installed)
pytest -v
```

### 6. Random Playout Fuzz Testing & Invariant Validation
Simulate complete random games to termination, asserting strict mathematical and ruleset invariants after **every single move**:
```bash
# Run 100 random games with up to 300 plies each
python3 validity_test.py -n 100 -m 300

# Run with a fixed seed and verbose per-game output
python3 validity_test.py -n 10 -m 200 --seed 42 --verbose
```

#### CLI Parameters for `validity_test.py`:
| Parameter | Short | Default | Description |
|---|:---:|:---:|---|
| `--num-games` | `-n` | `100` | Number of complete games to simulate |
| `--max-moves` | `-m` | `300` | Maximum half-moves (plies) per game before cutoff draw |
| `--seed` | `-s` | `None` | Random seed for deterministic reproducibility |
| `--perft-depth` | `-p` | `0` | Run Perft benchmark at given depth before rollouts |
| `--verbose` | `-v` | `False` | Print winner, plies, and reason for every individual game |
| `--quiet` | `-q` | `False` | Suppress progress logs and output summary table only |

#### The 7 Invariants Asserted at Every Ply:
1. **King Integrity**: Exactly 1 White King (`+6`) and 1 Black King (`-6`) exist on the board at all times; Kings can never be captured.
2. **Move Legality**: The executed move was strictly in `get_all_legal_moves()`; source square had a friendly piece, target square did not.
3. **Check Safety**: A player's move can **never** leave or place their own King in check (absolute pins and check evasion strictly enforced).
4. **Pawn Promotion**: No Pawns ever exist on back ranks (row 0 for White, row 7 for Black); always promoted strictly and automatically to a Minister.
5. **Material Monotonicity**: Total pieces $\le 32$, non-increasing. Moving player's piece count is unchanged; opponent's is unchanged or decreased by 1 on capture.
6. **Board Geometry**: Array strictly $(8, 8)$ with values in range $[-6, 6]$.
7. **Terminal Soundness**: Bare King (opponent left with only their King) terminates immediately with victory; Stalemated player wins (+1); Checkmated player loses (-1).

### 7. Perft Benchmarks & Move Divide
Calculate exact leaf node counts to verify move generator completeness:
```bash
# Standard Perft count
python3 perft.py --depth 2

# Perft Divide: Displays child node counts broken down by root move
python3 perft.py --depth 2 --divide

# Combined Perft + Fuzz Testing
python3 perft.py --depth 2 --fuzz 50 --max-moves 200
```

### 8. Using as a Python Package
```python
from chaturanga import ChaturangaEnv, make_move, get_all_legal_moves

# Initialize environment
env = ChaturangaEnv()
env.render()

# Retrieve legal moves: [((r1, c1), (r2, c2)), ...]
legal_moves = env.get_legal_moves()
print(f"Available legal moves: {len(legal_moves)}")

# Execute a move: returns (new_board, outcome, done)
board, outcome, done = env.step(legal_moves[0])
env.render()
```

---

## Comprehensive Study Guide: How the System Works

### A. Board Representations (2D Matrix vs. 3D Tensor)

The project manages two representations of the board:

#### 1. Internal Engine Representation (2D Matrix)
Used by the rules engine for fast arithmetic, move generation, and state transitions.
* **Type**: `np.ndarray` of shape `(8, 8)`, `dtype=np.int8`.
* **Values**:
  * `0`: Empty square
  * `+1` to `+6`: White pieces (`PAWN=1`, `HORSE=2`, `ELEPHANT=3`, `CHARIOT=4`, `MINISTER=5`, `KING=6`)
  * `-1` to `-6`: Black pieces (`-PAWN=-1`, `-HORSE=-2`, etc.)

#### 2. Deep Learning / AlphaZero Representation (3D Tensor)
Used by the Convolutional Neural Network (CNN/ResNet) for Policy and Value prediction.
* **Shape**: $(13, 8, 8)$ (13 binary feature planes of $8 \times 8$).
* **Why not a single 2D matrix?** If piece values 1–6 are fed into a CNN, the convolutional filters perform multiplication and addition. The network would mistakenly assume an Elephant (`3`) is worth three Pawns (`1`), or that a King (`6`) equals a Horse (`2`) plus a Chariot (`4`). One-hot planes isolate each piece type into an independent spatial feature map.
* **Channel Breakdown**:
  * **Channels 0–5**: White pieces (Pawns, Horses, Elephants, Chariots, Ministers, King).
  * **Channels 6–11**: Black pieces (Pawns, Horses, Elephants, Chariots, Ministers, King).
  * **Channel 12**: **Turn Indicator Plane** (Entire $8 \times 8$ grid filled with `1.0` if White's turn, `0.0` if Black's turn).
* **Why a Turn Plane instead of a Boolean?** 2D Convolutions require all inputs to share the exact same spatial dimensions $(8, 8)$. By tiling the turn indicator across an $8 \times 8$ plane, the convolutional kernels can combine piece positions with tempo/turn at every square in the very first layer without requiring auxiliary inputs.

---

### B. Action Space Representation (4,096 Dimensions)

To keep the AlphaZero action space simple and computationally feasible on consumer hardware, actions are modeled as a **flattened from-square to to-square mapping**:

$$\text{Action Space Size} = 64 \times 64 = 4096$$

* Square index: `sq = r * 8 + c` (where $0 \le \text{sq} < 64$)
* Action index: `action = start_sq * 64 + end_sq` (where $0 \le \text{action} < 4096$)
* **Illegal Move Masking**: Out of 4,096 theoretical moves, typically only 15–35 are legal at any turn. Before computing softmax in the Policy Head, illegal action indices are masked with $-\infty$, guaranteeing that the network only samples strictly legal moves.

---

### C. Move Generation & Check Detection Pipeline

The move generation pipeline is split into two phases:

```
[ Board State ]
       │
       ▼
1. Pseudo-Legal Move Generation (chaturanga.moves)
   - Orthogonal raycasting (Chariots)
   - L-shaped leaping (Horses)
   - 2-square diagonal leaping (Elephants)
   - 1-square diagonal stepping (Ministers)
   - 8-direction stepping (Kings)
   - 1-step forward / diagonal capture (Pawns)
       │
       ▼
2. Legal Move Filtering (chaturanga.rules)
   - For each pseudo-legal move:
       simulated_board = make_move(board, start, end)
       if not is_in_check(simulated_board, player):
           keep move
       │
       ▼
[ Strictly Legal Moves ]
```

#### Fast Raycast Check Detection Optimization
In naive engines, `is_in_check` scans all 64 squares, generates full move lists for all enemy pieces, and checks if the King is attacked. This causes thousands of list allocations per turn.

In this engine, `is_in_check` looks **directly outward from the King's coordinates**:
1. Casts 4 orthogonal rays outward $\to$ if first piece hit is an enemy Chariot, return `True`.
2. Checks the 8 horse jump squares $\to$ if enemy Horse present, return `True`.
3. Checks the 4 elephant leap squares $\to$ if enemy Elephant present, return `True`.
4. Checks the 4 minister diagonal squares $\to$ if enemy Minister present, return `True`.
5. Checks the 8 adjacent squares $\to$ if enemy King present, return `True`.
6. Checks the 2 pawn capture squares $\to$ if enemy Pawn present, return `True`.

This reduces check detection from hundreds of loop iterations down to ~25 fast memory lookups, yielding a **50x speedup** across simulations.

---

### D. Dual-Layer Testing Philosophy (Perft vs. Fuzzing)

| Feature | Perft (Performance Test) | Random Playout Fuzz Testing |
|---|---|---|
| **Search Pattern** | **Exhaustive BFS/DFS**: Expands *every single* legal move. | **Monte Carlo Trajectory**: Picks *one random* legal move per step. |
| **Search Depth** | **Shallow** (depths 1–4). Exponential branching ($b \approx 16$) prevents deep searches. | **Deep** (100–300+ plies per game). |
| **What it Catches** | **Omission/Addition of moves**: Discrepancies in leap logic, square bounds, or pin detection. | **State corruption over time**: Pawn promotion failures, illegal captures, bare king trigger errors, or crashes. |
| **Output Type** | Deterministic leaf-node integer (e.g. Depth 1 = 16, Depth 2 = 256). | Pass/Fail assertions across 7 invariants + outcome statistics. |

---

### E. Roadmap to Milestone 2 & AlphaZero

* **Milestone 1 (Complete)**: Flawless rule engine, check detection, terminal conditions, unit tests, and browser UI.
* **Milestone 2 (Next)**: 
  * State tensor converter (`chaturanga/tensor.py`): $8 \times 8 \to (13, 8, 8)$ tensor.
  * Lightweight PyTorch ResNet (`chaturanga/model.py`): 4–6 residual blocks with dual heads (Policy: 4,096 logits, Value: scalar $\in [-1, 1]$).
  * PUCT Monte Carlo Tree Search (`chaturanga/mcts.py`): Guided tree expansion using network priors ($P(s, a)$) and node evaluations ($v$) without slow random rollouts.
* **Milestones 3–4**:
  * Self-play training pipeline and experience replay buffer.
  * Continuous loss optimization (MSE value loss + Cross-Entropy policy loss).
  * Arena benchmarking against baseline players.
