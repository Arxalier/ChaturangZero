# ChaturangZero: Self-Play AlphaZero Agent for Chaturanga

ChaturangZero is an artificial intelligence project aiming to train an AlphaZero-style reinforcement learning agent for **Chaturanga**, the ancient Indian ancestor of Chess, Shogi, and Makruk, played on an 8x8 uncheckered grid (the *Ashtapada*).

This repository implements the **Murray/Golombek ruleset** with complete test coverage, modular architecture, and support for future MCTS and deep neural network integration.

---

## The Rules of Chaturanga (Murray/Golombek Ruleset)

Note: The source used to study the Murray/Golombek Ruleset is https://www.chessvariants.com/historic.dir/chaturanga.html

### 1. The Board & Setup
- **Ashtapada**: An 8x8 uncheckered board.
- **Symmetry**: 16 pieces per player mirroring each other.
  - The **Kings (*Rāja*)** face each other on the e-file (col 4).
  - The **Ministers (*Mantri*)** face each other on the d-file (col 3).

### 2. Piece Movement & Sanskrit Names

| English Name | Sanskrit Name | Movement Mechanics | Modern Equivalent |
|---|---|---|---|
| **King** | *Rāja* | 1 square in any direction (orthogonal or diagonal). Cannot move into check. No castling. | King |
| **Minister** | *Mantri* | Exactly 1 square diagonally in any of the 4 directions. | Queen (restricted) |
| **Elephant** | *Gaja / Hasti* | Exactly 2 squares diagonally, leaping over intervening pieces. | Alfil |
| **Horse** | *Ashwa* | L-shaped leap (2x3 rectangle), jumping over intervening pieces. | Knight |
| **Chariot** | *Ratha* | Orthogonal slider along ranks and files. | Rook |
| **Foot Soldier** | *Padāti* | Moves 1 square forward, captures 1 square diagonally forward. No initial double-step, no en passant. | Pawn |

### 3. Automatic Pawn Promotion
When a Foot Soldier (*Padāti*) reaches the furthest rank on the opponent's side (rank 0 for White, rank 7 for Black), it **promotes exclusively and automatically to a Minister (*Mantri*)**.

### 4. Terminal States & Reward Triggers
The game ends immediately upon meeting any of the following conditions:
- **Checkmate (+1 Win / -1 Loss)**: The opponent's King is under check and has no legal move to escape, block, or capture the attacker.
- **Baring the King (+1 Win / -1 Loss)**: A player captures the opponent's last remaining non-King piece, leaving the opponent's King isolated on the board.
- **Stalemate (+1 Win / -1 Loss)**: A player is **not** in check but has zero legal moves. Per Murray/Golombek rules, **the stalemated player wins the game (+1)**!

---

## Repository Layout

```
ChaturangZero/
├── chaturanga/              # Core game engine package
│   ├── __init__.py          # Package exports
│   ├── pieces.py            # Piece constants, player indicators, Sanskrit aliases
│   ├── moves.py             # Pseudo-legal move generators for each piece type
│   ├── board.py             # Board creation, make_move state transition, rendering
│   ├── rules.py             # Check detection, legal moves, game outcomes, perft
│   └── env.py               # ChaturangaEnv environment class (RL/Gym wrapper)
├── tests/                   # Automated unit test suite (pytest)
│   ├── __init__.py
│   ├── test_moves.py        # Move tests for all piece types & leaps
│   ├── test_rules.py        # Check, promotion, bare king, stalemate tests
│   └── test_perft.py        # Perft node count verification
├── ui/                      # Browser UI for demos (stdlib server + static page)
│   ├── server.py            # JSON API over the engine, no extra dependencies
│   ├── engine_api.py        # Engine output -> JSON, plus an engine call trace
│   ├── scenarios.py         # Seven preset demo positions
│   └── static/              # index.html, style.css, app.js
├── play.py                  # Interactive CLI play loop
├── perft.py                 # CLI perft performance benchmark tool
├── requirements.txt         # Package dependencies
├── .gitignore               # Standard Python gitignore
├── milestone1.txt           # Milestone 1 rules reference
└── projectDetails.txt       # Full project overview and roadmap
```

---

## Quickstart

### Installation
Ensure Python 3.8+ is installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### Interactive Play
Play manually from the terminal:
```bash
python play.py
```
Move format: `start_row,start_col end_row,end_col` (e.g. `6,4 5,4`).
Type `perft <depth>` during the game to run tree calculations, or `q` to quit.

### Visual Engine Explorer (presentation UI)
Launch a browser UI that exposes every part of the engine built so far — board, legal
moves, pseudo-legal vs legal filtering, check and threat maps, terminal states, perft,
and the raw int8 board tensor:
```bash
python ui/server.py
```
It opens `http://127.0.0.1:8000`. The page is drawing only: every rule decision comes
from the `chaturanga` package over a small JSON API, so the UI cannot disagree with the
engine. It needs no dependencies beyond the `numpy` the engine already uses.
Press <kbd>1</kbd>–<kbd>7</kbd> for preset demo positions (checkmate in 1, stalemate,
baring the King, an absolute pin, promotion, leapers in a crowd). See
[ui/README.md](ui/README.md) for deep links and shortcuts.

### Running Automated Tests
Run the complete test suite with `pytest`:
```bash
pytest -v
```

### Running Perft Benchmarks
Verify the move generator across different depths:
```bash
python perft.py --depth 2
```

### Using as a Python Package
```python
from chaturanga import ChaturangaEnv, make_move, get_all_legal_moves

env = ChaturangaEnv()
env.render()

# Retrieve legal moves: [((r1, c1), (r2, c2)), ...]
legal_moves = env.get_legal_moves()
print(f"Legal moves: {len(legal_moves)}")

# Execute a move
env.step(legal_moves[0])
env.render()
```
