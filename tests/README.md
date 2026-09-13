# ChaturangZero Test Suite (Milestone 1)

This directory is designated for the automated unit testing suite for the Chaturanga rules engine.

## Objective for Teammate (Milestone 1)
Verify 100% accuracy of the rules engine under the **Murray/Golombek ruleset** before AI/MCTS development begins.

### Key Areas to Test:
1. **Piece Movement & Leaps**:
   - **Foot Soldier (*Padāti*)**: 1-square forward step, diagonal captures, no double-step or en passant.
   - **Horse (*Ashwa*)**: L-shaped leaps jumping over any intervening pieces.
   - **Elephant (*Gaja*)**: Strict 2-square diagonal leaps jumping over intervening squares.
   - **Chariot (*Ratha*)**: Orthogonal rays, correct obstruction and capture termination.
   - **Minister (*Mantri*)**: 1-square diagonal moves.
   - **King (*Rāja*)**: 1-square steps in all 8 directions.
2. **Pawn Promotion**:
   - Automatic promotion strictly to Minister (*Mantri*) at row 0 (White) and row 7 (Black).
3. **King Check & Pins**:
   - In-check detection across all attacking piece types.
   - Check resolution via blocking, capture, or King evasion.
   - Absolute pins: Pinned pieces must not make moves that expose their King.
4. **Terminal State Outcomes**:
   - **Bare King**: Immediate win (+1) for the player who bares the opponent's King.
   - **Checkmate**: Attacking player wins (+1), mated player loses (-1).
   - **Stalemate**: Stalemated player **WINS (+1)** per Murray/Golombek rules (not a draw).
5. **Perft (Performance / Move Generation Correctness)**:
   - Recursive leaf-node enumeration from starting board (Depth 1 = 16 nodes, Depth 2 = 256 nodes, etc.).

### Running Tests
Tests can be authored using standard Python `unittest` or `pytest`:
```bash
python3 -m unittest discover -v tests
# or
pytest -v
```
