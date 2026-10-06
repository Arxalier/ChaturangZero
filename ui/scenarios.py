"""
ui.scenarios
------------
Hand-built demo positions for the presentation UI.

Boards are written as 8 rows of 8 characters using the same letters as
`chaturanga.pieces.PIECE_SYMBOLS` (uppercase = White, lowercase = Black,
'.' = empty), then parsed into the int8 arrays the engine expects.
"""

from typing import Dict, List, Tuple

import numpy as np

from chaturanga import (
    EMPTY, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING, WHITE, BLACK,
    create_starting_board,
)

CHAR_TO_PIECE = {
    '.': EMPTY,
    'P': PAWN, 'H': HORSE, 'E': ELEPHANT, 'R': CHARIOT, 'M': MINISTER, 'K': KING,
    'p': -PAWN, 'h': -HORSE, 'e': -ELEPHANT, 'r': -CHARIOT, 'm': -MINISTER, 'k': -KING,
}


def parse(rows: List[str]) -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            board[r, c] = CHAR_TO_PIECE[ch]
    return board


# Each scenario: id -> (label, subtitle, rows, player to move, teaching note)
SCENARIOS: Dict[str, Dict] = {
    "opening": {
        "label": "Opening position",
        "subtitle": "Murray/Golombek setup",
        "rows": None,  # built by create_starting_board()
        "player": WHITE,
        "note": "Kings face each other on the e-file, Ministers on the d-file. "
                "16 legal moves at depth 1 - every Padati step, since no other piece "
                "can leave the back rank yet.",
    },
    "checkmate": {
        "label": "Checkmate in 1",
        "subtitle": "White to play: Rf8 mates",
        "rows": [
            "k.......",
            ".......R",
            "........",
            "p.......",
            "P.......",
            "........",
            ".....R..",
            "....K...",
        ],
        "player": WHITE,
        "note": "The Chariot on h7 seals rank 7. Play f2-f8: the Raja on a8 is in check "
                "with no escape, so rules.get_game_outcome returns -1 for Black. "
                "Black's frozen Padati on a5 is there on purpose - without it the Bare King "
                "rule would fire first.",
    },
    "stalemate": {
        "label": "Stalemate (Black wins!)",
        "subtitle": "Black to play, not in check, no moves",
        "rows": [
            "k.......",
            "........",
            "HPM.....",
            "p.......",
            "P.......",
            "........",
            "........",
            "....K...",
        ],
        "player": BLACK,
        "note": "Black's Raja is NOT attacked, yet every square around it is covered and the "
                "Padati on a5 is blocked. Zero legal moves without check: under "
                "Murray/Golombek the stalemated player WINS, so get_game_outcome returns "
                "+1 for Black. This is the biggest departure from modern chess.",
    },
    "bareking": {
        "label": "Baring the King",
        "subtitle": "White to play: Rxe4 ends it",
        "rows": [
            "k.......",
            "........",
            "........",
            "........",
            "....p..R",
            "........",
            "........",
            "....K...",
        ],
        "player": WHITE,
        "note": "Capture the last black piece other than the Raja (h4xe4) and Black is left "
                "bare: instant +1 for White, no mate needed. This terminal state is why "
                "self-play rollouts stay short.",
    },
    "pin": {
        "label": "Absolute pin",
        "subtitle": "Pseudo-legal vs legal",
        "rows": [
            "k...r...",
            "p.......",
            "P.......",
            "........",
            "........",
            "........",
            "....H...",
            "....K...",
        ],
        "player": WHITE,
        "note": "Select the Ashwa on e2 and switch on 'pseudo-legal'. "
                "chaturanga.moves.get_horse_moves offers it several leaps; "
                "rules.get_all_legal_moves deletes every one of them, because each exposes "
                "the Raja to the Chariot on e8.",
    },
    "promotion": {
        "label": "Promotion to Mantri",
        "subtitle": "White Padati one step from rank 8",
        "rows": [
            "k.......",
            "...P....",
            "........",
            "........",
            "........",
            "........",
            "....p...",
            "....K...",
        ],
        "player": WHITE,
        "note": "Push d7-d8: board.make_move rewrites the Padati as a Minister automatically. "
                "There is no choice of promotion piece in Chaturanga. The black Padati on e2 "
                "promotes the same way on rank 1.",
    },
    "leapers": {
        "label": "Leapers in a crowd",
        "subtitle": "Gaja and Ashwa jump; Ratha does not",
        "rows": [
            "k.......",
            "........",
            "..ppp...",
            "..pEp...",
            "..ppp...",
            "......R.",
            ".....ppp",
            "K.....p.",
        ],
        "player": WHITE,
        "note": "The Gaja on d5 is walled in, yet it still has its four 2-square diagonal "
                "leaps - get_elephant_moves never looks at the intervening square. "
                "The Chariot on g3, by contrast, stops the moment it meets a piece.",
    },
}


def build(scenario_id: str) -> Tuple[np.ndarray, int, str, str]:
    """Returns (board, player_to_move, label, note) for a scenario id."""
    spec = SCENARIOS[scenario_id]
    board = create_starting_board() if spec["rows"] is None else parse(spec["rows"])
    return board, spec["player"], spec["label"], spec["note"]


def catalogue() -> List[Dict]:
    """Scenario metadata for the UI picker."""
    return [
        {"id": key, "label": spec["label"], "subtitle": spec["subtitle"]}
        for key, spec in SCENARIOS.items()
    ]
