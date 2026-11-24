"""
arena.match
-----------
Plays one game between two BasePlayers and records why it ended.

The stopping rules match training/self_play.py exactly, so arena results and
self-play statistics mean the same thing:

    * Checkmate, stalemate (the stalemated side WINS) and bare king come from
      chaturanga.get_game_outcome.
    * A position (board + side to move) seen for the third time is a draw.
    * Reaching ``max_plies`` half-moves is a draw (train.py default: 120).

Players that keep per-game state get ``reset()`` before each game, and players
whose ``get_move`` accepts ``past_position_counts`` receive the game's position
counts, the same history self-play gives MCTS.

A player that returns an illegal move or raises an exception loses the game;
the reason is recorded so bugs in bots show up in the tables instead of
crashing a long tournament.
"""

from __future__ import annotations

import inspect
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chaturanga import (
    BasePlayer,
    WHITE,
    create_starting_board,
    get_all_legal_moves,
    get_game_outcome,
    is_in_check,
    make_move,
)

Move = Tuple[Tuple[int, int], Tuple[int, int]]

_TAKES_HISTORY: Dict[type, bool] = {}


def reset_player(agent: BasePlayer) -> None:
    """Clear any per-game state a player keeps (e.g. ChaturangPlayer's visited positions)."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


def takes_history(agent: BasePlayer) -> bool:
    """True if agent.get_move accepts ``past_position_counts`` (repetition-aware search)."""
    cls = type(agent)
    if cls not in _TAKES_HISTORY:
        try:
            params = inspect.signature(agent.get_move).parameters
            _TAKES_HISTORY[cls] = "past_position_counts" in params
        except (TypeError, ValueError):
            _TAKES_HISTORY[cls] = False
    return _TAKES_HISTORY[cls]

DEFAULT_MAX_PLIES = 120

# Reasons a game can end
CHECKMATE = "checkmate"
STALEMATE = "stalemate"
BARE_KING = "bare_king"
REPETITION = "repetition"
MAX_PLIES = "max_plies"
ILLEGAL_MOVE = "illegal_move"
CRASH = "crash"
DECISIVE_REASONS = (CHECKMATE, STALEMATE, BARE_KING, ILLEGAL_MOVE, CRASH)
DRAW_REASONS = (REPETITION, MAX_PLIES)


@dataclass
class GameResult:
    white: str
    black: str
    winner: int              # +1 White won, -1 Black won, 0 draw
    reason: str
    plies: int
    moves: List[Move] = field(default_factory=list)
    opening_plies: int = 0
    seed: Optional[int] = None
    think_time: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def score_for(self, label: str) -> float:
        """1 for a win, 0.5 for a draw, 0 for a loss (from ``label``'s side)."""
        if self.winner == 0:
            return 0.5
        won_as = self.white if self.winner == WHITE else self.black
        return 1.0 if won_as == label else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["moves"] = [[list(a), list(b)] for a, b in self.moves]
        return d

    @staticmethod
    def from_dict(d: dict) -> "GameResult":
        d = dict(d)
        d["moves"] = [(tuple(a), tuple(b)) for a, b in d.get("moves", [])]
        return GameResult(**d)


def classify_terminal(board: np.ndarray, player: int, legal_moves: List[Move]) -> Optional[Tuple[int, str]]:
    """
    Returns ``(winner, reason)`` if ``player`` (side to move) faces a terminal
    position, else None. ``winner`` is +1/-1 for White/Black.
    """
    outcome = get_game_outcome(board, player, legal_moves)
    if outcome is None:
        return None
    winner = player if outcome == 1 else -player
    own = int(np.count_nonzero(board * player > 0))
    enemy = int(np.count_nonzero(board * player < 0))
    if enemy == 1 or own == 1:
        reason = BARE_KING
    elif not legal_moves and is_in_check(board, player):
        reason = CHECKMATE
    else:
        reason = STALEMATE
    return winner, reason


def random_opening(num_plies: int, rng: random.Random) -> List[Move]:
    """A short random opening used to diversify games between deterministic players."""
    board = create_starting_board()
    player = WHITE
    moves: List[Move] = []
    for _ in range(num_plies):
        legal = get_all_legal_moves(board, player)
        if not legal or get_game_outcome(board, player, legal) is not None:
            break
        move = rng.choice(legal)
        moves.append(move)
        board = make_move(board, move[0], move[1])
        player = -player
    return moves


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def play_game(
    white: BasePlayer,
    black: BasePlayer,
    white_label: str = "white",
    black_label: str = "black",
    max_plies: int = DEFAULT_MAX_PLIES,
    opening: Sequence[Move] = (),
    seed: Optional[int] = None,
    start_board: Optional[np.ndarray] = None,
    start_player: int = WHITE,
) -> GameResult:
    """Play one game to completion and return a GameResult."""
    if seed is not None:
        seed_everything(seed)

    board = create_starting_board() if start_board is None else start_board.copy()
    player = start_player
    players = {WHITE: (white, white_label), -WHITE: (black, black_label)}
    moves: List[Move] = []
    seen: Dict[bytes, int] = {}
    think = {white_label: 0.0, black_label: 0.0}
    opening = list(opening)
    reset_player(white)
    if black is not white:
        reset_player(black)

    def finish(winner: int, reason: str, error: Optional[str] = None) -> GameResult:
        return GameResult(white_label, black_label, winner, reason, len(moves), moves,
                          opening_plies=min(len(opening), len(moves)), seed=seed,
                          think_time=think, error=error)

    while len(moves) < max_plies:
        key = board.tobytes() + (b"\x01" if player == WHITE else b"\x00")
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return finish(0, REPETITION)

        legal = get_all_legal_moves(board, player)
        terminal = classify_terminal(board, player, legal)
        if terminal is not None:
            return finish(*terminal)

        ply = len(moves)
        if ply < len(opening):
            move = tuple(map(tuple, opening[ply]))
        else:
            agent, label = players[player]
            t0 = time.perf_counter()
            try:
                if takes_history(agent):
                    # Real game history, so the search can see threefold repetition.
                    move = agent.get_move(board.copy(), player, past_position_counts=dict(seen))
                else:
                    move = agent.get_move(board.copy(), player)
                move = (tuple(move[0]), tuple(move[1]))
            except Exception:  # a crashing bot loses, the tournament continues
                think[label] += time.perf_counter() - t0
                return finish(-player, CRASH, traceback.format_exc(limit=3))
            think[label] += time.perf_counter() - t0

        if move not in legal:
            return finish(-player, ILLEGAL_MOVE, f"{players[player][1]} played {move}")

        moves.append(move)
        board = make_move(board, move[0], move[1])
        player = -player

    # Ply cap reached: the final position may still be terminal.
    legal = get_all_legal_moves(board, player)
    terminal = classify_terminal(board, player, legal)
    if terminal is not None:
        return finish(*terminal)
    return finish(0, MAX_PLIES)
