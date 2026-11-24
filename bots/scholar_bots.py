"""
bots.scholar_bots
-----------------
Heuristic baseline bots for ChaturangZero with distinct playstyles.
All bots inherit from chaturanga.BasePlayer and implement get_move(board, player).
"""

import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from chaturanga import (
    BasePlayer,
    Move,
    WHITE,
    BLACK,
    EMPTY,
    PAWN,
    HORSE,
    ELEPHANT,
    CHARIOT,
    MINISTER,
    KING,
    make_move,
    find_king,
    is_in_check,
    get_elephant_moves,
    get_all_legal_moves,
    get_game_outcome,
)

# Material weights matching breakdown.md (ChanakyaBot & greedy capture baseline)
PIECE_VALUES: Dict[int, float] = {
    PAWN: 1.0,
    HORSE: 3.0,
    ELEPHANT: 3.0,
    CHARIOT: 4.0,
    MINISTER: 5.0,
    KING: 1000.0,
}

TERMINAL_WIN_SCORE = 100_000.0


def _order_moves(board: np.ndarray, moves: List[Move]) -> List[Move]:
    """
    Orders moves to maximize alpha-beta cutoffs:
    captures first using MVV-LVA (high-value victim, low-value attacker),
    then promotions, then quiet moves.
    """
    def score_move(move: Move) -> float:
        (sr, sc), (er, ec) = move
        attacker = abs(int(board[sr, sc]))
        victim = abs(int(board[er, ec]))
        score = 0.0
        if victim != EMPTY:
            score += 10.0 * PIECE_VALUES.get(victim, 0.0) - PIECE_VALUES.get(attacker, 0.0)
        if attacker == PAWN and (er == 0 or er == 7):
            score += 4.0  # Promotion to Mantri
        return score

    return sorted(moves, key=score_move, reverse=True)


class _AlphaBetaBot(BasePlayer):
    """
    Shared Negamax + Alpha-Beta pruning search engine for heuristic Scholar Bots.

    Subclasses override evaluate(board, player) to return a heuristic score
    from `player`'s perspective (positive = good for `player`).
    """

    def __init__(
        self,
        depth: int = 2,
        tie_break_noise: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.depth = depth
        self.tie_break_noise = tie_break_noise
        self._rng = random.Random(seed)

    def reset(self) -> None:
        """Optional lifecycle hook for tournament / UI resets."""
        pass

    def evaluate(self, board: np.ndarray, player: int) -> float:
        """Heuristic evaluation from `player`'s perspective."""
        raise NotImplementedError

    def _negamax(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        player: int,
        ply: int,
    ) -> float:
        legal_moves = get_all_legal_moves(board, player)
        outcome = get_game_outcome(board, player, legal_moves=legal_moves)

        # Murray/Golombek terminal handling (Checkmate, Bare King, Stalemate-wins):
        # outcome is +1 if `player` wins (including being stalemated), -1 if `player` loses.
        if outcome is not None:
            if outcome == 1:
                return TERMINAL_WIN_SCORE - ply
            if outcome == -1:
                return -TERMINAL_WIN_SCORE + ply
            return 0.0

        if depth == 0:
            return self.evaluate(board, player)

        best_val = -float("inf")
        for move in _order_moves(board, legal_moves):
            next_board = make_move(board, move[0], move[1])
            val = -self._negamax(
                next_board, depth - 1, -beta, -alpha, -player, ply + 1
            )
            if val > best_val:
                best_val = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        return best_val

    def get_move(self, board: np.ndarray, player: int) -> Move:
        legal_moves = get_all_legal_moves(board, player)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        if len(legal_moves) == 1:
            return legal_moves[0]

        ordered_moves = _order_moves(board, legal_moves)
        alpha = -float("inf")
        beta = float("inf")

        best_move = ordered_moves[0]
        best_score = -float("inf")

        for move in ordered_moves:
            next_board = make_move(board, move[0], move[1])
            score = -self._negamax(
                next_board, self.depth - 1, -beta, -alpha, -player, ply=1
            )

            # Tiny jitter prevents deterministic shuffling loops on equal moves
            # without ever overriding real tactical or terminal score differences.
            if self.tie_break_noise > 0.0 and abs(score) < TERMINAL_WIN_SCORE * 0.5:
                score += self._rng.uniform(-self.tie_break_noise, self.tie_break_noise)

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_move


class NoviceBot(BasePlayer):
    """
    Baseline player: 80% random legal move, 20% direct greedy capture.
    """

    def __init__(
        self,
        capture_prob: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.capture_prob = capture_prob
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def get_move(self, board: np.ndarray, player: int) -> Move:
        legal_moves = get_all_legal_moves(board, player)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if self._rng.random() < self.capture_prob:
            captures = [
                move for move in legal_moves
                if int(board[move[1][0], move[1][1]]) != EMPTY
            ]
            if captures:
                best_val = max(
                    PIECE_VALUES.get(abs(int(board[m[1][0], m[1][1]])), 0.0)
                    for m in captures
                )
                top_captures = [
                    m for m in captures
                    if PIECE_VALUES.get(abs(int(board[m[1][0], m[1][1]])), 0.0) == best_val
                ]
                return self._rng.choice(top_captures)

        return self._rng.choice(legal_moves)

class ChanakyaBot(_AlphaBetaBot):
    """
    Aggressive Tactician (Depth-2 Alpha-Beta Minimax).
    Weights piece values heavily with bonuses for giving check, eliminating
    opponent material, hunting the enemy Raja, and driving toward Bare King.
    """

    def __init__(
        self,
        depth: int = 2,
        tie_break_noise: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(depth=depth, tie_break_noise=tie_break_noise, seed=seed)

    def evaluate(self, board: np.ndarray, player: int) -> float:
        score = 0.0
        my_non_king = 0
        opp_non_king = 0

        # 1. Heavy material evaluation (King:1000, Minister:5, Chariot:4, Horse:3, Elephant:3, Pawn:1)
        for p_type, val in PIECE_VALUES.items():
            my_cnt = int(np.count_nonzero(board == p_type * player))
            opp_cnt = int(np.count_nonzero(board == -p_type * player))
            score += val * (my_cnt - opp_cnt)
            if p_type != KING:
                my_non_king += my_cnt
                opp_non_king += opp_cnt

        # 2. Aggression / capture pressure: rewards trading down & removing enemy pieces
        score += 0.35 * (15 - opp_non_king) - 0.15 * (15 - my_non_king)

        # 3. Bare King pressure: steep bonus as opponent nears 0 non-King pieces
        if 0 < opp_non_king <= 4:
            score += 6.0 / opp_non_king
        if 0 < my_non_king <= 4:
            score -= 6.0 / my_non_king

        # 4. Check bonuses / penalties
        if is_in_check(board, -player):
            score += 1.5
        if is_in_check(board, player):
            score -= 1.5

        # 5. King-hunt pressure: nudge non-King pieces toward the enemy Raja
        opp_king = find_king(board, -player)
        if opp_king is not None:
            kr, kc = opp_king
            my_coords = np.argwhere((board * player > 0) & (np.abs(board) != KING))
            if len(my_coords) > 0:
                dists = np.abs(my_coords[:, 0] - kr) + np.abs(my_coords[:, 1] - kc)
                score += float(np.sum(14 - dists)) * 0.04

        return score

# ---------------------------------------------------------------------------
# Piece-Square Tables (White's perspective: row 7 = home rank, row 0 = enemy rank)
# Black's tables are vertically mirrored via [::-1].
# ---------------------------------------------------------------------------

_CENTER_PST = np.array([
    [0.00, 0.05, 0.10, 0.15, 0.15, 0.10, 0.05, 0.00],
    [0.05, 0.15, 0.25, 0.30, 0.30, 0.25, 0.15, 0.05],
    [0.10, 0.25, 0.45, 0.55, 0.55, 0.45, 0.25, 0.10],
    [0.15, 0.30, 0.55, 0.85, 0.85, 0.55, 0.30, 0.15],
    [0.15, 0.30, 0.55, 0.85, 0.85, 0.55, 0.30, 0.15],
    [0.10, 0.25, 0.45, 0.55, 0.55, 0.45, 0.25, 0.10],
    [0.05, 0.15, 0.25, 0.30, 0.30, 0.25, 0.15, 0.05],
    [0.00, 0.05, 0.10, 0.15, 0.15, 0.10, 0.05, 0.00],
], dtype=np.float32)

_PAWN_PST_WHITE = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Row 0 (promotes to Mantri)
    [0.85, 0.90, 0.98, 1.05, 1.05, 0.98, 0.90, 0.85],  # 1 step from promotion
    [0.55, 0.62, 0.72, 0.82, 0.82, 0.72, 0.62, 0.55],
    [0.32, 0.40, 0.52, 0.65, 0.65, 0.52, 0.40, 0.32],  # Central ranks
    [0.18, 0.24, 0.35, 0.48, 0.48, 0.35, 0.24, 0.18],
    [0.08, 0.12, 0.18, 0.25, 0.25, 0.18, 0.12, 0.08],
    [0.00, 0.00, 0.05, 0.08, 0.08, 0.05, 0.00, 0.00],  # Starting rank
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
], dtype=np.float32)

_ELEPHANT_PST_WHITE = np.array([
    [0.00, 0.00, 0.10, 0.10, 0.10, 0.10, 0.00, 0.00],
    [0.00, 0.10, 0.20, 0.25, 0.25, 0.20, 0.10, 0.00],
    [0.10, 0.20, 0.55, 0.60, 0.60, 0.55, 0.20, 0.10],
    [0.10, 0.25, 0.60, 0.80, 0.80, 0.60, 0.25, 0.10],
    [0.10, 0.25, 0.60, 0.80, 0.80, 0.60, 0.25, 0.10],
    [0.10, 0.20, 0.55, 0.60, 0.60, 0.55, 0.20, 0.10],
    [0.00, 0.10, 0.20, 0.25, 0.25, 0.20, 0.10, 0.00],
    [0.00, 0.00, 0.10, 0.10, 0.10, 0.10, 0.00, 0.00],
], dtype=np.float32)

_KING_PST_WHITE = np.array([
    [-0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.40],
    [-0.30, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30],
    [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
    [-0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20],
    [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15],
    [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
    [ 0.10,  0.15,  0.15,  0.15,  0.15,  0.15,  0.15,  0.10],
    [ 0.20,  0.25,  0.25,  0.30,  0.30,  0.25,  0.25,  0.20],
], dtype=np.float32)

_PST_WHITE: Dict[int, np.ndarray] = {
    PAWN: _PAWN_PST_WHITE,
    HORSE: _CENTER_PST * 1.15,
    ELEPHANT: _ELEPHANT_PST_WHITE,
    CHARIOT: _CENTER_PST * 0.70,
    MINISTER: _CENTER_PST * 1.00,
    KING: _KING_PST_WHITE,
}

_PST_BLACK: Dict[int, np.ndarray] = {
    p_type: table[::-1].copy() for p_type, table in _PST_WHITE.items()
}


class AryabhataBot(_AlphaBetaBot):
    """
    Positional Sage (Depth-2 Alpha-Beta Minimax with Piece-Square Tables).
    Favors central control (rows 3-4, cols 3-4), Padati advancement toward
    Mantri promotion, and Gaja (Elephant) 2-square diagonal leaping mobility.
    """

    # Balanced Chaturanga material values alongside strong positional weighting
    _MATERIAL: Dict[int, float] = {
        PAWN: 1.0,
        ELEPHANT: 2.2,
        MINISTER: 2.6,
        HORSE: 3.2,
        CHARIOT: 5.0,
        KING: 1000.0,
    }

    def __init__(
        self,
        depth: int = 2,
        tie_break_noise: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(depth=depth, tie_break_noise=tie_break_noise, seed=seed)

    def evaluate(self, board: np.ndarray, player: int) -> float:
        score = 0.0

        # 1. Material + Piece-Square Table (central control & pawn advancement)
        for p_type, val in self._MATERIAL.items():
            w_mask = board == p_type
            b_mask = board == -p_type

            w_score = val * int(np.count_nonzero(w_mask)) + float(_PST_WHITE[p_type][w_mask].sum())
            b_score = val * int(np.count_nonzero(b_mask)) + float(_PST_BLACK[p_type][b_mask].sum())

            score += (w_score - b_score) * player

        # 2. Elephant (Gaja) leaping mobility bonus
        for r, c in np.argwhere(board == ELEPHANT * player):
            score += 0.25 * len(get_elephant_moves(board, int(r), int(c), player))
        for r, c in np.argwhere(board == -ELEPHANT * player):
            score -= 0.25 * len(get_elephant_moves(board, int(r), int(c), -player))

        return score

# ---------------------------------------------------------------------------
# Fast single-pass attack/defense maps for PaniniBot
# ---------------------------------------------------------------------------

_ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_KING_DIRS = _ORTHO_DIRS + _DIAG_DIRS
_HORSE_JUMPS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
_ELEPHANT_LEAPS = ((-2, -2), (-2, 2), (2, -2), (2, 2))


def _compute_attack_maps(board: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Computes 8x8 square-control counts for White and Black in a single pass.
    Includes friendly-occupied squares so piece protection (defense) is counted.
    """
    w_atk = [[0] * 8 for _ in range(8)]
    b_atk = [[0] * 8 for _ in range(8)]
    grid = board.tolist()

    for r in range(8):
        row = grid[r]
        for c in range(8):
            val = row[c]
            if val == 0:
                continue

            if val > 0:
                atk = w_atk
                p_type = val
                fwd = -1
            else:
                atk = b_atk
                p_type = -val
                fwd = 1

            if p_type == PAWN:
                nr = r + fwd
                if 0 <= nr < 8:
                    if c > 0:
                        atk[nr][c - 1] += 1
                    if c < 7:
                        atk[nr][c + 1] += 1
            elif p_type == HORSE:
                for dr, dc in _HORSE_JUMPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        atk[nr][nc] += 1
            elif p_type == ELEPHANT:
                for dr, dc in _ELEPHANT_LEAPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        atk[nr][nc] += 1
            elif p_type == MINISTER:
                for dr, dc in _DIAG_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        atk[nr][nc] += 1
            elif p_type == KING:
                for dr, dc in _KING_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        atk[nr][nc] += 1
            elif p_type == CHARIOT:
                for dr, dc in _ORTHO_DIRS:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 8 and 0 <= nc < 8:
                        atk[nr][nc] += 1
                        if grid[nr][nc] != 0:
                            break
                        nr += dr
                        nc += dc

    return w_atk, b_atk


class PaniniBot(_AlphaBetaBot):
    """
    Defensive / Solid Strategist (Depth-2 Alpha-Beta Minimax).
    Applies strong defensive penalties for leaving friendly pieces undefended
    or hanging, and for exposing the Raja (King) without adjacent shelter.
    """

    _MATERIAL: Dict[int, float] = {
        PAWN: 1.0,
        ELEPHANT: 2.5,
        MINISTER: 3.0,
        HORSE: 3.0,
        CHARIOT: 4.8,
        KING: 1000.0,
    }

    def __init__(
        self,
        depth: int = 2,
        tie_break_noise: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(depth=depth, tie_break_noise=tie_break_noise, seed=seed)

    def _side_safety_score(
        self,
        grid: List[List[int]],
        side: int,
        own_atk: List[List[int]],
        opp_atk: List[List[int]],
    ) -> float:
        score = 0.0
        king_pos: Optional[Tuple[int, int]] = None

        for r in range(8):
            row = grid[r]
            for c in range(8):
                val = row[c]
                if val * side <= 0:
                    continue

                p_type = abs(val)
                pval = self._MATERIAL[p_type]
                score += pval

                if p_type == KING:
                    king_pos = (r, c)
                    continue

                defenders = own_atk[r][c]
                attackers = opp_atk[r][c]

                # Penalty for loose / undefended or hanging pieces
                if defenders == 0:
                    score -= 0.22 * pval
                    if attackers > 0:
                        score -= 0.65 * pval
                elif attackers > defenders:
                    score -= 0.35 * pval
                else:
                    # Small solidity bonus for mutually defended pieces
                    score += 0.10 * min(defenders, 2)

        # King safety: home-rank shelter, adjacent bodyguards, and enemy zone pressure
        if king_pos is not None:
            kr, kc = king_pos

            # Penalize marching the Raja into open central/enemy ranks
            exposure = max(0, 6 - kr) if side == WHITE else max(0, kr - 1)
            score -= 0.45 * exposure

            # Penalize being in check directly
            if opp_atk[kr][kc] > 0:
                score -= 1.50

            # Count friendly shield pieces and enemy pressure around the Raja
            shield_count = 0
            zone_pressure = 0
            for dr, dc in _KING_DIRS:
                nr, nc = kr + dr, kc + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if grid[nr][nc] * side > 0:
                        shield_count += 1
                    zone_pressure += opp_atk[nr][nc]

            if shield_count == 0:
                score -= 0.85
            else:
                score += 0.30 * min(shield_count, 3)

            score -= 0.25 * zone_pressure

        return score

    def evaluate(self, board: np.ndarray, player: int) -> float:
        w_atk, b_atk = _compute_attack_maps(board)
        grid = board.tolist()

        w_score = self._side_safety_score(grid, WHITE, w_atk, b_atk)
        b_score = self._side_safety_score(grid, BLACK, b_atk, w_atk)
        return (w_score - b_score) * player