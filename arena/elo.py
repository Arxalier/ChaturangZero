"""
arena.elo
---------
Elo ratings from a set of games.

Elo is a *relative* scale: ratings only mean something against the other
players in the same pool. Every rating here is therefore reported relative to
an anchor player (by default ``Random`` fixed at 0), with a bootstrap 95%
confidence interval.

Method: Bradley-Terry maximum likelihood, fitted with Hunter's (2004) MM
algorithm. A draw counts as half a win for each side. Each pair of players
that met also receives ``prior_draws`` virtual draws, which keeps ratings
finite when one player wins every game (otherwise the MLE is +infinity) and
gently shrinks ratings that rest on few games. Elo = 400 * log10(strength).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ELO_SCALE = 400.0 / math.log(10.0)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability-weighted score of A against B under the Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_score(score: float, eps: float = 1e-3) -> float:
    """Elo difference implied by an average score in [0, 1] (clipped)."""
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def pair_matrix(players: Sequence[str], games: Iterable[Tuple[str, str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build score and game-count matrices from (a, b, score_of_a) triples.

    Returns:
        S[i, j]: total points player i scored against player j
        N[i, j]: games played between i and j
    """
    idx = {p: k for k, p in enumerate(players)}
    n = len(players)
    S = np.zeros((n, n))
    N = np.zeros((n, n))
    for a, b, score in games:
        i, j = idx[a], idx[b]
        S[i, j] += score
        S[j, i] += 1.0 - score
        N[i, j] += 1
        N[j, i] += 1
    return S, N


def fit_bradley_terry(S: np.ndarray, N: np.ndarray, prior_draws: float = 1.0,
                      max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    """Return natural-log strengths (mean zero) via the MM algorithm."""
    n = S.shape[0]
    met = (N > 0).astype(float)
    S = S + 0.5 * prior_draws * met
    N = N + prior_draws * met
    wins = S.sum(axis=1)
    gamma = np.ones(n)
    for _ in range(max_iter):
        denom = (N / (gamma[:, None] + gamma[None, :])).sum(axis=1)
        new = np.where(denom > 0, wins / np.maximum(denom, 1e-300), gamma)
        new = np.maximum(new, 1e-300)
        new /= math.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(gamma))) < tol:
            gamma = new
            break
        gamma = new
    return np.log(gamma)


@dataclass
class EloTable:
    players: List[str]
    elo: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    anchor: str
    anchor_elo: float
    games: np.ndarray  # games played by each player

    def as_rows(self) -> List[dict]:
        order = np.argsort(-self.elo)
        return [
            {
                "player": self.players[i],
                "elo": float(self.elo[i]),
                "ci_low": float(self.ci_low[i]),
                "ci_high": float(self.ci_high[i]),
                "games": int(self.games[i]),
            }
            for i in order
        ]

    def get(self, player: str) -> float:
        return float(self.elo[self.players.index(player)])


def compute_elo(players: Sequence[str], games: Sequence[Tuple[str, str, float]],
                anchor: Optional[str] = None, anchor_elo: float = 0.0,
                prior_draws: float = 1.0, bootstrap: int = 500,
                seed: int = 0) -> EloTable:
    """
    Fit Elo ratings and bootstrap confidence intervals.

    Args:
        players: every player label in the pool.
        games: (player_a, player_b, score_of_a) triples, score in {0, 0.5, 1}.
        anchor: player whose rating is fixed to ``anchor_elo``. Defaults to
            ``Random`` if present, else the lowest-rated player.
        bootstrap: number of resamples (games are resampled within each pair).
    """
    players = list(players)
    games = list(games)
    S, N = pair_matrix(players, games)
    logs = fit_bradley_terry(S, N, prior_draws)
    elo = logs * ELO_SCALE

    if anchor is None:
        anchor = "Random" if "Random" in players else players[int(np.argmin(elo))]
    a_idx = players.index(anchor)
    shift = anchor_elo - elo[a_idx]
    elo = elo + shift

    # Bootstrap: resample each pair's games with replacement.
    rng = np.random.default_rng(seed)
    by_pair: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = {}
    for g in games:
        by_pair.setdefault(tuple(sorted((g[0], g[1]))), []).append(g)
    samples = []
    for _ in range(bootstrap):
        resampled = []
        for pair_games in by_pair.values():
            pick = rng.integers(0, len(pair_games), size=len(pair_games))
            resampled.extend(pair_games[k] for k in pick)
        Sb, Nb = pair_matrix(players, resampled)
        e = fit_bradley_terry(Sb, Nb, prior_draws) * ELO_SCALE
        samples.append(e - e[a_idx] + anchor_elo)
    if samples:
        arr = np.array(samples)
        lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
    else:
        lo = hi = elo.copy()

    return EloTable(players, elo, lo, hi, anchor, anchor_elo, N.sum(axis=1))
