"""
arena
-----
Benchmarking suite for ChaturangZero (Person 4).

Modules:
    - players:              Baseline players (Random, Greedy 1-ply, Minimax) and a
                            registry that builds any player from a string spec.
    - match:                Single-game runner with the same draw rules as self-play.
    - elo:                  Bradley-Terry / Elo fitting with bootstrap confidence intervals.
    - tournament:           Parallel round-robin tournaments (python -m arena.tournament).
    - elo_tuning:           Skill tiers of one trained network, measured and calibrated.
    - ablation_experiments: "What didn't work" experiments.
    - plots:                Publication figures.
"""

from .match import GameResult, play_game
from .players import build_player, available_players

__all__ = ["GameResult", "play_game", "build_player", "available_players"]
