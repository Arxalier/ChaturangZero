# Elo tiers of one network

300 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 120 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 270 | [225, 320] | 100 |
| 2 | Greedy-1ply | 217 | [173, 260] | 100 |
| 3 | AZ-Intermediate | 43 | [18, 68] | 100 |
| 4 | AZ-Master | 31 | [3, 59] | 100 |
| 5 | AZ-Novice | 3 | [-21, 29] | 100 |
| 6 | Random | 0 | [0, 0] | 100 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 55 | 0 | 45 | 77.5% | 98.5 |
| Greedy-1ply | 56 | 16 | 28 | 70.0% | 92.5 |
| AZ-Intermediate | 0 | 16 | 84 | 42.0% | 84.2 |
| AZ-Master | 0 | 20 | 80 | 40.0% | 79.0 |
| AZ-Novice | 0 | 29 | 71 | 35.5% | 80.5 |
| Random | 1 | 31 | 68 | 35.0% | 106.6 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | AZ-Master | AZ-Intermediate | AZ-Novice |
|---|---|---|---|---|---|---|
| **Random** | – | 0-15-5 | 0-16-4 | 1-0-19 | 0-0-20 | 0-0-20 |
| **Greedy-1ply** | 15-0-5 | – | 0-16-4 | 13-0-7 | 11-0-9 | 17-0-3 |
| **Minimax-d2** | 16-0-4 | 16-0-4 | – | 6-0-14 | 5-0-15 | 12-0-8 |
| **AZ-Master** | 0-1-19 | 0-13-7 | 0-6-14 | – | 0-0-20 | 0-0-20 |
| **AZ-Intermediate** | 0-0-20 | 0-11-9 | 0-5-15 | 0-0-20 | – | 0-0-20 |
| **AZ-Novice** | 0-0-20 | 0-17-3 | 0-12-8 | 0-0-20 | 0-0-20 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | won_bare_king | won_checkmate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 68 | 0 | 26 | 5 | 0 | 1 |
| Greedy-1ply | 28 | 0 | 14 | 2 | 53 | 3 |
| Minimax-d2 | 41 | 4 | 0 | 0 | 40 | 15 |
| AZ-Master | 38 | 42 | 14 | 6 | 0 | 0 |
| AZ-Intermediate | 43 | 41 | 16 | 0 | 0 | 0 |
| AZ-Novice | 30 | 41 | 23 | 6 | 0 | 0 |
