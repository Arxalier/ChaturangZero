# Tournament: final

300 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 300 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 1026 | [928, 1161] | 100 |
| 2 | AZ-Master | 632 | [533, 769] | 100 |
| 3 | Greedy-1ply | 585 | [502, 687] | 100 |
| 4 | AZ-Intermediate | 285 | [215, 379] | 100 |
| 5 | AZ-Novice | 167 | [96, 246] | 100 |
| 6 | Random | 0 | [0, 0] | 100 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 97 | 0 | 3 | 98.5% | 109.6 |
| AZ-Master | 64 | 24 | 12 | 70.0% | 154.9 |
| Greedy-1ply | 65 | 34 | 1 | 65.5% | 115.3 |
| AZ-Intermediate | 25 | 55 | 20 | 35.0% | 154.8 |
| AZ-Novice | 14 | 68 | 18 | 23.0% | 164.5 |
| Random | 0 | 84 | 16 | 8.0% | 152.5 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | AZ-Master | AZ-Intermediate | AZ-Novice |
|---|---|---|---|---|---|---|
| **Random** | – | 0-20-0 | 0-20-0 | 0-17-3 | 0-14-6 | 0-13-7 |
| **Greedy-1ply** | 20-0-0 | – | 0-20-0 | 6-14-0 | 19-0-1 | 20-0-0 |
| **Minimax-d2** | 20-0-0 | 20-0-0 | – | 18-0-2 | 19-0-1 | 20-0-0 |
| **AZ-Master** | 17-0-3 | 14-6-0 | 0-18-2 | – | 16-0-4 | 17-0-3 |
| **AZ-Intermediate** | 14-0-6 | 0-19-1 | 0-19-1 | 0-16-4 | – | 11-1-8 |
| **AZ-Novice** | 13-0-7 | 0-20-0 | 0-20-0 | 0-17-3 | 1-11-8 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | won_bare_king | won_checkmate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 12 | 4 | 76 | 8 | 0 | 0 |
| Greedy-1ply | 1 | 0 | 30 | 4 | 60 | 5 |
| Minimax-d2 | 0 | 3 | 0 | 0 | 71 | 26 |
| AZ-Master | 7 | 5 | 17 | 7 | 60 | 4 |
| AZ-Intermediate | 8 | 12 | 46 | 9 | 25 | 0 |
| AZ-Novice | 10 | 8 | 60 | 8 | 13 | 1 |
