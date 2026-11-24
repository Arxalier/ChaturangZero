# Tournament: final

900 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 300 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 672 | [612, 737] | 180 |
| 2 | ChanakyaBot | 618 | [546, 688] | 180 |
| 3 | PaniniBot | 618 | [551, 687] | 180 |
| 4 | AryabhataBot | 608 | [543, 680] | 180 |
| 5 | AZ-Master | 499 | [433, 566] | 180 |
| 6 | Greedy-1ply | 429 | [372, 489] | 180 |
| 7 | AZ-Intermediate | 275 | [217, 335] | 180 |
| 8 | AZ-Novice | 152 | [96, 206] | 180 |
| 9 | NoviceBot | 149 | [94, 203] | 180 |
| 10 | Random | 0 | [0, 0] | 180 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 129 | 21 | 30 | 80.0% | 119.2 |
| ChanakyaBot | 106 | 19 | 55 | 74.2% | 101.8 |
| PaniniBot | 92 | 5 | 83 | 74.2% | 103.2 |
| AryabhataBot | 90 | 7 | 83 | 73.1% | 97.2 |
| AZ-Master | 78 | 41 | 61 | 60.3% | 138.2 |
| Greedy-1ply | 86 | 79 | 15 | 51.9% | 107.4 |
| AZ-Intermediate | 38 | 95 | 47 | 34.2% | 136.4 |
| AZ-Novice | 20 | 122 | 38 | 21.7% | 148.1 |
| NoviceBot | 26 | 129 | 25 | 21.4% | 137.7 |
| Random | 1 | 148 | 31 | 9.2% | 135.3 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | NoviceBot | AryabhataBot | ChanakyaBot | PaniniBot | AZ-Master | AZ-Intermediate | AZ-Novice |
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | – | 0-20-0 | 0-20-0 | 1-18-1 | 0-15-5 | 0-16-4 | 0-15-5 | 0-17-3 | 0-14-6 | 0-13-7 |
| **Greedy-1ply** | 20-0-0 | – | 0-20-0 | 19-0-1 | 1-14-5 | 1-19-0 | 0-12-8 | 6-14-0 | 19-0-1 | 20-0-0 |
| **Minimax-d2** | 20-0-0 | 20-0-0 | – | 20-0-0 | 3-8-9 | 5-9-6 | 4-4-12 | 18-0-2 | 19-0-1 | 20-0-0 |
| **NoviceBot** | 18-1-1 | 0-19-1 | 0-20-0 | – | 0-18-2 | 0-19-1 | 0-19-1 | 0-14-6 | 0-13-7 | 8-6-6 |
| **AryabhataBot** | 15-0-5 | 14-1-5 | 8-3-9 | 18-0-2 | – | 6-0-14 | 0-3-17 | 5-0-15 | 10-0-10 | 14-0-6 |
| **ChanakyaBot** | 16-0-4 | 19-1-0 | 9-5-6 | 19-0-1 | 0-6-14 | – | 1-7-12 | 10-0-10 | 16-0-4 | 16-0-4 |
| **PaniniBot** | 15-0-5 | 12-0-8 | 4-4-12 | 19-0-1 | 3-0-17 | 7-1-12 | – | 2-0-18 | 14-0-6 | 16-0-4 |
| **AZ-Master** | 17-0-3 | 14-6-0 | 0-18-2 | 14-0-6 | 0-5-15 | 0-10-10 | 0-2-18 | – | 16-0-4 | 17-0-3 |
| **AZ-Intermediate** | 14-0-6 | 0-19-1 | 0-19-1 | 13-0-7 | 0-10-10 | 0-16-4 | 0-14-6 | 0-16-4 | – | 11-1-8 |
| **AZ-Novice** | 13-0-7 | 0-20-0 | 0-20-0 | 6-8-6 | 0-14-6 | 0-16-4 | 0-16-4 | 0-17-3 | 1-11-8 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | won_bare_king | won_checkmate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 13 | 18 | 114 | 34 | 0 | 1 |
| Greedy-1ply | 2 | 13 | 53 | 26 | 77 | 9 |
| Minimax-d2 | 6 | 24 | 2 | 19 | 95 | 34 |
| NoviceBot | 14 | 11 | 86 | 43 | 25 | 1 |
| AryabhataBot | 0 | 83 | 4 | 3 | 43 | 47 |
| ChanakyaBot | 2 | 53 | 15 | 4 | 52 | 54 |
| PaniniBot | 4 | 79 | 4 | 1 | 32 | 60 |
| AZ-Master | 12 | 49 | 22 | 19 | 73 | 5 |
| AZ-Intermediate | 10 | 37 | 64 | 31 | 38 | 0 |
| AZ-Novice | 15 | 23 | 89 | 33 | 18 | 2 |
