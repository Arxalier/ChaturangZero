# Elo tiers of one network

900 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 300 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 709 | [645, 786] | 180 |
| 2 | AryabhataBot | 661 | [600, 741] | 180 |
| 3 | ChanakyaBot | 621 | [559, 693] | 180 |
| 4 | PaniniBot | 611 | [550, 692] | 180 |
| 5 | AZ-Master | 540 | [475, 616] | 180 |
| 6 | Greedy-1ply | 438 | [385, 501] | 180 |
| 7 | AZ-Intermediate | 287 | [227, 356] | 180 |
| 8 | NoviceBot | 200 | [138, 267] | 180 |
| 9 | AZ-Novice | 174 | [118, 237] | 180 |
| 10 | Random | 0 | [0, 0] | 180 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 133 | 20 | 27 | 81.4% | 125.1 |
| AryabhataBot | 97 | 2 | 81 | 76.4% | 99.5 |
| ChanakyaBot | 99 | 20 | 61 | 71.9% | 107.0 |
| PaniniBot | 79 | 4 | 97 | 70.8% | 109.3 |
| AZ-Master | 84 | 39 | 57 | 62.5% | 136.5 |
| Greedy-1ply | 85 | 84 | 11 | 50.3% | 113.0 |
| AZ-Intermediate | 38 | 99 | 43 | 33.1% | 139.0 |
| NoviceBot | 28 | 121 | 31 | 24.2% | 140.7 |
| AZ-Novice | 17 | 119 | 44 | 21.7% | 150.0 |
| Random | 1 | 153 | 26 | 7.8% | 131.0 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | NoviceBot | AryabhataBot | ChanakyaBot | PaniniBot | AZ-Master | AZ-Intermediate | AZ-Novice |
|---|---|---|---|---|---|---|---|---|---|---|
| **Random** | – | 0-20-0 | 0-20-0 | 1-16-3 | 0-19-1 | 0-17-3 | 0-17-3 | 0-17-3 | 0-14-6 | 0-13-7 |
| **Greedy-1ply** | 20-0-0 | – | 0-20-0 | 20-0-0 | 0-16-4 | 0-19-1 | 0-15-5 | 6-14-0 | 19-0-1 | 20-0-0 |
| **Minimax-d2** | 20-0-0 | 20-0-0 | – | 19-0-1 | 2-11-7 | 14-3-3 | 1-6-13 | 18-0-2 | 19-0-1 | 20-0-0 |
| **NoviceBot** | 16-1-3 | 0-20-0 | 0-19-1 | – | 0-14-6 | 0-18-2 | 0-15-5 | 0-19-1 | 2-12-6 | 10-3-7 |
| **AryabhataBot** | 19-0-1 | 16-0-4 | 11-2-7 | 14-0-6 | – | 4-0-16 | 0-0-20 | 6-0-14 | 15-0-5 | 12-0-8 |
| **ChanakyaBot** | 17-0-3 | 19-0-1 | 3-14-3 | 18-0-2 | 0-4-16 | – | 1-2-17 | 8-0-12 | 17-0-3 | 16-0-4 |
| **PaniniBot** | 17-0-3 | 15-0-5 | 6-1-13 | 15-0-5 | 0-0-20 | 2-1-17 | – | 1-1-18 | 10-1-9 | 13-0-7 |
| **AZ-Master** | 17-0-3 | 14-6-0 | 0-18-2 | 19-0-1 | 0-6-14 | 0-8-12 | 1-1-18 | – | 16-0-4 | 17-0-3 |
| **AZ-Intermediate** | 14-0-6 | 0-19-1 | 0-19-1 | 12-2-6 | 0-15-5 | 0-17-3 | 1-10-9 | 0-16-4 | – | 11-1-8 |
| **AZ-Novice** | 13-0-7 | 0-20-0 | 0-20-0 | 3-10-7 | 0-12-8 | 0-16-4 | 0-13-7 | 0-17-3 | 1-11-8 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | lost_stalemate | won_bare_king | won_checkmate | won_stalemate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 14 | 12 | 113 | 40 | 0 | 0 | 0 | 1 |
| Greedy-1ply | 1 | 10 | 64 | 20 | 0 | 77 | 8 | 0 |
| Minimax-d2 | 8 | 19 | 2 | 18 | 0 | 98 | 35 | 0 |
| NoviceBot | 14 | 17 | 86 | 34 | 1 | 28 | 0 | 0 |
| AryabhataBot | 2 | 79 | 2 | 0 | 0 | 49 | 48 | 0 |
| ChanakyaBot | 1 | 60 | 15 | 5 | 0 | 59 | 40 | 0 |
| PaniniBot | 5 | 92 | 2 | 1 | 1 | 32 | 47 | 0 |
| AZ-Master | 8 | 49 | 27 | 12 | 0 | 78 | 6 | 0 |
| AZ-Intermediate | 12 | 31 | 68 | 31 | 0 | 37 | 0 | 1 |
| AZ-Novice | 17 | 27 | 95 | 24 | 0 | 16 | 1 | 0 |
