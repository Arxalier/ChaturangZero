# Tournament: progression

300 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 120 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 244 | [199, 289] | 100 |
| 2 | Greedy-1ply | 210 | [171, 255] | 100 |
| 3 | gen_final | 40 | [7, 72] | 100 |
| 4 | gen_0 | 37 | [0, 70] | 100 |
| 5 | gen_mid | 22 | [-6, 50] | 100 |
| 6 | Random | 0 | [0, 0] | 100 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 50 | 1 | 49 | 74.5% | 100.6 |
| Greedy-1ply | 56 | 17 | 27 | 69.5% | 94.2 |
| gen_final | 0 | 17 | 83 | 41.5% | 92.0 |
| gen_0 | 7 | 25 | 68 | 41.0% | 103.4 |
| gen_mid | 0 | 23 | 77 | 38.5% | 89.2 |
| Random | 3 | 33 | 64 | 35.0% | 105.2 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | gen_0 | gen_mid | gen_final |
|---|---|---|---|---|---|---|
| **Random** | – | 0-15-5 | 0-16-4 | 1-2-17 | 0-0-20 | 2-0-18 |
| **Greedy-1ply** | 15-0-5 | – | 0-16-4 | 16-1-3 | 15-0-5 | 10-0-10 |
| **Minimax-d2** | 16-0-4 | 16-0-4 | – | 8-1-11 | 6-0-14 | 4-0-16 |
| **gen_0** | 2-1-17 | 1-16-3 | 1-8-11 | – | 2-0-18 | 1-0-19 |
| **gen_mid** | 0-0-20 | 0-15-5 | 0-6-14 | 0-2-18 | – | 0-0-20 |
| **gen_final** | 0-2-18 | 0-10-10 | 0-4-16 | 0-1-19 | 0-0-20 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | won_bare_king | won_checkmate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 63 | 1 | 26 | 7 | 0 | 3 |
| Greedy-1ply | 27 | 0 | 14 | 3 | 48 | 8 |
| Minimax-d2 | 42 | 7 | 0 | 1 | 33 | 17 |
| gen_0 | 59 | 9 | 19 | 6 | 0 | 7 |
| gen_mid | 51 | 26 | 16 | 7 | 0 | 0 |
| gen_final | 54 | 29 | 6 | 11 | 0 | 0 |
