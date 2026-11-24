# Tournament: progression

300 games, 20 per pair, colours alternated, 2 random opening plies shared by each colour-swapped pair, draw at threefold repetition or 300 plies.

## Elo (anchor: Random = 0, 95% bootstrap CI)

| Rank | Player | Elo | 95% CI | Games |
|---:|---|---:|---|---:|
| 1 | Minimax-d2 | 623 | [536, 742] | 100 |
| 2 | Greedy-1ply | 396 | [341, 462] | 100 |
| 3 | gen_final | 209 | [154, 269] | 100 |
| 4 | gen_0 | 88 | [30, 144] | 100 |
| 5 | pureRL-local | 59 | [12, 102] | 100 |
| 6 | Random | 0 | [0, 0] | 100 |

## Standings

| Player | W | L | D | Score | Avg plies |
|---|---:|---:|---:|---:|---:|
| Minimax-d2 | 89 | 1 | 10 | 94.0% | 125.1 |
| Greedy-1ply | 72 | 26 | 2 | 73.0% | 112.1 |
| gen_final | 30 | 31 | 39 | 49.5% | 166.9 |
| gen_0 | 14 | 48 | 38 | 33.0% | 176.2 |
| pureRL-local | 1 | 43 | 56 | 29.0% | 187.0 |
| Random | 5 | 62 | 33 | 21.5% | 189.5 |

## Head to head (row W-L-D vs column)

| | Random | Greedy-1ply | Minimax-d2 | gen_0 | pureRL-local | gen_final |
|---|---|---|---|---|---|---|
| **Random** | – | 0-20-0 | 0-20-0 | 1-8-11 | 4-0-16 | 0-14-6 |
| **Greedy-1ply** | 20-0-0 | – | 0-20-0 | 20-0-0 | 20-0-0 | 12-6-2 |
| **Minimax-d2** | 20-0-0 | 20-0-0 | – | 19-1-0 | 12-0-8 | 18-0-2 |
| **gen_0** | 8-1-11 | 0-20-0 | 1-19-0 | – | 4-1-15 | 1-7-12 |
| **pureRL-local** | 0-4-16 | 0-20-0 | 0-12-8 | 1-4-15 | – | 0-3-17 |
| **gen_final** | 14-0-6 | 6-12-2 | 0-18-2 | 7-1-12 | 3-0-17 | – |

## How games ended (from each player's side)

| Player | draw_max_plies | draw_repetition | lost_bare_king | lost_checkmate | won_bare_king | won_checkmate |
|---|---:|---:|---:|---:|---:|---:|
| Random | 30 | 3 | 53 | 9 | 3 | 2 |
| Greedy-1ply | 0 | 2 | 23 | 3 | 65 | 7 |
| Minimax-d2 | 3 | 7 | 0 | 1 | 65 | 24 |
| gen_0 | 23 | 15 | 42 | 6 | 7 | 7 |
| pureRL-local | 28 | 28 | 27 | 16 | 0 | 1 |
| gen_final | 12 | 27 | 21 | 10 | 26 | 4 |
