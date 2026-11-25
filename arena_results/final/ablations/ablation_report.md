# Ablation experiments: what didn't work

Seed 0, full run, checkpoint: `checkpoints/gen_final.pt`.

## Experiment 1: random rollouts vs value network

- 2000 uniformly random games from the start: median 411 plies, 90th percentile 692; 99.1% run past 100 plies.
- Endings: bare king 94.8%, checkmate 4.5%, stalemate 0.8%.
- With the same 120-ply cap as real games, 93.7% of rollouts never finish and score 0, so most rollouts carry no signal; the ones that do finish are ±1 coin flips (standard deviation 0.994).
- Correlation with the result Minimax-d2 actually reaches from the same position: material 0.799, value net untrained 0.006, value net trained 0.770; rollouts: k=1 0.278, k=2 0.339, k=4 0.407, k=8 0.477, k=16 0.523, k=32 0.540, k=64 0.554, k=128 0.560.
- Rollout-MCTS picks the same move across seeds in 30.0% (at 10 sims), 31.7% (at 25 sims), 32.5% (at 50 sims), 25.0% (at 100 sims), 28.3% (at 200 sims); it costs 40.8 ms per simulation vs 1.6 ms for one network evaluation.
- Head to head at 50 sims: network-MCTS scored 62.5% against rollout-MCTS over 12 games.

## Experiment 2: unmasked vs masked policy

- On average 19.5 of the 4096 actions are legal (0.5%).
- Untrained net: 99.5% of an unmasked softmax lands on illegal moves.
- Trained net: 100.0% of an unmasked softmax lands on illegal moves.
- Imitation run (800 steps, 3200 positions): held-out accuracy masked 14.6% vs unmasked 7.2% (chance 6.5%); steps to 90% of the best accuracy: masked 26, unmasked None.
- Unmasked network still puts 74.8% on illegal moves after training; share of the policy-logit gradient on legal moves: masked 100.0%, unmasked 57.9%.

## Experiment 3: greedy 1-ply vs lookahead

- Trap suite of 200 positions (a capture is punished by an Elephant leap with net material loss; a safe move exists). Poisoned capture taken: Random 3.0%, Greedy-1ply 48.5%, Minimax-d2 1.5%, Minimax-d3 0.0%, AZ-raw 12.5%, AZ-80 10.0%.
- Greedy-1ply vs Random (30 games): score 90.0%; Elephant blunders per 100 moves 0.22 vs 0.07.
- Minimax-d2 vs Random (30 games): score 88.3%; Elephant blunders per 100 moves 0.00 vs 0.00.
- Greedy-1ply vs Minimax-d2 (30 games): score 15.0%; Elephant blunders per 100 moves 0.35 vs 0.00.

Figures: `exp*.png` in this folder. Raw numbers: `ablation_results.json`.
