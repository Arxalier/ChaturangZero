ChaturangZero: A Reinforcement Learning Agent to play Chaturanga

Team Syllogism

2024A7PS0375G Aaryan Pandey
2024A7PS0606G Vraj Makwana
2024A7PS0870G Harshal Lahoti
2024A7PS0485G Sahaj Jhamb

CS F407 Artificial Intelligence, AY 2026–27 Semester I, BITS Pilani, K.K. Birla Goa Campus

Chaturanga is an ancient Indian-origin board game with much more restrictive piece mobility compared to chess (for example, the queen can only move one step diagonally). Under these conditions, standard self-play reinforcement learning collapses into cyclic repetition draws, making learning from scratch ineffective without massive compute. To get around this problem, we developed ChaturangZero, pairing a dual-headed CNN with Polynomial Upper Confidence Trees (PUCT) Monte Carlo Tree Search. We found that training purely by self-play over 450 games left the model at par with uniform random play, as the value head got stuck outputting a near-constant evaluation. However, by deriving piece values from move mobility and training on a dataset of heuristic-backed games (about 10,000 greedy moves), the model improved significantly. The resulting agent consistently defeats random play with a 90.0% win rate, and achieves an 87.5% unbeaten rate (25.0% wins, 62.5% draws, 12.5% losses) against four heuristic bots.

---

## 1. Introduction

Chaturanga is played on an $8 \times 8$ uncheckered grid called *Ashtapada* [5, 6]. While it is the ancestor of modern chess, its mechanics create a different ruleset:
1. Foot Soldiers (*Padāti*) cannot double-step on turn one.
2. Ministers (*Mantri*) move only one step diagonally.
3. Elephants (*Gaja*) strictly leap two squares diagonally.
4. Pawns promote exclusively to Ministers upon reaching the back rank.
5. Capturing all opponent non-king pieces (*Bare King*) triggers an immediate win.
6. Stalemate counts as an outright win for the stalemated player.

These restrictions cause extreme game lengths. In random play, the median game lasts 411 plies. When AlphaZero-style self-play is attempted with limited compute, both players rarely engage before hitting repetitive moves. Due to the longevity of the game, the search agent sparsely receives terminal signals, causing policy and value convergence to collapse. 

### Input and Output Specification
* **Input**: A $(13, 8, 8)$ binary tensor encoding the board for 6 piece types per player and a constant turn plane.
* **Output**: A 4,096-dimensional masked legal policy vector $\boldsymbol{\pi} \in \mathbb{R}^{4096}$ and a scalar position evaluation $v \in [-1, +1]$.

### Contributions
1. We implemented an end-to-end ancient Chaturanga engine and dual-headed ResNet paired with PUCT MCTS.
2. We showed why training purely from scratch fails due to endless repetition draws.
3. We introduced a first-principles derivation method for heuristics, calibrated with human feedback, to bootstrap the network. By bootstrapping the method, we mean to supply it with training data of games played with greedy rules so it can learn the intrinsic value of pieces. It is worth noting that this violates the condition of the bot learning purely through self-play. Despite this, we wanted to see how well the bot can perform with some pre-trained insights.
4. We evaluated the agent against random baselines and four distinct minimax heuristic bots.

---

## 2. Background and Related Work

### AlphaZero and PUCT Search
AlphaZero replaces evaluation functions with a deep neural network $f_\theta(s) = (\mathbf{p}, v)$ trained through reinforcement learning from self-play [1]. In each turn, Monte Carlo Tree Search (MCTS) uses a four step process to evaluate the game's search tree.
the network priors $\mathbf{p}$ and scalar values $v$ to guide tree expansion via the Polynomial Upper Confidence Trees (PUCT) formula:

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$


### Deep Reinforcement Learning in Board Games
Modern game-playing agents rely on policy iteration where MCTS acts as a policy improver [2]. However, tabular and deep actor-critic methods struggle when reward signals are delayed over hundreds of steps [3]. While multi-variant classical engines like Fairy-Stockfish can be configured to play Chaturanga, there are no deep reinforcement learning counterparts (like AlphaZero for Chess). We aim to bridge that gap by modeling an AlphaZero-like architecture over the base rules of ancient Chaturanga.

---

## 3. Method

```
[ Board State (8x8) ] ──> [ Board-to-Tensor Encoder ] ──> [ (13, 8, 8) Tensor ]
                                                                │
                                                ┌───────────────┴───────────────┐
                                                ▼                               ▼
                                       [ 4x ResNet Blocks ]           [ Action Mask (4096) ]
                                                │                               │
                                ┌───────────────┴───────────────┐               │
                                ▼                               ▼               │
                        [ Policy Head ]                  [ Value Head ]         │
                                │                               │               │
                        [ Masked Logits ] ◄─────────────────────┼───────────────┘
                                │                               │
                                ▼                               ▼
                         π(a|s) [4096]                      v(s) [-1, 1]
                                │                               │
                                └───────────────┬───────────────┘
                                                ▼
                                    [ PUCT Neural MCTS ]
                                                │
                                                ▼
                                         [ Action a* ]
```

### 3.1 Board Encoding and Neural Architecture
The board is converted into 13 binary feature planes:
* Channels 0–5: White $P, H, E, R, M, K$.
* Channels 6–11: Black $P, H, E, R, M, K$.
* Channel 12: Turn indicator plane ($1.0$ for White, $0.0$ for Black).

The action space maps all from-square to to-square transitions:
$$\text{action} = (r_{\text{start}} \times 8 + c_{\text{start}}) \times 64 + (r_{\text{end}} \times 8 + c_{\text{end}})$$

The network backbone contains an initial convolutional layer (64 filters, $3 \times 3$) followed by 4 residual blocks with batch normalization and ReLU activations. The policy head applies a $1 \times 1$ convolution, flattens to 4,096 logits, and masks out illegal moves to $-\infty$ before softmax. The value head applies a $1 \times 1$ convolution, dense layers, and a $\tanh$ activation returning $v \in [-1, +1]$.
can you also specify layers, and parameters?

### 3.2 Why Pure Self-Play Underperformed
Our initial attempt used pure self-play. We trained over 30 iterations (15 games per iteration, 450 games total) using MCTS with 20 simulations per move. 

The training failed due to two factors:
1. **Horizon Length**: In Chaturanga, players require 30–50 plies simply to cross the board. Pawns cannot leap forward, and ministers cover only 32 squares of the board.
2. **Repetition Cycles**: Because the value head was untrained, state evaluations were uniform. MCTS drifted into cyclic piece shuffling, terminating in 3-fold repetition draws by ply 45.

With over 90% of games ending in draws, target outcomes $z \in \{-1, 0, +1\}$ were zero. The value head converged to a degenerate constant $V(s) \approx +0.1330$ everywhere. Because $Q$-values across all branches were identical, MCTS could not identify captures.

### 3.3 Autonomous Value Induction from Move Graphs
To eliminate hardcoded human bias while resolving the exploration problem, we wrote an autonomous script (`discover_values.py`) that derives piece values and board centrality directly from the move generators:

$$\mathbb{E}[M_p] = \frac{1}{64} \sum_{r=0}^{7} \sum_{c=0}^{7} |\text{moves}(p, r, c)|$$

Summing across all 64 squares of an empty Ashtapada:
* Pawn average reach: $\text{base} = 1.64$ moves.
* Elephant reach: $2.25$ moves.
* Minister reach: $3.06$ moves.
* Horse reach: $5.25$ moves.
* Chariot reach: $14.00$ moves.

To account for piece obstruction on sliding pieces, we applied sublinear dampening ($\alpha = 0.85$):
$$V_{\text{raw}}(p) = \left(\frac{\mathbb{E}[M_p]}{\text{base}}\right)^{0.85}$$

This derived:
* Pawn: $1.00$
* Elephant: $1.31$
* Minister: $1.70$
* Horse: $2.69$
* Chariot: $6.19$

Simultaneously, summing move counts per square across short-range pieces yielded an $8 \times 8$ degree centrality matrix. Min-max normalization scaled this to a $[0.00, 0.20]$ positional bonus favoring the central $4 \times 4$ squares.

### 3.4 Manual Calibration and Supervised Bootstrapping
Training a CNN with only the base engine rules via self-play requires immense compute and millions of parallel games. By incorporating human feedback and move heuristics, we embed the heavy-lifting of figuring out basic piece values directly into a small training subset. We rounded and slightly adjusted these raw numbers by hand to account for early-game crowding:
$$\mathbf{w}_{\text{aligned}} = [\text{Pawn}: 1.0, \text{Elephant}: 1.5, \text{Minister}: 2.0, \text{Horse}: 3.0, \text{Chariot}: 5.0]$$

We generated a dataset of 150 curriculum games using 2-ply teacher evaluations with randomized exploration noise ($\epsilon = 0.15$). We fine-tuned the existing CNN weights using multi-task loss with action masking:

$$\mathcal{L} = 4.0 \cdot (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c \|\theta\|^2$$

Over 12 epochs, value loss dropped from $0.1752$ to $0.0099$, and policy loss dropped from $2.7178$ to $0.2218$.

---

## 4. Experimental Setup

### Baselines
1. **Uniform Random Bot**: Selects uniformly from legal moves.
2. **NoviceBot**: Selects 80% random moves, 20% greedy captures.
3. **ChanakyaBot**: Depth-2 Minimax with alpha-beta pruning and greedy material evaluation.
4. **AryabhataBot**: Depth-2 Minimax with piece-square positional tables favoring center control.
5. **PaniniBot**: Depth-2 Minimax with defensive penalties against piece vulnerability.

### Hardware and Evaluation
All benchmarks were run locally on an Intel x86_64 CPU (16 logical cores) running PyTorch 2.6. Inference used single-thread CPU execution to prevent OpenMP contention. 

---

## 5. Results

### 5.1 Material Valuation Diagnostic Probe
To verify whether the value head learned real material semantics, we passed differential board positions through $V(s)$:

| Piece Configuration | White Advantage | White Disadvantage | Net Delta |
|---|:---:|:---:|:---:|
| **Pawn (*Padati*)** | +0.0589 | -0.0213 | **+0.0803** |
| **Elephant (*Gaja*)** | +0.0611 | -0.0773 | **+0.1383** |
| **Minister (*Mantri*)** | +0.0794 | -0.0146 | **+0.0939** |
| **Horse (*Ashwa*)** | +0.3545 | -0.1142 | **+0.4687** |
| **Chariot (*Ratha*)** | +0.2340 | -0.2410 | **+0.4750** |
| **Major Combo (+Ratha +Ashwa)** | +0.5748 | -0.3850 | **+0.9598** |

The value head responds monotonically to material changes. Horses and Chariots produce the strongest positive evaluations, matching the tactical reality of the game.

### 5.2 Benchmark vs. Uniform Random Bot
We evaluated 10 games between ChaturangZero and the Uniform Random Bot across training regimes:

| Training Regime | Search Sims | Max Plies | AI Wins | Random Wins | Draws |
|---|:---:|:---:|:---:|:---:|:---:|
| Pure Self-Play (30 iters) | 20 | 120 | 0.0% (0) | 10.0% (1) | 90.0% (9) |
| Bootstrapped Fine-Tuning | 40 | 300 | 90.0% (9) | 0.0% (0) | 10.0% (1) |
| Bootstrapped Fine-Tuning | 200 | 250 | **90.0% (9)** | **0.0% (0)** | **10.0% (1)** |

At 200 simulations, the agent converted games in an average of 144 plies, steadily picking off pieces until it bared the opponent's king.

### 5.3 Tournament Benchmark vs. Heuristic Scholar Bots
We pitted the trained agent against the four heuristic bots in an 8-game tournament (1 game White, 1 game Black per bot) evaluated at 300 MCTS simulations per move with a cutoff of 260 plies:

| Opponent | AI Wins | Bot Wins | Draws | AI Win Rate |
|---|:---:|:---:|:---:|:---:|
| **NoviceBot** | 2 | 0 | 0 | **100.0%** |
| **ChanakyaBot** | 0 | 0 | 2 | 0.0% (100% draw) |
| **AryabhataBot** | 0 | 0 | 2 | 0.0% (100% draw) |
| **PaniniBot** | 0 | 1 | 1 | 0.0% (50% draw) |
| **Total** | **2 (25.0%)** | **1 (12.5%)** | **5 (62.5%)** | **25.0%** |

The agent achieved an 87.5% unbeaten rate (25.0% wins, 62.5% draws, 12.5% losses) across the tournament. It cleanly defeated NoviceBot in both matches (100% win rate) while holding ChanakyaBot and AryabhataBot to draws in every game.

---

## 6. Discussion and Limitations

### The Draw Tendency
The tournament results highlight a persistent issue: 62.5% of games against heuristic bots ended in 3-fold repetition draws. 
There are two reasons for this:
1. **Opponent Behavior**: The scholar bots do not maintain move history. When faced with equal evaluations, they oscillate pieces between two squares.
2. **Neutral Draw Evaluations**: The neural network evaluates equal positions near $V(s) \approx 0.0$. Because a draw is also evaluated as $0.0$, the agent sees no negative value in accepting repetition. In Game 6 against Aryabhata, the agent held an 8-piece to 4-piece advantage, yet allowed a repetition draw at ply 112.

### Search Depth and Compute Constraints
The agent's strength is heavily bottlenecked by MCTS simulations. At 30 simulations, the agent searches only 1–2 plies deep, falling prey to depth-2 alpha-beta traps. At 300 simulations, the agent plays strong moves but requires ~30,000 forward passes per game. On CPU, this takes 4–5 minutes per game, making large evaluation sweeps slow.

### Future Work
1. **Contempt Factor**: Incorporating a negative penalty for repetition draws ($V = -0.25$) when ahead in material will force the agent to avoid lazy repetitions.
2. **GPU Batched MCTS**: Running vectorized leaf evaluations in parallel will allow 300+ simulations to execute in under 10 seconds per game.
3. **Iterative Self-Play Reinforcement**: Using the bootstrapped weights as the starting checkpoint for an extended AlphaZero self-play run.

---

## 7. Conclusion

We built ChaturangZero, a neural MCTS agent for ancient Chaturanga. We showed that training purely from scratch via self-play fails in low-mobility board games due to extreme game horizons and draw collapse. By deriving piece weights from move mobility and bootstrapping the network on 2-ply games, our agent achieved a 90% win rate against random play and an 87.5% unbeaten rate against depth-2 minimax engines.

---

## Individual Member Contributions

| Name | Roll Number | Contribution |
|---|---|---|
| **Aaryan Pandey** | 2024A7PS0375G | Training pipeline lead: MCTS-NN integration, self-play loop, curriculum generation, teacher bootstrapping, loss masking. |
| **Vraj Makwana** | 2024A7PS0606G | Deep learning architect: Board-to-tensor encoder, action space indexing, 4-block dual-headed ResNet architecture. |
| **Sahaj Jhamb** | 2024A7PS0485G | Heuristic engine & UI: Implemented Scholar bots (Novice, Chanakya, Aryabhata, Panini), web board UI, opponent selector. |
| **Harshal Lahoti** | 2024A7PS0870G | Benchmarking & analysis: Arena tournament harness, material diagnostic probes, ablation experiments, report figures. |

*All members have read this report and agree with the statement above.*

---

## Generative AI Disclosure

Generative AI tools were used during this project as follows:
* **Gemini 3.8 Flash & Gemini Neon**: Used for drafting environment unit tests, debugging PyTorch tensor indexing in the action masking logic, and formatting benchmark tables.
* **Claude Opus 5**: Used for code review, validating the mathematical formulation of sublinear piece mobility, and refining report prose.

All code and report text were manually verified, debugged, and tested by the team members, who take full academic responsibility for the contents.

---

## Code Availability

The complete source code, trained neural network weights, evaluation logs, and browser UI are publicly available at:  
**https://github.com/Arxalier/ChaturangZero**

---

## References

[1] D. Silver, T. Hubert, J. Schrittwieser, et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, no. 6419, pp. 1140–1144, 2018.  
[2] D. Silver, J. Schrittwieser, K. Simonyan, et al., "Mastering the game of Go without human knowledge," *Nature*, vol. 550, no. 7676, pp. 354–359, 2017.  
[3] T. Dash, "Deep Reinforcement Learning," Lecture Slides, CS F407 Artificial Intelligence, BITS Pilani, AY 2025–26 Sem II. [Online]. Available: `https://github.com/tirtharajdash/CS-F407-AI-AY2025-26-S2/blob/main/lectures/deep_rl.pdf`  
[4] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2021.  
[5] H. J. R. Murray, *A History of Chess*, Oxford University Press, 1913.  
[6] H. Golombek, *Chess: A History*, G. P. Putnam's Sons, 1976.
