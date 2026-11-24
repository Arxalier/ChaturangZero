To ensure complete decoupling and prevent merge conflicts, all players (Random, Heuristic Scholar Bots, and AlphaZero) must inherit from and implement this exact interface:

python  
import numpy as np  
from typing import Tuple  
class BasePlayer:  
   def get\_move(self, board: np.ndarray, player: int) \-\> Tuple\[Tuple\[int, int\], Tuple\[int, int\]\]:  
       """  
       Takes the current 8x8 numpy board state and the active player (1 for White, \-1 for Black).  
       Returns a legal move formatted as ((start\_row, start\_col), (end\_row, end\_col)).  
       """  
       raise NotImplementedError  
---

### **Repository Structure & Folder Ownership**

ChaturangZero/  
├── chaturanga/          \# Core rules and board logic (already complete and verified)  
├── neural/              \# Person 1: Neural network architecture and tensor encoding \- Vraj  
├── training/            \# Person 2: MCTS-NN integration and self-play training loop  
├── bots/                \# Person 3: Scholar heuristic bots  
├── ui/                  \# Person 3: Web interface integration  
└── arena/               \# Person 4: Benchmarking, Elo tuning, and ablation studies  
---

### **Person 1: Deep Learning Architect (Model & Tensor Encoders)**

\-\> Vraj Makwana

**Directory:** `neural/`  
 **Objective:** Construct the input/output translation layer and the dual-headed ResNet architecture in PyTorch.

* **File: `neural/encoder.py`**  
  * `board_to_tensor(board: np.ndarray, player: int) -> torch.Tensor`: Converts the 8×88 \\times 88×8 board into a (13,8,8)(13, 8, 8)(13,8,8) binary tensor (channels 0–5 for White pieces, 6–11 for Black pieces, channel 12 filled with 1s if White to move, 0s if Black).  
  * `move_to_action(move: Tuple[Tuple[int, int], Tuple[int, int]]) -> int`: Flattens a move to an index in 0…40950 \\dots 40950…4095 via ((start\_r×8+start\_c)×64)+(end\_r×8+end\_c)((\\text{start\\\_r} \\times 8 \+ \\text{start\\\_c}) \\times 64\) \+ (\\text{end\\\_r} \\times 8 \+ \\text{end\\\_c})((start\_r×8+start\_c)×64)+(end\_r×8+end\_c).  
  * `action_to_move(action_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]`: Reverse mapping of the above.  
  * `get_action_mask(board: np.ndarray, player: int) -> np.ndarray`: Returns a 4096-dimensional boolean vector where only indices corresponding to strictly legal moves are `True`.  
* **File: `neural/model.py`**  
  * Implements `ChaturangaNet(nn.Module)`.  
  * Backbone: 1 input convolutional block followed by 4 residual blocks (64 filters, kernel size 3, batch norm, ReLU).  
  * Policy Head: Conv 1×1→1 \\times 1 \\rightarrow1×1→ BatchNorm →\\rightarrow→ ReLU →\\rightarrow→ Flatten →\\rightarrow→ Linear(4096). Applies the action mask before computing log-softmax.  
  * Value Head: Conv 1×1→1 \\times 1 \\rightarrow1×1→ BatchNorm →\\rightarrow→ ReLU →\\rightarrow→ Flatten →\\rightarrow→ Linear(64) →\\rightarrow→ ReLU →\\rightarrow→ Linear(1) →tanh⁡\\rightarrow \\tanh→tanh, returning a scalar v∈\[−1,+1\]v \\in \[-1, \+1\]v∈\[−1,+1\].  
* **Standalone Verification:** `neural/test_model.py` passes a dummy batch of shape (B,13,8,8)(B, 13, 8, 8)(B,13,8,8) through both heads, asserts output shapes (B,4096)(B, 4096)(B,4096) and (B,1)(B, 1)(B,1), and verifies that backward gradients compute without errors.

---

### **Person 2: Training Pipeline Lead (MCTS-NN Connection & Self-Play)**

\-\> Aaryan Pandey

**Directory:** `training/`  
 **Objective:** Replace random rollouts in MCTS with the neural evaluator, generate self-play data, and execute the training loop.

* **File: `training/mcts_agent.py`**  
  * Adapts the existing `mcts.py` so leaf node evaluation queries the neural network via `with torch.no_grad(): policy, value = model(tensor)`.  
  * Adds Dirichlet noise (α=0.3,ϵ=0.25\\alpha \= 0.3, \\epsilon \= 0.25α=0.3,ϵ=0.25) to root node priors during self-play for exploration.  
  * Implements `AlphaZeroPlayer(BasePlayer)` with configurable `num_sims`, `temperature`, and `blunder_rate`.  
  * *Note:* Uses a small dummy network stub locally so the training code can be written and tested immediately before Person 1 finishes the final ResNet.  
* **File: `training/self_play.py`**  
  * Implements game generation: the current model plays both sides.  
  * Stores state transitions (st,πt,z)(s\_t, \\boldsymbol{\\pi}\_t, z)(st​,πt​,z) in a replay buffer, where πt\\boldsymbol{\\pi}\_tπt​ is the MCTS visit distribution (N(s,a)1/τ/∑NN(s, a)^{1/\\tau} / \\sum NN(s,a)1/τ/∑N) and z∈{−1,+1}z \\in \\{-1, \+1\\}z∈{−1,+1} is the final game winner.  
* **File: `training/train.py` & Colab Runner**  
  * Implements multi-task loss: L=(z−v)2−π⊤log⁡p+c∥θ∥2\\mathcal{L} \= (z \- v)^2 \- \\boldsymbol{\\pi}^\\top \\log \\mathbf{p} \+ c \\\|\\theta\\\|^2L=(z−v)2−π⊤logp+c∥θ∥2.  
  * Executes the training loop using Adam (learning rate 10−310^{-3}10−3, weight decay 10−410^{-4}10−4).  
  * Saves checkpoints using `torch.save(model.state_dict(), filename)`: specifically `gen_0.pt` (untrained initialization), `gen_mid.pt`, and `gen_final.pt`.  
* **Standalone Verification:** `training/test_pipeline.py` runs a 1-game self-play simulation on CPU with 10 MCTS visits and performs 1 gradient update step.

---

### **Person 3: UI & Themed Scholar Bots (Heuristics & Web Interface)**

\-\> Sahaj Jhamb

**Directories:** `bots/` and `ui/`  
 **Objective:** Construct heuristic baseline bots with distinct playstyles and integrate them into the existing web application.

* **File: `bots/scholar_bots.py`**  
  * All bots implement `BasePlayer`.  
  * **ChanakyaBot (Aggressive Tactician):** Depth-2 Minimax with alpha-beta pruning. Evaluation weights piece values heavily (King: 1000, Minister: 5, Chariot: 4, Horse: 3, Elephant: 3, Pawn: 1), with large bonuses for giving check, capturing pieces, and reducing the opponent toward a Bare King.  
  * **AryabhataBot (Positional Sage):** Evaluates board states using Piece-Square Tables favoring central control (rows 3–4, cols 3–4), elephant leaping mobility, and pawn advancement toward promotion.  
  * **PaniniBot (Defensive / Solid):** Minimax with defensive penalties for leaving friendly pieces undefended or exposing the King.  
  * **NoviceBot:** Baseline player making 80% random legal moves and 20% direct greedy captures.  
* **Files in `ui/` (`engine_api.py`, `server.py`)**  
  * Exposes an endpoint or dropdown allowing users to select an opponent bot (Chanakya, Aryabhata, Panini, or AlphaZero).  
  * Passes moves between the frontend Ashtapada board and the selected bot instance via `get_move()`.  
* **Standalone Verification:** `bots/test_bots.py` runs an automated 5-game terminal match between ChanakyaBot and AryabhataBot, asserting valid moves and printing the winner.

---

### **Person 4: Benchmarking Suite, Elo Tuning & Failure Analysis**

\-\> Harshal Lahoti

**Directory:** `arena/`  
 **Objective:** Measure agent performance, calibrate varying Elo ratings from the trained CNN, execute ablation experiments on what failed, and prepare presentation figures.

* **File: `arena/tournament.py`**  
  * Runs round-robin matches between all players over NNN games, alternating White and Black to eliminate color advantage.  
  * Computes Win/Loss/Draw matrices, average game lengths, and estimated Elo ratings.  
* **File: `arena/elo_tuning.py` (Simulating Elo Variation)**  
  * Tunes the final single CNN into different skill tiers post-training by adjusting inference parameters:  
    * Master Tier (\~1800 Elo): 80 MCTS simulations, temperature τ=0.05\\tau \= 0.05τ=0.05.  
    * Intermediate Tier (\~1300 Elo): 15 MCTS simulations, temperature τ=0.60\\tau \= 0.60τ=0.60.  
    * Novice Tier (\~700 Elo): 1 MCTS simulation (raw policy distribution), blunder rate ϵ=0.15\\epsilon \= 0.15ϵ=0.15.  
* **File: `arena/ablation_experiments.py` ("What Didn't Work" Section)**  
  * **Experiment 1 (Classical Rollouts vs. Value Network):** Demonstrates that pure random rollouts in Chaturanga frequently enter 100+ move cycles or accidental Murray/Golombek stalemates, producing extreme variance that prevents MCTS convergence.  
  * **Experiment 2 (Unmasked vs. Masked Action Space):** Demonstrates that without masking the 4096-dimensional output, over 95% of policy probability is wasted on illegal moves, crippling gradient efficiency.  
  * **Experiment 3 (Greedy 1-Ply vs. Multi-Step Lookahead):** Demonstrates that 1-ply greedy bots consistently walk into 2-square diagonal elephant leap traps, proving the necessity of tree search.  
* **File: `arena/plots.py`**  
  * Generates publication-ready figures: head-to-head win rate heatmaps, training loss convergence plots, and Elo progression charts.  
* **Standalone Verification:** `arena/test_arena.py` executes a 2-game match between NoviceBot and AryabhataBot and exports a sample plot.

---

### **Final Integration Steps**

1. In `training/mcts_agent.py`, replace the dummy network stub with:  
    python  
2. from neural.model import ChaturangaNet  
3. from neural.encoder import board\_to\_tensor, action\_to\_move, get\_action\_mask  
4.   
5. In `arena/tournament.py` and `ui/engine_api.py`, load the trained checkpoint:  
    python  
6. from neural.model import ChaturangaNet  
7. from training.mcts\_agent import AlphaZeroPlayer  
8. from bots.scholar\_bots import ChanakyaBot, AryabhataBot, PaniniBot  
9. model \= ChaturangaNet()  
10. model.load\_state\_dict(torch.load("training/gen\_final.pt"))  
11. model.eval()  
12. alphazero\_master \= AlphaZeroPlayer(model, num\_sims=80, temperature=0.05)  
13.   
14. Person 4 runs `python3 -m arena.tournament` to produce the final evaluation tables and charts.

