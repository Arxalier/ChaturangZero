import math
import random
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

try:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    _ModuleBase = object

# Import universal player contract and engine rules
from chaturanga import BasePlayer, get_all_legal_moves, get_game_outcome, make_move

# Fallback default encoder utilities if neural.encoder is not present
def default_move_to_action(move: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
    (sr, sc), (er, ec) = move
    return ((sr * 8 + sc) * 64) + (er * 8 + ec)

def default_action_to_move(action_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    start_sq = action_idx // 64
    end_sq = action_idx % 64
    return (start_sq // 8, start_sq % 8), (end_sq // 8, end_sq % 8)

def default_board_to_tensor(board: np.ndarray, player: int):
    if torch is not None:
        tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
        for r in range(8):
            for c in range(8):
                val = board[r, c]
                if val > 0:
                    tensor[val - 1, r, c] = 1.0
                elif val < 0:
                    tensor[abs(val) + 5, r, c] = 1.0
        if player == 1:
            tensor[12, :, :] = 1.0
        return tensor  # Shape (13, 8, 8)
    return np.zeros((13, 8, 8), dtype=np.float32)

class DummyNeuralNet(_ModuleBase):
    """Fallback stub matching ChaturangaNet interface."""
    def __init__(self):
        super().__init__()
        if torch is not None:
            self.conv = nn.Conv2d(13, 16, kernel_size=3, padding=1)
            self.pol = nn.Linear(16 * 8 * 8, 4096)
            self.val = nn.Linear(16 * 8 * 8, 1)

    def forward(self, x, mask=None):
        if torch is None:
            return np.zeros((1, 4096)), np.zeros((1, 1))
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h = torch.relu(self.conv(x)).flatten(1)
        p = self.pol(h)
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)
        log_p = torch.log_softmax(p, dim=1)
        v = torch.tanh(self.val(h))
        return log_p, v

    def __call__(self, tensor, mask=None):
        return self.forward(tensor, mask=mask)


# Top-level encoder & model resolution
try:
    from neural.encoder import board_to_tensor, move_to_action
except ImportError:
    board_to_tensor = default_board_to_tensor
    move_to_action = default_move_to_action

try:
    from neural.model import ChaturangaNet
except ImportError:
    ChaturangaNet = DummyNeuralNet


Move = Tuple[Tuple[int, int], Tuple[int, int]]


class MCTSNode:
    """Represents a board state node in the search tree."""
    def __init__(self, current_player: int, prior_prob: float = 0.0, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        self.parent = parent
        self.move = move
        self.children: Dict[Move, 'MCTSNode'] = {}
        self.current_player = current_player

        self.N: int = 0            # Visit count
        self.W: float = 0.0        # Total accumulated value from child perspective
        self.Q: float = 0.0        # Mean action value (W / N)
        self.P: float = prior_prob   # Prior probability from policy head

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, action_probs: Dict[Move, float]) -> None:
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = MCTSNode(
                    current_player=-self.current_player,
                    prior_prob=prob,
                    parent=self,
                    move=move
                )

    def get_puct_value(self, c_puct: float = 1.25) -> float:
        parent_visits = self.parent.N if self.parent else 1
        exploration_term = c_puct * self.P * (math.sqrt(parent_visits) / (1 + self.N))
        # Critical fix: self.Q is stored from child perspective (opponent).
        # For the parent choosing this action, expected value is -self.Q.
        return -self.Q + exploration_term

    def update(self, value: float) -> None:
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class NeuralMCTS:
    """MCTS driven by a dual-headed policy/value neural network."""
    def __init__(self, model=None, c_puct: float = 1.25, num_simulations: int = 40, draw_penalty: float = 0.0):
        if model is not None:
            self.model = model
        else:
            try:
                self.model = ChaturangaNet()
            except Exception:
                self.model = DummyNeuralNet()
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.draw_penalty = draw_penalty

        # Cache device once to eliminate repeated parameter inspections
        if torch is not None and hasattr(self.model, "parameters"):
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = "cpu"

    def search(
        self,
        initial_board: np.ndarray,
        current_player: int,
        add_dirichlet: bool = False,
        past_position_counts: Optional[Dict[bytes, int]] = None
    ) -> MCTSNode:
        """
        Executes MCTS simulations from current root.
        Natively evaluates 3-fold repetition as terminal draw (0.0).
        """
        root_node = MCTSNode(current_player=current_player)

        # Expand root immediately
        legal_moves = get_all_legal_moves(initial_board, current_player)
        if not legal_moves:
            return root_node

        action_probs, _ = self._evaluate(initial_board, current_player, legal_moves)

        # Add exploration noise to the root node if in self-play mode
        if add_dirichlet and len(legal_moves) > 0:
            alpha = 0.3
            epsilon = 0.25
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                action_probs[move] = (1 - epsilon) * action_probs[move] + epsilon * noise[i]

        root_node.expand(action_probs)

        # Base history tracking for 3-fold repetition awareness in search tree
        root_key = initial_board.tobytes() + bytes([1 if current_player == 1 else 0])
        base_counts = dict(past_position_counts) if past_position_counts else {}
        if root_key not in base_counts:
            base_counts[root_key] = 1

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root_node
            board = initial_board.copy()
            search_path = [node]
            sim_counts = dict(base_counts)
            is_rep_draw = False

            # 1. Selection
            while node.is_expanded():
                best_move, best_child = max(
                    node.children.items(),
                    key=lambda item: item[1].get_puct_value(self.c_puct)
                )
                board = make_move(board, best_move[0], best_move[1])
                node = best_child
                search_path.append(node)

                pos_key = board.tobytes() + bytes([1 if node.current_player == 1 else 0])
                sim_counts[pos_key] = sim_counts.get(pos_key, 0) + 1
                if sim_counts[pos_key] >= 3:
                    is_rep_draw = True
                    break

            # 2. Evaluation & Expansion
            if is_rep_draw:
                # Positive value from leaf_player (-mover) perspective penalizes mover by -draw_penalty
                value = float(self.draw_penalty)
            else:
                legal = get_all_legal_moves(board, node.current_player)
                outcome = get_game_outcome(board, node.current_player, legal_moves=legal)

                if outcome is not None:
                    value = float(outcome)
                elif not legal:
                    value = 0.0
                else:
                    action_probs, leaf_value = self._evaluate(board, node.current_player, legal)
                    node.expand(action_probs)
                    value = leaf_value

            # 3. Backpropagation (zero-sum alternating perspective)
            leaf_player = node.current_player
            for n in reversed(search_path):
                n_value = value if n.current_player == leaf_player else -value
                n.update(n_value)

        return root_node

    def _evaluate(self, board: np.ndarray, player: int, legal_moves: List[Move]) -> Tuple[Dict[Move, float], float]:
        """Queries the network for legal move probabilities and position value."""
        inp_tensor = board_to_tensor(board, player)

        if torch is not None:
            if not isinstance(inp_tensor, torch.Tensor):
                inp_tensor = torch.tensor(inp_tensor, dtype=torch.float32)
            if inp_tensor.ndim == 3:
                inp_tensor = inp_tensor.unsqueeze(0)  # Shape (1, 13, 8, 8)

            # Ensure model is in eval mode during search to prevent BatchNorm errors
            if hasattr(self.model, "eval"):
                self.model.eval()

            inp_tensor = inp_tensor.to(self.device)

            with torch.no_grad():
                out = self.model(inp_tensor)
                policy_logits, value_tensor = out[0], out[1]
                policy_np = policy_logits.squeeze(0).cpu().numpy()
                val = float(value_tensor.item())
        else:
            policy_np = np.zeros(4096, dtype=np.float32)
            val = 0.0

        # Mask and softmax over strictly legal moves
        action_indices = [move_to_action(m) for m in legal_moves]
        sub_logits = np.array([policy_np[idx] for idx in action_indices], dtype=np.float64)

        # Numerically stable softmax
        exp_logits = np.exp(sub_logits - np.max(sub_logits))
        sum_exp = np.sum(exp_logits)
        probs = exp_logits / (sum_exp if sum_exp > 0 else 1.0)

        action_probs = {move: float(probs[i]) for i, move in enumerate(legal_moves)}
        return action_probs, val

    def get_action_distribution(self, root_node: MCTSNode, temperature: float = 1.0) -> Dict[Move, float]:
        """Calculates pi = N^(1/tau) / sum(N^(1/tau)) for training."""
        if not root_node.children:
            return {}

        moves = list(root_node.children.keys())
        visits = np.array([root_node.children[m].N for m in moves], dtype=np.float64)

        # Low temperature: deterministic argmax to prevent float overflow
        if temperature < 0.05:
            best_idx = int(np.argmax(visits))
            return {moves[best_idx]: 1.0}

        # Prevent overflow by normalizing with max visits
        visits_norm = visits / np.max(visits)
        visits_temp = visits_norm ** (1.0 / max(temperature, 1e-4))
        sum_visits = np.sum(visits_temp)
        probs = visits_temp / (sum_visits if sum_visits > 0 else 1.0)

        return {moves[i]: float(probs[i]) for i in range(len(moves))}


class ChaturangPlayer(BasePlayer):
    """Universal player wrapper implementing BasePlayer with configurable Elo degradation and game-theoretic repetition awareness."""
    def __init__(
        self,
        model=None,
        num_sims: int = 40,
        temperature: float = 0.1,
        top_k: Optional[int] = None,
        blunder_rate: float = 0.0,
        draw_penalty: float = 0.0
    ):
        self.num_sims = num_sims
        self.temperature = temperature
        self.top_k = top_k
        self.blunder_rate = blunder_rate
        self.draw_penalty = draw_penalty
        self.position_counts: Dict[bytes, int] = {}
        self.mcts = NeuralMCTS(model=model, num_simulations=num_sims, draw_penalty=draw_penalty)

    def reset(self):
        """Clears visited positions for a fresh game."""
        self.position_counts.clear()

    def get_move(
        self,
        board: np.ndarray,
        player: int,
        past_position_counts: Optional[Dict[bytes, int]] = None
    ) -> Move:
        legal_moves = get_all_legal_moves(board, player)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Blunder injection (optional random move)
        if self.blunder_rate > 0.0 and random.random() < self.blunder_rate:
            return random.choice(legal_moves)

        pos_key = board.tobytes() + bytes([1 if player == 1 else 0])
        self.position_counts[pos_key] = self.position_counts.get(pos_key, 0) + 1

        effective_counts = dict(self.position_counts)
        if past_position_counts:
            effective_counts.update(past_position_counts)

        # 1. Novice Tier: num_sims == 0 (sample directly from raw policy head with zero lookahead)
        if self.num_sims == 0:
            action_probs, _ = self.mcts._evaluate(board, player, legal_moves)
            moves = list(action_probs.keys())
            probs = np.array([action_probs[m] for m in moves], dtype=np.float64)

            # Apply top-k filtering if specified
            if self.top_k is not None and 0 < self.top_k < len(moves):
                top_idx = np.argsort(probs)[-self.top_k:]
                mask = np.zeros_like(probs, dtype=bool)
                mask[top_idx] = True
                probs[~mask] = 0.0
                sum_p = probs.sum()
                probs = probs / (sum_p if sum_p > 0 else 1.0)

            if self.temperature < 0.05:
                return moves[int(np.argmax(probs))]

            # Safeguard: normalize by max probability before temperature scaling to prevent underflow
            probs = probs / np.max(probs)
            probs = probs ** (1.0 / max(self.temperature, 1e-4))
            sum_p = probs.sum()
            probs = probs / (sum_p if sum_p > 0 else 1.0)
            return random.choices(moves, weights=probs.tolist(), k=1)[0]

        # 2. Normal MCTS Lookahead Search (Intermediate / Master Tiers)
        # Search is natively value-aware of 3-fold repetition: seeks repetition if losing (<0), avoids if winning (>0)
        root = self.mcts.search(board, player, add_dirichlet=False, past_position_counts=effective_counts)
        dist = self.mcts.get_action_distribution(root, temperature=self.temperature)

        if not dist:
            return random.choice(legal_moves)

        moves = list(dist.keys())
        probs = np.array([dist[m] for m in moves], dtype=np.float64)

        # Apply top-k filtering
        if self.top_k is not None and 0 < self.top_k < len(moves):
            top_idx = np.argsort(probs)[-self.top_k:]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_idx] = True
            probs[~mask] = 0.0
            sum_p = probs.sum()
            probs = probs / (sum_p if sum_p > 0 else 1.0)

        return random.choices(moves, weights=probs.tolist(), k=1)[0]


ChaturangZeroPlayer = ChaturangPlayer
AlphaZeroPlayer = ChaturangPlayer  # Backward compatibility
