import random
from typing import List, Tuple, Dict
import numpy as np

from chaturanga import ChaturangaEnv, get_game_outcome
from training.mcts_agent import NeuralMCTS, default_move_to_action

# Try to use Person 1's encoder if available, otherwise use default
try:
    from neural.encoder import move_to_action
except ImportError:
    move_to_action = default_move_to_action

# Transition tuple: (board_array, player_turn, target_pi_vector_4096, value_target_z)
TrainingSample = Tuple[np.ndarray, int, np.ndarray, float]


class ReplayBuffer:
    """Fixed-capacity buffer storing self-play transitions."""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[TrainingSample] = []

    def push_game(self, samples: List[TrainingSample]) -> None:
        self.buffer.extend(samples)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample_batch(self, batch_size: int) -> List[TrainingSample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class SelfPlayAgent:
    """Runs self-play games to generate training data."""
    def __init__(
        self,
        model,
        num_simulations: int = 25,
        temp_threshold: int = 35,
        max_turns: int = 100,
        rep_penalty: float = -0.15,
    ):
        self.mcts = NeuralMCTS(model=model, num_simulations=num_simulations)
        self.temp_threshold = temp_threshold  # High temp (1.0) for first N moves, 0.25 after
        self.max_turns = max_turns            # Prevents infinite repetition cycles
        self.rep_penalty = rep_penalty        # Small penalty to break 3-fold repetition draw traps
        self.last_result: str = "max_turns"

    def play_game(self) -> List[TrainingSample]:
        # Ensure model is in eval mode during self-play
        if hasattr(self.mcts.model, "eval"):
            self.mcts.model.eval()

        env = ChaturangaEnv()
        game_history = []  # Stores (board, current_player, pi_vector_4096)
        position_counts: Dict[bytes, int] = {}
        turn_count = 0
        ended_by_rep = False

        while turn_count < self.max_turns:
            board = env.board.copy()
            player = env.current_player

            # Check 3-fold repetition draw
            pos_key = board.tobytes() + bytes([1 if player == 1 else 0])
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
            if position_counts[pos_key] >= 3:
                ended_by_rep = True
                break

            outcome = get_game_outcome(board, player)
            if outcome is not None:
                break

            # Tau=1.0 for first 35 turns, Tau=0.25 after (avoids premature deterministic cycling)
            tau = 1.0 if turn_count < self.temp_threshold else 0.25

            # Run MCTS with Dirichlet exploration noise
            root = self.mcts.search(board, player, add_dirichlet=True)
            dist = self.mcts.get_action_distribution(root, temperature=tau)

            if not dist:
                break

            # Build the 4096-dimensional policy vector pi
            pi = np.zeros(4096, dtype=np.float32)
            for move, prob in dist.items():
                act_idx = move_to_action(move)
                pi[act_idx] = prob

            game_history.append((board, player, pi))

            # Pick move sampled from distribution
            moves = list(dist.keys())
            probs = list(dist.values())
            chosen_move = random.choices(moves, weights=probs, k=1)[0]

            env.step(chosen_move)
            turn_count += 1

        # Determine final winner
        final_outcome = get_game_outcome(env.board, env.current_player)
        if final_outcome is None:
            winner = 0
            self.last_result = "repetition" if ended_by_rep else "max_turns"
        else:
            winner = env.current_player if final_outcome == 1 else -env.current_player
            self.last_result = "white" if winner == 1 else "black"

        # Build final training samples with target z (+1 for win, -1 for loss, -0.15 for repetition)
        samples = []
        for board_snap, step_player, pi_vec in game_history:
            if winner != 0:
                z = 1.0 if step_player == winner else -1.0
            else:
                z = self.rep_penalty if ended_by_rep else 0.0
            samples.append((board_snap, step_player, pi_vec, z))

        return samples


# Alias for backward compatibility
SelfPlayWorker = SelfPlayAgent
