import random
from typing import List, Tuple, Dict
import numpy as np

from chaturanga import (
    ChaturangaEnv, get_game_outcome, is_in_check,
    WHITE, BLACK, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING
)
from training.mcts_agent import NeuralMCTS, default_move_to_action

try:
    from neural.encoder import move_to_action
except ImportError:
    move_to_action = default_move_to_action

# (board_array, player_turn, target_pi_4096, value_target_z, legal_mask_4096)
TrainingSample = Tuple[np.ndarray, int, np.ndarray, float, np.ndarray]

_PIECE_POOL = [PAWN] * 6 + [HORSE] * 2 + [ELEPHANT] * 2 + [CHARIOT] * 2 + [MINISTER]


def create_curriculum_board(min_pieces: int = 3, max_pieces: int = 8) -> np.ndarray:
    """Generates a valid random midgame/endgame board (pure rules, no heuristics)."""
    for _ in range(100):
        board = np.zeros((8, 8), dtype=np.int8)
        # Place both Kings at least 2 squares apart
        wk = (random.randint(0, 7), random.randint(0, 7))
        bk = (random.randint(0, 7), random.randint(0, 7))
        if max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1:
            continue
        board[wk] = KING
        board[bk] = -KING

        n_white = random.randint(min_pieces, max_pieces)
        n_black = max(2, min(max_pieces, n_white + random.choice([-1, 0, 0, 1])))

        for sign, count in ((WHITE, n_white), (BLACK, n_black)):
            chosen = random.sample(_PIECE_POOL, min(count, len(_PIECE_POOL)))
            for p in chosen:
                for _ in range(30):
                    r = random.randint(1, 6) if p == PAWN else random.randint(0, 7)
                    c = random.randint(0, 7)
                    if board[r, c] == 0:
                        board[r, c] = p * sign
                        break

        if (
            not is_in_check(board, WHITE)
            and not is_in_check(board, BLACK)
            and get_game_outcome(board, WHITE) is None
            and get_game_outcome(board, BLACK) is None
        ):
            return board
    return ChaturangaEnv().board


class ReplayBuffer:
    """Fixed-capacity buffer storing self-play transitions."""
    def __init__(self, max_size: int = 15000):
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
    """Runs pure-RL self-play games with optional curriculum starting positions."""
    def __init__(
        self,
        model,
        num_simulations: int = 25,
        temp_threshold: int = 25,
        max_turns: int = 120,
        curriculum_prob: float = 0.0,
    ):
        self.mcts = NeuralMCTS(model=model, num_simulations=num_simulations)
        self.temp_threshold = temp_threshold
        self.max_turns = max_turns
        self.curriculum_prob = curriculum_prob
        self.last_result: str = "max_turns"

    def play_game(self) -> List[TrainingSample]:
        if hasattr(self.mcts.model, "eval"):
            self.mcts.model.eval()

        env = ChaturangaEnv()
        if self.curriculum_prob > 0.0 and random.random() < self.curriculum_prob:
            env.set_state(create_curriculum_board(), random.choice([WHITE, BLACK]))

        game_history = []
        position_counts: Dict[bytes, int] = {}
        turn_count = 0
        ended_by_rep = False

        while turn_count < self.max_turns:
            board = env.board.copy()
            player = env.current_player

            pos_key = board.tobytes() + bytes([1 if player == 1 else 0])
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
            if position_counts[pos_key] >= 3:
                ended_by_rep = True
                break

            outcome = get_game_outcome(board, player)
            if outcome is not None:
                break

            tau = 1.0 if turn_count < self.temp_threshold else 0.15

            # Pass position_counts into MCTS so the tree natively avoids losing/winning repetitions
            root = self.mcts.search(
                board, player, add_dirichlet=True, past_position_counts=position_counts
            )
            dist = self.mcts.get_action_distribution(root, temperature=tau)
            if not dist:
                break

            pi = np.zeros(4096, dtype=np.float32)
            mask = np.zeros(4096, dtype=bool)
            for move in root.children.keys():
                mask[move_to_action(move)] = True
            for move, prob in dist.items():
                pi[move_to_action(move)] = prob

            game_history.append((board, player, pi, mask))

            moves = list(dist.keys())
            probs = list(dist.values())
            env.step(random.choices(moves, weights=probs, k=1)[0])
            turn_count += 1

        final_outcome = get_game_outcome(env.board, env.current_player)
        if final_outcome is None:
            winner = 0
            self.last_result = "repetition" if ended_by_rep else "max_turns"
        else:
            winner = env.current_player if final_outcome == 1 else -env.current_player
            self.last_result = "white" if winner == 1 else "black"

        samples: List[TrainingSample] = []
        for board_snap, step_player, pi_vec, mask_vec in game_history:
            z = 1.0 if step_player == winner else (-1.0 if winner != 0 else 0.0)
            samples.append((board_snap, step_player, pi_vec, z, mask_vec))

        return samples


SelfPlayWorker = SelfPlayAgent
