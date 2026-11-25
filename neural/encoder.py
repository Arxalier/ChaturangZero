"""
neural.encoder
--------------
Translates 8x8 Chaturanga boards into (13, 8, 8) PyTorch tensors and flattens
board moves into a 4,096-action index space with legal move masking.
"""

import numpy as np, torch
from chaturanga import get_all_legal_moves

def board_to_tensor(board: np.ndarray, player: int) -> torch.Tensor:
    t = np.zeros((13, 8, 8), dtype=np.float32)
    for k in range(1, 7):
        t[k - 1] = (board == k)        # White pieces (positive values)
        t[6 + k - 1] = (board == -k)   # Black pieces (negative values)
    if player == 1:
        t[12] = 1.0
    return torch.from_numpy(t)

def move_to_action(move) -> int:
    (sr, sc), (er, ec) = move
    return (sr * 8 + sc) * 64 + (er * 8 + ec)

def action_to_move(a: int):
    f, t = divmod(int(a), 64)
    return (divmod(f, 8), divmod(t, 8))

def get_action_mask(board: np.ndarray, player: int) -> np.ndarray:
    mask = np.zeros(4096, dtype=bool)
    for m in get_all_legal_moves(board, player):
        mask[move_to_action(m)] = True
    return mask