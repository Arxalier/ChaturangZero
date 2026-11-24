# loss function, optimizer, checkpointing
# converts batches from ReplayBuffer into PyTorch tensors
# uses multi-task loss, which is used by ChaturangZero as well
# adam optimizer is used.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from training.self_play import ReplayBuffer, SelfPlayAgent
from training.mcts_agent import default_board_to_tensor

# Attempt to import Person 1's model & encoder; use fallbacks if not yet present
try:
    from neural.model import ChaturangaNet
    from neural.encoder import board_to_tensor
except ImportError:
    from training.mcts_agent import DummyNeuralNet as ChaturangaNet
    board_to_tensor = default_board_to_tensor


class ChaturangZeroLoss(nn.Module):
    """Calculates combined Value (MSE) + Policy (Cross-Entropy) loss for ChaturangZero."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_logits, pred_values, target_pis, target_zs):
        # Value loss: (z - v)^2
        val_loss = self.mse(pred_values.squeeze(-1), target_zs)

        # Policy loss: - sum(pi * log_softmax(logits))
        log_probs = torch.log_softmax(pred_logits, dim=1)
        pol_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))

        return val_loss + pol_loss, val_loss.item(), pol_loss.item()


# Backwards compatibility alias
AlphaZeroLoss = ChaturangZeroLoss


def train_step(model, optimizer, criterion, batch, device):
    """Performs one gradient update over a sampled mini-batch."""
    model.train()

    # Unpack batch
    boards, players, pis, zs = zip(*batch)

    # Convert boards to (B, 13, 8, 8) tensor
    tensor_list = [board_to_tensor(b, p).squeeze(0) if torch.is_tensor(board_to_tensor(b, p))
                   else torch.tensor(board_to_tensor(b, p)).squeeze(0)
                   for b, p in zip(boards, players)]
    batch_x = torch.stack(tensor_list).to(device).float()
    batch_pi = torch.tensor(np.array(pis), dtype=torch.float32, device=device)
    batch_z = torch.tensor(np.array(zs), dtype=torch.float32, device=device)

    optimizer.zero_grad()
    pred_logits, pred_v = model(batch_x)
    total_loss, v_loss, p_loss = criterion(pred_logits, pred_v, batch_pi, batch_z)

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), v_loss, p_loss


def run_training(
    num_iterations: int = 10,
    games_per_iter: int = 5,
    sims_per_move: int = 40,
    batch_size: int = 32,
    epochs_per_iter: int = 3,
    save_dir: str = "checkpoints"
):
    """Main self-play training loop (suitable for local run or Google Colab)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting ChaturangZero training on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    model = ChaturangaNet().to(device)

    # Save gen_0 (untrained baseline)
    torch.save(model.state_dict(), os.path.join(save_dir, "gen_0.pt"))
    print(f"Saved initial checkpoint: {save_dir}/gen_0.pt")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = ChaturangZeroLoss()
    buffer = ReplayBuffer(max_size=5000)

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")

        # 1. Self-Play Phase
        agent = SelfPlayAgent(model=model, num_simulations=sims_per_move)
        for g in range(games_per_iter):
            game_samples = agent.play_game()
            buffer.push_game(game_samples)
            print(f"Iteration {iteration} | Game {g + 1}/{games_per_iter} completed ({len(game_samples)} turns)")

        # 2. Network Optimization Phase
        if len(buffer) >= batch_size:
            for epoch in range(epochs_per_iter):
                batch = buffer.sample_batch(batch_size)
                tot_loss, v_loss, p_loss = train_step(model, optimizer, criterion, batch, device)
                print(f"Epoch {epoch + 1}/{epochs_per_iter} - Loss: {tot_loss:.4f} (Val: {v_loss:.4f}, Pol: {p_loss:.4f})")

        # Save mid-way checkpoint
        if iteration == num_iterations // 2:
            torch.save(model.state_dict(), os.path.join(save_dir, "gen_mid.pt"))
            print(f"Saved intermediate checkpoint: {save_dir}/gen_mid.pt")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "gen_final.pt"))
    print(f"\nTraining complete! Final model saved to: {save_dir}/gen_final.pt")


if __name__ == "__main__":
    run_training(num_iterations=2, games_per_iter=2, sims_per_move=10, batch_size=8)