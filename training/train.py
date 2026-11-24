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

    def forward(self, pred_policy, pred_values, target_pis, target_zs):
        # Value loss: (z - v)^2
        val_loss = self.mse(pred_values.squeeze(-1), target_zs)

        # Policy loss:
        # If output from model is raw logits, compute log_softmax.
        # If already log_softmax (<= 0), use directly to prevent double log_softmax.
        if pred_policy.max() > 0.0:
            log_probs = torch.log_softmax(pred_policy, dim=1)
        else:
            log_probs = pred_policy

        pol_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))
        return val_loss + pol_loss, val_loss.item(), pol_loss.item()


# Backwards compatibility alias
AlphaZeroLoss = ChaturangZeroLoss


def train_step(model, optimizer, criterion, batch, device):
    """Performs one gradient update over a sampled mini-batch."""
    model.train()

    # Unpack batch
    boards, players, pis, zs = zip(*batch)

    # Convert boards to (B, 13, 8, 8) tensor without redundant calls
    tensor_list = []
    for b, p in zip(boards, players):
        t = board_to_tensor(b, p)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        if t.ndim == 4:
            t = t.squeeze(0)
        tensor_list.append(t)

    batch_x = torch.stack(tensor_list).to(device).float()
    batch_pi = torch.tensor(np.array(pis), dtype=torch.float32, device=device)
    batch_z = torch.tensor(np.array(zs), dtype=torch.float32, device=device)

    optimizer.zero_grad()
    pred_policy, pred_v = model(batch_x)
    total_loss, v_loss, p_loss = criterion(pred_policy, pred_v, batch_pi, batch_z)

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

        # 1. Self-Play Phase (ensure model is in eval mode during MCTS)
        model.eval()
        agent = SelfPlayAgent(model=model, num_simulations=sims_per_move)
        for g in range(games_per_iter):
            game_samples = agent.play_game()
            buffer.push_game(game_samples)
            print(f"Iteration {iteration} | Game {g + 1}/{games_per_iter} completed ({len(game_samples)} turns)")

        # 2. Network Optimization Phase
        if len(buffer) >= batch_size:
            train_steps = max(epochs_per_iter, 5)
            for step in range(train_steps):
                batch = buffer.sample_batch(batch_size)
                tot_loss, v_loss, p_loss = train_step(model, optimizer, criterion, batch, device)
                print(f"Step {step + 1}/{train_steps} - Loss: {tot_loss:.4f} (Val: {v_loss:.4f}, Pol: {p_loss:.4f})")

        # Save mid-way checkpoint
        if iteration == num_iterations // 2:
            torch.save(model.state_dict(), os.path.join(save_dir, "gen_mid.pt"))
            print(f"Saved intermediate checkpoint: {save_dir}/gen_mid.pt")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "gen_final.pt"))
    print(f"\nTraining complete! Final model saved to: {save_dir}/gen_final.pt")


if __name__ == "__main__":
    run_training(num_iterations=2, games_per_iter=2, sims_per_move=10, batch_size=8)
