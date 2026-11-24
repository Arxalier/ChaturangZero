import argparse
import json
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

    def forward(self, pred_log_p, pred_values, target_pis, target_zs):
        # Value loss: (z - v)^2
        val_loss = self.mse(pred_values.squeeze(-1), target_zs)

        # Policy loss: - sum(pi * log_p) (pred_log_p is already log_softmax from ChaturangaNet)
        pol_loss = -torch.mean(torch.sum(target_pis * pred_log_p, dim=1))
        return val_loss + pol_loss, val_loss.item(), pol_loss.item()


# Backwards compatibility alias
AlphaZeroLoss = ChaturangZeroLoss


# Action permutation indices for horizontal reflection symmetry
_FLIP_ACTION_MAP = None


def _get_flip_map(device):
    """Precomputes or returns cached horizontal reflection action map."""
    global _FLIP_ACTION_MAP
    if _FLIP_ACTION_MAP is None:
        mapping = np.zeros(4096, dtype=np.int64)
        for a in range(4096):
            f, t = divmod(a, 64)
            sr, sc = divmod(f, 8)
            er, ec = divmod(t, 8)
            f_flip = sr * 8 + (7 - sc)
            t_flip = er * 8 + (7 - ec)
            mapping[a] = f_flip * 64 + t_flip
        _FLIP_ACTION_MAP = torch.tensor(mapping, dtype=torch.long)
    return _FLIP_ACTION_MAP.to(device)


def train_step(model, optimizer, criterion, batch, device, augment: bool = True):
    """Performs one gradient update over a sampled mini-batch, with optional symmetry augmentation."""
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

    # Left-right horizontal flip data augmentation (2x data efficiency for free)
    if augment:
        flip_map = _get_flip_map(device)
        batch_x_flip = torch.flip(batch_x, dims=[3])  # Width/column reflection
        batch_pi_flip = batch_pi[:, flip_map]
        batch_x = torch.cat([batch_x, batch_x_flip], dim=0)
        batch_pi = torch.cat([batch_pi, batch_pi_flip], dim=0)
        batch_z = torch.cat([batch_z, batch_z], dim=0)

    optimizer.zero_grad()
    pred_log_p, pred_v = model(batch_x)
    total_loss, v_loss, p_loss = criterion(pred_log_p, pred_v, batch_pi, batch_z)

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), v_loss, p_loss


def run_training(
    num_iterations: int = 20,
    games_per_iter: int = 15,
    sims_per_move: int = 20,
    max_turns: int = 90,
    batch_size: int = 32,
    train_steps_per_iter: int = 20,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints",
    resume_from: str = None
):
    """Main self-play training loop with logging and resume support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting ChaturangZero training on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.json")

    model = ChaturangaNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = ChaturangZeroLoss()
    buffer = ReplayBuffer(max_size=10000)

    start_iteration = 1
    history_log = []

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint.get("iteration", 0) + 1
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    history_log = json.load(f)
            except Exception:
                pass
        print(f"Resuming at iteration {start_iteration}")
    else:
        # Save gen_0 (untrained initialization)
        torch.save(model.state_dict(), os.path.join(save_dir, "gen_0.pt"))
        print(f"Saved initial baseline checkpoint: {save_dir}/gen_0.pt")

    for iteration in range(start_iteration, num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")

        # 1. Self-Play Phase (model in eval mode for search)
        model.eval()
        agent = SelfPlayAgent(model=model, num_simulations=sims_per_move, max_turns=max_turns)
        turns_this_iter = []
        for g in range(games_per_iter):
            game_samples = agent.play_game()
            buffer.push_game(game_samples)
            turns_this_iter.append(len(game_samples))
            print(f"Iter {iteration} | Game {g + 1}/{games_per_iter} completed ({len(game_samples)} turns)")

        avg_game_len = float(np.mean(turns_this_iter)) if turns_this_iter else 0.0

        # 2. Network Optimization Phase
        epoch_losses = []
        epoch_v_losses = []
        epoch_p_losses = []

        if len(buffer) >= batch_size:
            for step in range(train_steps_per_iter):
                batch = buffer.sample_batch(batch_size)
                tot_loss, v_loss, p_loss = train_step(model, optimizer, criterion, batch, device)
                epoch_losses.append(tot_loss)
                epoch_v_losses.append(v_loss)
                epoch_p_losses.append(p_loss)

            mean_tot = float(np.mean(epoch_losses))
            mean_v = float(np.mean(epoch_v_losses))
            mean_p = float(np.mean(epoch_p_losses))
            print(f"Optimization - Avg Loss: {mean_tot:.4f} (Val: {mean_v:.4f}, Pol: {mean_p:.4f})")
        else:
            mean_tot, mean_v, mean_p = 0.0, 0.0, 0.0

        # Record metrics
        iter_record = {
            "iteration": iteration,
            "total_loss": mean_tot,
            "val_loss": mean_v,
            "pol_loss": mean_p,
            "avg_game_length": avg_game_len,
            "buffer_size": len(buffer),
        }
        history_log.append(iter_record)

        # Write training_log.json to disk (Person 4 will plot this)
        with open(log_file, "w") as f:
            json.dump(history_log, f, indent=2)

        # Save checkpoint_latest.pt for crash resilience
        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(save_dir, "checkpoint_latest.pt"))

        # Save mid-way checkpoint
        if iteration == num_iterations // 2:
            torch.save(model.state_dict(), os.path.join(save_dir, "gen_mid.pt"))
            print(f"Saved intermediate checkpoint: {save_dir}/gen_mid.pt")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "gen_final.pt"))
    print(f"\nTraining complete! Final model saved to: {save_dir}/gen_final.pt")
    print(f"Training log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Train ChaturangZero via Self-Play Reinforcement Learning")
    parser.add_argument("--iterations", type=int, default=20, help="Total training iterations (default: 20)")
    parser.add_argument("--games-per-iter", type=int, default=15, help="Self-play games per iteration (default: 15)")
    parser.add_argument("--sims", type=int, default=20, help="MCTS simulations per move (default: 20)")
    parser.add_argument("--max-turns", type=int, default=90, help="Max turns per game before draw (default: 90)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--train-steps", type=int, default=20, help="Gradient steps per iteration (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint_latest.pt to resume from")

    args = parser.parse_args()

    run_training(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        sims_per_move=args.sims,
        max_turns=args.max_turns,
        batch_size=args.batch_size,
        train_steps_per_iter=args.train_steps,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
