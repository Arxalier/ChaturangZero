"""
training.train
--------------
AlphaZero training loop: masked multi-task policy/value loss, horizontal-flip
data augmentation, live material-probe diagnostics, and checkpoint persistence.
"""

import argparse
import json
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chaturanga import ChaturangaEnv, create_starting_board, get_all_legal_moves
from training.self_play import ReplayBuffer, SelfPlayAgent
from training.mcts_agent import ChaturangPlayer, default_board_to_tensor

try:
    from neural.model import ChaturangaNet
    from neural.encoder import board_to_tensor
except ImportError:
    from training.mcts_agent import DummyNeuralNet as ChaturangaNet
    board_to_tensor = default_board_to_tensor


class ChaturangZeroLoss(nn.Module):
    """Balanced multi-task loss: val_weight * MSE(v, z) + Masked Cross-Entropy(log_p, pi)."""
    def __init__(self, val_weight: float = 4.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.val_weight = val_weight

    def forward(self, pred_log_p, pred_values, target_pis, target_zs):
        val_loss = self.mse(pred_values.squeeze(-1), target_zs)
        # Safe masked cross-entropy (ignores zero-probability illegal indices)
        safe_terms = torch.where(target_pis > 0, target_pis * pred_log_p, torch.zeros_like(pred_log_p))
        pol_loss = -torch.mean(torch.sum(safe_terms, dim=1))
        total = self.val_weight * val_loss + pol_loss
        return total, val_loss.item(), pol_loss.item()


AlphaZeroLoss = ChaturangZeroLoss
_FLIP_ACTION_MAP = None


def _get_flip_map(device):
    global _FLIP_ACTION_MAP
    if _FLIP_ACTION_MAP is None:
        mapping = np.zeros(4096, dtype=np.int64)
        for a in range(4096):
            f, t = divmod(a, 64)
            sr, sc = divmod(f, 8)
            er, ec = divmod(t, 8)
            mapping[a] = (sr * 8 + (7 - sc)) * 64 + (er * 8 + (7 - ec))
        _FLIP_ACTION_MAP = torch.tensor(mapping, dtype=torch.long)
    return _FLIP_ACTION_MAP.to(device)


def train_step(model, optimizer, criterion, batch, device, augment: bool = True):
    """Performs one masked gradient update with horizontal flip augmentation."""
    model.train()

    boards, players, pis, zs, masks = zip(*batch)

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
    batch_mask = torch.tensor(np.array(masks), dtype=torch.bool, device=device)

    if augment:
        flip_map = _get_flip_map(device)
        batch_x = torch.cat([batch_x, torch.flip(batch_x, dims=[3])], dim=0)
        batch_pi = torch.cat([batch_pi, batch_pi[:, flip_map]], dim=0)
        batch_z = torch.cat([batch_z, batch_z], dim=0)
        batch_mask = torch.cat([batch_mask, batch_mask[:, flip_map]], dim=0)

    optimizer.zero_grad()
    pred_log_p, pred_v = model(batch_x, mask=batch_mask)
    total_loss, v_loss, p_loss = criterion(pred_log_p, pred_v, batch_pi, batch_z)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), v_loss, p_loss


def evaluate_probe(model, device):
    """Live diagnostic: checks whether Value Head distinguishes material advantage."""
    model.eval()
    b0 = create_starting_board()
    b_up = b0.copy(); b_up[0, 0] = 0      # White up a Chariot
    b_dn = b0.copy(); b_dn[7, 0] = 0      # White down a Chariot

    vals = []
    with torch.no_grad():
        for b in (b0, b_up, b_dn):
            t = board_to_tensor(b, 1)
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            t = t.to(device).float()
            _, v = model(t)
            vals.append(float(v.item()))
    print(f"  [Probe] Start: {vals[0]:+.3f} | Up Chariot: {vals[1]:+.3f} | Down Chariot: {vals[2]:+.3f}")
    return {"start": vals[0], "up_chariot": vals[1], "down_chariot": vals[2]}


def run_training(
    num_iterations: int = 30,
    games_per_iter: int = 20,
    sims_per_move: int = 25,
    max_turns: int = 120,
    batch_size: int = 64,
    train_steps_per_iter: int = 30,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints",
    resume_from: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Pure-RL ChaturangZero training on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.json")

    model = ChaturangaNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-4)
    criterion = ChaturangZeroLoss(val_weight=4.0)
    buffer = ReplayBuffer(max_size=15000)

    start_iteration = 1
    history_log = []

    if resume_from and os.path.exists(resume_from):
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
        print(f"Resumed from {resume_from} at iteration {start_iteration}")
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "gen_0.pt"))
        print(f"Saved initial baseline checkpoint: {save_dir}/gen_0.pt")

    for iteration in range(start_iteration, num_iterations + 1):
        # Pure-RL curriculum schedule: start 75% endgame/midgame -> fade to 20%
        progress = (iteration - 1) / max(1, num_iterations - 1)
        curriculum_prob = max(0.20, 0.75 - 0.55 * progress)

        print(f"\n--- Iteration {iteration}/{num_iterations} (curriculum={curriculum_prob:.0%}) ---")
        model.eval()
        agent = SelfPlayAgent(
            model=model,
            num_simulations=sims_per_move,
            max_turns=max_turns,
            curriculum_prob=curriculum_prob,
        )

        turns_this_iter = []
        outcomes = {"white": 0, "black": 0, "repetition": 0, "max_turns": 0}
        decisive_games = 0
        draw_games_pushed = 0

        for g in range(games_per_iter):
            game_samples = agent.play_game()
            res = agent.last_result
            outcomes[res] = outcomes.get(res, 0) + 1
            turns_this_iter.append(len(game_samples))

            # Always keep decisive wins/losses; cap draw games at 25% so draws never drown the buffer
            if res in ("white", "black"):
                buffer.push_game(game_samples)
                decisive_games += 1
            elif draw_games_pushed < max(2, games_per_iter // 4):
                buffer.push_game(game_samples)
                draw_games_pushed += 1

        print(
            f"Iter {iteration} Outcomes: W={outcomes['white']}, B={outcomes['black']}, "
            f"Rep={outcomes['repetition']}, Cap={outcomes['max_turns']} "
            f"(avg {np.mean(turns_this_iter):.1f} turns, buf={len(buffer)})"
        )

        epoch_losses, epoch_v_losses, epoch_p_losses = [], [], []
        if len(buffer) >= batch_size:
            for _ in range(train_steps_per_iter):
                batch = buffer.sample_batch(batch_size)
                tot_loss, v_loss, p_loss = train_step(model, optimizer, criterion, batch, device)
                epoch_losses.append(tot_loss)
                epoch_v_losses.append(v_loss)
                epoch_p_losses.append(p_loss)
            scheduler.step()

            mean_tot = float(np.mean(epoch_losses))
            mean_v = float(np.mean(epoch_v_losses))
            mean_p = float(np.mean(epoch_p_losses))
            print(f"Optimization - Avg Loss: {mean_tot:.4f} (Val: {mean_v:.4f}, Pol: {mean_p:.4f})")
        else:
            mean_tot = mean_v = mean_p = 0.0

        probe = evaluate_probe(model, device) if (iteration % 3 == 0 or iteration == 1 or iteration == num_iterations) else None

        history_log.append({
            "iteration": iteration,
            "total_loss": mean_tot,
            "val_loss": mean_v,
            "pol_loss": mean_p,
            "avg_game_length": float(np.mean(turns_this_iter)),
            "buffer_size": len(buffer),
            "outcomes": outcomes,
            "probe": probe,
        })
        with open(log_file, "w") as f:
            json.dump(history_log, f, indent=2)

        torch.save({
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(save_dir, "checkpoint_latest.pt"))

        if iteration == num_iterations // 2:
            torch.save(model.state_dict(), os.path.join(save_dir, "gen_mid.pt"))
            print(f"Saved intermediate checkpoint: {save_dir}/gen_mid.pt")

    torch.save(model.state_dict(), os.path.join(save_dir, "gen_final.pt"))
    print(f"\nTraining complete! Final model saved to: {save_dir}/gen_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Train ChaturangZero via Pure Self-Play RL")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--sims", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
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
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
