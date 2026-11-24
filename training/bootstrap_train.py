import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chaturanga import (
    ChaturangaEnv, create_starting_board, get_all_legal_moves,
    get_game_outcome, make_move, PAWN, HORSE, ELEPHANT, CHARIOT, MINISTER, KING
)
from neural.model import ChaturangaNet
from neural.encoder import board_to_tensor, move_to_action
from training.train import ChaturangZeroLoss, _get_flip_map, evaluate_probe

VALS = {PAWN: 1.0, ELEPHANT: 1.5, MINISTER: 2.0, HORSE: 3.0, CHARIOT: 5.0, KING: 0.0}
CENTER_BONUS = np.array([
    [0.00, 0.02, 0.04, 0.05, 0.05, 0.04, 0.02, 0.00],
    [0.02, 0.05, 0.08, 0.10, 0.10, 0.08, 0.05, 0.02],
    [0.04, 0.08, 0.12, 0.15, 0.15, 0.12, 0.08, 0.04],
    [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05],
    [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05],
    [0.04, 0.08, 0.12, 0.15, 0.15, 0.12, 0.08, 0.04],
    [0.02, 0.05, 0.08, 0.10, 0.10, 0.08, 0.05, 0.02],
    [0.00, 0.02, 0.04, 0.05, 0.05, 0.04, 0.02, 0.00],
], dtype=np.float32)


def eval_board(board: np.ndarray, player: int) -> float:
    score = 0.0
    for p, v in VALS.items():
        wm = (board == p)
        bm = (board == -p)
        score += v * (np.count_nonzero(wm) - np.count_nonzero(bm))
        if p != KING:
            score += float(CENTER_BONUS[wm].sum() - CENTER_BONUS[bm].sum())
    return score * player


def teacher_2ply(board: np.ndarray, player: int, noise: float = 0.15):
    legal = get_all_legal_moves(board, player)
    if not legal:
        return None, 0.0, legal
    best_score = -1e9
    best_move = legal[0]
    alpha = -1e9
    for move in legal:
        nb = make_move(board, move[0], move[1])
        out = get_game_outcome(nb, -player)
        if out is not None:
            sc = -100.0 * out
        else:
            opp_legal = get_all_legal_moves(nb, -player)
            if not opp_legal:
                sc = eval_board(nb, player)
            else:
                worst = 1e9
                for om in opp_legal:
                    s2 = eval_board(make_move(nb, om[0], om[1]), player)
                    if s2 < worst:
                        worst = s2
                    if worst <= alpha:
                        break
                sc = worst
        sc += random.uniform(-noise, noise)
        if sc > best_score:
            best_score = sc
            best_move = move
        if best_score > alpha:
            alpha = best_score
    return best_move, best_score, legal


def generate_dataset(num_games: int = 200, max_turns: int = 100):
    samples = []
    t0 = time.time()
    for g in range(num_games):
        env = ChaturangaEnv()
        vs_random = (g % 2 == 0)
        rand_side = random.choice([1, -1]) if vs_random else 0
        game_rows = []
        pos_counts = {}

        for turn in range(max_turns):
            b = env.board.copy()
            p = env.current_player
            key = b.tobytes() + bytes([1 if p == 1 else 0])
            pos_counts[key] = pos_counts.get(key, 0) + 1
            if pos_counts[key] >= 3 or get_game_outcome(b, p) is not None:
                break

            best_move, best_score, legal = teacher_2ply(b, p, noise=0.15)
            if best_move is None:
                break

            mask = np.zeros(4096, dtype=bool)
            for m in legal:
                mask[move_to_action(m)] = True
            act_idx = move_to_action(best_move)
            val_target = float(np.tanh(best_score / 8.0))
            game_rows.append((b, p, act_idx, val_target, mask))

            if vs_random and p == rand_side:
                env.step(random.choice(legal))
            else:
                env.step(best_move)

        out = get_game_outcome(env.board, env.current_player)
        winner = 0 if out is None else (env.current_player if out == 1 else -env.current_player)
        for b, p, act_idx, v_eval, mask in game_rows:
            if winner != 0:
                z_term = 1.0 if p == winner else -1.0
                z = 0.65 * v_eval + 0.35 * z_term
            else:
                z = v_eval
            samples.append((b, p, act_idx, float(np.clip(z, -1.0, 1.0)), mask))

        if (g + 1) % 40 == 0:
            print(f"Generated {g + 1}/{num_games} games ({len(samples)} positions) in {time.time() - t0:.1f}s", flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--init-weights", type=str, default="checkpoints/gen_final.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Bootstrap training on {device}...")

    model = ChaturangaNet().to(device)
    if args.init_weights and os.path.exists(args.init_weights):
        try:
            ckpt = torch.load(args.init_weights, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"✓ Warm-starting from existing weights: {args.init_weights}")
        except Exception as e:
            print(f"Warning loading init weights: {e}, starting from scratch.")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "gen_0.pt"))

    samples = generate_dataset(num_games=args.games)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = ChaturangZeroLoss(val_weight=4.0)
    flip_map = _get_flip_map(device)
    log = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        tot_l, val_l, pol_l = [], [], []

        for i in range(0, len(samples), args.batch_size):
            chunk = samples[i:i + args.batch_size]
            if len(chunk) < 2:
                continue
            bx = torch.stack([board_to_tensor(s[0], s[1]) for s in chunk]).to(device).float()
            bpi = torch.zeros((len(chunk), 4096), dtype=torch.float32, device=device)
            for idx, s in enumerate(chunk):
                bpi[idx, s[2]] = 1.0
            bz = torch.tensor([s[3] for s in chunk], dtype=torch.float32, device=device)
            bmask = torch.tensor(np.array([s[4] for s in chunk]), dtype=torch.bool, device=device)

            # Horizontal flip augmentation
            bx = torch.cat([bx, torch.flip(bx, dims=[3])], dim=0)
            bpi = torch.cat([bpi, bpi[:, flip_map]], dim=0)
            bz = torch.cat([bz, bz], dim=0)
            bmask = torch.cat([bmask, bmask[:, flip_map]], dim=0)

            optimizer.zero_grad()
            log_p, v = model(bx, mask=bmask)
            loss, vl, pl = criterion(log_p, v, bpi, bz)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_l.append(loss.item()); val_l.append(vl); pol_l.append(pl)

        scheduler.step()
        mt, mv, mp = float(np.mean(tot_l)), float(np.mean(val_l)), float(np.mean(pol_l))
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {mt:.4f} (Val: {mv:.4f}, Pol: {mp:.4f})", flush=True)
        probe = evaluate_probe(model, device) if epoch in (1, args.epochs // 2, args.epochs) else None
        log.append({"iteration": epoch, "total_loss": mt, "val_loss": mv, "pol_loss": mp, "probe": probe})

        if epoch == args.epochs // 2:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "gen_mid.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "gen_final.pt"))
    with open(os.path.join(args.save_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved {args.save_dir}/gen_final.pt")


if __name__ == "__main__":
    main()
