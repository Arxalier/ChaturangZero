"""
arena.ablation_experiments
--------------------------
"What didn't work": three experiments that test the design choices behind
ChaturangZero. Each one measures; none of them assumes its conclusion. The
report states whatever the numbers show.

Experiment 1  Classical random rollouts vs. a value network
    1a  Random playouts: how long are they, how often do they pass 100 plies,
        and how often do they end in the Murray/Golombek stalemate win?
    1b  Rollout value estimates: their variance, and how well they (and the
        value network, and plain material count) predict the result that a
        stronger player (Minimax depth 2 vs itself) actually gets from the
        same position.
    1c  Rollout-MCTS: cost per simulation and how stable its chosen move is
        across random seeds, for growing simulation budgets.
    1d  (needs --checkpoint) Rollout-MCTS vs. network-MCTS head to head at
        the same simulation budget.

Experiment 2  Unmasked vs. masked 4096-way policy
    2a  How much of an unmasked softmax lands on illegal moves, for an
        untrained network and (if given) the trained checkpoint.
    2b  Controlled training run: two identical networks learn to imitate a
        Minimax teacher on the same data and batches, one with the legal-move
        mask applied before log-softmax and one without.

Experiment 3  Greedy 1-ply vs. multi-step lookahead
    3a  Elephant-trap suite: positions where a capture can be immediately
        punished by a 2-square diagonal Elephant leap (with a net material
        loss) while a safe alternative exists. Measures how often each player
        takes the poisoned capture.
    3b  In-game rate of Elephant-leap blunders per 100 moves, and the
        Greedy vs. Minimax head-to-head score.

    python -m arena.ablation_experiments                       # all, default sizes
    python -m arena.ablation_experiments --quick               # small smoke run
    python -m arena.ablation_experiments --exp 2 --checkpoint checkpoints/gen_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chaturanga import (
    ELEPHANT,
    WHITE,
    create_starting_board,
    get_all_legal_moves,
    get_game_outcome,
    make_move,
)

from . import plots
from .match import (BARE_KING, CHECKMATE, STALEMATE, classify_terminal, play_game, reset_player,
                    seed_everything)
from .players import (
    GreedyPlayer,
    MinimaxPlayer,
    RandomPlayer,
    build_player,
    load_model,
    material_balance,
    resolve_checkpoint,
)

Move = Tuple[Tuple[int, int], Tuple[int, int]]
Position = Tuple[np.ndarray, int]  # (board, side to move)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_worker_init)


def _map(fn, items, workers: int):
    items = list(items)
    if workers <= 1 or len(items) < 4:
        return [fn(x) for x in items]
    with _pool(workers) as pool:
        chunk = max(1, len(items) // (workers * 4))
        return list(pool.map(fn, items, chunksize=chunk))


def random_playout(board: np.ndarray, player: int, rng: random.Random,
                   cap: Optional[int] = None) -> Tuple[Optional[int], str, int]:
    """
    Plays uniformly random moves until the game ends.
    Returns (winner or None if capped, reason, plies).
    """
    plies = 0
    while cap is None or plies < cap:
        legal = get_all_legal_moves(board, player)
        term = classify_terminal(board, player, legal)
        if term is not None:
            return term[0], term[1], plies
        m = rng.choice(legal)
        board = make_move(board, m[0], m[1])
        player = -player
        plies += 1
    return None, "cap", plies


def _sample_positions_worker(args) -> List[Tuple[list, int, int]]:
    seed, plies_lo, plies_hi = args
    rng = random.Random(seed)
    seed_everything(seed)
    # Mixed-strength games give a spread of realistic positions.
    kinds = [GreedyPlayer(), MinimaxPlayer(2), RandomPlayer()]
    white, black = rng.choice(kinds[:2]), rng.choice(kinds)
    if rng.random() < 0.5:
        white, black = black, white
    board, player = create_starting_board(), WHITE
    for _ in range(rng.randint(2, 6)):  # random opening
        legal = get_all_legal_moves(board, player)
        m = rng.choice(legal)
        board, player = make_move(board, m[0], m[1]), -player
    out = []
    targets = set(rng.sample(range(plies_lo, plies_hi), k=3))
    for ply in range(plies_hi):
        legal = get_all_legal_moves(board, player)
        if get_game_outcome(board, player, legal) is not None:
            break
        if ply in targets:
            out.append((board.tolist(), player, ply))
        agent = white if player == WHITE else black
        m = agent.get_move(board.copy(), player)
        board, player = make_move(board, m[0], m[1]), -player
    return out


def sample_positions(n: int, seed: int, workers: int, plies=(6, 70)) -> List[Position]:
    """Non-terminal positions taken from games between baseline players."""
    games = max(1, math.ceil(n / 3 * 1.3))
    chunks = _map(_sample_positions_worker, [(seed * 100_003 + g, *plies) for g in range(games)], workers)
    positions = [(np.array(b, dtype=np.int8), p) for chunk in chunks for b, p, _ in chunk]
    random.Random(seed).shuffle(positions)
    return positions[:n]


def _net_eval(model, board: np.ndarray, player: int):
    import torch
    from neural.encoder import board_to_tensor

    with torch.no_grad():
        log_p, v = model(board_to_tensor(board, player).unsqueeze(0))
    return log_p[0].numpy(), float(v.item())


def _write(out_dir: str, name: str, payload: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
# Rollout MCTS (same PUCT search as training/mcts_agent.py, random-rollout leaves)
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("player", "P", "N", "W", "children")

    def __init__(self, player: int, prior: float):
        self.player, self.P, self.N, self.W, self.children = player, prior, 0, 0.0, {}

    def q(self) -> float:
        return self.W / self.N if self.N else 0.0


class RolloutMCTSPlayer:
    """
    PUCT search with uniform priors whose leaves are scored by ONE random
    playout (capped at ``rollout_cap`` plies, cap = draw). Everything else
    matches the network-guided search, so only the evaluator differs.
    """

    def __init__(self, sims: int = 50, c_puct: float = 1.25, rollout_cap: int = 120,
                 seed: Optional[int] = None):
        self.sims, self.c_puct, self.rollout_cap = sims, c_puct, rollout_cap
        self.rng = random.Random(seed)

    def search(self, board: np.ndarray, player: int) -> _Node:
        root = _Node(player, 1.0)
        for _ in range(self.sims):
            node, b, path = root, board.copy(), [root]
            while node.children:
                sqrt_n = math.sqrt(node.N)
                move, node = max(node.children.items(),
                                 key=lambda kv: -kv[1].q() + self.c_puct * kv[1].P * sqrt_n / (1 + kv[1].N))
                b = make_move(b, move[0], move[1])
                path.append(node)
            legal = get_all_legal_moves(b, node.player)
            outcome = get_game_outcome(b, node.player, legal)
            if outcome is not None:
                value = float(outcome)
            else:
                for m in legal:
                    node.children[m] = _Node(-node.player, 1.0 / len(legal))
                winner, _, _ = random_playout(b, node.player, self.rng, self.rollout_cap)
                value = 0.0 if winner is None else (1.0 if winner == node.player else -1.0)
            for n in reversed(path):
                n.N += 1
                n.W += value if n.player == node.player else -value
        return root

    def get_move(self, board: np.ndarray, player: int) -> Move:
        root = self.search(board, player)
        if not root.children:
            return self.rng.choice(get_all_legal_moves(board, player))
        best = max(c.N for c in root.children.values())
        return self.rng.choice([m for m, c in root.children.items() if c.N == best])


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------
def _playout_stats_worker(seed: int):
    rng = random.Random(seed)
    winner, reason, plies = random_playout(create_starting_board(), WHITE, rng, cap=2000)
    return reason, plies


def _rollout_values_worker(args):
    board_list, player, k, seed = args
    rng = random.Random(seed)
    board = np.array(board_list, dtype=np.int8)
    vals = []
    for _ in range(k):
        w, _, _ = random_playout(board, player, rng, cap=120)
        vals.append(0.0 if w is None else (1.0 if w == player else -1.0))
    return vals


def _ground_truth_worker(args):
    """Average result for the side to move when Minimax-d2 plays both sides."""
    board_list, player, games, seed = args
    board = np.array(board_list, dtype=np.int8)
    total = 0.0
    for g in range(games):
        r = play_game(MinimaxPlayer(2), MinimaxPlayer(2), "w", "b", max_plies=120,
                      seed=seed + g, start_board=board, start_player=player)
        total += 0.0 if r.winner == 0 else (1.0 if r.winner == player else -1.0)
    return total / games


def _stability_worker(args):
    board_list, player, sims, seed = args
    board = np.array(board_list, dtype=np.int8)
    t0 = time.perf_counter()
    m = RolloutMCTSPlayer(sims=sims, seed=seed).get_move(board, player)
    return m, time.perf_counter() - t0


def experiment_1(out: str, workers: int, seed: int, ckpt: Optional[str], quick: bool) -> dict:
    print("\n=== Experiment 1: random rollouts vs value network ===")
    report: Dict[str, object] = {}

    # 1a. Playout statistics from the starting position.
    n_play = 200 if quick else 2000
    stats = _map(_playout_stats_worker, [seed * 7919 + i for i in range(n_play)], workers)
    lengths = np.array([p for _, p in stats])
    reasons = Counter(r for r, _ in stats)
    report["1a_playouts"] = {
        "n": n_play,
        "mean_plies": float(lengths.mean()), "median_plies": float(np.median(lengths)),
        "p90_plies": float(np.percentile(lengths, 90)),
        "share_over_100_plies": float((lengths > 100).mean()),
        "share_over_120_plies": float((lengths > 120).mean()),
        "end_reasons": {k: v / n_play for k, v in reasons.items()},
    }
    print(json.dumps(report["1a_playouts"], indent=2))
    report["1a_playouts"]["lengths"] = lengths.tolist()

    # 1b. Variance and predictive power of rollout estimates.
    n_pos = 16 if quick else 60
    k_max = 32 if quick else 128
    gt_games = 2 if quick else 6
    positions = sample_positions(n_pos, seed, workers)
    print(f"1b: {len(positions)} positions, {k_max} rollouts each, {gt_games} Minimax games each for ground truth")
    rollouts = _map(_rollout_values_worker,
                    [(b.tolist(), p, k_max, seed + i) for i, (b, p) in enumerate(positions)], workers)
    truth = np.array(_map(_ground_truth_worker,
                          [(b.tolist(), p, gt_games, seed * 31 + i) for i, (b, p) in enumerate(positions)],
                          workers))
    R = np.array(rollouts)  # (positions, k)
    material = np.array([material_balance(b, p) for b, p in positions])

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= k_max]
    rng = np.random.default_rng(seed)
    rollout_corr, rollout_corr_lo, rollout_corr_hi = [], [], []
    for k in ks:
        cs = []
        for _ in range(30):
            idx = rng.choice(k_max, size=k, replace=False)
            cs.append(corr(R[:, idx].mean(axis=1), truth))
        cs = np.array(cs)
        rollout_corr.append(float(np.nanmean(cs)))
        rollout_corr_lo.append(float(np.nanpercentile(cs, 10)))
        rollout_corr_hi.append(float(np.nanpercentile(cs, 90)))
    single_std = float(R.std(axis=1).mean())
    report["1b_value_estimates"] = {
        "positions": len(positions), "rollouts_per_position": k_max,
        "ground_truth": f"mean result of {gt_games} Minimax-d2 self-play games per position",
        "single_rollout_std": single_std,
        "share_rollouts_drawn_by_cap": float((R == 0).mean()),
        "single_rollout_std_when_decided": float(np.std(R[R != 0])) if (R != 0).any() else float("nan"),
        "corr_vs_truth": {f"rollouts_k{k}": c for k, c in zip(ks, rollout_corr)},
        "corr_material_vs_truth": corr(material, truth),
    }
    series = {"Random rollouts (mean of k)": rollout_corr}
    refs = {"Material count": corr(material, truth)}
    for name, path in (("untrained", None), ("trained", ckpt)):
        if name == "trained" and not path:
            continue
        model = _load_or_untrained(path, seed)
        v = np.array([_net_eval(model, b, p)[1] for b, p in positions])
        report["1b_value_estimates"][f"corr_value_net_{name}_vs_truth"] = corr(v, truth)
        refs[f"Value net ({name})"] = corr(v, truth)
    print(json.dumps(report["1b_value_estimates"], indent=2))
    report["1b_value_estimates"]["plot"] = {"ks": ks, "rollout_corr": rollout_corr,
                                            "band": [rollout_corr_lo, rollout_corr_hi], "refs": refs}

    # 1c. Rollout-MCTS stability and cost.
    budgets = [10, 25, 50] if quick else [10, 25, 50, 100, 200]
    n_stab_pos = 6 if quick else 15
    seeds = 4 if quick else 8
    stab_positions = positions[:n_stab_pos]
    stability, sec_per_sim = [], []
    for sims in budgets:
        jobs = [(b.tolist(), p, sims, seed * 1009 + s) for b, p in stab_positions for s in range(seeds)]
        res = _map(_stability_worker, jobs, workers)
        agree = []
        for i in range(len(stab_positions)):
            chunk = [tuple(map(tuple, m)) for m, _ in res[i * seeds:(i + 1) * seeds]]
            agree.append(Counter(chunk).most_common(1)[0][1] / seeds)
        stability.append(float(np.mean(agree)))
        sec_per_sim.append(float(np.mean([t for _, t in res]) / sims))
        print(f"1c: sims={sims:>4}  same move across seeds={stability[-1]:.0%}  {sec_per_sim[-1]*1000:.1f} ms/sim")
    net_ms = _time_net_eval(seed)
    # Reference: the same statistic for a player that picks uniformly at random.
    rng_ref = random.Random(seed)
    ref = []
    for _ in range(200):
        for b, p in stab_positions:
            legal = get_all_legal_moves(b.copy(), p)
            picks = [rng_ref.randrange(len(legal)) for _ in range(seeds)]
            ref.append(Counter(picks).most_common(1)[0][1] / seeds)
    report["1c_rollout_mcts"] = {
        "budgets": budgets, "positions": n_stab_pos, "seeds": seeds,
        "agreement_with_modal_move": stability, "seconds_per_sim": sec_per_sim,
        "random_player_agreement": float(np.mean(ref)),
        "network_eval_ms": net_ms,
        "note": "agreement = share of seeds that pick the most common move; 1.0 = fully stable",
    }
    print(f"1c: a uniformly random player would agree {np.mean(ref):.0%} of the time")

    # 1d. Head to head at equal simulations.
    if ckpt:
        sims = 25 if quick else 50
        games = 4 if quick else 12
        jobs = [(ckpt, sims, g, seed) for g in range(games)]
        results = _map(_h2h_worker, jobs, workers)
        az_score = float(np.mean(results))
        report["1d_head_to_head"] = {"sims": sims, "games": games, "network_mcts_score": az_score}
        print(f"1d: network-MCTS scored {az_score:.0%} vs rollout-MCTS over {games} games at {sims} sims")
    return report


def _h2h_worker(args):
    ckpt, sims, g, seed = args
    import torch

    torch.set_num_threads(1)
    from training.mcts_agent import AlphaZeroPlayer

    az = AlphaZeroPlayer(load_model(ckpt), num_sims=sims, temperature=0.05)
    ro = RolloutMCTSPlayer(sims=sims, seed=seed + g)
    if g % 2 == 0:
        r = play_game(az, ro, "az", "ro", seed=seed + g)
        return r.score_for("az")
    r = play_game(ro, az, "ro", "az", seed=seed + g)
    return r.score_for("az")


def _load_or_untrained(path: Optional[str], seed: int):
    import torch
    from neural.model import ChaturangaNet

    if path:
        return load_model(path)
    torch.manual_seed(seed)
    m = ChaturangaNet()
    m.eval()
    return m


def _time_net_eval(seed: int) -> float:
    import torch

    torch.set_num_threads(1)
    m = _load_or_untrained(None, seed)
    b = create_starting_board()
    for _ in range(5):
        _net_eval(m, b, WHITE)
    t0 = time.perf_counter()
    for _ in range(50):
        _net_eval(m, b, WHITE)
    return (time.perf_counter() - t0) / 50 * 1000


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------
def _teacher_label_worker(args):
    board_list, player, seed = args
    seed_everything(seed)
    board = np.array(board_list, dtype=np.int8)
    m = MinimaxPlayer(2).get_move(board, player)
    return [list(m[0]), list(m[1])]


def experiment_2(out: str, workers: int, seed: int, ckpt: Optional[str], quick: bool) -> dict:
    import torch
    import torch.nn.functional as F
    from neural.encoder import board_to_tensor, get_action_mask, move_to_action
    from neural.model import ChaturangaNet

    print("\n=== Experiment 2: unmasked vs masked policy ===")
    torch.set_num_threads(max(1, workers))
    report: Dict[str, object] = {}

    n_pos = 600 if quick else 4000
    positions = sample_positions(n_pos, seed + 2, workers)
    X = torch.stack([board_to_tensor(b, p) for b, p in positions])
    M = torch.from_numpy(np.stack([get_action_mask(b, p) for b, p in positions]))
    legal_counts = M.sum(dim=1).float()

    # 2a. Illegal mass of a raw softmax.
    def illegal_mass(model) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            lp, _ = model(X)  # no mask
        return 1.0 - (lp.exp() * M).sum(dim=1).numpy()

    masses = {"Untrained net": illegal_mass(_load_or_untrained(None, seed))}
    if ckpt:
        masses["Trained net"] = illegal_mass(load_model(ckpt))
    report["2a_illegal_mass"] = {
        "positions": len(positions),
        "mean_legal_moves": float(legal_counts.mean()),
        "legal_share_of_action_space": float(legal_counts.mean() / 4096),
        **{f"{k.lower().replace(' ', '_')}_mean_illegal_mass": float(v.mean()) for k, v in masses.items()},
    }
    print(json.dumps(report["2a_illegal_mass"], indent=2))
    report["2a_illegal_mass"]["by_network"] = {k: float(v.mean()) for k, v in masses.items()}

    # 2b. Controlled imitation run.
    labels = _map(_teacher_label_worker,
                  [(b.tolist(), p, seed + i) for i, (b, p) in enumerate(positions)], workers)
    y = torch.tensor([move_to_action(((a[0], a[1]), (c[0], c[1]))) for a, c in labels])
    n_test = len(positions) // 5
    Xtr, Mtr, ytr = X[n_test:], M[n_test:], y[n_test:]
    Xte, Mte, yte = X[:n_test], M[:n_test], y[:n_test]
    steps = 150 if quick else 800
    batch = 64
    eval_every = max(10, steps // 30)

    def run(masked: bool):
        torch.manual_seed(seed)
        net = ChaturangaNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        hist = {"step": [], "test_acc": [], "test_ce_legal": [], "illegal_mass": [], "legal_grad_share": []}
        for step in range(1, steps + 1):
            net.train()
            idx = torch.randint(0, len(Xtr), (batch,), generator=g)
            lp, _ = net(Xtr[idx], Mtr[idx] if masked else None)
            loss = F.nll_loss(lp, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                net.eval()
                with torch.no_grad():
                    lp_raw, _ = net(Xte)
                    lp_m = lp_raw.masked_fill(~Mte, -1e9).log_softmax(dim=1)
                    acc = (lp_m.argmax(dim=1) == yte).float().mean().item()
                    ce = F.nll_loss(lp_m, yte).item()
                    ill = 1.0 - (lp_raw.exp() * Mte).sum(dim=1).mean().item()
                hist["step"].append(step)
                hist["test_acc"].append(acc)
                hist["test_ce_legal"].append(ce)
                hist["illegal_mass"].append(ill)
        # Share of the policy-logit gradient that lands on legal moves, final batch.
        net.train()
        idx = torch.arange(min(256, len(Xtr)))
        x = Xtr[idx]
        h = net.tower(net.stem(x))
        z = net.p_fc(torch.relu(net.p_bn(net.p_conv(h))).flatten(1))
        z.retain_grad()
        zz = z.masked_fill(~Mtr[idx], -1e9) if masked else z
        F.nll_loss(zz.log_softmax(dim=1), ytr[idx]).backward()
        gabs = z.grad.abs()
        hist["legal_grad_share"] = float((gabs * Mtr[idx]).sum() / gabs.sum().clamp_min(1e-12))
        return hist

    print(f"2b: training two networks for {steps} steps on {len(Xtr)} Minimax-labelled positions")
    t0 = time.time()
    masked_hist = run(True)
    unmasked_hist = run(False)
    print(f"2b: done in {time.time() - t0:.0f}s")

    def steps_to(hist, frac):
        target = frac * max(masked_hist["test_acc"][-1], unmasked_hist["test_acc"][-1])
        for s, a in zip(hist["step"], hist["test_acc"]):
            if a >= target:
                return s
        return None

    report["2b_training"] = {
        "train_positions": len(Xtr), "test_positions": len(Xte), "steps": steps, "batch": batch,
        "teacher": "Minimax depth 2 move (one-hot)",
        "final_test_accuracy": {"masked": masked_hist["test_acc"][-1], "unmasked": unmasked_hist["test_acc"][-1]},
        "final_test_ce_over_legal": {"masked": masked_hist["test_ce_legal"][-1],
                                     "unmasked": unmasked_hist["test_ce_legal"][-1]},
        "steps_to_90pct_of_best_accuracy": {"masked": steps_to(masked_hist, 0.9),
                                            "unmasked": steps_to(unmasked_hist, 0.9)},
        "final_illegal_mass_unmasked": unmasked_hist["illegal_mass"][-1],
        "gradient_share_on_legal_logits": {"masked": masked_hist["legal_grad_share"],
                                           "unmasked": unmasked_hist["legal_grad_share"]},
        "chance_accuracy": float((1.0 / legal_counts[:n_test]).mean()),
        "curves": {"masked": masked_hist, "unmasked": unmasked_hist},
    }
    summary = {k: v for k, v in report["2b_training"].items() if k != "curves"}
    print(json.dumps(summary, indent=2))
    return report


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------
def elephant_punishments(board: np.ndarray, player: int, move: Move) -> float:
    """
    Net material change for ``player`` if ``move`` is answered by the
    opponent's best Elephant capture and ``player`` then makes its best
    recapture. Returns +inf if no Elephant capture is available afterwards.
    """
    before = material_balance(board, player)
    b1 = make_move(board, move[0], move[1])
    opp = -player
    if get_game_outcome(b1, opp) is not None:
        return math.inf
    eleph_caps = [m for m in get_all_legal_moves(b1, opp)
                  if abs(int(b1[m[0]])) == ELEPHANT and b1[m[1]] * player > 0]
    if not eleph_caps:
        return math.inf
    worst = math.inf
    for e in eleph_caps:
        b2 = make_move(b1, e[0], e[1])
        legal = get_all_legal_moves(b2, player)
        if get_game_outcome(b2, player, legal) is not None:
            continue
        best_reply = max(material_balance(make_move(b2, r[0], r[1]), player) for r in legal)
        worst = min(worst, best_reply - before)
    return worst


def is_elephant_trap_move(board: np.ndarray, player: int, move: Move, threshold: float = -0.99) -> bool:
    """A capture that the opponent's Elephant leap punishes with a net material loss."""
    return board[move[1]] * player < 0 and elephant_punishments(board, player, move) <= threshold


def _trap_scan_worker(args) -> List[list]:
    """Find trap positions in games between assorted players."""
    seed, n_games = args
    rng = random.Random(seed)
    seed_everything(seed)
    found = []
    players = [GreedyPlayer(), MinimaxPlayer(2), RandomPlayer()]
    for g in range(n_games):
        white, black = rng.choice(players), rng.choice(players)
        board, player = create_starting_board(), WHITE
        for ply in range(100):
            legal = get_all_legal_moves(board, player)
            if get_game_outcome(board, player, legal) is not None:
                break
            caps = [m for m in legal if board[m[1]] * player < 0]
            traps = [m for m in caps if is_elephant_trap_move(board, player, m)]
            if traps:
                # Require a safe alternative: some move that is not punished by an Elephant.
                safe = [m for m in legal if m not in traps
                        and elephant_punishments(board, player, m) > -0.99]
                if safe:
                    found.append([board.tolist(), player, [[list(a), list(b)] for a, b in traps]])
            agent = white if player == WHITE else black
            m = agent.get_move(board.copy(), player)
            board, player = make_move(board, m[0], m[1]), -player
    return found


def _trap_eval_worker(args):
    spec, items, seed = args
    import torch

    torch.set_num_threads(1)
    seed_everything(seed)
    _, agent = build_player(spec)
    fell = 0
    for board_list, player, traps in items:
        board = np.array(board_list, dtype=np.int8)
        reset_player(agent)  # each position is a fresh, unrelated game
        m = agent.get_move(board.copy(), player)
        m = [list(m[0]), list(m[1])]
        fell += int(m in traps)
    return fell


def _blunder_rate_worker(args):
    spec_a, spec_b, g, seed = args
    import torch

    torch.set_num_threads(1)
    la, a = build_player(spec_a)
    lb, b = build_player(spec_b)
    white, black, wl, bl = (a, b, la, lb) if g % 2 == 0 else (b, a, lb, la)
    r = play_game(white, black, wl, bl, seed=seed + g)
    board, player = create_starting_board(), WHITE
    moves = {la: 0, lb: 0}
    blunders = {la: 0, lb: 0}
    for m in r.moves:
        who = wl if player == WHITE else bl
        moves[who] += 1
        if is_elephant_trap_move(board, player, m):
            blunders[who] += 1
        board, player = make_move(board, m[0], m[1]), -player
    return {"moves": moves, "blunders": blunders, "score_a": r.score_for(la), "a": la, "b": lb}


def experiment_3(out: str, workers: int, seed: int, ckpt: Optional[str], quick: bool) -> dict:
    print("\n=== Experiment 3: greedy 1-ply vs lookahead ===")
    report: Dict[str, object] = {}

    # 3a. Trap suite.
    scan_jobs = 8 if quick else 40
    games_per_job = 2 if quick else 3
    found = [x for chunk in _map(_trap_scan_worker,
                                 [(seed * 97 + j, games_per_job) for j in range(scan_jobs)], workers)
             for x in chunk]
    # De-duplicate positions.
    uniq = {}
    for b, p, t in found:
        uniq.setdefault((bytes(np.array(b, dtype=np.int8)), p), [b, p, t])
    suite = list(uniq.values())
    random.Random(seed).shuffle(suite)
    suite = suite[: (40 if quick else 200)]
    print(f"3a: {len(suite)} elephant-trap positions (a capture loses material to an Elephant leap, "
          f"and a safe move exists)")
    specs = ["random", "greedy", "minimax:depth=2", "minimax:depth=3"]
    if ckpt:
        specs += [f"AZ-raw=az:ckpt={ckpt},sims=0,temp=0.01", f"AZ-80=az-master:ckpt={ckpt}"]
    rates = {}
    if suite:
        for spec in specs:
            label = build_player(spec)[0]
            per = max(1, len(suite) // max(1, workers))
            chunks = [suite[i:i + per] for i in range(0, len(suite), per)]
            fell = sum(_map(_trap_eval_worker, [(spec, c, seed + i) for i, c in enumerate(chunks)], workers))
            rates[label] = fell / len(suite)
            print(f"3a: {label:<14} took the poisoned capture in {rates[label]:.0%} of positions")
    avg_traps = float(np.mean([len(t) for _, _, t in suite])) if suite else 0.0
    report["3a_trap_suite"] = {"positions": len(suite), "avg_poisoned_captures_per_position": avg_traps,
                               "fall_rate": rates}
    with open(os.path.join(out, "exp3_trap_suite.json"), "w") as f:
        json.dump(suite, f)

    # 3b. In-game blunder rate and head-to-head.
    games = 6 if quick else 30
    pairs = [("greedy", "random"), ("minimax:depth=2", "random"), ("greedy", "minimax:depth=2")]
    rows = []
    for a, b in pairs:
        res = _map(_blunder_rate_worker, [(a, b, g, seed * 13) for g in range(games)], workers)
        la, lb = res[0]["a"], res[0]["b"]
        mv = {la: sum(r["moves"][la] for r in res), lb: sum(r["moves"][lb] for r in res)}
        bl = {la: sum(r["blunders"][la] for r in res), lb: sum(r["blunders"][lb] for r in res)}
        rows.append({"player": la, "opponent": lb, "games": games,
                     "score": float(np.mean([r["score_a"] for r in res])),
                     "elephant_blunders_per_100_moves": 100 * bl[la] / max(1, mv[la]),
                     "opponent_elephant_blunders_per_100_moves": 100 * bl[lb] / max(1, mv[lb])})
        print(f"3b: {la} vs {lb}: score {rows[-1]['score']:.0%}, blunders/100 moves "
              f"{rows[-1]['elephant_blunders_per_100_moves']:.2f} vs {rows[-1]['opponent_elephant_blunders_per_100_moves']:.2f}")
    report["3b_games"] = rows
    return report


# ---------------------------------------------------------------------------
# Figures (drawn from the saved results, so --replot never reruns anything)
# ---------------------------------------------------------------------------
def make_figures(report: dict, out: str) -> List[str]:
    made = []
    j = os.path.join
    e1 = report.get("exp1") or {}
    a = e1.get("1a_playouts")
    if a:
        if a.get("lengths"):
            made.append(plots.plot_histograms(
                {"Random playout length": a["lengths"]}, j(out, "exp1a_playout_lengths.png"),
                "Length of uniformly random games from the start position", "Plies", bins=40))
        order = [CHECKMATE, STALEMATE, BARE_KING]
        made.append(plots.plot_bars(
            [r.replace("_", " ") for r in order], [a["end_reasons"].get(r, 0.0) for r in order],
            j(out, "exp1a_playout_endings.png"), "How random playouts end", "Share of playouts",
            ylim=(0, 1.05)))
    b = (e1.get("1b_value_estimates") or {}).get("plot")
    if b:
        made.append(plots.plot_lines(
            b["ks"], {"Random rollouts (mean of k)": b["rollout_corr"]},
            j(out, "exp1b_value_correlation.png"),
            "How well each evaluator predicts the Minimax self-play result",
            "Rollouts per position (k)", "Pearson correlation with result", logx=True,
            bands={"Random rollouts (mean of k)": tuple(b["band"])}, refs=b["refs"]))
    c = e1.get("1c_rollout_mcts")
    if c:
        refs = {"Uniformly random player": c["random_player_agreement"]} if "random_player_agreement" in c else None
        made.append(plots.plot_lines(
            c["budgets"], {"Rollout-MCTS": c["agreement_with_modal_move"]},
            j(out, "exp1c_rollout_stability.png"), "Does rollout-MCTS settle on one move?",
            "Simulations per move", "Share of seeds choosing the modal move",
            fmt="{:.0%}", logx=True, refs=refs, ylim=(0, 1.05)))
    e2 = report.get("exp2") or {}
    a = e2.get("2a_illegal_mass")
    if a and a.get("by_network"):
        made.append(plots.plot_bars(
            list(a["by_network"]), list(a["by_network"].values()), j(out, "exp2a_illegal_mass.png"),
            "Probability on illegal moves without a mask", "Mean illegal probability",
            ylim=(0, 1.05), fmt="{:.1%}"))
    t = e2.get("2b_training")
    if t:
        m, u = t["curves"]["masked"], t["curves"]["unmasked"]
        made.append(plots.plot_lines(
            m["step"], {"Masked": m["test_acc"], "Unmasked": u["test_acc"]},
            j(out, "exp2b_accuracy.png"),
            "Imitating a Minimax teacher: held-out top-1 accuracy (legal moves only)",
            "Training step", "Accuracy", fmt="{:.0%}",
            refs={"Chance (random legal move)": t["chance_accuracy"]}))
        made.append(plots.plot_lines(
            u["step"], {"Unmasked network": u["illegal_mass"]}, j(out, "exp2b_illegal_mass.png"),
            "Unmasked network: probability still on illegal moves (held-out)",
            "Training step", "Illegal probability", fmt="{:.0%}", ylim=(0, 1.05)))
    e3 = report.get("exp3") or {}
    a = e3.get("3a_trap_suite")
    if a and a.get("fall_rate"):
        made.append(plots.plot_bars(
            list(a["fall_rate"]), list(a["fall_rate"].values()), j(out, "exp3a_trap_rate.png"),
            f"Elephant-trap suite ({a['positions']} positions): poisoned capture taken",
            "Share of positions", ylim=(0, 1.05)))
    rows = e3.get("3b_games")
    if rows:
        by_player: Dict[str, List[float]] = {}
        for r in rows:
            by_player.setdefault(r["player"], []).append(r["elephant_blunders_per_100_moves"])
            by_player.setdefault(r["opponent"], []).append(r["opponent_elephant_blunders_per_100_moves"])
        names = list(by_player)
        made.append(plots.plot_bars(
            names, [float(np.mean(by_player[n])) for n in names], j(out, "exp3b_blunders_per_100.png"),
            "Elephant-leap blunders in real games", "Per 100 moves", fmt="{:.2f}"))
    return made


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.1%}" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_markdown(report: dict, out: str) -> str:
    L = ["# Ablation experiments: what didn't work", ""]
    cfg = report.get("config", {})
    L.append(f"Seed {cfg.get('seed')}, {'quick' if cfg.get('quick') else 'full'} run, "
             f"checkpoint: `{cfg.get('checkpoint') or 'none (untrained network only)'}`.")
    e1 = report.get("exp1")
    if e1:
        a, b, c = e1["1a_playouts"], e1["1b_value_estimates"], e1["1c_rollout_mcts"]
        L += ["", "## Experiment 1: random rollouts vs value network", "",
              f"- {a['n']} uniformly random games from the start: median {a['median_plies']:.0f} plies, "
              f"90th percentile {a['p90_plies']:.0f}; {_fmt(a['share_over_100_plies'], True)} run past 100 plies.",
              "- Endings: " + ", ".join(f"{k.replace('_', ' ')} {_fmt(v, True)}" for k, v in sorted(a["end_reasons"].items())) + ".",
              f"- With the same 120-ply cap as real games, {_fmt(b['share_rollouts_drawn_by_cap'], True)} of rollouts "
              f"never finish and score 0, so most rollouts carry no signal; the ones that do finish are ±1 coin flips "
              f"(standard deviation {_fmt(b['single_rollout_std_when_decided'])}).",
              "- Correlation with the result Minimax-d2 actually reaches from the same position: "
              + ", ".join(f"{k.replace('corr_', '').replace('_vs_truth', '').replace('_', ' ')} {_fmt(v)}"
                          for k, v in b.items() if k.startswith("corr_") and not isinstance(v, dict))
              + "; rollouts: " + ", ".join(f"k={k.split('k')[-1]} {_fmt(v)}" for k, v in b["corr_vs_truth"].items()) + ".",
              f"- Rollout-MCTS picks the same move across seeds in "
              + ", ".join(f"{_fmt(s, True)} (at {n} sims)" for n, s in zip(c["budgets"], c["agreement_with_modal_move"]))
              + f"; it costs {c['seconds_per_sim'][-1]*1000:.1f} ms per simulation vs {c['network_eval_ms']:.1f} ms for one network evaluation."]
        if "1d_head_to_head" in e1:
            d = e1["1d_head_to_head"]
            L.append(f"- Head to head at {d['sims']} sims: network-MCTS scored {_fmt(d['network_mcts_score'], True)} "
                     f"against rollout-MCTS over {d['games']} games.")
    e2 = report.get("exp2")
    if e2:
        a, b = e2["2a_illegal_mass"], e2["2b_training"]
        L += ["", "## Experiment 2: unmasked vs masked policy", "",
              f"- On average {a['mean_legal_moves']:.1f} of the 4096 actions are legal ({_fmt(a['legal_share_of_action_space'], True)})."]
        for k, v in a.items():
            if k.endswith("mean_illegal_mass"):
                L.append(f"- {k.replace('_mean_illegal_mass', '').replace('_', ' ').capitalize()}: "
                         f"{_fmt(v, True)} of an unmasked softmax lands on illegal moves.")
        L += [f"- Imitation run ({b['steps']} steps, {b['train_positions']} positions): held-out accuracy "
              f"masked {_fmt(b['final_test_accuracy']['masked'], True)} vs unmasked {_fmt(b['final_test_accuracy']['unmasked'], True)} "
              f"(chance {_fmt(b['chance_accuracy'], True)}); steps to 90% of the best accuracy: "
              f"masked {b['steps_to_90pct_of_best_accuracy']['masked']}, unmasked {b['steps_to_90pct_of_best_accuracy']['unmasked']}.",
              f"- Unmasked network still puts {_fmt(b['final_illegal_mass_unmasked'], True)} on illegal moves after training; "
              f"share of the policy-logit gradient on legal moves: masked {_fmt(b['gradient_share_on_legal_logits']['masked'], True)}, "
              f"unmasked {_fmt(b['gradient_share_on_legal_logits']['unmasked'], True)}."]
    e3 = report.get("exp3")
    if e3:
        a = e3["3a_trap_suite"]
        L += ["", "## Experiment 3: greedy 1-ply vs lookahead", "",
              f"- Trap suite of {a['positions']} positions (a capture is punished by an Elephant leap with net "
              f"material loss; a safe move exists). Poisoned capture taken: "
              + ", ".join(f"{k} {_fmt(v, True)}" for k, v in a["fall_rate"].items()) + "."]
        for r in e3["3b_games"]:
            L.append(f"- {r['player']} vs {r['opponent']} ({r['games']} games): score {_fmt(r['score'], True)}; "
                     f"Elephant blunders per 100 moves {r['elephant_blunders_per_100_moves']:.2f} vs "
                     f"{r['opponent_elephant_blunders_per_100_moves']:.2f}.")
    L += ["", "Figures: `exp*.png` in this folder. Raw numbers: `ablation_results.json`.", ""]
    text = "\n".join(L)
    with open(os.path.join(out, "ablation_report.md"), "w") as f:
        f.write(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--checkpoint", help="trained model; experiments that need it are skipped without one")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="small sizes for a smoke run")
    p.add_argument("--out", default=os.path.join("arena_results", "ablations"))
    p.add_argument("--replot", action="store_true",
                   help="only redraw figures and the report from the saved results")
    args = p.parse_args(argv)

    if args.replot:
        with open(os.path.join(args.out, "ablation_results.json")) as f:
            report = json.load(f)
        for m in make_figures(report, args.out):
            print("wrote", m)
        print(write_markdown(report, args.out))
        return

    ckpt = None
    try:
        ckpt = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError:
        if args.checkpoint:
            raise
        print("No trained checkpoint found: running the parts that do not need one.")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ablation_results.json")
    report: Dict[str, object] = {}
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
    report["config"] = {"seed": args.seed, "quick": args.quick, "checkpoint": ckpt}
    runners = {1: experiment_1, 2: experiment_2, 3: experiment_3}
    for e in args.exp:
        t0 = time.time()
        report[f"exp{e}"] = runners[e](args.out, args.workers, args.seed, ckpt, args.quick)
        report[f"exp{e}"]["seconds"] = time.time() - t0
        _write(args.out, "ablation_results.json", report)
        make_figures(report, args.out)
    print("\n" + write_markdown(report, args.out))
    print(f"Written to {args.out}/")


if __name__ == "__main__":
    main()
