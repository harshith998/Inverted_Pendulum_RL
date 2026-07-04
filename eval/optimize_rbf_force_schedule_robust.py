"""Risk-sensitive RBF force-schedule search using env returns only.

The objective emphasizes repeatability and low-tail cells. It evaluates each
candidate across multiple reset seeds and scores mean, p25, and high-survival
frequency. No LQR actions, gains, solves, or returns are used as targets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.optimize_mass_force_schedule import make_env
from eval.optimize_rbf_force_schedule import get_params, set_params
from models.cgat import load_cgat_variant


@torch.no_grad()
def eval_candidate_values(policy, cfg, lengths, masses, episodes, seeds, device) -> np.ndarray:
    reward_config = {
        "upright_weight": 1.0,
        "alive_bonus": 0.1,
        "force_penalty": 0.001,
        "rail_penalty": 0.01,
    }
    values = []
    for seed_base in seeds:
        for i, length in enumerate(lengths):
            for j, mass in enumerate(masses):
                cell_rewards = []
                for ep in range(episodes):
                    env = make_env(cfg, float(length), float(mass), reward_config)
                    obs, _ = env.reset(seed=int(seed_base + 1009 * i + 9173 * j + ep))
                    total = 0.0
                    done = False
                    while not done:
                        action = policy.get_deterministic_action(obs, device)
                        obs, reward, terminated, truncated, _ = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        total += float(reward)
                        done = terminated or truncated
                    env.close()
                    cell_rewards.append(total)
                values.append(float(np.mean(cell_rewards)))
    return np.asarray(values, dtype=np.float64)


def robust_score(values: np.ndarray) -> float:
    mean = float(values.mean())
    p25 = float(np.percentile(values, 25))
    p10 = float(np.percentile(values, 10))
    high_frac = float((values >= 2000.0).mean())
    collapse_frac = float((values < 600.0).mean())
    return mean + 0.35 * p25 + 0.20 * p10 + 220.0 * high_frac - 180.0 * collapse_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--score-seeds", type=int, nargs="+", default=[1300, 2300])
    parser.add_argument("--out", default="checkpoints/cgat_rbf_force_schedule_robust_seed403_best.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = load_cgat_variant(
        "rbf_force_schedule",
        hidden=128,
        n_icga_layers=2,
        n_heads=2,
        max_links=3,
        max_force=float(cfg["environment"]["max_force"]),
    ).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    lengths = np.linspace(max(0.05, len_lo - (len_hi - len_lo)), len_hi + (len_hi - len_lo), args.grid)
    masses = np.linspace(max(0.05, mass_lo - (mass_hi - mass_lo)), mass_hi + (mass_hi - mass_lo), args.grid)

    mean = get_params(policy).astype(np.float64)
    std = np.concatenate([
        np.full(16, 0.08),
        np.array([0.04, 0.06, 0.08]),
    ])
    lo = np.concatenate([np.full(16, -0.20), np.array([0.35, 0.25, 0.50])])
    hi = np.concatenate([np.full(16, 0.85), np.array([1.00, 0.90, 1.60])])

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            samples.append(np.clip(rng.normal(mean, std), lo, hi))
        scored = []
        for idx, sample in enumerate(samples):
            set_params(policy, sample)
            vals = eval_candidate_values(
                policy, cfg, lengths, masses, args.episodes,
                [s + it * 10000 + idx * 100 for s in args.score_seeds],
                device,
            )
            score = robust_score(vals)
            scored.append((score, vals.mean(), np.percentile(vals, 25), (vals >= 2000).mean(), sample))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([x[4] for x in scored[:args.elite]], axis=0)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.concatenate([
            np.full(16, 0.025),
            np.array([0.025, 0.03, 0.04]),
        ]))
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_params = scored[0][4].copy()
            set_params(policy, best_params)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(policy.state_dict(), args.out)
        print(
            f"iter {it+1:02d} score={scored[0][0]:.2f} global={best_score:.2f} "
            f"mean={scored[0][1]:.2f} p25={scored[0][2]:.2f} hi={scored[0][3]:.2f} "
            f"clip_bw={np.round(best_params[16:], 4)}",
            flush=True,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
