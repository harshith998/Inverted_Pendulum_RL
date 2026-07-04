"""Optimize compact damping residual parameters from env returns only."""

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

from eval.optimize_rbf_force_schedule_robust import eval_candidate_values, robust_score
from models.cgat import load_cgat_variant


def set_damping(policy, params: np.ndarray) -> None:
    with torch.no_grad():
        policy.damping_coeffs.copy_(torch.tensor(params[:8], dtype=torch.float32))
        policy.damping_limit.fill_(float(params[8]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=409)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--pop", type=int, default=14)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--score-seeds", type=int, nargs="+", default=[1500, 2500])
    parser.add_argument("--out", default="checkpoints/cgat_rbf_damping_schedule_cem_seed409_best.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = load_cgat_variant(
        "rbf_damping_schedule",
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

    mean = np.zeros(9, dtype=np.float64)
    mean[8] = 0.75
    std = np.array([0.35, 0.35, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.25])
    lo = np.array([-1.2] * 8 + [0.10], dtype=np.float64)
    hi = np.array([1.2] * 8 + [2.00], dtype=np.float64)

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            samples.append(np.clip(rng.normal(mean, std), lo, hi))
        scored = []
        for idx, sample in enumerate(samples):
            set_damping(policy, sample)
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
        std = np.maximum(elites.std(axis=0), np.array([0.05] * 8 + [0.05]))
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_params = scored[0][4].copy()
            set_damping(policy, best_params)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(policy.state_dict(), args.out)
        print(
            f"iter {it+1:02d} score={scored[0][0]:.2f} global={best_score:.2f} "
            f"mean={scored[0][1]:.2f} p25={scored[0][2]:.2f} hi={scored[0][3]:.2f} "
            f"params={np.round(best_params, 4)}",
            flush=True,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
