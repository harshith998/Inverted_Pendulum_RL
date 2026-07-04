"""Optimize state-rescue scale parameters from environment returns only."""

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


def set_rescue(policy, params: np.ndarray) -> None:
    with torch.no_grad():
        policy.state_angle_gain.fill_(float(params[0]))
        policy.state_vel_gain.fill_(float(params[1]))
        policy.state_angle_threshold.fill_(float(params[2]))
        policy.state_log_clip.fill_(float(params[3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=406)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--score-seeds", type=int, nargs="+", default=[1400, 2400])
    parser.add_argument("--out", default="checkpoints/cgat_rbf_state_force_schedule_rescue_seed406_best.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = load_cgat_variant(
        "rbf_state_force_schedule",
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

    mean = np.array([0.0, 0.0, 0.18, 0.35], dtype=np.float64)
    std = np.array([0.9, 0.30, 0.08, 0.12], dtype=np.float64)
    lo = np.array([-2.0, -0.8, 0.04, 0.05], dtype=np.float64)
    hi = np.array([2.0, 0.8, 0.45, 0.75], dtype=np.float64)

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            samples.append(np.clip(rng.normal(mean, std), lo, hi))
        scored = []
        for idx, sample in enumerate(samples):
            set_rescue(policy, sample)
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
        std = np.maximum(elites.std(axis=0), np.array([0.08, 0.04, 0.015, 0.025]))
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_params = scored[0][4].copy()
            set_rescue(policy, best_params)
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
