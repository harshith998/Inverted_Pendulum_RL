"""Black-box optimize polynomial force schedule parameters from env returns.

No LQR actions, gains, Riccati solves, or LQR costs are used. The optimized
parameters live inside a CGAT policy checkpoint and are evaluated through
eval_cgat.py.
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

from eval.optimize_mass_force_schedule import eval_candidate
from models.cgat import load_cgat_variant


def set_poly(policy, params: np.ndarray) -> None:
    with torch.no_grad():
        policy.poly_scale_coeffs.copy_(torch.tensor(params[:6], dtype=torch.float32))
        policy.poly_log_clip.fill_(float(params[6]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--elite", type=int, default=5)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default="checkpoints/cgat_poly_force_schedule_cem_seed395_best.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = load_cgat_variant(
        "poly_force_schedule",
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

    # Approximate the strongest mass-force schedule as a polynomial starting point.
    mean = np.array([
        0.34,   # bias
        0.00,   # length
        0.22,   # mass
        0.00,   # length*mass
        0.00,   # length^2
        0.02,   # mass^2
        0.70,   # clip
    ], dtype=np.float64)
    std = np.array([0.18, 0.18, 0.20, 0.18, 0.12, 0.12, 0.12], dtype=np.float64)
    bounds = np.array([
        [-0.20, 0.70],
        [-0.55, 0.55],
        [-0.40, 0.75],
        [-0.55, 0.55],
        [-0.35, 0.35],
        [-0.35, 0.35],
        [0.30, 1.05],
    ], dtype=np.float64)

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            sample = rng.normal(mean, std)
            samples.append(np.clip(sample, bounds[:, 0], bounds[:, 1]))
        scored = []
        for idx, sample in enumerate(samples):
            set_poly(policy, sample)
            score = eval_candidate(
                policy, cfg, lengths, masses, args.episodes,
                args.seed + it * 10000 + idx * 100,
                device,
            )
            scored.append((score, sample))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([x[1] for x in scored[:args.elite]], axis=0)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.array([0.025, 0.025, 0.025, 0.025, 0.02, 0.02, 0.025]))
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_params = scored[0][1].copy()
            set_poly(policy, best_params)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(policy.state_dict(), args.out)
        print(
            f"iter {it+1:02d} best={scored[0][0]:.2f} global={best_score:.2f} "
            f"params={np.round(best_params, 4)} std={np.round(std, 4)}",
            flush=True,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
