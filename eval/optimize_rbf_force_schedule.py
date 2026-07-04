"""Black-box optimize RBF force-schedule parameters from env returns only."""

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


def get_params(policy) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([
            policy.log_scale_table.detach().cpu().numpy().reshape(-1),
            np.array([
                float(policy.log_scale_clip.detach().cpu()),
                float(policy.length_bandwidth.detach().cpu()),
                float(policy.mass_bandwidth.detach().cpu()),
            ]),
        ])


def set_params(policy, params: np.ndarray) -> None:
    with torch.no_grad():
        policy.log_scale_table.copy_(torch.tensor(params[:16], dtype=torch.float32).view(4, 4))
        policy.log_scale_clip.fill_(float(params[16]))
        policy.length_bandwidth.fill_(float(params[17]))
        policy.mass_bandwidth.fill_(float(params[18]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=397)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default="checkpoints/cgat_rbf_force_schedule_cem_seed397_best.pt")
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
        np.full(16, 0.16),
        np.array([0.08, 0.12, 0.18]),
    ])
    lo = np.concatenate([np.full(16, -0.25), np.array([0.35, 0.22, 0.45])])
    hi = np.concatenate([np.full(16, 0.95), np.array([1.05, 0.95, 1.80])])

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            samples.append(np.clip(rng.normal(mean, std), lo, hi))
        scored = []
        for idx, sample in enumerate(samples):
            set_params(policy, sample)
            score = eval_candidate(
                policy, cfg, lengths, masses, args.episodes,
                args.seed + it * 10000 + idx * 100,
                device,
            )
            scored.append((score, sample))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([x[1] for x in scored[:args.elite]], axis=0)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.concatenate([
            np.full(16, 0.035),
            np.array([0.03, 0.035, 0.05]),
        ]))
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_params = scored[0][1].copy()
            set_params(policy, best_params)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(policy.state_dict(), args.out)
        print(
            f"iter {it+1:02d} best={scored[0][0]:.2f} global={best_score:.2f} "
            f"clip_bw={np.round(best_params[16:], 4)} table={np.round(best_params[:16].reshape(4,4), 3)}",
            flush=True,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
