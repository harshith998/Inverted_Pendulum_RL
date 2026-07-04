"""Black-box optimize the in-policy mass-force schedule parameters.

This is policy search over four neural policy parameters, using environment
returns only. It does not use LQR actions, LQR gains, Riccati solves, or LQR
costs as targets.
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

from env.pendulum_env import VariablePendulumEnv
from models.cgat import load_cgat_variant


def obs_to_tensor(obs: dict, device: torch.device) -> dict:
    return {
        k: torch.tensor(v, dtype=torch.float32 if v.dtype != np.int64 else torch.int64)
        .unsqueeze(0)
        .to(device)
        for k, v in obs.items()
    }


def set_schedule(policy, params: np.ndarray) -> None:
    with torch.no_grad():
        policy.log_low_scale.fill_(float(params[0]))
        policy.log_high_scale.fill_(float(params[1]))
        policy.mass_switch.fill_(float(params[2]))
        policy.mass_sharpness.fill_(float(params[3]))


def make_env(cfg: dict, length: float, mass: float, reward_config: dict | None = None):
    env_cfg = cfg["environment"]
    return VariablePendulumEnv(
        n_links_range=(3, 3),
        cart_mass_range=tuple(env_cfg["cart_mass_range"]),
        link_length_range=(float(length), float(length)),
        link_mass_range=(float(mass), float(mass)),
        rail_limit=float(env_cfg["rail_limit"]),
        max_force=float(env_cfg["max_force"]),
        timestep=float(env_cfg["timestep"]),
        frame_skip=int(env_cfg["frame_skip"]),
        max_episode_steps=int(env_cfg["max_episode_steps"]),
        termination_angle=float(env_cfg["termination_angle"]),
        max_links=int(env_cfg["max_links"]),
        reward_config=reward_config,
    )


def eval_candidate(policy, cfg, lengths, masses, episodes, seed, device) -> float:
    rewards = []
    reward_config = {
        "upright_weight": 1.0,
        "alive_bonus": 0.1,
        "force_penalty": 0.001,
        "rail_penalty": 0.01,
    }
    with torch.no_grad():
        for i, length in enumerate(lengths):
            for j, mass in enumerate(masses):
                for ep in range(episodes):
                    env = make_env(cfg, float(length), float(mass), reward_config)
                    obs, _ = env.reset(seed=seed + 1009 * i + 9173 * j + ep)
                    done = False
                    total = 0.0
                    while not done:
                        action = policy.get_deterministic_action(obs, device)
                        obs, reward, terminated, truncated, _ = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        total += float(reward)
                        done = terminated or truncated
                    env.close()
                    rewards.append(total)
    vals = np.asarray(rewards, dtype=np.float64)
    return float(vals.mean() + 0.15 * np.percentile(vals, 25))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--pop", type=int, default=18)
    parser.add_argument("--elite", type=int, default=5)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default="checkpoints/cgat_mass_force_schedule_cem_seed391_best.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = load_cgat_variant(
        "mass_force_schedule",
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
    eval_lengths = np.linspace(max(0.05, len_lo - (len_hi - len_lo)), len_hi + (len_hi - len_lo), args.grid)
    eval_masses = np.linspace(max(0.05, mass_lo - (mass_hi - mass_lo)), mass_hi + (mass_hi - mass_lo), args.grid)

    mean = np.array([0.18232156, 0.64185387, 1.20, 8.0], dtype=np.float64)
    std = np.array([0.18, 0.16, 0.28, 3.0], dtype=np.float64)
    bounds = np.array([
        [-0.20, 0.50],
        [0.10, 0.95],
        [0.45, 2.35],
        [2.0, 18.0],
    ], dtype=np.float64)

    best_score = -np.inf
    best_params = mean.copy()
    for it in range(args.iters):
        samples = [mean]
        while len(samples) < args.pop:
            sample = rng.normal(mean, std)
            sample = np.clip(sample, bounds[:, 0], bounds[:, 1])
            samples.append(sample)

        scored = []
        for idx, sample in enumerate(samples):
            set_schedule(policy, sample)
            score = eval_candidate(policy, cfg, eval_lengths, eval_masses, args.episodes, args.seed + it * 10000 + idx * 100, device)
            scored.append((score, sample))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([x[1] for x in scored[:args.elite]], axis=0)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.array([0.025, 0.025, 0.04, 0.35]))
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_params = scored[0][1].copy()
            set_schedule(policy, best_params)
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
