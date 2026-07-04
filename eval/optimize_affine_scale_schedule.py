"""CEM search for a compact length/mass action-scale schedule.

The base action is from a fixed learned CGAT checkpoint. The searched schedule
is a five-parameter deterministic policy wrapper scored only by simulation
reward. It does not use LQR gains, labels, Riccati solves, or controller
recomputation.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import load_policy, make_fixed_env


def scale_action(action: float, length: float, mass: float, params: np.ndarray) -> float:
    length_n = (length - 1.075) / 1.025
    mass_n = (mass - 1.975) / 1.925
    log_scale = (
        params[0]
        + params[1] * length_n
        + params[2] * mass_n
        + params[3] * length_n * mass_n
    )
    limit = abs(float(params[4]))
    return float(action * np.exp(np.clip(log_scale, -limit, limit)))


def eval_params(policy, cfg: dict, device: torch.device, params: np.ndarray,
                n_grid: int, n_episodes: int, seed: int) -> tuple[float, float, float, int]:
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    lengths = np.linspace(max(0.05, len_lo - (len_hi - len_lo)), len_hi + (len_hi - len_lo), n_grid)
    masses = np.linspace(max(0.05, mass_lo - (mass_hi - mass_lo)), mass_hi + (mass_hi - mass_lo), n_grid)
    rewards = []
    for i, length in enumerate(lengths):
        for j, mass in enumerate(masses):
            env = make_fixed_env(cfg, float(length), float(mass), n_links=3)
            try:
                for ep in range(n_episodes):
                    obs, _ = env.reset(seed=int(seed + 1009 * i + 9173 * j + ep))
                    total = 0.0
                    done = False
                    while not done:
                        action = policy.get_deterministic_action(obs, device)
                        action = scale_action(action, float(length), float(mass), params)
                        obs, reward, terminated, truncated, _ = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        total += reward
                        done = terminated or truncated
                    rewards.append(total)
            finally:
                env.close()
    arr = np.asarray(rewards, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 25)),
        float(np.median(arr)),
        int((arr >= 2000.0).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="param_residual")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--elite_frac", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--n_grid", type=int, default=6)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=354)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, cfg, device, variant=args.variant)
    rng = np.random.default_rng(args.seed)
    mean = np.array([0.0, 0.0, 0.35, 0.0, 0.70], dtype=np.float64)
    std = np.full(5, float(args.sigma), dtype=np.float64)
    n_elite = max(1, int(round(args.pop * args.elite_frac)))
    best = None

    print(
        f"Device: {device} | iters={args.iters} pop={args.pop} "
        f"elite={n_elite} sigma={args.sigma}",
        flush=True,
    )
    for it in range(args.iters):
        samples = []
        if it == 0:
            samples.append(mean.copy())
        while len(samples) < args.pop:
            sample = rng.normal(mean, std).astype(np.float64)
            sample[4] = np.clip(abs(sample[4]), 0.05, 1.20)
            samples.append(sample)

        candidates = []
        for idx, params in enumerate(samples):
            mean_r, p25, median, high = eval_params(
                policy, cfg, device, params, args.n_grid, args.n_eval_episodes,
                args.seed + 100 * it,
            )
            metric = mean_r + 0.20 * p25 + 30.0 * high
            record = (metric, mean_r, p25, median, high, params.copy())
            candidates.append(record)
            if best is None or metric > best[0]:
                best = record
                np.savetxt(args.output, params.reshape(1, -1), fmt="%.10f")
            print(
                f"iter={it:02d} cand={idx:02d} mean={mean_r:.2f} "
                f"p25={p25:.2f} med={median:.2f} high={high} "
                f"params={np.round(params, 3).tolist()} metric={metric:.2f}",
                flush=True,
            )

        candidates.sort(reverse=True, key=lambda item: item[0])
        elite = np.stack([item[-1] for item in candidates[:n_elite]], axis=0)
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.03)
        mean[4] = np.clip(abs(mean[4]), 0.05, 1.20)
        top = candidates[0]
        print(
            f"ITER {it:02d} TOP mean={top[1]:.2f} p25={top[2]:.2f} "
            f"med={top[3]:.2f} high={top[4]} metric={top[0]:.2f}",
            flush=True,
        )

    metric, mean_r, p25, median, high, params = best
    print(
        f"BEST mean={mean_r:.2f} p25={p25:.2f} med={median:.2f} "
        f"high={high} metric={metric:.2f} params={params.tolist()}",
        flush=True,
    )
    print(f"Saved params -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
