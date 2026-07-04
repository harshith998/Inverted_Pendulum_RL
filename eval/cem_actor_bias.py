"""CEM search over a tiny learned actor-bias subspace.

This optimizes fixed neural checkpoints directly in simulation. It never uses
LQR labels, gains, Riccati solves, or runtime controller recomputation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import eval_point, load_policy, make_fixed_env


BIAS_SPECS = (
    ("action_scale_net.3.bias", 1),
    ("gain_net.5.bias", 9),
    ("mean_head.bias", 1),
)


def vector_dim() -> int:
    return sum(dim for _, dim in BIAS_SPECS)


def apply_vector(base: dict[str, torch.Tensor], vector: np.ndarray) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in base.items()}
    offset = 0
    for key, dim in BIAS_SPECS:
        if key not in state:
            raise KeyError(key)
        chunk = torch.tensor(vector[offset:offset + dim], dtype=state[key].dtype)
        state[key] = state[key] + chunk.reshape_as(state[key])
        offset += dim
    return state


def screen(policy, cfg: dict, device: torch.device, n_grid: int,
           n_episodes: int, seed: int) -> tuple[float, float, float, float, int]:
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    lengths = np.linspace(max(0.05, len_lo - (len_hi - len_lo)),
                          len_hi + (len_hi - len_lo), n_grid)
    masses = np.linspace(max(0.05, mass_lo - (mass_hi - mass_lo)),
                         mass_hi + (mass_hi - mass_lo), n_grid)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rewards = []
    for length in lengths:
        for mass in masses:
            env = make_fixed_env(
                cfg, link_length=float(length), link_mass=float(mass), n_links=3)
            reward = eval_point(policy, env, n_episodes, device)
            env.close()
            rewards.append(reward)
    arr = np.asarray(rewards, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.median(arr)),
        int((arr > 2000.0).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="action_scale")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--elite_frac", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--n_grid", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=171)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base = torch.load(args.checkpoint, map_location="cpu")
    tmp_path = "checkpoints/_tmp_cem_actor_bias.pt"

    dim = vector_dim()
    mean = np.zeros(dim, dtype=np.float64)
    std = np.full(dim, args.sigma, dtype=np.float64)
    n_elite = max(1, int(round(args.pop * args.elite_frac)))
    best = None

    print(
        f"Device: {device} | dim={dim} iters={args.iters} pop={args.pop} "
        f"elite={n_elite} sigma={args.sigma}",
        flush=True,
    )

    for it in range(args.iters):
        candidates = []
        if it == 0:
            samples = [np.zeros(dim, dtype=np.float64)]
            samples += [
                rng.normal(mean, std).astype(np.float64)
                for _ in range(args.pop - 1)
            ]
        else:
            samples = [
                rng.normal(mean, std).astype(np.float64)
                for _ in range(args.pop)
            ]

        for idx, vector in enumerate(samples):
            state = apply_vector(base, vector)
            torch.save(state, tmp_path)
            policy = load_policy(tmp_path, cfg, device, variant=args.variant)
            mean_r, p10, p25, median, high = screen(
                policy, cfg, device, args.n_grid, args.n_eval_episodes,
                args.seed + it,
            )
            metric = mean_r + 0.35 * p25 + 80.0 * high
            record = (metric, mean_r, p10, p25, median, high, vector.copy())
            candidates.append(record)
            if best is None or metric > best[0]:
                best = record
                torch.save(state, args.output)
            print(
                f"iter={it:02d} cand={idx:02d} mean={mean_r:.2f} "
                f"p10={p10:.2f} p25={p25:.2f} med={median:.2f} "
                f"high={high} metric={metric:.2f}",
                flush=True,
            )

        candidates.sort(reverse=True, key=lambda item: item[0])
        elite = np.stack([item[-1] for item in candidates[:n_elite]], axis=0)
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.01)
        top = candidates[0]
        print(
            f"ITER {it:02d} TOP mean={top[1]:.2f} p25={top[3]:.2f} "
            f"high={top[5]} metric={top[0]:.2f}",
            flush=True,
        )

    metric, mean_r, p10, p25, median, high, _ = best
    print(
        f"BEST mean={mean_r:.2f} p10={p10:.2f} p25={p25:.2f} "
        f"med={median:.2f} high={high} metric={metric:.2f}",
        flush=True,
    )
    print(f"Saved best -> {args.output}", flush=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
