"""CEM search over low-dimensional learned param-residual feedback biases.

This optimizes neural policy parameters directly in simulation. It does not use
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


BIAS_KEYS = (
    "gain_net.5.bias",
    "param_residual_net.5.bias",
    "mean_head.bias",
)


def vectorize(base: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int]]]:
    pieces = []
    specs = []
    for key in BIAS_KEYS:
        if key not in base:
            raise KeyError(key)
        value = base[key].detach().cpu().float().reshape(-1)
        pieces.append(value.numpy())
        specs.append((key, tuple(base[key].shape), value.numel()))
    return np.concatenate(pieces).astype(np.float64), specs


def apply_delta(base: dict[str, torch.Tensor],
                specs: list[tuple[str, tuple[int, ...], int]],
                delta: np.ndarray) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in base.items()}
    offset = 0
    for key, shape, size in specs:
        chunk = torch.tensor(delta[offset:offset + size], dtype=state[key].dtype)
        state[key] = state[key] + chunk.reshape(shape)
        offset += size
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
    rewards = []
    for i, length in enumerate(lengths):
        for j, mass in enumerate(masses):
            env = make_fixed_env(
                cfg, link_length=float(length), link_mass=float(mass), n_links=3)
            try:
                for ep in range(n_episodes):
                    obs, _ = env.reset(seed=int(seed + 1009 * i + 9173 * j + ep))
                    total = 0.0
                    done = False
                    while not done:
                        action = policy.get_deterministic_action(obs, device)
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
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.median(arr)),
        int((arr >= 2000.0).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--elite_frac", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--n_grid", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=356)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base = torch.load(args.checkpoint, map_location="cpu")
    _, specs = vectorize(base)
    dim = sum(size for _, _, size in specs)
    mean = np.zeros(dim, dtype=np.float64)
    std = np.full(dim, args.sigma, dtype=np.float64)
    n_elite = max(1, int(round(args.pop * args.elite_frac)))
    tmp_path = "checkpoints/_tmp_param_residual_bias_cem.pt"
    best = None

    print(
        f"Device: {device} | dim={dim} iters={args.iters} pop={args.pop} "
        f"elite={n_elite} sigma={args.sigma}",
        flush=True,
    )
    for it in range(args.iters):
        samples = []
        if it == 0:
            samples.append(np.zeros(dim, dtype=np.float64))
        while len(samples) < args.pop:
            samples.append(rng.normal(mean, std).astype(np.float64))

        candidates = []
        for idx, delta in enumerate(samples):
            state = apply_delta(base, specs, delta)
            torch.save(state, tmp_path)
            policy = load_policy(tmp_path, cfg, device, variant="param_residual")
            mean_r, p10, p25, median, high = screen(
                policy, cfg, device, args.n_grid, args.n_eval_episodes,
                args.seed + 100 * it,
            )
            metric = mean_r + 0.25 * p25 + 45.0 * high
            record = (metric, mean_r, p10, p25, median, high, delta.copy())
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
        std = np.maximum(elite.std(axis=0), 0.006)
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
