"""CEM optimize the learned param-action-scale head in simulation.

This is pure learned-policy optimization: candidates are neural policy
parameters, scored only by MuJoCo rollout reward. It does not use LQR actions,
gains, labels, Riccati solves, or controller recomputation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import load_policy, make_fixed_env


SCALE_KEYS = (
    "param_action_scale_net.3.weight",
    "param_action_scale_net.3.bias",
)


def vectorize(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int]]]:
    pieces = []
    specs = []
    for key in SCALE_KEYS:
        value = state[key].detach().cpu().float().reshape(-1)
        pieces.append(value.numpy())
        specs.append((key, tuple(state[key].shape), value.numel()))
    return np.concatenate(pieces).astype(np.float64), specs


def apply_vector(base: dict[str, torch.Tensor],
                 specs: list[tuple[str, tuple[int, ...], int]],
                 vector: np.ndarray) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in base.items()}
    offset = 0
    for key, shape, size in specs:
        chunk = torch.tensor(vector[offset:offset + size], dtype=state[key].dtype)
        state[key] = chunk.reshape(shape)
        offset += size
    return state


def eval_policy(policy, cfg: dict, device: torch.device, n_grid: int,
                n_episodes: int, seed: int) -> tuple[float, float, float, int]:
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_width = len_hi - len_lo
    mass_width = mass_hi - mass_lo
    lengths = np.linspace(max(0.05, len_lo - len_width), len_hi + len_width, n_grid)
    masses = np.linspace(max(0.05, mass_lo - mass_width), mass_hi + mass_width, n_grid)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--elite_frac", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--n_grid", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=353)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base = torch.load(args.checkpoint, map_location="cpu")
    base_vec, specs = vectorize(base)
    mean = base_vec.copy()
    std = np.full_like(mean, float(args.sigma), dtype=np.float64)
    n_elite = max(1, int(round(args.pop * args.elite_frac)))
    tmp_path = "checkpoints/_tmp_param_action_scale_cem.pt"
    best = None

    print(
        f"Device: {device} | dim={mean.size} iters={args.iters} pop={args.pop} "
        f"elite={n_elite} sigma={args.sigma}",
        flush=True,
    )
    for it in range(args.iters):
        candidates = []
        samples = []
        if it == 0:
            samples.append(base_vec.copy())
        while len(samples) < args.pop:
            samples.append(rng.normal(mean, std).astype(np.float64))
        for idx, vector in enumerate(samples):
            state = apply_vector(base, specs, vector)
            torch.save(state, tmp_path)
            policy = load_policy(tmp_path, cfg, device, variant="param_action_scale")
            mean_r, p25, median, high = eval_policy(
                policy, cfg, device, args.n_grid, args.n_eval_episodes,
                args.seed + 100 * it,
            )
            metric = mean_r + 0.25 * p25 + 35.0 * high
            record = (metric, mean_r, p25, median, high, vector.copy())
            candidates.append(record)
            if best is None or metric > best[0]:
                best = record
                torch.save(state, args.output)
            print(
                f"iter={it:02d} cand={idx:02d} mean={mean_r:.2f} "
                f"p25={p25:.2f} med={median:.2f} high={high} "
                f"metric={metric:.2f}",
                flush=True,
            )

        candidates.sort(reverse=True, key=lambda item: item[0])
        elite = np.stack([item[-1] for item in candidates[:n_elite]], axis=0)
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.005)
        top = candidates[0]
        print(
            f"ITER {it:02d} TOP mean={top[1]:.2f} p25={top[2]:.2f} "
            f"med={top[3]:.2f} high={top[4]} metric={top[0]:.2f}",
            flush=True,
        )

    metric, mean_r, p25, median, high, _ = best
    print(
        f"BEST mean={mean_r:.2f} p25={p25:.2f} med={median:.2f} "
        f"high={high} metric={metric:.2f}",
        flush=True,
    )
    print(f"Saved best -> {args.output}", flush=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
