"""Screen learned-only CGAT checkpoint soups on the 3-link OOD grid.

This is an experiment driver, not part of the official benchmark. It creates
weighted averages of compatible neural policy checkpoints and evaluates each
candidate on a coarse length/mass grid. No LQR gains, Riccati solves, oracle
actions, or environment-specific controller recomputation are used.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import eval_point, load_policy, make_fixed_env


DEFAULT_CHECKPOINTS = [
    "checkpoints/cgat_gain_feedback_ppo_seed122_best.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed122_valbest.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed124_best.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed124_valbest.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed125_alpha1.50.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed126_actionscale1.40.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed127_b1.40_g0.70.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed128_best.pt",
    "checkpoints/cgat_gain_feedback_ppo_seed128_valbest.pt",
]


def _safe_name(path: str) -> str:
    return Path(path).stem.replace("cgat_gain_feedback_ppo_", "")


def load_state_dicts(paths: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    states = {}
    reference_keys = None
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu")
        keys = set(state)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"Incompatible checkpoint keys for {path}: "
                f"missing={missing}, extra={extra}"
            )
        states[path] = state
    return states


def soup_state(
    states: dict[str, dict[str, torch.Tensor]],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Soup weights must sum to a positive value")
    norm = {path: weight / total for path, weight in weights.items()}
    first = states[next(iter(norm))]
    mixed = {}
    for key, value in first.items():
        if torch.is_floating_point(value):
            out = torch.zeros_like(value)
            for path, weight in norm.items():
                out.add_(states[path][key], alpha=weight)
            mixed[key] = out
        else:
            mixed[key] = value.clone()
    return mixed


def candidate_weights(paths: list[str], pair_grid: list[float], random_count: int,
                      seed: int) -> list[tuple[str, dict[str, float]]]:
    candidates = []

    for path in paths:
        candidates.append((_safe_name(path), {path: 1.0}))

    for a, b in itertools.combinations(paths, 2):
        for weight in pair_grid:
            if weight <= 0.0 or weight >= 1.0:
                continue
            name = f"soup_{_safe_name(a)}_{weight:.2f}_{_safe_name(b)}_{1 - weight:.2f}"
            candidates.append((name, {a: weight, b: 1.0 - weight}))

    if random_count > 0:
        rng = np.random.default_rng(seed)
        for idx in range(random_count):
            raw = rng.dirichlet(np.ones(len(paths)))
            weights = {path: float(weight) for path, weight in zip(paths, raw)}
            candidates.append((f"random_{idx:03d}", weights))

    return candidates


def screen_policy(policy, cfg: dict, device: torch.device, n_grid: int,
                  n_episodes: int, seed: int) -> tuple[float, float, float, float]:
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_eval = np.linspace(max(0.05, len_lo - (len_hi - len_lo)),
                           len_hi + (len_hi - len_lo), n_grid)
    mass_eval = np.linspace(max(0.05, mass_lo - (mass_hi - mass_lo)),
                            mass_hi + (mass_hi - mass_lo), n_grid)

    # Fixed order and fixed seed keep soup comparisons paired.
    np.random.seed(seed)
    torch.manual_seed(seed)

    rewards = []
    for length in len_eval:
        for mass in mass_eval:
            env = make_fixed_env(cfg, link_length=float(length), link_mass=float(mass), n_links=3)
            reward = eval_point(policy, env, n_episodes, device)
            env.close()
            rewards.append(reward)
    arr = np.asarray(rewards, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.median(arr)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="gain_feedback")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--pair_grid", nargs="+", type=float,
                        default=[0.25, 0.4, 0.5, 0.6, 0.75])
    parser.add_argument("--random_count", type=int, default=32)
    parser.add_argument("--n_grid", type=int, default=7)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=140)
    parser.add_argument("--save_best", default="checkpoints/cgat_gain_feedback_ppo_seed140_soup.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = [path for path in args.checkpoints if os.path.exists(path)]
    if len(paths) < 2:
        raise ValueError("Need at least two existing checkpoints to soup")
    states = load_state_dicts(paths)
    candidates = candidate_weights(paths, args.pair_grid, args.random_count, args.seed)

    tmp_path = "checkpoints/_tmp_cgat_soup_screen.pt"
    best = None
    results = []
    print(f"Device: {device}", flush=True)
    print(
        f"Candidates: {len(candidates)} | grid={args.n_grid}x{args.n_grid} "
        f"eps={args.n_eval_episodes}",
        flush=True,
    )
    for idx, (name, weights) in enumerate(candidates, start=1):
        state = soup_state(states, weights)
        torch.save(state, tmp_path)
        policy = load_policy(tmp_path, cfg, device, variant=args.variant)
        mean, p10, p25, median = screen_policy(
            policy, cfg, device, args.n_grid, args.n_eval_episodes, args.seed)
        metric = mean + 0.55 * p25 + 0.20 * p10
        record = (metric, mean, p10, p25, median, name, weights)
        results.append(record)
        if best is None or metric > best[0]:
            best = record
            torch.save(state, args.save_best)
        print(
            f"[{idx:03d}/{len(candidates):03d}] {name} "
            f"mean={mean:.2f} p10={p10:.2f} p25={p25:.2f} "
            f"med={median:.2f} metric={metric:.2f}"
            ,
            flush=True,
        )

    results.sort(reverse=True, key=lambda item: item[0])
    print("\nTop candidates:", flush=True)
    for rank, (metric, mean, p10, p25, median, name, weights) in enumerate(results[:10], start=1):
        compact = ", ".join(f"{_safe_name(path)}:{weight:.3f}" for path, weight in weights.items()
                            if weight > 1e-4)
        print(
            f"TOP {rank}: {name} mean={mean:.2f} p10={p10:.2f} "
            f"p25={p25:.2f} med={median:.2f} metric={metric:.2f} | {compact}"
            ,
            flush=True,
        )
    print(f"\nSaved best soup -> {args.save_best}", flush=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
