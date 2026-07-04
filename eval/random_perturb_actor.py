"""Random-search small learned actor-head perturbations on a coarse OOD grid.

This is a learned-policy optimizer: it perturbs neural policy parameters and
keeps the candidate that scores best in simulation. It does not call LQR, use
oracle actions, or recompute any controller at runtime.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import eval_point, load_policy, make_fixed_env


PERTURB_KEYS = (
    "action_scale_net.3.weight",
    "action_scale_net.3.bias",
    "gain_net.5.weight",
    "gain_net.5.bias",
    "mean_head.weight",
    "mean_head.bias",
)


def perturb_state(base: dict[str, torch.Tensor], rng: np.random.Generator,
                  sigma: float) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in base.items()}
    for key in PERTURB_KEYS:
        if key not in state or not torch.is_floating_point(state[key]):
            continue
        noise = torch.tensor(
            rng.normal(0.0, sigma, size=tuple(state[key].shape)),
            dtype=state[key].dtype,
        )
        state[key] = state[key] + noise
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
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.015)
    parser.add_argument("--n_grid", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=170)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base = torch.load(args.checkpoint, map_location="cpu")
    tmp_path = "checkpoints/_tmp_actor_perturb.pt"

    best = None
    print(
        f"Device: {device} | candidates={args.candidates} "
        f"sigma={args.sigma} grid={args.n_grid}x{args.n_grid}",
        flush=True,
    )
    for idx in range(args.candidates + 1):
        if idx == 0:
            state = {key: value.clone() for key, value in base.items()}
            name = "base"
        else:
            state = perturb_state(base, rng, args.sigma)
            name = f"perturb_{idx:03d}"
        torch.save(state, tmp_path)
        policy = load_policy(tmp_path, cfg, device, variant=args.variant)
        mean, p10, p25, median, high = screen(
            policy, cfg, device, args.n_grid, args.n_eval_episodes, args.seed)
        metric = mean + 0.5 * p25 + 18.0 * high
        record = (metric, mean, p10, p25, median, high, name, state)
        if best is None or metric > best[0]:
            best = record
            torch.save(state, args.output)
        print(
            f"[{idx:03d}/{args.candidates:03d}] {name} "
            f"mean={mean:.2f} p10={p10:.2f} p25={p25:.2f} "
            f"med={median:.2f} high={high} metric={metric:.2f}",
            flush=True,
        )

    metric, mean, p10, p25, median, high, name, _ = best
    print(
        f"BEST {name}: mean={mean:.2f} p10={p10:.2f} p25={p25:.2f} "
        f"med={median:.2f} high={high} metric={metric:.2f}",
        flush=True,
    )
    print(f"Saved best -> {args.output}", flush=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
