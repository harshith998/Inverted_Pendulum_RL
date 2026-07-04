"""Screen small value-routed learned-policy ensembles on a coarse grid."""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_learned_ensemble import eval_point
from eval.eval_cgat import load_policy


CANDIDATES = [
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed160_best.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed167_globalscale1.20.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed164_soup.pt"),
    ("action_nonlinear", "checkpoints/cgat_action_nonlinear_ppo_seed162_best.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed163_best.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed163_valbest.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed170_perturb.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed171_cem_bias.pt"),
]


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def short_name(member: tuple[str, str]) -> str:
    return os.path.basename(member[1]).replace("cgat_", "").replace("_ppo_", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--subset_size", type=int, default=4)
    parser.add_argument("--n_grid", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=283)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = [member for member in CANDIDATES if os.path.exists(member[1])]
    loaded = {
        member: load_policy(member[1], cfg, device, variant=member[0])
        for member in available
    }

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    lengths = np.linspace(eval_len_lo, eval_len_hi, args.n_grid)
    masses = np.linspace(eval_mass_lo, eval_mass_hi, args.n_grid)

    results = []
    combos = list(itertools.combinations(available, args.subset_size))
    print(f"Device: {device} | subsets={len(combos)} size={args.subset_size}", flush=True)
    for idx, combo in enumerate(combos, start=1):
        policies = [loaded[member] for member in combo]
        rewards = []
        for length in lengths:
            for mass in masses:
                rewards.append(eval_point(
                    policies, cfg, float(length), float(mass),
                    args.n_eval_episodes, device, mode="value",
                ))
        arr = np.asarray(rewards, dtype=np.float64)
        metric = float(arr.mean() + 0.35 * np.percentile(arr, 25) + 60.0 * (arr > 2000).sum())
        results.append((metric, float(arr.mean()), float(np.percentile(arr, 25)),
                        float(np.median(arr)), int((arr > 2000).sum()), combo))
        print(
            f"[{idx:03d}/{len(combos):03d}] mean={arr.mean():.2f} "
            f"p25={np.percentile(arr,25):.2f} med={np.median(arr):.2f} "
            f"high={(arr>2000).sum()} metric={metric:.2f} | "
            + ", ".join(short_name(m) for m in combo),
            flush=True,
        )

    results.sort(reverse=True, key=lambda item: item[0])
    print("\nTop subsets:", flush=True)
    for rank, (metric, mean, p25, median, high, combo) in enumerate(results[:8], start=1):
        print(
            f"TOP {rank}: mean={mean:.2f} p25={p25:.2f} med={median:.2f} "
            f"high={high} metric={metric:.2f} | "
            + " ".join(f"--member {variant}={path}" for variant, path in combo),
            flush=True,
        )


if __name__ == "__main__":
    main()
