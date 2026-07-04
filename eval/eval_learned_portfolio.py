"""Evaluate a parameter-routed portfolio of learned CGAT policies.

This is a diagnostic experiment: it routes by link length/mass to one of several
fixed neural checkpoints using a table built from learned-policy validation
heatmaps. It does not use LQR gains, LQR actions, Riccati solves, or controller
recomputation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import eval_point, load_policy, make_fixed_env
from eval.topology import plot_topology_heatmaps, topology_reward_ceilings


DEFAULT_CANDIDATES = [
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed160_best.pt",
     "eval/results/cgat_action_scale_ppo_seed261_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed160_valbest.pt",
     "eval/results/cgat_action_scale_ppo_seed262_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed161_best.pt",
     "eval/results/cgat_action_scale_ppo_seed263_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed161_valbest.pt",
     "eval/results/cgat_action_scale_ppo_seed264_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed163_valbest.pt",
     "eval/results/cgat_action_scale_ppo_seed267_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed163_best.pt",
     "eval/results/cgat_action_scale_ppo_seed268_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed164_soup.pt",
     "eval/results/cgat_action_scale_ppo_seed269_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed165_soup_p25.pt",
     "eval/results/cgat_action_scale_ppo_seed270_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed166_valbest.pt",
     "eval/results/cgat_action_scale_ppo_seed271_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed166_best.pt",
     "eval/results/cgat_action_scale_ppo_seed272_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed167_globalscale1.20.pt",
     "eval/results/cgat_action_scale_ppo_seed273_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed170_perturb.pt",
     "eval/results/cgat_action_scale_ppo_seed274_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed170_perturb_highcells.pt",
     "eval/results/cgat_action_scale_ppo_seed275_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed171_cem_bias.pt",
     "eval/results/cgat_action_scale_ppo_seed276_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed172_cem_bias_grid10.pt",
     "eval/results/cgat_action_scale_ppo_seed277_test3.npz"),
    ("action_nonlinear", "checkpoints/cgat_action_nonlinear_ppo_seed162_valbest.pt",
     "eval/results/cgat_action_nonlinear_ppo_seed265_test3.npz"),
    ("action_nonlinear", "checkpoints/cgat_action_nonlinear_ppo_seed162_best.pt",
     "eval/results/cgat_action_nonlinear_ppo_seed266_test3.npz"),
]


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def load_candidates(cfg: dict, device: torch.device):
    candidates = []
    tables = []
    for idx, (variant, checkpoint, result_path) in enumerate(DEFAULT_CANDIDATES):
        if not (os.path.exists(checkpoint) and os.path.exists(result_path)):
            continue
        policy = load_policy(checkpoint, cfg, device, variant=variant)
        rewards = np.load(result_path)["rewards"].astype(np.float64)
        if rewards.shape != (1, 10, 10):
            raise ValueError(f"Expected (1,10,10) rewards in {result_path}, got {rewards.shape}")
        candidates.append((idx, variant, checkpoint, policy))
        tables.append(rewards[0])
    if not candidates:
        raise ValueError("No portfolio candidates were available")
    stack = np.stack(tables, axis=0)
    route = stack.argmax(axis=0)
    return candidates, route, stack


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.abs(values - value).argmin())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--n_grid", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=278)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates, route, stack = load_candidates(cfg, device)
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    length_grid = np.linspace(eval_len_lo, eval_len_hi, args.n_grid)
    mass_grid = np.linspace(eval_mass_lo, eval_mass_hi, args.n_grid)
    route_lengths = np.linspace(eval_len_lo, eval_len_hi, 10)
    route_masses = np.linspace(eval_mass_lo, eval_mass_hi, 10)

    reward_grid = np.zeros((args.n_grid, args.n_grid), dtype=np.float64)
    total = args.n_grid * args.n_grid
    done = 0
    print(f"Device: {device} | portfolio candidates={len(candidates)}", flush=True)
    print(f"Route-table oracle screen mean={stack.max(axis=0).mean():.2f}", flush=True)
    for i, length in enumerate(length_grid):
        for j, mass in enumerate(mass_grid):
            ri = nearest_index(route_lengths, float(length))
            rj = nearest_index(route_masses, float(mass))
            candidate_idx = int(route[rj, ri])
            _, variant, checkpoint, policy = candidates[candidate_idx]
            env = make_fixed_env(
                cfg, link_length=float(length), link_mass=float(mass), n_links=3)
            reward = eval_point(policy, env, args.n_eval_episodes, device)
            env.close()
            reward_grid[j, i] = reward
            done += 1
            if done % 20 == 0 or done == total:
                print(
                    f"[{done:3d}/{total}] length={length:.3f} mass={mass:.3f} "
                    f"reward={reward:.2f} via {variant}:{os.path.basename(checkpoint)}",
                    flush=True,
                )

    n_vals = np.array([3])
    reward_cube = reward_grid[None, :, :]
    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/plots", exist_ok=True)
    result_path = f"eval/results/cgat_learned_portfolio_seed{args.seed}_test3.npz"
    np.savez(
        result_path,
        lengths=length_grid,
        masses=mass_grid,
        n_links=n_vals,
        rewards=reward_cube,
        len_bounds=np.array([len_lo, len_hi]),
        mass_bounds=np.array([mass_lo, mass_hi]),
        train_topology=np.array(env_cfg["n_links_range"]),
    )
    plot_topology_heatmaps(
        length_grid,
        mass_grid,
        reward_cube,
        n_vals,
        (len_lo, len_hi),
        (mass_lo, mass_hi),
        tuple(env_cfg["n_links_range"]),
        "Learned Portfolio Heatmap",
        f"eval/plots/cgat_learned_portfolio_seed{args.seed}_ood_heatmaps_by_topology.png",
        max_rewards=topology_reward_ceilings(cfg, n_vals),
    )
    flat = reward_grid.reshape(-1)
    print(
        f"Saved {result_path} | mean={flat.mean():.2f} "
        f"p25={np.percentile(flat,25):.2f} med={np.median(flat):.2f} "
        f"p90={np.percentile(flat,90):.2f} high2000={(flat > 2000).sum()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
