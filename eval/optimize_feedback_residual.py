"""Black-box optimize a tiny learned feedback residual on top of CGAT.

This is a rollout-only learned residual. It does not solve LQR, use LQR gains,
use oracle actions, or recompute a controller from system matrices. The residual
is a small linear feedback term optimized by CEM on environment returns.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from eval.eval_cgat import load_policy, make_fixed_env
from eval.topology import plot_topology_heatmaps, topology_reward_ceilings


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def state_features(obs: dict) -> np.ndarray:
    node = obs["node_features"]
    edge = obs["edge_features"]
    cart = node[0]
    joints = node[1:4]
    sin_t = joints[:, 3]
    cos_t = joints[:, 4]
    theta = np.arctan2(sin_t, cos_t)
    theta_dot = joints[:, 5]
    length = edge[0:6:2, 0] * 0.9 + 0.3
    mass = edge[0:6:2, 1] * 1.9 + 0.1
    feats = np.concatenate(
        [
            [cart[6], cart[7]],
            theta,
            theta_dot,
            np.tanh(theta_dot / 0.5),
            sin_t,
            sin_t * theta_dot,
            np.log(np.clip(length, 0.03, None) / 0.6),
            np.log(np.clip(mass, 0.02, None) / 1.0),
        ]
    )
    return np.clip(feats, -5.0, 5.0).astype(np.float64)


def residual_action(policy, obs: dict, device: torch.device, weights: np.ndarray,
                    residual_scale: float, max_force: float) -> float:
    base = policy.get_deterministic_action(obs, device)
    residual = residual_scale * float(np.dot(weights, state_features(obs)))
    return float(np.clip(base + residual, -max_force, max_force))


def eval_candidate(policy, cfg, device, weights, points, n_episodes, residual_scale):
    env_cfg = cfg["environment"]
    rewards = []
    for length, mass in points:
        env = make_fixed_env(cfg, float(length), float(mass), n_links=3)
        try:
            for _ in range(n_episodes):
                obs, _ = env.reset()
                total = 0.0
                done = False
                while not done:
                    action = residual_action(
                        policy, obs, device, weights,
                        residual_scale, env_cfg["max_force"],
                    )
                    obs, reward, terminated, truncated, _ = env.step(
                        np.array([action], dtype=np.float32)
                    )
                    total += reward
                    done = terminated or truncated
                rewards.append(total)
        finally:
            env.close()
    rewards = np.asarray(rewards, dtype=np.float64)
    return float(rewards.mean()), float(np.percentile(rewards, 25))


def cem_optimize(policy, cfg, device, points, seed, iterations, population,
                 elite_frac, sigma, n_episodes, residual_scale, score_p25_weight):
    rng = np.random.default_rng(seed)
    probe_env = make_fixed_env(cfg, points[0][0], points[0][1], n_links=3)
    try:
        dim = len(state_features(probe_env.reset()[0]))
    finally:
        probe_env.close()
    mean = np.zeros(dim, dtype=np.float64)
    std = np.ones(dim, dtype=np.float64) * sigma
    best_w = mean.copy()
    best_score = -np.inf
    best_stats = (0.0, 0.0)
    elite_n = max(2, int(population * elite_frac))

    for it in range(iterations):
        samples = rng.normal(mean, std, size=(population, dim))
        scored = []
        for w in samples:
            mean_r, p25 = eval_candidate(
                policy, cfg, device, w, points, n_episodes, residual_scale
            )
            score = mean_r + score_p25_weight * p25
            scored.append((score, mean_r, p25, w))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([x[3] for x in scored[:elite_n]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 0.05 * sigma
        if scored[0][0] > best_score:
            best_score, best_stats, best_w = scored[0][0], scored[0][1:3], scored[0][3].copy()
        print(
            f"iter {it+1:02d}/{iterations} best_score={best_score:.1f} "
            f"mean={best_stats[0]:.1f} p25={best_stats[1]:.1f} "
            f"sigma={std.mean():.3f}",
            flush=True,
        )
    return best_w


def heatmap_eval(policy, cfg, device, weights, n_grid, n_episodes, seed, residual_scale):
    np.random.seed(seed)
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    lengths = np.linspace(eval_len_lo, eval_len_hi, n_grid)
    masses = np.linspace(eval_mass_lo, eval_mass_hi, n_grid)
    rewards = np.zeros((n_grid, n_grid), dtype=np.float64)
    total = n_grid * n_grid
    done = 0
    for i, length in enumerate(lengths):
        for j, mass in enumerate(masses):
            mean_r, _ = eval_candidate(
                policy, cfg, device, weights, [(length, mass)], n_episodes, residual_scale
            )
            rewards[j, i] = mean_r
            done += 1
            if done % 20 == 0 or done == total:
                print(
                    f"[{done:3d}/{total}] length={length:.3f} mass={mass:.3f} "
                    f"reward={mean_r:.2f}",
                    flush=True,
                )
    n_vals = np.array([3])
    cube = rewards[None, :, :]
    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/plots", exist_ok=True)
    result_path = f"eval/results/cgat_feedback_residual_cem_seed{seed}_test3.npz"
    np.savez(
        result_path,
        lengths=lengths,
        masses=masses,
        n_links=n_vals,
        rewards=cube,
        weights=weights,
        len_bounds=np.array([len_lo, len_hi]),
        mass_bounds=np.array([mass_lo, mass_hi]),
        train_topology=np.array(env_cfg["n_links_range"]),
    )
    plot_topology_heatmaps(
        lengths,
        masses,
        cube,
        n_vals,
        (len_lo, len_hi),
        (mass_lo, mass_hi),
        tuple(env_cfg["n_links_range"]),
        "CGAT + CEM Feedback Residual",
        f"eval/plots/cgat_feedback_residual_cem_seed{seed}_ood_heatmaps_by_topology.png",
        max_rewards=topology_reward_ceilings(cfg, n_vals),
    )
    flat = rewards.reshape(-1)
    print(
        f"Saved {result_path} | mean={flat.mean():.2f} "
        f"p25={np.percentile(flat,25):.2f} med={np.median(flat):.2f} "
        f"p90={np.percentile(flat,90):.2f} high2000={(flat >= 2000).sum()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="action_scale")
    parser.add_argument("--checkpoint", default="checkpoints/cgat_action_scale_ppo_seed160_best.pt")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite_frac", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--residual_scale", type=float, default=2.0)
    parser.add_argument("--n_train_episodes", type=int, default=1)
    parser.add_argument("--n_eval_episodes", type=int, default=3)
    parser.add_argument("--n_grid", type=int, default=10)
    parser.add_argument("--score_p25_weight", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, cfg, device, variant=args.variant)

    points = [
        (0.05, 0.05), (0.28, 0.05), (0.50, 0.05), (0.73, 0.05),
        (1.19, 0.05), (1.64, 0.05), (2.10, 0.05), (0.05, 0.48),
        (0.28, 0.48), (0.50, 0.48), (0.73, 0.48), (0.05, 1.76),
        (0.28, 3.90), (1.20, 2.00), (1.64, 2.19), (2.10, 3.90),
    ]
    weights = cem_optimize(
        policy,
        cfg,
        device,
        points,
        args.seed,
        args.iterations,
        args.population,
        args.elite_frac,
        args.sigma,
        args.n_train_episodes,
        args.residual_scale,
        args.score_p25_weight,
    )
    print("best_weights", " ".join(f"{x:.5f}" for x in weights), flush=True)
    heatmap_eval(
        policy,
        cfg,
        device,
        weights,
        args.n_grid,
        args.n_eval_episodes,
        args.seed,
        args.residual_scale,
    )


if __name__ == "__main__":
    main()
