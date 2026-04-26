# Run: python3.12 eval/eval_ppo.py --policy gnn_mpnn
#      python3.12 eval/eval_ppo.py --policy gnn_transformer
#      python3.12 eval/eval_ppo.py --policy mlp
#      python3.12 eval/eval_ppo.py --policy random
#      python3.12 eval/eval_ppo.py --policy gnn_mpnn --tests 1 2       # skip heatmap
#      python3.12 eval/eval_ppo.py --policy gnn_mpnn --tests 3          # heatmap only (uses cache)
#      python3.12 eval/eval_ppo.py --policy gnn_mpnn --checkpoint checkpoints/my_model.pt

"""
OOD evaluation suite for PPO policies (GNN-MPNN, GNN-Transformer, or MLP).

Tests
-----
  1. 1D sweep — link_length   (100 pts × 200 eps each)
  2. 1D sweep — link_mass     (100 pts × 200 eps each)
  3. 2D heatmap — link_length × link_mass  (20×20 grid, reuses Tests 1+2 cache)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.pendulum_env import VariablePendulumEnv
from models.gnn_mpnn_ppo import GNNMPNNPPOPolicy
from models.gnn_transformer_ppo import GNNTransformerPPOPolicy
from models.mlp_ppo import MLPPPOPolicy
from models.random_baseline import RandomPPOPolicy


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

N_SWEEP_POINTS  = 100
N_GRID_1D       = 20
N_EVAL_EPISODES = 10
MIN_PARAM_VAL   = 0.05

COLOR_IN_DIST = "#5ba4cf"
COLOR_OOD     = "#1a3a5c"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy(policy_name: str, checkpoint_path: str, cfg: dict,
                device: torch.device):
    env_cfg  = cfg["environment"]
    ppo_cfg  = cfg["ppo"]
    max_links = env_cfg["n_links_range"][1]
    hidden    = ppo_cfg["hidden_dim"]
    n_layers  = ppo_cfg["n_layers"]
    n_heads   = ppo_cfg["n_heads"]
    dropout   = ppo_cfg["dropout"]
    max_force = env_cfg["max_force"]

    t_cfg = ppo_cfg.get("gnn_transformer", {})

    if policy_name == "random":
        return RandomPPOPolicy(max_force=max_force)

    if policy_name == "gnn_mpnn":
        policy = GNNMPNNPPOPolicy(
            hidden=hidden, n_layers=n_layers, max_links=max_links,
            dropout=dropout, max_force=max_force)
    elif policy_name == "gnn_transformer":
        t_heads  = t_cfg.get("n_heads",  n_heads)
        t_layers = t_cfg.get("n_layers", n_layers)
        policy = GNNTransformerPPOPolicy(
            hidden=hidden, n_layers=t_layers, n_heads=t_heads,
            max_links=max_links, dropout=dropout, max_force=max_force)
    else:
        policy = MLPPPOPolicy(
            hidden=hidden, max_links=max_links,
            dropout=dropout, max_force=max_force)

    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] checkpoint missing keys: {missing}")
    if unexpected:
        print(f"  [warn] checkpoint unexpected keys (ignored): {unexpected}")
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_fixed_env(cfg: dict, link_length: float, link_mass: float,
                   cart_mass: float | None = None) -> VariablePendulumEnv:
    env_cfg = cfg["environment"]
    if cart_mass is None:
        lo, hi = env_cfg["cart_mass_range"]
        cart_mass = (lo + hi) / 2.0
    n_links = env_cfg["n_links_range"][0]
    return VariablePendulumEnv(
        n_links_range     = (n_links, n_links),
        cart_mass_range   = (cart_mass, cart_mass),
        link_length_range = (link_length, link_length),
        link_mass_range   = (link_mass, link_mass),
        rail_limit        = env_cfg["rail_limit"],
        max_force         = env_cfg["max_force"],
        timestep          = env_cfg["timestep"],
        frame_skip        = env_cfg["frame_skip"],
        max_episode_steps = env_cfg["max_episode_steps"],
        termination_angle = env_cfg["termination_angle"],
    )


# ---------------------------------------------------------------------------
# Core evaluation — continuous action, no bins
# ---------------------------------------------------------------------------

def eval_point(policy, env, n_episodes: int, device: torch.device) -> float:
    rewards = []
    for _ in range(n_episodes):
        try:
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                # get_action returns (force_float, log_prob, value) — only need force
                action, _, _ = policy.get_action(obs, device)
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32))
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        except Exception:
            rewards.append(0.0)
    return float(np.mean(rewards)) if rewards else 0.0


# ---------------------------------------------------------------------------
# Eval range helpers
# ---------------------------------------------------------------------------

def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width   = hi - lo
    eval_lo = max(MIN_PARAM_VAL, lo - width)
    eval_hi = hi + width
    return eval_lo, eval_hi


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _key(length: float, mass: float) -> tuple:
    return (round(float(length), 6), round(float(mass), 6))


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    return {_key(l, m): float(r)
            for l, m, r in zip(data["lengths"], data["masses"], data["rewards"])}


def save_cache(path: str, cache: dict):
    if not cache:
        return
    keys    = list(cache.keys())
    lengths = np.array([k[0] for k in keys], dtype=np.float64)
    masses  = np.array([k[1] for k in keys], dtype=np.float64)
    rewards = np.array([cache[k] for k in keys], dtype=np.float64)
    np.savez(path, lengths=lengths, masses=masses, rewards=rewards)


# ---------------------------------------------------------------------------
# Test 1 — 1D link_length sweep
# ---------------------------------------------------------------------------

def run_test1(policy, cfg, device, cache, n_episodes, n_points):
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_mid = sum(env_cfg["link_mass_range"]) / 2.0

    eval_lo, eval_hi = compute_eval_range(len_lo, len_hi)
    length_vals      = np.linspace(eval_lo, eval_hi, n_points)
    rewards          = []

    print(f"\n{'='*60}")
    print(f"[Test 1] Link Length sweep")
    print(f"  Eval range : {eval_lo:.3f}m → {eval_hi:.3f}m  ({n_points} pts)")
    print(f"  Train dist : [{len_lo:.3f}, {len_hi:.3f}]m")
    print(f"  Link mass fixed at {mass_mid:.3f} kg  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, length in enumerate(length_vals):
        k = _key(length, mass_mid)
        if k in cache:
            r   = cache[k]
            tag = "cached"
        else:
            env = make_fixed_env(cfg, link_length=length, link_mass=mass_mid)
            r   = eval_point(policy, env, n_episodes, device)
            env.close()
            cache[k] = r
            tag = "IN " if len_lo <= length <= len_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] length={length:.4f}m  reward={r:8.2f}  [{tag}]")

    return length_vals, np.array(rewards), len_lo, len_hi


# ---------------------------------------------------------------------------
# Test 2 — 1D link_mass sweep
# ---------------------------------------------------------------------------

def run_test2(policy, cfg, device, cache, n_episodes, n_points):
    env_cfg = cfg["environment"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_mid = sum(env_cfg["link_length_range"]) / 2.0

    eval_lo, eval_hi = compute_eval_range(mass_lo, mass_hi)
    mass_vals        = np.linspace(eval_lo, eval_hi, n_points)
    rewards          = []

    print(f"\n{'='*60}")
    print(f"[Test 2] Link Mass sweep")
    print(f"  Eval range : {eval_lo:.3f}kg → {eval_hi:.3f}kg  ({n_points} pts)")
    print(f"  Train dist : [{mass_lo:.3f}, {mass_hi:.3f}]kg")
    print(f"  Link length fixed at {len_mid:.3f} m  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, mass in enumerate(mass_vals):
        k = _key(len_mid, mass)
        if k in cache:
            r   = cache[k]
            tag = "cached"
        else:
            env = make_fixed_env(cfg, link_length=len_mid, link_mass=mass)
            r   = eval_point(policy, env, n_episodes, device)
            env.close()
            cache[k] = r
            tag = "IN " if mass_lo <= mass <= mass_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] mass={mass:.4f}kg  reward={r:8.2f}  [{tag}]")

    return mass_vals, np.array(rewards), mass_lo, mass_hi


# ---------------------------------------------------------------------------
# Test 3 — 2D heatmap (link_length × link_mass)
# ---------------------------------------------------------------------------

def run_test3(policy, cfg, device, cache, n_episodes, n_grid):
    env_cfg = cfg["environment"]
    len_lo, len_hi   = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]

    len_eval_lo,  len_eval_hi  = compute_eval_range(len_lo,  len_hi)
    mass_eval_lo, mass_eval_hi = compute_eval_range(mass_lo, mass_hi)

    length_vals = np.linspace(len_eval_lo,  len_eval_hi,  n_grid)
    mass_vals   = np.linspace(mass_eval_lo, mass_eval_hi, n_grid)
    reward_grid = np.full((n_grid, n_grid), np.nan)

    to_compute = []
    cache_hits = 0
    for i, length in enumerate(length_vals):
        for j, mass in enumerate(mass_vals):
            k = _key(length, mass)
            if k in cache:
                reward_grid[j, i] = cache[k]
                cache_hits += 1
            else:
                to_compute.append((i, j, length, mass))

    total = n_grid * n_grid
    print(f"\n{'='*60}")
    print(f"[Test 3] 2D Heatmap — {n_grid}×{n_grid} grid ({total} cells)")
    print(f"  Length : {len_eval_lo:.3f}m → {len_eval_hi:.3f}m")
    print(f"  Mass   : {mass_eval_lo:.3f}kg → {mass_eval_hi:.3f}kg")
    print(f"  Cache hits: {cache_hits}/{total} | To compute: {len(to_compute)}")
    print(f"{'='*60}")

    for idx, (i, j, length, mass) in enumerate(to_compute):
        env = make_fixed_env(cfg, link_length=length, link_mass=mass)
        r   = eval_point(policy, env, n_episodes, device)
        env.close()
        k                 = _key(length, mass)
        cache[k]          = r
        reward_grid[j, i] = r
        if (idx + 1) % 10 == 0 or (idx + 1) == len(to_compute):
            print(f"  [{idx+1:4d}/{len(to_compute)}] "
                  f"length={length:.3f}m  mass={mass:.3f}kg  reward={r:.2f}")

    return (length_vals, mass_vals, reward_grid,
            (len_lo, len_hi), (mass_lo, mass_hi))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_1d(values, rewards, train_lo, train_hi,
            param_name, units, policy_name, plot_dir):
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = [COLOR_IN_DIST if train_lo <= v <= train_hi else COLOR_OOD
              for v in values]

    ax.plot(values, rewards, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax.scatter(values, rewards, c=colors, s=18, zorder=3, edgecolors="none")

    ax.axvspan(train_lo, train_hi, alpha=0.10, color=COLOR_IN_DIST)
    ax.axvline(train_lo, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(train_hi, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)

    ymin, ymax = ax.get_ylim()
    mid_y = (ymin + ymax) / 2
    ax.text(train_lo, mid_y, f" train lo\n {train_lo:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)
    ax.text(train_hi, mid_y, f" train hi\n {train_hi:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)

    in_patch  = mpatches.Patch(color=COLOR_IN_DIST, label="In-distribution")
    ood_patch = mpatches.Patch(color=COLOR_OOD,     label="OOD")
    ax.legend(handles=[in_patch, ood_patch], fontsize=10)

    ax.set_ylim(0, 2000)
    ax.set_xlabel(f"{param_name} ({units})", fontsize=12)
    ax.set_ylabel(f"Mean Reward ({N_EVAL_EPISODES} eps)", fontsize=12)
    ax.set_title(f"OOD Generalisation — {param_name} | {policy_name.upper()} PPO",
                 fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    slug = param_name.lower().replace(" ", "_")
    path = os.path.join(plot_dir, f"{policy_name}_ppo_{slug}_sweep.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


def plot_2d(length_vals, mass_vals, reward_grid,
            len_bounds, mass_bounds, policy_name, plot_dir):
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(length_vals, mass_vals, reward_grid,
                       cmap="Greens", vmin=0, vmax=2000, shading="auto")

    len_lo, len_hi   = len_bounds
    mass_lo, mass_hi = mass_bounds
    rect = mpatches.Rectangle(
        (len_lo, mass_lo), len_hi - len_lo, mass_hi - mass_lo,
        linewidth=2, edgecolor="white", facecolor="none",
        linestyle="--", label="Training distribution",
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Reward", fontsize=11)

    ax.set_xlabel("Link Length (m)", fontsize=12)
    ax.set_ylabel("Link Mass (kg)", fontsize=12)
    ax.set_title(
        f"OOD Heatmap — Length × Mass | {policy_name.upper()} PPO\n"
        f"(white dashed box = training distribution)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"{policy_name}_ppo_ood_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OOD evaluation for PPO policies.")
    parser.add_argument("--policy",
                        choices=["gnn_mpnn", "gnn_transformer", "mlp", "random"],
                        default="gnn_mpnn")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
        help="Path to .pt checkpoint. Defaults to checkpoints/{policy}_ppo_best.pt")
    parser.add_argument("--tests", nargs="+", type=int, choices=[1, 2, 3],
        default=[1, 2, 3])
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--n_sweep_points",  type=int, default=N_SWEEP_POINTS)
    parser.add_argument("--n_grid",          type=int, default=N_GRID_1D)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint or f"checkpoints/{args.policy}_ppo_best.pt"
    if args.policy != "random" and not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Run training first:  python3.12 training/train_ppo.py --policy {args.policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Policy     : {args.policy}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Tests      : {args.tests}")
    print(f"Episodes/pt: {args.n_eval_episodes}")

    policy = load_policy(args.policy, checkpoint, cfg, device)

    # ------------------------------------------------------------------
    # Inference timing benchmark
    # ------------------------------------------------------------------
    _timing_env = make_fixed_env(cfg, link_length=0.75, link_mass=1.05)
    _obs, _     = _timing_env.reset()
    _timing_env.close()
    N_TIMING    = 1000
    # Warmup
    for _ in range(50):
        policy.get_action(_obs, device)
    # Timed runs
    _t0 = time.perf_counter()
    for _ in range(N_TIMING):
        policy.get_action(_obs, device)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    print(f"\nInference timing ({N_TIMING} calls):")
    print(f"  Total   : {_elapsed_ms:.2f} ms")
    print(f"  Per call: {_elapsed_ms / N_TIMING:.4f} ms  "
          f"({1000 / (_elapsed_ms / N_TIMING):.0f} inferences/sec)\n")

    os.makedirs("eval/plots", exist_ok=True)
    os.makedirs("eval/cache", exist_ok=True)
    cache_path = f"eval/cache/{args.policy}_ppo_ood_cache.npz"
    cache      = load_cache(cache_path)
    print(f"Cache      : {len(cache)} existing entries  ({cache_path})")

    try:
        if 1 in args.tests:
            l_vals, l_rewards, l_lo, l_hi = run_test1(
                policy, cfg, device, cache,
                args.n_eval_episodes, args.n_sweep_points)
            save_cache(cache_path, cache)
            plot_1d(l_vals, l_rewards, l_lo, l_hi,
                    "Link Length", "m", args.policy, "eval/plots")

        if 2 in args.tests:
            m_vals, m_rewards, m_lo, m_hi = run_test2(
                policy, cfg, device, cache,
                args.n_eval_episodes, args.n_sweep_points)
            save_cache(cache_path, cache)
            plot_1d(m_vals, m_rewards, m_lo, m_hi,
                    "Link Mass", "kg", args.policy, "eval/plots")

        if 3 in args.tests:
            l_v, m_v, r_grid, l_bounds, m_bounds = run_test3(
                policy, cfg, device, cache,
                args.n_eval_episodes, args.n_grid)
            save_cache(cache_path, cache)
            plot_2d(l_v, m_v, r_grid, l_bounds, m_bounds,
                    args.policy, "eval/plots")

    finally:
        save_cache(cache_path, cache)
        print(f"\nCache saved: {len(cache)} total entries → {cache_path}")
        print("Done.")


if __name__ == "__main__":
    main()
