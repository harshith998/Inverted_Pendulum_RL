# Run: python3.12 eval/eval_dqn.py --policy gnn
#      python3.12 eval/eval_dqn.py --policy mlp
#      python3.12 eval/eval_dqn.py --policy random
#      python3.12 eval/eval_dqn.py --policy gnn --tests 1 2       # skip heatmap
#      python3.12 eval/eval_dqn.py --policy gnn --tests 3          # heatmap only
#      python3.12 eval/eval_dqn.py --policy gnn --checkpoint checkpoints/my_model.pt

"""
OOD evaluation suite for DQN policies (GNN or MLP).
Tests
-----
  1. 1D sweep — link_length   (100 pts × 200 eps each)
  2. 1D sweep — link_mass     (100 pts × 200 eps each)
  3. 2D heatmap — link_length × link_mass  (20×20 grid)
  4. Few-shot adaptation is not implemented for DQN in this script
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
from models.gnn_dqn import GNNDQNPolicy
from models.mlp_dqn import MLPDQNPolicy
from models.random_baseline import RandomDQNPolicy
from eval.topology import (
    plot_topology_heatmaps,
    plot_topology_sweep,
    seed_suffix,
    topology_tag,
    topology_values,
)


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

N_SWEEP_POINTS   = 100   # points along each 1D sweep
N_GRID_1D        = 20    # grid size per axis for Test 3  (20×20 = 400 cells)
N_EVAL_EPISODES  = 200   # episodes per evaluation point
MIN_PARAM_VAL    = 0.05  # physical floor — prevents degenerate near-zero lengths/masses

# Plot colours
COLOR_IN_DIST = "#5ba4cf"   # light blue  — points inside training distribution
COLOR_OOD     = "#1a3a5c"   # dark navy   — points outside training distribution
CMAP_HEATMAP  = "RdYlGn"    # 2D heatmap: red (low) → yellow → green (high)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_policy(policy_name: str, checkpoint_path: str, cfg: dict,
                device: torch.device):
    env_cfg = cfg["environment"]
    dqn_cfg = cfg["dqn"]
    max_links    = env_cfg.get("max_links", env_cfg["n_links_range"][1])
    n_bins    = dqn_cfg["n_action_bins"]
    hidden    = dqn_cfg["hidden_dim"]
    n_layers  = dqn_cfg["n_mp_layers"]

    if policy_name == "random":
        return RandomDQNPolicy(n_bins)

    if policy_name == "gnn":
        policy = GNNDQNPolicy(n_bins, hidden=hidden, n_mp_layers=n_layers,
                              max_links=max_links)
    else:
        policy = MLPDQNPolicy(n_bins, max_links=max_links, hidden=hidden)

    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] checkpoint missing keys (new arch keys init to default): {missing}")
    if unexpected:
        print(f"  [warn] checkpoint has unexpected keys (ignored): {unexpected}")
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Environment factory — fixes all params at given values
# ---------------------------------------------------------------------------

def make_fixed_env(cfg: dict, link_length: float, link_mass: float,
                   cart_mass: float | None = None,
                   n_links: int | None = None) -> VariablePendulumEnv:
    """Build an env where every reset produces exactly the given parameters."""
    env_cfg = cfg["environment"]
    if cart_mass is None:
        lo, hi = env_cfg["cart_mass_range"]
        cart_mass = (lo + hi) / 2.0
    if n_links is None:
        n_links = env_cfg["n_links_range"][0]
    return VariablePendulumEnv(
        n_links_range       = (n_links, n_links),
        cart_mass_range     = (cart_mass, cart_mass),
        link_length_range   = (link_length, link_length),
        link_mass_range     = (link_mass, link_mass),
        rail_limit          = env_cfg["rail_limit"],
        max_force           = env_cfg["max_force"],
        timestep            = env_cfg["timestep"],
        frame_skip          = env_cfg["frame_skip"],
        max_episode_steps   = env_cfg["max_episode_steps"],
        termination_angle   = env_cfg["termination_angle"],
        max_links           = env_cfg.get("max_links"),
    )


# ---------------------------------------------------------------------------
# Core evaluation — runs n_episodes and returns mean reward
# ---------------------------------------------------------------------------

def eval_point(policy, env, n_episodes: int, act_bins: np.ndarray,
               device: torch.device) -> float:
    rewards = []
    for _ in range(n_episodes):
        try:
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                idx   = policy.get_action(obs, epsilon=0.0, device=device)
                force = act_bins[idx]
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([force], dtype=np.float32))
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        except Exception:
            rewards.append(0.0)   # treat MuJoCo crashes on extreme params as 0
    return float(np.mean(rewards)) if rewards else 0.0


# ---------------------------------------------------------------------------
# Eval range helpers
# ---------------------------------------------------------------------------

def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    """Extend [lo, hi] by 100% of its width on each side, clamped to MIN_PARAM_VAL."""
    width    = hi - lo
    eval_lo  = max(MIN_PARAM_VAL, lo - width)
    eval_hi  = hi + width
    return eval_lo, eval_hi


# ---------------------------------------------------------------------------
# Test 1 — 1D link_length sweep
# ---------------------------------------------------------------------------

def run_test1(policy, cfg, act_bins, device, n_episodes, n_points):
    env_cfg  = cfg["environment"]
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
        env = make_fixed_env(cfg, link_length=length, link_mass=mass_mid)
        r = eval_point(policy, env, n_episodes, act_bins, device)
        env.close()
        tag = "IN " if len_lo <= length <= len_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] length={length:.4f}m  reward={r:8.2f}  [{tag}]")

    return length_vals, np.array(rewards), len_lo, len_hi


# ---------------------------------------------------------------------------
# Test 2 — 1D link_mass sweep
# ---------------------------------------------------------------------------

def run_test2(policy, cfg, act_bins, device, n_episodes, n_points):
    env_cfg    = cfg["environment"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_mid    = sum(env_cfg["link_length_range"]) / 2.0

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
        env = make_fixed_env(cfg, link_length=len_mid, link_mass=mass)
        r = eval_point(policy, env, n_episodes, act_bins, device)
        env.close()
        tag = "IN " if mass_lo <= mass <= mass_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] mass={mass:.4f}kg  reward={r:8.2f}  [{tag}]")

    return mass_vals, np.array(rewards), mass_lo, mass_hi


# ---------------------------------------------------------------------------
# Test 2.5 — topology sweep
# ---------------------------------------------------------------------------

def run_test25(policy, cfg, act_bins, device, n_episodes):
    env_cfg = cfg["environment"]
    len_mid = sum(env_cfg["link_length_range"]) / 2.0
    mass_mid = sum(env_cfg["link_mass_range"]) / 2.0
    train_topology = tuple(env_cfg["n_links_range"])
    n_links_vals = topology_values(cfg)
    rewards = []

    print(f"\n{'='*60}")
    print("[Test 2.5] Topology sweep")
    print(f"  n_links : {n_links_vals}")
    print(f"  Train topology : {train_topology}")
    print(f"  Length={len_mid:.3f}m  Mass={mass_mid:.3f}kg  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, n_links in enumerate(n_links_vals):
        env = make_fixed_env(
            cfg, link_length=len_mid, link_mass=mass_mid, n_links=n_links)
        r = eval_point(policy, env, n_episodes, act_bins, device)
        env.close()
        rewards.append(r)
        print(f"  [{i+1:3d}/{len(n_links_vals)}] n_links={n_links}  "
              f"reward={r:8.2f}  [{topology_tag(n_links, train_topology)}]")

    return np.array(n_links_vals), np.array(rewards), train_topology


# ---------------------------------------------------------------------------
# Test 3 — topology-aware heatmaps
# ---------------------------------------------------------------------------

def run_test3(policy, cfg, act_bins, device, n_episodes, n_grid):
    env_cfg  = cfg["environment"]
    len_lo, len_hi   = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    n_links_vals = topology_values(cfg)

    len_eval_lo,  len_eval_hi  = compute_eval_range(len_lo,  len_hi)
    mass_eval_lo, mass_eval_hi = compute_eval_range(mass_lo, mass_hi)

    length_vals = np.linspace(len_eval_lo,  len_eval_hi,  n_grid)
    mass_vals   = np.linspace(mass_eval_lo, mass_eval_hi, n_grid)

    reward_cube = np.full((len(n_links_vals), n_grid, n_grid), np.nan)

    total = len(n_links_vals) * n_grid * n_grid
    print(f"\n{'='*60}")
    print(f"[Test 3] Topology heatmaps — {len(n_links_vals)}×{n_grid}×{n_grid} grid ({total} cells)")
    print(f"  n_links: {n_links_vals}")
    print(f"  Length : {len_eval_lo:.3f}m → {len_eval_hi:.3f}m")
    print(f"  Mass   : {mass_eval_lo:.3f}kg → {mass_eval_hi:.3f}kg")
    print(f"{'='*60}")

    done = 0
    for k, n_links in enumerate(n_links_vals):
        for i, length in enumerate(length_vals):
            for j, mass in enumerate(mass_vals):
                env = make_fixed_env(
                    cfg, link_length=length, link_mass=mass, n_links=n_links)
                r = eval_point(policy, env, n_episodes, act_bins, device)
                env.close()
                reward_cube[k, j, i] = r
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done:4d}/{total}] n_links={n_links} "
                          f"length={length:.3f}m  mass={mass:.3f}kg  reward={r:.2f}")

    return (length_vals, mass_vals, reward_cube, np.array(n_links_vals),
            (len_lo, len_hi), (mass_lo, mass_hi), tuple(env_cfg["n_links_range"]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_1d(values, rewards, train_lo, train_hi,
            param_name, units, policy_name, plot_dir, file_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = [COLOR_IN_DIST if train_lo <= v <= train_hi else COLOR_OOD
              for v in values]

    # Connecting line (grey, thin) then scatter on top
    ax.plot(values, rewards, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax.scatter(values, rewards, c=colors, s=18, zorder=3, edgecolors="none")

    # Training-distribution shaded band
    ax.axvspan(train_lo, train_hi, alpha=0.10, color=COLOR_IN_DIST)
    ax.axvline(train_lo, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(train_hi, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)

    # Vertical labels for training bounds
    ymin, ymax = ax.get_ylim()
    mid_y = (ymin + ymax) / 2
    ax.text(train_lo, mid_y, f" train lo\n {train_lo:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)
    ax.text(train_hi, mid_y, f" train hi\n {train_hi:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)

    in_patch  = mpatches.Patch(color=COLOR_IN_DIST, label="In-distribution")
    ood_patch = mpatches.Patch(color=COLOR_OOD, label="OOD")
    ax.legend(handles=[in_patch, ood_patch], fontsize=10)

    ax.set_ylim(0, 2000)
    ax.set_xlabel(f"{param_name} ({units})", fontsize=12)
    ax.set_ylabel(f"Mean Reward ({N_EVAL_EPISODES} eps)", fontsize=12)
    ax.set_title(f"OOD Generalisation — {param_name} | {policy_name.upper()} DQN",
                 fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    slug = param_name.lower().replace(" ", "_")
    path = os.path.join(plot_dir, f"{policy_name}_dqn{file_suffix}_{slug}_sweep.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


def plot_2d(length_vals, mass_vals, reward_grid,
            len_bounds, mass_bounds, policy_name, plot_dir):
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(length_vals, mass_vals, reward_grid,
                       cmap="Greens", vmin=0, vmax=2000, shading="auto")

    # Training distribution rectangle
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
        f"OOD Heatmap — Length × Mass | {policy_name.upper()} DQN\n"
        f"(white dashed box = training distribution)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"{policy_name}_dqn_ood_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OOD evaluation for DQN policies.")
    parser.add_argument("--policy",   choices=["gnn", "mlp"], default="gnn")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
        help="Path to .pt checkpoint. Defaults to checkpoints/{policy}_dqn_best.pt")
    parser.add_argument("--tests", nargs="+",
        choices=["1", "2", "2.5", "3", "4"],
        default=["1", "2", "2.5", "3"],
        help="Which tests to run (default: 1 2 2.5 3)")
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES,
        help=f"Episodes per eval point (default {N_EVAL_EPISODES})")
    parser.add_argument("--n_sweep_points", type=int, default=N_SWEEP_POINTS,
        help=f"Points along each 1D sweep (default {N_SWEEP_POINTS})")
    parser.add_argument("--n_grid", type=int, default=N_GRID_1D,
        help=f"Grid size per axis for 2D heatmap (default {N_GRID_1D} → {N_GRID_1D}²={N_GRID_1D**2} cells)")
    parser.add_argument("--seed", type=int, default=None,
        help="Eval seed and checkpoint suffix. Uses checkpoints/{policy}_dqn_seedN_best.pt.")
    args = parser.parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "4" in args.tests:
        raise NotImplementedError(
            "Test 4 few-shot adaptation is implemented for PPO/CGAT policies. "
            "DQN needs a separate replay-buffer/target-network adaptation protocol.")

    checkpoint = args.checkpoint or f"checkpoints/{args.policy}_dqn{seed_suffix(args.seed)}_best.pt"
    if args.policy != "random" and not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Run training first:  python3.12 training/train_dqn.py --policy {args.policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Policy     : {args.policy}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Tests      : {args.tests}")
    print(f"Seed       : {args.seed if args.seed is not None else 'none'}")
    print(f"Episodes/pt: {args.n_eval_episodes}")

    policy = load_policy(args.policy, checkpoint, cfg, device)

    # ------------------------------------------------------------------
    # Inference timing benchmark
    # ------------------------------------------------------------------
    env_cfg  = cfg["environment"]
    dqn_cfg  = cfg["dqn"]
    act_bins = np.linspace(-env_cfg["max_force"], env_cfg["max_force"],
                           dqn_cfg["n_action_bins"])

    _timing_env = make_fixed_env(cfg, link_length=0.75, link_mass=1.05)
    _obs, _     = _timing_env.reset()
    _timing_env.close()
    N_TIMING    = 1000
    # Warmup
    for _ in range(50):
        policy.get_action(_obs, epsilon=0.0, device=device)
    # Timed runs
    _t0 = time.perf_counter()
    for _ in range(N_TIMING):
        policy.get_action(_obs, epsilon=0.0, device=device)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    print(f"\nInference timing ({N_TIMING} calls):")
    print(f"  Total   : {_elapsed_ms:.2f} ms")
    print(f"  Per call: {_elapsed_ms / N_TIMING:.4f} ms  "
          f"({1000 / (_elapsed_ms / N_TIMING):.0f} inferences/sec)\n")

    os.makedirs("eval/plots", exist_ok=True)
    os.makedirs("eval/results", exist_ok=True)

    suffix = seed_suffix(args.seed)

    if "1" in args.tests:
        l_vals, l_rewards, l_lo, l_hi = run_test1(
            policy, cfg, act_bins, device,
            args.n_eval_episodes, args.n_sweep_points)
        plot_1d(l_vals, l_rewards, l_lo, l_hi,
                "Link Length", "m", args.policy, "eval/plots", suffix)
        np.savez(
            f"eval/results/{args.policy}_dqn{suffix}_test1_length.npz",
            values=l_vals, rewards=l_rewards,
            bounds=np.array([l_lo, l_hi]),
        )

    if "2" in args.tests:
        m_vals, m_rewards, m_lo, m_hi = run_test2(
            policy, cfg, act_bins, device,
            args.n_eval_episodes, args.n_sweep_points)
        plot_1d(m_vals, m_rewards, m_lo, m_hi,
                "Link Mass", "kg", args.policy, "eval/plots", suffix)
        np.savez(
            f"eval/results/{args.policy}_dqn{suffix}_test2_mass.npz",
            values=m_vals, rewards=m_rewards,
            bounds=np.array([m_lo, m_hi]),
        )

    if "2.5" in args.tests:
        n_vals, topo_rewards, topo_bounds = run_test25(
            policy, cfg, act_bins, device, args.n_eval_episodes)
        plot_topology_sweep(
            n_vals, topo_rewards, topo_bounds,
            f"Topology Sweep | {args.policy.upper()} DQN",
            f"eval/plots/{args.policy}_dqn{suffix}_topology_sweep.png",
        )
        np.savez(
            f"eval/results/{args.policy}_dqn{suffix}_test25_topology.npz",
            n_links=n_vals, rewards=topo_rewards,
            train_topology=np.array(topo_bounds),
        )

    if "3" in args.tests:
        l_v, m_v, r_cube, n_vals, l_bounds, m_bounds, topo_bounds = run_test3(
            policy, cfg, act_bins, device,
            args.n_eval_episodes, args.n_grid)
        plot_topology_heatmaps(
            l_v, m_v, r_cube, n_vals, l_bounds, m_bounds, topo_bounds,
            f"OOD Heatmaps | {args.policy.upper()} DQN",
            f"eval/plots/{args.policy}_dqn{suffix}_ood_heatmaps_by_topology.png",
        )
        np.savez(
            f"eval/results/{args.policy}_dqn{suffix}_test3.npz",
            lengths=l_v, masses=m_v, n_links=n_vals, rewards=r_cube,
            len_bounds=np.array(l_bounds), mass_bounds=np.array(m_bounds),
            train_topology=np.array(topo_bounds),
        )

    print("Done.")


if __name__ == "__main__":
    main()
