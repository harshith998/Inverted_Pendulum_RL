# Run: python3.12 eval/eval_lqr.py

"""
LQR Oracle evaluation matching eval_ppo.py structure for Tests 1, 2, 2.5, and 3.

For in-distribution cells, K is recomputed from the exact physical parameters
(optimal, non-generalizable). For OOD cells, K is computed from the nearest
in-distribution parameters (clamped to training bounds) — simulating what a
non-generalizing oracle does outside its known range.

Tests
-----
  1. 1D sweep — link_length
  2. 1D sweep — link_mass
  2.5 Topology sweep — n_links
  3. Topology heatmaps — link_length × link_mass per n_links

Also prints overall win rate across all evaluated episodes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

from env.pendulum_env import VariablePendulumEnv
from eval.topology import (
    plot_topology_heatmaps,
    plot_topology_sweep,
    seed_suffix,
    topology_tag,
    topology_values,
)


# ── Constants (must match graph/graph_builder.py) ─────────────────────────────
_RAIL_LIMIT   = 2.5
_CART_VEL_MAX = 5.0
_ANG_VEL_MAX  = 10.0
_G            = 9.81

N_SWEEP_POINTS  = 100
N_GRID          = 20
N_EPISODES      = 10
MIN_PARAM_VAL   = 0.05


# ── LQR cost matrices ─────────────────────────────────────────────────────────
# State order: [x, θ₁,…,θₙ, ẋ, θ̇₁,…,θ̇ₙ]
def _make_QR(n_links):
    n_q       = n_links + 1
    state_dim = 2 * n_q
    Q = np.zeros((state_dim, state_dim))
    Q[0, 0] = 0.1                      # cart position
    for j in range(1, n_q):
        Q[j, j] = 10.0                 # joint angles — heavily penalise tilt
    for j in range(n_q, state_dim):
        Q[j, j] = 1.0                  # velocities
    R = np.array([[0.001]])            # control effort (matches force_penalty)
    return Q, R


# ── Mass matrix at upright (θ=0, cos=1) ──────────────────────────────────────
def _mass_matrix_upright(lengths, masses, cart_mass):
    n, n_q = len(lengths), len(lengths) + 1
    L, m   = np.array(lengths), np.array(masses)
    distal = np.array([m[j:].sum() for j in range(n)])
    M = np.zeros((n_q, n_q))
    M[0, 0] = cart_mass + m.sum()
    for j in range(n):
        v = L[j] * (distal[j] - m[j] / 2)
        M[0, j+1] = M[j+1, 0] = v
    for j in range(n):
        M[j+1, j+1] = L[j]**2 * (m[j] / 3 + (distal[j] - m[j]))
    for j in range(n):
        for k in range(j+1, n):
            v = L[j] * L[k] * (distal[k] - m[k] / 2)
            M[j+1, k+1] = M[k+1, j+1] = v
    return M


# ── Gravity stiffness (linearised around upright) ─────────────────────────────
def _gravity_stiffness(lengths, masses):
    n, n_q = len(lengths), len(lengths) + 1
    L, m   = np.array(lengths), np.array(masses)
    distal = np.array([m[j:].sum() for j in range(n)])
    G = np.zeros((n_q, n_q))
    for j in range(n):
        G[j+1, j+1] = _G * L[j] * (distal[j] - m[j] / 2)
    return G


# ── Compute LQR gain K for given physical parameters ─────────────────────────
def compute_lqr_gain(lengths, masses, cart_mass, n_links):
    n_q       = n_links + 1
    state_dim = 2 * n_q
    M0        = _mass_matrix_upright(lengths, masses, cart_mass)
    G         = _gravity_stiffness(lengths, masses)
    M0_inv    = np.linalg.inv(M0)

    A = np.zeros((state_dim, state_dim))
    A[:n_q, n_q:] = np.eye(n_q)
    A[n_q:, :n_q] = M0_inv @ G

    e1 = np.zeros(n_q);  e1[0] = 1.0
    B  = np.zeros((state_dim, 1))
    B[n_q:, 0] = M0_inv @ e1

    Q, R = _make_QR(n_links)
    P    = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K    = np.linalg.solve(R, B.T @ P)   # (1, state_dim)
    return K


# ── For OOD: clamp to nearest in-distribution config ─────────────────────────
def lqr_gain_for_eval(length, mass, cfg, n_links: int | None = None):
    env_cfg   = cfg["environment"]
    if n_links is None:
        n_links = env_cfg["n_links_range"][0]
    len_lo, len_hi   = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]
    cart_mass = (cart_lo + cart_hi) / 2.0

    # Clamp to training range for K computation
    k_length = float(np.clip(length, len_lo, len_hi))
    k_mass   = float(np.clip(mass,   mass_lo, mass_hi))

    return compute_lqr_gain(
        [k_length] * n_links,
        [k_mass]   * n_links,
        cart_mass,
        n_links,
    )


# ── Extract raw state [x,θ₁,…,θₙ, ẋ,θ̇₁,…,θ̇ₙ] from graph obs ───────────────
def extract_state(obs, n_links):
    nf        = obs["node_features"]
    x         = float(nf[0, 6]) * _RAIL_LIMIT
    xdot      = float(nf[0, 7]) * _CART_VEL_MAX
    thetas    = [float(np.arctan2(nf[i+1, 3], nf[i+1, 4])) for i in range(n_links)]
    thetadots = [float(nf[i+1, 5]) * _ANG_VEL_MAX            for i in range(n_links)]
    return np.array([x] + thetas + [xdot] + thetadots)


# ── Run one episode ───────────────────────────────────────────────────────────
def run_episode(env, K, n_links, max_force, max_steps):
    obs, _ = env.reset()
    total  = 0.0
    for step in range(max_steps):
        state = extract_state(obs, n_links)
        u     = float(-(K @ state)[0])
        u     = np.clip(u, -max_force, max_force)
        obs, reward, terminated, truncated, _ = env.step(
            np.array([u], dtype=np.float32))
        total += reward
        if terminated or truncated:
            return total, False
    return total, True


# ── Env factory ───────────────────────────────────────────────────────────────
def make_fixed_env(cfg, link_length, link_mass, n_links: int | None = None):
    env_cfg   = cfg["environment"]
    if n_links is None:
        n_links = env_cfg["n_links_range"][0]
    lo, hi    = env_cfg["cart_mass_range"]
    cart_mass = (lo + hi) / 2.0
    return VariablePendulumEnv(
        n_links_range     = (n_links, n_links),
        cart_mass_range   = (cart_mass, cart_mass),
        link_length_range = (link_length, link_length),
        link_mass_range   = (link_mass,   link_mass),
        rail_limit        = env_cfg["rail_limit"],
        max_force         = env_cfg["max_force"],
        timestep          = env_cfg["timestep"],
        frame_skip        = env_cfg["frame_skip"],
        max_episode_steps = env_cfg["max_episode_steps"],
        termination_angle = env_cfg["termination_angle"],
        max_links         = env_cfg.get("max_links"),
    )


# ── Eval a single (length, mass) point ───────────────────────────────────────
def eval_point(cfg, length, mass, n_episodes, n_links: int | None = None):
    env_cfg   = cfg["environment"]
    if n_links is None:
        n_links = env_cfg["n_links_range"][0]
    max_force = env_cfg["max_force"]
    max_steps = env_cfg["max_episode_steps"]

    K   = lqr_gain_for_eval(length, mass, cfg, n_links=n_links)
    env = make_fixed_env(cfg, length, mass, n_links=n_links)
    rewards, wins = [], []
    for _ in range(n_episodes):
        try:
            r, w = run_episode(env, K, n_links, max_force, max_steps)
        except Exception:
            r, w = 0.0, False
        rewards.append(r)
        wins.append(int(w))
    env.close()
    return float(np.mean(rewards)), float(np.mean(wins))


# ── Eval range helper (same logic as eval_ppo.py) ────────────────────────────
def compute_eval_range(lo, hi):
    width   = hi - lo
    eval_lo = max(MIN_PARAM_VAL, lo - width)
    eval_hi = hi + width
    return eval_lo, eval_hi


# ── Plotting helpers ──────────────────────────────────────────────────────────
COLOR_IN  = "#5ba4cf"
COLOR_OOD = "#1a3a5c"

def plot_1d(values, rewards, train_lo, train_hi, param_name, units, plot_dir,
            n_episodes=N_EPISODES):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = [COLOR_IN if train_lo <= v <= train_hi else COLOR_OOD for v in values]
    ax.plot(values, rewards, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax.scatter(values, rewards, c=colors, s=18, zorder=3, edgecolors="none")
    ax.axvspan(train_lo, train_hi, alpha=0.10, color=COLOR_IN)
    ax.axvline(train_lo, color=COLOR_IN, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(train_hi, color=COLOR_IN, linewidth=1.5, linestyle="--", alpha=0.7)
    in_p  = mpatches.Patch(color=COLOR_IN,  label="In-distribution (exact K)")
    ood_p = mpatches.Patch(color=COLOR_OOD, label="OOD (nearest in-dist K)")
    ax.legend(handles=[in_p, ood_p], fontsize=10)
    ax.set_ylim(0, 2000)
    ax.set_xlabel(f"{param_name} ({units})", fontsize=12)
    ax.set_ylabel(f"Mean Reward ({n_episodes} eps)", fontsize=12)
    ax.set_title(f"LQR Oracle — {param_name} Sweep", fontsize=13)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    slug = param_name.lower().replace(" ", "_")
    path = os.path.join(plot_dir, f"lqr_{slug}_sweep.png")
    plt.savefig(path, dpi=150);  plt.close()
    print(f"  Plot saved → {path}")


def plot_2d(length_vals, mass_vals, reward_grid, len_bounds, mass_bounds,
            overall_winrate, plot_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(length_vals, mass_vals, reward_grid,
                       cmap="Greens", vmin=0, vmax=2000, shading="auto")
    len_lo, len_hi   = len_bounds
    mass_lo, mass_hi = mass_bounds
    rect = mpatches.Rectangle(
        (len_lo, mass_lo), len_hi - len_lo, mass_hi - mass_lo,
        linewidth=2, edgecolor="white", facecolor="none",
        linestyle="--", label="Training distribution (exact K)",
    )
    ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Reward", fontsize=11)
    ax.set_xlabel("Link Length (m)",  fontsize=12)
    ax.set_ylabel("Link Mass (kg)",   fontsize=12)
    ax.set_title(
        f"LQR Oracle Heatmap — Length × Mass\n"
        f"(inside box = exact K, outside = nearest in-dist K)  "
        f"Win rate: {overall_winrate:.1f}%",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    path = os.path.join(plot_dir, "lqr_heatmap.png")
    plt.savefig(path, dpi=150);  plt.close()
    print(f"  Plot saved → {path}")


# ── Test helpers ─────────────────────────────────────────────────────────────
def run_topology_sweep(cfg, n_episodes):
    env_cfg = cfg["environment"]
    len_mid = sum(env_cfg["link_length_range"]) / 2.0
    mass_mid = sum(env_cfg["link_mass_range"]) / 2.0
    train_topology = tuple(env_cfg["n_links_range"])
    n_links_vals = topology_values(cfg)
    rewards, wins = [], []

    print(f"\n{'='*60}")
    print("[Test 2.5] LQR Topology sweep")
    print(f"  n_links : {n_links_vals}")
    print(f"  Train topology : {train_topology}")
    print(f"  Length={len_mid:.3f}m  Mass={mass_mid:.3f}kg  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, n_links in enumerate(n_links_vals):
        r, w = eval_point(cfg, len_mid, mass_mid, n_episodes, n_links=n_links)
        rewards.append(r)
        wins.append(w)
        print(f"  [{i+1:3d}/{len(n_links_vals)}] n_links={n_links}  "
              f"reward={r:8.2f}  win={w*100:.0f}%  "
              f"[{topology_tag(n_links, train_topology)}]")
    return np.array(n_links_vals), np.array(rewards), np.array(wins), train_topology


def run_topology_heatmaps(cfg, n_episodes, n_grid):
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    n_links_vals = topology_values(cfg)

    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    length_grid = np.linspace(eval_len_lo, eval_len_hi, n_grid)
    mass_grid = np.linspace(eval_mass_lo, eval_mass_hi, n_grid)
    reward_cube = np.zeros((len(n_links_vals), n_grid, n_grid))
    win_cube = np.zeros((len(n_links_vals), n_grid, n_grid))
    total_cells = len(n_links_vals) * n_grid * n_grid

    print(f"\n{'='*60}")
    print(f"[Test 3] LQR Topology heatmaps — {len(n_links_vals)}×{n_grid}×{n_grid} grid ({total_cells} cells)")
    print(f"  n_links: {n_links_vals}")
    print(f"  Length : {eval_len_lo:.3f}m → {eval_len_hi:.3f}m")
    print(f"  Mass   : {eval_mass_lo:.3f}kg → {eval_mass_hi:.3f}kg")
    print(f"{'='*60}")

    done = 0
    for k, n_links in enumerate(n_links_vals):
        for i, length in enumerate(length_grid):
            for j, mass in enumerate(mass_grid):
                r, w = eval_point(cfg, length, mass, n_episodes, n_links=n_links)
                reward_cube[k, j, i] = r
                win_cube[k, j, i] = w
                done += 1
                if done % 20 == 0 or done == total_cells:
                    print(f"  [{done:4d}/{total_cells}]  n_links={n_links} "
                          f"length={length:.3f}m  mass={mass:.3f}kg  "
                          f"reward={r:7.2f}  win={w*100:.0f}%")

    return (length_grid, mass_grid, reward_cube, win_cube, np.array(n_links_vals),
            (len_lo, len_hi), (mass_lo, mass_hi), tuple(env_cfg["n_links_range"]))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LQR oracle OOD evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tests", nargs="+",
        choices=["1", "2", "2.5", "3"],
        default=["1", "2", "2.5", "3"])
    parser.add_argument("--n_eval_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--n_sweep_points", type=int, default=N_SWEEP_POINTS)
    parser.add_argument("--n_grid", type=int, default=N_GRID)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg          = cfg["environment"]
    len_lo,  len_hi  = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]

    os.makedirs("eval/plots", exist_ok=True)
    os.makedirs("eval/results", exist_ok=True)

    all_rewards, all_wins = [], []

    suffix = seed_suffix(args.seed)

    if "1" in args.tests:
        mass_mid = (mass_lo + mass_hi) / 2.0
        eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
        length_vals = np.linspace(eval_len_lo, eval_len_hi, args.n_sweep_points)
        t1_rewards = []

        print(f"\n{'='*60}")
        print(f"[Test 1] LQR Link Length sweep")
        print(f"  Eval  : {eval_len_lo:.3f}m → {eval_len_hi:.3f}m  ({args.n_sweep_points} pts)")
        print(f"  Train : [{len_lo:.3f}, {len_hi:.3f}]m  |  mass fixed={mass_mid:.3f}kg")
        print(f"{'='*60}")

        for i, length in enumerate(length_vals):
            r, w = eval_point(cfg, length, mass_mid, args.n_eval_episodes)
            t1_rewards.append(r)
            all_rewards.append(r);  all_wins.append(w)
            tag = "IN " if len_lo <= length <= len_hi else "OOD"
            if (i + 1) % 20 == 0 or (i + 1) == args.n_sweep_points:
                print(f"  [{i+1:3d}/{args.n_sweep_points}] length={length:.4f}m  "
                      f"reward={r:8.2f}  win={w*100:.0f}%  [{tag}]")

        plot_1d(length_vals, np.array(t1_rewards), len_lo, len_hi,
                "Link Length", "m", "eval/plots", args.n_eval_episodes)
        np.savez(
            f"eval/results/lqr_oracle{suffix}_test1_length.npz",
            values=length_vals, rewards=np.array(t1_rewards),
            bounds=np.array([len_lo, len_hi]),
        )

    if "2" in args.tests:
        len_mid = (len_lo + len_hi) / 2.0
        eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
        mass_vals_sweep = np.linspace(eval_mass_lo, eval_mass_hi, args.n_sweep_points)
        t2_rewards = []

        print(f"\n{'='*60}")
        print(f"[Test 2] LQR Link Mass sweep")
        print(f"  Eval  : {eval_mass_lo:.3f}kg → {eval_mass_hi:.3f}kg  ({args.n_sweep_points} pts)")
        print(f"  Train : [{mass_lo:.3f}, {mass_hi:.3f}]kg  |  length fixed={len_mid:.3f}m")
        print(f"{'='*60}")

        for i, mass in enumerate(mass_vals_sweep):
            r, w = eval_point(cfg, len_mid, mass, args.n_eval_episodes)
            t2_rewards.append(r)
            all_rewards.append(r);  all_wins.append(w)
            tag = "IN " if mass_lo <= mass <= mass_hi else "OOD"
            if (i + 1) % 20 == 0 or (i + 1) == args.n_sweep_points:
                print(f"  [{i+1:3d}/{args.n_sweep_points}] mass={mass:.4f}kg  "
                      f"reward={r:8.2f}  win={w*100:.0f}%  [{tag}]")

        plot_1d(mass_vals_sweep, np.array(t2_rewards), mass_lo, mass_hi,
                "Link Mass", "kg", "eval/plots", args.n_eval_episodes)
        np.savez(
            f"eval/results/lqr_oracle{suffix}_test2_mass.npz",
            values=mass_vals_sweep, rewards=np.array(t2_rewards),
            bounds=np.array([mass_lo, mass_hi]),
        )

    if "2.5" in args.tests:
        n_vals, topo_rewards, topo_wins, topo_bounds = run_topology_sweep(
            cfg, args.n_eval_episodes)
        all_rewards.extend(topo_rewards.tolist())
        all_wins.extend(topo_wins.tolist())
        plot_topology_sweep(
            n_vals, topo_rewards, topo_bounds,
            "LQR Oracle Topology Sweep",
            f"eval/plots/lqr{suffix}_topology_sweep.png",
        )
        np.savez(
            f"eval/results/lqr_oracle{suffix}_test25_topology.npz",
            n_links=n_vals, rewards=topo_rewards,
            wins=topo_wins, train_topology=np.array(topo_bounds),
        )

    if "3" in args.tests:
        length_grid, mass_grid, reward_cube, win_cube, n_vals, l_bounds, m_bounds, topo_bounds = (
            run_topology_heatmaps(cfg, args.n_eval_episodes, args.n_grid)
        )
        all_rewards.extend(reward_cube.reshape(-1).tolist())
        all_wins.extend(win_cube.reshape(-1).tolist())

        plot_topology_heatmaps(
            length_grid, mass_grid, reward_cube, n_vals,
            l_bounds, m_bounds, topo_bounds,
            "LQR Oracle Heatmaps",
            f"eval/plots/lqr{suffix}_heatmaps_by_topology.png",
        )

        np.savez(
            f"eval/results/lqr_oracle{suffix}_test3.npz",
            lengths=length_grid, masses=mass_grid, n_links=n_vals,
            rewards=reward_cube, wins=win_cube,
            len_bounds=np.array(l_bounds),
            mass_bounds=np.array(m_bounds),
            train_topology=np.array(topo_bounds),
        )
        print(f"  Results saved -> eval/results/lqr_oracle{suffix}_test3.npz")

    print(f"\n{'='*55}")
    if all_wins:
        print(f"  Overall win rate : {float(np.mean(all_wins)) * 100:.1f}%")
    if all_rewards:
        print(f"  Mean reward      : {np.mean(all_rewards):.2f}")
    print(f"{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()
