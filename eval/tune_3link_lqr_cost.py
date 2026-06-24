"""Coarse reward tuning for 3-link LQR Q/R.

The repository's LQR baseline uses a hand-chosen cost:
    cart_q=0.1, angle_q=10, velocity_q=10, R=0.1

That is a stabilizing controller, but not necessarily the best controller for
the environment reward, which penalizes force and rail drift. This script
searches a compact set of diagonal Q/R values on the fixed 3-link task.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.linalg
import yaml

from env.pendulum_env import VariablePendulumEnv
from eval.eval_lqr import _linearize_mujoco, extract_state


def make_qr(n_links: int, cart_q: float, angle_q: float, vel_q: float, r: float):
    n_q = n_links + 1
    state_dim = 2 * n_q
    Q = np.zeros((state_dim, state_dim))
    Q[0, 0] = cart_q
    for j in range(1, n_q):
        Q[j, j] = angle_q
    for j in range(n_q, state_dim):
        Q[j, j] = vel_q
    R = np.array([[r]], dtype=float)
    return Q, R


def compute_gain(length, mass, cart_mass, cfg, costs, cache):
    key = (round(length, 4), round(mass, 4), round(cart_mass, 4), costs)
    if key in cache:
        return cache[key]
    env_cfg = cfg["environment"]
    A, B = _linearize_mujoco(
        [length] * 3,
        [mass] * 3,
        cart_mass,
        3,
        env_cfg["rail_limit"],
        env_cfg["max_force"],
        env_cfg["timestep"],
    )
    Q, R = make_qr(3, *costs)
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    cache[key] = K
    return K


def make_env(cfg, length, mass):
    env_cfg = cfg["environment"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]
    cart_mass = (cart_lo + cart_hi) / 2.0
    return VariablePendulumEnv(
        n_links_range=(3, 3),
        cart_mass_range=(cart_mass, cart_mass),
        link_length_range=(length, length),
        link_mass_range=(mass, mass),
        rail_limit=env_cfg["rail_limit"],
        max_force=env_cfg["max_force"],
        timestep=env_cfg["timestep"],
        frame_skip=env_cfg["frame_skip"],
        max_episode_steps=env_cfg["max_episode_steps"],
        termination_angle=env_cfg["termination_angle"],
        max_links=3,
    )


def eval_costs(cfg, lengths, masses, costs, episodes, cache):
    env_cfg = cfg["environment"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]
    cart_mass = (cart_lo + cart_hi) / 2.0
    rewards = []
    wins = []
    for length in lengths:
        for mass in masses:
            K = compute_gain(float(length), float(mass), cart_mass, cfg, costs, cache)
            env = make_env(cfg, float(length), float(mass))
            try:
                for _ in range(episodes):
                    obs, _ = env.reset()
                    total = 0.0
                    done = False
                    steps = 0
                    while not done:
                        state = extract_state(obs, 3)
                        u = float(-(K @ state)[0])
                        u = float(np.clip(u, -env_cfg["max_force"], env_cfg["max_force"]))
                        obs, reward, terminated, truncated, _ = env.step(
                            np.array([u], dtype=np.float32)
                        )
                        total += reward
                        steps += 1
                        done = terminated or truncated
                    rewards.append(total)
                    wins.append(1 if steps >= env_cfg["max_episode_steps"] else 0)
            finally:
                env.close()
    return float(np.mean(rewards)), float(np.mean(wins))


def main():
    parser = argparse.ArgumentParser(description="Tune 3-link LQR reward costs.")
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--grid", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_width = len_hi - len_lo
    mass_width = mass_hi - mass_lo
    lengths = np.linspace(max(0.05, len_lo - len_width), len_hi + len_width, args.grid)
    masses = np.linspace(max(0.05, mass_lo - mass_width), mass_hi + mass_width, args.grid)

    candidates = []
    for cart_q in [0.03, 0.1, 0.3]:
        for angle_q in [3.0, 10.0, 30.0]:
            for vel_q in [3.0, 10.0, 30.0]:
                for r in [0.05, 0.1, 0.3, 1.0]:
                    candidates.append((cart_q, angle_q, vel_q, r))

    cache = {}
    rows = []
    best = None
    print(f"Tuning {len(candidates)} candidates on {args.grid}x{args.grid} grid, {args.episodes} eps/cell")
    for idx, costs in enumerate(candidates, start=1):
        mean_reward, win_rate = eval_costs(cfg, lengths, masses, costs, args.episodes, cache)
        rows.append((*costs, mean_reward, win_rate))
        if best is None or mean_reward > best[-2]:
            best = rows[-1]
            print(
                f"  new best {idx:3d}/{len(candidates)} costs={costs} "
                f"mean={mean_reward:.1f} win={win_rate*100:.1f}%"
            )
        elif idx % 20 == 0:
            print(f"  checked {idx:3d}/{len(candidates)} current={mean_reward:.1f} best={best[-2]:.1f}")

    os.makedirs("eval/results", exist_ok=True)
    out = "eval/results/three_link_lqr_cost_tuning.npz"
    np.savez(
        out,
        rows=np.array(rows, dtype=float),
        columns=np.array(["cart_q", "angle_q", "vel_q", "r", "mean_reward", "win_rate"]),
        lengths=lengths,
        masses=masses,
        best=np.array(best, dtype=float),
    )
    print(f"Best cart_q angle_q vel_q r mean win: {best}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
