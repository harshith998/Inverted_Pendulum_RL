"""Focused 3-link evaluation for CGAT against simulated LQR.

This script avoids topology variation and compares controllers on the same
3-link length/mass grid. It reports:
  - CGAT deterministic policy reward
  - simulated LQR with K recomputed for the evaluated physical parameters
  - clamped LQR using train-range-clamped parameters, matching the old OOD idea
  - old in-distribution ceiling, which is useful but not a simulated controller
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from env.pendulum_env import VariablePendulumEnv
from eval.eval_cgat import load_policy
from eval.eval_lqr import compute_lqr_gain, extract_state
from eval.few_shot import select_eval_action
from models.cgat import load_cgat_variant


MIN_PARAM_VAL = 0.05


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(MIN_PARAM_VAL, lo - width), hi + width


def make_env(cfg: dict, length: float, mass: float) -> VariablePendulumEnv:
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


def deterministic_policy_action(policy, obs, device):
    return select_eval_action(policy, obs, device, stochastic=False)


def eval_cgat_point(policy, cfg, device, length, mass, episodes, *,
                    residual_lqr=False, residual_gate: str = "none",
                    residual_lqr_costs=None, reset_seeds=None):
    rewards = []
    env = make_env(cfg, length, mass)
    K = None
    if residual_lqr:
        env_cfg = cfg["environment"]
        cart_lo, cart_hi = env_cfg["cart_mass_range"]
        cart_mass = (cart_lo + cart_hi) / 2.0
        if residual_lqr_costs is None:
            K = compute_lqr_gain(
                [float(length)] * 3,
                [float(mass)] * 3,
                cart_mass,
                3,
                rail_limit=env_cfg["rail_limit"],
                max_force=env_cfg["max_force"],
                timestep=env_cfg["timestep"],
            )
        else:
            import scipy.linalg
            from eval.eval_lqr import _linearize_mujoco

            cart_q, angle_q, vel_q, r = residual_lqr_costs
            A, B = _linearize_mujoco(
                [float(length)] * 3,
                [float(mass)] * 3,
                cart_mass,
                3,
                env_cfg["rail_limit"],
                env_cfg["max_force"],
                env_cfg["timestep"],
            )
            Q = np.zeros((8, 8))
            Q[0, 0] = cart_q
            Q[1:4, 1:4] = np.eye(3) * angle_q
            Q[4:8, 4:8] = np.eye(4) * vel_q
            R = np.array([[r]], dtype=float)
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ P)
    try:
        for ep in range(episodes):
            seed = None if reset_seeds is None else int(reset_seeds[ep])
            obs, _ = env.reset(seed=seed)
            total = 0.0
            done = False
            while not done:
                action = deterministic_policy_action(policy, obs, device)
                if residual_lqr:
                    base = float(-(K @ extract_state(obs, 3))[0])
                    if residual_gate == "same_sign" and base * action < 0.0:
                        action = 0.0
                    elif residual_gate == "half_base":
                        limit = 0.5 * abs(base)
                        action = float(np.clip(action, -limit, limit))
                    action = float(np.clip(base + action, -cfg["environment"]["max_force"], cfg["environment"]["max_force"]))
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32)
                )
                total += reward
                done = terminated or truncated
            rewards.append(total)
    finally:
        env.close()
    return float(np.mean(rewards))


def load_residual_policy(checkpoint_path: str, cfg: dict, device: torch.device,
                         variant: str, residual_scale: float):
    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    h_cfg = ppo_cfg.get("cgat", {})
    policy = load_cgat_variant(
        variant,
        hidden=h_cfg.get("hidden_dim", ppo_cfg["hidden_dim"]),
        n_icga_layers=h_cfg.get("n_icga_layers", 2),
        n_heads=h_cfg.get("n_heads", 2),
        max_links=env_cfg.get("max_links", 3),
        max_force=residual_scale,
    )
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    policy.to(device)
    policy.eval()
    return policy


def eval_lqr_point(cfg, length, mass, episodes, *, clamped: bool, reset_seeds=None):
    env_cfg = cfg["environment"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]
    cart_mass = (cart_lo + cart_hi) / 2.0
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    k_length = float(np.clip(length, len_lo, len_hi)) if clamped else float(length)
    k_mass = float(np.clip(mass, mass_lo, mass_hi)) if clamped else float(mass)

    K = compute_lqr_gain(
        [k_length] * 3,
        [k_mass] * 3,
        cart_mass,
        3,
        rail_limit=env_cfg["rail_limit"],
        max_force=env_cfg["max_force"],
        timestep=env_cfg["timestep"],
    )

    rewards = []
    env = make_env(cfg, length, mass)
    try:
        for ep in range(episodes):
            seed = None if reset_seeds is None else int(reset_seeds[ep])
            obs, _ = env.reset(seed=seed)
            total = 0.0
            done = False
            while not done:
                state = extract_state(obs, 3)
                action = float(-(K @ state)[0])
                action = float(np.clip(action, -env_cfg["max_force"], env_cfg["max_force"]))
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32)
                )
                total += reward
                done = terminated or truncated
            rewards.append(total)
    finally:
        env.close()
    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description="3-link-only CGAT vs LQR eval.")
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--residual_lqr", action="store_true",
        help="Treat checkpoint as a residual policy added on top of exact LQR.")
    parser.add_argument("--residual_scale", type=float, default=4.0)
    parser.add_argument("--residual_gate", choices=["none", "same_sign", "half_base"],
        default="none")
    parser.add_argument("--residual_lqr_costs", nargs=4, type=float, default=None,
        metavar=("CART_Q", "ANGLE_Q", "VEL_Q", "R"))
    parser.add_argument("--eval_seed", type=int, default=12345)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "" if args.seed is None else f"_seed{args.seed}"
    checkpoint = args.checkpoint or f"checkpoints/cgat_{args.variant}_ppo{suffix}_best.pt"
    if args.residual_lqr:
        policy = load_residual_policy(
            checkpoint, cfg, device, args.variant, args.residual_scale)
    else:
        policy = load_policy(checkpoint, cfg, device, variant=args.variant)

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)

    if args.quick:
        lengths = np.array([eval_len_lo, (len_lo + len_hi) / 2.0, eval_len_hi])
        masses = np.array([eval_mass_lo, (mass_lo + mass_hi) / 2.0, eval_mass_hi])
    else:
        lengths = np.linspace(eval_len_lo, eval_len_hi, args.grid)
        masses = np.linspace(eval_mass_lo, eval_mass_hi, args.grid)

    cgat = np.zeros((len(masses), len(lengths)))
    lqr_exact = np.zeros_like(cgat)
    lqr_clamped = np.zeros_like(cgat)

    total = cgat.size
    done = 0
    for i, mass in enumerate(masses):
        for j, length in enumerate(lengths):
            cell_seed = args.eval_seed + i * 1000 + j * 100
            reset_seeds = [cell_seed + ep for ep in range(args.episodes)]
            cgat[i, j] = eval_cgat_point(
                policy, cfg, device, length, mass, args.episodes,
                residual_lqr=args.residual_lqr,
                residual_gate=args.residual_gate,
                residual_lqr_costs=args.residual_lqr_costs,
                reset_seeds=reset_seeds)
            lqr_exact[i, j] = eval_lqr_point(
                cfg, length, mass, args.episodes, clamped=False,
                reset_seeds=reset_seeds)
            lqr_clamped[i, j] = eval_lqr_point(
                cfg, length, mass, args.episodes, clamped=True,
                reset_seeds=reset_seeds)
            done += 1
            print(
                f"[{done:3d}/{total}] L={length:.3f} M={mass:.3f} "
                f"CGAT={cgat[i,j]:7.1f} LQR_exact={lqr_exact[i,j]:7.1f} "
                f"LQR_clamped={lqr_clamped[i,j]:7.1f}"
            )

    os.makedirs("eval/results", exist_ok=True)
    out = f"eval/results/three_link_{args.variant}{suffix}_compare.npz"
    np.savez(
        out,
        lengths=lengths,
        masses=masses,
        cgat=cgat,
        lqr_exact=lqr_exact,
        lqr_clamped=lqr_clamped,
        train_length_bounds=np.array([len_lo, len_hi]),
        train_mass_bounds=np.array([mass_lo, mass_hi]),
    )

    def summary(name, arr):
        return f"{name}: mean={np.mean(arr):.1f} min={np.min(arr):.1f} p10={np.percentile(arr,10):.1f}"

    print("\nSummary")
    print(summary("CGAT       ", cgat))
    print(summary("LQR exact  ", lqr_exact))
    print(summary("LQR clamped", lqr_clamped))
    print(f"CGAT > LQR exact cells   : {100*np.mean(cgat > lqr_exact):.1f}%")
    print(f"CGAT > LQR clamped cells : {100*np.mean(cgat > lqr_clamped):.1f}%")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
