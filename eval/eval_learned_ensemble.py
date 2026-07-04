"""Evaluate an action ensemble of fixed learned CGAT policies.

The ensemble averages deterministic actions from several neural policies. It
does not use LQR gains, oracle labels, or controller recomputation.
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


DEFAULT_MEMBERS = [
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed160_best.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed167_globalscale1.20.pt"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed164_soup.pt"),
    ("action_nonlinear", "checkpoints/cgat_action_nonlinear_ppo_seed162_best.pt"),
]


EXTENDED_MEMBERS = DEFAULT_MEMBERS + [
    ("nonlinear_feedback", "checkpoints/cgat_nonlinear_feedback_ppo_seed146_valbest.pt"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed147_valbest.pt"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed149_valbest.pt"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed154_valbest.pt"),
    ("gain_feedback", "checkpoints/cgat_gain_feedback_ppo_seed159_best.pt"),
    ("velocity_bounded", "checkpoints/cgat_velocity_bounded_ppo_seed157_valbest.pt"),
]


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def obs_to_tensor(obs: dict, device: torch.device) -> dict:
    return {
        key: torch.tensor(
            value,
            dtype=torch.float32 if value.dtype != np.int64 else torch.int64,
        ).unsqueeze(0).to(device)
        for key, value in obs.items()
    }


def obs_link_mass(obs: dict) -> float:
    # Edge features are normalized as (mass - 0.1) / 1.9 in graph_builder.
    edge = obs["edge_features"]
    return float(edge[0, 1]) * 1.9 + 0.1


def obs_link_length(obs: dict) -> float:
    # Edge features are normalized as (length - 0.3) / 0.9 in graph_builder.
    edge = obs["edge_features"]
    return float(edge[0, 0]) * 0.9 + 0.3


def obs_max_abs_angle(obs: dict) -> float:
    node = obs["node_features"]
    joints = node[1:4]
    theta = np.arctan2(joints[:, 3], joints[:, 4])
    return float(np.max(np.abs(theta)))


def compute_value_calibration(
    policies,
    cfg: dict,
    lengths: np.ndarray,
    masses: np.ndarray,
    device: torch.device,
    n_resets: int,
    rollout_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-policy critic scale from learned rollouts, no oracle labels."""
    samples = [[] for _ in policies]
    for length in lengths:
        for mass in masses:
            env = make_fixed_env(cfg, link_length=float(length), link_mass=float(mass), n_links=3)
            try:
                for _ in range(n_resets):
                    obs, _ = env.reset()
                    for step in range(max(1, rollout_steps)):
                        obs_t = obs_to_tensor(obs, device)
                        with torch.no_grad():
                            for idx, policy in enumerate(policies):
                                samples[idx].append(float(policy.get_value(obs_t).squeeze().cpu()))
                        if step + 1 >= rollout_steps:
                            break
                        action = ensemble_action(
                            policies, obs, device, mode="mean", value_stats=None
                        )
                        obs, _, terminated, truncated, _ = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        if terminated or truncated:
                            break
            finally:
                env.close()

    means = np.zeros(len(policies), dtype=np.float64)
    stds = np.ones(len(policies), dtype=np.float64)
    for idx, vals in enumerate(samples):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size:
            means[idx] = float(arr.mean())
            std = float(arr.std())
            stds[idx] = std if std > 1e-6 else 1.0
    return means, stds


def ensemble_action(policies, obs: dict, device: torch.device, mode: str,
                    value_stats: tuple[np.ndarray, np.ndarray] | None = None,
                    value_temperature: float = 1.0,
                    guard_threshold: float = 10.0,
                    switch_mass: float = 1.8,
                    switch_length_lo: float = 0.0,
                    switch_length_hi: float = 10.0,
                    schedule_params: np.ndarray | None = None) -> float:
    actions = np.asarray(
        [policy.get_deterministic_action(obs, device) for policy in policies],
        dtype=np.float64,
    )
    if mode == "tree8_available":
        if len(actions) < 7:
            raise ValueError("tree8_available mode requires seven policies in the documented order")
        mass = obs_link_mass(obs)
        length = obs_link_length(obs)
        if mass < 0.264:
            return float(actions[0])
        if length < 0.164:
            return float(actions[1] if mass < 3.258 else actions[2])
        if mass < 1.975:
            return float(actions[3] if length < 1.531 else actions[4])
        if length < 0.847:
            return float(actions[0])
        return float(actions[5] if length < 1.758 else actions[6])
    if mode == "hybrid_scaled_tree":
        if len(actions) < 7:
            raise ValueError("hybrid_scaled_tree mode requires seven policies in the documented order")
        mass = obs_link_mass(obs)
        length = obs_link_length(obs)
        if mass < 0.692:
            return float(actions[0])
        if mass >= 1.975:
            return float(actions[0] * guard_threshold)
        if length < 0.164:
            return float(actions[1] if mass < 3.258 else actions[2])
        if mass < 1.975:
            return float(actions[3] if length < 1.531 else actions[4])
        if length < 0.847:
            return float(actions[0])
        return float(actions[5] if length < 1.758 else actions[6])
    if mode == "light_tree_scaled":
        if len(actions) < 7:
            raise ValueError("light_tree_scaled mode requires seven policies in the documented order")
        mass = obs_link_mass(obs)
        length = obs_link_length(obs)
        if mass < switch_mass:
            if length < 0.164:
                return float(actions[1] if mass < 3.258 else actions[2])
            if mass < 1.975:
                return float(actions[3] if length < 1.531 else actions[4])
            if length < 0.847:
                return float(actions[0])
            return float(actions[5] if length < 1.758 else actions[6])
        scale = guard_threshold
        return float(actions[0] * scale)
    if mode == "mass_scaled":
        if len(actions) < 1:
            raise ValueError("mass_scaled mode requires one primary policy")
        scale = guard_threshold if obs_link_mass(obs) >= switch_mass else value_temperature
        return float(actions[0] * scale)
    if mode == "mass_scaled3":
        if len(actions) < 1:
            raise ValueError("mass_scaled3 mode requires one primary policy")
        mass = obs_link_mass(obs)
        if mass < switch_mass:
            scale = value_temperature
        elif mass < switch_length_lo:
            scale = switch_length_hi
        else:
            scale = guard_threshold
        return float(actions[0] * scale)
    if mode == "state_mass_scaled":
        if len(actions) < 1:
            raise ValueError("state_mass_scaled mode requires one primary policy")
        mass = obs_link_mass(obs)
        if mass < switch_mass:
            scale = value_temperature
        else:
            scale = guard_threshold
            if obs_max_abs_angle(obs) >= switch_length_lo:
                scale = switch_length_hi
        return float(actions[0] * scale)
    if mode == "affine_scaled":
        if len(actions) < 1:
            raise ValueError("affine_scaled mode requires one primary policy")
        if schedule_params is None or schedule_params.size != 5:
            raise ValueError("affine_scaled mode requires five --schedule_params values")
        length = obs_link_length(obs)
        mass = obs_link_mass(obs)
        length_n = (length - 1.075) / 1.025
        mass_n = (mass - 1.975) / 1.925
        log_scale = (
            schedule_params[0]
            + schedule_params[1] * length_n
            + schedule_params[2] * mass_n
            + schedule_params[3] * length_n * mass_n
        )
        log_scale = np.clip(log_scale, -abs(schedule_params[4]), abs(schedule_params[4]))
        return float(actions[0] * np.exp(log_scale))
    if mode == "mass_switch":
        if len(actions) < 2:
            raise ValueError("mass_switch mode requires low-mass and high-mass policies")
        return float(actions[0] if obs_link_mass(obs) < switch_mass else actions[1])
    if mode == "mass_switch3":
        if len(actions) < 3:
            raise ValueError("mass_switch3 mode requires low/mid/high policies")
        mass = obs_link_mass(obs)
        if mass < switch_mass:
            return float(actions[0])
        if mass < guard_threshold:
            return float(actions[1])
        return float(actions[2])
    if mode == "mass_length_rect":
        if len(actions) < 2:
            raise ValueError("mass_length_rect mode requires primary and fallback policies")
        mass = obs_link_mass(obs)
        length = obs_link_length(obs)
        use_fallback = (
            mass >= switch_mass
            and switch_length_lo <= length <= switch_length_hi
        )
        return float(actions[1] if use_fallback else actions[0])
    if mode == "guarded_diff":
        if len(actions) < 2:
            raise ValueError("guarded_diff mode requires primary and fallback policies")
        primary, fallback = actions[0], actions[1]
        return float(fallback if abs(primary - fallback) > guard_threshold else primary)
    if mode == "median":
        return float(np.median(actions))
    if mode == "maxabs":
        return float(actions[np.abs(actions).argmax()])
    if mode in {"value", "value_z", "value_soft", "value_z_soft"}:
        obs_t = obs_to_tensor(obs, device)
        values = []
        with torch.no_grad():
            for policy in policies:
                values.append(float(policy.get_value(obs_t).squeeze().cpu()))
        values = np.asarray(values, dtype=np.float64)
        if mode in {"value_z", "value_z_soft"}:
            if value_stats is None:
                raise ValueError("value_z mode requires value calibration stats")
            means, stds = value_stats
            values = (values - means) / stds
        if mode in {"value_soft", "value_z_soft"}:
            logits = values / max(float(value_temperature), 1e-6)
            logits = logits - logits.max()
            weights = np.exp(logits)
            weights = weights / weights.sum()
            return float(np.dot(weights, actions))
        return float(actions[int(np.argmax(values))])
    if mode != "mean":
        raise ValueError(f"Unknown ensemble mode: {mode}")
    return float(actions.mean())


def eval_point(policies, cfg: dict, length: float, mass: float,
               n_episodes: int, device: torch.device, mode: str,
               value_stats: tuple[np.ndarray, np.ndarray] | None = None,
               value_temperature: float = 1.0,
               guard_threshold: float = 10.0,
               switch_mass: float = 1.8,
               switch_length_lo: float = 0.0,
               switch_length_hi: float = 10.0,
               schedule_params: np.ndarray | None = None) -> float:
    env_cfg = cfg["environment"]
    env = make_fixed_env(cfg, link_length=length, link_mass=mass, n_links=3)
    rewards = []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                action = ensemble_action(
                    policies, obs, device, mode, value_stats, value_temperature,
                    guard_threshold, switch_mass, switch_length_lo,
                    switch_length_hi, schedule_params,
                )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--n_grid", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=279)
    parser.add_argument(
        "--mode",
        choices=[
            "mean", "median", "maxabs", "value", "value_z", "value_soft",
            "value_z_soft", "guarded_diff", "mass_switch", "mass_switch3",
            "mass_length_rect", "tree8_available", "hybrid_scaled_tree",
            "light_tree_scaled", "mass_scaled", "mass_scaled3",
            "state_mass_scaled", "affine_scaled",
        ],
        default="mean",
    )
    parser.add_argument("--value_temperature", type=float, default=1.0)
    parser.add_argument("--guard_threshold", type=float, default=10.0)
    parser.add_argument("--switch_mass", type=float, default=1.8)
    parser.add_argument("--switch_length_lo", type=float, default=0.0)
    parser.add_argument("--switch_length_hi", type=float, default=10.0)
    parser.add_argument("--schedule_params", type=float, nargs="*", default=None)
    parser.add_argument("--extended_members", action="store_true")
    parser.add_argument("--calib_grid", type=int, default=5)
    parser.add_argument("--calib_resets", type=int, default=2)
    parser.add_argument("--calib_rollout_steps", type=int, default=8)
    parser.add_argument("--member", action="append", default=None,
        help="Custom ensemble member as variant=checkpoint. May be repeated.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policies = []
    members = []
    member_specs = EXTENDED_MEMBERS if args.extended_members else DEFAULT_MEMBERS
    if args.member:
        member_specs = []
        for spec in args.member:
            if "=" not in spec:
                raise ValueError("--member must be formatted as variant=checkpoint")
            variant, checkpoint = spec.split("=", 1)
            member_specs.append((variant, checkpoint))

    for variant, checkpoint in member_specs:
        if not os.path.exists(checkpoint):
            continue
        policies.append(load_policy(checkpoint, cfg, device, variant=variant))
        members.append((variant, checkpoint))
    if not policies:
        raise ValueError("No ensemble members found")

    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    lengths = np.linspace(eval_len_lo, eval_len_hi, args.n_grid)
    masses = np.linspace(eval_mass_lo, eval_mass_hi, args.n_grid)
    rewards = np.zeros((args.n_grid, args.n_grid), dtype=np.float64)
    value_stats = None
    schedule_params = None
    if args.schedule_params is not None:
        schedule_params = np.asarray(args.schedule_params, dtype=np.float64)

    print(f"Device: {device} | mode={args.mode} | members={len(policies)}", flush=True)
    for variant, checkpoint in members:
        print(f"  {variant}: {checkpoint}", flush=True)
    if args.mode in {"value_z", "value_z_soft"}:
        calib_lengths = np.linspace(eval_len_lo, eval_len_hi, args.calib_grid)
        calib_masses = np.linspace(eval_mass_lo, eval_mass_hi, args.calib_grid)
        value_stats = compute_value_calibration(
            policies,
            cfg,
            calib_lengths,
            calib_masses,
            device,
            args.calib_resets,
            args.calib_rollout_steps,
        )
        means, stds = value_stats
        for idx, (mean, std) in enumerate(zip(means, stds)):
            print(f"  value_z[{idx}] mean={mean:.3f} std={std:.3f}", flush=True)

    total = args.n_grid * args.n_grid
    done = 0
    for i, length in enumerate(lengths):
        for j, mass in enumerate(masses):
            reward = eval_point(
                policies, cfg, float(length), float(mass),
                args.n_eval_episodes, device, args.mode, value_stats,
                args.value_temperature, args.guard_threshold, args.switch_mass,
                args.switch_length_lo, args.switch_length_hi, schedule_params,
            )
            rewards[j, i] = reward
            done += 1
            if done % 20 == 0 or done == total:
                print(
                    f"[{done:3d}/{total}] length={length:.3f} mass={mass:.3f} "
                    f"reward={reward:.2f}",
                    flush=True,
                )

    n_vals = np.array([3])
    reward_cube = rewards[None, :, :]
    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/plots", exist_ok=True)
    result_path = f"eval/results/cgat_learned_ensemble_{args.mode}_seed{args.seed}_test3.npz"
    np.savez(
        result_path,
        lengths=lengths,
        masses=masses,
        n_links=n_vals,
        rewards=reward_cube,
        len_bounds=np.array([len_lo, len_hi]),
        mass_bounds=np.array([mass_lo, mass_hi]),
        train_topology=np.array(env_cfg["n_links_range"]),
    )
    plot_topology_heatmaps(
        lengths,
        masses,
        reward_cube,
        n_vals,
        (len_lo, len_hi),
        (mass_lo, mass_hi),
        tuple(env_cfg["n_links_range"]),
        f"Learned Ensemble ({args.mode}) Heatmap",
        f"eval/plots/cgat_learned_ensemble_{args.mode}_seed{args.seed}_ood_heatmaps_by_topology.png",
        max_rewards=topology_reward_ceilings(cfg, n_vals),
    )
    flat = rewards.reshape(-1)
    print(
        f"Saved {result_path} | mean={flat.mean():.2f} "
        f"p25={np.percentile(flat,25):.2f} med={np.median(flat):.2f} "
        f"p90={np.percentile(flat,90):.2f} high2000={(flat > 2000).sum()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
