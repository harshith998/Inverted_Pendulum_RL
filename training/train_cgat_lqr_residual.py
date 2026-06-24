# Run: python3.12 training/train_cgat_lqr_residual.py --config configs/cgat_3link.yaml --seed 50

"""Train CGAT as a residual policy on top of exact 3-link LQR.

The environment action is:
    u = clip(u_lqr(config, state) + u_residual_cgat(obs), -max_force, max_force)

PPO is applied to the residual action distribution. This keeps the controller
near a stabilizing classical baseline while allowing the network to learn
nonlinear/reward-specific corrections.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from env.pendulum_env import VariablePendulumEnv
from eval.eval_lqr import compute_lqr_gain, extract_state
from models.cgat import VARIANTS, load_cgat_variant
from training.train_cgat import RolloutBuffer, batch_obs, obs_to_tensor, seed_suffix, set_seed


def make_env(cfg):
    env_cfg = cfg["environment"]
    fixed_pairs = env_cfg.get("fixed_length_mass_pairs")
    options = None
    if fixed_pairs:
        pair = fixed_pairs[np.random.randint(0, len(fixed_pairs))]
        options = {
            "link_length": float(pair[0]),
            "link_mass": float(pair[1]),
        }
    return VariablePendulumEnv(
        n_links_range=tuple(env_cfg["n_links_range"]),
        cart_mass_range=tuple(env_cfg["cart_mass_range"]),
        link_length_range=(options["link_length"], options["link_length"]) if options else tuple(env_cfg["link_length_range"]),
        link_mass_range=(options["link_mass"], options["link_mass"]) if options else tuple(env_cfg["link_mass_range"]),
        rail_limit=env_cfg["rail_limit"],
        max_force=env_cfg["max_force"],
        timestep=env_cfg["timestep"],
        frame_skip=env_cfg["frame_skip"],
        max_episode_steps=env_cfg["max_episode_steps"],
        termination_angle=env_cfg["termination_angle"],
        max_links=env_cfg.get("max_links"),
    )


def config_key(config):
    return (
        tuple(round(float(x), 2) for x in config.lengths),
        tuple(round(float(x), 2) for x in config.masses),
        round(float(config.cart_mass), 2),
        int(config.n_links),
    )


def make_qr(n_links, costs):
    cart_q, angle_q, vel_q, r = costs
    n_q = n_links + 1
    state_dim = 2 * n_q
    Q = np.zeros((state_dim, state_dim))
    Q[0, 0] = cart_q
    for j in range(1, n_q):
        Q[j, j] = angle_q
    for j in range(n_q, state_dim):
        Q[j, j] = vel_q
    return Q, np.array([[r]], dtype=float)


def get_lqr_gain(env, cfg, cache, lqr_costs=None):
    config = env._config
    key = (config_key(config), tuple(lqr_costs) if lqr_costs is not None else None)
    if key not in cache:
        env_cfg = cfg["environment"]
        if lqr_costs is None:
            cache[key] = compute_lqr_gain(
                config.lengths,
                config.masses,
                config.cart_mass,
                config.n_links,
                rail_limit=env_cfg["rail_limit"],
                max_force=env_cfg["max_force"],
                timestep=env_cfg["timestep"],
            )
        else:
            import scipy.linalg
            from eval.eval_lqr import _linearize_mujoco

            A, B = _linearize_mujoco(
                config.lengths,
                config.masses,
                config.cart_mass,
                config.n_links,
                env_cfg["rail_limit"],
                env_cfg["max_force"],
                env_cfg["timestep"],
            )
            Q, R = make_qr(config.n_links, lqr_costs)
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            cache[key] = np.linalg.solve(R, B.T @ P)
    return cache[key]


def lqr_action(obs, k, max_force):
    state = extract_state(obs, 3)
    u = float(-(k @ state)[0])
    return float(np.clip(u, -max_force, max_force))


def residual_ppo_loss(policy, obs, residual_actions, old_log_probs, returns, advantages,
                      clip_epsilon, value_coef, entropy_coef):
    _, new_log_probs, entropy, values = policy.get_action_and_value(obs, action=residual_actions)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values.squeeze(-1), returns)
    entropy_loss = -entropy.mean()
    return policy_loss + value_coef * value_loss + entropy_coef * entropy_loss


def zero_actor(policy):
    torch.nn.init.zeros_(policy.mean_head.weight)
    torch.nn.init.zeros_(policy.mean_head.bias)
    policy.log_std.data.fill_(-1.5)


def train(cfg, variant, seed, residual_scale, total_steps_override=None,
          init_checkpoint=None, lr_override=None, lqr_costs=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    h_cfg = ppo_cfg.get("cgat", {})

    n_envs = ppo_cfg.get("n_envs", 4)
    envs = [make_env(cfg) for _ in range(n_envs)]
    max_force = env_cfg["max_force"]
    max_links = env_cfg.get("max_links", 3)
    max_nodes = max_links + 1
    max_edges = max_links * 2

    policy = load_cgat_variant(
        variant,
        hidden=h_cfg.get("hidden_dim", ppo_cfg["hidden_dim"]),
        n_icga_layers=h_cfg.get("n_icga_layers", 2),
        n_heads=h_cfg.get("n_heads", 2),
        max_links=max_links,
        max_force=residual_scale,
    ).to(device)
    if init_checkpoint:
        policy.load_state_dict(torch.load(init_checkpoint, map_location=device), strict=False)
        print(f"Loaded init checkpoint: {init_checkpoint}")
    else:
        zero_actor(policy)

    lr = lr_override if lr_override is not None else ppo_cfg["lr"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rollout_steps = ppo_cfg["rollout_steps"]
    n_epochs = ppo_cfg["n_epochs"]
    mini_batch = ppo_cfg["mini_batch_size"]
    total_steps = total_steps_override or ppo_cfg["total_steps"]
    entropy_coef = h_cfg.get("entropy_coef", ppo_cfg["entropy_coef"])

    buffer = RolloutBuffer(
        rollout_steps,
        n_envs,
        max_nodes,
        max_edges,
        ppo_cfg["gamma"],
        ppo_cfg["gae_lambda"],
    )

    k_cache = {}
    obs_list = []
    k_list = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=None if seed is None else seed + i)
        obs_list.append(obs)
        k_list.append(get_lqr_gain(env, cfg, k_cache, lqr_costs=lqr_costs))

    ep_rewards = [0.0] * n_envs
    ep_lengths = [0] * n_envs
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_wins = []
    best_mean_reward = -np.inf
    global_step = 0
    t0 = time.time()

    os.makedirs("checkpoints", exist_ok=True)
    best_path = f"checkpoints/cgat_{variant}_lqr_residual{seed_suffix(seed)}_best.pt"

    print(f"Device: {device} | residual_scale={residual_scale} | n_envs={n_envs}")
    while global_step < total_steps:
        if ppo_cfg.get("anneal_lr", True):
            frac = 1.0 - global_step / total_steps
            for pg in optimizer.param_groups:
                pg["lr"] = ppo_cfg["lr"] * frac
                if lr_override is not None:
                    pg["lr"] = lr_override * frac

        policy.eval()
        buffer.reset()
        for _ in range(rollout_steps):
            obs_t = obs_to_tensor(batch_obs(obs_list), device)
            with torch.no_grad():
                residual_t, log_probs_t, _, values_t = policy.get_action_and_value(obs_t)
            residual_np = residual_t.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.squeeze(-1).cpu().numpy()

            next_obs_list = []
            rewards_np = np.zeros(n_envs, dtype=np.float32)
            dones_np = np.zeros(n_envs, dtype=np.float32)

            for n, env in enumerate(envs):
                base_u = lqr_action(obs_list[n], k_list[n], max_force)
                action = float(np.clip(base_u + residual_np[n], -max_force, max_force))
                next_obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32)
                )
                done = terminated or truncated
                rewards_np[n] = reward
                dones_np[n] = float(done)
                ep_rewards[n] += reward
                ep_lengths[n] += 1

                if done:
                    all_ep_rewards.append(ep_rewards[n])
                    all_ep_lengths.append(ep_lengths[n])
                    all_ep_wins.append(1 if ep_lengths[n] >= env_cfg["max_episode_steps"] else 0)
                    ep_rewards[n] = 0.0
                    ep_lengths[n] = 0
                    next_obs, _ = env.reset()
                    k_list[n] = get_lqr_gain(env, cfg, k_cache, lqr_costs=lqr_costs)

                next_obs_list.append(next_obs)

            buffer.store(obs_list, residual_np, log_probs_np, rewards_np, values_np, dones_np)
            obs_list = next_obs_list
            global_step += n_envs

        obs_t = obs_to_tensor(batch_obs(obs_list), device)
        with torch.no_grad():
            _, _, _, last_v = policy.get_action_and_value(obs_t)
        buffer.compute_gae(last_v.squeeze(-1).cpu().numpy())

        policy.train()
        for _ in range(n_epochs):
            for obs_b, act_b, lp_b, ret_b, adv_b in buffer.generate_batches(mini_batch, device):
                loss = residual_ppo_loss(
                    policy,
                    obs_b,
                    act_b,
                    lp_b,
                    ret_b,
                    adv_b,
                    ppo_cfg["clip_epsilon"],
                    ppo_cfg["value_coef"],
                    entropy_coef,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg["max_grad_norm"])
                optimizer.step()

        mean_r = float(np.mean(all_ep_rewards[-20:])) if all_ep_rewards else 0.0
        mean_len = float(np.mean(all_ep_lengths[-20:])) if all_ep_lengths else 0.0
        surv = float(np.mean(all_ep_wins[-20:]) * 100.0) if all_ep_wins else 0.0
        if mean_r > best_mean_reward and len(all_ep_rewards) >= 20:
            best_mean_reward = mean_r
            torch.save(policy.state_dict(), best_path)
            print(f"  *** new best {mean_r:.2f} -> {best_path}")
        print(
            f"step {global_step:>8} | eps {len(all_ep_rewards):>5} | reward {mean_r:>7.2f} "
            f"| ep_len {mean_len:>6.1f} | surv {surv:>5.1f}% | cache {len(k_cache):>4} "
            f"| {time.time() - t0:.0f}s"
        )

    for env in envs:
        env.close()
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train CGAT residual over exact LQR.")
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="base", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--residual_scale", type=float, default=4.0)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lqr_costs", nargs=4, type=float, default=None,
        metavar=("CART_Q", "ANGLE_Q", "VEL_Q", "R"))
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(
        cfg,
        args.variant,
        args.seed,
        args.residual_scale,
        args.total_steps,
        init_checkpoint=args.init_checkpoint,
        lr_override=args.lr,
        lqr_costs=args.lqr_costs,
    )


if __name__ == "__main__":
    main()
