# Run: python3.12 training/train_cgat.py
#      python3.12 training/train_cgat.py --config configs/default.yaml
#      python3.12 training/train_cgat.py --variant perhead
#      python3.12 training/train_cgat.py --variant gravity --config configs/default.yaml

"""
Training script for the Coupled Graph Attention Transformer (CGAT).

Drop-in replacement for train_ppo.py — identical PPO loop, same environment,
same hyperparameter file.  Key differences:
  • Uses a CGAT variant (base|perhead|directional|gravity|perc).
  • Logs β (physics attention scale) each update.
  • Default config overrides for CGAT (n_heads=2, n_icga_layers=2).

Variants
--------
  base        — scalar β·M̃ per layer  (default)
  perhead     — per-head β·M̃
  directional — β_fwd/β_bwd directional scales
  gravity     — scalar β·M̃ + gravity torque node injection
  perc        — scalar β·M̃ + PERC critic (w_H init=1)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt

from env.pendulum_env import VariablePendulumEnv
from models.cgat import load_cgat_variant, VARIANTS


def set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_suffix(seed: int | None) -> str:
    return "" if seed is None else f"_seed{seed}"


# ── Observation helpers (identical to train_ppo.py) ─────────────────────────

def batch_obs(obs_list: list) -> dict:
    return {
        "node_features": np.stack([o["node_features"] for o in obs_list]),
        "edge_index":    np.stack([o["edge_index"]    for o in obs_list]),
        "edge_features": np.stack([o["edge_features"] for o in obs_list]),
        "n_nodes":       np.stack([o["n_nodes"]       for o in obs_list]),
        "n_edges":       np.stack([o["n_edges"]       for o in obs_list]),
    }


def obs_to_tensor(obs_batch: dict, device: torch.device) -> dict:
    return {
        "node_features": torch.tensor(obs_batch["node_features"], dtype=torch.float32).to(device),
        "edge_index":    torch.tensor(obs_batch["edge_index"],    dtype=torch.int64).to(device),
        "edge_features": torch.tensor(obs_batch["edge_features"], dtype=torch.float32).to(device),
        "n_nodes":       torch.tensor(obs_batch["n_nodes"],       dtype=torch.int64).to(device),
        "n_edges":       torch.tensor(obs_batch["n_edges"],       dtype=torch.int64).to(device),
    }


# ── Rollout Buffer (identical to train_ppo.py) ───────────────────────────────

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, max_nodes, max_edges, gamma, gae_lambda):
        self.rollout_steps = rollout_steps
        self.n_envs        = n_envs
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda

        T, N = rollout_steps, n_envs
        self.node_feat  = np.zeros((T, N, max_nodes, 9),  dtype=np.float32)
        self.edge_index = np.zeros((T, N, 2, max_edges),  dtype=np.int64)
        self.edge_feat  = np.zeros((T, N, max_edges, 2),  dtype=np.float32)
        self.n_nodes    = np.zeros((T, N, 1),              dtype=np.int64)
        self.n_edges    = np.zeros((T, N, 1),              dtype=np.int64)

        self.actions    = np.zeros((T, N), dtype=np.float32)
        self.log_probs  = np.zeros((T, N), dtype=np.float32)
        self.rewards    = np.zeros((T, N), dtype=np.float32)
        self.values     = np.zeros((T, N), dtype=np.float32)
        self.dones      = np.zeros((T, N), dtype=np.float32)
        self.returns    = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.pos = 0

    def store(self, obs_list, actions, log_probs, rewards, values, dones):
        t = self.pos
        for n, obs in enumerate(obs_list):
            self.node_feat[t, n]  = obs["node_features"]
            self.edge_index[t, n] = obs["edge_index"]
            self.edge_feat[t, n]  = obs["edge_features"]
            self.n_nodes[t, n]    = obs["n_nodes"]
            self.n_edges[t, n]    = obs["n_edges"]
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.values[t]    = values
        self.dones[t]     = dones
        self.pos += 1

    def compute_gae(self, last_values):
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_val = last_values if t == self.rollout_steps - 1 else self.values[t + 1]
            delta    = (self.rewards[t]
                        + self.gamma * next_val * (1.0 - self.dones[t])
                        - self.values[t])
            gae      = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def generate_batches(self, batch_size, device):
        T, N  = self.rollout_steps, self.n_envs
        total = T * N
        indices = np.random.permutation(total)

        nf_f  = self.node_feat.reshape(total, *self.node_feat.shape[2:])
        ei_f  = self.edge_index.reshape(total, *self.edge_index.shape[2:])
        ef_f  = self.edge_feat.reshape(total, *self.edge_feat.shape[2:])
        nn_f  = self.n_nodes.reshape(total, 1)
        ne_f  = self.n_edges.reshape(total, 1)
        act_f = self.actions.reshape(total)
        lp_f  = self.log_probs.reshape(total)
        ret_f = self.returns.reshape(total)
        adv_f = self.advantages.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            obs_b = {
                "node_features": torch.tensor(nf_f[idx],  dtype=torch.float32).to(device),
                "edge_index":    torch.tensor(ei_f[idx],  dtype=torch.int64).to(device),
                "edge_features": torch.tensor(ef_f[idx],  dtype=torch.float32).to(device),
                "n_nodes":       torch.tensor(nn_f[idx],  dtype=torch.int64).to(device),
                "n_edges":       torch.tensor(ne_f[idx],  dtype=torch.int64).to(device),
            }
            yield (
                obs_b,
                torch.tensor(act_f[idx], dtype=torch.float32).to(device).unsqueeze(1),
                torch.tensor(lp_f[idx],  dtype=torch.float32).to(device),
                torch.tensor(ret_f[idx], dtype=torch.float32).to(device),
                torch.tensor(adv_f[idx], dtype=torch.float32).to(device),
            )

    def reset(self):
        self.pos = 0


# ── PPO loss (identical to train_ppo.py) ─────────────────────────────────────

def compute_ppo_loss(policy, obs, actions, old_log_probs, returns, advantages,
                     clip_epsilon, value_coef, entropy_coef):
    _, new_log_probs, entropy, values = policy.get_action_and_value(obs, action=actions)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio       = (new_log_probs - old_log_probs).exp()
    surr1       = ratio * advantages
    surr2       = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss   = F.mse_loss(values.squeeze(-1), returns)
    entropy_loss = -entropy.mean()

    total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total, policy_loss.item(), value_loss.item(), (-entropy_loss).item()


# ── Beta extraction helper ────────────────────────────────────────────────────

def _get_beta_val(policy, variant: str) -> float:
    """Extract a representative β scalar for logging (variant-aware)."""
    layer = policy.encoder.icga_layers[0]
    if not hasattr(layer, "physics_beta") and variant != "directional":
        return 0.0
    if variant == "directional":
        b_fwd = float(torch.tanh(layer.physics_beta_fwd).item())
        b_bwd = float(torch.tanh(layer.physics_beta_bwd).item())
        return (b_fwd + b_bwd) / 2.0
    beta = layer.physics_beta
    if beta.numel() > 1:
        return float(torch.tanh(beta).mean().item())
    return float(beta.item())


def _beta_label(policy, variant: str) -> str:
    """Human-readable beta string for the log line."""
    layer = policy.encoder.icga_layers[0]
    if not hasattr(layer, "physics_beta") and variant != "directional":
        return "β n/a"
    if variant == "directional":
        b_fwd = float(torch.tanh(layer.physics_beta_fwd).item())
        b_bwd = float(torch.tanh(layer.physics_beta_bwd).item())
        return f"β_fwd {b_fwd:+.3f} β_bwd {b_bwd:+.3f}"
    beta = layer.physics_beta
    if beta.numel() > 1:
        vals = torch.tanh(beta).tolist()
        mean = sum(vals) / len(vals)
        return f"β̄ {mean:+.3f}"
    return f"β {float(beta.item()):+.3f}"


def _fixed_eval_env(cfg: dict, link_length: float, link_mass: float,
                    reward_config: dict | None = None) -> VariablePendulumEnv:
    env_cfg = cfg["environment"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]
    cart_mass = (cart_lo + cart_hi) / 2.0
    return VariablePendulumEnv(
        n_links_range=(3, 3),
        cart_mass_range=(cart_mass, cart_mass),
        link_length_range=(link_length, link_length),
        link_mass_range=(link_mass, link_mass),
        rail_limit=env_cfg["rail_limit"],
        max_force=env_cfg["max_force"],
        timestep=env_cfg["timestep"],
        frame_skip=env_cfg["frame_skip"],
        max_episode_steps=env_cfg["max_episode_steps"],
        termination_angle=env_cfg["termination_angle"],
        angle_noise=cfg.get("init", {}).get("angle_noise", 0.05),
        vel_noise=cfg.get("init", {}).get("vel_noise", 0.01),
        reward_config=cfg.get("rewards", {}) if reward_config is None else reward_config,
        max_links=env_cfg.get("max_links"),
    )


def deterministic_validation_score(policy, cfg: dict, device: torch.device,
                                   points: list[list[float]],
                                   n_episodes: int,
                                   reward_config: dict | None = None,
                                   aggregate: str = "mean") -> float:
    scores = []
    policy.eval()
    for link_length, link_mass in points:
        env = _fixed_eval_env(
            cfg, float(link_length), float(link_mass),
            reward_config=reward_config,
        )
        try:
            for _ in range(n_episodes):
                obs, _ = env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    action = policy.get_deterministic_action(obs, device)
                    obs, reward, terminated, truncated, _ = env.step(
                        np.array([action], dtype=np.float32)
                    )
                    ep_reward += reward
                    done = terminated or truncated
                scores.append(ep_reward)
        finally:
            env.close()
    if not scores:
        return -np.inf
    scores_arr = np.array(scores, dtype=np.float32)
    if aggregate == "min":
        return float(np.min(scores_arr))
    if aggregate.startswith("p"):
        percentile = float(aggregate[1:])
        return float(np.percentile(scores_arr, percentile))
    if aggregate == "median":
        return float(np.median(scores_arr))
    if aggregate != "mean":
        raise ValueError(f"Unknown validation aggregate: {aggregate}")
    return float(np.mean(scores_arr))


def _eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def heatmap_validation_score(policy, cfg: dict, device: torch.device,
                             n_grid: int,
                             n_episodes: int,
                             reward_config: dict | None = None,
                             aggregate: str = "mean",
                             seed: int | None = None) -> float:
    """Small eval-shaped validation grid for checkpoint selection."""
    env_cfg = cfg["environment"]
    len_lo, len_hi = _eval_range(*env_cfg["link_length_range"])
    mass_lo, mass_hi = _eval_range(*env_cfg["link_mass_range"])
    lengths = np.linspace(len_lo, len_hi, n_grid)
    masses = np.linspace(mass_lo, mass_hi, n_grid)

    scores = []
    policy.eval()
    for i, link_length in enumerate(lengths):
        for j, link_mass in enumerate(masses):
            env = _fixed_eval_env(
                cfg, float(link_length), float(link_mass),
                reward_config=reward_config,
            )
            try:
                for ep in range(n_episodes):
                    reset_seed = None
                    if seed is not None:
                        reset_seed = int(seed + 1009 * i + 9173 * j + ep)
                    obs, _ = env.reset(seed=reset_seed)
                    ep_reward = 0.0
                    done = False
                    while not done:
                        action = policy.get_deterministic_action(obs, device)
                        obs, reward, terminated, truncated, _ = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        ep_reward += reward
                        done = terminated or truncated
                    scores.append(ep_reward)
            finally:
                env.close()

    if not scores:
        return -np.inf
    scores_arr = np.array(scores, dtype=np.float32)
    if aggregate == "min":
        return float(np.min(scores_arr))
    if aggregate.startswith("p"):
        percentile = float(aggregate[1:])
        return float(np.percentile(scores_arr, percentile))
    if aggregate == "median":
        return float(np.median(scores_arr))
    if aggregate != "mean":
        raise ValueError(f"Unknown validation aggregate: {aggregate}")
    return float(np.mean(scores_arr))


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg, variant: str = "base", plot: bool = True, show_plot: bool = True,
          seed: int | None = None, init_checkpoint: str | None = None,
          strict_init: bool = True):
    set_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    h_cfg   = ppo_cfg.get("cgat", {})     # CGAT-specific overrides
    n_envs  = ppo_cfg.get("n_envs", 4)

    print(f"Device: {device}  |  Policy: CGAT-{variant}  |  Parallel envs: {n_envs}"
          f"  |  Seed: {seed if seed is not None else 'none'}")

    def make_env():
        return VariablePendulumEnv(
            n_links_range    = tuple(env_cfg["n_links_range"]),
            cart_mass_range  = tuple(env_cfg["cart_mass_range"]),
            link_length_range= tuple(env_cfg["link_length_range"]),
            link_mass_range  = tuple(env_cfg["link_mass_range"]),
            rail_limit       = env_cfg["rail_limit"],
            max_force        = env_cfg["max_force"],
            timestep         = env_cfg["timestep"],
            frame_skip       = env_cfg["frame_skip"],
            max_episode_steps= env_cfg["max_episode_steps"],
            termination_angle= env_cfg["termination_angle"],
            angle_noise      = cfg.get("init", {}).get("angle_noise", 0.05),
            vel_noise        = cfg.get("init", {}).get("vel_noise", 0.01),
            reward_config    = cfg.get("rewards", {}),
            parameter_regions= env_cfg.get("parameter_regions"),
            max_links        = env_cfg.get("max_links"),
        )

    envs = [make_env() for _ in range(n_envs)]

    max_links    = env_cfg.get("max_links", env_cfg["n_links_range"][1])
    max_nodes    = max_links + 1
    max_edges    = max_links * 2
    max_force    = env_cfg["max_force"]
    max_ep_steps = env_cfg["max_episode_steps"]

    # CGAT hyperparameters (fall back to ppo defaults if not overridden)
    hidden       = h_cfg.get("hidden_dim",    ppo_cfg["hidden_dim"])
    n_icga       = h_cfg.get("n_icga_layers", 2)
    n_heads      = h_cfg.get("n_heads",       2)
    entropy_coef = h_cfg.get("entropy_coef",  ppo_cfg["entropy_coef"])

    policy = load_cgat_variant(
        variant, hidden=hidden, n_icga_layers=n_icga,
        n_heads=n_heads, max_links=max_links, max_force=max_force,
    )
    if init_checkpoint:
        state_dict = torch.load(init_checkpoint, map_location=device)
        missing, unexpected = policy.load_state_dict(state_dict, strict=strict_init)
        print(f"  initialized from learned checkpoint: {init_checkpoint}")
        if not strict_init:
            print(f"  non-strict init: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if "init_log_std" in ppo_cfg:
        with torch.no_grad():
            policy.log_std.fill_(float(ppo_cfg["init_log_std"]))
        print(f"  set initial log_std to {float(policy.log_std.item()):+.3f}")
    if ppo_cfg.get("freeze_log_std", False):
        policy.log_std.requires_grad_(False)
        print("  froze log_std during fine-tune")
    trainable_prefixes = ppo_cfg.get("trainable_prefixes", [])
    if trainable_prefixes:
        trainable_prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
        for name, param in policy.named_parameters():
            param.requires_grad_(name.startswith(trainable_prefixes))
        trainable_count = sum(
            p.numel() for p in policy.parameters() if p.requires_grad
        )
        print(
            "  scoped fine-tune prefixes: "
            + ", ".join(trainable_prefixes)
            + f" ({trainable_count:,} trainable params)"
        )
    policy.to(device)

    print(f"  CGAT-{variant}: hidden={hidden}, n_icga={n_icga}, n_heads={n_heads}, "
          f"entropy_coef={entropy_coef}")
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    lr_init   = ppo_cfg["lr"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_init)
    anneal_lr = ppo_cfg.get("anneal_lr", True)

    rollout_steps  = ppo_cfg["rollout_steps"]
    n_epochs       = ppo_cfg["n_epochs"]
    mini_batch     = ppo_cfg["mini_batch_size"]
    gamma          = ppo_cfg["gamma"]
    gae_lambda     = ppo_cfg["gae_lambda"]
    clip_epsilon   = ppo_cfg["clip_epsilon"]
    value_coef     = ppo_cfg["value_coef"]
    max_grad_norm  = ppo_cfg["max_grad_norm"]
    total_steps    = ppo_cfg["total_steps"]
    action_scale   = float(ppo_cfg.get("action_multiplier", 1.0))
    if action_scale != 1.0:
        print(f"  rollout action multiplier: {action_scale:.3f}")

    buffer = RolloutBuffer(rollout_steps, n_envs, max_nodes, max_edges,
                           gamma, gae_lambda)

    os.makedirs("checkpoints", exist_ok=True)
    best_mean_reward = -np.inf
    best_val_reward = -np.inf
    best_model_path  = f"checkpoints/cgat_{variant}_ppo{seed_suffix(seed)}_best.pt"
    best_val_model_path = f"checkpoints/cgat_{variant}_ppo{seed_suffix(seed)}_valbest.pt"
    legacy_model_path = f"checkpoints/cgat_{variant}_ppo_best.pt" if seed == 0 else None
    val_cfg = cfg.get("validation", {})
    val_points = val_cfg.get("points", [])
    val_every = int(val_cfg.get("interval_updates", 0))
    val_episodes = int(val_cfg.get("n_episodes", 1))
    val_reward_config = val_cfg.get("reward_config")
    val_aggregate = val_cfg.get("aggregate", "mean")
    heatmap_val_cfg = val_cfg.get("heatmap", {})
    heatmap_val_grid = int(heatmap_val_cfg.get("n_grid", 0))
    heatmap_val_episodes = int(heatmap_val_cfg.get("n_episodes", val_episodes))
    heatmap_val_aggregate = heatmap_val_cfg.get("aggregate", val_aggregate)
    heatmap_val_reward_config = heatmap_val_cfg.get("reward_config", val_reward_config)
    heatmap_val_seed = heatmap_val_cfg.get("seed", seed)
    update_idx = 0

    obs_list   = [
        env.reset(seed=None if seed is None else seed + i)[0]
        for i, env in enumerate(envs)
    ]
    ep_rewards = [0.0] * n_envs
    ep_lengths = [0]   * n_envs
    ep_count   = 0
    all_ep_rewards, all_ep_lengths, all_ep_wins = [], [], []

    log_steps, log_mean_reward, log_mean_length, log_survival = [], [], [], []
    log_beta = []   # CGAT-specific diagnostics

    global_step = 0
    t_start     = time.time()

    while global_step < total_steps:
        update_idx += 1

        # LR annealing
        if anneal_lr:
            frac = 1.0 - global_step / total_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * frac

        # ── Rollout collection ────────────────────────────────────────────────
        policy.eval()
        buffer.reset()

        for _ in range(rollout_steps):
            obs_t = obs_to_tensor(batch_obs(obs_list), device)

            with torch.no_grad():
                actions_t, log_probs_t, _, values_t = policy.get_action_and_value(obs_t)

            actions_np   = actions_t.squeeze(-1).cpu().numpy()
            env_actions_np = actions_np * action_scale
            log_probs_np = log_probs_t.cpu().numpy()
            values_np    = values_t.squeeze(-1).cpu().numpy()

            next_obs_list = []
            rewards_np    = np.zeros(n_envs, dtype=np.float32)
            dones_np      = np.zeros(n_envs, dtype=np.float32)

            for n, (env, action) in enumerate(zip(envs, env_actions_np)):
                next_obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32))
                done = terminated or truncated

                rewards_np[n] = reward
                dones_np[n]   = float(done)
                ep_rewards[n] += reward
                ep_lengths[n] += 1

                if done:
                    all_ep_rewards.append(ep_rewards[n])
                    all_ep_lengths.append(ep_lengths[n])
                    all_ep_wins.append(1 if ep_lengths[n] >= max_ep_steps else 0)
                    ep_count      += 1
                    ep_rewards[n]  = 0.0
                    ep_lengths[n]  = 0
                    next_obs, _    = env.reset()

                next_obs_list.append(next_obs)

            buffer.store(obs_list, actions_np, log_probs_np,
                         rewards_np, values_np, dones_np)
            obs_list     = next_obs_list
            global_step += n_envs

        # Bootstrap final values
        obs_t = obs_to_tensor(batch_obs(obs_list), device)
        with torch.no_grad():
            _, _, _, last_v = policy.get_action_and_value(obs_t)
        buffer.compute_gae(last_v.squeeze(-1).cpu().numpy())

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        for _ in range(n_epochs):
            for obs_b, act_b, lp_b, ret_b, adv_b in buffer.generate_batches(
                    mini_batch, device):
                loss, pl, vl, ent = compute_ppo_loss(
                    policy, obs_b, act_b, lp_b, ret_b, adv_b,
                    clip_epsilon, value_coef, entropy_coef)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        window       = 20
        mean_r       = np.mean(all_ep_rewards[-window:]) if all_ep_rewards else 0.0
        mean_len     = np.mean(all_ep_lengths[-window:]) if all_ep_lengths else 0.0
        survival_pct = np.mean(all_ep_wins[-window:]) * 100 if all_ep_wins else 0.0
        elapsed      = time.time() - t_start

        # CGAT diagnostics
        beta_val   = _get_beta_val(policy, variant)
        beta_label = _beta_label(policy, variant)

        log_steps.append(global_step);    log_mean_reward.append(mean_r)
        log_mean_length.append(mean_len); log_survival.append(survival_pct)
        log_beta.append(beta_val)

        if mean_r > best_mean_reward and len(all_ep_rewards) >= 20:
            best_mean_reward = mean_r
            state_dict = policy.state_dict()
            torch.save(state_dict, best_model_path)
            if legacy_model_path is not None:
                torch.save(state_dict, legacy_model_path)
            print(f"  *** new best {mean_r:.2f} → {best_model_path}")

        val_msg = ""
        if val_every > 0 and update_idx % val_every == 0 and (
                val_points or heatmap_val_grid > 1):
            if heatmap_val_grid > 1:
                val_score = heatmap_validation_score(
                    policy, cfg, device, heatmap_val_grid, heatmap_val_episodes,
                    reward_config=heatmap_val_reward_config,
                    aggregate=heatmap_val_aggregate,
                    seed=heatmap_val_seed,
                )
                val_kind = f"heat{heatmap_val_grid}"
            else:
                val_score = deterministic_validation_score(
                    policy, cfg, device, val_points, val_episodes,
                    reward_config=val_reward_config,
                    aggregate=val_aggregate,
                )
                val_kind = "val"
            val_msg = f" | val {val_score:>7.2f}"
            if val_score > best_val_reward:
                best_val_reward = val_score
                torch.save(policy.state_dict(), best_val_model_path)
                print(f"  *** new {val_kind} best {val_score:.2f} → {best_val_model_path}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"step {global_step:>8} | eps {ep_count:>5} "
              f"| reward {mean_r:>7.2f} | ep_len {mean_len:>6.1f} "
              f"| surv {survival_pct:>5.1f}% "
              f"| {beta_label} "
              f"| lr {cur_lr:.2e}{val_msg} | {elapsed:.0f}s")

    for env in envs:
        env.close()
    print("Training complete.")

    if plot:
        _plot_training(log_steps, log_mean_reward, log_mean_length,
                       log_survival, log_beta, variant, show=show_plot,
                       seed=seed)

    return log_steps, log_mean_reward, log_mean_length, log_survival


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_training(steps, rewards, lengths, survival, betas, variant: str = "base",
                   show: bool = True, seed: int | None = None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"CGAT-{variant} PPO Training")

    axes[0].plot(steps, rewards,  color="steelblue")
    axes[0].set_ylabel("Mean Reward (last 20 eps)"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, lengths,  color="seagreen")
    axes[1].set_ylabel("Mean Episode Length");       axes[1].grid(alpha=0.3)

    axes[2].plot(steps, survival, color="tomato")
    axes[2].set_ylabel("Survival Rate %");           axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 105)

    axes[3].plot(steps, betas,    color="darkorange")
    axes[3].set_ylabel("β (physics attn scale)");   axes[3].grid(alpha=0.3)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_xlabel("Training Steps")

    plt.tight_layout()
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/cgat_{variant}_ppo{seed_suffix(seed)}_training_curve.png"
    plt.savefig(path, dpi=150)
    print(f"  plot saved → {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CGAT variant with PPO.")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--variant", default="base",
        choices=list(VARIANTS),
        help="CGAT variant to train (default: base)")
    parser.add_argument("--no-show", action="store_true",
        help="Save training plots without opening a blocking window")
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed. Adds _seedN to checkpoint/plot names.")
    parser.add_argument("--init-checkpoint", default=None,
        help="Initialize from an existing learned CGAT checkpoint before PPO.")
    parser.add_argument("--init-nonstrict", action="store_true",
        help="Allow partial checkpoint initialization for architecture ablations.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(
        cfg,
        variant=args.variant,
        show_plot=not args.no_show,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        strict_init=not args.init_nonstrict,
    )
