# Run: python3.12 training/train_hgfn.py
#      python3.12 training/train_hgfn.py --config configs/default.yaml

"""
Training script for the Hamiltonian Graph Flow Network (HGFN).

Drop-in replacement for train_ppo.py — identical PPO loop, same environment,
same hyperparameter file.  Key differences:
  • Uses HGFNPPOPolicy instead of GNNTransformerPPOPolicy.
  • Logs additional diagnostics: β (physics attention scale), w_H (Hamiltonian
    critic weight), and the mean normalised Hamiltonian value of rollout states.
  • Default config overrides for HGFN (n_heads=2, n_icga_layers=2).
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
from models.hgfn_ppo import HGFNPPOPolicy, compute_hamiltonian


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
        self.node_feat  = np.zeros((T, N, max_nodes, 8),  dtype=np.float32)
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


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg, plot: bool = True):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    h_cfg   = ppo_cfg.get("hgfn", {})     # HGFN-specific overrides
    n_envs  = ppo_cfg.get("n_envs", 4)

    print(f"Device: {device}  |  Policy: HGFN  |  Parallel envs: {n_envs}")

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
        )

    envs = [make_env() for _ in range(n_envs)]

    max_links    = env_cfg["n_links_range"][1]
    max_nodes    = max_links + 1
    max_edges    = max_links * 2
    max_force    = env_cfg["max_force"]
    max_ep_steps = env_cfg["max_episode_steps"]

    # HGFN hyperparameters (fall back to ppo defaults if not overridden)
    hidden       = h_cfg.get("hidden_dim",    ppo_cfg["hidden_dim"])
    n_icga       = h_cfg.get("n_icga_layers", 2)
    n_heads      = h_cfg.get("n_heads",       2)
    entropy_coef = h_cfg.get("entropy_coef",  ppo_cfg["entropy_coef"])

    policy = HGFNPPOPolicy(
        hidden=hidden, n_icga_layers=n_icga,
        n_heads=n_heads, max_links=max_links, max_force=max_force,
    )
    policy.to(device)

    print(f"  HGFN: hidden={hidden}, n_icga={n_icga}, n_heads={n_heads}, "
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

    buffer = RolloutBuffer(rollout_steps, n_envs, max_nodes, max_edges,
                           gamma, gae_lambda)

    os.makedirs("checkpoints", exist_ok=True)
    best_mean_reward = -np.inf
    best_model_path  = "checkpoints/hgfn_ppo_best.pt"

    obs_list   = [env.reset()[0] for env in envs]
    ep_rewards = [0.0] * n_envs
    ep_lengths = [0]   * n_envs
    ep_count   = 0
    all_ep_rewards, all_ep_lengths, all_ep_wins = [], [], []

    log_steps, log_mean_reward, log_mean_length, log_survival = [], [], [], []
    log_beta, log_wH, log_H_mean = [], [], []   # HGFN-specific diagnostics

    global_step = 0
    t_start     = time.time()

    while global_step < total_steps:

        # LR annealing
        if anneal_lr:
            frac = 1.0 - global_step / total_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * frac

        # ── Rollout collection ────────────────────────────────────────────────
        policy.eval()
        buffer.reset()
        rollout_H = []   # track Hamiltonian values during rollout

        for _ in range(rollout_steps):
            obs_t = obs_to_tensor(batch_obs(obs_list), device)

            with torch.no_grad():
                actions_t, log_probs_t, _, values_t = policy.get_action_and_value(obs_t)
                H_vals = compute_hamiltonian(obs_t).cpu().numpy()
                rollout_H.extend(H_vals.tolist())

            actions_np   = actions_t.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np    = values_t.squeeze(-1).cpu().numpy()

            next_obs_list = []
            rewards_np    = np.zeros(n_envs, dtype=np.float32)
            dones_np      = np.zeros(n_envs, dtype=np.float32)

            for n, (env, action) in enumerate(zip(envs, actions_np)):
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

        # HGFN diagnostics
        beta_val  = float(policy.encoder.icga_layers[0].physics_beta.item())
        wH_val    = float(policy.w_H.item())
        H_mean    = float(np.mean(rollout_H)) if rollout_H else 0.0

        log_steps.append(global_step);         log_mean_reward.append(mean_r)
        log_mean_length.append(mean_len);      log_survival.append(survival_pct)
        log_beta.append(beta_val);             log_wH.append(wH_val)
        log_H_mean.append(H_mean)

        if mean_r > best_mean_reward and len(all_ep_rewards) >= 20:
            best_mean_reward = mean_r
            torch.save(policy.state_dict(), best_model_path)
            print(f"  *** new best {mean_r:.2f} → {best_model_path}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"step {global_step:>8} | eps {ep_count:>5} "
              f"| reward {mean_r:>7.2f} | ep_len {mean_len:>6.1f} "
              f"| surv {survival_pct:>5.1f}% "
              f"| β {beta_val:+.3f} | w_H {wH_val:+.3f} | H̄ {H_mean:.2f} "
              f"| lr {cur_lr:.2e} | {elapsed:.0f}s")

    for env in envs:
        env.close()
    print("Training complete.")

    if plot:
        _plot_training(log_steps, log_mean_reward, log_mean_length,
                       log_survival, log_beta, log_wH, log_H_mean)

    return log_steps, log_mean_reward, log_mean_length, log_survival


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_training(steps, rewards, lengths, survival, betas, wHs, H_means):
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("HGFN PPO Training")

    axes[0].plot(steps, rewards,   color="steelblue")
    axes[0].set_ylabel("Mean Reward (last 20 eps)");  axes[0].grid(alpha=0.3)

    axes[1].plot(steps, lengths,   color="seagreen")
    axes[1].set_ylabel("Mean Episode Length");        axes[1].grid(alpha=0.3)

    axes[2].plot(steps, survival,  color="tomato")
    axes[2].set_ylabel("Survival Rate %");            axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 105)

    axes[3].plot(steps, betas,     color="darkorange", label="β (physics attn)")
    axes[3].plot(steps, wHs,       color="purple",     label="w_H (Hamiltonian)")
    axes[3].set_ylabel("Physics Weights");             axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=8)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")

    axes[4].plot(steps, H_means,   color="teal")
    axes[4].set_ylabel("Mean Ĥ (normalised energy)"); axes[4].grid(alpha=0.3)
    axes[4].set_xlabel("Training Steps")

    plt.tight_layout()
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/hgfn_ppo_training_curve.png"
    plt.savefig(path, dpi=150)
    print(f"  plot saved → {path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)