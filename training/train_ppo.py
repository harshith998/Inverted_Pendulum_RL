# Run: python3.12 training/train_ppo.py --policy gnn_mpnn
#      python3.12 training/train_ppo.py --policy gnn_transformer
#      python3.12 training/train_ppo.py --policy mlp

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
from models.gnn_mpnn_ppo import GNNMPNNPPOPolicy
from models.gnn_transformer_ppo import GNNTransformerPPOPolicy
from models.mlp_ppo import MLPPPOPolicy


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    On-policy buffer: collect exactly `rollout_steps` transitions,
    compute GAE returns/advantages, then yield mini-batches for N epochs.
    Wiped after each update — pure on-policy.
    """

    def __init__(self, rollout_steps: int, max_nodes: int, max_edges: int,
                 gamma: float, gae_lambda: float):
        self.rollout_steps = rollout_steps
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda

        # Observations (pre-allocated)
        self.node_feat  = np.zeros((rollout_steps, max_nodes, 8),  dtype=np.float32)
        self.edge_index = np.zeros((rollout_steps, 2, max_edges),  dtype=np.int64)
        self.edge_feat  = np.zeros((rollout_steps, max_edges, 2),  dtype=np.float32)
        self.n_nodes    = np.zeros((rollout_steps, 1),              dtype=np.int64)
        self.n_edges    = np.zeros((rollout_steps, 1),              dtype=np.int64)

        self.actions    = np.zeros(rollout_steps, dtype=np.float32)
        self.log_probs  = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards    = np.zeros(rollout_steps, dtype=np.float32)
        self.values     = np.zeros(rollout_steps, dtype=np.float32)
        self.dones      = np.zeros(rollout_steps, dtype=np.float32)

        # Computed after collection
        self.returns    = np.zeros(rollout_steps, dtype=np.float32)
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)

        self.pos = 0

    def store(self, obs, action, log_prob, reward, value, done):
        i = self.pos
        self.node_feat[i]  = obs["node_features"]
        self.edge_index[i] = obs["edge_index"]
        self.edge_feat[i]  = obs["edge_features"]
        self.n_nodes[i]    = obs["n_nodes"]
        self.n_edges[i]    = obs["n_edges"]
        self.actions[i]    = action
        self.log_probs[i]  = log_prob
        self.rewards[i]    = reward
        self.values[i]     = value
        self.dones[i]      = float(done)
        self.pos += 1

    def compute_gae(self, last_value: float):
        """
        Generalised Advantage Estimation (GAE-λ).
        last_value: V(s_{T+1}) bootstrapped from the policy after rollout ends.
        """
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            next_val   = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            next_done  = self.dones[t]
            delta      = (self.rewards[t]
                          + self.gamma * next_val * (1.0 - next_done)
                          - self.values[t])
            gae        = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def generate_batches(self, batch_size: int, device: torch.device):
        """Yield shuffled mini-batches as dicts of tensors."""
        indices = np.random.permutation(self.rollout_steps)
        for start in range(0, self.rollout_steps, batch_size):
            idx = indices[start:start + batch_size]

            obs_batch = {
                "node_features": torch.tensor(self.node_feat[idx],  dtype=torch.float32).to(device),
                "edge_index":    torch.tensor(self.edge_index[idx],  dtype=torch.int64).to(device),
                "edge_features": torch.tensor(self.edge_feat[idx],  dtype=torch.float32).to(device),
                "n_nodes":       torch.tensor(self.n_nodes[idx],     dtype=torch.int64).to(device),
                "n_edges":       torch.tensor(self.n_edges[idx],     dtype=torch.int64).to(device),
            }
            yield (
                obs_batch,
                torch.tensor(self.actions[idx],    dtype=torch.float32).to(device).unsqueeze(1),
                torch.tensor(self.log_probs[idx],  dtype=torch.float32).to(device),
                torch.tensor(self.returns[idx],    dtype=torch.float32).to(device),
                torch.tensor(self.advantages[idx], dtype=torch.float32).to(device),
            )

    def reset(self):
        self.pos = 0


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def compute_ppo_loss(policy, obs, actions, old_log_probs, returns, advantages,
                     clip_epsilon, value_coef, entropy_coef):
    """
    Clipped surrogate objective + value loss + entropy bonus.

    Returns total loss and diagnostic scalars.
    """
    _, new_log_probs, entropy, values = policy.get_action_and_value(obs, action=actions)

    # Normalise advantages (within mini-batch)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy (surrogate) loss
    ratio       = (new_log_probs - old_log_probs).exp()
    surr1       = ratio * advantages
    surr2       = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (MSE between predicted and GAE returns)
    value_loss  = F.mse_loss(values.squeeze(-1), returns)

    # Entropy bonus (encourages exploration)
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    return total_loss, policy_loss.item(), value_loss.item(), (-entropy_loss).item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, policy_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Policy: {policy_name}")

    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]

    env = VariablePendulumEnv(
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

    max_links = env_cfg["n_links_range"][1]
    max_nodes = max_links + 1
    max_edges = max_links * 2
    max_force = env_cfg["max_force"]

    # --- build policy ---
    hidden   = ppo_cfg["hidden_dim"]
    n_layers = ppo_cfg["n_layers"]
    n_heads  = ppo_cfg["n_heads"]
    dropout  = ppo_cfg["dropout"]

    if policy_name == "gnn_mpnn":
        policy = GNNMPNNPPOPolicy(
            hidden=hidden, n_layers=n_layers, max_links=max_links,
            dropout=dropout, max_force=max_force)
    elif policy_name == "gnn_transformer":
        policy = GNNTransformerPPOPolicy(
            hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            max_links=max_links, dropout=dropout, max_force=max_force)
    else:
        policy = MLPPPOPolicy(
            hidden=hidden, max_links=max_links,
            dropout=dropout, max_force=max_force)

    policy.to(device)

    lr_init = ppo_cfg["lr"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_init)
    anneal_lr = ppo_cfg.get("anneal_lr", False)

    rollout_steps  = ppo_cfg["rollout_steps"]
    n_epochs       = ppo_cfg["n_epochs"]
    mini_batch     = ppo_cfg["mini_batch_size"]
    gamma          = ppo_cfg["gamma"]
    gae_lambda     = ppo_cfg["gae_lambda"]
    clip_epsilon   = ppo_cfg["clip_epsilon"]
    value_coef     = ppo_cfg["value_coef"]
    entropy_coef   = ppo_cfg["entropy_coef"]
    max_grad_norm  = ppo_cfg["max_grad_norm"]
    total_steps    = ppo_cfg["total_steps"]
    log_interval   = ppo_cfg["log_interval"]
    save_interval  = ppo_cfg["save_interval"]
    max_ep_steps   = env_cfg["max_episode_steps"]

    buffer = RolloutBuffer(rollout_steps, max_nodes, max_edges, gamma, gae_lambda)

    os.makedirs("checkpoints", exist_ok=True)
    best_mean_reward = -np.inf
    best_model_path  = f"checkpoints/{policy_name}_ppo_best.pt"

    obs, _     = env.reset()
    ep_reward  = 0.0
    ep_length  = 0
    ep_count   = 0
    ep_rewards = []
    ep_lengths = []
    ep_wins    = []

    log_steps       = []
    log_mean_reward = []
    log_mean_length = []
    log_survival    = []

    global_step = 0
    t_start     = time.time()

    while global_step < total_steps:

        # ----------------------------------------------------------------
        # 0. LR annealing — linear decay from lr_init → 0
        # ----------------------------------------------------------------
        if anneal_lr:
            frac = 1.0 - global_step / total_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_init * frac

        # ----------------------------------------------------------------
        # 1. Collect rollout
        # ----------------------------------------------------------------
        policy.eval()
        buffer.reset()

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs, device)

            next_obs, reward, terminated, truncated, _ = env.step(
                np.array([action], dtype=np.float32))
            done = terminated or truncated

            buffer.store(obs, action, log_prob, reward, value, done)
            ep_reward += reward
            ep_length += 1
            obs        = next_obs
            global_step += 1

            if done:
                obs, _  = env.reset()
                ep_count += 1
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_wins.append(1 if ep_length >= max_ep_steps else 0)
                ep_reward = 0.0
                ep_length = 0

        # Bootstrap value for the last state (handles mid-episode rollout end)
        with torch.no_grad():
            _, _, last_value = policy.get_action(obs, device)
        buffer.compute_gae(last_value)

        # ----------------------------------------------------------------
        # 2. PPO update — N epochs over the rollout buffer
        # ----------------------------------------------------------------
        policy.train()
        for _ in range(n_epochs):
            for obs_b, act_b, lp_b, ret_b, adv_b in buffer.generate_batches(mini_batch, device):
                loss, pl, vl, ent = compute_ppo_loss(
                    policy, obs_b, act_b, lp_b, ret_b, adv_b,
                    clip_epsilon, value_coef, entropy_coef)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # ----------------------------------------------------------------
        # 3. Logging
        # ----------------------------------------------------------------
        if global_step % log_interval < rollout_steps:
            window       = 20
            mean_r       = np.mean(ep_rewards[-window:]) if ep_rewards else 0.0
            mean_len     = np.mean(ep_lengths[-window:]) if ep_lengths else 0.0
            survival_pct = np.mean(ep_wins[-window:]) * 100 if ep_wins else 0.0
            elapsed      = time.time() - t_start

            log_steps.append(global_step)
            log_mean_reward.append(mean_r)
            log_mean_length.append(mean_len)
            log_survival.append(survival_pct)

            if mean_r > best_mean_reward and len(ep_rewards) >= 20:
                best_mean_reward = mean_r
                torch.save(policy.state_dict(), best_model_path)
                print(f"  *** new best reward {mean_r:.2f} → saved {best_model_path}")

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"step {global_step:>8} | episodes {ep_count:>5} "
                  f"| reward {mean_r:>7.2f} | ep_len {mean_len:>6.1f} "
                  f"| survival {survival_pct:>5.1f}% | lr {cur_lr:.2e} | {elapsed:.0f}s")

    env.close()
    print("Training complete.")
    _plot_training(log_steps, log_mean_reward, log_mean_length, log_survival, policy_name)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_training(steps, rewards, lengths, survival, policy_name):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"PPO Training — {policy_name.upper()} policy")

    axes[0].plot(steps, rewards, color="steelblue")
    axes[0].set_ylabel("Mean Reward (last 20 eps)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, lengths, color="seagreen")
    axes[1].set_ylabel("Mean Episode Length (steps)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, survival, color="tomato")
    axes[2].set_ylabel("Survival Rate % (last 20 eps)")
    axes[2].set_xlabel("Training Steps")
    axes[2].set_ylim(0, 105)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{policy_name}_ppo_training_curve.png"
    plt.savefig(path, dpi=150)
    print(f"  plot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        choices=["gnn_mpnn", "gnn_transformer", "mlp"],
                        default="gnn_mpnn")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.policy)
