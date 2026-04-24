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
# Observation helpers
# ---------------------------------------------------------------------------

def batch_obs(obs_list: list) -> dict:
    """Stack N per-env obs dicts into a single batched obs dict (B=N)."""
    return {
        "node_features": np.stack([o["node_features"] for o in obs_list]),
        "edge_index":    np.stack([o["edge_index"]    for o in obs_list]),
        "edge_features": np.stack([o["edge_features"] for o in obs_list]),
        "n_nodes":       np.stack([o["n_nodes"]       for o in obs_list]),
        "n_edges":       np.stack([o["n_edges"]       for o in obs_list]),
    }


def obs_to_tensor(obs_batch: dict, device: torch.device) -> dict:
    """Convert batched numpy obs dict to tensors on device."""
    return {
        "node_features": torch.tensor(obs_batch["node_features"], dtype=torch.float32).to(device),
        "edge_index":    torch.tensor(obs_batch["edge_index"],    dtype=torch.int64).to(device),
        "edge_features": torch.tensor(obs_batch["edge_features"], dtype=torch.float32).to(device),
        "n_nodes":       torch.tensor(obs_batch["n_nodes"],       dtype=torch.int64).to(device),
        "n_edges":       torch.tensor(obs_batch["n_edges"],       dtype=torch.int64).to(device),
    }


# ---------------------------------------------------------------------------
# Rollout Buffer  (T × N layout)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    On-policy buffer for N parallel environments.
    Layout: (rollout_steps, n_envs, ...).
    GAE is computed per-env along the time axis.
    Flattened to (T*N, ...) when yielding mini-batches.
    """

    def __init__(self, rollout_steps: int, n_envs: int,
                 max_nodes: int, max_edges: int,
                 gamma: float, gae_lambda: float):
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

    def store(self, obs_list: list, actions: np.ndarray, log_probs: np.ndarray,
              rewards: np.ndarray, values: np.ndarray, dones: np.ndarray):
        """Store one timestep of data from all N envs. obs_list has length N."""
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

    def compute_gae(self, last_values: np.ndarray):
        """
        GAE-λ computed jointly across all N envs.
        last_values: (N,) bootstrapped V(s_{T+1}) from the final state.
        Done flags mask the carry-over between episodes correctly.
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_val  = last_values if t == self.rollout_steps - 1 else self.values[t + 1]
            delta     = (self.rewards[t]
                         + self.gamma * next_val * (1.0 - self.dones[t])
                         - self.values[t])
            gae       = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def generate_batches(self, batch_size: int, device: torch.device):
        """Flatten (T, N) → (T*N,) and yield shuffled mini-batches as tensor dicts."""
        T, N  = self.rollout_steps, self.n_envs
        total = T * N
        indices = np.random.permutation(total)

        # Reshape (T, N, ...) → (T*N, ...)
        node_feat_f  = self.node_feat.reshape(total, *self.node_feat.shape[2:])
        edge_index_f = self.edge_index.reshape(total, *self.edge_index.shape[2:])
        edge_feat_f  = self.edge_feat.reshape(total, *self.edge_feat.shape[2:])
        n_nodes_f    = self.n_nodes.reshape(total, 1)
        n_edges_f    = self.n_edges.reshape(total, 1)
        actions_f    = self.actions.reshape(total)
        log_probs_f  = self.log_probs.reshape(total)
        returns_f    = self.returns.reshape(total)
        adv_f        = self.advantages.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            obs_batch = {
                "node_features": torch.tensor(node_feat_f[idx],  dtype=torch.float32).to(device),
                "edge_index":    torch.tensor(edge_index_f[idx], dtype=torch.int64).to(device),
                "edge_features": torch.tensor(edge_feat_f[idx],  dtype=torch.float32).to(device),
                "n_nodes":       torch.tensor(n_nodes_f[idx],    dtype=torch.int64).to(device),
                "n_edges":       torch.tensor(n_edges_f[idx],    dtype=torch.int64).to(device),
            }
            yield (
                obs_batch,
                torch.tensor(actions_f[idx],   dtype=torch.float32).to(device).unsqueeze(1),
                torch.tensor(log_probs_f[idx], dtype=torch.float32).to(device),
                torch.tensor(returns_f[idx],   dtype=torch.float32).to(device),
                torch.tensor(adv_f[idx],       dtype=torch.float32).to(device),
            )

    def reset(self):
        self.pos = 0


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def compute_ppo_loss(policy, obs, actions, old_log_probs, returns, advantages,
                     clip_epsilon, value_coef, entropy_coef):
    """Clipped surrogate objective + value loss + entropy bonus."""
    _, new_log_probs, entropy, values = policy.get_action_and_value(obs, action=actions)

    # Normalise advantages within mini-batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio       = (new_log_probs - old_log_probs).exp()
    surr1       = ratio * advantages
    surr2       = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss   = F.mse_loss(values.squeeze(-1), returns)
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss.item(), value_loss.item(), (-entropy_loss).item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, policy_name: str, plot: bool = True):
    """
    Train a PPO policy. Returns (log_steps, log_mean_reward, log_mean_length, log_survival)
    so ablation scripts can collect and compare curves without re-running.
    Set plot=False to skip saving/showing the training curve (used by ablation_ppo.py).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    n_envs  = ppo_cfg.get("n_envs", 1)

    print(f"Device: {device}  |  Policy: {policy_name}  |  Parallel envs: {n_envs}")

    # Create N independent environments
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

    # --- build policy ---
    hidden   = ppo_cfg["hidden_dim"]
    n_layers = ppo_cfg["n_layers"]
    n_heads  = ppo_cfg["n_heads"]
    dropout  = ppo_cfg["dropout"]

    # Transformer-specific overrides (fewer heads/layers than MPNN default)
    t_cfg    = ppo_cfg.get("gnn_transformer", {})

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
        print(f"  GNN Transformer: n_heads={t_heads}, n_layers={t_layers}")
    else:
        policy = MLPPPOPolicy(
            hidden=hidden, max_links=max_links,
            dropout=dropout, max_force=max_force)

    policy.to(device)

    lr_init   = ppo_cfg["lr"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_init)
    anneal_lr = ppo_cfg.get("anneal_lr", False)

    rollout_steps     = ppo_cfg["rollout_steps"]
    n_epochs          = ppo_cfg["n_epochs"]
    mini_batch        = ppo_cfg["mini_batch_size"]
    gamma             = ppo_cfg["gamma"]
    gae_lambda        = ppo_cfg["gae_lambda"]
    clip_epsilon      = ppo_cfg["clip_epsilon"]
    value_coef        = ppo_cfg["value_coef"]
    # Transformer uses a slightly higher entropy coef (std collapses faster with attention)
    entropy_coef      = (t_cfg.get("entropy_coef", ppo_cfg["entropy_coef"])
                         if policy_name == "gnn_transformer" else ppo_cfg["entropy_coef"])
    max_grad_norm     = ppo_cfg["max_grad_norm"]
    total_steps       = ppo_cfg["total_steps"]
    buffer = RolloutBuffer(rollout_steps, n_envs, max_nodes, max_edges, gamma, gae_lambda)

    os.makedirs("checkpoints", exist_ok=True)
    best_mean_reward = -np.inf
    best_model_path  = f"checkpoints/{policy_name}_ppo_best.pt"

    # Initialise all envs
    obs_list   = [env.reset()[0] for env in envs]
    ep_rewards = [0.0] * n_envs
    ep_lengths = [0]   * n_envs
    ep_count   = 0
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_wins    = []

    log_steps       = []
    log_mean_reward = []
    log_mean_length = []
    log_survival    = []

    global_step = 0
    t_start     = time.time()

    while global_step < total_steps:

        # LR annealing — linear decay from lr_init → 0
        if anneal_lr:
            frac = 1.0 - global_step / total_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * frac

        # ----------------------------------------------------------------
        # 1. Collect rollout_steps steps from each of the N envs
        # ----------------------------------------------------------------
        policy.eval()
        buffer.reset()

        for _ in range(rollout_steps):
            # Batch obs from all N envs → single policy forward
            obs_t = obs_to_tensor(batch_obs(obs_list), device)

            with torch.no_grad():
                actions_t, log_probs_t, _, values_t = policy.get_action_and_value(obs_t)

            actions_np   = actions_t.squeeze(-1).cpu().numpy()    # (N,)
            log_probs_np = log_probs_t.cpu().numpy()              # (N,)
            values_np    = values_t.squeeze(-1).cpu().numpy()     # (N,)

            # Step each env
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
                    ep_count  += 1
                    ep_rewards[n] = 0.0
                    ep_lengths[n] = 0
                    next_obs, _ = env.reset()

                next_obs_list.append(next_obs)

            buffer.store(obs_list, actions_np, log_probs_np, rewards_np, values_np, dones_np)
            obs_list     = next_obs_list
            global_step += n_envs   # each inner step advances N global steps

        # Bootstrap last values for the current (possibly mid-episode) states
        obs_t = obs_to_tensor(batch_obs(obs_list), device)
        with torch.no_grad():
            _, _, _, last_values_t = policy.get_action_and_value(obs_t)
        buffer.compute_gae(last_values_t.squeeze(-1).cpu().numpy())

        # ----------------------------------------------------------------
        # 2. PPO update — N epochs over the full (T*N) rollout buffer
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
        # 3. Logging — once per update
        # ----------------------------------------------------------------
        window       = 20
        mean_r       = np.mean(all_ep_rewards[-window:]) if all_ep_rewards else 0.0
        mean_len     = np.mean(all_ep_lengths[-window:]) if all_ep_lengths else 0.0
        survival_pct = np.mean(all_ep_wins[-window:]) * 100 if all_ep_wins else 0.0
        elapsed      = time.time() - t_start

        log_steps.append(global_step)
        log_mean_reward.append(mean_r)
        log_mean_length.append(mean_len)
        log_survival.append(survival_pct)

        if mean_r > best_mean_reward and len(all_ep_rewards) >= 20:
            best_mean_reward = mean_r
            torch.save(policy.state_dict(), best_model_path)
            print(f"  *** new best reward {mean_r:.2f} → saved {best_model_path}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"step {global_step:>8} | eps {ep_count:>5} "
              f"| reward {mean_r:>7.2f} | ep_len {mean_len:>6.1f} "
              f"| survival {survival_pct:>5.1f}% | lr {cur_lr:.2e} | {elapsed:.0f}s")

    for env in envs:
        env.close()
    print("Training complete.")
    if plot:
        _plot_training(log_steps, log_mean_reward, log_mean_length, log_survival, policy_name)
    return log_steps, log_mean_reward, log_mean_length, log_survival


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
