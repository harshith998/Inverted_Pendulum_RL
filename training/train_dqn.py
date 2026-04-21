# Run: python3.12 training/train_dqn.py --policy gnn
#      python3.12 training/train_dqn.py --policy mlp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt

from env.pendulum_env import VariablePendulumEnv
from models.gnn_dqn import GNNDQNPolicy
from models.mlp_dqn import MLPDQNPolicy


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    #Fixed-size circular buffer storing (obs, action, reward, next_obs, done).

    def __init__(self, capacity, max_nodes, max_edges):
        self.capacity  = capacity
        self.pos       = 0
        self.size      = 0
        max_n = max_nodes
        max_e = max_edges

        # pre-allocate all arrays up front for speed
        self.node_feat      = np.zeros((capacity, max_n, 8),  dtype=np.float32)
        self.edge_index     = np.zeros((capacity, 2, max_e),  dtype=np.int64)
        self.edge_feat      = np.zeros((capacity, max_e, 2),  dtype=np.float32)
        self.n_nodes        = np.zeros((capacity, 1),          dtype=np.int64)
        self.n_edges        = np.zeros((capacity, 1),          dtype=np.int64)

        self.next_node_feat = np.zeros((capacity, max_n, 8),  dtype=np.float32)
        self.next_edge_index= np.zeros((capacity, 2, max_e),  dtype=np.int64)
        self.next_edge_feat = np.zeros((capacity, max_e, 2),  dtype=np.float32)
        self.next_n_nodes   = np.zeros((capacity, 1),          dtype=np.int64)
        self.next_n_edges   = np.zeros((capacity, 1),          dtype=np.int64)

        self.actions        = np.zeros(capacity, dtype=np.int64)
        self.rewards        = np.zeros(capacity, dtype=np.float32)
        self.dones          = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.node_feat[i]       = obs["node_features"]
        self.edge_index[i]      = obs["edge_index"]
        self.edge_feat[i]       = obs["edge_features"]
        self.n_nodes[i]         = obs["n_nodes"]
        self.n_edges[i]         = obs["n_edges"]

        self.next_node_feat[i]  = next_obs["node_features"]
        self.next_edge_index[i] = next_obs["edge_index"]
        self.next_edge_feat[i]  = next_obs["edge_features"]
        self.next_n_nodes[i]    = next_obs["n_nodes"]
        self.next_n_edges[i]    = next_obs["n_edges"]

        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i]   = float(done)

        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        def t(arr, dtype=torch.float32):
            return torch.tensor(arr[idx], dtype=dtype).to(device)

        obs = {
            "node_features": t(self.node_feat),
            "edge_index":    t(self.edge_index, torch.int64),
            "edge_features": t(self.edge_feat),
            "n_nodes":       t(self.n_nodes, torch.int64),
            "n_edges":       t(self.n_edges, torch.int64),
        }
        next_obs = {
            "node_features": t(self.next_node_feat),
            "edge_index":    t(self.next_edge_index, torch.int64),
            "edge_features": t(self.next_edge_feat),
            "n_nodes":       t(self.next_n_nodes, torch.int64),
            "n_edges":       t(self.next_n_edges, torch.int64),
        }
        actions = t(self.actions, torch.int64)
        rewards = t(self.rewards)
        dones   = t(self.dones)
        return obs, actions, rewards, next_obs, dones


# ---------------------------------------------------------------------------
# TD loss
# ---------------------------------------------------------------------------

def compute_td_loss(policy, target, obs, actions, rewards, next_obs, dones, gamma, device,
                    aux_weight=0.1):
    """Bellman loss + optional auxiliary param prediction loss (GNN only)."""

    # current Q for taken actions
    q_all    = policy.get_q_values(obs)                          # (B, n_bins)
    q_taken  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # target Q via double-DQN style: argmax from policy, value from target
    with torch.no_grad():
        next_actions = policy.get_q_values(next_obs).argmax(1)   # (B,)
        next_q       = target.get_q_values(next_obs)             # (B, n_bins)
        next_q_taken = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q     = rewards + gamma * next_q_taken * (1.0 - dones)

    td_loss = F.smooth_l1_loss(q_taken, target_q)

    # auxiliary param prediction (GNN only — MLP has no forward() returning aux)
    # Use policy.forward() so encoder runs once for both Q and aux (not twice).
    aux_loss = torch.tensor(0.0, device=device)
    if hasattr(policy, "max_links") and callable(getattr(policy, "forward", None)):
        _, pred = policy(obs)                            # (B, n_bins), (B, max_links * 2)
        # build ground-truth from edge_features: edges 0,2,4,... are forward edges → [length, mass]
        # edge_features shape (B, max_edges, 2), forward edges at even indices
        ef   = obs["edge_features"]                      # (B, max_edges, 2)
        n_e  = obs["n_edges"]                            # (B, 1)
        # pick forward edges (0, 2, 4, ...) up to n_links = n_edges // 2
        max_links = policy.max_links
        gt = ef[:, 0:max_links*2:2, :].reshape(ef.shape[0], -1)  # (B, max_links * 2)
        # only penalise real rods
        n_links = (n_e.float() / 2).long()               # (B, 1)
        rod_mask = (torch.arange(max_links, device=device).unsqueeze(0)
                    < n_links).float()                   # (B, max_links)
        rod_mask = rod_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(ef.shape[0], -1)
        aux_loss = (F.mse_loss(pred * rod_mask, gt * rod_mask, reduction="sum")
                    / rod_mask.sum().clamp(min=1))

    return td_loss + aux_weight * aux_loss, td_loss.item(), aux_loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, policy_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Policy: {policy_name}")

    # --- env ---
    env_cfg = cfg["environment"]
    env = VariablePendulumEnv(
        n_links_range   = tuple(env_cfg["n_links_range"]),
        cart_mass_range = tuple(env_cfg["cart_mass_range"]),
        link_length_range=tuple(env_cfg["link_length_range"]),
        link_mass_range = tuple(env_cfg["link_mass_range"]),
        rail_limit      = env_cfg["rail_limit"],
        max_force       = env_cfg["max_force"],
        timestep        = env_cfg["timestep"],
        frame_skip      = env_cfg["frame_skip"],
        max_episode_steps=env_cfg["max_episode_steps"],
        termination_angle=env_cfg["termination_angle"],
    )

    max_links = env_cfg["n_links_range"][1]
    max_nodes = max_links + 1
    max_edges = max_links * 2

    # --- action bins: 51 evenly spaced forces from -max_force to +max_force ---
    dqn_cfg  = cfg["dqn"]
    n_bins   = dqn_cfg["n_action_bins"]
    act_bins = np.linspace(-env_cfg["max_force"], env_cfg["max_force"], n_bins)

    # --- policy + target ---
    hidden    = dqn_cfg["hidden_dim"]
    n_layers  = dqn_cfg["n_mp_layers"]

    if policy_name == "gnn":
        policy = GNNDQNPolicy(n_bins, hidden=hidden, n_mp_layers=n_layers, max_links=max_links)
    else:
        policy = MLPDQNPolicy(n_bins, max_links=max_links, hidden=hidden)

    policy.to(device)
    target = copy.deepcopy(policy).to(device)
    target.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=dqn_cfg["lr"],
                                 weight_decay=dqn_cfg.get("weight_decay", 1e-4))
    buffer    = ReplayBuffer(dqn_cfg["replay_capacity"], max_nodes, max_edges)

    # --- bookkeeping ---
    total_steps     = dqn_cfg["total_steps"]
    warmup          = dqn_cfg["warmup_steps"]
    update_freq     = dqn_cfg["update_freq"]
    target_freq     = dqn_cfg["target_update_freq"]
    batch_size      = dqn_cfg["batch_size"]
    gamma           = dqn_cfg["gamma"]
    eps_start       = dqn_cfg["eps_start"]
    eps_end         = dqn_cfg["eps_end"]
    eps_decay       = dqn_cfg["eps_decay_steps"]
    log_interval    = dqn_cfg["log_interval"]
    save_interval   = dqn_cfg["save_interval"]

    os.makedirs("checkpoints", exist_ok=True)
    best_mean_reward = -np.inf
    best_model_path  = f"checkpoints/{policy_name}_dqn_best.pt"

    max_ep_steps = env_cfg["max_episode_steps"]
    obs, _       = env.reset()
    ep_reward    = 0.0
    ep_length    = 0
    ep_count     = 0
    ep_rewards   = []   # reward per episode
    ep_lengths   = []   # steps per episode
    ep_wins      = []   # 1 if survived max_episode_steps, else 0

    # logged every log_interval steps for smooth curves
    log_steps        = []
    log_mean_reward  = []
    log_mean_length  = []
    log_survival     = []

    t_start      = time.time()

    for step in range(1, total_steps + 1):
        # linearly decay epsilon
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)

        action_idx = policy.get_action(obs, epsilon, device)
        force      = act_bins[action_idx]

        next_obs, reward, terminated, truncated, _ = env.step(np.array([force], dtype=np.float32))
        done = terminated or truncated

        buffer.push(obs, action_idx, reward, next_obs, done)
        ep_reward += reward
        ep_length += 1
        obs        = next_obs

        if done:
            obs, _  = env.reset()
            ep_count += 1
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            ep_wins.append(1 if ep_length >= max_ep_steps else 0)  # survived = win
            ep_reward = 0.0
            ep_length = 0

        # gradient update
        if step >= warmup and step % update_freq == 0 and buffer.size >= batch_size:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size, device)
            policy.train()
            loss, td, aux = compute_td_loss(
                policy, target, b_obs, b_act, b_rew, b_next, b_done, gamma, device
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)  # prevent exploding grads
            optimizer.step()

        # sync target network to live network
        if step % target_freq == 0:
            target.load_state_dict(policy.state_dict())

        # logging
        if step % log_interval == 0:
            window       = 20
            mean_r       = np.mean(ep_rewards[-window:])  if ep_rewards  else 0.0
            mean_len     = np.mean(ep_lengths[-window:])  if ep_lengths  else 0.0
            survival_pct = np.mean(ep_wins[-window:]) * 100 if ep_wins   else 0.0
            elapsed      = time.time() - t_start

            log_steps.append(step)
            log_mean_reward.append(mean_r)
            log_mean_length.append(mean_len)
            log_survival.append(survival_pct)

            # save only when mean reward improves
            if mean_r > best_mean_reward and len(ep_rewards) >= 20:
                best_mean_reward = mean_r
                torch.save(policy.state_dict(), best_model_path)
                print(f"  *** new best reward {mean_r:.2f} → saved {best_model_path}")

            print(f"step {step:>7} | eps {epsilon:.3f} | episodes {ep_count:>5} "
                  f"| reward {mean_r:>7.2f} | ep_len {mean_len:>6.1f} "
                  f"| survival {survival_pct:>5.1f}% | {elapsed:.0f}s")

    env.close()
    print("Training complete.")
    _plot_training(log_steps, log_mean_reward, log_mean_length, log_survival, policy_name)


def _plot_training(steps, rewards, lengths, survival, policy_name):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"DQN Training — {policy_name.upper()} policy")

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
    path = f"checkpoints/{policy_name}_dqn_training_curve.png"
    plt.savefig(path, dpi=150)
    print(f"  plot saved → {path}")
    plt.show()


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["gnn", "mlp"], default="gnn",
                        help="gnn = GNN-based DQN,  mlp = flat MLP baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.policy)
