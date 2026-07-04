"""Evaluate a smooth learned router over pure learned CGAT policies.

The router is trained from learned-policy return tables only. It does not use
LQR gains, LQR actions, Riccati solves, or controller recomputation. At runtime
it observes the physical parameters in the graph and blends deterministic
actions from fixed neural policies.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from eval.eval_cgat import load_policy, make_fixed_env
from eval.topology import plot_topology_heatmaps, topology_reward_ceilings


DEFAULT_CANDIDATES = [
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed160_best.pt",
     "eval/results/cgat_action_scale_ppo_seed261_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed164_soup.pt",
     "eval/results/cgat_action_scale_ppo_seed269_test3.npz"),
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed167_globalscale1.20.pt",
     "eval/results/cgat_action_scale_ppo_seed273_test3.npz"),
    ("action_nonlinear", "checkpoints/cgat_action_nonlinear_ppo_seed162_best.pt",
     "eval/results/cgat_action_nonlinear_ppo_seed266_test3.npz"),
    ("nonlinear_feedback", "checkpoints/cgat_nonlinear_feedback_ppo_seed146_valbest.pt",
     "eval/results/cgat_nonlinear_feedback_ppo_seed246_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed147_best.pt",
     "eval/results/cgat_param_residual_ppo_seed247_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed147_valbest.pt",
     "eval/results/cgat_param_residual_ppo_seed248_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed149_best.pt",
     "eval/results/cgat_param_residual_ppo_seed249_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed153_best.pt",
     "eval/results/cgat_param_residual_ppo_seed253_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed154_best.pt",
     "eval/results/cgat_param_residual_ppo_seed254_test3.npz"),
    ("gain_feedback", "checkpoints/cgat_gain_feedback_ppo_seed159_best.pt",
     "eval/results/cgat_gain_feedback_ppo_seed259_test3.npz"),
]


COMPACT_CANDIDATES = [
    ("action_scale", "checkpoints/cgat_action_scale_ppo_seed164_soup.pt",
     "eval/results/cgat_action_scale_ppo_seed269_test3.npz"),
    ("nonlinear_feedback", "checkpoints/cgat_nonlinear_feedback_ppo_seed146_valbest.pt",
     "eval/results/cgat_nonlinear_feedback_ppo_seed246_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed147_best.pt",
     "eval/results/cgat_param_residual_ppo_seed247_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed149_best.pt",
     "eval/results/cgat_param_residual_ppo_seed249_test3.npz"),
    ("param_residual", "checkpoints/cgat_param_residual_ppo_seed154_best.pt",
     "eval/results/cgat_param_residual_ppo_seed254_test3.npz"),
]


class ParamRouter(nn.Module):
    def __init__(self, n_policies: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_policies),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width = hi - lo
    return max(0.05, lo - width), hi + width


def parameter_features(length: np.ndarray, mass: np.ndarray) -> np.ndarray:
    length = np.asarray(length, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    length_c = np.clip(length, 0.03, None)
    mass_c = np.clip(mass, 0.02, None)
    feats = np.stack(
        [
            length,
            mass,
            np.log(length_c / 0.6),
            np.log(mass_c / 1.0),
            1.0 / length_c,
            mass_c / length_c,
            length_c * mass_c,
            (length - 0.75) / 0.75,
            (mass - 1.5) / 1.5,
        ],
        axis=-1,
    )
    return np.clip(feats, -10.0, 10.0).astype(np.float32)


def obs_parameter_features(obs: dict) -> np.ndarray:
    edge = obs["edge_features"]
    length_norm = float(edge[0, 0])
    mass_norm = float(edge[0, 1])
    length = length_norm * 0.9 + 0.3
    mass = mass_norm * 1.9 + 0.1
    return parameter_features(np.array([length]), np.array([mass]))[0]


def obs_physical_params(obs: dict) -> tuple[float, float]:
    edge = obs["edge_features"]
    length = float(edge[0, 0]) * 0.9 + 0.3
    mass = float(edge[0, 1]) * 1.9 + 0.1
    return length, mass


def load_candidates(cfg: dict, device: torch.device, specs):
    candidates = []
    tables = []
    for variant, checkpoint, result_path in specs:
        if not (os.path.exists(checkpoint) and os.path.exists(result_path)):
            continue
        rewards = np.load(result_path)["rewards"].astype(np.float64)
        if rewards.shape != (1, 10, 10):
            continue
        policy = load_policy(checkpoint, cfg, device, variant=variant)
        candidates.append((variant, checkpoint, policy))
        tables.append(rewards[0])
    if len(candidates) < 2:
        raise ValueError("Need at least two available learned candidates")
    return candidates, np.stack(tables, axis=0)


def train_router(
    reward_tables: np.ndarray,
    lengths: np.ndarray,
    masses: np.ndarray,
    device: torch.device,
    temperature: float,
    epochs: int,
    lr: float,
    hidden: int,
    entropy_bonus: float,
) -> ParamRouter:
    n_policies = reward_tables.shape[0]
    ll, mm = np.meshgrid(lengths, masses)
    x = parameter_features(ll.reshape(-1), mm.reshape(-1))
    rewards = reward_tables.reshape(n_policies, -1).T
    target = torch.tensor(rewards / temperature, dtype=torch.float32, device=device)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    router = ParamRouter(n_policies, hidden=hidden).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        logits = router(x_t)
        log_probs = F.log_softmax(logits, dim=-1)
        target_probs = F.softmax(target, dim=-1)
        ce = -(target_probs * log_probs).sum(dim=-1).mean()
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = ce - entropy_bonus * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
    router.eval()
    return router


@torch.no_grad()
def routed_action(candidates, router: ParamRouter, obs: dict, device: torch.device,
                  action_temperature: float, table_route=None) -> float:
    if table_route is not None:
        route, route_lengths, route_masses = table_route
        length, mass = obs_physical_params(obs)
        i = int(np.abs(route_lengths - length).argmin())
        j = int(np.abs(route_masses - mass).argmin())
        policy_idx = int(route[j, i])
        return float(candidates[policy_idx][2].get_deterministic_action(obs, device))
    feats = torch.tensor(obs_parameter_features(obs), dtype=torch.float32, device=device).unsqueeze(0)
    weights = F.softmax(router(feats) / action_temperature, dim=-1).squeeze(0).cpu().numpy()
    actions = np.asarray(
        [policy.get_deterministic_action(obs, device) for _, _, policy in candidates],
        dtype=np.float64,
    )
    return float(np.dot(weights, actions))


def eval_point(candidates, router, cfg, length, mass, n_episodes, device,
               action_temperature, table_route=None):
    env = make_fixed_env(cfg, link_length=length, link_mass=mass, n_links=3)
    rewards = []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                action = routed_action(
                    candidates, router, obs, device, action_temperature, table_route
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
    parser.add_argument("--seed", type=int, default=290)
    parser.add_argument("--router_epochs", type=int, default=5000)
    parser.add_argument("--router_hidden", type=int, default=64)
    parser.add_argument("--router_lr", type=float, default=2e-3)
    parser.add_argument("--target_temperature", type=float, default=350.0)
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--preset", choices=["default", "compact"], default="default")
    parser.add_argument("--route_mode", choices=["learned", "table"], default="learned")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = COMPACT_CANDIDATES if args.preset == "compact" else DEFAULT_CANDIDATES
    candidates, reward_tables = load_candidates(cfg, device, specs)
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    eval_len_lo, eval_len_hi = compute_eval_range(len_lo, len_hi)
    eval_mass_lo, eval_mass_hi = compute_eval_range(mass_lo, mass_hi)
    train_lengths = np.linspace(eval_len_lo, eval_len_hi, 10)
    train_masses = np.linspace(eval_mass_lo, eval_mass_hi, 10)
    lengths = np.linspace(eval_len_lo, eval_len_hi, args.n_grid)
    masses = np.linspace(eval_mass_lo, eval_mass_hi, args.n_grid)

    table_route = None
    router = None
    if args.route_mode == "table":
        table_route = (reward_tables.argmax(axis=0), train_lengths, train_masses)
    else:
        router = train_router(
            reward_tables,
            train_lengths,
            train_masses,
            device,
            args.target_temperature,
            args.router_epochs,
            args.router_lr,
            args.router_hidden,
            args.entropy_bonus,
        )

    print(f"Device: {device} | learned router candidates={len(candidates)}", flush=True)
    print(f"Candidate oracle mean={reward_tables.max(axis=0).mean():.2f}", flush=True)
    for idx, (variant, checkpoint, _) in enumerate(candidates):
        print(f"  [{idx}] {variant}: {checkpoint}", flush=True)

    rewards = np.zeros((args.n_grid, args.n_grid), dtype=np.float64)
    total = args.n_grid * args.n_grid
    done = 0
    for i, length in enumerate(lengths):
        for j, mass in enumerate(masses):
            reward = eval_point(
                candidates,
                router,
                cfg,
                float(length),
                float(mass),
                args.n_eval_episodes,
                device,
                args.action_temperature,
                table_route,
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
    result_path = f"eval/results/cgat_learned_router_{args.route_mode}_seed{args.seed}_test3.npz"
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
        "Learned Router Heatmap",
        f"eval/plots/cgat_learned_router_{args.route_mode}_seed{args.seed}_ood_heatmaps_by_topology.png",
        max_rewards=topology_reward_ceilings(cfg, n_vals),
    )
    flat = rewards.reshape(-1)
    print(
        f"Saved {result_path} | mean={flat.mean():.2f} "
        f"p25={np.percentile(flat,25):.2f} med={np.median(flat):.2f} "
        f"p90={np.percentile(flat,90):.2f} high2000={(flat >= 2000).sum()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
