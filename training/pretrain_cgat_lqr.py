# Run: python3.12 training/pretrain_cgat_lqr.py --config configs/cgat_3link.yaml --variant base --seed 40

"""Behavior-clone CGAT from exact 3-link LQR actions.

The PPO-from-scratch policy struggles to discover the stabilizing manifold for
3-link systems. This script trains the CGAT actor to imitate exact LQR around
the upright region, giving later RL fine-tuning a strong classical prior.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from env.mujoco_builder import PendulumConfig
from eval.eval_lqr import compute_lqr_gain
from graph.graph_builder import build_graph
from models.cgat import VARIANTS, load_cgat_variant


RAIL_LIMIT = 2.5
MAX_FORCE = 20.0


def set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_batch_to_tensor(obs_batch: dict, device: torch.device) -> dict:
    return {
        "node_features": torch.tensor(obs_batch["node_features"], dtype=torch.float32, device=device),
        "edge_index": torch.tensor(obs_batch["edge_index"], dtype=torch.int64, device=device),
        "edge_features": torch.tensor(obs_batch["edge_features"], dtype=torch.float32, device=device),
        "n_nodes": torch.tensor(obs_batch["n_nodes"], dtype=torch.int64, device=device),
        "n_edges": torch.tensor(obs_batch["n_edges"], dtype=torch.int64, device=device),
    }


def deterministic_action(policy, obs_t: dict) -> torch.Tensor:
    emb = policy.encode(obs_t)
    actor_h = policy.actor_trunk(emb)
    raw_mean = policy.mean_head(actor_h)
    return torch.tanh(raw_mean) * policy.max_force


def make_dataset(
    cfg: dict,
    configs_per_axis: int,
    states_per_config: int,
    seed: int | None,
    *,
    cart_pos_max: float,
    angle_max: float,
    cart_vel_max: float,
    angle_vel_max: float,
):
    rng = np.random.default_rng(seed)
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    cart_lo, cart_hi = env_cfg["cart_mass_range"]

    # Cover one train-width beyond both sides, matching the OOD eval protocol.
    len_width = len_hi - len_lo
    mass_width = mass_hi - mass_lo
    length_vals = np.linspace(max(0.05, len_lo - len_width), len_hi + len_width, configs_per_axis)
    mass_vals = np.linspace(max(0.05, mass_lo - mass_width), mass_hi + mass_width, configs_per_axis)
    cart_vals = np.array([(cart_lo + cart_hi) / 2.0], dtype=float)

    obs_rows = []
    actions = []
    n_links = 3
    state_dim = 2 * (n_links + 1)

    total_cfgs = len(length_vals) * len(mass_vals) * len(cart_vals)
    done_cfgs = 0

    for cart_mass in cart_vals:
        for length in length_vals:
            for mass in mass_vals:
                done_cfgs += 1
                K = compute_lqr_gain(
                    [float(length)] * n_links,
                    [float(mass)] * n_links,
                    float(cart_mass),
                    n_links,
                    rail_limit=env_cfg["rail_limit"],
                    max_force=env_cfg["max_force"],
                    timestep=env_cfg["timestep"],
                )
                config = PendulumConfig(
                    n_links=n_links,
                    lengths=[float(length)] * n_links,
                    masses=[float(mass)] * n_links,
                    cart_mass=float(cart_mass),
                )

                for _ in range(states_per_config):
                    x = np.zeros(state_dim, dtype=float)
                    x[0] = rng.uniform(-cart_pos_max, cart_pos_max)
                    x[1:4] = rng.uniform(-angle_max, angle_max, size=3)
                    x[4] = rng.uniform(-cart_vel_max, cart_vel_max)
                    x[5:8] = rng.uniform(-angle_vel_max, angle_vel_max, size=3)

                    action = float(-(K @ x)[0])
                    action = float(np.clip(action, -env_cfg["max_force"], env_cfg["max_force"]))

                    graph = build_graph(
                        config,
                        cart_pos=x[0],
                        cart_vel=x[4],
                        joint_angles=x[1:4],
                        joint_vels=x[5:8],
                    )
                    obs_rows.append(graph)
                    actions.append(action)

                if done_cfgs % 10 == 0 or done_cfgs == total_cfgs:
                    print(f"  generated {done_cfgs}/{total_cfgs} LQR configs")

    max_nodes = env_cfg.get("max_links", 3) + 1
    max_edges = 2 * env_cfg.get("max_links", 3)
    n = len(obs_rows)
    node_features = np.zeros((n, max_nodes, 9), dtype=np.float32)
    edge_index = np.zeros((n, 2, max_edges), dtype=np.int64)
    edge_features = np.zeros((n, max_edges, 2), dtype=np.float32)
    n_nodes = np.zeros((n, 1), dtype=np.int64)
    n_edges = np.zeros((n, 1), dtype=np.int64)

    for i, graph in enumerate(obs_rows):
        node_features[i, : graph.n_nodes] = graph.node_features
        edge_index[i, :, : graph.n_edges] = graph.edge_index
        edge_features[i, : graph.n_edges] = graph.edge_features
        n_nodes[i, 0] = graph.n_nodes
        n_edges[i, 0] = graph.n_edges

    obs = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
    }
    return obs, np.array(actions, dtype=np.float32).reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="Pretrain CGAT by imitating exact LQR.")
    parser.add_argument("--config", default="configs/cgat_3link.yaml")
    parser.add_argument("--variant", default="base", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--configs_per_axis", type=int, default=9)
    parser.add_argument("--states_per_config", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cart_pos_max", type=float, default=0.25)
    parser.add_argument("--angle_max", type=float, default=0.08)
    parser.add_argument("--cart_vel_max", type=float, default=0.25)
    parser.add_argument("--angle_vel_max", type=float, default=0.25)
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg["environment"]
    ppo_cfg = cfg["ppo"]
    h_cfg = ppo_cfg.get("cgat", {})

    policy = load_cgat_variant(
        args.variant,
        hidden=h_cfg.get("hidden_dim", ppo_cfg["hidden_dim"]),
        n_icga_layers=h_cfg.get("n_icga_layers", 2),
        n_heads=h_cfg.get("n_heads", 2),
        max_links=env_cfg.get("max_links", 3),
        max_force=env_cfg["max_force"],
    ).to(device)

    obs, actions = make_dataset(
        cfg,
        configs_per_axis=args.configs_per_axis,
        states_per_config=args.states_per_config,
        seed=args.seed,
        cart_pos_max=args.cart_pos_max,
        angle_max=args.angle_max,
        cart_vel_max=args.cart_vel_max,
        angle_vel_max=args.angle_vel_max,
    )

    n = actions.shape[0]
    target = torch.tensor(actions, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"Device: {device} | samples: {n} | variant: {args.variant}")
    indices = np.arange(n)
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(indices)
        losses = []
        maes = []
        for start in range(0, n, args.batch_size):
            idx = indices[start : start + args.batch_size]
            obs_b = {k: v[idx] for k, v in obs.items()}
            obs_t = obs_batch_to_tensor(obs_b, device)
            pred = deterministic_action(policy, obs_t)
            y = target[idx]
            loss = F.mse_loss(pred / MAX_FORCE, y / MAX_FORCE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            maes.append(float((pred - y).abs().mean().item()))

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d} | mse_norm={np.mean(losses):.6f} | action_mae={np.mean(maes):.3f} N")

    os.makedirs("checkpoints", exist_ok=True)
    suffix = "" if args.seed is None else f"_seed{args.seed}"
    out = f"checkpoints/cgat_{args.variant}_lqr_bc{suffix}.pt"
    torch.save(policy.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
