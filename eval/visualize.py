# macOS requires mjpython (MuJoCo's bundled Python) for the passive viewer.
# Run: mjpython eval/visualize.py
#      mjpython eval/visualize.py --policy gnn          # GNN DQN (default)
#      mjpython eval/visualize.py --policy mlp          # MLP DQN
#      mjpython eval/visualize.py --policy gnn_mpnn     # GNN MPNN PPO
#      mjpython eval/visualize.py --policy gnn_transformer
#      mjpython eval/visualize.py --policy random
#      mjpython eval/visualize.py --max_attempts 50     # give up after N episodes if no win

"""
Finds a winning episode for the chosen policy and replays it infinitely
in a MuJoCo viewer window so you can watch the pendulum stay balanced.

A "win" = episode survived to max_episode_steps (truncated, not terminated).
If no win is found within --max_attempts episodes, replays the best episode
seen instead.

Close the viewer window to exit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer as mjviewer

from env.pendulum_env import VariablePendulumEnv
from env.mujoco_builder import build_mjcf
from models.gnn_dqn import GNNDQNPolicy
from models.mlp_dqn import MLPDQNPolicy
from models.gnn_mpnn_ppo import GNNMPNNPPOPolicy
from models.gnn_transformer_ppo import GNNTransformerPPOPolicy
from models.mlp_ppo import MLPPPOPolicy
from models.random_baseline import RandomDQNPolicy, RandomPPOPolicy


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(policy_name: str, cfg: dict, device: torch.device,
                checkpoint: str | None = None):
    env_cfg = cfg["environment"]
    dqn_cfg = cfg["dqn"]
    ppo_cfg = cfg["ppo"]
    max_links = env_cfg["n_links_range"][1]
    max_force = env_cfg["max_force"]

    is_ppo = policy_name in ("gnn_mpnn", "gnn_transformer", "mlp_ppo")

    if policy_name == "random":
        # Return a DQN-interface random policy for simplicity
        return RandomDQNPolicy(dqn_cfg["n_action_bins"]), False

    if policy_name in ("gnn", "mlp"):
        n_bins  = dqn_cfg["n_action_bins"]
        hidden  = dqn_cfg["hidden_dim"]
        n_layers = dqn_cfg["n_mp_layers"]
        if policy_name == "gnn":
            policy = GNNDQNPolicy(n_bins, hidden=hidden, n_mp_layers=n_layers,
                                  max_links=max_links)
            ckpt = checkpoint or "checkpoints/gnn_dqn_best.pt"
        else:
            policy = MLPDQNPolicy(n_bins, max_links=max_links, hidden=hidden)
            ckpt = checkpoint or "checkpoints/mlp_dqn_best.pt"
        state_dict = torch.load(ckpt, map_location=device)
        policy.load_state_dict(state_dict, strict=False)
        policy.to(device).eval()
        return policy, False   # False = DQN interface

    # PPO policies
    hidden   = ppo_cfg["hidden_dim"]
    n_layers = ppo_cfg["n_layers"]
    n_heads  = ppo_cfg["n_heads"]
    dropout  = ppo_cfg["dropout"]
    t_cfg    = ppo_cfg.get("gnn_transformer", {})

    if policy_name == "gnn_mpnn":
        policy = GNNMPNNPPOPolicy(hidden=hidden, n_layers=n_layers,
                                  max_links=max_links, dropout=dropout,
                                  max_force=max_force)
        ckpt = checkpoint or "checkpoints/gnn_mpnn_ppo_best.pt"
    elif policy_name == "gnn_transformer":
        policy = GNNTransformerPPOPolicy(
            hidden=hidden,
            n_layers=t_cfg.get("n_layers", n_layers),
            n_heads=t_cfg.get("n_heads", n_heads),
            max_links=max_links, dropout=dropout, max_force=max_force)
        ckpt = checkpoint or "checkpoints/gnn_transformer_ppo_best.pt"
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    state_dict = torch.load(ckpt, map_location=device)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device).eval()
    return policy, True   # True = PPO interface


# ---------------------------------------------------------------------------
# Make env
# ---------------------------------------------------------------------------

def make_env(cfg: dict) -> VariablePendulumEnv:
    env_cfg = cfg["environment"]
    return VariablePendulumEnv(
        n_links_range     = tuple(env_cfg["n_links_range"]),
        cart_mass_range   = tuple(env_cfg["cart_mass_range"]),
        link_length_range = tuple(env_cfg["link_length_range"]),
        link_mass_range   = tuple(env_cfg["link_mass_range"]),
        rail_limit        = env_cfg["rail_limit"],
        max_force         = env_cfg["max_force"],
        timestep          = env_cfg["timestep"],
        frame_skip        = env_cfg["frame_skip"],
        max_episode_steps = env_cfg["max_episode_steps"],
        termination_angle = env_cfg["termination_angle"],
    )


# ---------------------------------------------------------------------------
# Run one episode, record trajectory
# ---------------------------------------------------------------------------

def run_episode(env: VariablePendulumEnv, policy, is_ppo: bool,
                act_bins: np.ndarray, device: torch.device):
    """
    Returns (trajectory, total_reward, won).

    trajectory: list of (qpos, qvel, ctrl) numpy arrays, one per env step.
    won: True if episode reached max_episode_steps.
    """
    obs, _ = env.reset()
    trajectory = []
    ep_reward  = 0.0
    done       = False

    while not done:
        # Snapshot state before step
        trajectory.append((
            env._mj_data.qpos.copy(),
            env._mj_data.qvel.copy(),
            env._mj_data.ctrl.copy(),
        ))

        if is_ppo:
            force, _, _ = policy.get_action(obs, device)
        else:
            idx   = policy.get_action(obs, epsilon=0.0, device=device)
            force = float(act_bins[idx])

        obs, reward, terminated, truncated, _ = env.step(
            np.array([force], dtype=np.float32))
        ep_reward += reward
        done = terminated or truncated

    won = truncated
    return trajectory, ep_reward, won


# ---------------------------------------------------------------------------
# Replay a recorded trajectory infinitely
# ---------------------------------------------------------------------------

def replay_loop(env: VariablePendulumEnv, trajectory: list, cfg: dict):
    """
    Replays trajectory frames in a MuJoCo viewer window, looping forever.
    Close the window to exit.
    """
    env_cfg    = cfg["environment"]
    frame_dt   = env_cfg["timestep"] * env_cfg["frame_skip"]   # real seconds per env step
    mj_model   = env._mj_model
    mj_data    = env._mj_data

    n_frames = len(trajectory)
    print(f"\nReplaying {n_frames}-step episode. Close window to exit.\n")

    with mjviewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            for qpos, qvel, ctrl in trajectory:
                if not viewer.is_running():
                    break
                step_start = time.perf_counter()

                mj_data.qpos[:] = qpos
                mj_data.qvel[:] = qvel
                mj_data.ctrl[:] = ctrl
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                # Sleep to maintain real-time playback speed
                elapsed = time.perf_counter() - step_start
                remaining = frame_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            # Brief pause between loop iterations
            time.sleep(0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="gnn",
        choices=["gnn", "mlp", "gnn_mpnn", "gnn_transformer", "random"],
        help="Policy to visualize (default: gnn)")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--checkpoint",   default=None,
        help="Override checkpoint path")
    parser.add_argument("--max_attempts", type=int, default=30,
        help="Max episodes to try before falling back to best seen (default: 30)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Policy     : {args.policy}")
    print(f"Device     : {device}")

    policy, is_ppo = load_policy(args.policy, cfg, device, args.checkpoint)

    env_cfg  = cfg["environment"]
    dqn_cfg  = cfg["dqn"]
    act_bins = np.linspace(-env_cfg["max_force"], env_cfg["max_force"],
                           dqn_cfg["n_action_bins"])

    env = make_env(cfg)

    # Search for a winning episode
    best_traj   = None
    best_reward = -np.inf

    print(f"\nSearching for a win (max {args.max_attempts} attempts)...")

    for attempt in range(1, args.max_attempts + 1):
        traj, reward, won = run_episode(env, policy, is_ppo, act_bins, device)
        status = "WIN" if won else f"fail ({len(traj)} steps)"
        print(f"  Episode {attempt:3d}: reward={reward:7.1f}  {status}")

        if reward > best_reward:
            best_reward = reward
            best_traj   = traj
            # Save the MuJoCo model/data state from this episode for replay
            best_config = env._config

        if won:
            print(f"\nWin found on attempt {attempt}! Reward: {reward:.1f}")
            break
    else:
        print(f"\nNo full win in {args.max_attempts} attempts. "
              f"Replaying best episode (reward={best_reward:.1f}, "
              f"{len(best_traj)} steps).")
        # Restore env to best config for replay
        env._load_model(best_config)

    replay_loop(env, best_traj, cfg)
    env.close()


if __name__ == "__main__":
    main()
