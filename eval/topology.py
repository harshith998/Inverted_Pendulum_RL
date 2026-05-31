"""Topology-eval helpers shared by PPO, CGAT, DQN, and LQR eval scripts."""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLOR_IN_DIST = "#5ba4cf"
COLOR_OOD = "#1a3a5c"


def seed_suffix(seed: int | None) -> str:
    return "" if seed is None else f"_seed{seed}"


def topology_values(cfg: dict) -> list[int]:
    env_cfg = cfg["environment"]
    vals = env_cfg.get("topology_eval_n_links")
    if vals:
        return [int(v) for v in vals]
    max_links = int(env_cfg.get("max_links", env_cfg["n_links_range"][1]))
    return list(range(1, max_links + 1))


def topology_tag(n_links: int, train_range: tuple[int, int]) -> str:
    return "IN " if train_range[0] <= n_links <= train_range[1] else "OOD"


def topology_reward_ceiling(cfg: dict, n_links: int) -> float:
    env_cfg = cfg["environment"]
    reward_cfg = cfg.get("rewards", {})
    upright_weight = float(reward_cfg.get("upright_weight", 1.0))
    alive_bonus = float(reward_cfg.get("alive_bonus", 0.1))
    max_steps = int(env_cfg["max_episode_steps"])
    win_bonus = 2.0
    return max_steps * (upright_weight * n_links + alive_bonus) + win_bonus


def topology_reward_ceilings(cfg: dict, n_links_vals) -> np.ndarray:
    return np.array(
        [topology_reward_ceiling(cfg, int(n_links)) for n_links in n_links_vals],
        dtype=np.float32,
    )


def plot_topology_sweep(
    n_links_vals,
    rewards,
    train_range,
    title: str,
    path: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        COLOR_IN_DIST if train_range[0] <= n <= train_range[1] else COLOR_OOD
        for n in n_links_vals
    ]
    ax.plot(n_links_vals, rewards, color="#777777", linewidth=1.2, zorder=1)
    ax.scatter(n_links_vals, rewards, c=colors, s=52, edgecolors="none", zorder=2)
    ax.axvspan(train_range[0] - 0.15, train_range[1] + 0.15,
               alpha=0.10, color=COLOR_IN_DIST)
    ax.set_xticks(n_links_vals)
    ax.set_xlabel("Number of links")
    ax.set_ylabel("Mean reward")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_IN_DIST, label="In-distribution topology"),
            mpatches.Patch(color=COLOR_OOD, label="Topology OOD"),
        ],
        fontsize=9,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {path}")


def plot_topology_heatmaps(
    length_vals,
    mass_vals,
    reward_cube,
    n_links_vals,
    len_bounds,
    mass_bounds,
    train_topology_range,
    title: str,
    path: str,
    max_rewards=None,
):
    n = len(n_links_vals)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    im = None
    len_lo, len_hi = len_bounds
    mass_lo, mass_hi = mass_bounds
    plot_cube = reward_cube
    cbar_label = "Mean Reward"
    vmin, vmax = 0, 2000
    if max_rewards is not None:
        max_rewards = np.asarray(max_rewards, dtype=np.float32).reshape(-1, 1, 1)
        plot_cube = np.divide(
            reward_cube,
            max_rewards,
            out=np.zeros_like(reward_cube, dtype=np.float32),
            where=max_rewards > 0,
        )
        plot_cube = np.clip(plot_cube, 0.0, 1.0)
        cbar_label = "Normalized Reward"
        vmin, vmax = 0.0, 1.0

    for idx, n_links in enumerate(n_links_vals):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.pcolormesh(
            length_vals, mass_vals, plot_cube[idx],
            cmap="Greens", vmin=vmin, vmax=vmax, shading="auto",
        )
        rect = mpatches.Rectangle(
            (len_lo, mass_lo), len_hi - len_lo, mass_hi - mass_lo,
            linewidth=2, edgecolor="white", facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        tag = topology_tag(n_links, train_topology_range)
        ax.set_title(f"{n_links} link(s) [{tag.strip()}]")
        ax.grid(alpha=0.12)
        ax.set_xlabel("Link Length (m)")
        ax.set_ylabel("Link Mass (kg)")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(cbar_label)
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")
