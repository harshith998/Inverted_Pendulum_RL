# Run: python3.12 eval/eval_cgat.py
#      python3.12 eval/eval_cgat.py --variant perhead
#      python3.12 eval/eval_cgat.py --checkpoint checkpoints/cgat_base_ppo_best.pt
#      python3.12 eval/eval_cgat.py --tests 1 2        # skip heatmap
#      python3.12 eval/eval_cgat.py --tests 3           # heatmap only
#      python3.12 eval/eval_cgat.py --compare           # run all variants side-by-side

"""
OOD evaluation suite for the Coupled Graph Attention Transformer (CGAT).

Identical structure to eval_ppo.py — same tests and plots
format — so results are directly comparable across all PPO baselines.

Tests
-----
  1. 1D sweep — link_length   (100 pts × 10 eps each)
  2. 1D sweep — link_mass     (100 pts × 10 eps each)
  3. 2D heatmap — link_length × link_mass  (20×20 grid)
  4. Few-shot OOD adaptation on far-OOD parameter points

Variants
--------
  base, perhead, directional, gravity, perc
  --compare runs all found checkpoints and overlays them on shared plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.pendulum_env import VariablePendulumEnv
from models.cgat import load_cgat_variant, VARIANTS
from eval.few_shot import (
    plot_few_shot,
    run_few_shot,
    save_few_shot_results,
    select_eval_action,
    summarize_few_shot,
)
from eval.topology import (
    plot_topology_heatmaps,
    plot_topology_sweep,
    seed_suffix,
    topology_tag,
    topology_reward_ceilings,
    topology_values,
)


# ── Defaults ──────────────────────────────────────────────────────────────────

N_SWEEP_POINTS  = 100
N_GRID_1D       = 20
N_EVAL_EPISODES = 10
MIN_PARAM_VAL   = 0.05

COLOR_IN_DIST = "#5ba4cf"
COLOR_OOD     = "#1a3a5c"

# Per-variant plot colours for --compare mode
VARIANT_COLORS = {
    "base":        "#e06c00",
    "perhead":     "#8e44ad",
    "directional": "#27ae60",
    "gravity":     "#c0392b",
    "perc":        "#2980b9",
    "no_physics":  "#555555",
    "shuffled":    "#d4a017",
}


def _default_ckpt(variant: str) -> str:
    return f"checkpoints/cgat_{variant}_ppo_best.pt"


def _seeded_ckpt(variant: str, seed: int | None) -> str:
    return f"checkpoints/cgat_{variant}_ppo{seed_suffix(seed)}_best.pt"


def resolve_checkpoint(variant: str, requested: str | None, seed: int | None) -> str:
    if requested is not None:
        return requested
    return _seeded_ckpt(variant, seed)


def set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, cfg: dict, device: torch.device,
                variant: str = "base"):
    env_cfg   = cfg["environment"]
    ppo_cfg   = cfg["ppo"]
    h_cfg     = ppo_cfg.get("cgat", {})

    max_links    = env_cfg.get("max_links", env_cfg["n_links_range"][1])
    hidden    = h_cfg.get("hidden_dim",    ppo_cfg["hidden_dim"])
    n_icga    = h_cfg.get("n_icga_layers", 2)
    n_heads   = h_cfg.get("n_heads",       2)
    max_force = env_cfg["max_force"]

    policy = load_cgat_variant(
        variant, hidden=hidden, n_icga_layers=n_icga,
        n_heads=n_heads, max_links=max_links, max_force=max_force,
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] checkpoint missing keys: {missing}")
    if unexpected:
        print(f"  [warn] checkpoint unexpected keys (ignored): {unexpected}")

    policy.to(device)
    policy.eval()
    return policy


def _beta_summary(policy, variant: str) -> str:
    """Single-line physics-weight summary for the eval header."""
    layer = policy.encoder.icga_layers[0]
    if not hasattr(layer, "physics_beta") and variant != "directional":
        return "β=n/a"
    if variant == "directional":
        import torch
        b_fwd = float(torch.tanh(layer.physics_beta_fwd).item())
        b_bwd = float(torch.tanh(layer.physics_beta_bwd).item())
        return f"β_fwd={b_fwd:+.4f}  β_bwd={b_bwd:+.4f}"
    import torch
    beta = layer.physics_beta
    if beta.numel() > 1:
        vals = torch.tanh(beta).tolist()
        return "β_heads=[" + ", ".join(f"{v:+.3f}" for v in vals) + "]"
    return f"β={float(beta.item()):+.4f}"


# ── Environment factory ───────────────────────────────────────────────────────

def make_fixed_env(cfg: dict, link_length: float, link_mass: float,
                   cart_mass: float | None = None,
                   n_links: int | None = None) -> VariablePendulumEnv:
    env_cfg = cfg["environment"]
    if cart_mass is None:
        lo, hi = env_cfg["cart_mass_range"]
        cart_mass = (lo + hi) / 2.0
    if n_links is None:
        n_links = env_cfg.get(
            "nominal_eval_n_links",
            round(sum(env_cfg["n_links_range"]) / 2),
        )
    return VariablePendulumEnv(
        n_links_range     = (n_links, n_links),
        cart_mass_range   = (cart_mass, cart_mass),
        link_length_range = (link_length, link_length),
        link_mass_range   = (link_mass, link_mass),
        rail_limit        = env_cfg["rail_limit"],
        max_force         = env_cfg["max_force"],
        timestep          = env_cfg["timestep"],
        frame_skip        = env_cfg["frame_skip"],
        max_episode_steps = env_cfg["max_episode_steps"],
        termination_angle = env_cfg["termination_angle"],
        max_links         = env_cfg.get("max_links"),
    )


# ── Core evaluation ───────────────────────────────────────────────────────────

def eval_point(policy, env, n_episodes: int, device: torch.device,
               stochastic_eval: bool = False) -> float:
    rewards = []
    for _ in range(n_episodes):
        try:
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = select_eval_action(
                    policy, obs, device, stochastic=stochastic_eval)
                obs, reward, terminated, truncated, _ = env.step(
                    np.array([action], dtype=np.float32))
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        except Exception:
            rewards.append(0.0)
    return float(np.mean(rewards)) if rewards else 0.0


# ── Eval range helpers ────────────────────────────────────────────────────────

def compute_eval_range(lo: float, hi: float) -> tuple[float, float]:
    width   = hi - lo
    eval_lo = max(MIN_PARAM_VAL, lo - width)
    eval_hi = hi + width
    return eval_lo, eval_hi


# ── Test 1 — 1D link_length sweep ────────────────────────────────────────────

def run_test1(policy, cfg, device, n_episodes, n_points, stochastic_eval=False):
    env_cfg = cfg["environment"]
    len_lo, len_hi = env_cfg["link_length_range"]
    mass_mid = sum(env_cfg["link_mass_range"]) / 2.0

    eval_lo, eval_hi = compute_eval_range(len_lo, len_hi)
    length_vals      = np.linspace(eval_lo, eval_hi, n_points)
    rewards          = []

    print(f"\n{'='*60}")
    print(f"[Test 1] Link Length sweep")
    print(f"  Eval range : {eval_lo:.3f}m → {eval_hi:.3f}m  ({n_points} pts)")
    print(f"  Train dist : [{len_lo:.3f}, {len_hi:.3f}]m")
    print(f"  Link mass fixed at {mass_mid:.3f} kg  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, length in enumerate(length_vals):
        env = make_fixed_env(cfg, link_length=length, link_mass=mass_mid)
        r = eval_point(policy, env, n_episodes, device,
                       stochastic_eval=stochastic_eval)
        env.close()
        tag = "IN " if len_lo <= length <= len_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] length={length:.4f}m  reward={r:8.2f}  [{tag}]")

    return length_vals, np.array(rewards), len_lo, len_hi


# ── Test 2 — 1D link_mass sweep ───────────────────────────────────────────────

def run_test2(policy, cfg, device, n_episodes, n_points, stochastic_eval=False):
    env_cfg = cfg["environment"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    len_mid = sum(env_cfg["link_length_range"]) / 2.0

    eval_lo, eval_hi = compute_eval_range(mass_lo, mass_hi)
    mass_vals        = np.linspace(eval_lo, eval_hi, n_points)
    rewards          = []

    print(f"\n{'='*60}")
    print(f"[Test 2] Link Mass sweep")
    print(f"  Eval range : {eval_lo:.3f}kg → {eval_hi:.3f}kg  ({n_points} pts)")
    print(f"  Train dist : [{mass_lo:.3f}, {mass_hi:.3f}]kg")
    print(f"  Link length fixed at {len_mid:.3f} m  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, mass in enumerate(mass_vals):
        env = make_fixed_env(cfg, link_length=len_mid, link_mass=mass)
        r = eval_point(policy, env, n_episodes, device,
                       stochastic_eval=stochastic_eval)
        env.close()
        tag = "IN " if mass_lo <= mass <= mass_hi else "OOD"

        rewards.append(r)
        print(f"  [{i+1:3d}/{n_points}] mass={mass:.4f}kg  reward={r:8.2f}  [{tag}]")

    return mass_vals, np.array(rewards), mass_lo, mass_hi


# ── Test 2.5 — topology sweep ────────────────────────────────────────────────

def run_test25(policy, cfg, device, n_episodes, stochastic_eval=False):
    env_cfg = cfg["environment"]
    len_mid = sum(env_cfg["link_length_range"]) / 2.0
    mass_mid = sum(env_cfg["link_mass_range"]) / 2.0
    train_topology = tuple(env_cfg["n_links_range"])
    n_links_vals = topology_values(cfg)
    rewards = []

    print(f"\n{'='*60}")
    print("[Test 2.5] Topology sweep")
    print(f"  n_links : {n_links_vals}")
    print(f"  Train topology : {train_topology}")
    print(f"  Length={len_mid:.3f}m  Mass={mass_mid:.3f}kg  |  {n_episodes} eps/pt")
    print(f"{'='*60}")

    for i, n_links in enumerate(n_links_vals):
        env = make_fixed_env(
            cfg, link_length=len_mid, link_mass=mass_mid, n_links=n_links)
        r = eval_point(policy, env, n_episodes, device,
                       stochastic_eval=stochastic_eval)
        env.close()
        rewards.append(r)
        print(f"  [{i+1:3d}/{len(n_links_vals)}] n_links={n_links}  "
              f"reward={r:8.2f}  [{topology_tag(n_links, train_topology)}]")

    return np.array(n_links_vals), np.array(rewards), train_topology


# ── Test 3 — topology-aware heatmaps ─────────────────────────────────────────

def run_test3(policy, cfg, device, n_episodes, n_grid, stochastic_eval=False,
              partial_path: str | None = None, partial_every: int = 10,
              resume_partial: bool = False):
    env_cfg = cfg["environment"]
    len_lo,  len_hi  = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    n_links_vals = topology_values(cfg)

    len_eval_lo,  len_eval_hi  = compute_eval_range(len_lo,  len_hi)
    mass_eval_lo, mass_eval_hi = compute_eval_range(mass_lo, mass_hi)

    length_vals = np.linspace(len_eval_lo,  len_eval_hi,  n_grid)
    mass_vals   = np.linspace(mass_eval_lo, mass_eval_hi, n_grid)
    reward_cube = np.full((len(n_links_vals), n_grid, n_grid), np.nan)

    if resume_partial and partial_path and os.path.exists(partial_path):
        partial = np.load(partial_path)
        saved_rewards = partial["rewards"]
        if saved_rewards.shape == reward_cube.shape:
            reward_cube[:] = saved_rewards
            print(f"  Resuming partial heatmap from {partial_path}")
        else:
            print(
                f"  [warn] partial heatmap shape {saved_rewards.shape} "
                f"does not match {reward_cube.shape}; ignoring"
            )

    def save_partial():
        if not partial_path:
            return
        np.savez(
            partial_path,
            lengths=length_vals,
            masses=mass_vals,
            n_links=np.array(n_links_vals),
            rewards=reward_cube,
            len_bounds=np.array((len_lo, len_hi)),
            mass_bounds=np.array((mass_lo, mass_hi)),
            train_topology=np.array(tuple(env_cfg["n_links_range"])),
        )

    total = len(n_links_vals) * n_grid * n_grid
    print(f"\n{'='*60}")
    print(f"[Test 3] Topology heatmaps — {len(n_links_vals)}×{n_grid}×{n_grid} grid ({total} cells)")
    print(f"  n_links: {n_links_vals}")
    print(f"  Length : {len_eval_lo:.3f}m → {len_eval_hi:.3f}m")
    print(f"  Mass   : {mass_eval_lo:.3f}kg → {mass_eval_hi:.3f}kg")
    print(f"{'='*60}")

    done = 0
    for k, n_links in enumerate(n_links_vals):
        for i, length in enumerate(length_vals):
            for j, mass in enumerate(mass_vals):
                if np.isfinite(reward_cube[k, j, i]):
                    done += 1
                    continue
                env = make_fixed_env(
                    cfg, link_length=length, link_mass=mass, n_links=n_links)
                r = eval_point(policy, env, n_episodes, device,
                               stochastic_eval=stochastic_eval)
                env.close()
                reward_cube[k, j, i] = r
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done:4d}/{total}] n_links={n_links} "
                          f"length={length:.3f}m  mass={mass:.3f}kg  reward={r:.2f}")
                if partial_path and (
                    done % max(1, partial_every) == 0 or done == total
                ):
                    save_partial()

    return (length_vals, mass_vals, reward_cube, np.array(n_links_vals),
            (len_lo, len_hi), (mass_lo, mass_hi), tuple(env_cfg["n_links_range"]))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_1d(values, rewards, train_lo, train_hi,
            param_name, units, plot_dir, variant: str = "base",
            file_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = [COLOR_IN_DIST if train_lo <= v <= train_hi else COLOR_OOD
              for v in values]

    ax.plot(values, rewards, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax.scatter(values, rewards, c=colors, s=18, zorder=3, edgecolors="none")

    ax.axvspan(train_lo, train_hi, alpha=0.10, color=COLOR_IN_DIST)
    ax.axvline(train_lo, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(train_hi, color=COLOR_IN_DIST, linewidth=1.5, linestyle="--", alpha=0.7)

    ymin, ymax = ax.get_ylim()
    mid_y = (ymin + ymax) / 2
    ax.text(train_lo, mid_y, f" train lo\n {train_lo:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)
    ax.text(train_hi, mid_y, f" train hi\n {train_hi:.2f}{units}",
            color=COLOR_IN_DIST, fontsize=7, va="center", alpha=0.8)

    in_patch  = mpatches.Patch(color=COLOR_IN_DIST, label="In-distribution")
    ood_patch = mpatches.Patch(color=COLOR_OOD,     label="OOD")
    ax.legend(handles=[in_patch, ood_patch], fontsize=10)

    ax.set_ylim(0, 2000)
    ax.set_xlabel(f"{param_name} ({units})", fontsize=12)
    ax.set_ylabel(f"Mean Reward ({N_EVAL_EPISODES} eps)", fontsize=12)
    ax.set_title(
        f"OOD Generalisation — {param_name} | CGAT-{variant} PPO", fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    slug = param_name.lower().replace(" ", "_")
    path = os.path.join(plot_dir, f"cgat_{variant}_ppo{file_suffix}_{slug}_sweep.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


def plot_2d(length_vals, mass_vals, reward_grid,
            len_bounds, mass_bounds, plot_dir, variant: str = "base"):
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(length_vals, mass_vals, reward_grid,
                       cmap="Greens", vmin=0, vmax=2000, shading="auto")

    len_lo, len_hi   = len_bounds
    mass_lo, mass_hi = mass_bounds
    rect = mpatches.Rectangle(
        (len_lo, mass_lo), len_hi - len_lo, mass_hi - mass_lo,
        linewidth=2, edgecolor="white", facecolor="none",
        linestyle="--", label="Training distribution",
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Reward", fontsize=11)

    ax.set_xlabel("Link Length (m)", fontsize=12)
    ax.set_ylabel("Link Mass (kg)", fontsize=12)
    ax.set_title(
        f"OOD Heatmap — Length × Mass | CGAT-{variant} PPO\n"
        f"(white dashed box = training distribution)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"cgat_{variant}_ppo_ood_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.close()


# ── Compare mode — overlay all variants on one plot ───────────────────────────

def run_compare(cfg, device, args):
    """Load every variant whose checkpoint exists and overlay on shared 1D plots."""
    import matplotlib.pyplot as plt

    env_cfg   = cfg["environment"]
    len_lo,  len_hi  = env_cfg["link_length_range"]
    mass_lo, mass_hi = env_cfg["link_mass_range"]
    mass_mid = (mass_lo + mass_hi) / 2.0
    len_mid  = (len_lo  + len_hi)  / 2.0

    len_eval_lo,  len_eval_hi  = compute_eval_range(len_lo,  len_hi)
    mass_eval_lo, mass_eval_hi = compute_eval_range(mass_lo, mass_hi)
    length_vals = np.linspace(len_eval_lo,  len_eval_hi,  args.n_sweep_points)
    mass_vals   = np.linspace(mass_eval_lo, mass_eval_hi, args.n_sweep_points)

    os.makedirs("eval/plots", exist_ok=True)

    fig_len, ax_len = plt.subplots(figsize=(12, 5))
    fig_mass, ax_mass = plt.subplots(figsize=(12, 5))

    for ax, param_name in [(ax_len, "Link Length"), (ax_mass, "Link Mass")]:
        train_lo = len_lo if "Length" in param_name else mass_lo
        train_hi = len_hi if "Length" in param_name else mass_hi
        units    = "m"    if "Length" in param_name else "kg"
        ax.axvspan(train_lo, train_hi, alpha=0.08, color=COLOR_IN_DIST,
                   label="Training dist.")
        ax.axvline(train_lo, color=COLOR_IN_DIST, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(train_hi, color=COLOR_IN_DIST, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlabel(f"{param_name} ({units})", fontsize=12)
        ax.set_ylabel(f"Mean Reward ({args.n_eval_episodes} eps)", fontsize=12)
        ax.set_ylim(0, 2000)
        ax.grid(alpha=0.25)

    found_any = False
    for variant in VARIANTS:
        ckpt = _default_ckpt(variant)
        if not os.path.exists(ckpt):
            print(f"  [skip] {variant} — checkpoint not found ({ckpt})")
            continue

        print(f"\n--- variant: {variant} ---")
        policy = load_policy(ckpt, cfg, device, variant=variant)
        color  = VARIANT_COLORS.get(variant, "#555555")

        # Length sweep
        l_rewards = []
        for length in length_vals:
            env = make_fixed_env(cfg, link_length=length, link_mass=mass_mid)
            r = eval_point(
                policy, env, args.n_eval_episodes, device,
                stochastic_eval=args.stochastic_eval)
            env.close()
            l_rewards.append(r)
        ax_len.plot(length_vals, l_rewards, color=color, linewidth=1.5,
                    label=f"CGAT-{variant}")

        # Mass sweep
        m_rewards = []
        for mass in mass_vals:
            env = make_fixed_env(cfg, link_length=len_mid, link_mass=mass)
            r = eval_point(
                policy, env, args.n_eval_episodes, device,
                stochastic_eval=args.stochastic_eval)
            env.close()
            m_rewards.append(r)
        ax_mass.plot(mass_vals, m_rewards, color=color, linewidth=1.5,
                     label=f"CGAT-{variant}")

        found_any = True

    if not found_any:
        print("No variant checkpoints found. Train at least one variant first.")
        return

    for ax, fig, fname, title in [
        (ax_len,  fig_len,  "eval/plots/cgat_compare_length_sweep.png",
         "CGAT Variant Comparison — Link Length"),
        (ax_mass, fig_mass, "eval/plots/cgat_compare_mass_sweep.png",
         "CGAT Variant Comparison — Link Mass"),
    ]:
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"  Compare plot saved → {fname}")
        plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OOD evaluation for CGAT PPO policy variants.")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--variant",    default="base",
        choices=list(VARIANTS),
        help="CGAT variant to evaluate (default: base)")
    parser.add_argument("--checkpoint", default=None,
        help="Path to .pt file. Defaults to checkpoints/cgat_{variant}_ppo_best.pt")
    parser.add_argument("--tests", nargs="+",
        choices=["1", "2", "2.5", "3", "4"],
        default=["1", "2", "2.5", "3", "4"])
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--n_sweep_points",  type=int, default=N_SWEEP_POINTS)
    parser.add_argument("--n_grid",          type=int, default=N_GRID_1D)
    parser.add_argument("--heatmap_partial", default=None,
        help="Optional .npz path for Test 3 partial-save/resume data")
    parser.add_argument("--heatmap_partial_every", type=int, default=10,
        help="Save Test 3 partial results every N cells when --heatmap_partial is set")
    parser.add_argument("--resume_heatmap", action="store_true",
        help="Resume Test 3 from --heatmap_partial if it exists")
    parser.add_argument("--stochastic_eval", action="store_true",
        help="Sample from the Gaussian during eval instead of using the mean action")
    parser.add_argument("--few_shot_budgets", nargs="+", type=int,
        default=[0, 1, 5, 10, 25],
        help="Fine-tuning episode budgets for Test 4")
    parser.add_argument("--few_shot_tasks", type=int, default=4,
        help="Number of far-OOD corner tasks for Test 4")
    parser.add_argument("--few_shot_epochs", type=int, default=4,
        help="PPO epochs per few-shot adaptation batch")
    parser.add_argument("--few_shot_lr", type=float, default=3e-5,
        help="Learning rate for Test 4 adaptation")
    parser.add_argument("--compare", action="store_true",
        help="Run all available variants and generate comparison plots")
    parser.add_argument("--seed", type=int, default=None,
        help="Eval seed and checkpoint suffix. Uses checkpoints/cgat_{variant}_ppo_seedN_best.pt.")
    args = parser.parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        print(f"\nDevice : {device}")
        print(f"Mode   : --compare (all variants)")
        run_compare(cfg, device, args)
        print("Done.")
        return

    # ── Single variant mode ───────────────────────────────────────────────────
    variant    = args.variant
    checkpoint = resolve_checkpoint(variant, args.checkpoint, args.seed)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Train first:  python3.12 training/train_cgat.py --variant {variant} --seed {args.seed}")

    print(f"\nDevice     : {device}")
    print(f"Policy     : CGAT-{variant}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Tests      : {args.tests}")
    print(f"Seed       : {args.seed if args.seed is not None else 'none'}")
    print(f"Episodes/pt: {args.n_eval_episodes}")
    print(f"Eval mode  : {'stochastic' if args.stochastic_eval else 'deterministic'}")

    policy = load_policy(checkpoint, cfg, device, variant=variant)

    print(f"\nLearned physics weights (from checkpoint):")
    print(f"  {_beta_summary(policy, variant)}")

    # Inference timing benchmark
    _timing_env = make_fixed_env(cfg, link_length=0.75, link_mass=1.05)
    _obs, _     = _timing_env.reset()
    _timing_env.close()
    N_TIMING = 1000
    for _ in range(50):
        select_eval_action(policy, _obs, device, stochastic=args.stochastic_eval)
    _t0 = time.perf_counter()
    for _ in range(N_TIMING):
        select_eval_action(policy, _obs, device, stochastic=args.stochastic_eval)
    _ms = (time.perf_counter() - _t0) * 1000
    print(f"\nInference timing ({N_TIMING} calls):")
    print(f"  Total   : {_ms:.2f} ms")
    print(f"  Per call: {_ms / N_TIMING:.4f} ms  "
          f"({1000 / (_ms / N_TIMING):.0f} inferences/sec)\n")

    os.makedirs("eval/plots", exist_ok=True)
    os.makedirs("eval/results", exist_ok=True)

    suffix = seed_suffix(args.seed)

    if "1" in args.tests:
        l_vals, l_rewards, l_lo, l_hi = run_test1(
            policy, cfg, device, args.n_eval_episodes, args.n_sweep_points,
            stochastic_eval=args.stochastic_eval)
        plot_1d(l_vals, l_rewards, l_lo, l_hi,
                "Link Length", "m", "eval/plots", variant=variant,
                file_suffix=suffix)
        np.savez(
            f"eval/results/cgat_{variant}_ppo{suffix}_test1_length.npz",
            values=l_vals, rewards=l_rewards,
            bounds=np.array([l_lo, l_hi]),
        )

    if "2" in args.tests:
        m_vals, m_rewards, m_lo, m_hi = run_test2(
            policy, cfg, device, args.n_eval_episodes, args.n_sweep_points,
            stochastic_eval=args.stochastic_eval)
        plot_1d(m_vals, m_rewards, m_lo, m_hi,
                "Link Mass", "kg", "eval/plots", variant=variant,
                file_suffix=suffix)
        np.savez(
            f"eval/results/cgat_{variant}_ppo{suffix}_test2_mass.npz",
            values=m_vals, rewards=m_rewards,
            bounds=np.array([m_lo, m_hi]),
        )

    if "2.5" in args.tests:
        n_vals, topo_rewards, topo_bounds = run_test25(
            policy, cfg, device, args.n_eval_episodes,
            stochastic_eval=args.stochastic_eval)
        plot_topology_sweep(
            n_vals, topo_rewards, topo_bounds,
            f"Topology Sweep | CGAT-{variant} PPO",
            f"eval/plots/cgat_{variant}_ppo{suffix}_topology_sweep.png",
        )
        np.savez(
            f"eval/results/cgat_{variant}_ppo{suffix}_test25_topology.npz",
            n_links=n_vals, rewards=topo_rewards,
            train_topology=np.array(topo_bounds),
        )

    if "3" in args.tests:
        l_v, m_v, r_cube, n_vals, l_bounds, m_bounds, topo_bounds = run_test3(
            policy, cfg, device, args.n_eval_episodes, args.n_grid,
            stochastic_eval=args.stochastic_eval,
            partial_path=args.heatmap_partial,
            partial_every=args.heatmap_partial_every,
            resume_partial=args.resume_heatmap)
        plot_topology_heatmaps(
            l_v, m_v, r_cube, n_vals, l_bounds, m_bounds, topo_bounds,
            f"OOD Heatmaps | CGAT-{variant} PPO",
            f"eval/plots/cgat_{variant}_ppo{suffix}_ood_heatmaps_by_topology.png",
            max_rewards=topology_reward_ceilings(cfg, n_vals),
        )
        np.savez(
            f"eval/results/cgat_{variant}_ppo{suffix}_test3.npz",
            lengths=l_v, masses=m_v, n_links=n_vals, rewards=r_cube,
            len_bounds=np.array(l_bounds), mass_bounds=np.array(m_bounds),
            train_topology=np.array(topo_bounds),
        )

    if "4" in args.tests:
        tasks, rewards = run_few_shot(
            policy, make_fixed_env, cfg, device,
            budgets=args.few_shot_budgets,
            n_tasks=args.few_shot_tasks,
            eval_episodes=args.n_eval_episodes,
            adapt_lr=args.few_shot_lr,
            adapt_epochs=args.few_shot_epochs,
            adapt_batch_size=cfg["ppo"]["mini_batch_size"],
            stochastic_eval=args.stochastic_eval,
        )
        budgets = sorted(set([0] + args.few_shot_budgets))
        result_path = f"eval/results/cgat_{variant}_ppo{suffix}_test4_fewshot.npz"
        save_few_shot_results(result_path, tasks, budgets, rewards)
        plot_few_shot(
            f"eval/plots/cgat_{variant}_ppo{suffix}_test4_fewshot.png",
            f"Few-Shot OOD Adaptation | CGAT-{variant} PPO",
            tasks, budgets, rewards,
        )
        summary = summarize_few_shot(budgets, rewards)
        print("\n[Test 4 summary]")
        for k, v in summary.items():
            print(f"  {k}: {v:.2f}")
        print(f"  Results saved -> {result_path}")

    print("Done.")


if __name__ == "__main__":
    main()
