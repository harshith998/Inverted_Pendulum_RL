# One-at-a-time hyperparameter ablation for PPO.
#
# Usage:
#   python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep lr
#   python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep all
#
# Each sweep fixes all other hyperparams at their default.yaml values and
# varies only the target parameter across a small set of candidates.
# Runs are shorter than full training (--steps, default 1_000_000) so the
# whole ablation finishes in reasonable wall-clock time.
#
# Outputs:
#   checkpoints/ablation/{policy}_{sweep}_{value}_curve.npz   — raw log data
#   checkpoints/ablation/{policy}_{sweep}_comparison.png      — comparison plot

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml

from training.train_ppo import train


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# Each entry: key in ppo config → list of candidate values to try.
# Defaults come from default.yaml; only the swept key is changed per run.
SWEEPS = {
    "lr": {
        "values": [3e-4, 1e-4, 3e-5],
        "label":  "Learning Rate",
        "fmt":    lambda v: f"lr={v:.0e}",
    },
    "n_envs": {
        "values": [2, 4, 8],
        "label":  "Parallel Envs",
        "fmt":    lambda v: f"n_envs={v}",
    },
    "rollout_steps": {
        "values": [2048, 4096, 8192],
        "label":  "Rollout Steps (per env)",
        "fmt":    lambda v: f"T={v}",
    },
    "gae_lambda": {
        "values": [0.90, 0.95, 0.99],
        "label":  "GAE Lambda",
        "fmt":    lambda v: f"λ={v}",
    },
    "entropy_coef": {
        "values": [0.005, 0.02, 0.05],
        "label":  "Entropy Coefficient",
        "fmt":    lambda v: f"ent={v}",
    },
    "n_epochs": {
        "values": [3, 5, 10],
        "label":  "PPO Epochs per Update",
        "fmt":    lambda v: f"epochs={v}",
    },
}


# ---------------------------------------------------------------------------
# Run one sweep
# ---------------------------------------------------------------------------

def run_sweep(base_cfg: dict, policy_name: str, sweep_key: str,
              ablation_steps: int, out_dir: str) -> dict:
    """
    Run all candidate values for a single sweep key.
    Returns {label_str: (steps, rewards)} for plotting.
    """
    sweep_def = SWEEPS[sweep_key]
    results   = {}

    for value in sweep_def["values"]:
        label = sweep_def["fmt"](value)
        cache_path = os.path.join(out_dir, f"{policy_name}_{sweep_key}_{label}.npz")

        if os.path.exists(cache_path):
            print(f"  [cached] {label}")
            d = np.load(cache_path)
            results[label] = (d["steps"].tolist(), d["rewards"].tolist())
            continue

        print(f"\n{'='*60}")
        print(f"  Sweep: {sweep_key}  |  Value: {value}  ({label})")
        print(f"{'='*60}")

        # Deep-copy config and patch the swept key
        cfg = copy.deepcopy(base_cfg)
        cfg["ppo"][sweep_key]    = value
        cfg["ppo"]["total_steps"] = ablation_steps

        steps, rewards, _, _ = train(cfg, policy_name, plot=False)

        np.savez(cache_path, steps=np.array(steps), rewards=np.array(rewards))
        results[label] = (steps, rewards)

    return results


# ---------------------------------------------------------------------------
# Comparison plot for one sweep
# ---------------------------------------------------------------------------

def plot_sweep(results: dict, sweep_key: str, policy_name: str,
               out_dir: str):
    sweep_def = SWEEPS[sweep_key]
    colors    = ["steelblue", "tomato", "seagreen", "goldenrod", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, (steps, rewards)), color in zip(results.items(), colors):
        ax.plot(steps, rewards, label=label, color=color, linewidth=1.8)

    ax.set_title(f"PPO Ablation — {policy_name.upper()} — {sweep_def['label']}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward (last 20 eps)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, f"{policy_name}_{sweep_key}_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  plot saved → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict, policy_name: str):
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY — {policy_name.upper()}")
    print(f"{'='*60}")
    print(f"  {'Sweep':<18} {'Config':<20} {'Final Reward':>12}")
    print(f"  {'-'*52}")
    for sweep_key, results in all_results.items():
        for label, (steps, rewards) in results.items():
            final_r = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
            print(f"  {sweep_key:<18} {label:<20} {final_r:>12.1f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        choices=["gnn_mpnn", "gnn_transformer", "mlp"],
                        default="gnn_mpnn",
                        help="Which PPO policy architecture to ablate")
    parser.add_argument("--sweep",
                        choices=list(SWEEPS.keys()) + ["all"],
                        default="lr",
                        help="Which hyperparameter to sweep (or 'all' for every sweep)")
    parser.add_argument("--steps",
                        type=int,
                        default=1_000_000,
                        help="Training steps per ablation run (shorter than full training)")
    parser.add_argument("--config",
                        default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    out_dir = os.path.join("checkpoints", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    sweep_keys = list(SWEEPS.keys()) if args.sweep == "all" else [args.sweep]
    all_results = {}

    for sweep_key in sweep_keys:
        print(f"\n>>> Starting sweep: {sweep_key}")
        results = run_sweep(base_cfg, args.policy, sweep_key, args.steps, out_dir)
        all_results[sweep_key] = results
        plot_sweep(results, sweep_key, args.policy, out_dir)

    print_summary(all_results, args.policy)
