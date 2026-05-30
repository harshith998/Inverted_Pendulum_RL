#!/usr/bin/env python3.12
"""Aggregate seed-specific eval result files into mean/std artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


SEED_RE = re.compile(r"_seed(\d+)")


def _group_key(path: Path) -> str:
    return SEED_RE.sub("", path.stem)


def _seed(path: Path) -> int | None:
    m = SEED_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _summary_metrics(data: dict) -> dict:
    rewards = data.get("rewards_mean")
    if rewards is None:
        return {}
    rewards = np.asarray(rewards, dtype=float)
    out = {"reward_mean": float(np.nanmean(rewards))}

    if "test4_fewshot" in data.get("_group", "") and "budgets" in data:
        budgets = np.asarray(data["budgets"], dtype=float)
        curve = rewards.mean(axis=0) if rewards.ndim == 2 else rewards
        out["zero_shot"] = float(curve[0])
        out["final"] = float(curve[-1])
        out["gain"] = float(curve[-1] - curve[0])
        out["auc"] = float(np.trapezoid(curve, budgets) /
                           max(1.0, budgets[-1] - budgets[0]))

    if "test25_topology" in data.get("_group", "") and "n_links" in data:
        n_links = np.asarray(data["n_links"], dtype=int)
        train_topology = np.asarray(data.get("train_topology", []), dtype=int)
        if train_topology.size == 2:
            mask = (n_links < train_topology[0]) | (n_links > train_topology[1])
            if np.any(mask):
                out["topology_ood_mean"] = float(np.nanmean(rewards[mask]))

    return out


def aggregate_results(results_dir: str | Path = "eval/results",
                      seeds: list[int] | None = None) -> list[dict]:
    results_dir = Path(results_dir)
    seed_filter = set(seeds or [])
    files = sorted(p for p in results_dir.glob("*.npz") if _seed(p) is not None)

    groups: dict[str, list[Path]] = {}
    for path in files:
        seed = _seed(path)
        if seed_filter and seed not in seed_filter:
            continue
        groups.setdefault(_group_key(path), []).append(path)

    rows = []
    for group, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        loaded = [np.load(p) for p in paths]
        if not all("rewards" in d.files for d in loaded):
            continue

        rewards = np.stack([d["rewards"].astype(float) for d in loaded], axis=0)
        first = loaded[0]
        out = {
            "seeds": np.array([_seed(p) for p in paths], dtype=np.int64),
            "rewards_mean": rewards.mean(axis=0),
            "rewards_std": rewards.std(axis=0),
        }
        for key in first.files:
            if key == "rewards":
                continue
            vals = [d[key] for d in loaded]
            if all(np.array_equal(vals[0], v) for v in vals[1:]):
                out[key] = vals[0]

        out_path = results_dir / f"{group}_seedavg.npz"
        np.savez(out_path, **out)

        summary_data = dict(out)
        summary_data["_group"] = group
        row = {"result": group, "n_seeds": len(paths), "path": str(out_path)}
        row.update(_summary_metrics(summary_data))
        rows.append(row)

    if rows:
        csv_path = results_dir / "seed_average_summary.csv"
        fieldnames = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Seed-average summary saved -> {csv_path}")
    else:
        print("No multi-seed result groups found to aggregate.")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="eval/results")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    aggregate_results(args.results_dir, args.seeds)


if __name__ == "__main__":
    main()

