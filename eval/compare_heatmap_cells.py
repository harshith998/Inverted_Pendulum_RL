"""Compare saved Test-3 heatmap cells across result files.

Useful for diagnosing exactly where a learned checkpoint gains or loses.
Result files must be produced by eval/eval_cgat.py or eval/eval_lqr.py.
"""

import argparse
from pathlib import Path

import numpy as np


def load_result(path: str):
    data = np.load(path)
    rewards_key = "rewards" if "rewards" in data.files else "reward_cube"
    rewards = data[rewards_key].astype(float)
    if rewards.ndim == 3:
        rewards = rewards[0]
    lengths = data["lengths"]
    masses = data["masses"]
    return lengths, masses, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    lengths, masses, cand = load_result(args.candidate)
    base_lengths, base_masses, base = load_result(args.baseline)
    if cand.shape != base.shape or not np.allclose(lengths, base_lengths) or not np.allclose(masses, base_masses):
        base_on_cand = np.empty_like(cand)
        for j, mass in enumerate(masses):
            bj = int(np.argmin(np.abs(base_masses - mass)))
            for i, length in enumerate(lengths):
                bi = int(np.argmin(np.abs(base_lengths - length)))
                base_on_cand[j, i] = base[bj, bi]
        base = base_on_cand

    ref = None
    if args.reference:
        ref_lengths, ref_masses, ref = load_result(args.reference)
        if cand.shape != ref.shape or not np.allclose(lengths, ref_lengths) or not np.allclose(masses, ref_masses):
            ref_on_cand = np.empty_like(cand)
            for j, mass in enumerate(masses):
                rj = int(np.argmin(np.abs(ref_masses - mass)))
                for i, length in enumerate(lengths):
                    ri = int(np.argmin(np.abs(ref_lengths - length)))
                    ref_on_cand[j, i] = ref[rj, ri]
            ref = ref_on_cand

    delta = cand - base
    cells = []
    for j, mass in enumerate(masses):
        for i, length in enumerate(lengths):
            item = {
                "length": float(length),
                "mass": float(mass),
                "candidate": float(cand[j, i]),
                "baseline": float(base[j, i]),
                "delta": float(delta[j, i]),
            }
            if ref is not None:
                item["reference"] = float(ref[j, i])
                item["ref_gap"] = float(ref[j, i] - cand[j, i])
            cells.append(item)

    print(f"candidate: {Path(args.candidate).name}")
    print(f"baseline : {Path(args.baseline).name}")
    if args.reference:
        print(f"reference: {Path(args.reference).name}")

    def summarize(name, arr):
        flat = arr[np.isfinite(arr)]
        print(
            f"{name:10s} mean={flat.mean():8.2f} p10={np.percentile(flat,10):8.2f} "
            f"p25={np.percentile(flat,25):8.2f} med={np.median(flat):8.2f} "
            f"p90={np.percentile(flat,90):8.2f}"
        )

    summarize("candidate", cand)
    summarize("baseline", base)
    if ref is not None:
        summarize("reference", ref)

    print("\nWorst candidate cells:")
    for item in sorted(cells, key=lambda x: x["candidate"])[:args.top]:
        extra = f" ref={item['reference']:8.2f} ref_gap={item['ref_gap']:8.2f}" if ref is not None else ""
        print(
            f"L={item['length']:.3f} M={item['mass']:.3f} "
            f"cand={item['candidate']:8.2f} base={item['baseline']:8.2f} "
            f"delta={item['delta']:8.2f}{extra}"
        )

    print("\nLargest losses vs baseline:")
    for item in sorted(cells, key=lambda x: x["delta"])[:args.top]:
        extra = f" ref={item['reference']:8.2f} ref_gap={item['ref_gap']:8.2f}" if ref is not None else ""
        print(
            f"L={item['length']:.3f} M={item['mass']:.3f} "
            f"cand={item['candidate']:8.2f} base={item['baseline']:8.2f} "
            f"delta={item['delta']:8.2f}{extra}"
        )


if __name__ == "__main__":
    main()
