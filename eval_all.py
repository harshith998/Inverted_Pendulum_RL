#!/usr/bin/env python3.12
"""
Run every OOD evaluation job sequentially.

Usage
-----
  python3.12 eval_all.py
  python3.12 eval_all.py --config configs/default.yaml
  python3.12 eval_all.py --skip dqn_mlp cgat_perc
  python3.12 eval_all.py --only ppo_gnn_mpnn cgat_base
  python3.12 eval_all.py --seeds 0 1 2
  python3.12 eval_all.py --tests 1 2 2.5 3

Jobs (in order)
---------------
  ppo_mlp
  ppo_gnn_mpnn
  ppo_gnn_transformer
  dqn_mlp
  dqn_gnn
  cgat_base
  cgat_perhead
  cgat_directional
  cgat_gravity
  cgat_perc
  cgat_no_physics
  cgat_shuffled
  lqr_oracle
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time


JOBS = [
    {"name": "ppo_mlp",             "cmd": ["python3.12", "-u", "eval/eval_ppo.py",  "--policy", "mlp"]},
    {"name": "ppo_gnn_mpnn",        "cmd": ["python3.12", "-u", "eval/eval_ppo.py",  "--policy", "gnn_mpnn"]},
    {"name": "ppo_gnn_transformer", "cmd": ["python3.12", "-u", "eval/eval_ppo.py",  "--policy", "gnn_transformer"]},
    {"name": "dqn_mlp",             "cmd": ["python3.12", "-u", "eval/eval_dqn.py",  "--policy", "mlp"]},
    {"name": "dqn_gnn",             "cmd": ["python3.12", "-u", "eval/eval_dqn.py",  "--policy", "gnn"]},
    {"name": "cgat_base",           "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "base"]},
    {"name": "cgat_perhead",        "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "perhead"]},
    {"name": "cgat_directional",    "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "directional"]},
    {"name": "cgat_gravity",        "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "gravity"]},
    {"name": "cgat_perc",           "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "perc"]},
    {"name": "cgat_no_physics",     "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "no_physics"]},
    {"name": "cgat_shuffled",       "cmd": ["python3.12", "-u", "eval/eval_cgat.py", "--variant", "shuffled"]},
    {"name": "lqr_oracle",          "cmd": ["python3.12", "-u", "eval/eval_lqr.py"]},
]


def _print_header(msg: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    cache_root = os.path.join(tempfile.gettempdir(), "inverted_pendulum_eval_all")
    mpl_config = os.path.join(cache_root, "matplotlib")
    xdg_cache = os.path.join(cache_root, "cache")
    os.makedirs(mpl_config, exist_ok=True)
    os.makedirs(xdg_cache, exist_ok=True)
    env["MPLCONFIGDIR"] = mpl_config
    env["XDG_CACHE_HOME"] = xdg_cache
    return env


def _should_echo(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    prefixes = (
        "Device", "Policy", "Checkpoint", "Config", "Tests", "Seed", "Episodes/pt",
        "Inference timing", "Total", "Per call", "Learned physics",
        "Plot saved", "Compare plot saved", "Done",
    )
    if stripped.startswith(prefixes):
        return True
    if stripped.startswith("[Test"):
        return True
    if stripped.startswith("[") and "/" in stripped:
        return True
    if "Checkpoint not found" in stripped or stripped.startswith("Traceback"):
        return True
    return False


def _build_cmd(job: dict, args, seed: int) -> list[str] | None:
    cmd = job["cmd"] + ["--config", args.config]
    tests = list(args.tests)
    if job["name"].startswith("dqn_"):
        tests = [t for t in tests if t != "4"]
        if not tests:
            return None
    if job["name"] == "lqr_oracle":
        tests = [t for t in tests if t in ("1", "2", "2.5", "3")]
        if not tests:
            return None
    cmd += ["--tests"] + [str(t) for t in tests]
    cmd += ["--seed", str(seed)]
    if args.n_eval_episodes is not None:
        cmd += ["--n_eval_episodes", str(args.n_eval_episodes)]
    if args.n_sweep_points is not None:
        cmd += ["--n_sweep_points", str(args.n_sweep_points)]
    if args.n_grid is not None:
        cmd += ["--n_grid", str(args.n_grid)]
    if not job["name"].startswith("dqn_") and job["name"] != "lqr_oracle":
        if args.stochastic_eval:
            cmd += ["--stochastic_eval"]
        cmd += ["--few_shot_budgets"] + [str(b) for b in args.few_shot_budgets]
        cmd += ["--few_shot_tasks", str(args.few_shot_tasks)]
        cmd += ["--few_shot_epochs", str(args.few_shot_epochs)]
        cmd += ["--few_shot_lr", str(args.few_shot_lr)]
    return cmd


def _cmd_arg(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    idx = cmd.index(flag)
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def _checkpoint_path(job: dict, seed: int) -> str | None:
    name = job["name"]
    if name == "lqr_oracle":
        return None

    if name.startswith("ppo_"):
        policy = _cmd_arg(job["cmd"], "--policy")
        return f"checkpoints/{policy}_ppo_seed{seed}_best.pt"

    if name.startswith("dqn_"):
        policy = _cmd_arg(job["cmd"], "--policy")
        return f"checkpoints/{policy}_dqn_seed{seed}_best.pt"

    if name.startswith("cgat_"):
        variant = _cmd_arg(job["cmd"], "--variant")
        return f"checkpoints/cgat_{variant}_ppo_seed{seed}_best.pt"

    return None


def _missing_checkpoint_message(job: dict, seed: int, seeded: str) -> str:
    name = job["name"]
    if name.startswith("ppo_"):
        policy = _cmd_arg(job["cmd"], "--policy")
        return f"python3.12 training/train_ppo.py --policy {policy} --seed {seed}"
    if name.startswith("dqn_"):
        policy = _cmd_arg(job["cmd"], "--policy")
        return f"python3.12 training/train_dqn.py --policy {policy} --seed {seed}"
    if name.startswith("cgat_"):
        variant = _cmd_arg(job["cmd"], "--variant")
        return f"python3.12 training/train_cgat.py --variant {variant} --seed {seed}"
    return f"missing checkpoint: {seeded}"


def _checkpoint_available(job: dict, seed: int) -> tuple[bool, str | None]:
    seeded = _checkpoint_path(job, seed)
    if seeded is None:
        return True, None
    if os.path.exists(seeded):
        return True, None
    return False, seeded


def run_job(job: dict, args, seed: int, job_idx: int, n_jobs: int) -> str:
    name = job["name"]
    cmd = _build_cmd(job, args, seed)
    if cmd is None:
        _print_header(f"[{job_idx}/{n_jobs}]  {name}")
        if name == "lqr_oracle":
            print("  Skipped: LQR oracle only supports Tests 1-3.")
        else:
            print("  Skipped: Test 4 is not implemented for DQN eval.")
        return "skip"

    _print_header(f"[{job_idx}/{n_jobs}]  {name}  seed={seed}")

    available, detail = _checkpoint_available(job, seed)
    if not available:
        print(f"  Skipped: checkpoint not found: {detail}")
        print(f"  Train it with: {_missing_checkpoint_message(job, seed, detail)}")
        return "skip"
    print(f"  Command : {' '.join(cmd)}")

    log_path = os.path.join(args.log_dir, f"{name}_seed{seed}.log")
    print(f"  Log     : {log_path}\n")

    if args.dry_run:
        return "ok"

    t_start = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
        )

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if _should_echo(line):
                print(f"  {line.rstrip()}", flush=True)

        proc.wait()

    elapsed = time.time() - t_start
    ok = proc.returncode == 0
    status = "DONE" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n  {name}: {status} in {elapsed / 60:.1f} min  "
          f"(log -> {log_path})")
    return "ok" if ok else "fail"


def main():
    parser = argparse.ArgumentParser(
        description="Run all OOD eval jobs sequentially.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip", nargs="*", default=[], metavar="JOB",
        help="Job names to skip")
    parser.add_argument("--only", nargs="*", default=None, metavar="JOB",
        help="Run only these jobs (overrides --skip)")
    parser.add_argument("--tests", nargs="+",
        choices=["1", "2", "2.5", "3", "4"],
        default=["1", "2", "2.5", "3", "4"],
        help="Eval stages: 1=length, 2=mass, 2.5=topology, 3=topology heatmaps, 4=few-shot")
    parser.add_argument("--n_eval_episodes", type=int, default=None,
        help="Override episodes per eval point")
    parser.add_argument("--n_sweep_points", type=int, default=None,
        help="Override points per 1D sweep")
    parser.add_argument("--n_grid", type=int, default=None,
        help="Override heatmap grid size per axis")
    parser.add_argument("--stochastic_eval", action="store_true",
        help="Use stochastic PPO/CGAT eval instead of deterministic mean actions")
    parser.add_argument("--few_shot_budgets", nargs="+", type=int,
        default=[0, 1, 5, 10, 25],
        help="Fine-tuning episode budgets for Test 4")
    parser.add_argument("--few_shot_tasks", type=int, default=4,
        help="Number of far-OOD corner tasks for Test 4")
    parser.add_argument("--few_shot_epochs", type=int, default=4,
        help="PPO epochs per few-shot adaptation batch")
    parser.add_argument("--few_shot_lr", type=float, default=3e-5,
        help="Learning rate for Test 4 adaptation")
    parser.add_argument("--log_dir", default="logs/eval",
        help="Directory for per-job eval logs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Seeds to evaluate. Default: 0 1 2")
    parser.add_argument("--dry-run", action="store_true",
        help="Print commands without running them")
    parser.add_argument("--no_aggregate", action="store_true",
        help="Do not aggregate seed result files after successful eval")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"Config not found: {args.config}")

    os.makedirs(args.log_dir, exist_ok=True)

    valid_names = {j["name"] for j in JOBS}
    bad = (set(args.only or []) | set(args.skip)) - valid_names
    if bad:
        print(f"[warn] Unknown job names: {bad}")
        print(f"       Valid names: {sorted(valid_names)}\n")

    jobs = JOBS
    if args.only is not None:
        jobs = [j for j in jobs if j["name"] in args.only]
    else:
        jobs = [j for j in jobs if j["name"] not in args.skip]

    if not jobs:
        sys.exit("No jobs to run after filtering.")

    expanded_jobs = [(job, seed) for seed in args.seeds for job in jobs]

    print(f"\neval_all  |  {len(expanded_jobs)}/{len(JOBS) * len(args.seeds)} runs")
    print(f"Config    : {args.config}")
    print(f"Tests     : {args.tests}")
    print(f"Seeds     : {args.seeds}")
    eval_size = []
    eval_size.append(
        f"{args.n_sweep_points} sweep pts"
        if args.n_sweep_points is not None else "script-default sweep pts"
    )
    eval_size.append(
        f"{args.n_grid}x{args.n_grid} heatmap"
        if args.n_grid is not None else "script-default heatmap"
    )
    eval_size.append(
        f"{args.n_eval_episodes} eps/pt"
        if args.n_eval_episodes is not None else "script-default eps/pt"
    )
    print(f"Eval size : {', '.join(eval_size)}")
    print(f"Logs      : {args.log_dir}/")
    if args.dry_run:
        print("Mode      : dry run")
    print("\nJobs to run:")
    for j, seed in expanded_jobs:
        print(f"  - {j['name']}  seed={seed}")

    results = {}
    t_all = time.time()
    for i, (job, seed) in enumerate(expanded_jobs, 1):
        status = run_job(job, args, seed, i, len(expanded_jobs))
        results[f"{job['name']}_seed{seed}"] = status

    total_elapsed = time.time() - t_all
    _print_header(f"Summary  ({total_elapsed / 60:.1f} min total)")
    labels = {"ok": "OK", "fail": "FAIL", "skip": "SKIP"}
    for name, status in results.items():
        print(f"  {labels[status]:<4} {name}")

    failed = [n for n, status in results.items() if status == "fail"]
    skipped = [n for n, status in results.items() if status == "skip"]
    if failed:
        print(f"\n  {len(failed)} job(s) failed: {failed}")
        sys.exit(1)
    if skipped:
        print(f"\n  {len(skipped)} job(s) skipped because required checkpoints/features were missing.")
        print("  No eval jobs failed.")
    else:
        print(f"\n  All {len(results)} eval job(s) completed successfully.")

    if skipped:
        print("  Seed aggregation skipped because not every requested seed ran.")
        return

    if not args.dry_run and not args.no_aggregate and len(args.seeds) > 1:
        from eval.aggregate_seeds import aggregate_results
        aggregate_results("eval/results", args.seeds)


if __name__ == "__main__":
    main()
