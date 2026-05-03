"""Run Exercise 2 experiment groups with fixed learning rates.

Usage from repo root:
    python3 -m experiments.ej2.run_all --group all --seeds 42 123 2026

This runner intentionally rejects learning-rate schedules. Every config must
use a constant learning rate for the whole training run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CONFIGS_DIR = Path("experiments/ej2/configs")
TRAINING_DIR = Path("results/ej2/training")
LOG_PATH = TRAINING_DIR / "run_all.log"
DEFAULT_SEEDS = [42, 123, 2026]

GROUP_CONFIGS = {
    "learning_rate": [
        "lr_0_0001_tanh_sgd.json",
        "lr_0_001_tanh_sgd.json",
        "lr_0_01_tanh_sgd.json",
        "lr_0_1_tanh_sgd.json",
        "lr_1_0_tanh_sgd.json",
        "lr_3_0_tanh_sgd.json",
    ],
    "architecture": [
        "arch_32_tanh_sgd.json",
        "arch_64_tanh_sgd.json",
        "arch_128_tanh_sgd.json",
        "arch_64_32_tanh_sgd.json",
    ],
    "optimizer": [
        "opt_sgd_tanh.json",
        "opt_momentum_tanh.json",
        "opt_adam_tanh_lr_0_001.json",
    ],
}

REQUIRED_HISTORY_KEYS = {
    "epoch",
    "train_loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "macro_f1",
    "val_macro_f1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=["all", *GROUP_CONFIGS],
        default="all",
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run for every config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when all expected artifacts already exist.",
    )
    parser.add_argument(
        "--with-per-run-plots",
        action="store_true",
        help="Let train.py create individual run plots. Aggregate plots are generated separately.",
    )
    return parser.parse_args()


def selected_configs(group: str) -> list[str]:
    if group == "all":
        names: list[str] = []
        for configs in GROUP_CONFIGS.values():
            for name in configs:
                if name not in names:
                    names.append(name)
        return names
    return GROUP_CONFIGS[group]


def config_name(config_path: Path) -> str:
    cfg = json.loads(config_path.read_text())
    return str(cfg["name"])


def validate_constant_lr(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())
    schedule = cfg.get("lr_schedule")
    if schedule not in (None, "", "none"):
        raise SystemExit(
            f"{config_path} uses lr_schedule={schedule!r}. "
            "Exercise 2 sweeps must use a fixed learning rate."
        )


def run_dir_for(config_path: Path, seed: int) -> Path:
    return TRAINING_DIR / f"{config_name(config_path)}__seed{seed}"


def _history_has_required_keys(history_path: Path) -> bool:
    if not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text())
    except json.JSONDecodeError:
        return False
    if not history:
        return False
    if history[0].get("epoch") != 0:
        return False
    keys = set().union(*(record.keys() for record in history))
    return REQUIRED_HISTORY_KEYS.issubset(keys)


def run_is_complete(run_dir: Path) -> bool:
    required_files = [
        run_dir / "config.json",
        run_dir / "history.json",
        run_dir / "evaluation.json",
        run_dir / "weights.npz",
    ]
    return all(path.exists() for path in required_files) and _history_has_required_keys(
        run_dir / "history.json"
    )


def append_log(text: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(text)


def main() -> int:
    args = parse_args()
    configs = selected_configs(args.group)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"# ej2 run_all started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"# group={args.group} seeds={args.seeds}\n"
        "# learning rates are constant; lr_schedule is rejected\n\n"
    )

    results: list[tuple[str, int, str, int, float]] = []
    overall_start = time.time()

    for config_basename in configs:
        config_path = CONFIGS_DIR / config_basename
        if not config_path.exists():
            append_log(f"\n=== {config_basename}: MISSING CONFIG ===\n")
            results.append((config_basename, -1, "missing_config", -1, 0.0))
            continue

        validate_constant_lr(config_path)

        for seed in args.seeds:
            run_dir = run_dir_for(config_path, seed)
            label = f"{config_basename} seed={seed}"

            if not args.force and run_is_complete(run_dir):
                print(f"[skip] {label} already complete")
                append_log(f"\n=== {label}: SKIP complete ===\n")
                results.append((config_basename, seed, "skipped", 0, 0.0))
                continue

            command = [
                sys.executable,
                "-m",
                "experiments.ej2.train",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            ]
            if not args.with_per_run_plots:
                command.append("--no-plots")

            header = f"\n=== {label} (start {time.strftime('%H:%M:%S')}) ===\n"
            print(header.strip(), flush=True)
            append_log(header)

            t0 = time.time()
            proc = subprocess.run(command, check=False, capture_output=True, text=True)
            elapsed = time.time() - t0

            append_log(f"-- command={' '.join(command)} --\n")
            append_log(f"-- returncode={proc.returncode} elapsed={elapsed:.1f}s --\n")
            append_log("-- stdout --\n")
            append_log(proc.stdout or "")
            append_log("\n-- stderr --\n")
            append_log(proc.stderr or "")
            append_log("\n")

            print(f"   -> rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
            results.append((config_basename, seed, "ran", proc.returncode, elapsed))

    overall_elapsed = time.time() - overall_start
    summary = ["\n## SUMMARY", f"Total elapsed: {overall_elapsed:.1f}s"]
    for config_basename, seed, status, rc, elapsed in results:
        summary.append(
            f"  {config_basename} seed={seed}: status={status}, rc={rc}, elapsed={elapsed:.1f}s"
        )
    summary_text = "\n".join(summary) + "\n"
    print(summary_text, flush=True)
    append_log(summary_text)

    failures = [row for row in results if row[3] != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
