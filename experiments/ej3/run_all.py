"""Run all ej3 sweep configs sequentially, capturing return codes and logs.

Usage (from repo root):

    python3 -m experiments.ej3.run_all

Each config is launched via ``subprocess.run(...)`` with ``check=False`` so a
divergent or failing run does not abort the rest of the sweep. ``run_all.log``
is written under ``results/ej3/training/`` with a per-config block.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Subset of configs to run by default (the 11 new sweep configs from T10 — the
# pre-event ablation_data_only run is excluded because it is already trained).
DEFAULT_CONFIGS = [
    "data_only_more.json",
    "arch_128_64_combined.json",
    "arch_256_128_combined.json",
    "arch_relu_128_combined.json",
    "opt_adam_lr_decay_combined.json",
    "opt_momentum_combined.json",
    "opt_adam_cosine_combined.json",
    "reg_l2_low_combined.json",
    "reg_l2_high_combined.json",
    "init_he_relu_combined.json",
    "init_xavier_tanh_combined.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-dir",
        default="experiments/ej3/configs",
        help="Directory containing the config JSON files",
    )
    parser.add_argument(
        "--log-path",
        default="results/ej3/training/run_all.log",
        help="Path to write the per-config log",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="If given, only run these basenames (e.g. arch_128_64_combined.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    selected = args.only if args.only else DEFAULT_CONFIGS

    results: list[tuple[str, int, float]] = []
    log_path.write_text(
        f"# ej3 sweep run_all started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"# {len(selected)} configs to run\n\n"
    )

    overall_start = time.time()
    for i, basename in enumerate(selected, start=1):
        cfg = configs_dir / basename
        if not cfg.exists():
            with log_path.open("a") as f:
                f.write(f"\n=== [{i}/{len(selected)}] {basename}: MISSING CONFIG ===\n")
            results.append((basename, -1, 0.0))
            continue

        header = (
            f"\n=== [{i}/{len(selected)}] {basename} "
            f"(start {time.strftime('%H:%M:%S')}) ===\n"
        )
        print(header.strip(), flush=True)
        with log_path.open("a") as f:
            f.write(header)

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "experiments.ej3.train", "--config", str(cfg)],
            check=False,
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0

        with log_path.open("a") as f:
            f.write(f"-- returncode={proc.returncode} elapsed={dt:.1f}s --\n")
            f.write("-- stdout --\n")
            f.write(proc.stdout or "")
            f.write("\n-- stderr --\n")
            f.write(proc.stderr or "")
            f.write("\n")

        results.append((basename, proc.returncode, dt))
        print(
            f"   -> rc={proc.returncode} elapsed={dt:.1f}s",
            flush=True,
        )

    overall_dt = time.time() - overall_start
    summary = ["\n## SUMMARY", f"Total elapsed: {overall_dt:.1f}s"]
    for basename, rc, dt in results:
        summary.append(f"  {basename}: rc={rc}, elapsed={dt:.1f}s")
    summary_str = "\n".join(summary) + "\n"
    print(summary_str, flush=True)
    with log_path.open("a") as f:
        f.write(summary_str)

    n_failed = sum(1 for _, rc, _ in results if rc != 0)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
