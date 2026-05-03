"""Run all new EJ2 one-parameter sweeps.

Usage from repo root:
    python3 -m experiments.ej2.experiments.run_all --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import activation, architecture, epochs, learning_rate, loss, optimizer, training_method, weight_init
from .common import DEFAULT_SEEDS, SweepSpec, run_sweep, validate_sweep_outputs


SPECS: list[SweepSpec] = [
    learning_rate.SPEC,
    optimizer.SPEC,
    architecture.SPEC,
    activation.SPEC,
    loss.SPEC,
    training_method.SPEC,
    epochs.SPEC,
    weight_init.SPEC,
]

SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EJ2 new comparison sweeps.")
    parser.add_argument(
        "--group",
        choices=["all", *SPEC_BY_NAME],
        default="all",
        help="Sweep group to run. Defaults to all groups.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run for every variant.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when expected artifacts already exist.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only generate effective configs and dry-run route summaries.",
    )
    parser.add_argument(
        "--base-config",
        default="experiments/ej2/config_base.json",
        help="Path to the base JSON config.",
    )
    return parser.parse_args()


def selected_specs(group: str) -> list[SweepSpec]:
    return SPECS if group == "all" else [SPEC_BY_NAME[group]]


def main() -> int:
    args = parse_args()
    specs = selected_specs(args.group)
    for spec in specs:
        run_sweep(
            spec,
            seeds=args.seeds,
            force=args.force,
            no_train=args.no_train,
            base_config=Path(args.base_config),
        )

    if not args.no_train:
        missing: list[str] = []
        for spec in specs:
            missing.extend(validate_sweep_outputs(spec, args.seeds))
        if missing:
            raise SystemExit("Incomplete EJ2 new sweep outputs: " + ", ".join(missing))

    print(f"[run_all] completed {len(specs)} group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

