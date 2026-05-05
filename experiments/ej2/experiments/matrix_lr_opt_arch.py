"""Phase 3 matrix sweep for EJ2: learning_rate × optimizer × architecture.

Runs after Phase 1 and Phase 2 have fixed the activation/loss/training_method/
weight_init winners in the base config. Explores the cartesian product of the
three axes that interact most strongly and warrant joint analysis.

The result is presented as 3 heatmap slices (one per optimizer) with rows=
architecture and columns=learning_rate, plus the standard summary/aggregate
CSVs and per-variant epoch curves.

Usage from repo root:
    # Full sweep (144 runs, ~3 hrs sequential, single process)
    python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --seeds 42 123 2026

    # One slice (= one optimizer's 4×4 sub-matrix, 16 cells × 3 seeds = 48 runs)
    python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --optimizers adam --seeds 42 123 2026

    # Final consolidation after slice runs (regenerates full summary + all heatmaps)
    python3 -m experiments.ej2.experiments.matrix_lr_opt_arch --seeds 42 123 2026

Output: results/ej2/comparasion/matrix_lr_opt_arch/
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Sequence

from .common import SweepSpec, Variant, add_common_args, run_sweep


LR_VALUES: list[float] = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1]

OPTIMIZERS: list[tuple[str, dict]] = [
    ("sgd", {}),
    ("momentum", {"momentum": 0.9}),
    ("adam", {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}),
]

ARCHITECTURES: list[list[int]] = [
    [784, 64, 10],        # ~50k  params · 1H · baseline pequeño
    [784, 128, 10],       # ~101k params · 1H · baseline (config base)
    [784, 256, 10],       # ~203k params · 1H · ancho
    [784, 128, 64, 10],   # ~109k params · 2H · param-matched con [128]
    [784, 256, 128, 10],  # ~218k params · 2H · param-matched con [256]
]


def _lr_slug(lr: float) -> str:
    return f"{lr:.6f}".rstrip("0").rstrip(".").replace(".", "_") or "0"


def _arch_slug(arch: list[int]) -> str:
    return "x".join(str(n) for n in arch[1:-1])


def _build_variants() -> list[Variant]:
    variants: list[Variant] = []
    for lr, (opt_name, opt_params), arch in product(LR_VALUES, OPTIMIZERS, ARCHITECTURES):
        slug = f"lr_{_lr_slug(lr)}__{opt_name}__h{_arch_slug(arch)}"
        label = f"lr={lr:g} | {opt_name} | h=[{_arch_slug(arch).replace('x', ',')}]"
        variants.append(
            Variant(
                slug,
                label,
                {
                    "learning_rate": lr,
                    "optimizer": opt_name,
                    "optimizer_params": dict(opt_params),
                    "architecture": list(arch),
                },
            )
        )
    return variants


SPEC = SweepSpec(
    name="matrix_lr_opt_arch",
    parameter="learning_rate × optimizer × architecture",
    allowed_changes=frozenset(
        {"learning_rate", "optimizer", "optimizer_params", "architecture"}
    ),
    description="EJ2 Phase 3 matrix sweep (LR × optimizer × arch).",
    variants=_build_variants(),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SPEC.description)
    add_common_args(parser)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=[name for name, _ in OPTIMIZERS],
        default=None,
        metavar="OPT",
        help="Run only these optimizer slices (default: all 3). Use to batch the matrix one slice at a time.",
    )
    args = parser.parse_args(argv)

    optimizers_to_run = args.optimizers or [name for name, _ in OPTIMIZERS]

    if args.optimizers:
        filtered_variants = [
            v for v in SPEC.variants if v.overrides["optimizer"] in optimizers_to_run
        ]
        train_spec = SweepSpec(
            name=SPEC.name,
            parameter=SPEC.parameter,
            allowed_changes=SPEC.allowed_changes,
            description=SPEC.description,
            variants=filtered_variants,
        )
    else:
        train_spec = SPEC

    result = run_sweep(
        train_spec,
        seeds=args.seeds,
        force=args.force,
        no_train=args.no_train,
        base_config=Path(args.base_config),
    )
    if not result.dry_run:
        from .plotting import plot_matrix_heatmaps

        plot_matrix_heatmaps(
            result.output_dir,
            optimizers=optimizers_to_run,
            lr_values=LR_VALUES,
            architectures=ARCHITECTURES,
            metric="val_macro_f1",
        )
        plot_matrix_heatmaps(
            result.output_dir,
            optimizers=optimizers_to_run,
            lr_values=LR_VALUES,
            architectures=ARCHITECTURES,
            metric="best_val_macro_f1",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
