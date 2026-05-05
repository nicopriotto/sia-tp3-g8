"""Phase 1 supplementary sweep for EJ2: fine-grained batch_size within mini-batch.

Complements `training_method` (online/mini/full categorical decision) by
exploring the quality-vs-speed tradeoff at finer resolution within the
mini-batch regime. The bs=32 cell is reused from training_method's
mini_batch_32 runs to avoid duplicate compute.

Usage from repo root:
    python3 -m experiments.ej2.experiments.batch_size --seeds 42 123 2026
"""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="batch_size",
    parameter="batch_size",
    allowed_changes=frozenset({"batch_size"}),
    description="EJ2 fine-grained batch_size sweep within mini-batch regime.",
    variants=[
        Variant("bs_8", "batch_size=8", {"batch_size": 8}),
        Variant("bs_16", "batch_size=16", {"batch_size": 16}),
        Variant("bs_32", "batch_size=32", {"batch_size": 32}),
        Variant("bs_64", "batch_size=64", {"batch_size": 64}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
