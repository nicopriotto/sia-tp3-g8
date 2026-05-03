"""Sweep online, mini-batch and full-batch training for EJ2."""
from __future__ import annotations

from .common import FULL_TRAIN_BATCH_SIZE, SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="training_method",
    parameter="batch_size",
    allowed_changes=frozenset({"batch_size"}),
    description="EJ2 training-method sweep via batch_size.",
    variants=[
        Variant("online", "online(batch_size=None)", {"batch_size": None}),
        Variant("mini_batch_32", "mini-batch(batch_size=32)", {"batch_size": 32}),
        Variant("batch_full", "batch(full train split)", {"batch_size": FULL_TRAIN_BATCH_SIZE}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

