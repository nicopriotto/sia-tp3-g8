"""Sweep learning rate for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="learning_rate",
    parameter="learning_rate",
    allowed_changes=frozenset({"learning_rate"}),
    description="EJ2 learning-rate sweep.",
    variants=[
        Variant("lr_0_0001", "lr=0.0001", {"learning_rate": 0.0001}),
        Variant("lr_0_001", "lr=0.001", {"learning_rate": 0.001}),
        Variant("lr_0_003", "lr=0.003", {"learning_rate": 0.003}),
        Variant("lr_0_01", "lr=0.01", {"learning_rate": 0.01}),
        Variant("lr_0_03", "lr=0.03", {"learning_rate": 0.03}),
        Variant("lr_0_1", "lr=0.1", {"learning_rate": 0.1}),
        Variant("lr_0_3", "lr=0.3", {"learning_rate": 0.3}),
        Variant("lr_1_0", "lr=1.0", {"learning_rate": 1.0}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

