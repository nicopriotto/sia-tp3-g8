"""Sweep loss function for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="loss",
    parameter="loss",
    allowed_changes=frozenset({"loss"}),
    description="EJ2 loss-function sweep.",
    variants=[
        Variant(
            "categorical_cross_entropy",
            "categorical_cross_entropy",
            {"loss": "categorical_cross_entropy"},
        ),
        Variant("mse", "mse", {"loss": "mse"}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

