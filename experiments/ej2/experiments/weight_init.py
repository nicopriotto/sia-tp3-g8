"""Sweep MLP weight initialization for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="weight_init",
    parameter="weight_init",
    allowed_changes=frozenset({"weight_init"}),
    description="EJ2 weight-initialization sweep.",
    variants=[
        Variant("uniform", "uniform", {"weight_init": "uniform"}),
        Variant("xavier", "xavier", {"weight_init": "xavier"}),
        Variant("he", "he", {"weight_init": "he"}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

