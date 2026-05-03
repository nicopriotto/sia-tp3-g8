"""Sweep hidden activation for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="activation",
    parameter="activation",
    allowed_changes=frozenset({"activation"}),
    description="EJ2 activation sweep.",
    variants=[
        Variant("tanh", "tanh", {"activation": "tanh"}),
        Variant("sigmoid", "sigmoid", {"activation": "sigmoid"}),
        Variant("relu", "relu", {"activation": "relu"}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

