"""Sweep maximum number of epochs for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="epochs",
    parameter="epochs",
    allowed_changes=frozenset({"epochs"}),
    description="EJ2 epochs sweep.",
    variants=[
        Variant("epochs_10", "epochs=10", {"epochs": 10}),
        Variant("epochs_30", "epochs=30", {"epochs": 30}),
        Variant("epochs_50", "epochs=50", {"epochs": 50}),
        Variant("epochs_75", "epochs=75", {"epochs": 75}),
        Variant("epochs_100", "epochs=100", {"epochs": 100}),
        Variant("epochs_150", "epochs=150", {"epochs": 150}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

