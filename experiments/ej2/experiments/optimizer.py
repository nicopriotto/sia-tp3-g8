"""Sweep optimizer for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="optimizer",
    parameter="optimizer + optimizer_params",
    allowed_changes=frozenset({"optimizer", "optimizer_params"}),
    description="EJ2 optimizer sweep.",
    variants=[
        Variant("sgd", "sgd", {"optimizer": "sgd", "optimizer_params": {}}),
        Variant(
            "momentum_0_9",
            "momentum(momentum=0.9)",
            {"optimizer": "momentum", "optimizer_params": {"momentum": 0.9}},
        ),
        Variant("adam", "adam", {"optimizer": "adam", "optimizer_params": {}}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

