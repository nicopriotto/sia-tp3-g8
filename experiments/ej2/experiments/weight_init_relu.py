"""Phase 2 weight-init sweep for EJ2, conditional on ReLU activation.

Runs after Phase 1 has fixed the activation/loss/training_method winners in
the base config. Compares only the inits that are theoretically meaningful
when the activation is ReLU: He (the recommended pairing) vs uniform (naive
baseline). Xavier is intentionally excluded — it is the right choice for
tanh/sigmoid, not for ReLU.

Usage from repo root:
    python3 -m experiments.ej2.experiments.weight_init_relu --seeds 42 123 2026
"""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="weight_init_relu",
    parameter="weight_init",
    allowed_changes=frozenset({"weight_init"}),
    description="EJ2 Phase 2 weight-init sweep (ReLU-conditional: he vs uniform).",
    variants=[
        Variant("he", "he", {"weight_init": "he"}),
        Variant("uniform", "uniform", {"weight_init": "uniform"}),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
