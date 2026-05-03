"""Sweep MLP architecture for EJ2."""
from __future__ import annotations

from .common import SweepSpec, Variant, run_sweep_from_cli


SPEC = SweepSpec(
    name="architecture",
    parameter="architecture",
    allowed_changes=frozenset({"architecture"}),
    description="EJ2 architecture sweep.",
    variants=[
        Variant("h_16", "[784,16,10]", {"architecture": [784, 16, 10]}),
        Variant("h_32", "[784,32,10]", {"architecture": [784, 32, 10]}),
        Variant("h_64", "[784,64,10]", {"architecture": [784, 64, 10]}),
        Variant("h_128", "[784,128,10]", {"architecture": [784, 128, 10]}),
        Variant("h_256", "[784,256,10]", {"architecture": [784, 256, 10]}),
        Variant("h_64_32", "[784,64,32,10]", {"architecture": [784, 64, 32, 10]}),
        Variant("h_128_64", "[784,128,64,10]", {"architecture": [784, 128, 64, 10]}),
        Variant("h_256_128", "[784,256,128,10]", {"architecture": [784, 256, 128, 10]}),
        Variant(
            "h_128_64_32",
            "[784,128,64,32,10]",
            {"architecture": [784, 128, 64, 32, 10]},
        ),
    ],
)


def main() -> int:
    return run_sweep_from_cli(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())

