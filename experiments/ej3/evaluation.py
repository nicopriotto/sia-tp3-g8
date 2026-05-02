"""Re-export of multiclass evaluation utilities from ``experiments._common``.

Kept as a thin shim so ej3 scripts can ``from experiments.ej3.evaluation
import evaluate_multiclass`` symmetrically with ej2.
"""
from experiments._common.evaluation import (  # noqa: F401
    categorical_cross_entropy,
    classification_report_per_class,
    evaluate_multiclass,
    macro_average,
    save_evaluation,
)

__all__ = [
    "categorical_cross_entropy",
    "classification_report_per_class",
    "evaluate_multiclass",
    "macro_average",
    "save_evaluation",
]
