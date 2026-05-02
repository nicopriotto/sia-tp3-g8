"""Re-export of plot helpers from ``experiments._common``.

Kept as a thin shim so ej3 scripts can ``from experiments.ej3.plots import
plot_loss_curve`` symmetrically with ej2.
"""
from experiments._common.plots import (  # noqa: F401
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    save_fig,
)

__all__ = [
    "plot_accuracy_curve",
    "plot_confusion_matrix",
    "plot_loss_curve",
    "save_fig",
]
