"""Final evaluation of the selected ej2 model on digits_test.csv.

Run from repo root:
    python3 -m experiments.ej2.evaluate_final
    python3 -m experiments.ej2.evaluate_final --run-dir results/ej2/training/<run_id>

Reads the candidate selected by Task 010 (or the explicit `--run-dir`) and writes:
    results/ej2/final_test/evaluation.json
    results/ej2/final_test/confusion_matrix_test.png
    results/ej2/final_test/label_distribution_test.png
    results/ej2/final_test/final_test_report.md

This script is the only place where digits_test.csv is consumed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ej2.data_pipeline import CLASS_LABELS, DIGITS_TEST_CSV, load_test
from experiments.ej2.evaluation import evaluate_multiclass, save_evaluation
from experiments.ej2.plots import save_fig
from perceptron.persistence import load_model

DEFAULT_SELECTION = Path("results/ej2/selection/selected_model.json")
DEFAULT_OUTPUT = Path("results/ej2/final_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None,
                        help="Path to the selected training run directory.")
    parser.add_argument("--selection",
                        default=str(DEFAULT_SELECTION),
                        help="Path to selected_model.json (used when --run-dir is omitted).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> tuple[Path, dict | None]:
    if args.run_dir is not None:
        return Path(args.run_dir), None

    selection_path = Path(args.selection)
    if not selection_path.exists():
        raise SystemExit(
            f"No --run-dir given and selection file not found at {selection_path}. "
            "Run experiments.ej2.summarize first."
        )
    selection = json.loads(selection_path.read_text())
    disclaimer = selection.get("test_disclaimer", "")
    if "digits_test.csv was not used" not in disclaimer:
        raise SystemExit(
            f"Refusing to evaluate: selection file at {selection_path} does not "
            "explicitly state that digits_test.csv was untouched during selection."
        )
    return Path(selection["run_dir"]), selection


def plot_test_confusion(matrix: list[list[int]], labels: list[int], output_path: Path) -> None:
    arr = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix (digits_test.csv)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if value > arr.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)
    save_fig(fig, output_path)


def plot_label_distribution(labels: np.ndarray, class_labels: list[int], output_path: Path) -> None:
    counts = Counter(labels.tolist())
    values = [counts.get(c, 0) for c in class_labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(c) for c in class_labels], values)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Test label distribution ({DIGITS_TEST_CSV.name})")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    save_fig(fig, output_path)


def write_report(
    output_dir: Path,
    run_dir: Path,
    selection: dict | None,
    config: dict,
    test_evaluation: dict,
    test_distribution: dict[int, int],
    val_evaluation: dict | None,
) -> None:
    lines: list[str] = ["# Ej2 — Final test report", ""]
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Test data: `{DIGITS_TEST_CSV}`")
    lines.append(f"- Test samples: {test_evaluation['n_samples']}")
    lines.append("")
    if selection is not None:
        lines.append("## Selection trace")
        lines.append("")
        lines.append(f"- Selected at: {selection.get('selected_at')}")
        lines.append(f"- Selection metric: {selection.get('selection_metric')}")
        lines.append(f"- Selection reason: {selection.get('selection_reason')}")
        lines.append(f"- `{selection.get('test_disclaimer')}` (selection phase)")
        lines.append("")

    arch = config.get("architecture")
    lines.append("## Model")
    lines.append("")
    lines.append(f"- architecture: `{arch}`")
    lines.append(f"- activation: `{config.get('activation')}` / output `{config.get('output_activation')}`")
    lines.append(f"- loss: `{config.get('loss')}`")
    lines.append(f"- optimizer: `{config.get('optimizer')}` params=`{config.get('optimizer_params')}`")
    lines.append(f"- learning_rate: {config.get('learning_rate')}")
    lines.append(f"- batch_size: {config.get('batch_size')} | epochs configured: {config.get('epochs')}")
    lines.append(f"- seed: {config.get('seed')}")
    lines.append("")

    lines.append("## Test metrics")
    lines.append("")
    lines.append(f"- accuracy: {test_evaluation['accuracy']:.4f}")
    lines.append(f"- macro_precision: {test_evaluation['macro_precision']:.4f}")
    lines.append(f"- macro_recall: {test_evaluation['macro_recall']:.4f}")
    lines.append(f"- macro_f1: {test_evaluation['macro_f1']:.4f}")
    lines.append(f"- categorical_cross_entropy: {test_evaluation['loss']:.4f}")
    lines.append("")

    if val_evaluation is not None:
        lines.append("## Validation vs Test")
        lines.append("")
        lines.append("| metric | validation | test | delta |")
        lines.append("|---|---|---|---|")
        for key, label in [
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("macro_precision", "macro_precision"),
            ("macro_recall", "macro_recall"),
            ("loss", "loss"),
        ]:
            v = val_evaluation.get(key)
            t = test_evaluation.get(key)
            if v is None or t is None:
                continue
            delta = t - v
            lines.append(f"| {label} | {v:.4f} | {t:.4f} | {delta:+.4f} |")
        lines.append("")

    lines.append("## Per-class metrics (test)")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support | count in test |")
    lines.append("|---|---|---|---|---|---|")
    per_class = test_evaluation.get("per_class", {})
    for c in CLASS_LABELS:
        row = per_class.get(str(c), {})
        lines.append(
            "| {c} | {p:.4f} | {r:.4f} | {f:.4f} | {s} | {n} |".format(
                c=c,
                p=row.get("precision", 0.0),
                r=row.get("recall", 0.0),
                f=row.get("f1", 0.0),
                s=row.get("support", 0),
                n=test_distribution.get(c, 0),
            )
        )
    lines.append("")

    lines.append("## Confusion matrix (test)")
    lines.append("")
    cm = test_evaluation.get("confusion_matrix") or []
    header = "| true \\ pred | " + " | ".join(str(l) for l in CLASS_LABELS) + " |"
    sep = "|" + "|".join(["---"] * (len(CLASS_LABELS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for label, row in zip(CLASS_LABELS, cm):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    lines.append("## Test label distribution")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for c in CLASS_LABELS:
        lines.append(f"| {c} | {test_distribution.get(c, 0)} |")
    lines.append("")

    eight_count = test_distribution.get(8, 0)
    eight_metrics = per_class.get("8", {})
    eight_correct = (cm[8][8] if len(cm) > 8 and len(cm[8]) > 8 else 0)
    lines.append("## Class 8 — special case")
    lines.append("")
    lines.append(
        "El conjunto de entrenamiento `digits.csv` no contiene la clase `8` "
        "(el EDA lo confirma)."
    )
    lines.append(
        f"En `digits_test.csv` la clase `8` aparece {eight_count} vez/veces."
    )
    if eight_count == 0:
        lines.append(
            "El test no contiene ejemplos de la clase 8 en esta corrida, asi que el modelo "
            "no enfrenta el caso fuera de distribucion."
        )
    else:
        lines.append(
            f"El modelo predijo correctamente {eight_correct} de esos {eight_count}. "
            "Como nunca vio un 8 durante entrenamiento, todo acierto sobre esta clase es "
            "espureo: no aprendio a reconocer 8, solo es accidental que su salida elegida "
            "sea esa. Esto es esperable y no debe contar como evidencia de generalizacion."
        )
        lines.append(
            f"Per-class de 8: precision={eight_metrics.get('precision', 0.0):.4f}, "
            f"recall={eight_metrics.get('recall', 0.0):.4f}, f1={eight_metrics.get('f1', 0.0):.4f}."
        )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- `{output_dir}/evaluation.json`")
    lines.append(f"- `{output_dir}/confusion_matrix_test.png`")
    lines.append(f"- `{output_dir}/label_distribution_test.png`")
    lines.append(f"- `{output_dir}/final_test_report.md`")
    lines.append("")

    (output_dir / "final_test_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir, selection = resolve_run_dir(args)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    print(f"Loading model from {run_dir}")
    model, config = load_model(run_dir)

    val_evaluation = None
    eval_path = run_dir / "evaluation.json"
    if eval_path.exists():
        run_eval = json.loads(eval_path.read_text())
        val_evaluation = run_eval.get("validation")

    print(f"Loading test set from {DIGITS_TEST_CSV}")
    X_test, y_test, labels_test = load_test()
    print(f"Test samples: {len(labels_test)}")

    outputs = model.predict(X_test)
    test_evaluation = evaluate_multiclass(y_test, outputs, labels=CLASS_LABELS)
    test_distribution = {int(k): int(v) for k, v in sorted(Counter(labels_test.tolist()).items())}

    payload = {
        "run_dir": str(run_dir),
        "test_data": str(DIGITS_TEST_CSV),
        "class_labels": CLASS_LABELS,
        "test_distribution": test_distribution,
        "metrics": test_evaluation,
        "selection": selection,
    }
    save_evaluation(payload, output_dir / "evaluation.json")

    plot_test_confusion(
        test_evaluation["confusion_matrix"],
        labels=CLASS_LABELS,
        output_path=output_dir / "confusion_matrix_test.png",
    )
    plot_label_distribution(
        labels_test,
        class_labels=CLASS_LABELS,
        output_path=output_dir / "label_distribution_test.png",
    )

    config_dict = {
        "architecture": config.architecture,
        "activation": config.activation,
        "output_activation": config.output_activation,
        "loss": config.loss,
        "optimizer": config.optimizer,
        "optimizer_params": config.optimizer_params,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "seed": config.seed,
    }
    write_report(
        output_dir=output_dir,
        run_dir=run_dir,
        selection=selection,
        config=config_dict,
        test_evaluation=test_evaluation,
        test_distribution=test_distribution,
        val_evaluation=val_evaluation,
    )

    print(
        "[test] "
        f"acc={test_evaluation['accuracy']:.4f} "
        f"macro_f1={test_evaluation['macro_f1']:.4f} "
        f"loss={test_evaluation['loss']:.4f}"
    )
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
