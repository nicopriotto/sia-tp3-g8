"""Final evaluation of the selected ej3 model on digits_test.csv.

Run from repo root:
    python3 -m experiments.ej3.evaluate_final
    python3 -m experiments.ej3.evaluate_final --run-dir results/ej3/training/<run_id>

Reads the candidate selected by T11 (or the explicit `--run-dir`) and writes:
    results/ej3/final_test/evaluation.json
    results/ej3/final_test/confusion_matrix_test.png
    results/ej3/final_test/label_distribution_test.png
    results/ej3/final_test/final_test_report.md

This script is the only place in ej3 where digits_test.csv is consumed.
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
from experiments._common.evaluation import evaluate_multiclass, save_evaluation
from experiments._common.plots import save_fig
from perceptron.persistence import load_model

DEFAULT_SELECTION = Path("results/ej3/selection/selected_model.json")
DEFAULT_OUTPUT = Path("results/ej3/final_test")
TARGET_ACCURACY = 0.98


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
            "Run experiments.ej3.summarize first."
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
    lines: list[str] = ["# Ej3 — Final test report", ""]
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
    lines.append(f"- weight_init: `{config.get('weight_init')}`")
    lines.append(f"- loss: `{config.get('loss')}`")
    lines.append(f"- optimizer: `{config.get('optimizer')}` params=`{config.get('optimizer_params')}`")
    lines.append(f"- learning_rate: {config.get('learning_rate')}")
    lines.append(
        f"- lr_schedule: `{config.get('lr_schedule')}` "
        f"params=`{config.get('lr_schedule_params')}`"
    )
    lines.append(f"- batch_size: {config.get('batch_size')} | epochs configured: {config.get('epochs')}")
    lines.append(f"- seed: {config.get('seed')}")
    lines.append(f"- data_strategy: `{config.get('data_strategy')}`")
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
    lines.append("## Clase 8 — contraste con ej2")
    lines.append("")
    lines.append(
        "En ej2 el train (`digits.csv`) no contenia la clase 8; el modelo seleccionado "
        "obtuvo `precision=0.0000`, `recall=0.0000`, `f1=0.0000` sobre los 243 ochos del "
        "test (cota teorica de accuracy `~0.9027`). El reporte de ej2 (`results/ej2/final_test/`) "
        "lo lista como caso patologico."
    )
    lines.append(
        f"En ej3 el modelo ganador uso `data_strategy={config.get('data_strategy')!r}`, que "
        "incluye `more_digits.csv` y por lo tanto si tiene ejemplos de la clase 8 en train."
    )
    lines.append(
        f"En `digits_test.csv` la clase `8` aparece {eight_count} vez/veces."
    )
    if eight_count == 0:
        lines.append(
            "El test no contiene ejemplos de la clase 8 en esta corrida; no hay contraste "
            "directo posible."
        )
    else:
        lines.append(
            f"El modelo predijo correctamente {eight_correct} de esos {eight_count}. "
            f"Per-class de 8: precision={eight_metrics.get('precision', 0.0):.4f}, "
            f"recall={eight_metrics.get('recall', 0.0):.4f}, "
            f"f1={eight_metrics.get('f1', 0.0):.4f}."
        )
        lines.append(
            "Esto confirma que sumar `more_digits.csv` al train remueve el techo del "
            "0.9027 que dominaba ej2: ahora la clase 8 contribuye a accuracy real, no "
            "a un aciertos espureo."
        )
    lines.append("")

    accuracy_test = float(test_evaluation["accuracy"])
    delta_target = accuracy_test - TARGET_ACCURACY
    lines.append("## Target del enunciado (accuracy >= 0.98)")
    lines.append("")
    lines.append(f"- Target: accuracy >= {TARGET_ACCURACY:.2f} sobre `digits_test.csv`.")
    lines.append(f"- Obtenido: accuracy = {accuracy_test:.4f}.")
    lines.append(f"- Delta vs target: {delta_target:+.4f}.")
    if accuracy_test >= TARGET_ACCURACY:
        lines.append(
            f"- **Target alcanzado.** El modelo supera el 98% por {delta_target:+.4f} "
            "puntos absolutos sobre el set de test reservado."
        )
    else:
        gap = TARGET_ACCURACY - accuracy_test
        lines.append(
            f"- **Target no alcanzado** por {gap:.4f} puntos absolutos. "
            "Candidatos para una iteracion adicional:"
        )
        lines.append(
            "  - regularizacion mas fuerte / dropout para cerrar el gap train-val si "
            "hay overfitting,"
        )
        lines.append(
            "  - data augmentation (rotaciones, traslaciones, ruido pixel) para "
            "robustecer la frontera de decision sobre `digits_test.csv`,"
        )
        lines.append(
            "  - arquitectura mas grande o ensembling de seeds,"
        )
        lines.append(
            "  - mas epocas con lr_schedule decreciente y early stopping mas paciente."
        )
    lines.append("")

    if val_evaluation is not None:
        val_acc = float(val_evaluation.get("accuracy", 0.0))
        gap = val_acc - accuracy_test
        lines.append("## Overfitting indicator (validation vs test)")
        lines.append("")
        lines.append(
            f"- val_accuracy = {val_acc:.4f}, test_accuracy = {accuracy_test:.4f}, "
            f"gap = {gap:+.4f}."
        )
        if abs(gap) <= 0.01:
            lines.append(
                "- Gap muy chico: la validation fue una buena proxy del test, "
                "el modelo generaliza."
            )
        elif gap > 0.03:
            lines.append(
                "- Gap > 3 puntos: T11 podria haber penalizado mas la diferencia "
                "train-val. Documentar en T13."
            )
        elif gap > 0.01:
            lines.append(
                "- Gap moderado (1-3 puntos): drop esperable entre validation y test, "
                "no es senal de overfitting fuerte."
            )
        else:
            lines.append(
                "- gap negativo: el modelo rinde mejor en test que en validation, "
                "muy probablemente por la composicion del split (no es preocupante)."
            )
        lines.append("")

    lines.append("## Referencia ej2")
    lines.append("")
    lines.append(
        "Como referencia, el ganador de ej2 (`baseline_tanh_sgd__seed42`) obtuvo "
        "`accuracy=0.8074`, `macro_f1=0.7592` sobre el mismo `digits_test.csv` "
        "(ver `results/ej2/final_test/final_test_report.md`). La diferencia "
        f"vs ej3 ({accuracy_test:.4f} accuracy, {test_evaluation['macro_f1']:.4f} macro_f1) "
        "se origina principalmente en haber sumado `more_digits.csv` (clase 8 "
        "incluida) al train, mas la regularizacion L2 y el optimizador Adam que "
        "introdujo T10."
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

    data_strategy = None
    if isinstance(config.extra, dict):
        data_strategy = config.extra.get("data_strategy")

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
        "weight_init": getattr(config, "weight_init", None),
        "lr_schedule": getattr(config, "lr_schedule", None),
        "lr_schedule_params": getattr(config, "lr_schedule_params", {}),
        "data_strategy": data_strategy,
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
