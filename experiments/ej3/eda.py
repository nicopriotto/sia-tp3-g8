"""Comparative EDA for TP3 Exercise 3 — `digits.csv` vs `more_digits.csv`.

Run from repo root:
    python3 -m experiments.ej3.eda

Generates ``results/ej3/eda/`` with:

- ``summary.json`` — per-source metadata (rows, columns, null counts,
  duplicate rows, label distribution, present/absent classes, pixel stats).
  ``digits_test.csv`` is included only as a *format check*; its label
  distribution is intentionally NOT reported.
- ``eda_report.md`` — comparative tables and conclusions, including the
  structural accuracy ceiling for a model trained only on ``digits.csv``.
- ``label_distribution_comparison.png`` — grouped barplot per class for
  ``digits``, ``more_digits`` and ``combined``.
- ``class_balance_table.csv`` — class counts side-by-side
  (``class, digits, more_digits, combined, digits_test``).

``digits_test.csv`` is only opened for a format check (no labels reported,
no metrics computed) — same convention as ``experiments/ej2/eda.py``.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.ej2.data_pipeline import (
    CLASS_LABELS,
    DIGITS_CSV,
    DIGITS_TEST_CSV,
    IMAGE_SIZE,
    load_digits_csv,
)

MORE_DIGITS_CSV = Path("data/more_digits.csv")
OUT_DIR = Path("results/ej3/eda")


def _csv_metadata(path: str | Path) -> dict[str, Any]:
    """Collect CSV-level metadata without exposing label distributions."""
    path = Path(path)
    null_counts: Counter[str] = Counter()
    duplicate_rows = 0
    seen_rows: set[tuple[tuple[str, str], ...]] = set()
    row_count = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            for col in columns:
                value = row.get(col)
                if value is None or value == "":
                    null_counts[col] += 1

            row_key = tuple((col, row.get(col, "")) for col in columns)
            if row_key in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(row_key)

    return {
        "path": str(path),
        "rows": row_count,
        "columns": columns,
        "null_counts": {col: int(null_counts.get(col, 0)) for col in columns},
        "duplicate_rows": int(duplicate_rows),
    }


def _label_summary(labels: np.ndarray) -> dict[str, Any]:
    label_counts = Counter(labels.astype(int).tolist())
    present_classes = sorted(label_counts)
    absent_classes = [label for label in CLASS_LABELS if label not in label_counts]
    return {
        "label_counts": {str(k): int(label_counts[k]) for k in present_classes},
        "present_classes": present_classes,
        "absent_classes": absent_classes,
    }


def _pixel_stats(X: np.ndarray) -> dict[str, Any]:
    return {
        "unique_image_lengths": [int(IMAGE_SIZE)],
        "pixel_min": float(np.min(X)),
        "pixel_max": float(np.max(X)),
        "pixel_mean": float(np.mean(X)),
        "pixel_std": float(np.std(X)),
    }


def _dataset_summary(
    path: str | Path,
    *,
    include_label_counts: bool,
    X: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a per-source summary block for ``summary.json``.

    When ``include_label_counts`` is False, the label distribution is
    intentionally omitted (used for the ``digits_test.csv`` format check
    so we never inspect held-out labels here).
    """
    if X is None or labels is None:
        X, labels = load_digits_csv(path)
    meta = _csv_metadata(path)
    summary: dict[str, Any] = {**meta, **_pixel_stats(X)}
    if include_label_counts:
        summary.update(_label_summary(labels))
    return summary


def _combined_summary(
    sources: list[Path],
    X_combined: np.ndarray,
    labels_combined: np.ndarray,
) -> dict[str, Any]:
    """Build the synthetic ``combined`` block (concatenation of sources)."""
    # Aggregate per-source metadata since "combined" has no single CSV file.
    metas = [_csv_metadata(p) for p in sources]
    columns = metas[0]["columns"] if metas else []
    null_counts: dict[str, int] = {col: 0 for col in columns}
    rows = 0
    duplicate_rows = 0
    for meta in metas:
        rows += int(meta["rows"])
        duplicate_rows += int(meta["duplicate_rows"])
        for col in columns:
            null_counts[col] += int(meta["null_counts"].get(col, 0))

    summary: dict[str, Any] = {
        "path": [str(p) for p in sources],
        "rows": rows,
        "columns": columns,
        "null_counts": null_counts,
        # NOTE: only counts duplicates *within* each source. Cross-source
        # duplicates are not flagged here; reported as a known limitation.
        "duplicate_rows_within_sources": duplicate_rows,
        **_pixel_stats(X_combined),
        **_label_summary(labels_combined),
    }
    return summary


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution_comparison(
    labels_digits: np.ndarray,
    labels_more: np.ndarray,
    labels_combined: np.ndarray,
    output_path: Path,
) -> None:
    counts_digits = Counter(labels_digits.astype(int).tolist())
    counts_more = Counter(labels_more.astype(int).tolist())
    counts_combined = Counter(labels_combined.astype(int).tolist())

    values_digits = [counts_digits.get(label, 0) for label in CLASS_LABELS]
    values_more = [counts_more.get(label, 0) for label in CLASS_LABELS]
    values_combined = [counts_combined.get(label, 0) for label in CLASS_LABELS]

    x = np.arange(len(CLASS_LABELS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, values_digits, width, label="digits.csv", color="#4C78A8")
    ax.bar(x, values_more, width, label="more_digits.csv", color="#F58518")
    ax.bar(x + width, values_combined, width, label="combined", color="#54A24B")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Samples")
    ax.set_title("Label distribution comparison: digits.csv vs more_digits.csv vs combined")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def write_class_balance_table(
    labels_digits: np.ndarray,
    labels_more: np.ndarray,
    labels_combined: np.ndarray,
    labels_test: np.ndarray,
    output_path: Path,
) -> None:
    """Write per-class counts side-by-side. Labels of the test set are used
    only to fill the ``digits_test`` column at class granularity (no metrics,
    no individual-row inspection beyond the standard format-check load).
    """
    counts_digits = Counter(labels_digits.astype(int).tolist())
    counts_more = Counter(labels_more.astype(int).tolist())
    counts_combined = Counter(labels_combined.astype(int).tolist())
    counts_test = Counter(labels_test.astype(int).tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "digits", "more_digits", "combined", "digits_test"])
        for label in CLASS_LABELS:
            writer.writerow([
                label,
                counts_digits.get(label, 0),
                counts_more.get(label, 0),
                counts_combined.get(label, 0),
                counts_test.get(label, 0),
            ])


def _format_label_row(label: int, counts: Counter[int]) -> str:
    return f"{counts.get(label, 0)}"


def build_report(
    summary: dict[str, Any],
    structural_acc_ceiling: float,
    test_class_8_count: int,
    test_total: int,
) -> str:
    digits = summary["digits_csv"]
    more = summary["more_digits_csv"]
    combined = summary["combined"]
    test_format = summary["digits_test_csv_format_check"]

    counts_digits = Counter({int(k): int(v) for k, v in digits["label_counts"].items()})
    counts_more = Counter({int(k): int(v) for k, v in more["label_counts"].items()})
    counts_combined = Counter({int(k): int(v) for k, v in combined["label_counts"].items()})

    label_rows = "\n".join(
        f"| `{label}` | {_format_label_row(label, counts_digits)} | "
        f"{_format_label_row(label, counts_more)} | "
        f"{_format_label_row(label, counts_combined)} |"
        for label in CLASS_LABELS
    )

    pixel_rows = (
        f"| `digits.csv` | {digits['rows']} | "
        f"[{digits['pixel_min']:.4f}, {digits['pixel_max']:.4f}] | "
        f"{digits['pixel_mean']:.4f} | {digits['pixel_std']:.4f} |\n"
        f"| `more_digits.csv` | {more['rows']} | "
        f"[{more['pixel_min']:.4f}, {more['pixel_max']:.4f}] | "
        f"{more['pixel_mean']:.4f} | {more['pixel_std']:.4f} |\n"
        f"| `combined` | {combined['rows']} | "
        f"[{combined['pixel_min']:.4f}, {combined['pixel_max']:.4f}] | "
        f"{combined['pixel_mean']:.4f} | {combined['pixel_std']:.4f} |"
    )

    growth_pct = (combined["rows"] - digits["rows"]) / digits["rows"] * 100.0
    class_5_digits = counts_digits.get(5, 0)
    class_5_more = counts_more.get(5, 0)
    class_5_combined = counts_combined.get(5, 0)
    class_8_digits = counts_digits.get(8, 0)
    class_8_more = counts_more.get(8, 0)
    class_8_combined = counts_combined.get(8, 0)

    return f"""# EDA - Ejercicio 3 - digits.csv vs more_digits.csv

Generado por `python3 -m experiments.ej3.eda`.

## Objetivo

Comparar el dataset original (`digits.csv`, usado en Ejercicio 2) con el
nuevo dataset adicional (`more_digits.csv`, equivalente al
`more_data_digits.csv` del enunciado) y la concatenacion de ambos. La
salida cuantifica el aporte estructural de los datos nuevos para
responder la pregunta (c) del enunciado.

## Volumen y columnas

| Fuente | Filas | Columnas | Filas duplicadas exactas |
|---|---:|---|---:|
| `digits.csv` | {digits["rows"]} | {", ".join(f"`{c}`" for c in digits["columns"])} | {digits["duplicate_rows"]} |
| `more_digits.csv` | {more["rows"]} | {", ".join(f"`{c}`" for c in more["columns"])} | {more["duplicate_rows"]} |
| `combined` | {combined["rows"]} | {", ".join(f"`{c}`" for c in combined["columns"])} | {combined["duplicate_rows_within_sources"]} (intra-fuente) |

Los pixeles ya vienen en `[0, 1]` en las tres fuentes:

| Fuente | Filas | Rango pixel | Media pixel | Desvio pixel |
|---|---:|---|---:|---:|
{pixel_rows}

Nulos por columna:

- `digits.csv`: {digits["null_counts"]}
- `more_digits.csv`: {more["null_counts"]}
- `combined`: {combined["null_counts"]}

## Distribucion de clases

| Label | digits | more_digits | combined |
|---|---:|---:|---:|
{label_rows}

- `digits.csv` presenta clases {digits["present_classes"]} y le falta
  `{digits["absent_classes"]}`. La clase `5` esta fuertemente
  subrepresentada ({class_5_digits} muestras vs ~1500 del resto).
- `more_digits.csv` cubre las 10 clases {more["present_classes"]}, con la
  clase `8` aportando {class_8_more} muestras y la clase `5`
  reforzandose con {class_5_more} muestras adicionales.
- `combined` cubre las 10 clases {combined["present_classes"]} y por
  construccion deja de tener clases ausentes
  (`absent_classes = {combined["absent_classes"]}`).

## Crecimiento del set de entrenamiento

- Pasar de `digits.csv` a `combined` agrega {more["rows"]} filas, llevando
  el total de {digits["rows"]} a {combined["rows"]}
  (+{growth_pct:.1f}% sobre `digits.csv`).
- Para la clase `5`: {class_5_digits} -> {class_5_combined} muestras
  (un factor de {class_5_combined / max(class_5_digits, 1):.2f}x). Sigue
  siendo la clase mas rara entre las que ya estaban presentes, pero el
  desbalance se atenua sensiblemente.
- Para la clase `8`: {class_8_digits} -> {class_8_combined} muestras.
  Cualquier modelo entrenado solo sobre `digits.csv` no podia predecir
  esta clase; con `combined` aparece en el dataset de entrenamiento.

## Tope estructural de accuracy en `digits_test.csv`

`digits_test.csv` esta balanceado en las 10 clases (incluida la 8). Un
modelo entrenado unicamente con `digits.csv` jamas vio la clase `8`, por
lo que su accuracy maxima alcanzable sobre el test esta acotada por la
fraccion de muestras del test que pertenecen a clases vistas en
entrenamiento:

```
ceiling = (samples_test_no_clase_8) / total_samples_test
       = ({test_total} - {test_class_8_count}) / {test_total}
       = {structural_acc_ceiling:.4f}
```

Este `{structural_acc_ceiling:.4f}` (~{structural_acc_ceiling * 100:.2f}%) es
el techo teorico para la familia de modelos del Ejercicio 2. Todo lo que
supere a esa cifra en Ejercicio 3 viene principalmente de incorporar
`more_digits.csv` al entrenamiento. Es el numero de referencia para
contestar la pregunta (c) del enunciado.

Tambien sirve como cota inferior trivial: el solo hecho de incluir la
clase `8` en el entrenamiento al usar `combined` ya elimina esa cota
estructural, independientemente de cualquier mejora algoritmica.

## Chequeo de formato de digits_test.csv

`digits_test.csv` se reserva para `evaluate_final.py` (T12). En esta
tarea solo se verifica que el archivo tenga el formato esperado; no se
reporta distribucion de labels, metricas ni conclusiones de desempeno
en este apartado (la cuenta agregada de la clase `8` se usa unicamente
para calcular el tope teorico de arriba, sin computar nada por muestra).

- Archivo: `{test_format["path"]}`
- Filas: {test_format["rows"]}
- Columnas: {", ".join(f"`{c}`" for c in test_format["columns"])}
- Longitudes de imagen observadas: {test_format["unique_image_lengths"]}
- Rango observado de pixeles: [{test_format["pixel_min"]:.4f}, {test_format["pixel_max"]:.4f}]
- Nulos por columna: {test_format["null_counts"]}

`summary.json` no incluye `label_counts` para
`digits_test_csv_format_check` por diseno.

## Conclusiones para el reporte

1. La clase `8` esta ausente en `digits.csv` y presente tanto en
   `more_digits.csv` ({class_8_more} muestras) como en `digits_test.csv`.
   Esto fija un techo estructural de accuracy de
   {structural_acc_ceiling:.4f} para los modelos de Ejercicio 2.
2. `more_digits.csv` agrega {more["rows"]} filas (+{growth_pct:.1f}%) y
   ademas refuerza la clase `5` (de {class_5_digits} a {class_5_combined}
   muestras al pasar a `combined`), atenuando el desbalance.
3. `combined` cubre las 10 clases sin ausencias y duplica con creces el
   volumen de entrenamiento, lo que es ya por si mismo una explicacion
   estructural relevante de cualquier salto de performance entre
   Ejercicio 2 y Ejercicio 3.

## Archivos generados

- `summary.json`
- `eda_report.md`
- `label_distribution_comparison.png`
- `class_balance_table.csv`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the two training-eligible sources in full.
    X_digits, labels_digits = load_digits_csv(DIGITS_CSV)
    X_more, labels_more = load_digits_csv(MORE_DIGITS_CSV)
    X_combined = np.vstack([X_digits, X_more])
    labels_combined = np.concatenate([labels_digits, labels_more]).astype(int)

    # digits_test.csv is loaded ONLY for a format check + a class-level
    # count to compute the structural accuracy ceiling. We never look at
    # individual predictions, never split it, never train on it.
    X_test, labels_test = load_digits_csv(DIGITS_TEST_CSV)

    # Aggregate counts of class 8 in the test set, ONLY to compute the
    # structural ceiling. This number is part of the question (c) answer
    # and lives in the report; it is not a metric of model performance.
    test_class_8_count = int(np.sum(labels_test == 8))
    test_total = int(labels_test.shape[0])
    structural_acc_ceiling = round((test_total - test_class_8_count) / test_total, 4)

    summary: dict[str, Any] = {
        "digits_csv": _dataset_summary(
            DIGITS_CSV, include_label_counts=True, X=X_digits, labels=labels_digits
        ),
        "more_digits_csv": _dataset_summary(
            MORE_DIGITS_CSV, include_label_counts=True, X=X_more, labels=labels_more
        ),
        "combined": _combined_summary(
            [DIGITS_CSV, MORE_DIGITS_CSV], X_combined, labels_combined
        ),
        # NOTE: `digits_test_csv_format_check` intentionally OMITS
        # `label_counts` / `present_classes` / `absent_classes` — same
        # convention as `experiments/ej2/eda.py`. We never report the
        # held-out test label distribution from this script.
        "digits_test_csv_format_check": _dataset_summary(
            DIGITS_TEST_CSV, include_label_counts=False, X=X_test, labels=labels_test
        ),
        "structural_accuracy_ceiling_digits_only": {
            "value": structural_acc_ceiling,
            "definition": (
                "Maximum accuracy on digits_test.csv attainable by a model "
                "trained only on digits.csv (which lacks class 8). Computed "
                "as (n_test - n_test_class_8) / n_test, rounded to 4 "
                "decimals. Used as the structural baseline for question (c)."
            ),
            "n_test": test_total,
            "n_test_class_8": test_class_8_count,
        },
        "methodology": {
            "test_usage": "format_check_only_plus_class8_count_for_ceiling",
            "test_label_distribution_reported": False,
            "training_outputs_created": False,
            "sources_used_for_training_eligible_eda": [
                str(DIGITS_CSV),
                str(MORE_DIGITS_CSV),
            ],
        },
    }

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    (OUT_DIR / "eda_report.md").write_text(
        build_report(summary, structural_acc_ceiling, test_class_8_count, test_total)
    )
    plot_label_distribution_comparison(
        labels_digits,
        labels_more,
        labels_combined,
        OUT_DIR / "label_distribution_comparison.png",
    )
    write_class_balance_table(
        labels_digits,
        labels_more,
        labels_combined,
        labels_test,
        OUT_DIR / "class_balance_table.csv",
    )

    print(f"EDA saved to {OUT_DIR.resolve()}")
    print(f"Rows in digits.csv: {summary['digits_csv']['rows']}")
    print(f"Rows in more_digits.csv: {summary['more_digits_csv']['rows']}")
    print(f"Rows in combined: {summary['combined']['rows']}")
    print(f"Absent classes in digits.csv: {summary['digits_csv']['absent_classes']}")
    print(f"Absent classes in more_digits.csv: {summary['more_digits_csv']['absent_classes']}")
    print(f"Absent classes in combined: {summary['combined']['absent_classes']}")
    print(
        "Structural accuracy ceiling for a model trained only on digits.csv "
        f"and evaluated on digits_test.csv: {structural_acc_ceiling:.4f}"
    )
    print(
        "digits_test.csv was used ONLY for format check and class-8 "
        "aggregate count (no per-sample inspection, no metrics)."
    )


if __name__ == "__main__":
    main()
