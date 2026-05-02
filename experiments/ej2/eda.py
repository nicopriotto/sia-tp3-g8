"""Minimal EDA for TP3 Exercise 2 digits data.

Run from repo root:
    python3 -m experiments.ej2.eda
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
    FEATURE_SHAPE,
    IMAGE_SIZE,
    load_digits_csv,
)

OUT_DIR = Path("results/ej2/eda")


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


def _dataset_summary(path: str | Path, *, include_label_counts: bool) -> dict[str, Any]:
    X, labels = load_digits_csv(path)
    meta = _csv_metadata(path)
    summary: dict[str, Any] = {
        **meta,
        "unique_image_lengths": [int(IMAGE_SIZE)],
        "pixel_min": float(np.min(X)),
        "pixel_max": float(np.max(X)),
        "pixel_mean": float(np.mean(X)),
        "pixel_std": float(np.std(X)),
    }
    if include_label_counts:
        label_counts = Counter(labels.astype(int).tolist())
        present_classes = sorted(label_counts)
        absent_classes = [label for label in CLASS_LABELS if label not in label_counts]
        summary.update(
            {
                "label_counts": {str(k): int(label_counts[k]) for k in present_classes},
                "present_classes": present_classes,
                "absent_classes": absent_classes,
            }
        )
    return summary


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(labels: np.ndarray, output_path: Path) -> None:
    counts = Counter(labels.astype(int).tolist())
    values = [counts.get(label, 0) for label in CLASS_LABELS]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CLASS_LABELS, values, color="#4C78A8")
    ax.set_xticks(CLASS_LABELS)
    ax.set_xlabel("Label")
    ax.set_ylabel("Samples")
    ax.set_title("digits.csv label distribution")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    _save(fig, output_path)


def plot_sample_grid(X: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    first_by_label: dict[int, int] = {}
    for idx, label in enumerate(labels.astype(int).tolist()):
        first_by_label.setdefault(label, idx)

    selected_labels = sorted(first_by_label)
    ncols = 5
    nrows = int(np.ceil(len(selected_labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.2 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, label in zip(axes_arr, selected_labels):
        image = X[first_by_label[label]].reshape(FEATURE_SHAPE)
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"label {label}")
        ax.axis("off")

    for ax in axes_arr[len(selected_labels):]:
        ax.axis("off")

    fig.suptitle("Sample images from digits.csv", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


def plot_pixel_intensity_hist(X: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(X.ravel(), bins=50, color="#59A14F", edgecolor="none")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    ax.set_title("digits.csv pixel intensity histogram")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def build_report(summary: dict[str, Any]) -> str:
    train = summary["digits_csv"]
    test_format = summary["digits_test_csv_format_check"]
    label_counts = train["label_counts"]
    label_rows = "\n".join(
        f"| `{label}` | {label_counts.get(str(label), 0)} |" for label in CLASS_LABELS
    )
    null_rows = "\n".join(
        f"| `{col}` | {count} |" for col, count in train["null_counts"].items()
    )
    test_null_rows = "\n".join(
        f"| `{col}` | {count} |" for col, count in test_format["null_counts"].items()
    )

    return f"""# EDA - Ejercicio 2 digits.csv

Generado por `python3 -m experiments.ej2.eda`.

## Columnas

- `label`: clase entera del digito manuscrito, en el rango `0..9`.
- `image`: lista serializada de `{IMAGE_SIZE}` pixeles, equivalente a una imagen `{FEATURE_SHAPE[0]}x{FEATURE_SHAPE[1]}`.

## Dataset de aprendizaje

- Archivo: `{DIGITS_CSV}`
- Filas: {train["rows"]}
- Columnas: {", ".join(f"`{col}`" for col in train["columns"])}
- Longitudes de imagen observadas: {train["unique_image_lengths"]}
- Rango observado de pixeles: [{train["pixel_min"]:.6f}, {train["pixel_max"]:.6f}]
- Media de pixeles: {train["pixel_mean"]:.6f}
- Desvio de pixeles: {train["pixel_std"]:.6f}
- Filas duplicadas exactas: {train["duplicate_rows"]}

Nulos por columna:

| Columna | Nulos |
|---|---:|
{null_rows}

Distribucion de labels de `digits.csv`:

| Label | Muestras |
|---|---:|
{label_rows}

## Restricciones detectadas

- La clase `8` esta ausente en `digits.csv`.
- La clase `5` esta desbalanceada respecto del resto: {label_counts.get("5", 0)} muestras.
- La salida del MLP debe tener siempre `10` neuronas; no se debe inferir la cantidad de clases desde `digits.csv`.

## Preprocesamiento inicial

Los pixeles ya vienen en `[0, 1]`. Para el baseline no se aplica z-score ni otra
normalizacion adicional.

## Chequeo de formato de digits_test.csv

`digits_test.csv` se reserva para Task 011, como evaluacion final de produccion. En esta
tarea solo se verifica que el archivo tenga el formato esperado; no se reporta
distribucion de labels, metricas ni conclusiones de desempeno.

- Archivo: `{DIGITS_TEST_CSV}`
- Filas: {test_format["rows"]}
- Columnas: {", ".join(f"`{col}`" for col in test_format["columns"])}
- Longitudes de imagen observadas: {test_format["unique_image_lengths"]}
- Rango observado de pixeles: [{test_format["pixel_min"]:.6f}, {test_format["pixel_max"]:.6f}]

Nulos por columna en el chequeo de formato:

| Columna | Nulos |
|---|---:|
{test_null_rows}

## Archivos generados

- `summary.json`
- `label_distribution_train_source.png`
- `sample_grid.png`
- `pixel_intensity_hist.png`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_source, labels_train_source = load_digits_csv(DIGITS_CSV)
    summary = {
        "digits_csv": _dataset_summary(DIGITS_CSV, include_label_counts=True),
        "digits_test_csv_format_check": _dataset_summary(DIGITS_TEST_CSV, include_label_counts=False),
        "methodology": {
            "test_usage": "format_check_only",
            "test_label_distribution_reported": False,
            "training_outputs_created": False,
        },
    }

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    (OUT_DIR / "eda_report.md").write_text(build_report(summary))
    plot_label_distribution(labels_train_source, OUT_DIR / "label_distribution_train_source.png")
    plot_sample_grid(X_train_source, labels_train_source, OUT_DIR / "sample_grid.png")
    plot_pixel_intensity_hist(X_train_source, OUT_DIR / "pixel_intensity_hist.png")

    print(f"EDA saved to {OUT_DIR.resolve()}")
    print(f"Rows in digits.csv: {summary['digits_csv']['rows']}")
    print(f"Absent classes in digits.csv: {summary['digits_csv']['absent_classes']}")
    print("digits_test.csv was used for format checks only.")


if __name__ == "__main__":
    main()
