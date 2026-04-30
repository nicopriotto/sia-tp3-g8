"""EDA for ej1 fraud dataset.

Run from repo root:
    python3 -m experiments.ej1.eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

CSV_PATH = "data/fraud_dataset.csv"
OUT_DIR = Path("results/ej1/eda")
TARGET_COL = "big_model_fraud_probability"
FLAG_COL = "flagged_fraud"


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_histograms(df: pd.DataFrame, feature_cols: list[str]) -> None:
    n = len(feature_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col], bins=40, edgecolor="none", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature distributions", fontsize=11)
    fig.tight_layout()
    _save(fig, "feature_histograms.png")


def plot_target_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[TARGET_COL], bins=50, edgecolor="none", alpha=0.85, color="steelblue")
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("count")
    ax.set_title("Target distribution (BigModel fraud probability)")
    ax.axvline(df[TARGET_COL].mean(), color="red", linestyle="--", label=f"mean={df[TARGET_COL].mean():.3f}")
    ax.legend()
    _save(fig, "target_histogram.png")


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str]) -> None:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    _save(fig, "correlation_matrix.png")


def plot_bigmodel_vs_flag(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in df.groupby(FLAG_COL):
        ax.hist(
            group[TARGET_COL], bins=40, alpha=0.6,
            label=f"flagged={int(label)} (n={len(group)})", density=True,
        )
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("density")
    ax.set_title("BigModel probability by fraud label")
    ax.legend()
    _save(fig, "bigmodel_vs_flag.png")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Shape: {df.shape}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"\n{df.describe().T.to_string()}\n")

    feature_cols = [c for c in df.columns if c not in (TARGET_COL, FLAG_COL)]

    # Plots
    plot_feature_histograms(df, feature_cols)
    plot_target_histogram(df)
    all_analysis_cols = feature_cols + [TARGET_COL, FLAG_COL]
    plot_correlation_matrix(df, all_analysis_cols)
    plot_bigmodel_vs_flag(df)

    # Feature correlations with target
    correlations = {
        col: round(float(df[col].corr(df[TARGET_COL])), 6)
        for col in feature_cols
    }

    # Oracle upper bound: BigModel scores directly vs flagged_fraud
    oracle_auc_roc = float(roc_auc_score(df[FLAG_COL], df[TARGET_COL]))
    oracle_auc_pr = float(average_precision_score(df[FLAG_COL], df[TARGET_COL]))

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_mean": round(float(df[TARGET_COL].mean()), 6),
        "target_std": round(float(df[TARGET_COL].std()), 6),
        "flag_positive_rate": round(float(df[FLAG_COL].mean()), 6),
        "bigmodel_vs_flag_auc_roc": round(oracle_auc_roc, 6),
        "bigmodel_vs_flag_auc_pr": round(oracle_auc_pr, 6),
        "feature_correlations_with_target": correlations,
        "null_counts": df.isnull().sum().to_dict(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== EDA Summary ===")
    print(f"Rows: {summary['n_rows']}, Cols: {summary['n_columns']}")
    print(f"Target mean: {summary['target_mean']:.4f}, std: {summary['target_std']:.4f}")
    print(f"Flag positive rate: {summary['flag_positive_rate']:.4f}")
    print(f"Oracle AUC-ROC (BigModel vs flagged_fraud): {oracle_auc_roc:.4f}")
    print(f"Oracle AUC-PR  (BigModel vs flagged_fraud): {oracle_auc_pr:.4f}")
    print(f"\nCorrelations with target:")
    for col, v in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        print(f"  {col:40s} {v:+.4f}")
    print(f"\nResults saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
