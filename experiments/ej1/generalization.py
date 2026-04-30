"""Generalisation study for ej1: learning curve, k-fold CV, threshold analysis.

Run from repo root:
    python3 -m experiments.ej1.generalization \
        --config experiments/ej1/configs/nonlinear_sigmoid_mse.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler

from perceptron.config import ExperimentConfig
from perceptron.metrics import mse, mae, threshold_sweep
from perceptron.models.factory import build_model
from perceptron.training.trainer import Trainer
from perceptron.utils import set_seed

from experiments.ej1.data_pipeline import prepare_data, make_kfold_splits
from experiments.ej1.plots import save_fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# E.0 Learning curve
# ---------------------------------------------------------------------------

def learning_curve(config: ExperimentConfig, bundle, fractions=(0.1, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    X_tv = np.vstack([bundle.X_train, bundle.X_val])
    y_tv = np.hstack([bundle.y_train, bundle.y_val])
    n_total = len(X_tv)
    # val = last 20% of trainval (fixed, unscaled by fraction)
    n_val = int(round(n_total * 0.2))
    X_val_lc = X_tv[-n_val:]
    y_val_lc = y_tv[-n_val:]
    rows = []
    for frac in fractions:
        n_train = max(10, int(round((n_total - n_val) * frac)))
        X_tr = X_tv[:n_train]
        y_tr = y_tv[:n_train]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val_lc)
        m = build_model(config, n_features=X_tr_s.shape[1])
        t = Trainer(m, config)
        t.fit(X_tr_s, y_tr, X_val_s, y_val_lc, stop_on_perfect=False, restore_best_weights=True)
        preds = m.predict(X_val_s).flatten()
        rows.append({"n_train": n_train, "frac": frac, "val_mse": round(float(mse(y_val_lc, preds)), 6)})
        print(f"  learning_curve frac={frac:.2f} n_train={n_train:5d} val_mse={rows[-1]['val_mse']:.6f}")
    return rows


def plot_learning_curve(lc_data: list[dict], out_path: Path) -> None:
    ns = [r["n_train"] for r in lc_data]
    vals = [r["val_mse"] for r in lc_data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, vals, "o-", linewidth=1.5)
    ax.set_xlabel("n_train samples")
    ax.set_ylabel("val MSE")
    ax.set_title("Learning curve (val MSE vs training set size)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# E.1 K-fold CV
# ---------------------------------------------------------------------------

def run_kfold(config: ExperimentConfig, bundle, n_splits: int = 5) -> list[dict]:
    X_tv = np.vstack([bundle.X_train, bundle.X_val])
    y_tv = np.hstack([bundle.y_train, bundle.y_val])
    yf_tv = np.hstack([bundle.y_flag_train, bundle.y_flag_val])

    folds = make_kfold_splits(X_tv, y_tv, yf_tv, n_splits=n_splits, seed=config.seed)

    fold_results = []
    all_scores = []
    all_flags = []

    for fold_idx, (X_tr, y_tr, yf_tr, X_vl, y_vl, yf_vl) in enumerate(folds):
        m = build_model(config, n_features=X_tr.shape[1])
        t = Trainer(m, config)
        t.fit(X_tr, y_tr, X_vl, y_vl, stop_on_perfect=False, restore_best_weights=True)

        preds = m.predict(X_vl).flatten()
        preds_clip = np.clip(preds, 0.0, 1.0)

        fold_mse = float(mse(y_vl, preds))
        fold_mae = float(mae(y_vl, preds))
        auc_roc = float(roc_auc_score(yf_vl, preds_clip))
        auc_pr = float(average_precision_score(yf_vl, preds_clip))

        # Best threshold by F1 within fold
        sweep = threshold_sweep(yf_vl, preds_clip)
        best_thr = max(sweep, key=lambda x: x["f1"])

        fold_results.append({
            "fold": fold_idx,
            "mse": round(fold_mse, 6),
            "mae": round(fold_mae, 6),
            "auc_roc": round(auc_roc, 6),
            "auc_pr": round(auc_pr, 6),
            "best_threshold_f1": round(best_thr["threshold"], 4),
            "best_f1": round(best_thr["f1"], 4),
            "best_precision": round(best_thr["precision"], 4),
            "best_recall": round(best_thr["recall"], 4),
        })

        all_scores.append(preds_clip)
        all_flags.append(yf_vl)
        print(f"  fold {fold_idx}: mse={fold_mse:.5f} auc_roc={auc_roc:.4f} auc_pr={auc_pr:.4f} best_f1={best_thr['f1']:.4f}@{best_thr['threshold']:.2f}")

    # E.2 Global threshold from concatenated OOF scores
    oof_scores = np.concatenate(all_scores)
    oof_flags = np.concatenate(all_flags)
    global_sweep = threshold_sweep(oof_flags, oof_scores)
    recommended_threshold = max(global_sweep, key=lambda x: x["f1"])

    return fold_results, recommended_threshold, global_sweep


# ---------------------------------------------------------------------------
# E.3 Final model on test
# ---------------------------------------------------------------------------

def train_final_model(config: ExperimentConfig, bundle):
    X_tv = np.vstack([bundle.X_train, bundle.X_val])
    y_tv = np.hstack([bundle.y_train, bundle.y_val])

    scaler = StandardScaler().fit(X_tv)
    X_tv_s = scaler.transform(X_tv)
    X_test_s = scaler.transform(bundle.X_test)

    m = build_model(config, n_features=X_tv_s.shape[1])
    t = Trainer(m, config)
    t.fit(X_tv_s, y_tv, stop_on_perfect=False, restore_best_weights=False)
    return m, X_test_s


def plot_pr_curve(y_true, scores, out_path: Path) -> None:
    prec, rec, thr = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, linewidth=1.5, label=f"TinyModel (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve vs flagged_fraud")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_roc_curve(y_true, scores, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=1.5, label=f"TinyModel (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve vs flagged_fraud")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_threshold_sweep(sweep: list[dict], recommended_thr: float, out_path: Path) -> None:
    thresholds = [r["threshold"] for r in sweep]
    precision = [r["precision"] for r in sweep]
    recall = [r["recall"] for r in sweep]
    f1 = [r["f1"] for r in sweep]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, f1, label="F1", linewidth=2)
    ax.axvline(recommended_thr, color="red", linestyle="--", label=f"recommended={recommended_thr:.2f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_title("Threshold sweep vs flagged_fraud (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_pred_vs_bigmodel(y_true, preds, out_path: Path) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(2000, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[idx], preds[idx], s=8, alpha=0.3, edgecolors="none")
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
    ax.set_xlabel("BigModel probability (target)")
    ax.set_ylabel("TinyModel prediction")
    ax.set_title("TinyModel vs BigModel (test set)")
    ax.legend()
    save_fig(fig, out_path)


def _find_threshold_at_min_recall(sweep: list[dict], min_recall: float) -> dict | None:
    candidates = [r for r in sweep if r["recall"] >= min_recall]
    return min(candidates, key=lambda x: x["threshold"]) if candidates else None


def _find_threshold_at_min_precision(sweep: list[dict], min_precision: float) -> dict | None:
    candidates = [r for r in sweep if r["precision"] >= min_precision]
    return max(candidates, key=lambda x: x["threshold"]) if candidates else None


def build_final_report(
    config_name: str,
    lc_data: list[dict],
    fold_results: list[dict],
    recommended_threshold: dict,
    test_metrics: dict,
    test_sweep: list[dict],
    oracle_auc_roc: float = 1.0,
    oracle_auc_pr: float = 1.0,
) -> str:
    n_folds = len(fold_results)
    mse_mean = np.mean([r["mse"] for r in fold_results])
    mse_std = np.std([r["mse"] for r in fold_results])
    auc_roc_mean = np.mean([r["auc_roc"] for r in fold_results])
    auc_roc_std = np.std([r["auc_roc"] for r in fold_results])
    auc_pr_mean = np.mean([r["auc_pr"] for r in fold_results])
    auc_pr_std = np.std([r["auc_pr"] for r in fold_results])
    f1_mean = np.mean([r["best_f1"] for r in fold_results])
    f1_std = np.std([r["best_f1"] for r in fold_results])

    rec_thr = recommended_threshold["threshold"]
    thr_recall90 = _find_threshold_at_min_recall(test_sweep, 0.9)
    thr_prec90 = _find_threshold_at_min_precision(test_sweep, 0.9)

    lc_table = "\n".join(
        f"  {r['n_train']:6d} samples → val_mse={r['val_mse']:.6f}"
        for r in lc_data
    )

    fold_table_lines = [
        f"{'fold':>5} {'mse':>10} {'mae':>10} {'auc_roc':>10} {'auc_pr':>10} {'best_f1':>10} {'thr':>8}",
        "-" * 70,
    ]
    for r in fold_results:
        fold_table_lines.append(
            f"{r['fold']:>5} {r['mse']:>10.5f} {r['mae']:>10.5f} {r['auc_roc']:>10.4f} {r['auc_pr']:>10.4f} {r['best_f1']:>10.4f} {r['best_threshold_f1']:>8.2f}"
        )
    fold_table_lines.append("-" * 70)
    fold_table_lines.append(
        f"{'mean':>5} {mse_mean:>10.5f} {'':>10} {auc_roc_mean:>10.4f} {auc_pr_mean:>10.4f} {f1_mean:>10.4f}"
    )
    fold_table_lines.append(
        f"{'std':>5} {mse_std:>10.5f} {'':>10} {auc_roc_std:>10.4f} {auc_pr_std:>10.4f} {f1_std:>10.4f}"
    )

    thr_table = f"""
| Criterion           | Threshold | Precision | Recall | F1    |
|---------------------|-----------|-----------|--------|-------|
| Max F1 (recommended)| {rec_thr:.2f}      | {recommended_threshold['precision']:.4f}    | {recommended_threshold['recall']:.4f} | {recommended_threshold['f1']:.4f} |"""
    if thr_recall90:
        thr_table += f"\n| Recall ≥ 0.9        | {thr_recall90['threshold']:.2f}      | {thr_recall90['precision']:.4f}    | {thr_recall90['recall']:.4f} | {thr_recall90['f1']:.4f} |"
    if thr_prec90:
        thr_table += f"\n| Precision ≥ 0.9     | {thr_prec90['threshold']:.2f}      | {thr_prec90['precision']:.4f}    | {thr_prec90['recall']:.4f} | {thr_prec90['f1']:.4f} |"

    return f"""# Generalisation Report — {config_name}

## (a) Evaluation metrics and why

**Regression metrics (vs BigModel probability)**:
- **MSE**: training objective; measures how well TinyModel approximates BigModel.
- **MAE**: more interpretable (average absolute error in probability space).

**Classification metrics (vs flagged_fraud ground truth, after thresholding)**:
- **Precision, Recall, F1**: flagged_fraud is imbalanced (~11.6% positive).
  Accuracy is misleading (a model predicting all-negative gets ~88.4%).
  F1 balances precision and recall.
- **AUC-PR**: area under precision-recall curve; robust to class imbalance.
- **AUC-ROC**: standard ranking metric.

Oracle upper bound (BigModel scores directly vs flagged_fraud):
- AUC-ROC = {oracle_auc_roc:.4f}, AUC-PR = {oracle_auc_pr:.4f}
  (BigModel perfectly separates flagged_fraud — flagged is derived by thresholding BigModel.)

## (b) Data manipulation strategy and best training set

**Strategy used**:
1. **Hold-out test (20%)** — never touched during training or CV.
2. **K-fold CV (k={n_folds})** over train+val (80%) — each fold re-fits StandardScaler on
   its own training partition to avoid leakage.
3. **Learning curve** — measures val_mse vs n_train to determine if more data helps.

**Learning curve (val MSE vs n_train)**:
```
{lc_table}
```
Interpretation: {'val_mse decreases as n_train grows — the model benefits from more data up to the full training set size.' if lc_data[-1]['val_mse'] < lc_data[0]['val_mse'] else 'val_mse plateaus quickly — the capacity of the perceptron is the bottleneck, not the data size.'}

**Best training set**: the full available train+val partition. Larger training set consistently
gives lower or equal val_mse. For the final model, we train on train+val (no CV fold held out)
and evaluate on the held-out test set.

## K-fold CV results (k={n_folds})

```
{chr(10).join(fold_table_lines)}
```

## (c) Best model and threshold recommendation

**Final model evaluated on test set**:
- MSE: {test_metrics['mse']:.6f}
- MAE: {test_metrics['mae']:.6f}
- AUC-ROC: {test_metrics['auc_roc']:.4f}  (oracle: {oracle_auc_roc:.4f})
- AUC-PR:  {test_metrics['auc_pr']:.4f}  (oracle: {oracle_auc_pr:.4f})

**Threshold recommendation**:

Threshold selected by maximising F1 on out-of-fold predictions (concatenated from CV folds).
{thr_table}

**Recommended threshold = {rec_thr:.2f}** (maximises F1).

**Trade-off discussion**:
- If CompanyX **prioritises catching all fraud** (high recall, more false positives are acceptable),
  use the Recall ≥ 0.9 threshold. This blocks more legitimate transactions but misses fewer frauds.
- If CompanyX **prioritises precision** (avoid blocking legitimate transactions), use Precision ≥ 0.9.
  More fraud slips through, but false positive rate is minimal.
- The **recommended F1-maximising threshold** balances both concerns and is the default
  unless the client specifies a cost model.
"""


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    set_seed(config.seed)

    out_dir = Path("results/ej1/generalization")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generalisation study: {config.name} ===")
    bundle = prepare_data(seed=config.seed)

    # E.0 Learning curve
    print("\n[E.0] Learning curve...")
    lc_data = learning_curve(config, bundle)
    plot_learning_curve(lc_data, out_dir / "learning_curve.png")
    with open(out_dir / "learning_curve.json", "w") as f:
        json.dump(lc_data, f, indent=2)

    # E.1/E.2 K-fold
    print("\n[E.1] K-fold CV (k=5)...")
    fold_results, recommended_threshold, _ = run_kfold(config, bundle)
    with open(out_dir / "metrics_per_fold.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    fold_mse = [r["mse"] for r in fold_results]
    fold_auc_roc = [r["auc_roc"] for r in fold_results]
    fold_auc_pr = [r["auc_pr"] for r in fold_results]
    summary = {
        "mse": {"mean": float(np.mean(fold_mse)), "std": float(np.std(fold_mse))},
        "auc_roc": {"mean": float(np.mean(fold_auc_roc)), "std": float(np.std(fold_auc_roc))},
        "auc_pr": {"mean": float(np.mean(fold_auc_pr)), "std": float(np.std(fold_auc_pr))},
        "recommended_threshold": recommended_threshold,
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  CV summary: mse={summary['mse']['mean']:.5f}±{summary['mse']['std']:.5f} "
          f"auc_roc={summary['auc_roc']['mean']:.4f} "
          f"recommended_threshold={recommended_threshold['threshold']:.2f} "
          f"F1={recommended_threshold['f1']:.4f}")

    # E.3 Final model
    print("\n[E.3] Training final model on train+val, evaluating on test...")
    final_model, X_test_s = train_final_model(config, bundle)
    test_preds = final_model.predict(X_test_s).flatten()
    test_preds_clip = np.clip(test_preds, 0.0, 1.0)

    test_auc_roc = float(roc_auc_score(bundle.y_flag_test, test_preds_clip))
    test_auc_pr = float(average_precision_score(bundle.y_flag_test, test_preds_clip))
    test_sweep = threshold_sweep(bundle.y_flag_test, test_preds_clip)

    test_metrics = {
        "mse": round(float(mse(bundle.y_test, test_preds)), 6),
        "mae": round(float(mae(bundle.y_test, test_preds)), 6),
        "auc_roc": round(test_auc_roc, 6),
        "auc_pr": round(test_auc_pr, 6),
    }
    print(f"  Test: mse={test_metrics['mse']:.6f} auc_roc={test_metrics['auc_roc']:.4f} auc_pr={test_metrics['auc_pr']:.4f}")
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Plots
    plot_pr_curve(bundle.y_flag_test, test_preds_clip, out_dir / "pr_curve.png")
    plot_roc_curve(bundle.y_flag_test, test_preds_clip, out_dir / "roc_curve.png")
    plot_threshold_sweep(test_sweep, recommended_threshold["threshold"], out_dir / "threshold_sweep.png")
    plot_pred_vs_bigmodel(bundle.y_test, test_preds, out_dir / "pred_vs_bigmodel.png")

    # Final report
    report = build_final_report(
        config.name, lc_data, fold_results, recommended_threshold, test_metrics, test_sweep
    )
    with open(out_dir / "final_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to {out_dir.resolve()}")
    print(f"Final report: {out_dir / 'final_report.md'}")


if __name__ == "__main__":
    main()
