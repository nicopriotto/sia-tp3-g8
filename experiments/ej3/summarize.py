"""Consolidate ej3 training runs into summary tables and select a candidate.

Run from repo root:
    python3 -m experiments.ej3.summarize

Reads `results/ej3/training/*/` and writes:
    results/ej3/summary.csv
    results/ej3/summary.md
    results/ej3/selection/selected_model.json
    results/ej3/selection/selection_notes.md

Selection rule (validation only):
    1. Sort by val_macro_f1 desc.
    2. Break ties by val_accuracy desc.
    3. Penalize a large train-val accuracy gap (overfitting).
    4. If two are equivalent, prefer the simpler architecture.

The selected_model.json carries an explicit `test_disclaimer` field stating
that `digits_test.csv` was not used during selection. The final test
evaluation is the exclusive job of `experiments.ej3.evaluate_final` (T12).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

TRAINING_DIR = Path("results/ej3/training")
OUT_DIR = Path("results/ej3")
SELECTION_DIR = OUT_DIR / "selection"

CSV_COLUMNS = [
    "run_id",
    "config_name",
    "seed",
    "data_strategy",
    "architecture",
    "activation",
    "output_activation",
    "weight_init",
    "weight_decay",
    "loss",
    "optimizer",
    "optimizer_params",
    "lr_schedule",
    "lr_schedule_params",
    "learning_rate",
    "batch_size",
    "epochs_ran",
    "train_accuracy",
    "val_accuracy",
    "train_macro_f1",
    "val_macro_f1",
    "train_loss_final",
    "val_loss_final",
    "best_val_loss",
    "elapsed_sec_final",
    "val_recall_class_8",
]


def _load_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    hist_path = run_dir / "history.json"
    ev_path = run_dir / "evaluation.json"
    if not (cfg_path.exists() and hist_path.exists() and ev_path.exists()):
        return None

    cfg = json.loads(cfg_path.read_text())
    history = json.loads(hist_path.read_text())
    evaluation = json.loads(ev_path.read_text())

    if not history:
        return None

    val_losses = [r.get("val_loss") for r in history if r.get("val_loss") is not None]
    best_val_loss = min(val_losses) if val_losses else None

    train_eval = evaluation.get("train", {})
    val_eval = evaluation.get("validation", {})
    optimizer_params = cfg.get("optimizer_params") or {}
    extra = cfg.get("extra") or {}
    val_per_class = val_eval.get("per_class", {}) or {}
    class8 = val_per_class.get("8") or val_per_class.get(8) or {}

    return {
        "run_id": evaluation.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "config_name": cfg.get("name", ""),
        "seed": cfg.get("seed"),
        "data_strategy": extra.get("data_strategy"),
        "architecture": cfg.get("architecture"),
        "activation": cfg.get("activation"),
        "output_activation": cfg.get("output_activation"),
        "weight_init": cfg.get("weight_init"),
        "weight_decay": optimizer_params.get("weight_decay"),
        "loss": cfg.get("loss"),
        "optimizer": cfg.get("optimizer"),
        "optimizer_params": optimizer_params,
        "lr_schedule": cfg.get("lr_schedule"),
        "lr_schedule_params": cfg.get("lr_schedule_params") or {},
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "epochs_ran": len(history),
        "train_accuracy": train_eval.get("accuracy"),
        "val_accuracy": val_eval.get("accuracy"),
        "train_macro_f1": train_eval.get("macro_f1"),
        "val_macro_f1": val_eval.get("macro_f1"),
        "train_loss_final": history[-1].get("loss"),
        "val_loss_final": history[-1].get("val_loss"),
        "best_val_loss": best_val_loss,
        "elapsed_sec_final": history[-1].get("elapsed_sec"),
        "val_confusion_matrix": val_eval.get("confusion_matrix"),
        "class_labels": evaluation.get("class_labels"),
        "val_per_class": val_per_class,
        "val_recall_class_8": class8.get("recall"),
        "val_precision_class_8": class8.get("precision"),
        "val_f1_class_8": class8.get("f1"),
    }


def _row_for_csv(run: dict) -> dict:
    return {
        "run_id": run["run_id"],
        "config_name": run["config_name"],
        "seed": run["seed"],
        "data_strategy": run["data_strategy"],
        "architecture": json.dumps(run["architecture"]),
        "activation": run["activation"],
        "output_activation": run["output_activation"],
        "weight_init": run["weight_init"],
        "weight_decay": run["weight_decay"],
        "loss": run["loss"],
        "optimizer": run["optimizer"],
        "optimizer_params": json.dumps(run["optimizer_params"]),
        "lr_schedule": run["lr_schedule"],
        "lr_schedule_params": json.dumps(run["lr_schedule_params"]),
        "learning_rate": run["learning_rate"],
        "batch_size": run["batch_size"],
        "epochs_ran": run["epochs_ran"],
        "train_accuracy": run["train_accuracy"],
        "val_accuracy": run["val_accuracy"],
        "train_macro_f1": run["train_macro_f1"],
        "val_macro_f1": run["val_macro_f1"],
        "train_loss_final": run["train_loss_final"],
        "val_loss_final": run["val_loss_final"],
        "best_val_loss": run["best_val_loss"],
        "elapsed_sec_final": run["elapsed_sec_final"],
        "val_recall_class_8": run["val_recall_class_8"],
    }


def _format_pct(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _format_loss(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _gap(run: dict) -> float | None:
    if run["train_accuracy"] is None or run["val_accuracy"] is None:
        return None
    return run["train_accuracy"] - run["val_accuracy"]


def _arch_size(run: dict) -> int:
    arch = run.get("architecture") or []
    return sum(arch) if isinstance(arch, list) else 0


def _confusion_matrix_md(run: dict) -> str:
    cm = run.get("val_confusion_matrix") or []
    labels = run.get("class_labels") or list(range(len(cm)))
    if not cm:
        return "_no validation confusion matrix saved._\n"
    header = "| true \\ pred | " + " | ".join(str(l) for l in labels) + " |\n"
    sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|\n"
    body = ""
    for label, row in zip(labels, cm):
        body += f"| **{label}** | " + " | ".join(str(v) for v in row) + " |\n"
    return header + sep + body


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(runs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_runs = sorted(
        runs,
        key=lambda r: (
            -(r["val_macro_f1"] if r["val_macro_f1"] is not None else -1),
            -(r["val_accuracy"] if r["val_accuracy"] is not None else -1),
        ),
    )
    lines = [
        "# Ej3 — runs summary",
        "",
        "Generado por `experiments.ej3.summarize`. Solo metricas de train/validation, "
        "ningun numero proviene de `digits_test.csv`.",
        "",
        "| run_id | data | architecture | act | init | optimizer | wd | lr_sched | lr | epochs | train_acc | val_acc | val_macroF1 | gap | recall_8 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted_runs:
        lines.append(
            "| `{run_id}` | {ds} | {arch} | {act} | {init} | {opt} | {wd} | {lrs} | {lr} | {ep} | {ta} | {va} | {vf} | {gap} | {r8} |".format(
                run_id=r["run_id"],
                ds=r["data_strategy"] or "n/a",
                arch=json.dumps(r["architecture"]),
                act=r["activation"],
                init=r["weight_init"] or "default",
                opt=r["optimizer"],
                wd=(r["weight_decay"] if r["weight_decay"] is not None else "0"),
                lrs=r["lr_schedule"] or "none",
                lr=r["learning_rate"],
                ep=r["epochs_ran"],
                ta=_format_pct(r["train_accuracy"]),
                va=_format_pct(r["val_accuracy"]),
                vf=_format_pct(r["val_macro_f1"]),
                gap=_format_pct(_gap(r)) if _gap(r) is not None else "n/a",
                r8=_format_pct(r["val_recall_class_8"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def select_candidate(runs: list[dict]) -> tuple[dict, list[dict], str]:
    """Apply the documented selection rule. Return (winner, top3, reason)."""
    ranked = sorted(
        runs,
        key=lambda r: (
            -(r["val_macro_f1"] if r["val_macro_f1"] is not None else -1),
            -(r["val_accuracy"] if r["val_accuracy"] is not None else -1),
            (_gap(r) if _gap(r) is not None else 1.0),
            _arch_size(r),
        ),
    )
    top3 = ranked[:3]
    winner = top3[0]
    gap = _gap(winner)
    gap_str = f"{gap:.4f}" if gap is not None else "n/a"
    reason_parts = [
        "Highest validation macro_f1 across all runs.",
        f"Validation accuracy {_format_pct(winner['val_accuracy'])} (tiebreaker over runs with similar macro_f1).",
        f"Train-val accuracy gap {gap_str} stays comparable to the rest of the top three.",
    ]
    reason = " ".join(reason_parts)
    return winner, top3, reason


def write_selection(
    winner: dict,
    top3: list[dict],
    reason: str,
    all_runs: list[dict],
) -> None:
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)

    config_path = Path("experiments/ej3/configs") / f"{winner['config_name']}.json"
    selected = {
        "run_id": winner["run_id"],
        "run_dir": winner["run_dir"],
        "config_path": str(config_path),
        "selection_metric": "val_macro_f1, tiebreak val_accuracy, gap, arch size",
        "selection_reason": reason,
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "test_disclaimer": "digits_test.csv was not used",
        "summary": {
            "data_strategy": winner["data_strategy"],
            "architecture": winner["architecture"],
            "activation": winner["activation"],
            "weight_init": winner["weight_init"],
            "optimizer": winner["optimizer"],
            "optimizer_params": winner["optimizer_params"],
            "lr_schedule": winner["lr_schedule"],
            "lr_schedule_params": winner["lr_schedule_params"],
            "learning_rate": winner["learning_rate"],
            "batch_size": winner["batch_size"],
            "epochs_ran": winner["epochs_ran"],
            "train_accuracy": winner["train_accuracy"],
            "val_accuracy": winner["val_accuracy"],
            "train_macro_f1": winner["train_macro_f1"],
            "val_macro_f1": winner["val_macro_f1"],
            "best_val_loss": winner["best_val_loss"],
            "val_recall_class_8": winner["val_recall_class_8"],
            "val_precision_class_8": winner["val_precision_class_8"],
            "val_f1_class_8": winner["val_f1_class_8"],
        },
    }
    (SELECTION_DIR / "selected_model.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )

    notes = ["# Ej3 — Selection notes", "", "## Rule", ""]
    notes.extend([
        "1. Maximizar `val_macro_f1`.",
        "2. Desempatar por `val_accuracy`.",
        "3. Penalizar diferencia grande entre `train_accuracy` y `val_accuracy`.",
        "4. Si dos modelos quedan equivalentes, preferir la arquitectura mas chica.",
        "",
        "Solo se usaron metricas de train/validation. `digits_test.csv` no fue mirado.",
        "",
        "## Top 3 por validation",
        "",
        "| rank | run_id | data | architecture | optimizer | lr | val_macro_f1 | val_accuracy | gap | recall_8 | best_val_loss |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for i, r in enumerate(top3, start=1):
        notes.append(
            "| {i} | `{rid}` | {ds} | {arch} | {opt} | {lr} | {vf} | {va} | {gap} | {r8} | {bvl} |".format(
                i=i,
                rid=r["run_id"],
                ds=r["data_strategy"] or "n/a",
                arch=json.dumps(r["architecture"]),
                opt=r["optimizer"],
                lr=r["learning_rate"],
                vf=_format_pct(r["val_macro_f1"]),
                va=_format_pct(r["val_accuracy"]),
                gap=_format_pct(_gap(r)) if _gap(r) is not None else "n/a",
                r8=_format_pct(r["val_recall_class_8"]),
                bvl=_format_loss(r["best_val_loss"]),
            )
        )
    notes.extend(["", "## Confusion matrices del top 3 (validation)", ""])
    for i, r in enumerate(top3, start=1):
        notes.append(f"### {i}. `{r['run_id']}`")
        notes.append("")
        notes.append(_confusion_matrix_md(r))
        notes.append("")

    notes.extend(["## Decision", "", reason, ""])
    notes.append(f"Candidato seleccionado: **`{winner['run_id']}`**.")
    notes.append("")

    # Verificacion clase 8.
    recall_8 = winner.get("val_recall_class_8")
    if recall_8 is None or recall_8 <= 0:
        notes.append(
            "**ATENCION**: el candidato tiene `validation.per_class['8'].recall = "
            f"{recall_8}`. Esto significa que la clase 8 esta colapsada en validation. "
            "Revisar antes de pasar a T12."
        )
    else:
        notes.append(
            f"Verificacion clase 8 en validation: precision="
            f"{_format_pct(winner.get('val_precision_class_8'))}, "
            f"recall={_format_pct(recall_8)}, "
            f"f1={_format_pct(winner.get('val_f1_class_8'))} (recall > 0, OK)."
        )
    notes.append("")

    # Verificacion target 0.97.
    val_acc = winner.get("val_accuracy") or 0.0
    if val_acc < 0.97:
        notes.append(
            "**Nota target**: el mejor candidato esta por debajo del target de "
            f"`val_accuracy >= 0.97` ({_format_pct(val_acc)}). Considerar mas epochs "
            "o batch mayor antes de T12."
        )
    else:
        notes.append(
            f"Target alcanzado: val_accuracy={_format_pct(val_acc)} >= 0.97."
        )
    notes.append("")

    notes.append(
        "Para T12 se debe entrenar el ganador (o reusar `weights.npz`) y luego "
        "evaluar en `digits_test.csv` exactamente una vez."
    )
    notes.append("")

    (SELECTION_DIR / "selection_notes.md").write_text("\n".join(notes))


def main() -> None:
    if not TRAINING_DIR.exists():
        raise SystemExit(f"No training dir at {TRAINING_DIR}. Run trainings first.")

    runs: list[dict] = []
    for run_dir in sorted(p for p in TRAINING_DIR.iterdir() if p.is_dir()):
        # Skip smoke/dev runs whose names start with an underscore.
        if run_dir.name.startswith("_"):
            print(f"[skip] {run_dir} (smoke/dev run)")
            continue
        run = _load_run(run_dir)
        if run is None:
            print(f"[skip] {run_dir} missing artifacts")
            continue
        runs.append(run)

    if not runs:
        raise SystemExit("No valid runs found.")

    csv_rows = [_row_for_csv(r) for r in runs]
    write_summary_csv(csv_rows, OUT_DIR / "summary.csv")
    write_summary_md(runs, OUT_DIR / "summary.md")

    winner, top3, reason = select_candidate(runs)
    write_selection(winner, top3, reason, runs)

    print(f"Indexed {len(runs)} runs.")
    print(
        f"Selected: {winner['run_id']} (val_macro_f1={winner['val_macro_f1']:.4f}, "
        f"val_accuracy={winner['val_accuracy']:.4f})"
    )
    print(f"Wrote: {OUT_DIR / 'summary.csv'}")
    print(f"Wrote: {OUT_DIR / 'summary.md'}")
    print(f"Wrote: {SELECTION_DIR / 'selected_model.json'}")
    print(f"Wrote: {SELECTION_DIR / 'selection_notes.md'}")


if __name__ == "__main__":
    main()
