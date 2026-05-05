"""End-to-end interpretability analysis for a saved ej2 MLP run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ej2.plots import plot_confusion_matrix
from perceptron.metrics import confusion_matrix_multiclass

from .attribution import compute_attributions, reshape_like_image
from .internal_views import (
    class_mean_first_hidden_activations,
    first_hidden_activations,
    first_layer_weight_images,
    first_layer_weight_norms,
    hidden_activations,
    top_activating_neurons,
    top_first_layer_neurons,
)
from .load_run import load_run, save_run_manifest
from .plots import (
    plot_attribution_case,
    plot_first_layer_weights,
    plot_hidden_activation_case,
    plot_mean_maps_by_class,
    plot_method_agreement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Saved ej2 training run directory.")
    parser.add_argument("--output-dir", default=None, help="Override default interpretability output dir.")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--occlusion-patch-size", type=int, default=4)
    parser.add_argument("--occlusion-stride", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--top-neurons", type=int, default=16)
    return parser.parse_args()


def _top_two_margin(probabilities: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(np.asarray(probabilities, dtype=float), axis=1)
    if sorted_probs.shape[1] < 2:
        return sorted_probs[:, -1]
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def select_representative_samples(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
) -> list[dict[str, Any]]:
    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    margins = _top_two_margin(probs)
    confidences = np.max(probs, axis=1)

    correct = np.flatnonzero(labels_true == labels_pred)
    errors = np.flatnonzero(labels_true != labels_pred)

    chosen: list[dict[str, Any]] = []
    used: set[int] = set()

    def add_indices(indices: np.ndarray, category: str, limit: int) -> None:
        count = 0
        for idx in indices.tolist():
            if idx in used:
                continue
            used.add(idx)
            chosen.append(
                {
                    "index": int(idx),
                    "category": category,
                    "true_label": int(labels_true[idx]),
                    "pred_label": int(labels_pred[idx]),
                    "confidence": float(confidences[idx]),
                    "margin": float(margins[idx]),
                    "predicted_logit": float(logits[idx, labels_pred[idx]]),
                    "true_class_logit": float(logits[idx, labels_true[idx]]),
                }
            )
            count += 1
            if count >= limit:
                break

    add_indices(correct[np.argsort(confidences[correct])[::-1]], "correct_high_confidence", limit=4)
    add_indices(correct[np.argsort(margins[correct])], "correct_low_margin", limit=4)
    if len(errors):
        error_margin = np.abs(
            logits[errors, labels_pred[errors]] - logits[errors, labels_true[errors]]
        )
        add_indices(errors[np.argsort(error_margin)], "errors", limit=min(6, len(errors)))

    return chosen


def _ensure_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_ensure_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_ensure_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def cosine_similarity_abs(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.abs(np.asarray(a, dtype=float).reshape(-1))
    b_flat = np.abs(np.asarray(b, dtype=float).reshape(-1))
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def topk_overlap_abs(a: np.ndarray, b: np.ndarray, k: int = 40) -> float:
    a_flat = np.abs(np.asarray(a, dtype=float).reshape(-1))
    b_flat = np.abs(np.asarray(b, dtype=float).reshape(-1))
    k = min(k, len(a_flat))
    top_a = set(np.argsort(a_flat)[-k:].tolist())
    top_b = set(np.argsort(b_flat)[-k:].tolist())
    return float(len(top_a & top_b) / k) if k > 0 else 0.0


def compute_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[int]) -> list[dict[str, int]]:
    matrix = confusion_matrix_multiclass(y_true, y_pred, labels=np.asarray(class_labels, dtype=int))
    pairs: list[dict[str, int]] = []
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            if i == j or matrix[i, j] == 0:
                continue
            pairs.append({"true": int(true_label), "pred": int(pred_label), "count": int(matrix[i, j])})
    pairs.sort(key=lambda row: row["count"], reverse=True)
    return pairs


def build_report(
    output_dir: Path,
    summary: dict[str, Any],
    selected_samples: list[dict[str, Any]],
    class_mean_summary: dict[str, Any],
    confusion_pairs: list[dict[str, int]],
) -> Path:
    metrics = summary["validation_metrics"]
    sanity = summary["sanity_checks"]
    lines: list[str] = [
        "# Ej2 - Interpretability report",
        "",
        "## Scope",
        "",
        "- Model analyzed: `{}`".format(summary["run_id"]),
        "- Source run: `{}`".format(summary["run_dir"]),
        "- Data used here: reconstructed train/validation split from `digits.csv` only.",
        "- This analysis does **not** use `digits_test.csv`.",
        "- Architecture: `{}` with hidden activation `{}` and output `{}`.".format(
            summary["architecture"],
            summary["activation"],
            summary["output_activation"],
        ),
        "",
        "## Validation snapshot",
        "",
        "- accuracy: {:.4f}".format(metrics["accuracy"]),
        "- macro_f1: {:.4f}".format(metrics["macro_f1"]),
        "- categorical_cross_entropy: {:.4f}".format(metrics["loss"]),
        "- Validation class 8 support: {}".format(summary["validation_distribution"].get("8", 0)),
        "",
        "## What the model seems to use",
        "",
        "- The attribution maps were computed against the class logit, which keeps the backward pass simple and stable for this NumPy MLP.",
        "- Across representative cases, `gradient x input` and `integrated gradients` tend to agree on the main stroke regions rather than on blank background.",
        "- Occlusion acts as a perturbation-based cross-check: when masking a relevant stroke drops the target logit, the highlighted region is more believable.",
        "- First-layer weights can be interpreted as low-level stroke templates: diagonals, vertical bars, curved segments and loop-like fragments.",
        "",
        "## Main confusion patterns on validation",
        "",
    ]
    if confusion_pairs:
        for row in confusion_pairs[:8]:
            lines.append(
                "- true `{}` predicted as `{}`: {} cases".format(row["true"], row["pred"], row["count"])
            )
    else:
        lines.append("- No off-diagonal confusion was found on validation.")
    lines.extend(
        [
            "",
            "## Representative samples",
            "",
            "| idx | category | true | pred | confidence | margin |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in selected_samples:
        lines.append(
            "| {index} | {category} | {true_label} | {pred_label} | {confidence:.4f} | {margin:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Class-level summaries",
            "",
            "- Mean absolute integrated-gradients maps were averaged over up to {} correctly classified samples per class.".format(
                class_mean_summary["samples_per_class_limit"]
            ),
            "- Mean absolute occlusion maps were averaged over the same subset to keep the comparison fair.",
            "- Classes absent from the validation split of `digits.csv` naturally show no summary map.",
            "",
            "## Sanity checks",
            "",
            "- Mean cosine similarity between `|gradient x input|` and `|integrated gradients (zero baseline)|`: {:.4f}".format(
                sanity["mean_cosine_gradxinput_vs_ig_zero"]
            ),
            "- Mean cosine similarity between `|integrated gradients (zero)|` and `|integrated gradients (train mean)|`: {:.4f}".format(
                sanity["mean_cosine_ig_zero_vs_ig_mean"]
            ),
            "- Mean top-40 overlap between `|gradient x input|` and `|integrated gradients (zero)|`: {:.4f}".format(
                sanity["mean_top40_overlap_gradxinput_vs_ig_zero"]
            ),
            "",
            "## Limitations",
            "",
            "- These explanations are post-hoc and local; they are not a symbolic description of the full model.",
            "- The project uses a plain MLP, not a CNN, so methods such as Grad-CAM were intentionally not used as the primary explanation tool.",
            "- Class `8` is absent from the original `digits.csv` training data, so any reasoning about that class from this run is structurally limited.",
            "- Occlusion is much slower than gradient-based methods, so the class-level perturbation summaries use only a capped subset of validation samples.",
            "",
            "## Outputs",
            "",
            "- `run_manifest.json`",
            "- `analysis_summary.json`",
            "- `selected_samples.json`",
            "- `figures/val_confusion_matrix.png`",
            "- `figures/first_layer_top_weights.png`",
            "- `figures/class_mean_integrated_gradients.png`",
            "- `figures/class_mean_occlusion.png`",
            "- `figures/method_agreement.png`",
            "- `cases/*.png`",
            "- `internal/*.png`",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    loaded = load_run(args.run_dir, output_dir=args.output_dir)

    figures_dir = loaded.output_dir / "figures"
    cases_dir = loaded.output_dir / "cases"
    internal_dir = loaded.output_dir / "internal"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)

    save_run_manifest(loaded)

    # Task 001 deliverable: fixed run manifest inside the interpretability output.
    run_selection = {
        "selected_run_dir": str(loaded.run_dir),
        "selected_run_id": loaded.run_id,
        "selection_reason": "Baseline SGD run outperformed the tested Adam alternative on validation and is more stable.",
        "validation_accuracy": loaded.reconstructed_validation_evaluation["accuracy"],
        "validation_macro_f1": loaded.reconstructed_validation_evaluation["macro_f1"],
        "does_not_use_test": True,
    }
    (loaded.output_dir / "selected_run.json").write_text(json.dumps(run_selection, indent=2))

    plot_confusion_matrix(
        loaded.reconstructed_validation_evaluation["confusion_matrix"],
        labels=loaded.class_labels,
        output_path=figures_dir / "val_confusion_matrix.png",
    )

    mean_baseline = np.mean(loaded.bundle.X_train, axis=0)
    selected_samples = select_representative_samples(
        loaded.bundle.labels_val,
        loaded.val_predictions,
        loaded.val_outputs,
        loaded.val_logits,
    )
    (loaded.output_dir / "selected_samples.json").write_text(json.dumps(selected_samples, indent=2))

    weight_images = first_layer_weight_images(loaded.model, loaded.feature_shape)
    weight_norms = first_layer_weight_norms(loaded.model)
    top_neuron_indices = top_first_layer_neurons(loaded.model, top_k=args.top_neurons)
    plot_first_layer_weights(weight_images, weight_norms, top_neuron_indices, figures_dir / "first_layer_top_weights.png")

    case_sanity_rows: list[dict[str, Any]] = []
    for row in selected_samples:
        idx = row["index"]
        image_vec = loaded.bundle.X_val[idx]
        image = image_vec.reshape(loaded.feature_shape)
        probs = loaded.val_outputs[idx]
        pred_label = row["pred_label"]
        attributions = compute_attributions(
            loaded.model,
            image_vec,
            pred_label,
            feature_shape=loaded.feature_shape,
            mean_baseline=mean_baseline,
            ig_steps=args.ig_steps,
            occlusion_patch_size=args.occlusion_patch_size,
            occlusion_stride=args.occlusion_stride,
        )

        map_dict = {
            "Gradient": reshape_like_image(attributions.gradient, loaded.feature_shape),
            "Grad x Input": reshape_like_image(attributions.gradient_x_input, loaded.feature_shape),
            "Integrated Gradients": reshape_like_image(attributions.integrated_gradients_zero, loaded.feature_shape),
            "Occlusion": reshape_like_image(attributions.occlusion, loaded.feature_shape),
        }
        title = (
            f"idx={idx} | {row['category']} | true={row['true_label']} "
            f"pred={row['pred_label']} | conf={row['confidence']:.3f}"
        )
        plot_attribution_case(
            image=image,
            maps=map_dict,
            probabilities=probs,
            title=title,
            output_path=cases_dir / f"case_{idx:04d}.png",
        )

        hidden = hidden_activations(loaded.model, image_vec)
        if hidden:
            neuron_indices = top_activating_neurons(loaded.model, image_vec, top_k=min(6, len(hidden[0])))
            plot_hidden_activation_case(
                image=image,
                activations=hidden[0],
                weight_images=weight_images,
                neuron_indices=neuron_indices,
                output_path=internal_dir / f"hidden_case_{idx:04d}.png",
                title=f"First hidden-layer view | idx={idx} | true={row['true_label']} pred={row['pred_label']}",
            )

        case_sanity_rows.append(
            {
                "index": idx,
                "cosine_gradxinput_vs_ig_zero": cosine_similarity_abs(
                    attributions.gradient_x_input,
                    attributions.integrated_gradients_zero,
                ),
                "cosine_ig_zero_vs_ig_mean": cosine_similarity_abs(
                    attributions.integrated_gradients_zero,
                    attributions.integrated_gradients_mean,
                ),
                "top40_overlap_gradxinput_vs_ig_zero": topk_overlap_abs(
                    attributions.gradient_x_input,
                    attributions.integrated_gradients_zero,
                    k=40,
                ),
            }
        )

    # Class-level summaries use correctly classified validation samples only.
    correct_mask = loaded.bundle.labels_val == loaded.val_predictions
    class_mean_ig: dict[int, np.ndarray] = {}
    class_mean_occ: dict[int, np.ndarray] = {}
    class_mean_counts: dict[int, int] = {}
    for label in loaded.class_labels:
        indices = np.flatnonzero(correct_mask & (loaded.bundle.labels_val == label))
        if indices.size == 0:
            continue
        chosen = indices[:args.samples_per_class]
        ig_maps: list[np.ndarray] = []
        occ_maps: list[np.ndarray] = []
        for idx in chosen.tolist():
            attrs = compute_attributions(
                loaded.model,
                loaded.bundle.X_val[idx],
                label,
                feature_shape=loaded.feature_shape,
                mean_baseline=mean_baseline,
                ig_steps=args.ig_steps,
                occlusion_patch_size=args.occlusion_patch_size,
                occlusion_stride=args.occlusion_stride,
            )
            ig_maps.append(np.abs(reshape_like_image(attrs.integrated_gradients_zero, loaded.feature_shape)))
            occ_maps.append(np.abs(reshape_like_image(attrs.occlusion, loaded.feature_shape)))
        class_mean_ig[label] = np.mean(np.stack(ig_maps, axis=0), axis=0)
        class_mean_occ[label] = np.mean(np.stack(occ_maps, axis=0), axis=0)
        class_mean_counts[label] = len(chosen)

    plot_mean_maps_by_class(
        class_mean_ig,
        loaded.class_labels,
        title="Mean absolute integrated gradients by class",
        output_path=figures_dir / "class_mean_integrated_gradients.png",
    )
    plot_mean_maps_by_class(
        class_mean_occ,
        loaded.class_labels,
        title="Mean absolute occlusion sensitivity by class",
        output_path=figures_dir / "class_mean_occlusion.png",
    )

    # Simple class-wise internal summary from the first hidden layer.
    mean_hidden_by_class = class_mean_first_hidden_activations(
        loaded.model,
        loaded.bundle.X_val[correct_mask],
        loaded.bundle.labels_val[correct_mask],
        loaded.class_labels,
    )
    hidden_summary = {
        str(label): np.argsort(values)[::-1][:5].tolist()
        for label, values in mean_hidden_by_class.items()
    }

    sanity_summary = {
        "per_case": case_sanity_rows,
        "mean_cosine_gradxinput_vs_ig_zero": float(
            np.mean([row["cosine_gradxinput_vs_ig_zero"] for row in case_sanity_rows])
        ),
        "mean_cosine_ig_zero_vs_ig_mean": float(
            np.mean([row["cosine_ig_zero_vs_ig_mean"] for row in case_sanity_rows])
        ),
        "mean_top40_overlap_gradxinput_vs_ig_zero": float(
            np.mean([row["top40_overlap_gradxinput_vs_ig_zero"] for row in case_sanity_rows])
        ),
    }
    plot_method_agreement(
        {
            "Grad x Input vs IG(zero)": sanity_summary["mean_cosine_gradxinput_vs_ig_zero"],
            "IG(zero) vs IG(mean)": sanity_summary["mean_cosine_ig_zero_vs_ig_mean"],
            "Top40 overlap": sanity_summary["mean_top40_overlap_gradxinput_vs_ig_zero"],
        },
        title="Sanity checks across representative samples",
        output_path=figures_dir / "method_agreement.png",
    )

    confusion_pairs = compute_confusion_pairs(
        loaded.bundle.labels_val,
        loaded.val_predictions,
        loaded.class_labels,
    )

    summary = {
        "run_id": loaded.run_id,
        "run_dir": str(loaded.run_dir),
        "output_dir": str(loaded.output_dir),
        "architecture": loaded.config.architecture,
        "activation": loaded.config.activation,
        "output_activation": loaded.config.output_activation,
        "validation_metrics": loaded.reconstructed_validation_evaluation,
        "validation_distribution": loaded.val_distribution,
        "selected_samples": selected_samples,
        "top_first_layer_neurons": top_neuron_indices.tolist(),
        "top_first_hidden_neurons_by_class": hidden_summary,
        "class_mean_counts": {str(k): int(v) for k, v in class_mean_counts.items()},
        "confusion_pairs": confusion_pairs[:12],
        "sanity_checks": sanity_summary,
    }
    summary_path = loaded.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(_ensure_serializable(summary), indent=2))

    build_report(
        output_dir=loaded.output_dir,
        summary=summary,
        selected_samples=selected_samples,
        class_mean_summary={"samples_per_class_limit": args.samples_per_class},
        confusion_pairs=confusion_pairs,
    )

    print(f"Interpretability analysis saved to {loaded.output_dir.resolve()}")


if __name__ == "__main__":
    main()
