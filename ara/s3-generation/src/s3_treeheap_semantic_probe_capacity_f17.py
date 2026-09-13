#!/usr/bin/env python3
"""Recheck F16 semantic recoverability with a covariance-aware ridge probe."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_cross_language_semantic_recoverability_f16 as f16
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-SEMANTIC-PROBE-CAPACITY-F17"
EXPECTED_MANIFEST = "940c7d11f201c7df45c9ac37f0482d567134d34090f5ced4a51cdf0f5eff61ed"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path, sp) -> tuple[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != payload.get("sha256") or digest != EXPECTED_MANIFEST:
        raise ValueError(f"manifest mismatch: {digest} != {payload.get('sha256')}")
    restored = []
    for row in rows:
        token_ids = sp.encode(row["zh"], out_type=int)
        if len(token_ids) != row["source_pieces"] or not 2 <= len(token_ids) <= 30:
            raise ValueError(f"token contract mismatch at corpus line {row['line']}")
        restored.append({**row, "token_ids": token_ids})
    return digest, restored


def ridge_predictions(features, labels, train_mask, test_mask, controls, ridge, device):
    x_train = features[train_mask].to(device=device, dtype=torch.float64)
    x_test = features[test_mask].to(device=device, dtype=torch.float64)
    true_train = labels[train_mask]
    label_columns = [true_train, *controls]
    targets = torch.stack(label_columns, dim=1).to(device=device, dtype=torch.float64)
    target_means = targets.mean(0, keepdim=True)
    centered_targets = targets - target_means

    mean = x_train.mean(0)
    scale = x_train.std(0, unbiased=False)
    active = scale > 1e-8
    train = (x_train[:, active] - mean[active]) / scale[active]
    test = (x_test[:, active] - mean[active]) / scale[active]
    regularization = ridge * max(1, train.shape[1])
    if train.shape[1] > train.shape[0]:
        gram = train @ train.T
        coefficient = torch.linalg.solve(
            gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype, device=device),
            centered_targets,
        )
        predictions = test @ train.T @ coefficient + target_means
    else:
        gram = train.T @ train
        coefficient = torch.linalg.solve(
            gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype, device=device),
            train.T @ centered_targets,
        )
        predictions = test @ coefficient + target_means
    return predictions.cpu(), int(active.sum())


def metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    labels = labels.long()
    prediction = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "auroc": f16.auc(scores, labels),
        "average_precision": f16.average_precision(scores, labels),
        "balanced_accuracy": float(0.5 * (
            prediction[positive].double().mean() + (~prediction[negative]).double().mean()
        )),
        "score_gap": float(scores[positive].mean() - scores[negative].mean()),
        "finite": bool(torch.isfinite(scores).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--f16-summary", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12801)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload, _, sp, model, pieces, eos, _, source_hash = load_runtime(
        Path(args.checkpoint), args.device,
    )
    f04.freeze_model(model)
    expected_state = payload["trainable_state_dict"]
    manifest_hash, rows = load_manifest(Path(args.cases), sp)
    f16_summary = json.loads(Path(args.f16_summary).read_text(encoding="utf-8"))
    if f16_summary.get("sample_manifest_sha256") != manifest_hash:
        raise ValueError("F16 summary and case manifest disagree")

    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    model_features = f16.extract_features(model, rows, pieces, eos, args.batch_size, args.device)
    all_features = {"input_bow": f16.input_bow(rows, pieces)}
    for depth, stages in model_features.items():
        for stage, levels in stages.items():
            for level, values in levels.items():
                all_features[f"d{depth}:{stage}:{level}"] = values

    result_rows = []
    finite = True
    family_names = [name for name, _ in f16.FAMILIES]
    for held_out in family_names:
        train_indices = [index for index, row in enumerate(rows) if row["family"] != held_out]
        test_indices = [index for index, row in enumerate(rows) if row["family"] == held_out]
        train_mask = torch.zeros(len(rows), dtype=torch.bool)
        test_mask = torch.zeros(len(rows), dtype=torch.bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        controls = [
            f16.shuffled_train_labels(rows, train_indices, args.seed + 1000 * repeat + len(held_out))
            for repeat in range(args.shuffle_repeats)
        ]
        test_labels = labels[test_mask]
        for feature_name, values in all_features.items():
            predictions, active = ridge_predictions(
                values, labels, train_mask, test_mask, controls, args.ridge, args.device,
            )
            observed = metrics(predictions[:, 0], test_labels)
            control_metrics = [metrics(predictions[:, index], test_labels) for index in range(1, predictions.shape[1])]
            control_aurocs = [row["auroc"] for row in control_metrics]
            finite = finite and observed["finite"] and all(row["finite"] for row in control_metrics)
            result_rows.append({
                "held_out_family": held_out,
                "feature": feature_name,
                **observed,
                "active_features": active,
                "shuffled_auroc_mean": sum(control_aurocs) / len(control_aurocs),
                "shuffled_auroc_max": max(control_aurocs),
            })
    write_csv(output / "probe_results.csv", result_rows)

    def macro(feature_suffix: str, field: str = "auroc") -> float:
        values = [row[field] for row in result_rows if row["feature"].endswith(feature_suffix)]
        return sum(values) / len(values)

    fold_root = macro(":fold:root")
    fold_leaf = macro(":fold:leaf")
    read_root = macro(":read_facing:root")
    read_leaf = macro(":read_facing:leaf")
    root_controls = [
        row["shuffled_auroc_mean"] for row in result_rows
        if row["feature"].endswith(":read_facing:root")
    ]
    f16_macro = f16_summary["macro_auroc"]
    counts = {
        family: {str(label): sum(row["family"] == family and row["label"] == label for row in rows) for label in (0, 1)}
        for family in family_names
    }
    gates = {
        "O0_f16_manifest_contract": (
            len(rows) == 480 and len({row["line"] for row in rows}) == 480
            and all(value == 40 for family in counts.values() for value in family.values())
        ),
        "O1_finite": finite,
        "O2_frozen_checkpoint": f04.frozen_exact(model, expected_state),
        "C0_shuffled_control_rejected": sum(root_controls) / len(root_controls) < 0.60,
        "P1_fold_root_semantic_recoverable": fold_root >= 0.65,
        "P2_read_root_semantic_recoverable": read_root >= 0.65,
        "P3_fold_compression_loss_pattern": fold_leaf >= 0.70 and fold_root <= fold_leaf - 0.10,
        "P3_read_compression_loss_pattern": read_leaf >= 0.70 and read_root <= read_leaf - 0.10,
        "P4_fold_root_probe_underfit": fold_root >= f16_macro["fold_root"] + 0.05,
        "P4_read_root_probe_underfit": read_root >= f16_macro["read_facing_root"] + 0.05,
    }
    if gates["P3_fold_compression_loss_pattern"]:
        diagnosis = "strong probe supports FOLD compression loss"
    elif gates["P3_read_compression_loss_pattern"]:
        diagnosis = "strong probe isolates READ-facing root degradation"
    elif gates["P1_fold_root_semantic_recoverable"] and gates["P2_read_root_semantic_recoverable"]:
        diagnosis = "strong probe recovers root target-label direction; decoder alignment remains"
    elif fold_leaf < 0.70 and read_leaf < 0.70:
        diagnosis = "strong probe remains weak at leaf and root; build semantic protocol before FOLD change"
    else:
        diagnosis = "mixed strong-probe result; inspect family-specific transfer"

    summary = {
        "claim": CLAIM,
        "host": os.uname().nodename,
        "checkpoint": args.checkpoint,
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_state_sha256": source_hash,
        "sample_manifest_sha256": manifest_hash,
        "case_count": len(rows),
        "family_counts": counts,
        "protocol_depths": list(DEPTHS),
        "ridge": args.ridge,
        "macro_auroc": {
            "input_bow": macro("input_bow"),
            "fold_root": fold_root,
            "fold_leaf": fold_leaf,
            "read_facing_root": read_root,
            "read_facing_leaf": read_leaf,
        },
        "f16_mean_difference_macro_auroc": f16_macro,
        "gates": gates,
        "diagnosis": diagnosis,
        "seconds": time.time() - started,
        "claim_boundary": "fixed-manifest ridge recoverability, not generation quality",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        f"# {CLAIM}\n\n- Cases: `{len(rows)}`\n- Macro AUROC: `{summary['macro_auroc']}`\n"
        f"- Gates: `{gates}`\n- Diagnosis: `{diagnosis}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
