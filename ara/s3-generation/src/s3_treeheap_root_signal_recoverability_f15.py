#!/usr/bin/env python3
"""Separate TreeHeap root information loss from decoder readout failure."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import torch

import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_coherent_accumulation_audit_f14 as f14
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-ROOT-SIGNAL-RECOVERABILITY-F15"
SOURCE_SLOTS = 8
SOURCE_WIDTH = 16


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_cases(push_id: int, neutral_id: int) -> list[dict]:
    cases = []
    ordinal_by_count = {count: 0 for count in range(SOURCE_SLOTS + 1)}
    for bits in range(1 << SOURCE_SLOTS):
        positions = [index for index in range(SOURCE_SLOTS) if bits & (1 << index)]
        count = len(positions)
        ordinal = ordinal_by_count[count]
        ordinal_by_count[count] += 1
        split = "train"
        if 0 < count < SOURCE_SLOTS and ordinal % 4 == 0:
            split = "test"
        tokens = [push_id if index in positions else neutral_id for index in range(SOURCE_SLOTS)]
        cases.append({
            "id": f"mask-{bits:03d}",
            "bits": bits,
            "count": count,
            "positions": positions,
            "split": split,
            "tokens": tokens,
        })
    return cases


def masked_mean(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(nodes.dtype)
    return (nodes * weight[:, :, None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1.0)


def extract_level_features(model, protocol) -> dict[str, list[torch.Tensor]]:
    base_levels, base_masks, _, _, _, extra = protocol
    stages = {
        "fold": (base_levels, base_masks),
        "read_facing": (model.reconstructor.convolve(base_levels, base_masks), base_masks),
    }
    extra_stages = None
    if extra is not None:
        extra_levels, extra_masks, _, _ = extra
        extra_stages = {
            "fold": (extra_levels, extra_masks),
            "read_facing": (model.extra.reconstructor.convolve(extra_levels, extra_masks), extra_masks),
        }

    output = {}
    for stage, (levels, masks) in stages.items():
        features = []
        for level, (nodes, mask) in enumerate(zip(levels, masks)):
            pooled = masked_mean(nodes, mask)
            if extra_stages is not None:
                extra_levels, extra_masks = extra_stages[stage]
                if len(extra_levels) != len(levels):
                    raise ValueError("base/extra level count mismatch")
                pooled = torch.cat(
                    (pooled, masked_mean(extra_levels[level], extra_masks[level])), dim=-1,
                )
            features.append(pooled.detach().float().cpu())
        output[stage] = features
    return output


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def ridge_fit_predict(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    ridge: float,
    permute_seed: int | None = None,
) -> dict:
    x_train = features[train_mask].double()
    x_test = features[test_mask].double()
    y_train = labels[train_mask].double()
    y_test = labels[test_mask].double()

    mean = x_train.mean(0)
    scale = x_train.std(0, unbiased=False)
    active = scale > 1e-8
    x_train = (x_train[:, active] - mean[active]) / scale[active]
    x_test = (x_test[:, active] - mean[active]) / scale[active]
    y_mean = y_train.mean()
    fit_labels = y_train - y_mean
    if permute_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(permute_seed)
        fit_labels = fit_labels[torch.randperm(fit_labels.shape[0], generator=generator)]

    if x_train.shape[1] == 0:
        prediction = torch.full_like(y_test, y_mean)
    else:
        regularization = ridge * max(1, x_train.shape[1])
        if x_train.shape[1] > x_train.shape[0]:
            gram = x_train @ x_train.T
            dual = torch.linalg.solve(
                gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype),
                fit_labels,
            )
            coefficient = x_train.T @ dual
        else:
            gram = x_train.T @ x_train
            coefficient = torch.linalg.solve(
                gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype),
                x_train.T @ fit_labels,
            )
        prediction = x_test @ coefficient + y_mean

    residual = (prediction - y_test).square().sum()
    baseline = (y_test - y_test.mean()).square().sum().clamp_min(1e-12)
    rounded = (prediction * SOURCE_SLOTS).round().clamp(0, SOURCE_SLOTS)
    target_count = y_test * SOURCE_SLOTS
    finite = bool(torch.isfinite(prediction).all())
    return {
        "r2": float(1.0 - residual / baseline),
        "mae_count": float((prediction - y_test).abs().mean() * SOURCE_SLOTS),
        "correlation": correlation(prediction, y_test),
        "exact_count_accuracy": float((rounded == target_count).double().mean()),
        "active_features": int(active.sum()),
        "finite": finite,
    }


def aggregate_margin(cases: list[dict], margins: list[float]) -> dict[str, float]:
    output = {}
    for count in range(SOURCE_SLOTS + 1):
        values = [margin for case, margin in zip(cases, margins) if case["count"] == count]
        output[str(count)] = sum(values) / len(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--permutation-seed", type=int, default=12601)
    args = parser.parse_args()

    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload, _, sp, model, pieces, eos, bos, source_hash = load_runtime(
        Path(args.checkpoint), args.device,
    )
    f04.freeze_model(model)
    expected_state = payload["trainable_state_dict"]

    push_source = sp.encode("推", out_type=int)
    neutral_source = sp.encode("看", out_type=int)
    push_target = sp.encode("push", out_type=int)
    token_contract = {
        "push_source": push_source,
        "neutral_source": neutral_source,
        "push_target": push_target,
        "push_target_pieces": [sp.id_to_piece(value) for value in push_target],
    }
    if not (len(push_source) == len(neutral_source) == len(push_target) == 1):
        raise ValueError(f"single-token contract failed: {token_contract}")

    cases = make_cases(push_source[0], neutral_source[0])
    direction_id = pieces + 2
    source, lengths = f14.collate_cases(cases, direction_id, eos, pieces, args.device)
    labels = torch.tensor([case["count"] / SOURCE_SLOTS for case in cases], dtype=torch.float64)
    train_mask = torch.tensor([case["split"] == "train" for case in cases])
    test_mask = ~train_mask
    target = torch.tensor([[push_target[0], eos]] * len(cases), dtype=torch.long, device=args.device)
    write_json(output / "cases.json", {"token_contract": token_contract, "cases": cases})

    probe_rows = []
    margin_rows = []
    finite = True
    for protocol_depth in DEPTHS:
        with torch.no_grad():
            protocols = {"native": model.protocol(source, lengths, protocol_depth)}
            with f14.routed_scale(0.5):
                protocols["mean"] = model.protocol(source, lengths, protocol_depth)

        for mode, protocol in protocols.items():
            with torch.no_grad():
                stage_features = extract_level_features(model, protocol)
                root_logits, _, = f14.combined_cutoff(model, protocol, bos, 0)
            target_metrics = f14.token_metrics(root_logits, push_target[0])
            margins = [row["target_margin"] for row in target_metrics]
            margin_by_count = aggregate_margin(cases, margins)
            for count, margin in margin_by_count.items():
                margin_rows.append({
                    "mode": mode,
                    "protocol_depth": protocol_depth,
                    "count": int(count),
                    "root_target_margin": margin,
                })
            finite = finite and all(math.isfinite(value) for value in margins)

            for stage, levels in stage_features.items():
                for level, features in enumerate(levels):
                    metrics = ridge_fit_predict(
                        features, labels, train_mask, test_mask, args.ridge,
                    )
                    shuffled = ridge_fit_predict(
                        features, labels, train_mask, test_mask, args.ridge,
                        permute_seed=args.permutation_seed + 100 * protocol_depth + level,
                    )
                    finite = finite and metrics["finite"] and shuffled["finite"]
                    probe_rows.append({
                        "mode": mode,
                        "protocol_depth": protocol_depth,
                        "stage": stage,
                        "level": level,
                        "level_from_leaf": len(levels) - 1 - level,
                        "node_count": int(protocol[0][level].shape[1]),
                        "feature_width": int(features.shape[1]),
                        "test_r2": metrics["r2"],
                        "test_mae_count": metrics["mae_count"],
                        "test_correlation": metrics["correlation"],
                        "test_exact_count_accuracy": metrics["exact_count_accuracy"],
                        "active_features": metrics["active_features"],
                        "shuffled_test_r2": shuffled["r2"],
                    })

    write_csv(output / "probe_results.csv", probe_rows)
    write_csv(output / "root_decoder_margins.csv", margin_rows)

    def select(mode: str, stage: str, level: int = 0) -> list[dict]:
        return [
            row for row in probe_rows
            if row["mode"] == mode and row["stage"] == stage and row["level"] == level
        ]

    native_fold_root = select("native", "fold")
    native_read_root = select("native", "read_facing")
    mean_fold_root = select("mean", "fold")
    mean_read_root = select("mean", "read_facing")
    train_counts = {case["count"] for case in cases if case["split"] == "train"}
    test_counts = {case["count"] for case in cases if case["split"] == "test"}
    gates = {
        "O0_exhaustive_position_holdout": (
            len(cases) == 256
            and train_counts == set(range(9))
            and test_counts == set(range(1, 8))
            and source.shape[1] == SOURCE_WIDTH
            and bool((lengths == SOURCE_SLOTS + 2).all())
        ),
        "O1_finite": finite,
        "O2_frozen_checkpoint": f04.frozen_exact(model, expected_state),
        "C0_shuffled_control_rejected": max(
            row["shuffled_test_r2"] for row in native_read_root
        ) < 0.20,
        "P1_native_fold_root_recoverable": min(
            row["test_r2"] for row in native_fold_root
        ) >= 0.80,
        "P2_native_read_root_recoverable": min(
            row["test_r2"] for row in native_read_root
        ) >= 0.80,
    }
    if not gates["P1_native_fold_root_recoverable"]:
        diagnosis = "supports fold-level root information loss"
    elif not gates["P2_native_read_root_recoverable"]:
        diagnosis = "fold preserves count signal; decoder convolution loses linear recoverability"
    else:
        diagnosis = "source count reaches read-facing root; F14 failure is access/alignment/scale, not erasure"

    summary = {
        "claim": CLAIM,
        "host": os.uname().nodename,
        "checkpoint": args.checkpoint,
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_state_sha256": source_hash,
        "token_contract": token_contract,
        "case_count": len(cases),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "protocol_depths": list(DEPTHS),
        "native_fold_root": native_fold_root,
        "native_read_facing_root": native_read_root,
        "mean_fold_root": mean_fold_root,
        "mean_read_facing_root": mean_read_root,
        "gates": gates,
        "diagnosis": diagnosis,
        "seconds": time.time() - started,
        "claim_boundary": "synthetic source-count recoverability; not target-language semantic proof",
    }
    write_json(output / "summary.json", summary)
    readme = [
        f"# {CLAIM}",
        "",
        f"- Cases: `{len(cases)}`; train/test: `{int(train_mask.sum())}/{int(test_mask.sum())}`",
        f"- Gates: `{gates}`",
        f"- Diagnosis: `{diagnosis}`",
        f"- Runtime seconds: `{summary['seconds']:.2f}`",
        "",
        "This frozen synthetic probe tests source-count recoverability, not translation quality.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
