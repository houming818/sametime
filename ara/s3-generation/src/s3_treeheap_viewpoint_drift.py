#!/usr/bin/env python3
"""Capture and compare read-only Butterfly viewpoint traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from s2_treeheap_butterfly_wmt import ButterflyRecursive, pair_indices  # noqa: E402


FORMAT_VERSION = 1
DIRECTIONS = ("en2zh", "zh2en")


def stable_split(direction: str, source: str) -> str:
    digest = hashlib.blake2b(
        f"{direction}\0{source}".encode("utf-8"), digest_size=8,
    ).digest()
    return "calibration" if int.from_bytes(digest, "big") % 2 == 0 else "heldout"


def parse_dreams(path: Path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or parts[0] not in DIRECTIONS:
            raise ValueError(f"malformed Dreams row {path}:{line_no}")
        rows.append({
            "id": f"dream-{len(rows) + 1:03d}",
            "direction": parts[0],
            "source": parts[1],
            "reference": parts[2] if len(parts) > 2 and parts[2] else None,
            "split": stable_split(parts[0], parts[1]),
        })
    if not rows:
        raise ValueError(f"no probes found in {path}")
    if {row["split"] for row in rows} != {"calibration", "heldout"}:
        raise ValueError("probe hash split must contain calibration and heldout rows")
    return rows


def load_model(checkpoint_path: Path, sp, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    pieces = sp.get_piece_size()
    pad = pieces
    model = ButterflyRecursive(
        vocab=pieces + 3,
        dim=int(cfg["dim"]),
        hidden=int(cfg["hidden"]),
        heap_width=int(cfg["heap_width"]),
        pad=pad,
        mode="butterfly",
        scale=float(cfg["coupling_scale"]),
        dynamic_width=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, pad


@torch.no_grad()
def communication_trace(model, source: torch.Tensor, length: torch.Tensor):
    encoder = model.encoder
    result, mask = encoder.raw_leaf(source, length)
    states = [result.detach().cpu()]
    active_depths = int(math.log2(result.shape[1]))
    schedule = encoder.communication.schedule("butterfly", active_depths)
    for step, address_stage in enumerate(schedule):
        left_index, right_index = pair_indices(
            result.shape[1], address_stage, result.device,
        )
        left, right = result[:, left_index], result[:, right_index]
        active = (mask[:, left_index] & mask[:, right_index])[:, :, None]
        gain = encoder.communication._gain(step)
        right_next = right + gain * encoder.communication.forward_kernel(left)
        left_next = left + gain * encoder.communication.backward_kernel(right_next)
        updated = result.clone()
        updated[:, left_index] = torch.where(active, left_next, left)
        updated[:, right_index] = torch.where(active, right_next, right)
        result = updated
        states.append(result.detach().cpu())
    return result, mask, states


@torch.no_grad()
def capture(args):
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    model, checkpoint, pad = load_model(Path(args.checkpoint), sp, args.device)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    records = []
    max_logit_diff = 0.0
    trace_final_diff = 0.0
    greedy_stable = True

    for row in parse_dreams(Path(args.dreams)):
        source_ids = [
            direction_ids[row["direction"]],
            *sp.encode(row["source"], out_type=int),
            eos,
        ]
        if len(source_ids) > model.encoder.heap_width:
            raise ValueError(f"{row['id']} exceeds heap width: {len(source_ids)}")
        source = torch.tensor([source_ids], dtype=torch.long, device=args.device)
        length = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)

        reference = row["reference"] or row["source"]
        target_ids = [*sp.encode(reference, out_type=int), eos]
        target = torch.tensor([target_ids], dtype=torch.long, device=args.device)
        logits_before, _ = model.teacher(source, length, target, bos)
        generated_before, _ = model.greedy(source, length, bos, eos, args.max_output)

        traced_final, leaf_mask, stages = communication_trace(model, source, length)
        leaf, root, details, masks = model.encoder.fold(source, length)
        trace_final_diff = max(
            trace_final_diff, float((traced_final - leaf).abs().max().cpu()),
        )

        logits_after, _ = model.teacher(source, length, target, bos)
        generated_after, route = model.greedy(
            source, length, bos, eos, args.max_output,
        )
        max_logit_diff = max(
            max_logit_diff, float((logits_before - logits_after).abs().max().cpu()),
        )
        greedy_stable = greedy_stable and torch.equal(generated_before, generated_after)

        row.update({
            "source_ids": source_ids,
            "target_ids": target_ids,
            "mask": leaf_mask[0].detach().cpu(),
            "stages": [state[0] for state in stages],
            "root": root[0].detach().cpu(),
            "details": [detail[0].detach().cpu() for detail in details],
            "detail_masks": [mask[0].detach().cpu() for mask in masks],
            "generated_ids": generated_after[0].detach().cpu(),
            "generated_text": sp.decode(generated_after[0].tolist()),
            "route": route.detach().cpu(),
        })
        records.append(row)

    report = {
        "format_version": FORMAT_VERSION,
        "claim": "S3-TREEHEAP-VIEW-DRIFT-C04",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "global_examples": checkpoint.get("global_examples"),
        "config": checkpoint["config"],
        "probe_count": len(records),
        "max_logit_diff": max_logit_diff,
        "trace_final_diff": trace_final_diff,
        "greedy_tokens_identical": greedy_stable,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(report, output)
    print(json.dumps({
        "output": str(output),
        "probe_count": len(records),
        "max_logit_diff": max_logit_diff,
        "trace_final_diff": trace_final_diff,
        "greedy_tokens_identical": greedy_stable,
        "P5_read_only": (
            max_logit_diff <= 1e-6
            and trace_final_diff <= 1e-6
            and greedy_stable
        ),
    }, ensure_ascii=False, indent=2))


def vectors_at_stage(report, split: str, stage: int):
    values = []
    for row in report["records"]:
        if row["split"] != split or stage >= len(row["stages"]):
            continue
        values.append(row["stages"][stage][row["mask"]].float())
    return torch.cat(values, dim=0) if values else None


def normalized_rmse(prediction: torch.Tensor, target: torch.Tensor):
    error = (prediction - target).square().mean().sqrt()
    scale = target.square().mean().sqrt().clamp_min(1e-12)
    return float(error / scale)


def fit_orthogonal(x: torch.Tensor, y: torch.Tensor):
    x_mean, y_mean = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd((x - x_mean).T @ (y - y_mean), full_matrices=False)
    return u @ vh, x_mean, y_mean


def compare(args):
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    left_keys = [(r["direction"], r["source"]) for r in left["records"]]
    right_keys = [(r["direction"], r["source"]) for r in right["records"]]
    if left_keys != right_keys:
        raise ValueError("trace files do not contain the same ordered probes")

    max_stage = min(
        max(len(row["stages"]) for row in left["records"]),
        max(len(row["stages"]) for row in right["records"]),
    )
    rows = []
    for stage in range(max_stage):
        x_cal = vectors_at_stage(left, "calibration", stage)
        y_cal = vectors_at_stage(right, "calibration", stage)
        x_test = vectors_at_stage(left, "heldout", stage)
        y_test = vectors_at_stage(right, "heldout", stage)
        if any(value is None for value in (x_cal, y_cal, x_test, y_test)):
            continue
        if x_cal.shape != y_cal.shape or x_test.shape != y_test.shape:
            raise ValueError(f"stage {stage} trace shape mismatch")
        rotation, x_mean, y_mean = fit_orthogonal(x_cal, y_cal)
        raw = normalized_rmse(x_test, y_test)
        aligned = normalized_rmse((x_test - x_mean) @ rotation + y_mean, y_test)
        rows.append({
            "stage": stage,
            "calibration_vectors": int(x_cal.shape[0]),
            "heldout_vectors": int(x_test.shape[0]),
            "raw_nrmse": raw,
            "aligned_nrmse": aligned,
            "alignment_gain": 1.0 - aligned / max(raw, 1e-12),
        })

    summary = {
        "claim": "S3-TREEHEAP-VIEW-DRIFT-C04",
        "left": args.left,
        "right": args.right,
        "stages": rows,
        "P2_two_stages_gain_20pct": sum(
            row["alignment_gain"] >= 0.20 for row in rows
        ) >= 2,
        "interpretation": (
            "Alignment supports coordinate recoverability only. Grammar/fact "
            "annotations are required before calling it semantic viewpoint drift."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def self_test():
    generator = torch.Generator().manual_seed(8104)
    x_cal = torch.randn(512, 32, generator=generator)
    x_test = torch.randn(256, 32, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator))
    offset = torch.randn(1, 32, generator=generator)
    y_cal, y_test = x_cal @ q + offset, x_test @ q + offset
    rotation, x_mean, y_mean = fit_orthogonal(x_cal, y_cal)
    raw = normalized_rmse(x_test, y_test)
    aligned = normalized_rmse((x_test - x_mean) @ rotation + y_mean, y_test)
    result = {"raw_nrmse": raw, "aligned_nrmse": aligned, "passed": aligned < 1e-5}
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--checkpoint", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    capture_parser.add_argument("--dreams", default="ara/s3-generation/dreams.txt")
    capture_parser.add_argument("--max-output", type=int, default=128)
    capture_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    capture_parser.set_defaults(func=capture)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(func=compare)

    test_parser = sub.add_parser("self-test")
    test_parser.set_defaults(func=lambda _args: self_test())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
