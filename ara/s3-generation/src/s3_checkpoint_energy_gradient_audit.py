#!/usr/bin/env python3
"""Audit local versus energy-carrying FOLD on a frozen real-text checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


EPSILON = 1e-8
WIDTHS = (4, 8, 16, 32, 64, 128, 256)


def tensor_stats(values: torch.Tensor):
    values = values.detach().double().flatten().cpu()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.0, 0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float64),
    )
    names = ("min", "p01", "p05", "p50", "p90", "p95", "p99", "max")
    return {
        **{name: float(value) for name, value in zip(names, quantiles)},
        "mean": float(values.mean()),
        "count": int(values.numel()),
    }


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten().cpu()
    right = right.detach().double().flatten().cpu()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float((left * right).sum() / denominator)


def expand_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.empty(
        left.shape[0], left.shape[1] * 2, left.shape[2],
        dtype=left.dtype, device=left.device,
    )
    result[:, 0::2], result[:, 1::2] = left, right
    return result


def pair_metrics(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor, parent: torch.Tensor):
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    cancellation = (left + right).norm(dim=-1) / (left_norm + right_norm).clamp_min(1e-30)
    return {
        "cancellation": cancellation,
        "scale": scale.squeeze(-1),
        "inverse_scale": scale.squeeze(-1).reciprocal(),
        "parent_norm": parent.norm(dim=-1),
    }


def fold_local(leaf: torch.Tensor):
    node = leaf
    levels = [node]
    details, scales, records = [], [], []
    while node.shape[1] > 1:
        left, right = node[:, 0::2], node[:, 1::2]
        scale = torch.sqrt(
            left.square().sum(-1, keepdim=True)
            + right.square().sum(-1, keepdim=True)
            + EPSILON
        )
        parent = (left + right) / (math.sqrt(2.0) * scale)
        detail = (right - left) / (math.sqrt(2.0) * scale)
        records.append(pair_metrics(left, right, scale, parent))
        details.append(detail)
        scales.append(scale)
        node = parent
        levels.append(node)
    return levels, details, scales, records


def unfold_local(root: torch.Tensor, details, scales):
    node = root
    for detail, scale in zip(reversed(details), reversed(scales)):
        left = scale * (node - detail) / math.sqrt(2.0)
        right = scale * (node + detail) / math.sqrt(2.0)
        node = expand_pair(left, right)
    return node


def fold_carrier(leaf: torch.Tensor):
    energy = torch.sqrt(leaf.square().sum(-1, keepdim=True) + EPSILON)
    direction = leaf / energy
    levels, energies = [direction], [energy]
    details, child_energies, ratios, records = [], [], [], []
    while direction.shape[1] > 1:
        left_u, right_u = direction[:, 0::2], direction[:, 1::2]
        left_e, right_e = energy[:, 0::2], energy[:, 1::2]
        left_x, right_x = left_e * left_u, right_e * right_u
        parent_e = torch.sqrt(left_e.square() + right_e.square())
        parent_u = (left_x + right_x) / (math.sqrt(2.0) * parent_e)
        detail = (right_x - left_x) / (math.sqrt(2.0) * parent_e)
        records.append(pair_metrics(left_x, right_x, parent_e, parent_u))
        details.append(detail)
        child_energies.append((left_e, right_e))
        ratios.append((left_e / parent_e, right_e / parent_e))
        direction, energy = parent_u, parent_e
        levels.append(direction)
        energies.append(energy)
    return levels, energies, details, child_energies, ratios, records


def unfold_carrier(root_u, root_e, details, child_energies):
    direction, energy = root_u, root_e
    for detail, (left_e, right_e) in zip(reversed(details), reversed(child_energies)):
        left_x = energy * (direction - detail) / math.sqrt(2.0)
        right_x = energy * (direction + detail) / math.sqrt(2.0)
        direction = expand_pair(left_x / left_e, right_x / right_e)
        energy = expand_pair(left_e, right_e)
    return direction * energy


def path_energy_error(energies, ratios):
    reconstructed = energies[-1]
    for left_ratio, right_ratio in reversed(ratios):
        reconstructed = expand_pair(
            reconstructed * left_ratio,
            reconstructed * right_ratio,
        )
    return float((reconstructed - energies[0]).abs().max().detach())


def root_gradient(root: torch.Tensor, leaf: torch.Tensor):
    probe = torch.linspace(-1.0, 1.0, root.shape[-1], dtype=root.dtype, device=root.device)
    probe = probe / probe.norm()
    objective = (root[:, 0] * probe).sum()
    gradient = torch.autograd.grad(objective, leaf, retain_graph=False)[0]
    return gradient.flatten(1).norm(dim=-1)


def append_records(target, records):
    for depth, record in enumerate(records):
        for name, values in record.items():
            target[depth][name].append(values.detach().cpu())


def summarize_depth_records(records):
    output = []
    for depth in sorted(records):
        merged = {
            name: torch.cat([part.flatten() for part in parts])
            for name, parts in records[depth].items()
        }
        cancellation = merged["cancellation"]
        output.append({
            "depth_from_leaf": depth,
            **{name: tensor_stats(values) for name, values in merged.items()},
            "cancellation_rates": {
                "lt_0_01": float((cancellation < 0.01).double().mean()),
                "lt_0_05": float((cancellation < 0.05).double().mean()),
                "lt_0_10": float((cancellation < 0.10).double().mean()),
            },
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=16221)
    parser.add_argument("--collect-rows", type=int, default=1024)
    parser.add_argument("--per-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    started = time.time()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["device"] = args.device
    model_args = SimpleNamespace(**config)
    sp = spm.SentencePieceProcessor(model_file=model_args.spm_model)
    pieces = sp.get_piece_size()
    pad, vocab = pieces, pieces + 3
    model = integrated.build_integrated_model(model_args, vocab, pad)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    c10.CONTEXT_WIDTHS = WIDTHS
    rows, collection = c10.collect_pretrain_rows(
        model_args, sp, "valid", args.collect_rows, args.seed,
    )
    by_width = defaultdict(list)
    for row in rows:
        if len(by_width[len(row[0])]) < args.per_width:
            by_width[len(row[0])].append(row)
    selected = {width: by_width[width] for width in WIDTHS if by_width[width]}

    local_depth_records = defaultdict(lambda: defaultdict(list))
    carrier_depth_records = defaultdict(lambda: defaultdict(list))
    sample_widths, sample_min_q = [], []
    local_gradients, carrier_gradients = [], []
    closure_local, closure_carrier, path_errors = [], [], []
    root_norm_local, root_norm_carrier = [], []

    encoder = model.encoder
    with torch.no_grad():
        for width in sorted(selected):
            bucket = selected[width]
            for start in range(0, len(bucket), args.batch_size):
                source, length, _ = c10.collate_rows(bucket[start:start + args.batch_size], pad, args.device)
                leaf, mask = encoder.inner.raw_leaf(source, length)
                leaf = encoder.inner.communication(leaf, mask, "butterfly")
                leaf64 = leaf.detach().double()

                local_leaf = leaf64.clone().requires_grad_(True)
                with torch.enable_grad():
                    local_levels, local_details, local_scales, local_records = fold_local(local_leaf)
                    local_grad = root_gradient(local_levels[-1], local_leaf)
                local_restored = unfold_local(
                    local_levels[-1].detach(),
                    [value.detach() for value in local_details],
                    [value.detach() for value in local_scales],
                )

                carrier_leaf = leaf64.clone().requires_grad_(True)
                with torch.enable_grad():
                    carrier = fold_carrier(carrier_leaf)
                    carrier_levels, energies, details, child_energies, ratios, carrier_records = carrier
                    carrier_grad = root_gradient(carrier_levels[-1], carrier_leaf)
                carrier_restored = unfold_carrier(
                    carrier_levels[-1].detach(), energies[-1].detach(),
                    [value.detach() for value in details],
                    [(left.detach(), right.detach()) for left, right in child_energies],
                )

                append_records(local_depth_records, local_records)
                append_records(carrier_depth_records, carrier_records)
                per_sample_q = torch.stack([
                    record["cancellation"].amin(dim=1) for record in local_records
                ], dim=1).amin(dim=1)
                count = leaf.shape[0]
                sample_widths.extend([width] * count)
                sample_min_q.append(per_sample_q.detach().cpu())
                local_gradients.append(local_grad.detach().cpu())
                carrier_gradients.append(carrier_grad.detach().cpu())
                root_norm_local.append(local_levels[-1][:, 0].norm(dim=-1).detach().cpu())
                root_norm_carrier.append(carrier_levels[-1][:, 0].norm(dim=-1).detach().cpu())
                closure_local.append(float((local_restored - leaf64).abs().max()))
                closure_carrier.append(float((carrier_restored - leaf64).abs().max()))
                path_errors.append(path_energy_error(energies, ratios))

    min_q = torch.cat(sample_min_q)
    local_grad = torch.cat(local_gradients)
    carrier_grad = torch.cat(carrier_gradients)
    gradient_ratio = local_grad / carrier_grad.clamp_min(1e-30)
    width_tensor = torch.tensor(sample_widths, dtype=torch.long)
    cancellation_signal = -torch.log10(min_q.clamp_min(1e-12))
    gradient_signal = torch.log10(gradient_ratio.clamp_min(1e-12))
    all_cancellation = torch.cat([
        torch.cat([part.flatten() for part in parts])
        for depth in local_depth_records.values()
        for name, parts in depth.items() if name == "cancellation"
    ])

    max_closure = max(closure_local + closure_carrier)
    max_path_error = max(path_errors)
    cancellation_rate = float((all_cancellation < 0.05).double().mean())
    local_stats = tensor_stats(local_grad)
    carrier_stats = tensor_stats(carrier_grad)
    ratio_stats = tensor_stats(gradient_ratio)
    correlation = pearson(cancellation_signal, gradient_signal)
    by_width_summary = {}
    for width in sorted(set(sample_widths)):
        selected_width = width_tensor.eq(width)
        by_width_summary[str(width)] = {
            "samples": int(selected_width.sum()),
            "minimum_cancellation": tensor_stats(min_q[selected_width]),
            "local_root_gradient": tensor_stats(local_grad[selected_width]),
            "carrier_root_gradient": tensor_stats(carrier_grad[selected_width]),
            "local_over_carrier_gradient_ratio": tensor_stats(gradient_ratio[selected_width]),
        }
    gates = {
        "A0_numerically_valid": (
            max_closure < 1e-6
            and max_path_error < 1e-8
            and bool(torch.isfinite(local_grad).all())
            and bool(torch.isfinite(carrier_grad).all())
        ),
        "A1_real_cancellation_tail": cancellation_rate >= 0.001,
        "A2_current_p99_at_least_2x_carrier": local_stats["p99"] >= 2.0 * carrier_stats["p99"],
        "A3_cancellation_gradient_ratio_correlation": correlation >= 0.2,
    }
    proceed = gates["A0_numerically_valid"] and gates["A1_real_cancellation_tail"] and (
        gates["A2_current_p99_at_least_2x_carrier"]
        or gates["A3_cancellation_gradient_ratio_correlation"]
    )
    payload = {
        "audit": "S3-STONE2-CHECKPOINT-ENERGY-GRADIENT-AUDIT",
        "host": socket.gethostname(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_state_sha256": checkpoint.get("state_sha256"),
        "seed": args.seed,
        "dtype": "float64-fold-audit",
        "selected_rows_by_width": {str(width): len(bucket) for width, bucket in selected.items()},
        "collection": collection,
        "local_depths": summarize_depth_records(local_depth_records),
        "carrier_depths": summarize_depth_records(carrier_depth_records),
        "sample_metrics": {
            "samples": len(sample_widths),
            "minimum_cancellation": tensor_stats(min_q),
            "local_root_gradient": local_stats,
            "carrier_root_gradient": carrier_stats,
            "local_over_carrier_gradient_ratio": ratio_stats,
            "cancellation_to_gradient_ratio_log_pearson": correlation,
            "local_root_norm": tensor_stats(torch.cat(root_norm_local)),
            "carrier_root_norm": tensor_stats(torch.cat(root_norm_carrier)),
            "by_width": by_width_summary,
        },
        "numerical": {
            "max_local_closure_abs": max(closure_local),
            "max_carrier_closure_abs": max(closure_carrier),
            "max_path_product_energy_error": max_path_error,
        },
        "gates": gates,
        "decision": "register_short_training_ablation" if proceed else "do_not_train_energy_carrier_yet",
        "seconds": time.time() - started,
        "boundary": (
            "Frozen real-text encoder/Jacobian audit only. No decoder adaptation, language-quality "
            "claim, S7 claim, or architecture promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "gates": gates, "decision": payload["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
