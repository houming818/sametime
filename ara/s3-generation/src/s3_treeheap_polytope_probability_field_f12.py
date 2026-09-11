#!/usr/bin/env python3
"""Audit whether TreeHeap depth-local loss geometry is multi-directional."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import torch
import torch.nn.functional as F

import s3_pretrain_task_posterior_pipeline as c10
import s3_recursive_depth_pressure_protocol_training as d07
import s3_filter_guided_theta_calibration_f04 as f04
import s3_structural_protocol_full_pipeline_d10 as d10
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-POLYTOPE-PROBABILITY-FIELD-F12"
NATIVE_SCALE = math.sqrt(0.5)
U_LIMIT = 0.5
MAX_MERGES = 5


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


class GroupwiseOrthogonalFold:
    """Apply one bounded orthogonal angle per channel group and merge depth."""

    def __init__(self, raw: torch.Tensor, groups: int):
        self.raw = raw
        self.groups = groups
        self.max_energy_relative_error = 0.0
        self.active_nodes = 0

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        batch, _, dim = slots.shape
        if batch != self.raw.shape[0] or dim % self.groups:
            raise ValueError(f"invalid grouped fold shape: slots={slots.shape}, raw={self.raw.shape}")
        if self.raw.shape[1:] != (MAX_MERGES, self.groups):
            raise ValueError(f"invalid raw coordinate shape: {self.raw.shape}")
        width_per_group = dim // self.groups
        levels, masks = [slots], [slot_mask]
        node, valid = slots, slot_mask
        merge = 0
        while node.shape[1] > 1:
            if merge >= MAX_MERGES:
                raise RuntimeError("TreeHeap requires more merge levels than F12 registered")
            left, right = node[:, 0::2], node[:, 1::2]
            left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
            both = left_valid & right_valid
            nodes = left.shape[1]
            left_group = left.reshape(batch, nodes, self.groups, width_per_group)
            right_group = right.reshape(batch, nodes, self.groups, width_per_group)
            common = (left_group + right_group) * NATIVE_SCALE
            difference = (left_group - right_group) * NATIVE_SCALE
            u = U_LIMIT * torch.tanh(self.raw[:, merge, :])[:, None, :, None]
            inv_norm = torch.rsqrt(1.0 + u.square())
            parent_pair = ((common + u * difference) * inv_norm).reshape_as(left)
            detail_pair = ((-u * common + difference) * inv_norm).reshape_as(left)

            child = torch.where(left_valid[:, :, None], left, right)
            parent = torch.where(both[:, :, None], parent_pair, child)
            next_valid = left_valid | right_valid
            parent = parent * next_valid[:, :, None]
            if bool(both.any()):
                before = left.square().sum(-1) + right.square().sum(-1)
                after = parent_pair.square().sum(-1) + detail_pair.square().sum(-1)
                relative = ((after - before).abs() / before.clamp_min(1e-12))[both]
                self.max_energy_relative_error = max(
                    self.max_energy_relative_error, float(relative.max().detach().cpu()),
                )
                self.active_nodes += int(both.sum().detach().cpu())
            levels.append(parent)
            masks.append(next_valid)
            node, valid = parent, next_valid
            merge += 1
        return list(reversed(levels)), list(reversed(masks))


class GroupwiseFoldRouter:
    def __init__(self, raw_base: torch.Tensor, raw_extra: torch.Tensor, groups: int, extra_dim: int):
        self.raw_base = raw_base
        self.raw_extra = raw_extra
        self.groups = groups
        self.extra_dim = extra_dim
        self.folds: list[GroupwiseOrthogonalFold] = []
        self.call_dims: list[int] = []

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        dim = slots.shape[-1]
        if dim == 256:
            raw = self.raw_base
        elif dim == self.extra_dim:
            raw = self.raw_extra
        else:
            raise ValueError(f"unexpected TreeHeap protocol dimension: {dim}")
        fold = GroupwiseOrthogonalFold(raw, self.groups)
        result = fold(slots, slot_mask)
        self.folds.append(fold)
        self.call_dims.append(dim)
        return result


@contextmanager
def routed_groupwise_fold(raw_base, raw_extra, groups: int, extra_dim: int):
    original = d07.fold_protocol
    router = GroupwiseFoldRouter(raw_base, raw_extra, groups, extra_dim)
    d07.fold_protocol = router
    try:
        yield router
    finally:
        d07.fold_protocol = original


def batch_forward(model, rows, pad, bos, device, depth, raw_base=None, raw_extra=None, groups=8):
    source, lengths, target = c10.collate_rows(rows, pad, device)
    if raw_base is None:
        logits, _, _, _, _ = model.teacher(source, lengths, target, bos, depth)
        router = None
    else:
        with routed_groupwise_fold(raw_base, raw_extra, groups, model.extra_dim) as router:
            logits, _, _, _, _ = model.teacher(source, lengths, target, bos, depth)
        if router.call_dims != [256, model.extra_dim]:
            raise RuntimeError(f"unexpected FOLD call order: {router.call_dims}")
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), target, ignore_index=pad, reduction="none",
    )
    counts = target.ne(pad).sum(1).clamp_min(1)
    losses = token_loss.sum(1) / counts
    return losses, logits, router


def project_scalar_subspace(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.mean(-1, keepdim=True).expand_as(gradient)


def normalized_descent(gradient: torch.Tensor) -> torch.Tensor:
    flat = gradient.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (-flat / norm).reshape_as(gradient)


def matrix_geometry(matrix: torch.Tensor) -> dict:
    matrix = matrix.double()
    norms = matrix.norm(dim=1)
    valid = norms > 1e-12
    matrix = matrix[valid]
    if matrix.shape[0] < 2:
        return {"valid_examples": int(matrix.shape[0]), "insufficient": True}
    unit = matrix / matrix.norm(dim=1, keepdim=True)
    cosine = unit @ unit.T
    upper = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    energy_sum = energy.sum().clamp_min(1e-24)
    centered = matrix - matrix.mean(0, keepdim=True)
    centered_singular = torch.linalg.svdvals(centered)
    centered_energy = centered_singular.square()
    centered_sum = centered_energy.sum().clamp_min(1e-24)
    return {
        "valid_examples": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "gradient_norm_mean": float(norms[valid].mean()),
        "pair_cosine_mean": float(upper.mean()),
        "pair_cosine_median": float(upper.median()),
        "pair_cosine_min": float(upper.min()),
        "pair_cosine_max": float(upper.max()),
        "negative_cosine_fraction": float((upper < 0).double().mean()),
        "pc1_explained_uncentered": float(energy[0] / energy_sum),
        "effective_rank_uncentered": float(energy_sum.square() / energy.square().sum().clamp_min(1e-24)),
        "pc1_explained_centered": float(centered_energy[0] / centered_sum),
        "singular_values": [float(value) for value in singular],
    }


def row_text(row, sp) -> tuple[str, str]:
    pair, direction = row[3], row[2]
    return (pair[1], pair[0]) if direction == "en2zh" else (pair[0], pair[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12301)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--step-norms", type=float, nargs="+", default=(0.5, 1.0))
    args = parser.parse_args()

    if 256 % args.groups or 384 % args.groups:
        raise ValueError("groups must divide both 256 and 384")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)

    payload, base_args, sp, model, pieces, eos, bos, source_hash = load_runtime(
        checkpoint, args.device,
    )
    if model.extra is None or model.extra_dim != 384:
        raise RuntimeError(f"F12 expects TreeHeap-106M base+extra dimensions, got {model.extra_dim}")
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    rows, _, pairs = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )

    geometry = {}
    example_rows = []
    origin_max_logit_delta = 0.0
    origin_max_nll_delta = 0.0
    max_energy_error = 0.0
    nonzero_finite_gradient = True
    protocol_finite = {}

    for depth in DEPTHS:
        depth_grad_base, depth_grad_extra = [], []
        finite_accumulator = {str(step): {"grouped": [], "scalar": []} for step in args.step_norms}
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start:batch_start + args.batch_size]
            size = len(batch)
            with torch.no_grad():
                native_losses, native_logits, _ = batch_forward(
                    model, batch, pieces, bos, args.device, depth,
                )
            raw_base = torch.zeros(
                size, MAX_MERGES, args.groups, device=args.device, requires_grad=True,
            )
            raw_extra = torch.zeros_like(raw_base, requires_grad=True)
            zero_losses, zero_logits, router = batch_forward(
                model, batch, pieces, bos, args.device, depth,
                raw_base, raw_extra, args.groups,
            )
            origin_max_logit_delta = max(
                origin_max_logit_delta,
                float((zero_logits.detach() - native_logits).abs().max().cpu()),
            )
            origin_max_nll_delta = max(
                origin_max_nll_delta,
                float((zero_losses.detach() - native_losses).abs().max().cpu()),
            )
            zero_losses.sum().backward()
            grad_base = raw_base.grad.detach()
            grad_extra = raw_extra.grad.detach()
            gradients_finite = bool(torch.isfinite(grad_base).all() and torch.isfinite(grad_extra).all())
            gradients_nonzero = float(grad_base.abs().max()) > 1e-12 or float(grad_extra.abs().max()) > 1e-12
            nonzero_finite_gradient &= gradients_finite and gradients_nonzero
            depth_grad_base.append(grad_base.cpu())
            depth_grad_extra.append(grad_extra.cpu())
            max_energy_error = max(
                max_energy_error,
                max((fold.max_energy_relative_error for fold in router.folds), default=0.0),
            )

            full = torch.cat((grad_base, grad_extra), dim=1)
            scalar = project_scalar_subspace(full)
            full_direction = normalized_descent(full)
            scalar_direction = normalized_descent(scalar)
            scalar_fraction = scalar.flatten(1).square().sum(1) / full.flatten(1).square().sum(1).clamp_min(1e-24)

            for step_norm in args.step_norms:
                grouped_raw = full_direction * step_norm
                scalar_raw = scalar_direction * step_norm
                with torch.no_grad():
                    grouped_losses, _, _ = batch_forward(
                        model, batch, pieces, bos, args.device, depth,
                        grouped_raw[:, :MAX_MERGES], grouped_raw[:, MAX_MERGES:], args.groups,
                    )
                    scalar_losses, _, _ = batch_forward(
                        model, batch, pieces, bos, args.device, depth,
                        scalar_raw[:, :MAX_MERGES], scalar_raw[:, MAX_MERGES:], args.groups,
                    )
                finite_accumulator[str(step_norm)]["grouped"].append(grouped_losses.cpu())
                finite_accumulator[str(step_norm)]["scalar"].append(scalar_losses.cpu())

            for index, row in enumerate(batch):
                source_text, target_text = row_text(row, sp)
                record = {
                    "protocol_depth": depth,
                    "row_index": batch_start + index,
                    "line_no": row[4],
                    "direction": row[2],
                    "source": source_text,
                    "target": target_text,
                    "native_nll": float(native_losses[index].cpu()),
                    "gradient_norm": float(full[index].norm().cpu()),
                    "scalar_energy_fraction": float(scalar_fraction[index].cpu()),
                }
                for step_norm in args.step_norms:
                    key = str(step_norm)
                    record[f"grouped_nll_{key}"] = float(
                        finite_accumulator[key]["grouped"][-1][index]
                    )
                    record[f"scalar_nll_{key}"] = float(
                        finite_accumulator[key]["scalar"][-1][index]
                    )
                example_rows.append(record)

        grad_base = torch.cat(depth_grad_base)
        grad_extra = torch.cat(depth_grad_extra)
        combined = torch.cat((grad_base, grad_extra), dim=1)
        scalar_projection = project_scalar_subspace(combined)
        depth_geometry = {
            "all_coordinates": matrix_geometry(combined.flatten(1)),
            "scalar_energy_fraction": float(
                scalar_projection.square().sum() / combined.square().sum().clamp_min(1e-24)
            ),
            "per_merge": {},
            "finite_steps": {},
        }
        for merge in range(MAX_MERGES):
            matrix = torch.cat((grad_base[:, merge], grad_extra[:, merge]), dim=1)
            depth_geometry["per_merge"][str(merge)] = matrix_geometry(matrix)
        native_depth = torch.tensor([
            row["native_nll"] for row in example_rows if row["protocol_depth"] == depth
        ])
        for step_norm in args.step_norms:
            key = str(step_norm)
            grouped_losses = torch.cat(finite_accumulator[key]["grouped"])
            scalar_losses = torch.cat(finite_accumulator[key]["scalar"])
            depth_geometry["finite_steps"][key] = {
                "native_mean_nll": float(native_depth.mean()),
                "grouped_mean_nll": float(grouped_losses.mean()),
                "scalar_mean_nll": float(scalar_losses.mean()),
                "grouped_minus_native": float((grouped_losses - native_depth).mean()),
                "scalar_minus_native": float((scalar_losses - native_depth).mean()),
                "grouped_minus_scalar": float((grouped_losses - scalar_losses).mean()),
                "grouped_win_fraction": float((grouped_losses < scalar_losses).float().mean()),
            }
        geometry[str(depth)] = depth_geometry
        protocol_finite[str(depth)] = {
            "gradient_norm_mean": depth_geometry["all_coordinates"]["gradient_norm_mean"],
            "scalar_energy_fraction": depth_geometry["scalar_energy_fraction"],
        }
        print(json.dumps({
            "event": "depth_complete", "depth": depth,
            "geometry": depth_geometry["all_coordinates"],
            "scalar_energy_fraction": depth_geometry["scalar_energy_fraction"],
            "finite_steps": depth_geometry["finite_steps"],
        }, ensure_ascii=False), flush=True)

    multi_cells = []
    for depth, depth_row in geometry.items():
        for merge, row in depth_row["per_merge"].items():
            if row.get("insufficient"):
                continue
            if (
                row["pc1_explained_uncentered"] < 0.80
                and row["effective_rank_uncentered"] > 1.5
                and row["negative_cosine_fraction"] > 0.10
            ):
                multi_cells.append({"protocol_depth": int(depth), "merge_depth": int(merge)})
    grouped_depths = []
    for depth, depth_row in geometry.items():
        if any(
            value["grouped_minus_scalar"] < -1e-4
            and value["grouped_win_fraction"] > 0.60
            for value in depth_row["finite_steps"].values()
        ):
            grouped_depths.append(int(depth))
    global_scalar_fraction = float(sum(
        row["gradient_norm"] ** 2 * row["scalar_energy_fraction"] for row in example_rows
    ) / max(1e-24, sum(row["gradient_norm"] ** 2 for row in example_rows)))

    gates = {
        "O0_exact_origin": origin_max_logit_delta <= 1e-7 and origin_max_nll_delta <= 1e-7,
        "O1_finite_gradient_and_energy": nonzero_finite_gradient and max_energy_error <= 1e-5,
        "O2_frozen_checkpoint": f04.frozen_exact(model, expected_model),
        "P1_multi_direction_cells": len(multi_cells) >= 2,
        "P2_grouped_finite_step_advantage": len(grouped_depths) >= 2,
        "P3_scalar_subspace_insufficient": global_scalar_fraction < 0.80,
    }
    summary = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "config": vars(args),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": f04.sha256(checkpoint),
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_sha256": source_hash,
        "wmt_rows": len(rows),
        "wmt_unique_pairs": len(pairs),
        "origin_max_logit_delta": origin_max_logit_delta,
        "origin_max_nll_delta": origin_max_nll_delta,
        "max_energy_relative_error": max_energy_error,
        "geometry": geometry,
        "protocol_summary": protocol_finite,
        "multi_direction_cells": multi_cells,
        "grouped_advantage_depths": grouped_depths,
        "global_scalar_energy_fraction": global_scalar_fraction,
        "gates": gates,
        "seconds": time.time() - started,
        "claim_boundary": (
            "frozen-checkpoint, target-conditioned local geometry audit; no learned router, "
            "production FOLD replacement, or literal polygon claim"
        ),
    }
    write_json(output / "summary.json", summary)
    write_csv(output / "examples.csv", example_rows)
    write_json(output / "contract.json", {
        "claim": CLAIM,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_state_sha256": summary["checkpoint_state_sha256"],
        "source_sha256": source_hash,
        "eval_wmt_data": args.eval_wmt_data,
        "rows": len(rows),
        "groups": args.groups,
        "merge_depths": MAX_MERGES,
        "protocol_depths": list(DEPTHS),
        "step_norms": args.step_norms,
        "seed": args.seed,
    })
    (output / "README.md").write_text(
        "# F12 TreeHeap polytope probability-field audit\n\n"
        f"- Claim: `{CLAIM}`\n"
        f"- Gates: `{json.dumps(gates, ensure_ascii=False)}`\n"
        f"- Multi-direction cells: `{json.dumps(multi_cells, ensure_ascii=False)}`\n"
        f"- Grouped-advantage depths: `{grouped_depths}`\n"
        f"- Global scalar energy fraction: `{global_scalar_fraction:.8f}`\n"
        f"- Runtime seconds: `{summary['seconds']:.3f}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete", "gates": gates,
        "multi_direction_cells": multi_cells,
        "grouped_advantage_depths": grouped_depths,
        "global_scalar_energy_fraction": global_scalar_fraction,
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)
    if not all(gates[key] for key in ("O0_exact_origin", "O1_finite_gradient_and_energy", "O2_frozen_checkpoint")):
        raise SystemExit(5)


if __name__ == "__main__":
    main()
