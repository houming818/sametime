#!/usr/bin/env python3
"""Probe a bounded orthogonal TreeHeap FOLD without training the checkpoint."""
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
import torch.nn as nn
import torch.nn.functional as F

import s3_recursive_depth_pressure_protocol_training as d07
import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_ancestor_jump_microscope_f09 as f09
import s3_cross_language_echo_controller_f08 as f08
from treeheap_epoch_translate_cli import load_runtime


CLAIM = "S3-TREEHEAP-ORTHOGONAL-ANNEALING-MICROSCOPE-F10"
RAW_AMPLITUDES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)
MODES = ("parent", "detail", "joint")
NATIVE_SCALE = math.sqrt(0.5)
U_LIMIT = 0.5
DETAIL_MIX = 0.5


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


class OrthogonalFold:
    """Recurse through p while exposing p, q, or a controlled p/q mixture."""

    def __init__(self, raw: torch.Tensor, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.raw = raw
        self.mode = mode
        self.max_energy_relative_error = 0.0
        self.active_nodes = 0
        self.u_values: list[torch.Tensor] = []

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        visible_levels = [slots]
        masks = [slot_mask]
        node, valid = slots, slot_mask
        cursor = 0
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
            both = left_valid & right_valid
            width = left.shape[1]
            local_raw = self.raw[:, cursor:cursor + width]
            if local_raw.shape[1] != width:
                raise RuntimeError("orthogonal parameter vector does not cover all parents")
            cursor += width

            u = U_LIMIT * torch.tanh(local_raw)
            inv_norm = torch.rsqrt(1.0 + u.square())
            common = (left + right) * NATIVE_SCALE
            difference = (left - right) * NATIVE_SCALE
            parent_pair = (common + u[:, :, None] * difference) * inv_norm[:, :, None]
            detail_pair = (-u[:, :, None] * common + difference) * inv_norm[:, :, None]

            child = torch.where(left_valid[:, :, None], left, right)
            parent = torch.where(both[:, :, None], parent_pair, child)
            detail = torch.where(both[:, :, None], detail_pair, torch.zeros_like(detail_pair))
            next_valid = left_valid | right_valid
            parent = parent * next_valid[:, :, None]
            detail = detail * next_valid[:, :, None]

            if bool(both.any()):
                before = left.square().sum(-1) + right.square().sum(-1)
                after = parent_pair.square().sum(-1) + detail_pair.square().sum(-1)
                relative = ((after - before).abs() / before.clamp_min(1e-12))[both]
                self.max_energy_relative_error = max(
                    self.max_energy_relative_error, float(relative.max().detach().cpu()),
                )
                self.active_nodes += int(both.sum().detach().cpu())
                self.u_values.append(u[both])

            if self.mode == "parent":
                visible = parent
            elif self.mode == "detail":
                visible = torch.where(both[:, :, None], detail, parent)
            else:
                mixed = (parent + DETAIL_MIX * detail) / math.sqrt(1.0 + DETAIL_MIX ** 2)
                visible = torch.where(both[:, :, None], mixed, parent)

            visible_levels.append(visible)
            masks.append(next_valid)
            node, valid = parent, next_valid

        if cursor != self.raw.shape[1]:
            raise RuntimeError(f"unused orthogonal coordinates: {self.raw.shape[1] - cursor}")
        return list(reversed(visible_levels)), list(reversed(masks))

    def summary(self) -> dict:
        values = torch.cat([value.detach().flatten() for value in self.u_values]) if self.u_values else None
        return {
            "mode": self.mode,
            "active_nodes": self.active_nodes,
            "max_energy_relative_error": self.max_energy_relative_error,
            "u_min": float(values.min().cpu()) if values is not None else 0.0,
            "u_max": float(values.max().cpu()) if values is not None else 0.0,
        }


class FoldRouter:
    def __init__(self, raw: torch.Tensor, mode: str):
        self.raw = raw
        self.mode = mode
        self.folds: list[OrthogonalFold] = []

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        fold = OrthogonalFold(self.raw, self.mode)
        result = fold(slots, slot_mask)
        self.folds.append(fold)
        return result

    def summary(self) -> dict:
        return {
            "calls": len(self.folds),
            "folds": [fold.summary() for fold in self.folds],
            "max_energy_relative_error": max(
                (fold.max_energy_relative_error for fold in self.folds), default=0.0,
            ),
        }


@contextmanager
def routed_fold(raw: torch.Tensor, mode: str):
    original = d07.fold_protocol
    router = FoldRouter(raw, mode)
    d07.fold_protocol = router
    try:
        yield router
    finally:
        d07.fold_protocol = original


def decode_trees(model, base_tree, base_masks, extra, bos, steps, prefixes=None):
    base_tree = model.reconstructor.convolve(base_tree, base_masks)
    base_hidden = base_tree[0].new_zeros((base_tree[0].shape[0], model.reconstructor.hidden))
    extra_tree, extra_masks, _, _ = extra
    extra_tree = model.extra.reconstructor.convolve(extra_tree, extra_masks)
    extra_hidden = extra_tree[0].new_zeros((extra_tree[0].shape[0], model.extra.reconstructor.hidden))
    previous = torch.full((base_tree[0].shape[0],), bos, dtype=torch.long, device=base_tree[0].device)
    logits_rows, predicted = [], []
    for step in range(steps):
        base_context, _ = model.reconstructor.read(base_hidden, base_tree, base_masks)
        base_hidden = model.reconstructor.cell(
            torch.cat((model.reconstructor.embedding(previous), base_context), dim=-1), base_hidden,
        )
        logits = model.reconstructor.output(torch.cat((base_hidden, base_context), dim=-1))
        extra_context, _ = model.extra.reconstructor.read(extra_hidden, extra_tree, extra_masks)
        extra_hidden = model.extra.reconstructor.cell(
            torch.cat((model.extra.reconstructor.embedding(previous), extra_context), dim=-1), extra_hidden,
        )
        extra_logits = model.extra.reconstructor.output(torch.cat((extra_hidden, extra_context), dim=-1))
        logits = logits + torch.tanh(model.extra_logit_gain) * extra_logits
        token = logits.argmax(-1)
        logits_rows.append(logits)
        predicted.append(token)
        previous = (prefixes[:, step] if prefixes is not None else token).detach()
    return torch.stack(logits_rows, dim=1), torch.stack(predicted, dim=1)


def decode(model, source, lengths, bos, depth, steps, raw=None, mode="native", prefixes=None):
    if mode == "native":
        base_tree, base_masks, budgets, _, _, extra = model.protocol(source, lengths, depth, "native")
        router_summary = None
    else:
        if raw is None:
            raise ValueError("orthogonal decode requires raw parameters")
        with routed_fold(raw, mode) as router:
            base_tree, base_masks, budgets, _, _, extra = model.protocol(source, lengths, depth, "native")
        if len(router.folds) != 2:
            raise RuntimeError(f"expected base and extra folds, got {len(router.folds)}")
        router_summary = router.summary()
    logits, tokens = decode_trees(
        model, base_tree, base_masks, extra, bos, steps, prefixes=prefixes,
    )
    return logits, tokens, budgets, router_summary


def make_raw(batch: int, amplitude: float, device: str, requires_grad=False):
    tensor = torch.full((batch, 31), amplitude, dtype=torch.float32, device=device)
    return nn.Parameter(tensor) if requires_grad else tensor


def target_stats(logits: torch.Tensor, alternatives: list[torch.Tensor]) -> dict:
    probability = F.softmax(logits[0].float(), dim=-1)
    return f09.target_stats(logits, alternatives) | {
        "probability_entropy": float(
            -(probability * probability.clamp_min(1e-12).log()).sum(-1).mean().detach().cpu()
        ),
    }


def evaluate(
    model, source, lengths, specimen, concepts_raw, sp, pieces, eos, bos,
    depth, steps, baseline_logits, baseline_tokens, mode, amplitude, device,
):
    raw = make_raw(1, amplitude, device)
    fixed_logits, _, _, fixed_router = decode(
        model, source, lengths, bos, depth, steps, raw=raw, mode=mode,
        prefixes=baseline_tokens,
    )
    free_logits, free_tokens, _, free_router = decode(
        model, source, lengths, bos, depth, steps, raw=raw, mode=mode,
    )
    alternatives = [
        torch.tensor(sp.encode(surface, out_type=int), dtype=torch.long, device=device)
        for surface in concepts_raw["push"]
    ]
    clean = f04.clean(free_tokens[0].tolist(), eos, pieces)
    hard_hit = any(f08.contains_subsequence(clean, target.tolist()) for target in alternatives)
    stats = target_stats(fixed_logits, alternatives)
    actual_u = U_LIMIT * math.tanh(amplitude)
    return {
        "specimen": specimen["id"],
        "source": specimen["source"],
        "mode": mode,
        "raw_amplitude": amplitude,
        "actual_u": actual_u,
        **stats,
        "fixed_logit_max_abs_delta": float((fixed_logits - baseline_logits).abs().max().detach().cpu()),
        "hard_push_hit": int(hard_hit),
        "free_text": sp.decode(clean),
        "fixed_energy_relative_error": fixed_router["max_energy_relative_error"],
        "free_energy_relative_error": free_router["max_energy_relative_error"],
        "active_nodes_base": fixed_router["folds"][0]["active_nodes"],
        "active_nodes_extra": fixed_router["folds"][1]["active_nodes"],
    }


def best(rows: list[dict], specimen: str, mode: str) -> dict:
    candidates = [row for row in rows if row["specimen"] == specimen and row["mode"] == mode]
    chosen = max(candidates, key=lambda row: (
        row["hard_push_hit"], row["push_coverage"], -row["push_best_rank"],
    ))
    return {key: chosen[key] for key in (
        "raw_amplitude", "actual_u", "push_coverage", "push_best_rank",
        "push_best_probability", "hard_push_hit", "free_text",
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12101)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    data_path = Path(args.data)

    concepts_raw, all_rows = f08.load_echo_data(data_path)
    by_id = {row["id"]: row for row in all_rows}
    push = by_id["test-push-single"]
    sisyphus = by_id["test-sisyphus-composition"]
    specimens = [
        {**push, "id": "push-single"},
        {**push, "id": "push-repeat8", "source": push["repeat_source"]},
        {**sisyphus, "id": "sisyphus-single"},
        {**sisyphus, "id": "sisyphus-repeat4", "source": sisyphus["repeat_source"]},
    ]

    payload, _, sp, model, pieces, eos, bos, source_hash = load_runtime(checkpoint, args.device)
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    original_dynamic_width = bool(model.frozen_source.encoder.dynamic_width)
    model.frozen_source.encoder.dynamic_width = False

    rows = []
    baselines = {}
    zero_deltas = []
    for specimen in specimens:
        source, lengths = f04.encode_rows([specimen], sp, pieces, eos, pieces, args.device)
        with torch.no_grad():
            baseline_logits, baseline_tokens, budgets, _ = decode(
                model, source, lengths, bos, args.depth, args.steps,
            )
            alternatives = [
                torch.tensor(sp.encode(surface, out_type=int), dtype=torch.long, device=args.device)
                for surface in concepts_raw["push"]
            ]
            clean = f04.clean(baseline_tokens[0].tolist(), eos, pieces)
            base_stats = target_stats(baseline_logits, alternatives)
            base_row = {
                "specimen": specimen["id"], "source": specimen["source"],
                "mode": "native", "raw_amplitude": 0.0, "actual_u": 0.0,
                **base_stats, "fixed_logit_max_abs_delta": 0.0,
                "hard_push_hit": int(any(
                    f08.contains_subsequence(clean, target.tolist()) for target in alternatives
                )),
                "free_text": sp.decode(clean), "fixed_energy_relative_error": 0.0,
                "free_energy_relative_error": 0.0, "active_nodes_base": -1,
                "active_nodes_extra": -1,
            }
            rows.append(base_row)
            baselines[specimen["id"]] = base_row
            for mode in MODES:
                for amplitude in RAW_AMPLITUDES:
                    row = evaluate(
                        model, source, lengths, specimen, concepts_raw, sp, pieces, eos, bos,
                        args.depth, args.steps, baseline_logits, baseline_tokens,
                        mode, amplitude, args.device,
                    )
                    rows.append(row)
                    if mode == "parent" and amplitude == 0.0:
                        zero_deltas.append(row["fixed_logit_max_abs_delta"])

    gradient_rows = [specimens[0], specimens[2]]
    source, lengths = f04.encode_rows(gradient_rows, sp, pieces, eos, pieces, args.device)
    with torch.no_grad():
        _, native_tokens, _, _ = decode(model, source, lengths, bos, args.depth, args.steps)
    raw = make_raw(len(gradient_rows), 0.0, args.device, requires_grad=True)
    logits, _, _, gradient_router = decode(
        model, source, lengths, bos, args.depth, args.steps,
        raw=raw, mode="parent", prefixes=native_tokens,
    )
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)
    loss, gradient_metrics = f04.behavior_loss(logits, gradient_rows, concepts, eos)
    loss.backward()
    gradient_finite = raw.grad is not None and bool(torch.isfinite(raw.grad).all())
    gradient_abs_max = float(raw.grad.detach().abs().max().cpu()) if raw.grad is not None else 0.0
    gradient_abs_mean = float(raw.grad.detach().abs().mean().cpu()) if raw.grad is not None else 0.0

    model.frozen_source.encoder.dynamic_width = original_dynamic_width
    frozen_model = f04.frozen_exact(model, expected_model)
    numeric_keys = (
        "raw_amplitude", "actual_u", "push_coverage", "push_best_probability",
        "probability_entropy", "fixed_logit_max_abs_delta",
        "fixed_energy_relative_error", "free_energy_relative_error",
    )
    finite = all(math.isfinite(float(row[key])) for row in rows for key in numeric_keys)
    maximum_energy_error = max(
        max(row["fixed_energy_relative_error"], row["free_energy_relative_error"])
        for row in rows
    )
    best_by_mode = {
        specimen["id"]: {
            mode: best(rows, specimen["id"], mode)
            for mode in MODES
        }
        for specimen in specimens
    }

    parent_lookup = {
        (row["specimen"], row["raw_amplitude"]): row
        for row in rows if row["mode"] == "parent"
    }
    sign_pairs = []
    for specimen in specimens:
        negative = parent_lookup[(specimen["id"], -4.0)]
        positive = parent_lookup[(specimen["id"], 4.0)]
        sign_pairs.append({
            "specimen": specimen["id"],
            "coverage_delta_positive_minus_negative": positive["push_coverage"] - negative["push_coverage"],
            "logit_delta_between_signs_lower_bound": abs(
                positive["fixed_logit_max_abs_delta"] - negative["fixed_logit_max_abs_delta"]
            ),
            "negative_text": negative["free_text"],
            "positive_text": positive["free_text"],
        })

    gates = {
        "O0_parent_zero_exact": max(zero_deltas, default=float("inf")) <= 1e-7,
        "O1_energy_closure": maximum_energy_error <= 1e-5,
        "O2_frozen_and_finite": frozen_model and finite,
        "O3_nonzero_finite_gradient": gradient_finite and gradient_abs_max > 0.0,
    }
    probes = {
        "P1_sign_asymmetry_on_complete_sentence": any(
            abs(row["coverage_delta_positive_minus_negative"]) > 1e-7
            for row in sign_pairs if row["specimen"] == "sisyphus-single"
        ),
        "P2_any_complete_sentence_hard_push": any(
            row["hard_push_hit"] for row in rows if row["specimen"] == "sisyphus-single"
        ),
        "P3_any_parent_soft_improvement": any(
            row["push_coverage"] > baselines[row["specimen"]]["push_coverage"]
            for row in rows if row["mode"] == "parent"
        ),
        "P4_any_detail_or_joint_soft_improvement": any(
            row["push_coverage"] > baselines[row["specimen"]]["push_coverage"]
            for row in rows if row["mode"] in ("detail", "joint")
        ),
    }
    summary = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": f04.sha256(checkpoint),
        "runtime_source_sha256": source_hash,
        "data": str(data_path),
        "data_sha256": f04.sha256(data_path),
        "seed": args.seed,
        "depth": args.depth,
        "steps": args.steps,
        "raw_amplitudes": list(RAW_AMPLITUDES),
        "u_limit": U_LIMIT,
        "detail_mix": DETAIL_MIX,
        "formula": {
            "common": "(left + right) / sqrt(2)",
            "difference": "(left - right) / sqrt(2)",
            "u": "0.5 * tanh(raw)",
            "parent": "(common + u * difference) / sqrt(1 + u^2)",
            "detail": "(-u * common + difference) / sqrt(1 + u^2)",
        },
        "gates": gates,
        "probes": probes,
        "maximum_energy_relative_error": maximum_energy_error,
        "zero_max_logit_delta": max(zero_deltas, default=None),
        "gradient": {
            "loss": float(loss.detach().cpu()),
            "metrics": gradient_metrics,
            "finite": gradient_finite,
            "abs_max": gradient_abs_max,
            "abs_mean": gradient_abs_mean,
            "router": gradient_router,
        },
        "sign_pairs": sign_pairs,
        "baselines": baselines,
        "best_by_mode": best_by_mode,
        "intervention_count": len(rows),
        "seconds": time.time() - started,
        "claim_boundary": "frozen orthogonal intervention microscope; detail/joint are interface diagnostics, not a production dual-channel READ",
    }
    write_csv(output / "interventions.csv", rows)
    write_json(output / "interventions.json", rows)
    write_json(output / "summary.json", summary)
    readme = [
        "# F10 bounded orthogonal annealing microscope",
        "",
        f"- Claim: `{CLAIM}`",
        f"- Gates: `{json.dumps(gates, ensure_ascii=False)}`",
        f"- Probes: `{json.dumps(probes, ensure_ascii=False)}`",
        f"- Maximum energy relative error: `{maximum_energy_error:.9g}`",
        f"- Zero maximum logit delta: `{max(zero_deltas):.9g}`",
        f"- Gradient abs max: `{gradient_abs_max:.9g}`",
        f"- Interventions: `{len(rows)}`",
        "",
        "See `summary.json` and `interventions.csv` for the complete frozen-checkpoint observations.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete", "gates": gates, "probes": probes,
        "maximum_energy_relative_error": maximum_energy_error,
        "zero_max_logit_delta": max(zero_deltas),
        "gradient_abs_max": gradient_abs_max,
        "best_by_mode": best_by_mode,
        "intervention_count": len(rows), "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
