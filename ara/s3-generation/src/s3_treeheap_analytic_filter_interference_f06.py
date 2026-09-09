#!/usr/bin/env python3
"""Probe frozen TreeHeap states with deterministic analytic filter patterns."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_filter_guided_theta_calibration_f04 as f04  # noqa: E402
import s3_treeheap_observation_instruments_f05 as f05  # noqa: E402
from treeheap_epoch_translate_cli import load_runtime  # noqa: E402


CLAIM = "S3-TREEHEAP-ANALYTIC-FILTER-INTERFERENCE-F06"
FILTER_NAMES = (
    "uniform-gain",
    "pascal-middle",
    "binomial-left-p025",
    "binomial-right-p075",
    "haar-local-sibling",
    "depth-fine-minus-coarse",
)
CENTERED_FILTERS = frozenset(FILTER_NAMES) - {"uniform-gain"}
INTERFERENCE_PAIRS = (
    ("binomial-left-p025", "binomial-right-p075"),
    ("pascal-middle", "haar-local-sibling"),
    ("pascal-middle", "depth-fine-minus-coarse"),
    ("uniform-gain", "pascal-middle"),
)
INTERFERENCE_EPSILONS = (0.01, 0.1)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def coordinate_metadata(coord: int) -> dict:
    merge, node = f05.coord_parts(coord)
    root_depth = 4 - merge
    bits = format(node, f"0{root_depth}b") if root_depth else ""
    right_count = bits.count("1")
    return {
        "coord": coord,
        "merge": merge,
        "node": node,
        "root_depth": root_depth,
        "path_bits": bits,
        "right_count": right_count,
    }


def raw_filter_value(name: str, meta: dict) -> float:
    depth = meta["root_depth"]
    rights = meta["right_count"]
    bits = meta["path_bits"]
    if name == "uniform-gain":
        return 1.0
    if name == "pascal-middle":
        return float(math.comb(depth, rights))
    if name == "binomial-left-p025":
        return (0.25 ** rights) * (0.75 ** (depth - rights))
    if name == "binomial-right-p075":
        return (0.75 ** rights) * (0.25 ** (depth - rights))
    if name == "haar-local-sibling":
        return 1.0 if meta["node"] % 2 == 0 else -1.0
    if name == "depth-fine-minus-coarse":
        return float(depth)
    raise KeyError(name)


def analytic_filter_bank(active: list[int], device: str) -> tuple[dict[str, torch.Tensor], list[dict]]:
    if len(active) < 2:
        raise ValueError("analytic filters require at least two active coordinates")
    metadata = [coordinate_metadata(coord) for coord in active]
    bank = {}
    rows = []
    for name in FILTER_NAMES:
        raw = torch.tensor(
            [raw_filter_value(name, meta) for meta in metadata],
            dtype=torch.float64,
        )
        centered = name in CENTERED_FILTERS
        values = raw - raw.mean() if centered else raw
        rms = values.square().mean().sqrt()
        if not torch.isfinite(rms) or float(rms) <= 0.0:
            raise RuntimeError(f"degenerate analytic filter: {name}")
        values = values / rms
        full = torch.zeros((1, 31), dtype=torch.float32, device=device)
        full[0, active] = values.to(dtype=torch.float32, device=device)
        bank[name] = full
        for meta, raw_value, value in zip(metadata, raw.tolist(), values.tolist()):
            rows.append({
                "filter": name,
                **meta,
                "raw_value": raw_value,
                "normalized_value": value,
                "centered": int(centered),
            })
    return bank, rows


def tree_delta_metrics(baseline: list[torch.Tensor], current: list[torch.Tensor]) -> dict:
    depth_rows = []
    total_sq = 0.0
    max_abs = 0.0
    for depth, (left, right) in enumerate(zip(baseline, current)):
        delta = (right - left).float()
        norm = float(delta.norm().cpu())
        local_max = float(delta.abs().max().cpu())
        total_sq += norm * norm
        max_abs = max(max_abs, local_max)
        depth_rows.append({"tree_depth": depth, "l2": norm, "max_abs": local_max})
    return {"l2": math.sqrt(total_sq), "max_abs": max_abs, "depths": depth_rows}


def tree_interference_metrics(
    baseline: list[torch.Tensor],
    arm_a: list[torch.Tensor],
    arm_b: list[torch.Tensor],
    combined: list[torch.Tensor],
) -> dict:
    residual_sq = 0.0
    delta_a_sq = 0.0
    delta_b_sq = 0.0
    depth_rows = []
    for depth, (y0, ya, yb, yab) in enumerate(zip(baseline, arm_a, arm_b, combined)):
        residual = (yab - ya - yb + y0).float()
        delta_a = (ya - y0).float()
        delta_b = (yb - y0).float()
        residual_norm = float(residual.norm().cpu())
        residual_sq += residual_norm * residual_norm
        delta_a_sq += float(delta_a.square().sum().cpu())
        delta_b_sq += float(delta_b.square().sum().cpu())
        depth_rows.append({
            "tree_depth": depth,
            "interaction_l2": residual_norm,
            "interaction_max_abs": float(residual.abs().max().cpu()),
        })
    interaction = math.sqrt(residual_sq)
    denominator = math.sqrt(delta_a_sq) + math.sqrt(delta_b_sq)
    return {
        "interaction_l2": interaction,
        "relative_interaction": interaction / max(denominator, 1e-30),
        "depths": depth_rows,
    }


def tensor_interference_metrics(y0, ya, yb, yab) -> dict:
    residual = (yab - ya - yb + y0).float()
    denominator = (ya - y0).float().norm() + (yb - y0).float().norm()
    return {
        "interaction_l2": float(residual.norm().cpu()),
        "interaction_max_abs": float(residual.abs().max().cpu()),
        "relative_interaction": float((residual.norm() / denominator.clamp_min(1e-30)).cpu()),
    }


def convolved_trees(model, trees: dict) -> dict:
    return {
        "base_tree": model.reconstructor.convolve(trees["base_tree"], trees["base_masks"]),
        "extra_tree": model.extra.reconstructor.convolve(trees["extra_tree"], trees["extra_masks"]),
    }


@torch.no_grad()
def run_arm(model, theta, source, lengths, depth, bos, eos, pieces, sp, steps, baseline_tokens,
            channel: str, filter_u: torch.Tensor | None):
    trees = f05.protocol_trees(
        model, theta, source, lengths, depth,
        base_filter=filter_u if channel == "base" else None,
        extra_filter=filter_u if channel == "extra" else None,
    )
    logits, fixed_tokens, routes = f05.decode_with_routes(
        model, trees, bos, steps, prefixes=baseline_tokens,
    )
    _, free_tokens, _ = f05.decode_with_routes(model, trees, bos, steps)
    free_text = sp.decode(f04.clean(free_tokens[0].tolist(), eos, pieces))
    return {
        "trees": trees,
        "convolved": convolved_trees(model, trees),
        "logits": logits,
        "fixed_tokens": fixed_tokens,
        "routes": routes,
        "free_text": free_text,
    }


def arm_summary(label, channel, name, epsilon, baseline, arm, concepts, concepts_raw, sp) -> tuple[dict, list[dict]]:
    local_routes = f05.compare_routes(label, baseline["routes"], arm["routes"])
    tree_key = f"{channel}_tree"
    tree_metrics = tree_delta_metrics(baseline["trees"][tree_key], arm["trees"][tree_key])
    conv_metrics = tree_delta_metrics(baseline["convolved"][tree_key], arm["convolved"][tree_key])
    logits_delta = (arm["logits"] - baseline["logits"]).float()
    return ({
        "label": label,
        "channel": channel,
        "filter": name,
        "epsilon": epsilon,
        "fold_tree_l2": tree_metrics["l2"],
        "fold_tree_max_abs": tree_metrics["max_abs"],
        "convolved_tree_l2": conv_metrics["l2"],
        "convolved_tree_max_abs": conv_metrics["max_abs"],
        "logit_l2": float(logits_delta.norm().cpu()),
        "logit_max_abs": float(logits_delta.abs().max().cpu()),
        "route_max_js": max(row["js"] for row in local_routes),
        "route_branch_flips": sum(row["branch_flip"] for row in local_routes),
        "fixed_argmax_changes": int((arm["fixed_tokens"] != baseline["fixed_tokens"]).sum().cpu()),
        "free_text": arm["free_text"],
        "spectrum": f05.spectrum_summary(label, arm["logits"], concepts, concepts_raw, sp),
        "fold_depths": tree_metrics["depths"],
        "convolved_depths": conv_metrics["depths"],
    }, local_routes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--theta-checkpoint", required=True)
    parser.add_argument("--specimens", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--sentence-id", default="test-sisyphus-push-stone")
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11701)
    parser.add_argument("--rank", type=int, default=16)
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
    theta_checkpoint = Path(args.theta_checkpoint)
    specimen_path = Path(args.specimens)

    concepts_raw, all_rows = f04.load_specimens(specimen_path)
    matches = [row for row in all_rows if row["id"] == args.sentence_id]
    if len(matches) != 1:
        raise ValueError(f"expected one specimen {args.sentence_id}, got {len(matches)}")
    specimen = matches[0]
    payload, _, sp, model, pieces, eos, bos, source_hash = load_runtime(checkpoint, args.device)
    f04.freeze_model(model)
    expected_model = payload["trainable_state_dict"]
    theta = f04.AnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
    theta_payload = torch.load(theta_checkpoint, map_location=args.device)
    theta.load_state_dict(theta_payload["theta_state_dict"])
    theta.eval()
    for parameter in theta.parameters():
        parameter.requires_grad_(False)
    expected_theta = {name: value.detach().cpu().clone() for name, value in theta.state_dict().items()}

    source, lengths = f04.encode_rows([specimen], sp, pieces, eos, pieces, args.device)
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)
    original_dynamic_width = bool(model.frozen_source.encoder.dynamic_width)
    model.frozen_source.encoder.dynamic_width = False
    baseline_trees = f05.protocol_trees(model, theta, source, lengths, args.depth)
    _, base_valid = theta.base.gain_vector()
    _, extra_valid = theta.extra.gain_vector()
    base_active = torch.nonzero(base_valid[0], as_tuple=False).flatten().tolist()
    extra_active = torch.nonzero(extra_valid[0], as_tuple=False).flatten().tolist()
    banks = {}
    filter_rows = []
    for channel, active in (("base", base_active), ("extra", extra_active)):
        bank, rows = analytic_filter_bank(active, args.device)
        banks[channel] = bank
        filter_rows.extend({"channel": channel, **row} for row in rows)
    f05.write_csv(output / "analytic_filter_bank.csv", filter_rows)

    baseline_logits, baseline_tokens, baseline_routes = f05.decode_with_routes(
        model, baseline_trees, bos, args.steps,
    )
    native_logits, native_tokens, _ = f04.decode_logits(
        model, theta, source, lengths, bos, args.depth, args.steps,
    )
    baseline = {
        "trees": baseline_trees,
        "convolved": convolved_trees(model, baseline_trees),
        "logits": baseline_logits,
        "fixed_tokens": baseline_tokens,
        "routes": baseline_routes,
        "free_text": sp.decode(f04.clean(baseline_tokens[0].tolist(), eos, pieces)),
    }
    reader_logit_delta = float((baseline_logits - native_logits).abs().max().cpu())
    reader_token_exact = bool(torch.equal(baseline_tokens, native_tokens))

    epsilons = (-0.1, -0.01, 0.01, 0.1)
    intervention_rows = []
    route_rows = []
    detailed = []
    positive_cache = {}
    for channel, name, epsilon in itertools.product(("base", "extra"), FILTER_NAMES, epsilons):
        filter_u = banks[channel][name] * epsilon
        arm = run_arm(
            model, theta, source, lengths, args.depth, bos, eos, pieces, sp, args.steps,
            baseline_tokens, channel, filter_u,
        )
        label = f"{channel}:{name}:eps{epsilon:+.2f}"
        summary, local_routes = arm_summary(
            label, channel, name, epsilon, baseline, arm, concepts, concepts_raw, sp,
        )
        intervention_rows.append({
            key: value for key, value in summary.items()
            if key not in {"spectrum", "fold_depths", "convolved_depths"}
        })
        detailed.append(summary)
        route_rows.extend(local_routes)
        if epsilon in INTERFERENCE_EPSILONS:
            positive_cache[(channel, name, epsilon)] = arm

    interference_rows = []
    interference_detail = []
    for channel in ("base", "extra"):
        tree_key = f"{channel}_tree"
        for (name_a, name_b), epsilon in itertools.product(
            INTERFERENCE_PAIRS, INTERFERENCE_EPSILONS,
        ):
            arm_a = positive_cache[(channel, name_a, epsilon)]
            arm_b = positive_cache[(channel, name_b, epsilon)]
            combined_filter = epsilon * (banks[channel][name_a] + banks[channel][name_b])
            combined = run_arm(
                model, theta, source, lengths, args.depth, bos, eos, pieces, sp, args.steps,
                baseline_tokens, channel, combined_filter,
            )
            fold_metrics = tree_interference_metrics(
                baseline["trees"][tree_key], arm_a["trees"][tree_key],
                arm_b["trees"][tree_key], combined["trees"][tree_key],
            )
            conv_metrics = tree_interference_metrics(
                baseline["convolved"][tree_key], arm_a["convolved"][tree_key],
                arm_b["convolved"][tree_key], combined["convolved"][tree_key],
            )
            logit_metrics = tensor_interference_metrics(
                baseline["logits"], arm_a["logits"], arm_b["logits"], combined["logits"],
            )
            row = {
                "channel": channel,
                "filter_a": name_a,
                "filter_b": name_b,
                "epsilon": epsilon,
                "fold_interaction_l2": fold_metrics["interaction_l2"],
                "fold_relative_interaction": fold_metrics["relative_interaction"],
                "convolved_interaction_l2": conv_metrics["interaction_l2"],
                "convolved_relative_interaction": conv_metrics["relative_interaction"],
                "logit_interaction_l2": logit_metrics["interaction_l2"],
                "logit_interaction_max_abs": logit_metrics["interaction_max_abs"],
                "logit_relative_interaction": logit_metrics["relative_interaction"],
                "combined_fixed_argmax_changes": int(
                    (combined["fixed_tokens"] != baseline_tokens).sum().cpu()
                ),
                "combined_free_text": combined["free_text"],
            }
            interference_rows.append(row)
            interference_detail.append({
                **row,
                "fold_depths": fold_metrics["depths"],
                "convolved_depths": conv_metrics["depths"],
                "logit_metrics": logit_metrics,
            })

    f05.write_csv(output / "filter_interventions.csv", intervention_rows)
    f05.write_csv(output / "read_route_responses.csv", route_rows)
    f05.write_csv(output / "filter_interference.csv", interference_rows)
    write_json(output / "filter_interventions.json", detailed)
    write_json(output / "filter_interference.json", interference_detail)

    model.frozen_source.encoder.dynamic_width = original_dynamic_width
    frozen_model = f04.frozen_exact(model, expected_model)
    frozen_theta = f05.tensor_equal_state(theta, expected_theta)
    analytic_contract = True
    for channel in ("base", "extra"):
        for name in FILTER_NAMES:
            values = [
                row["normalized_value"] for row in filter_rows
                if row["channel"] == channel and row["filter"] == name
            ]
            rms = math.sqrt(sum(value * value for value in values) / len(values))
            mean = sum(values) / len(values)
            analytic_contract &= abs(rms - 1.0) <= 1e-6
            if name in CENTERED_FILTERS:
                analytic_contract &= abs(mean) <= 1e-6
    finite = all(
        math.isfinite(float(row[key]))
        for row in intervention_rows
        for key in (
            "fold_tree_l2", "convolved_tree_l2", "logit_l2", "route_max_js",
        )
    ) and all(
        math.isfinite(float(row[key]))
        for row in interference_rows
        for key in (
            "fold_interaction_l2", "convolved_interaction_l2", "logit_interaction_l2",
        )
    )
    no_grad_contract = all(not parameter.requires_grad for parameter in model.parameters()) and all(
        not parameter.requires_grad for parameter in theta.parameters()
    )
    gates = {
        "A0_reader_parity_and_frozen": (
            reader_logit_delta <= 1e-6 and reader_token_exact and frozen_model and frozen_theta
        ),
        "A1_analytic_contract": analytic_contract,
        "A2_evidence_complete": (
            len(intervention_rows) == 2 * len(FILTER_NAMES) * len(epsilons)
            and bool(route_rows) and finite
        ),
        "A3_interference_complete": (
            len(interference_rows)
            == 2 * len(INTERFERENCE_PAIRS) * len(INTERFERENCE_EPSILONS)
        ),
        "A4_no_target_optimization": no_grad_contract,
    }
    summary = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": f04.sha256(checkpoint),
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "runtime_source_sha256": source_hash,
        "theta_checkpoint": str(theta_checkpoint),
        "theta_checkpoint_sha256": f04.sha256(theta_checkpoint),
        "specimens": str(specimen_path),
        "specimens_sha256": f04.sha256(specimen_path),
        "sentence_id": args.sentence_id,
        "source": specimen["source"],
        "source_pieces": int(lengths[0].cpu()),
        "depth": args.depth,
        "source_width_mode": "fixed_32",
        "active_coordinates": {"base": base_active, "extra": extra_active},
        "filters": list(FILTER_NAMES),
        "epsilons": list(epsilons),
        "interference_pairs": [list(pair) for pair in INTERFERENCE_PAIRS],
        "interference_epsilons": list(INTERFERENCE_EPSILONS),
        "baseline_generation": baseline["free_text"],
        "baseline_spectrum": f05.spectrum_summary(
            "baseline", baseline_logits, concepts, concepts_raw, sp,
        ),
        "reader_parity": {
            "logit_max_abs_delta": reader_logit_delta,
            "token_exact": reader_token_exact,
        },
        "frozen_model": frozen_model,
        "frozen_theta": frozen_theta,
        "gates": gates,
        "strongest_logit_responses": sorted(
            intervention_rows, key=lambda row: row["logit_l2"], reverse=True,
        )[:8],
        "strongest_interactions": sorted(
            interference_rows, key=lambda row: row["logit_relative_interaction"], reverse=True,
        ),
        "seconds": time.time() - started,
        "claim_boundary": "frozen analytic observation only; no semantic or quality claim",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "baseline_generation": baseline["free_text"],
        "strongest_logit_responses": summary["strongest_logit_responses"][:4],
        "strongest_interactions": summary["strongest_interactions"][:4],
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
