#!/usr/bin/env python3
"""Fit a shared frozen-model analytic-filter controller for cross-language echo."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import torch
import torch.nn.functional as F

import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_structured_filter_bank_f07 as f07
from treeheap_epoch_translate_cli import load_runtime


CLAIM = "S3-CROSS-LANGUAGE-ECHO-CONTROLLER-F08"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_echo_data(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts, rows = payload.get("concepts"), payload.get("rows")
    if not isinstance(concepts, dict) or not isinstance(rows, list):
        raise ValueError("invalid F08 data")
    if {row.get("split") for row in rows} != {"fit", "test"}:
        raise ValueError("F08 requires fit and test splits")
    if any(not row.get("repeat_source") for row in rows):
        raise ValueError("F08 requires repeat_source for every row")
    return concepts, rows


def row_basis(active: list[int], device: str) -> tuple[torch.Tensor, list[str]]:
    metadata = [f07.f06.coordinate_metadata(coord) for coord in active]
    vectors = []
    degenerate = []
    for name in f07.FILTER_NAMES:
        raw = f07.raw_values(name, metadata)
        values = raw - raw.mean() if name in f07.CENTERED_FILTERS else raw
        rms = values.square().mean().sqrt()
        if not torch.isfinite(rms):
            raise RuntimeError(f"non-finite basis {name}")
        full = torch.zeros(31, dtype=torch.float32, device=device)
        if float(rms) <= 1e-12:
            degenerate.append(name)
        else:
            full[active] = (values / rms).to(dtype=torch.float32, device=device)
        vectors.append(full)
    return torch.stack(vectors), degenerate


@torch.no_grad()
def build_basis(model, theta, source, lengths, bos, depth, steps, device):
    f04.decode_logits(model, theta, source, lengths, bos, depth, steps)
    _, valid = theta.base.gain_vector()
    rows = []
    active_rows = []
    degenerate_rows = []
    for index in range(source.shape[0]):
        active = torch.nonzero(valid[index], as_tuple=False).flatten().tolist()
        basis, degenerate = row_basis(active, device)
        rows.append(basis)
        active_rows.append(active)
        degenerate_rows.append(degenerate)
    return torch.stack(rows), active_rows, degenerate_rows


def compose_filter(raw_coefficients: torch.Tensor, basis: torch.Tensor, max_u: float) -> torch.Tensor:
    if raw_coefficients.ndim == 1:
        linear = torch.einsum("k,bkc->bc", raw_coefficients, basis)
    else:
        linear = torch.einsum("bk,bkc->bc", raw_coefficients, basis)
    return max_u * torch.tanh(linear)


def distribution_kl(logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    baseline = F.softmax(baseline_logits.detach().float(), dim=-1)
    current = F.log_softmax(logits.float(), dim=-1)
    return (baseline * (baseline.clamp_min(1e-12).log() - current)).sum(-1).mean()


def optimize_controller(
    model, theta, source, lengths, rows, concepts, bos, eos, depth, steps,
    basis, baseline_logits, baseline_tokens, optimization_steps, learning_rate,
    max_u, per_row: bool,
):
    shape = (source.shape[0], len(f07.FILTER_NAMES)) if per_row else (len(f07.FILTER_NAMES),)
    raw = torch.nn.Parameter(torch.zeros(shape, device=source.device))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    trace = []
    for step in range(optimization_steps + 1):
        filter_u = compose_filter(raw, basis, max_u)
        logits, _, _ = f04.decode_logits(
            model, theta, source, lengths, bos, depth, steps,
            filter_u=filter_u, prefixes=baseline_tokens,
        )
        behavior, metrics = f04.behavior_loss(logits, rows, concepts, eos)
        kl = distribution_kl(logits, baseline_logits)
        regularizer = raw.square().mean()
        loss = behavior + 0.02 * kl + 0.002 * regularizer
        if step % 10 == 0 or step == optimization_steps:
            trace.append({
                "step": step,
                "total_loss": float(loss.detach().cpu()),
                "behavior_loss": metrics["loss"],
                "positive_coverage": metrics["positive_coverage"],
                "negative_activation": metrics["negative_activation"],
                "kl_from_native": float(kl.detach().cpu()),
                "filter_rms": float(filter_u.square().mean().sqrt().detach().cpu()),
                "filter_max_abs": float(filter_u.abs().max().detach().cpu()),
            })
        if step == optimization_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise RuntimeError("non-finite controller gradient")
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        optimizer.step()
    return raw.detach(), trace


def contains_subsequence(tokens: list[int], target: list[int]) -> bool:
    if not target or len(target) > len(tokens):
        return False
    return any(tokens[index:index + len(target)] == target for index in range(len(tokens) - len(target) + 1))


@torch.no_grad()
def evaluate_case(
    label, model, theta, source, lengths, rows, concepts, concepts_raw,
    sp, pieces, eos, bos, depth, steps, filter_u=None, fixed_prefixes=None,
):
    fixed_logits, _, _ = f04.decode_logits(
        model, theta, source, lengths, bos, depth, steps,
        filter_u=filter_u, prefixes=fixed_prefixes,
    )
    free_logits, free_tokens, _ = f04.decode_logits(
        model, theta, source, lengths, bos, depth, steps, filter_u=filter_u,
    )
    _, fixed_metrics = f04.behavior_loss(fixed_logits, rows, concepts, eos)
    _, free_metrics = f04.behavior_loss(free_logits, rows, concepts, eos)
    outputs = []
    positive_hits = []
    for index, row in enumerate(rows):
        tokens = f04.clean(free_tokens[index].tolist(), eos, pieces)
        concept_hits = {}
        for name in row["positive"]:
            alternatives = [sp.encode(surface, out_type=int) for surface in concepts_raw[name]]
            hit = any(contains_subsequence(tokens, target) for target in alternatives)
            concept_hits[name] = hit
            positive_hits.append(float(hit))
        outputs.append({
            "id": row["id"],
            "source": row["source"],
            "text": sp.decode(tokens),
            "positive_hits": concept_hits,
            "all_positive_hit": all(concept_hits.values()),
        })
    return {
        "label": label,
        "fixed_metrics": fixed_metrics,
        "free_metrics": free_metrics,
        "positive_token_hit_rate": sum(positive_hits) / max(len(positive_hits), 1),
        "all_positive_row_hits": sum(row["all_positive_hit"] for row in outputs),
        "outputs": outputs,
    }


def filter_stats(filter_u: torch.Tensor, active_rows: list[list[int]]) -> list[dict]:
    result = []
    for row, active in zip(filter_u, active_rows):
        values = row[active].float()
        result.append({
            "rms": float(values.square().mean().sqrt().cpu()),
            "max_abs": float(values.abs().max().cpu()),
        })
    return result


def matched_uniform_filter(filter_u: torch.Tensor, active_rows: list[list[int]]) -> torch.Tensor:
    matched = torch.zeros_like(filter_u)
    for index, active in enumerate(active_rows):
        values = filter_u[index, active].float()
        amplitude = values.square().mean().sqrt()
        matched[index, active] = amplitude.to(dtype=matched.dtype)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--theta-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--optimization-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-u", type=float, default=0.2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11901)
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
    data_path = Path(args.data)
    concepts_raw, all_rows = load_echo_data(data_path)
    fit_rows = [row for row in all_rows if row["split"] == "fit"]
    test_rows = [row for row in all_rows if row["split"] == "test"]

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
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)

    fit_source, fit_lengths = f04.encode_rows(fit_rows, sp, pieces, eos, pieces, args.device)
    test_source, test_lengths = f04.encode_rows(test_rows, sp, pieces, eos, pieces, args.device)
    repeat_rows = [{**row, "source": row["repeat_source"]} for row in test_rows]
    repeat_source, repeat_lengths = f04.encode_rows(
        repeat_rows, sp, pieces, eos, pieces, args.device,
    )
    original_dynamic_width = bool(model.frozen_source.encoder.dynamic_width)
    model.frozen_source.encoder.dynamic_width = False

    fit_basis, fit_active, fit_degenerate = build_basis(
        model, theta, fit_source, fit_lengths, bos, args.depth, args.steps, args.device,
    )
    test_basis, test_active, test_degenerate = build_basis(
        model, theta, test_source, test_lengths, bos, args.depth, args.steps, args.device,
    )
    with torch.no_grad():
        fit_native_logits, fit_native_tokens, _ = f04.decode_logits(
            model, theta, fit_source, fit_lengths, bos, args.depth, args.steps,
        )
        test_native_logits, test_native_tokens, _ = f04.decode_logits(
            model, theta, test_source, test_lengths, bos, args.depth, args.steps,
        )
        _, repeat_native_tokens, _ = f04.decode_logits(
            model, theta, repeat_source, repeat_lengths, bos, args.depth, args.steps,
        )

    shared_raw, shared_trace = optimize_controller(
        model, theta, fit_source, fit_lengths, fit_rows, concepts, bos, eos,
        args.depth, args.steps, fit_basis, fit_native_logits, fit_native_tokens,
        args.optimization_steps, args.learning_rate, args.max_u, per_row=False,
    )
    oracle_raw, oracle_trace = optimize_controller(
        model, theta, test_source, test_lengths, test_rows, concepts, bos, eos,
        args.depth, args.steps, test_basis, test_native_logits, test_native_tokens,
        args.optimization_steps, args.learning_rate, args.max_u, per_row=True,
    )
    fit_shared_filter = compose_filter(shared_raw, fit_basis, args.max_u)
    test_shared_filter = compose_filter(shared_raw, test_basis, args.max_u)
    test_oracle_filter = compose_filter(oracle_raw, test_basis, args.max_u)
    test_uniform_filter = matched_uniform_filter(test_shared_filter, test_active)

    cases = {
        "fit_single_native": evaluate_case(
            "fit-single-native", model, theta, fit_source, fit_lengths, fit_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            fixed_prefixes=fit_native_tokens,
        ),
        "fit_single_shared": evaluate_case(
            "fit-single-shared", model, theta, fit_source, fit_lengths, fit_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            filter_u=fit_shared_filter, fixed_prefixes=fit_native_tokens,
        ),
        "test_single_native": evaluate_case(
            "test-single-native", model, theta, test_source, test_lengths, test_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            fixed_prefixes=test_native_tokens,
        ),
        "test_single_shared": evaluate_case(
            "test-single-shared", model, theta, test_source, test_lengths, test_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            filter_u=test_shared_filter, fixed_prefixes=test_native_tokens,
        ),
        "test_single_uniform_matched": evaluate_case(
            "test-single-uniform-matched", model, theta, test_source, test_lengths, test_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            filter_u=test_uniform_filter, fixed_prefixes=test_native_tokens,
        ),
        "test_repeat_native": evaluate_case(
            "test-repeat-native", model, theta, repeat_source, repeat_lengths, repeat_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            fixed_prefixes=repeat_native_tokens,
        ),
        "test_single_oracle": evaluate_case(
            "test-single-oracle", model, theta, test_source, test_lengths, test_rows,
            concepts, concepts_raw, sp, pieces, eos, bos, args.depth, args.steps,
            filter_u=test_oracle_filter, fixed_prefixes=test_native_tokens,
        ),
    }

    model.frozen_source.encoder.dynamic_width = original_dynamic_width
    frozen_model = f04.frozen_exact(model, expected_model)
    frozen_theta = f07.f06.f05.tensor_equal_state(theta, expected_theta)
    finite = all(
        math.isfinite(float(case[metric_group][metric]))
        for case in cases.values()
        for metric_group in ("fixed_metrics", "free_metrics")
        for metric in ("loss", "positive_coverage", "negative_activation")
    ) and all(math.isfinite(row["total_loss"]) for row in shared_trace + oracle_trace)
    no_test_leakage = shared_raw.ndim == 1 and shared_raw.shape[0] == len(f07.FILTER_NAMES)
    gates = {
        "A0_model_and_theta_frozen": frozen_model and frozen_theta,
        "A1_finite_and_bounded": finite and float(test_oracle_filter.abs().max().cpu()) <= args.max_u + 1e-6,
        "A2_fit_test_isolated": no_test_leakage,
        "A3_evidence_complete": len(cases) == 7 and bool(shared_trace) and bool(oracle_trace),
    }
    native = cases["test_single_native"]
    shared = cases["test_single_shared"]
    uniform = cases["test_single_uniform_matched"]
    oracle = cases["test_single_oracle"]
    repeat = cases["test_repeat_native"]
    probes = {
        "P1_fit_soft_improves": (
            cases["fit_single_shared"]["free_metrics"]["positive_coverage"]
            > cases["fit_single_native"]["free_metrics"]["positive_coverage"]
        ),
        "P2_test_soft_improves": shared["free_metrics"]["positive_coverage"] > native["free_metrics"]["positive_coverage"],
        "P3_test_hard_improves": shared["positive_token_hit_rate"] > native["positive_token_hit_rate"],
        "P4_oracle_soft_improves": oracle["free_metrics"]["positive_coverage"] > native["free_metrics"]["positive_coverage"],
        "P5_shared_beats_repeat_soft": shared["free_metrics"]["positive_coverage"] > repeat["free_metrics"]["positive_coverage"],
        "P6_shared_beats_matched_uniform_soft": shared["free_metrics"]["positive_coverage"] > uniform["free_metrics"]["positive_coverage"],
    }
    summary = {
        "claim": CLAIM,
        "host": socket.gethostname(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": f04.sha256(checkpoint),
        "runtime_source_sha256": source_hash,
        "theta_checkpoint": str(theta_checkpoint),
        "theta_checkpoint_sha256": f04.sha256(theta_checkpoint),
        "data": str(data_path),
        "data_sha256": f04.sha256(data_path),
        "seed": args.seed,
        "depth": args.depth,
        "steps": args.steps,
        "optimization_steps": args.optimization_steps,
        "max_u": args.max_u,
        "filter_names": list(f07.FILTER_NAMES),
        "shared_coefficients": dict(zip(f07.FILTER_NAMES, shared_raw.cpu().tolist())),
        "fit_active_coordinates": fit_active,
        "test_active_coordinates": test_active,
        "fit_degenerate_filters": fit_degenerate,
        "test_degenerate_filters": test_degenerate,
        "fit_filter_stats": filter_stats(fit_shared_filter, fit_active),
        "test_filter_stats": filter_stats(test_shared_filter, test_active),
        "matched_uniform_filter_stats": filter_stats(test_uniform_filter, test_active),
        "oracle_filter_stats": filter_stats(test_oracle_filter, test_active),
        "cases": cases,
        "shared_trace": shared_trace,
        "oracle_trace": oracle_trace,
        "frozen_model": frozen_model,
        "frozen_theta": frozen_theta,
        "gates": gates,
        "probes": probes,
        "seconds": time.time() - started,
        "claim_boundary": "frozen local controller probe; oracle is not generalization evidence",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "cases.json", cases)
    write_json(output / "optimization_trace.json", {
        "shared": shared_trace, "oracle": oracle_trace,
    })
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "probes": probes,
        "test_positive_coverage": {
            "single_native": native["free_metrics"]["positive_coverage"],
            "single_shared": shared["free_metrics"]["positive_coverage"],
            "single_uniform_matched": uniform["free_metrics"]["positive_coverage"],
            "repeat_native": repeat["free_metrics"]["positive_coverage"],
            "single_oracle": oracle["free_metrics"]["positive_coverage"],
        },
        "test_hit_rate": {
            "single_native": native["positive_token_hit_rate"],
            "single_shared": shared["positive_token_hit_rate"],
            "repeat_native": repeat["positive_token_hit_rate"],
            "single_oracle": oracle["positive_token_hit_rate"],
        },
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
