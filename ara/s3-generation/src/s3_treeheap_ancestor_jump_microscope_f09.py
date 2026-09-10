#!/usr/bin/env python3
"""Compare direct parent jumps with continuous TreeHeap ancestor interventions."""
from __future__ import annotations

import argparse
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

import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_observation_instruments_f05 as f05
import s3_cross_language_echo_controller_f08 as f08
from treeheap_epoch_translate_cli import load_runtime


CLAIM = "S3-TREEHEAP-ANCESTOR-JUMP-MICROSCOPE-F09"
AMPLITUDES = (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)


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


def active_ancestor_path(seed: int, active: set[int]) -> list[int]:
    merge, node = f05.coord_parts(seed)
    path = []
    for ancestor_merge in range(merge, len(f05.COORD_OFFSETS)):
        ancestor_node = node // (2 ** (ancestor_merge - merge))
        coord = f05.COORD_OFFSETS[ancestor_merge] + ancestor_node
        if coord in active:
            path.append(coord)
    return path


def make_filter(coords: list[int], amplitude: float, device: str) -> torch.Tensor:
    value = torch.zeros((1, 31), dtype=torch.float32, device=device)
    value[0, coords] = amplitude
    return value


def target_stats(logits: torch.Tensor, alternatives: list[torch.Tensor]) -> dict:
    probability = F.softmax(logits[0].float(), dim=-1)
    coverage = f04.concept_coverage(probability, alternatives)
    best_rank = probability.shape[-1]
    best_probability = 0.0
    best_token = None
    best_step = None
    for ids in alternatives:
        for token_id in ids.tolist():
            values = probability[:, token_id]
            for step in range(probability.shape[0]):
                value = float(values[step].cpu())
                rank = 1 + int((probability[step] > probability[step, token_id]).sum().cpu())
                if rank < best_rank or (rank == best_rank and value > best_probability):
                    best_rank = rank
                    best_probability = value
                    best_token = int(token_id)
                    best_step = step
    return {
        "push_coverage": float(coverage.cpu()),
        "push_best_rank": best_rank,
        "push_best_probability": best_probability,
        "push_best_token_id": best_token,
        "push_best_step": best_step,
    }


@torch.no_grad()
def evaluate(
    model, theta, source, lengths, row, concepts, concepts_raw, sp, pieces,
    bos, eos, depth, steps, baseline_logits, baseline_tokens, label,
    family, coords, amplitude, active_count, device,
) -> dict:
    filter_u = None if not coords else make_filter(coords, amplitude, device)
    fixed_logits, _, _ = f04.decode_logits(
        model, theta, source, lengths, bos, depth, steps,
        filter_u=filter_u, prefixes=baseline_tokens,
    )
    free_logits, free_tokens, _ = f04.decode_logits(
        model, theta, source, lengths, bos, depth, steps, filter_u=filter_u,
    )
    clean = f04.clean(free_tokens[0].tolist(), eos, pieces)
    alternatives = [sp.encode(surface, out_type=int) for surface in concepts_raw["push"]]
    hard_hit = any(f08.contains_subsequence(clean, target) for target in alternatives)
    stats = target_stats(fixed_logits, concepts["push"])
    actual_gain = 1.0 if not coords else math.exp(math.log(1.5) * math.tanh(amplitude))
    return {
        "specimen": row["id"],
        "source": row["source"],
        "label": label,
        "family": family,
        "coords": ",".join(map(str, coords)),
        "coordinate_count": len(coords),
        "active_count": active_count,
        "amplitude_u": amplitude,
        "per_node_gain": actual_gain,
        "filter_l2": abs(amplitude) * math.sqrt(len(coords)),
        **stats,
        "fixed_logit_max_abs_delta": float((fixed_logits - baseline_logits).abs().max().cpu()),
        "hard_push_hit": int(hard_hit),
        "free_text": sp.decode(clean),
    }


def best(rows: list[dict], specimen: str, family: str) -> dict | None:
    candidates = [row for row in rows if row["specimen"] == specimen and row["family"] == family]
    if not candidates:
        return None
    chosen = max(
        candidates,
        key=lambda row: (row["hard_push_hit"], row["push_coverage"], -row["push_best_rank"]),
    )
    return {key: chosen[key] for key in (
        "label", "coords", "amplitude_u", "per_node_gain", "push_coverage",
        "push_best_rank", "push_best_probability", "hard_push_hit", "free_text",
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--theta-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12001)
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
    theta = f04.AnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
    theta_payload = torch.load(theta_checkpoint, map_location=args.device)
    theta.load_state_dict(theta_payload["theta_state_dict"])
    theta.eval()
    for parameter in theta.parameters():
        parameter.requires_grad_(False)
    expected_theta = {name: value.detach().cpu().clone() for name, value in theta.state_dict().items()}
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)
    original_dynamic_width = bool(model.frozen_source.encoder.dynamic_width)
    model.frozen_source.encoder.dynamic_width = False

    rows = []
    topology = []
    for specimen in specimens:
        source, lengths = f04.encode_rows([specimen], sp, pieces, eos, pieces, args.device)
        baseline_logits, baseline_tokens, _ = f04.decode_logits(
            model, theta, source, lengths, bos, args.depth, args.steps,
        )
        _, base_valid = theta.base.gain_vector()
        _, extra_valid = theta.extra.gain_vector()
        base_active = torch.nonzero(base_valid[0], as_tuple=False).flatten().tolist()
        extra_active = torch.nonzero(extra_valid[0], as_tuple=False).flatten().tolist()
        active = sorted(set(base_active) & set(extra_active))
        active_set = set(active)
        fine = [coord for coord in active if f05.coord_parts(coord)[0] == 0]
        topology.append({
            "specimen": specimen["id"],
            "source": specimen["source"],
            "source_pieces": int(lengths[0].cpu()),
            "base_active": base_active,
            "extra_active": extra_active,
            "common_active": active,
            "fine_coords": fine,
            "paths": {str(coord): active_ancestor_path(coord, active_set) for coord in fine},
        })
        rows.append(evaluate(
            model, theta, source, lengths, specimen, concepts, concepts_raw, sp, pieces,
            bos, eos, args.depth, args.steps, baseline_logits, baseline_tokens,
            "baseline", "baseline", [], 0.0, len(active), args.device,
        ))
        for amplitude in AMPLITUDES:
            rows.append(evaluate(
                model, theta, source, lengths, specimen, concepts, concepts_raw, sp, pieces,
                bos, eos, args.depth, args.steps, baseline_logits, baseline_tokens,
                f"uniform-all-u{amplitude:+g}", "uniform-all", active, amplitude,
                len(active), args.device,
            ))
            for coord in active:
                merge, node = f05.coord_parts(coord)
                family = "fine-jump" if merge == 0 else "upper-jump"
                rows.append(evaluate(
                    model, theta, source, lengths, specimen, concepts, concepts_raw, sp, pieces,
                    bos, eos, args.depth, args.steps, baseline_logits, baseline_tokens,
                    f"{family}-m{merge}n{node}-u{amplitude:+g}", family, [coord], amplitude,
                    len(active), args.device,
                ))
            for seed in fine:
                path = active_ancestor_path(seed, active_set)
                if len(path) < 2:
                    continue
                rows.append(evaluate(
                    model, theta, source, lengths, specimen, concepts, concepts_raw, sp, pieces,
                    bos, eos, args.depth, args.steps, baseline_logits, baseline_tokens,
                    f"ancestor-chain-seed{seed}-u{amplitude:+g}", "ancestor-chain", path,
                    amplitude, len(active), args.device,
                ))
                off_path = [coord for coord in active if coord not in path][:len(path)]
                if len(off_path) == len(path):
                    rows.append(evaluate(
                        model, theta, source, lengths, specimen, concepts, concepts_raw, sp, pieces,
                        bos, eos, args.depth, args.steps, baseline_logits, baseline_tokens,
                        f"off-path-seed{seed}-u{amplitude:+g}", "off-path-matched", off_path,
                        amplitude, len(active), args.device,
                    ))

    model.frozen_source.encoder.dynamic_width = original_dynamic_width
    frozen_model = f04.frozen_exact(model, expected_model)
    frozen_theta = f05.tensor_equal_state(theta, expected_theta)
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("amplitude_u", "per_node_gain", "push_coverage", "push_best_probability",
                    "fixed_logit_max_abs_delta")
    )
    summaries = {
        specimen["id"]: {
            family: best(rows, specimen["id"], family)
            for family in ("baseline", "fine-jump", "upper-jump", "ancestor-chain",
                           "off-path-matched", "uniform-all")
        }
        for specimen in specimens
    }
    gates = {
        "O0_model_and_theta_frozen": frozen_model and frozen_theta,
        "O1_all_values_finite": finite,
        "O2_base_extra_topology_equal": all(row["base_active"] == row["extra_active"] for row in topology),
        "O3_all_specimens_observed": len(topology) == len(specimens),
        "O4_bidirectional_gain_scan": min(row["per_node_gain"] for row in rows) < 1.0 < max(row["per_node_gain"] for row in rows),
    }
    probes = {
        "P1_repeat_baseline_has_push": bool(summaries["push-repeat8"]["baseline"]["hard_push_hit"]),
        "P2_single_any_upper_jump_has_push": any(
            row["hard_push_hit"] for row in rows
            if row["specimen"] == "push-single" and row["family"] == "upper-jump"
        ),
        "P3_sisyphus_any_upper_jump_has_push": any(
            row["hard_push_hit"] for row in rows
            if row["specimen"] == "sisyphus-single" and row["family"] == "upper-jump"
        ),
        "P4_sisyphus_any_chain_has_push": any(
            row["hard_push_hit"] for row in rows
            if row["specimen"] == "sisyphus-single" and row["family"] == "ancestor-chain"
        ),
        "P5_chain_beats_off_path_on_sisyphus": (
            summaries["sisyphus-single"]["ancestor-chain"] is not None
            and summaries["sisyphus-single"]["off-path-matched"] is not None
            and summaries["sisyphus-single"]["ancestor-chain"]["push_coverage"]
            > summaries["sisyphus-single"]["off-path-matched"]["push_coverage"]
        ),
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
        "amplitudes": list(AMPLITUDES),
        "topology": topology,
        "best_by_family": summaries,
        "gates": gates,
        "probes": probes,
        "intervention_count": len(rows),
        "frozen_model": frozen_model,
        "frozen_theta": frozen_theta,
        "seconds": time.time() - started,
        "claim_boundary": "frozen radial intervention microscope; best cases are reachability observations, not learned routing",
    }
    write_csv(output / "interventions.csv", rows)
    write_json(output / "interventions.json", rows)
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "probes": probes,
        "best_by_family": summaries,
        "intervention_count": len(rows),
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
