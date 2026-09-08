#!/usr/bin/env python3
"""Observe TreeHeap parameters, FOLD impulse paths, READ routes, and token spectra."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_filter_guided_theta_calibration_f04 as f04  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
from treeheap_epoch_translate_cli import load_runtime  # noqa: E402


CLAIM = "S3-TREEHEAP-OBSERVATION-INSTRUMENTS-F05"
COORD_OFFSETS = (0, 16, 24, 28, 30)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_equal_state(module, expected: dict[str, torch.Tensor]) -> bool:
    state = module.state_dict()
    return all(name in state and torch.equal(value.detach().cpu(), expected[name]) for name, value in state.items())


def parameter_atlas(module, owner: str) -> list[dict]:
    rows = []
    for name, value in module.state_dict().items():
        tensor = value.detach()
        row = {
            "owner": owner,
            "name": name,
            "group": ".".join(name.split(".")[:2]),
            "shape": "x".join(map(str, tensor.shape)) or "scalar",
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "count": tensor.numel(),
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
            "l2": "",
        }
        if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
            work = tensor.float()
            row.update({
                "mean": float(work.mean().cpu()),
                "std": float(work.std(unbiased=False).cpu()),
                "min": float(work.min().cpu()),
                "max": float(work.max().cpu()),
                "l2": float(work.norm().cpu()),
            })
        rows.append(row)
    return rows


class ObservationRouter:
    def __init__(self, theta, base_filter=None, extra_filter=None):
        self.theta = theta
        self.base_filter = base_filter
        self.extra_filter = extra_filter
        self.calls = 0

    def __call__(self, slots, slot_mask):
        fold = self.theta.choose(slots.shape[-1])
        filter_u = self.base_filter if slots.shape[-1] == self.theta.base.dim else self.extra_filter
        levels, masks = fold(slots, slot_mask, filter_u)
        self.calls += 1
        return levels, masks


@contextmanager
def observed_fold(theta, base_filter=None, extra_filter=None):
    original = d07.fold_protocol
    router = ObservationRouter(theta, base_filter, extra_filter)
    d07.fold_protocol = router
    try:
        yield router
    finally:
        d07.fold_protocol = original


def protocol_trees(model, theta, source, lengths, depth, base_filter=None, extra_filter=None):
    with observed_fold(theta, base_filter, extra_filter) as router:
        base_tree, base_masks, budgets, base_slots, _, extra = model.protocol(
            source, lengths, depth, "native",
        )
    if router.calls != 2 or extra is None:
        raise RuntimeError(f"expected two FOLD calls and an extra channel, got {router.calls}")
    extra_tree, extra_masks, extra_slots, _ = extra
    return {
        "base_tree": base_tree,
        "base_masks": base_masks,
        "base_slots": base_slots,
        "extra_tree": extra_tree,
        "extra_masks": extra_masks,
        "extra_slots": extra_slots,
        "budgets": budgets,
    }


def coord_parts(coord: int) -> tuple[int, int]:
    for merge in range(4, -1, -1):
        if coord >= COORD_OFFSETS[merge]:
            return merge, coord - COORD_OFFSETS[merge]
    raise ValueError(coord)


def expected_path(coord: int) -> set[tuple[int, int]]:
    merge, node = coord_parts(coord)
    path = set()
    for ancestor_merge in range(merge, 5):
        ancestor_node = node // (2 ** (ancestor_merge - merge))
        tree_depth = 4 - ancestor_merge
        path.add((tree_depth, ancestor_node))
    return path


def clone_tree(tree):
    return [nodes.detach().clone() for nodes in tree]


def tree_delta_rows(baseline, pulse, path: set[tuple[int, int]]) -> tuple[list[dict], float, float]:
    rows = []
    on_energy = 0.0
    off_energy = 0.0
    for tree_depth, (left, right) in enumerate(zip(baseline, pulse)):
        delta = (right - left).float().norm(dim=-1)[0]
        for node, value in enumerate(delta.tolist()):
            relation = "path" if (tree_depth, node) in path else "off_path"
            energy = value * value
            if relation == "path":
                on_energy += energy
            else:
                off_energy += energy
            rows.append({
                "tree_depth": tree_depth,
                "node": node,
                "relation": relation,
                "delta_l2": value,
            })
    return rows, on_energy, off_energy


@torch.no_grad()
def scan_channel_impulses(theta_fold, slots, slot_mask, channel: str, epsilons: list[float]):
    zero = torch.zeros((slots.shape[0], 31), device=slots.device)
    baseline, _ = theta_fold(slots, slot_mask, zero)
    baseline = clone_tree(baseline)
    _, valid = theta_fold.gain_vector()
    active = torch.nonzero(valid[0], as_tuple=False).flatten().tolist()
    summaries = []
    details = []
    for coord in active:
        merge, node = coord_parts(coord)
        path = expected_path(coord)
        for epsilon in epsilons:
            pulse_filter = zero.clone()
            pulse_filter[0, coord] = epsilon
            pulse, _ = theta_fold(slots, slot_mask, pulse_filter)
            local_rows, on_energy, off_energy = tree_delta_rows(baseline, pulse, path)
            own_depth = 4 - merge
            own_delta = next(
                row["delta_l2"] for row in local_rows
                if row["tree_depth"] == own_depth and row["node"] == node
            )
            higher = [
                row["delta_l2"] for row in local_rows
                if row["relation"] == "path" and row["tree_depth"] < own_depth
            ]
            total = on_energy + off_energy
            summary = {
                "channel": channel,
                "coord": coord,
                "merge": merge,
                "node": node,
                "epsilon": epsilon,
                "own_delta_l2": own_delta,
                "max_ancestor_delta_l2": max(higher, default=0.0),
                "root_delta_l2": local_rows[0]["delta_l2"],
                "on_path_energy": on_energy,
                "off_path_energy": off_energy,
                "off_path_ratio": off_energy / max(total, 1e-30),
            }
            summaries.append(summary)
            for row in local_rows:
                details.append({**summary, **row})
    return baseline, active, summaries, details


def traced_read(decoder, hidden, tree, masks):
    base_query = decoder.query(hidden)
    query = base_query
    frontier = masks[0].to(query.dtype)
    gain = torch.sigmoid(decoder.read_gain_logit)
    last_depth = len(tree) - 1
    trace = []
    local = None
    for depth, (nodes, valid) in enumerate(zip(tree, masks)):
        frontier = frontier * valid.to(frontier.dtype)
        frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
        local = (frontier[:, :, None] * nodes).sum(1)
        depth_state = decoder.depth_embedding.weight[depth][None].expand_as(local)
        query = query + gain * decoder.read_kernel(query, local, depth_state)
        trace.append(frontier.detach().clone())
        if depth == last_depth:
            break
        children = tree[depth + 1].reshape(
            nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
        )
        child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
        branch_query = (decoder.branch(hidden) + gain * (query - base_query))[:, None, None]
        scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
        scores = scores.masked_fill(~child_valid, -1e9)
        probability = F.softmax(scores, dim=-1)
        probability = probability * child_valid.to(probability.dtype)
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
        frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
    return local + (query - base_query), trace


@torch.no_grad()
def decode_with_routes(model, trees, bos: int, steps: int, prefixes=None):
    base_tree = model.reconstructor.convolve(trees["base_tree"], trees["base_masks"])
    extra_tree = model.extra.reconstructor.convolve(trees["extra_tree"], trees["extra_masks"])
    batch = base_tree[0].shape[0]
    base_hidden = base_tree[0].new_zeros((batch, model.reconstructor.hidden))
    extra_hidden = extra_tree[0].new_zeros((batch, model.extra.reconstructor.hidden))
    previous = torch.full((batch,), bos, dtype=torch.long, device=base_tree[0].device)
    logits_rows = []
    predicted = []
    routes = {"base": [], "extra": []}
    for step in range(steps):
        base_context, base_route = traced_read(
            model.reconstructor, base_hidden, base_tree, trees["base_masks"],
        )
        base_hidden = model.reconstructor.cell(
            torch.cat((model.reconstructor.embedding(previous), base_context), dim=-1), base_hidden,
        )
        logits = model.reconstructor.output(torch.cat((base_hidden, base_context), dim=-1))
        extra_context, extra_route = traced_read(
            model.extra.reconstructor, extra_hidden, extra_tree, trees["extra_masks"],
        )
        extra_hidden = model.extra.reconstructor.cell(
            torch.cat((model.extra.reconstructor.embedding(previous), extra_context), dim=-1), extra_hidden,
        )
        extra_logits = model.extra.reconstructor.output(
            torch.cat((extra_hidden, extra_context), dim=-1),
        )
        logits = logits + torch.tanh(model.extra_logit_gain) * extra_logits
        token = logits.argmax(-1)
        logits_rows.append(logits)
        predicted.append(token)
        routes["base"].append(base_route)
        routes["extra"].append(extra_route)
        previous = (prefixes[:, step] if prefixes is not None else token).detach()
    return torch.stack(logits_rows, 1), torch.stack(predicted, 1), routes


def js_divergence(left, right) -> float:
    left = left.float().clamp_min(1e-12)
    right = right.float().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    value = 0.5 * (
        (left * (left.log() - middle.log())).sum()
        + (right * (right.log() - middle.log())).sum()
    )
    return float(value.cpu())


def compare_routes(label, baseline, pulse) -> list[dict]:
    rows = []
    for channel in ("base", "extra"):
        for step, (base_step, pulse_step) in enumerate(zip(baseline[channel], pulse[channel])):
            for depth, (left, right) in enumerate(zip(base_step, pulse_step)):
                p, q = left[0], right[0]
                rows.append({
                    "intervention": label,
                    "channel": channel,
                    "decode_step": step,
                    "tree_depth": depth,
                    "js": js_divergence(p, q),
                    "max_abs_delta": float((p - q).abs().max().cpu()),
                    "baseline_argmax": int(p.argmax().cpu()),
                    "pulse_argmax": int(q.argmax().cpu()),
                    "branch_flip": int(p.argmax() != q.argmax()),
                    "baseline_entropy": float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum().cpu()),
                    "pulse_entropy": float(-(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum().cpu()),
                })
    return rows


def spectrum_rows(label, logits, concepts_raw, sp) -> list[dict]:
    probability = F.softmax(logits[0].float(), dim=-1)
    rows = []
    for concept in ("push", "stone"):
        for surface in concepts_raw[concept]:
            ids = sp.encode(surface, out_type=int)
            for piece_index, token_id in enumerate(ids):
                for step in range(probability.shape[0]):
                    value = probability[step, token_id]
                    rank = 1 + int((probability[step] > value).sum().cpu())
                    rows.append({
                        "intervention": label,
                        "concept": concept,
                        "surface": surface,
                        "piece_index": piece_index,
                        "token_id": token_id,
                        "piece": sp.id_to_piece(token_id),
                        "decode_step": step,
                        "probability": float(value.cpu()),
                        "rank": rank,
                    })
    return rows


def spectrum_summary(label, logits, concepts, concepts_raw, sp) -> dict:
    probability = F.softmax(logits[0].float(), dim=-1)
    result = {"intervention": label, "concepts": {}}
    for concept in ("push", "stone"):
        coverage = f04.concept_coverage(probability, concepts[concept])
        best_piece_rank = probability.shape[-1]
        max_piece_probability = 0.0
        best_surface_rank = probability.shape[-1]
        best_surface_probability = 0.0
        best_surface = None
        for surface in concepts_raw[concept]:
            token_ids = sp.encode(surface, out_type=int)
            for token_id in token_ids:
                values = probability[:, token_id]
                max_piece_probability = max(max_piece_probability, float(values.max().cpu()))
                for step in range(probability.shape[0]):
                    rank = 1 + int((probability[step] > probability[step, token_id]).sum().cpu())
                    best_piece_rank = min(best_piece_rank, rank)
            if len(token_ids) == 1:
                token_id = token_ids[0]
                values = probability[:, token_id]
                local_probability = float(values.max().cpu())
                local_rank = min(
                    1 + int((probability[step] > probability[step, token_id]).sum().cpu())
                    for step in range(probability.shape[0])
                )
                if local_rank < best_surface_rank or (
                    local_rank == best_surface_rank and local_probability > best_surface_probability
                ):
                    best_surface_rank = local_rank
                    best_surface_probability = local_probability
                    best_surface = surface
        result["concepts"][concept] = {
            "phrase_coverage": float(coverage.cpu()),
            "best_any_piece_rank": best_piece_rank,
            "max_any_piece_probability": max_piece_probability,
            "best_single_token_surface": best_surface,
            "best_single_token_surface_rank": best_surface_rank if best_surface is not None else None,
            "best_single_token_surface_probability": (
                best_surface_probability if best_surface is not None else None
            ),
        }
    return result


@torch.no_grad()
def context_case(model, theta, single_source, single_lengths, batch_source, batch_lengths, bos, depth, steps):
    single_logits, single_tokens, _ = f04.decode_logits(
        model, theta, single_source, single_lengths, bos, depth, steps,
    )
    _, batch_tokens, _ = f04.decode_logits(
        model, theta, batch_source, batch_lengths, bos, depth, steps,
    )
    fixed_prefixes = batch_tokens.clone()
    fixed_prefixes[0] = single_tokens[0]
    batch_logits, batch_fixed_tokens, _ = f04.decode_logits(
        model, theta, batch_source, batch_lengths, bos, depth, steps,
        prefixes=fixed_prefixes,
    )
    return {
        "step0_logit_max_abs_delta": float(
            (single_logits[0, 0] - batch_logits[0, 0]).abs().max().cpu()
        ),
        "fixed_history_logit_max_abs_delta": float(
            (single_logits[0] - batch_logits[0]).abs().max().cpu()
        ),
        "local_argmax_changes": int((single_tokens[0] != batch_fixed_tokens[0]).sum().cpu()),
        "single_tokens": single_tokens[0].detach().cpu().tolist(),
        "batch_tokens": batch_fixed_tokens[0].detach().cpu().tolist(),
    }


def batch_width_audit(model, theta, test_rows, sp, pieces, eos, bos, device, depth, steps):
    single_source, single_lengths = f04.encode_rows(
        [test_rows[0]], sp, pieces, eos, pieces, device,
    )
    batch_source, batch_lengths = f04.encode_rows(
        test_rows, sp, pieces, eos, pieces, device,
    )
    narrow_source, narrow_lengths = f04.encode_rows(
        [test_rows[0]] * len(test_rows), sp, pieces, eos, pieces, device,
    )
    encoder = model.frozen_source.encoder
    original = bool(encoder.dynamic_width)
    try:
        encoder.dynamic_width = True
        dynamic = context_case(
            model, theta, single_source, single_lengths, batch_source, batch_lengths,
            bos, depth, steps,
        )
        dynamic_equal_batch = context_case(
            model, theta, narrow_source, narrow_lengths, batch_source, batch_lengths,
            bos, depth, steps,
        )
        encoder.dynamic_width = False
        fixed = context_case(
            model, theta, single_source, single_lengths, batch_source, batch_lengths,
            bos, depth, steps,
        )
        fixed_equal_batch = context_case(
            model, theta, narrow_source, narrow_lengths, batch_source, batch_lengths,
            bos, depth, steps,
        )
    finally:
        encoder.dynamic_width = original
    for case in (dynamic, fixed, dynamic_equal_batch, fixed_equal_batch):
        case["single_text"] = sp.decode(f04.clean(case.pop("single_tokens"), eos, pieces))
        case["batch_text"] = sp.decode(f04.clean(case.pop("batch_tokens"), eos, pieces))
    return {
        "original_dynamic_width": original,
        "single_source_width": int(single_source.shape[1]),
        "single_true_length": int(single_lengths[0]),
        "batch_source_width": int(batch_source.shape[1]),
        "batch_true_lengths": batch_lengths.detach().cpu().tolist(),
        "dynamic": dynamic,
        "fixed_32": fixed,
        "dynamic_equal_batch": dynamic_equal_batch,
        "fixed_32_equal_batch": fixed_equal_batch,
    }


def selected_route_coords(active: list[int]) -> list[int]:
    if not active:
        return []
    by_merge = sorted(active, key=lambda coord: (coord_parts(coord)[0], coord))
    return list(dict.fromkeys((by_merge[0], by_merge[-1])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--theta-checkpoint", required=True)
    parser.add_argument("--specimens", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--sentence-id", default="test-sisyphus-push-stone")
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11601)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    test_rows = [row for row in all_rows if row["split"] == "test"]
    batch_audit = batch_width_audit(
        model, theta, test_rows, sp, pieces, eos, bos, args.device, args.depth, args.steps,
    )
    source, lengths = f04.encode_rows([specimen], sp, pieces, eos, pieces, args.device)
    concepts = f04.compile_concepts(concepts_raw, sp, args.device)
    original_dynamic_width = bool(model.frozen_source.encoder.dynamic_width)
    model.frozen_source.encoder.dynamic_width = False
    baseline_trees = protocol_trees(model, theta, source, lengths, args.depth)

    atlas = parameter_atlas(model, "model") + parameter_atlas(theta, "f04-guided-theta")
    write_csv(output / "parameter_atlas.csv", atlas)
    write_json(output / "parameter_atlas.json", atlas)

    epsilons = [-0.1, -0.01, -0.001, 0.001, 0.01, 0.1]
    base_tree, base_active, base_summary, base_detail = scan_channel_impulses(
        theta.base, baseline_trees["base_slots"], baseline_trees["base_masks"][-1], "base", epsilons,
    )
    extra_tree, extra_active, extra_summary, extra_detail = scan_channel_impulses(
        theta.extra, baseline_trees["extra_slots"], baseline_trees["extra_masks"][-1], "extra", epsilons,
    )
    impulse_summary = base_summary + extra_summary
    write_csv(output / "path_impulses.csv", impulse_summary)
    write_json(output / "path_impulse_nodes.json", base_detail + extra_detail)

    baseline_traced_logits, baseline_tokens, baseline_routes = decode_with_routes(
        model, baseline_trees, bos, args.steps,
    )
    baseline_native_logits, baseline_native_tokens, _ = f04.decode_logits(
        model, theta, source, lengths, bos, args.depth, args.steps,
    )
    reader_logit_delta = float((baseline_traced_logits - baseline_native_logits).abs().max().cpu())
    reader_token_exact = bool(torch.equal(baseline_tokens, baseline_native_tokens))

    route_rows = []
    spectra = [spectrum_summary("baseline", baseline_traced_logits, concepts, concepts_raw, sp)]
    spectrum_detail = spectrum_rows("baseline", baseline_traced_logits, concepts_raw, sp)
    intervention_records = []
    for channel, active in (("base", base_active), ("extra", extra_active)):
        for coord in selected_route_coords(active):
            pulse = torch.zeros((1, 31), device=args.device)
            pulse[0, coord] = 0.1
            label = f"{channel}-coord{coord}-eps+0.1"
            trees = protocol_trees(
                model, theta, source, lengths, args.depth,
                base_filter=pulse if channel == "base" else None,
                extra_filter=pulse if channel == "extra" else None,
            )
            logits, tokens, routes = decode_with_routes(
                model, trees, bos, args.steps, prefixes=baseline_tokens,
            )
            local_route_rows = compare_routes(label, baseline_routes, routes)
            route_rows.extend(local_route_rows)
            spectra.append(spectrum_summary(label, logits, concepts, concepts_raw, sp))
            spectrum_detail.extend(spectrum_rows(label, logits, concepts_raw, sp))
            merge, node = coord_parts(coord)
            intervention_records.append({
                "label": label,
                "channel": channel,
                "coord": coord,
                "merge": merge,
                "node": node,
                "max_route_js": max(row["js"] for row in local_route_rows),
                "max_route_delta": max(row["max_abs_delta"] for row in local_route_rows),
                "branch_flips": sum(row["branch_flip"] for row in local_route_rows),
                "logit_max_abs_delta": float((logits - baseline_traced_logits).abs().max().cpu()),
                "local_argmax_changes": int((tokens != baseline_tokens).sum().cpu()),
            })

    write_csv(output / "read_route_scope.csv", route_rows)
    write_csv(output / "token_spectrum.csv", spectrum_detail)
    write_json(output / "token_spectrum_summary.json", spectra)

    max_off_path_ratio = max(row["off_path_ratio"] for row in impulse_summary)
    transport_rows = [row for row in impulse_summary if row["merge"] < 4]
    model.frozen_source.encoder.dynamic_width = original_dynamic_width
    frozen_model = f04.frozen_exact(model, expected_model)
    frozen_theta = tensor_equal_state(theta, expected_theta)
    gates = {
        "O0_reader_parity": reader_logit_delta <= 1e-6 and reader_token_exact,
        "O1_topology_locality": max_off_path_ratio <= 1e-12,
        "O2_ancestor_transport": all(
            row["own_delta_l2"] > 0.0 and row["max_ancestor_delta_l2"] > 0.0
            for row in transport_rows
        ),
        "O3_route_observed": bool(route_rows),
        "O4_spectrum_observed": bool(spectrum_detail),
        "O5_frozen_exact": frozen_model and frozen_theta,
        "O6_dynamic_width_isolated": (
            batch_audit["dynamic"]["step0_logit_max_abs_delta"] > 1e-6
            and batch_audit["fixed_32"]["fixed_history_logit_max_abs_delta"] <= 1e-6
            and batch_audit["fixed_32"]["local_argmax_changes"] == 0
        ),
        "O7_equal_batch_width_isolated": (
            batch_audit["dynamic_equal_batch"]["step0_logit_max_abs_delta"] > 1e-6
            and batch_audit["fixed_32_equal_batch"]["fixed_history_logit_max_abs_delta"] <= 1e-6
            and batch_audit["fixed_32_equal_batch"]["local_argmax_changes"] == 0
        ),
    }
    clean_ids = f04.clean(baseline_tokens[0].tolist(), eos, pieces)
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
        "observation_source_width_mode": "fixed_32",
        "batch_width_audit": batch_audit,
        "budget": int(baseline_trees["budgets"][0].cpu()),
        "baseline_generation": sp.decode(clean_ids),
        "reader_parity": {
            "logit_max_abs_delta": reader_logit_delta,
            "token_exact": reader_token_exact,
        },
        "parameter_tensors": len(atlas),
        "parameter_count": sum(int(row["count"]) for row in atlas),
        "active_filter_coordinates": {"base": base_active, "extra": extra_active},
        "impulses": {
            "count": len(impulse_summary),
            "max_off_path_ratio": max_off_path_ratio,
            "max_root_response": max(row["root_delta_l2"] for row in impulse_summary),
        },
        "route_interventions": intervention_records,
        "spectra": spectra,
        "frozen_model": frozen_model,
        "frozen_theta": frozen_theta,
        "gates": gates,
        "seconds": time.time() - started,
        "claim_boundary": "observation instruments only; no path law, semantic coordinate, or quality claim",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete",
        "gates": gates,
        "generation": summary["baseline_generation"],
        "active": summary["active_filter_coordinates"],
        "max_off_path_ratio": max_off_path_ratio,
        "route_interventions": intervention_records,
        "spectra": spectra,
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
