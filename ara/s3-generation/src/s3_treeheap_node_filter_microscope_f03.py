#!/usr/bin/env python3
"""Scan a per-node TreeHeap readout filter and preserve direct Decode evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import torch

import s3_recursive_depth_pressure_protocol_training as d07
import s3_structural_protocol_full_pipeline_d10 as d10
from treeheap_epoch_translate_cli import load_runtime
from treeheap_fold_matrix_decode_probe import adjacent_repetition, character_f2


CLAIM = "S3-TREEHEAP-NODE-FILTER-MICROSCOPE-F03"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_specimen(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("F03 requires exactly one preregistered specimen")
    row = rows[0]
    if row.get("direction") != "zh2en" or not row.get("source") or not row.get("reference"):
        raise ValueError("invalid F03 specimen")
    return row


class NodeFilter:
    def __init__(self, original, coordinate: int | None = None, weight: float = 1.0):
        self.original = original
        self.coordinate = coordinate
        self.weight = weight
        self.addresses: list[dict] = []
        self.original_tree: list[torch.Tensor] = []
        self.filtered_tree: list[torch.Tensor] = []

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        tree, masks = self.original(slots, slot_mask)
        self.original_tree = [level.detach().clone() for level in tree]
        filtered = []
        addresses = []
        cursor = 0
        last_level = len(tree) - 1
        for level_index, (level, mask) in enumerate(zip(tree, masks)):
            width = level.shape[1]
            weights = torch.ones(width, dtype=level.dtype, device=level.device)
            if self.coordinate is not None and cursor <= self.coordinate < cursor + width:
                weights[self.coordinate - cursor] = self.weight
            filtered.append(level * weights[None, :, None])
            band = "root" if level_index == 0 else ("leaf" if level_index == last_level else "internal")
            for position in range(width):
                addresses.append({
                    "index": cursor + position,
                    "level": level_index,
                    "band": band,
                    "level_width": width,
                    "position": position,
                    "active": bool(mask[0, position].item()),
                })
            cursor += width
        self.addresses = addresses
        self.filtered_tree = [level.detach().clone() for level in filtered]
        return filtered, masks


@contextmanager
def node_filter(coordinate: int | None = None, weight: float = 1.0):
    original = d07.fold_protocol
    intervention = NodeFilter(original, coordinate, weight)
    d07.fold_protocol = intervention
    try:
        yield intervention
    finally:
        d07.fold_protocol = original


def semantic_slots(text: str, slot_terms: dict[str, list[str]]) -> dict[str, bool]:
    lowered = text.lower()
    return {
        name: all(term.lower() in lowered for term in terms)
        for name, terms in slot_terms.items()
    }


@torch.inference_mode()
def decode(runtime, specimen: dict, depth: int, max_len: int, coordinate=None, weight=1.0, filtered=True):
    payload, args, sp, model, pieces, eos, bos, _ = runtime
    raw = sp.encode(specimen["source"].strip(), out_type=int)
    source_ids = [pieces + 2, *raw[:32], eos]
    source = torch.tensor([source_ids], dtype=torch.long, device=args.device)
    lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)
    if filtered:
        with node_filter(coordinate, weight) as observation:
            generated, route, budgets, _, _ = model.greedy(source, lengths, bos, eos, max_len, depth)
    else:
        observation = None
        generated, route, budgets, _, _ = model.greedy(source, lengths, bos, eos, max_len, depth)
    raw_ids = generated[0].tolist()
    eos_reached = eos in raw_ids
    clean_ids = d10.wmt.clean(raw_ids, eos, pieces)
    output = sp.decode(clean_ids)
    row = {
        "coordinate": coordinate,
        "weight": weight,
        "output": output,
        "raw_ids": raw_ids,
        "clean_ids": clean_ids,
        "output_pieces": len(clean_ids),
        "eos_reached": eos_reached,
        "adjacent_repetition_rate": adjacent_repetition(clean_ids),
        "character_f2_diagnostic": character_f2(output, specimen["reference"]),
        "semantic_slots": semantic_slots(output, specimen["semantic_slots"]),
        "route_depth_mass": [float(value) for value in route.detach().cpu()],
        "budget": int(budgets[0]),
    }
    return row, observation


def signature(row: dict) -> tuple[int, ...]:
    return tuple(row["clean_ids"])


def grid(start: int, stop: int, divisor: int) -> list[float]:
    return [value / divisor for value in range(start, stop + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.time()
    checkpoint = Path(args.checkpoint)
    specimen_path = Path(args.specimen)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    specimen = load_specimen(specimen_path)
    runtime = load_runtime(checkpoint, args.device)

    native, _ = decode(runtime, specimen, args.depth, args.max_new_tokens, filtered=False)
    identity, identity_observation = decode(runtime, specimen, args.depth, args.max_new_tokens)
    assert identity_observation is not None
    tree_exact = all(
        torch.equal(before, after)
        for before, after in zip(identity_observation.original_tree, identity_observation.filtered_tree)
    )
    token_exact = native["raw_ids"] == identity["raw_ids"]
    addresses = identity_observation.addresses
    active = [address for address in addresses if address["active"]]
    address_by_index = {address["index"]: address for address in active}

    if args.mode == "smoke":
        root = next(address for address in active if address["band"] == "root")
        middle = next(address for address in active if address["band"] == "internal")
        leaf = next(address for address in active if address["band"] == "leaf")
        scan_addresses = [root, middle, leaf]
        coarse_weights = [0.0, 0.5, 1.0, 1.5]
    else:
        scan_addresses = active
        coarse_weights = grid(0, 30, 20)

    coarse_path = output / "coarse_results.jsonl"
    if coarse_path.exists():
        coarse_path.unlink()
    rows_by_address: dict[int, list[dict]] = {}
    total = len(scan_addresses) * len(coarse_weights)
    completed = 0
    for address in scan_addresses:
        group = []
        for weight in coarse_weights:
            row, _ = decode(
                runtime, specimen, args.depth, args.max_new_tokens,
                coordinate=address["index"], weight=weight,
            )
            row["address"] = address
            append_jsonl(coarse_path, row)
            group.append(row)
            completed += 1
        rows_by_address[address["index"]] = group
        if completed == total or completed % max(1, len(coarse_weights) * 10) == 0:
            print(json.dumps({"event": "progress", "completed": completed, "total": total}), flush=True)

    atlas = []
    for coordinate, group in rows_by_address.items():
        distinct = len({signature(row) for row in group})
        transitions = [
            (left["weight"], right["weight"])
            for left, right in zip(group, group[1:])
            if signature(left) != signature(right)
        ]
        atlas.append({
            "address": address_by_index[coordinate],
            "distinct_outputs": distinct,
            "transition_count": len(transitions),
            "coarse_transition_intervals": transitions,
            "baseline_output": next(row["output"] for row in group if math.isclose(row["weight"], 1.0)),
        })

    refinement_plan = []
    fine_rows = []
    if args.mode == "formal":
        candidates = sorted(
            (item for item in atlas if item["transition_count"]),
            key=lambda item: (-item["distinct_outputs"], item["address"]["index"]),
        )[:8]
        for item in candidates:
            intervals = item["coarse_transition_intervals"]
            lower, upper = min(intervals, key=lambda pair: (abs((pair[0] + pair[1]) / 2 - 1.0), pair[0]))
            lower_milli, upper_milli = round(lower * 1000), round(upper * 1000)
            refinement_plan.append({"coordinate": item["address"]["index"], "lower": lower, "upper": upper})
            previous = None
            transitions = []
            for weight in grid(lower_milli, upper_milli, 1000):
                row, _ = decode(
                    runtime, specimen, args.depth, args.max_new_tokens,
                    coordinate=item["address"]["index"], weight=weight,
                )
                row["address"] = item["address"]
                fine_rows.append(row)
                if previous is None or signature(previous) != signature(row):
                    transitions.append({
                        "weight": weight,
                        "output": row["output"],
                        "clean_ids": row["clean_ids"],
                        "semantic_slots": row["semantic_slots"],
                    })
                previous = row
            item["fine_interval"] = [lower, upper]
            item["fine_transition_atlas"] = transitions
        fine_path = output / "fine_results.jsonl"
        if fine_path.exists():
            fine_path.unlink()
        for row in fine_rows:
            append_jsonl(fine_path, row)

    changed = [item for item in atlas if item["distinct_outputs"] > 1]
    changed_bands = sorted({item["address"]["band"] for item in changed})
    fine_resolved = any(len(item.get("fine_transition_atlas", [])) > 1 for item in atlas)
    expected_coarse = len(scan_addresses) * len(coarse_weights)
    actual_coarse = sum(len(group) for group in rows_by_address.values())
    finite = all(
        math.isfinite(row["character_f2_diagnostic"])
        and math.isfinite(row["adjacent_repetition_rate"])
        for group in rows_by_address.values() for row in group
    )
    gates = {
        "P0_identity_contract": tree_exact and token_exact,
        "P1_complete_finite_scan": actual_coarse == expected_coarse and finite,
        "P2_node_focus_effect": bool(changed),
        "P3_cross_resolution_effect": len(changed_bands) >= 2,
        "P4_fine_transition_resolved": fine_resolved if args.mode == "formal" else None,
    }
    report = {
        "claim": CLAIM,
        "mode": args.mode,
        "host": socket.gethostname(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "specimen_path": str(specimen_path),
        "specimen_sha256": sha256(specimen_path),
        "specimen": specimen,
        "depth": args.depth,
        "filter_position": "after native fold_protocol and before Decoder READ",
        "address_order": "root-to-leaf, level-major, left-to-right",
        "full_node_count": len(addresses),
        "active_node_count": len(active),
        "level_widths": [len([a for a in addresses if a["level"] == level]) for level in range(max(a["level"] for a in addresses) + 1)],
        "addresses": addresses,
        "native": native,
        "identity": identity,
        "identity_tree_exact": tree_exact,
        "identity_token_exact": token_exact,
        "coarse_weights": coarse_weights,
        "coarse_observations_expected": expected_coarse,
        "coarse_observations_actual": actual_coarse,
        "changed_node_count": len(changed),
        "changed_bands": changed_bands,
        "refinement_plan": refinement_plan,
        "fine_observations": len(fine_rows),
        "atlas": atlas,
        "gates": gates,
        "seconds": time.time() - started,
        "claim_boundary": "single checkpoint, specimen, and depth; direct Decode is primary evidence",
    }
    write_json(output / "summary.json", report)
    print(json.dumps({
        "event": "complete", "mode": args.mode,
        "active_nodes": len(active), "changed_nodes": len(changed),
        "changed_bands": changed_bands, "gates": gates,
        "seconds": report["seconds"],
    }, ensure_ascii=False), flush=True)
    if not gates["P0_identity_contract"] or not gates["P1_complete_finite_scan"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
