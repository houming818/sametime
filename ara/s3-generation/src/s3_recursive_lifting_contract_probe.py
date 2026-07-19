#!/usr/bin/env python3
"""Deterministic contract audit for addressed recursive lifting TreeHeap."""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_adaptive_lifting_wmt import AdaptiveLiftingEncoder
from s2_lifting_pump_wmt import RecursiveDecoder


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def topology(width: int) -> List[Dict[str, object]]:
    depths = int(math.log2(width))
    rows: List[Dict[str, object]] = []
    for depth in range(depths + 1):
        count = 1 << depth
        span_width = width // count
        for offset in range(count):
            address = count + offset
            start = offset * span_width
            stop = start + span_width
            rows.append(
                {
                    "address": address,
                    "parent": None if address == 1 else address // 2,
                    "children": [] if depth == depths else [2 * address, 2 * address + 1],
                    "depth": depth,
                    "span": [start, stop],
                }
            )
    return rows


def topology_gates(rows: List[Dict[str, object]], width: int) -> dict:
    by_address = {int(row["address"]): row for row in rows}
    expected = set(range(1, 2 * width))
    addresses = set(by_address)
    unique_parent = all(
        row["address"] == 1 or int(row["parent"]) == int(row["address"]) // 2
        for row in rows
    )
    children_match = all(
        not row["children"]
        or row["children"] == [2 * int(row["address"]), 2 * int(row["address"]) + 1]
        for row in rows
    )
    spans_partition = True
    max_depth = int(math.log2(width))
    for depth in range(max_depth + 1):
        spans = sorted(row["span"] for row in rows if row["depth"] == depth)
        cursor = 0
        for start, stop in spans:
            spans_partition &= start == cursor and stop > start
            cursor = stop
        spans_partition &= cursor == width
    return {
        "complete_heap_addresses": addresses == expected,
        "unique_parent": unique_parent,
        "children_match_heap_rule": children_match,
        "each_depth_partitions_leaf_span": spans_partition,
    }


def capped_read_audit(
    decoder: RecursiveDecoder,
    hidden: torch.Tensor,
    levels: List[torch.Tensor],
    masks: List[torch.Tensor],
    max_depth: int,
) -> dict:
    query = decoder.query(hidden)
    active = torch.ones(hidden.shape[0], 1, device=hidden.device)
    maximum_error = 0.0
    stop_mass_by_depth = []
    trace = []
    context = torch.zeros(hidden.shape[0], levels[0].shape[-1], device=hidden.device)
    for depth, (nodes, valid) in enumerate(zip(levels, masks)):
        active = active * valid.to(active.dtype)
        incoming = active
        last = depth == len(levels) - 1
        capped = depth >= max_depth
        if last or capped:
            stop_probability = torch.ones_like(incoming)
        else:
            depth_state = decoder.depth_embedding.weight[depth][None, None]
            expanded_query = query[:, None].expand_as(nodes)
            stop_probability = torch.sigmoid(
                decoder.stop(torch.cat((expanded_query, nodes + depth_state), -1)).squeeze(-1)
            )
        stopped = incoming * stop_probability
        context = context + (stopped[..., None] * nodes).sum(1)
        stop_mass_by_depth.append(float(stopped.sum(1).mean()))
        if last or capped:
            child_mass = torch.zeros(*incoming.shape, 2, device=hidden.device)
        else:
            children = levels[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2]
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            scores = (
                decoder.branch(hidden)[:, None, None] * children
            ).sum(-1) / math.sqrt(nodes.shape[-1])
            branch_probability = F.softmax(scores.masked_fill(~child_valid, -1e9), -1)
            child_mass = (
                incoming[..., None]
                * (1.0 - stop_probability[..., None])
                * branch_probability
            )
        local_error = (incoming - stopped - child_mass.sum(-1)).abs().max()
        maximum_error = max(maximum_error, float(local_error))
        trace.append(
            {
                "depth": depth,
                "active_nodes": incoming.shape[1],
                "incoming_mass": float(incoming.sum(1).mean()),
                "stop_mass": float(stopped.sum(1).mean()),
                "expanded_mass": float(child_mass.sum((1, 2)).mean()),
                "max_conservation_error": float(local_error),
            }
        )
        if not last and not capped:
            active = child_mass.flatten(1, 2)
        else:
            break
    return {
        "max_depth": max_depth,
        "maximum_error": maximum_error,
        "total_stop_mass": sum(stop_mass_by_depth),
        "stop_mass_by_depth": stop_mass_by_depth,
        "trace": trace,
        "context_norm": float(context.norm(dim=-1).mean().detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_recursive_lifting_contract_probe")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--seed", type=int, default=72064)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.width < 4 or args.width & (args.width - 1):
        raise ValueError("width must be a power of two and at least four")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    encoder = AdaptiveLiftingEncoder(
        args.vocab, args.dim, args.width, 0, learned_update=True, alternate=False
    ).to(device)
    tokens = torch.randint(1, args.vocab, (args.batch, args.width), device=device)
    lengths = torch.full((args.batch,), args.width, dtype=torch.long, device=device)

    leaf, root, details, masks = encoder.fold(tokens, lengths)
    levels, level_masks = encoder.unfold(root, details, masks)
    closure_mse = float(torch.mean((levels[-1] - leaf) ** 2).detach())
    closure_max_abs = float((levels[-1] - leaf).abs().max().detach())

    # Swap two finest-level residual addresses without changing their values.
    shifted = [row.clone() for row in details]
    shifted[0][:, [0, 1]] = shifted[0][:, [1, 0]]
    shifted_levels, _ = encoder.unfold(root, shifted, masks)
    difference = torch.mean((shifted_levels[-1] - leaf) ** 2, dim=(0, 2))
    affected_mse = float(difference[:4].mean().detach())
    outside_mse = float(difference[4:].mean().detach()) if args.width > 4 else 0.0

    rows = topology(args.width)
    topology_result = topology_gates(rows, args.width)
    decoder = RecursiveDecoder(
        args.vocab, args.dim, args.dim, int(math.log2(args.width))
    ).to(device)
    hidden = torch.randn(args.batch, args.dim, device=device)
    cap_audits = [
        capped_read_audit(decoder, hidden, levels, level_masks, cap)
        for cap in range(int(math.log2(args.width)) + 1)
    ]
    route = cap_audits[-1]
    maximum_cap_error = max(row["maximum_error"] for row in cap_audits)
    maximum_stop_mass_error = max(abs(row["total_stop_mass"] - 1.0) for row in cap_audits)
    gates = {
        "C1_full_residual_roundtrip": closure_mse < 1e-10,
        "C2_topology_complete": all(topology_result.values()),
        "C3_route_mass_conserved": maximum_cap_error < 1e-6 and maximum_stop_mass_error < 1e-6,
        "C4_address_swap_is_local": affected_mse > 1e-8 and outside_mse < 1e-12,
        "C5_finite": all(
            math.isfinite(value)
            for value in (
                closure_mse,
                closure_max_abs,
                affected_mse,
                outside_mse,
                maximum_cap_error,
                maximum_stop_mass_error,
            )
        ),
    }
    summary = {
        "claim_id": "S3-TREE-LIFT-RECURSIVE-C01",
        "phase": "deterministic_contract",
        "status": "pass" if all(gates.values()) else "fail",
        "host": socket.gethostname(),
        "config": vars(args),
        "closure": {"mse": closure_mse, "max_abs": closure_max_abs},
        "topology": {"nodes": rows, "gates": topology_result},
        "address_swap": {
            "detail_depth": 0,
            "swapped_detail_addresses": [0, 1],
            "expected_leaf_span": [0, 4],
            "affected_span_mse": affected_mse,
            "outside_span_mse": outside_mse,
        },
        "route_conservation": {
            "maximum_cap_error": maximum_cap_error,
            "maximum_stop_mass_error": maximum_stop_mass_error,
            "caps": cap_audits,
        },
        "gates": gates,
        "boundary": "Deterministic algebra and address contract only; no language or semantic claim.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Recursive lifting contract probe\n\nSee `summary.json`.\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
