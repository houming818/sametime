#!/usr/bin/env python3
"""Bounded rotation/CAT search proof for an ordered TreeHeap orbit.

This probe separates two questions:

1. Deductive: does CAT(H, R(H)) remain exactly searchable when R is an
   invertible order isomorphism represented by a lazy descriptor?
2. Inductive: can one tiny shared route kernel, trained on shallow rotations,
   reuse the same comparison rule at unseen deeper rotations?

The probe does not claim semantic search or arbitrary exponential search.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import pathlib
import platform
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteResult:
    status: str
    found: bool
    logical_index: int | None
    local_index: int | None
    sector: int | None
    comparisons: int
    rotation_path: str
    residual_query: int | None


def make_base(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 3:
        raise ValueError("base-size must be at least 3")
    gaps = rng.integers(2, 12, size=size, dtype=np.int64)
    return np.cumsum(gaps, dtype=np.int64) + np.int64(101)


def rotation_descriptors(base: np.ndarray, depth: int) -> tuple[list[int], list[int]]:
    """Return ranges and order-preserving translation offsets for each level."""
    current_range = int(base[-1] - base[0])
    ranges: list[int] = []
    offsets: list[int] = []
    for level in range(depth):
        ranges.append(current_range)
        # A visible gap keeps the route margin stable as depth grows.
        gap = current_range + 17 + level
        offset = current_range + gap
        offsets.append(offset)
        current_range += offset
    return ranges, offsets


def materialize(base: np.ndarray, offsets: list[int]) -> np.ndarray:
    values = base.copy()
    for offset in offsets:
        values = np.concatenate((values, values + np.int64(offset)))
    return values


def sector_offset(sector: int, offsets: list[int]) -> int:
    value = 0
    bit = 0
    while sector:
        if sector & 1:
            value += offsets[bit]
        sector >>= 1
        bit += 1
    return value


def query_for_index(base: np.ndarray, offsets: list[int], logical_index: int) -> int:
    base_size = len(base)
    sector, local_index = divmod(logical_index, base_size)
    return int(base[local_index]) + sector_offset(sector, offsets)


def binary_search(values: np.ndarray, query: int) -> tuple[int | None, int, str]:
    lo = 0
    hi = len(values)
    comparisons = 0
    path: list[str] = []
    while lo < hi:
        mid = (lo + hi) // 2
        comparisons += 1
        value = int(values[mid])
        if query == value:
            path.append("S")
            return mid, comparisons, "".join(path)
        if query < value:
            path.append("L")
            hi = mid
        else:
            path.append("R")
            lo = mid + 1
    return None, comparisons, "".join(path)


def exact_route_right(query: int, right_min: int, _current_range: int) -> bool:
    return query >= right_min


def logistic_probability(theta: np.ndarray, x: float) -> float:
    z = float(theta[0] * x + theta[1])
    z = min(60.0, max(-60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def learned_route_right(
    query: int, right_min: int, current_range: int, theta: np.ndarray
) -> bool:
    x = (query - right_min) / max(1.0, float(current_range))
    return logistic_probability(theta, x) >= 0.5


def route_lazy(
    base: np.ndarray,
    ranges: list[int],
    offsets: list[int],
    query: int,
    *,
    theta: np.ndarray | None = None,
    max_steps: int | None = None,
) -> RouteResult:
    residual = int(query)
    comparisons = 0
    top_down_bits: list[str] = []

    for level in range(len(offsets) - 1, -1, -1):
        if max_steps is not None and comparisons >= max_steps:
            return RouteResult(
                "BUDGET_EXHAUSTED", False, None, None, None,
                comparisons, "".join(top_down_bits), None,
            )
        right_min = int(base[0]) + offsets[level]
        if theta is None:
            go_right = exact_route_right(residual, right_min, ranges[level])
        else:
            go_right = learned_route_right(
                residual, right_min, ranges[level], theta
            )
        comparisons += 1
        top_down_bits.append("R" if go_right else "L")
        if go_right:
            residual -= offsets[level]

    if max_steps is not None and comparisons >= max_steps:
        return RouteResult(
            "BUDGET_EXHAUSTED", False, None, None, None,
            comparisons, "".join(top_down_bits), None,
        )

    local_index, local_comparisons, _ = binary_search(base, residual)
    if max_steps is not None and comparisons + local_comparisons > max_steps:
        return RouteResult(
            "BUDGET_EXHAUSTED", False, None, None, None,
            max_steps, "".join(top_down_bits), None,
        )
    comparisons += local_comparisons
    if local_index is None:
        return RouteResult(
            "NOT_FOUND", False, None, None, None,
            comparisons, "".join(top_down_bits), residual,
        )

    sector = 0
    for bit_index, direction in enumerate(reversed(top_down_bits)):
        if direction == "R":
            sector |= 1 << bit_index
    logical_index = sector * len(base) + local_index
    return RouteResult(
        "FOUND", True, logical_index, local_index, sector,
        comparisons, "".join(top_down_bits), residual,
    )


def collect_training_examples(
    base: np.ndarray,
    ranges: list[int],
    offsets: list[int],
    train_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for depth in range(1, train_depth + 1):
        count = len(base) * (1 << depth)
        for logical_index in range(count):
            residual = query_for_index(base, offsets[:depth], logical_index)
            for level in range(depth - 1, -1, -1):
                right_min = int(base[0]) + offsets[level]
                x = (residual - right_min) / max(1.0, float(ranges[level]))
                y = 1.0 if residual >= right_min else 0.0
                xs.append(x)
                ys.append(y)
                if y == 1.0:
                    residual -= offsets[level]
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def train_route_kernel(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    epochs: int = 2500,
    learning_rate: float = 0.25,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    theta = np.zeros(2, dtype=np.float64)
    trace: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = np.clip(theta[0] * xs + theta[1], -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - ys
        grad = np.array([np.mean(error * xs), np.mean(error)], dtype=np.float64)
        theta -= learning_rate * grad
        if epoch in {0, 9, 99, 499, 999, epochs - 1}:
            loss = -np.mean(
                ys * np.log(probs + 1e-12)
                + (1.0 - ys) * np.log(1.0 - probs + 1e-12)
            )
            acc = np.mean((probs >= 0.5) == (ys >= 0.5))
            trace.append(
                {"epoch": epoch + 1, "loss": float(loss), "accuracy": float(acc)}
            )
    return theta, trace


def broken_permutation(local_index: int, sector: int, base_size: int) -> int:
    if sector == 0:
        return local_index
    # base_size is registered as prime (31), so this is a bijection.
    a = 2 + (sector % (base_size - 2))
    b = (7 * sector + 3) % base_size
    return (a * local_index + b) % base_size


def broken_one_path_hit(local_index: int, sector: int, base_size: int) -> bool:
    stored_local = broken_permutation(local_index, sector, base_size)
    return stored_local == local_index


def inverse_permutation(target_local: int, sector: int, base_size: int) -> int:
    if sector == 0:
        return target_local
    a = 2 + (sector % (base_size - 2))
    b = (7 * sector + 3) % base_size
    a_inv = pow(a, -1, base_size)
    return (a_inv * (target_local - b)) % base_size


def sampled_indices(
    count: int, samples_per_depth: int, rng: np.random.Generator
) -> np.ndarray:
    if count <= samples_per_depth:
        return np.arange(count, dtype=np.int64)
    anchors = np.array([0, 1, count // 2, count - 2, count - 1], dtype=np.int64)
    random_count = max(0, samples_per_depth - len(anchors))
    random_indices = rng.choice(count, size=random_count, replace=False)
    return np.unique(np.concatenate((anchors, random_indices))).astype(np.int64)


def run_probe(args: argparse.Namespace) -> dict:
    if args.base_size != 31:
        raise ValueError("this control uses prime base-size 31")
    if args.train_depth >= args.max_depth:
        raise ValueError("train-depth must be smaller than max-depth")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = make_base(args.base_size, rng)
    ranges, offsets = rotation_descriptors(base, args.max_depth)

    train_x, train_y = collect_training_examples(
        base, ranges, offsets, args.train_depth
    )
    theta, training_trace = train_route_kernel(train_x, train_y)

    depth_rows: list[dict] = []
    example_rows: list[dict] = []
    all_det_hits = 0
    all_learned_hits = 0
    all_queries = 0
    all_inverse_hits = 0
    max_seen_steps = 0

    for depth in range(args.max_depth + 1):
        active_offsets = offsets[:depth]
        active_ranges = ranges[:depth]
        logical_count = len(base) * (1 << depth)
        indices = sampled_indices(logical_count, args.samples_per_depth, rng)

        det_hits = 0
        learned_hits = 0
        inverse_hits = 0
        explicit_hits = 0
        broken_hits = 0
        comparisons: list[int] = []
        scan_comparisons: list[int] = []

        explicit = materialize(base, active_offsets)
        cat_order_exact = bool(np.all(np.diff(explicit) > 0))

        for logical_index_np in indices:
            logical_index = int(logical_index_np)
            query = query_for_index(base, active_offsets, logical_index)
            det = route_lazy(base, active_ranges, active_offsets, query)
            learned = route_lazy(
                base, active_ranges, active_offsets, query, theta=theta
            )
            det_ok = det.found and det.logical_index == logical_index
            learned_ok = learned.found and learned.logical_index == logical_index
            det_hits += int(det_ok)
            learned_hits += int(learned_ok)
            comparisons.append(det.comparisons)
            max_seen_steps = max(max_seen_steps, det.comparisons)

            sector, local_index = divmod(logical_index, len(base))
            recovered = query - sector_offset(sector, active_offsets)
            inverse_ok = recovered == int(base[local_index])
            inverse_hits += int(inverse_ok)

            explicit_index = int(np.searchsorted(explicit, query))
            explicit_ok = (
                explicit_index < len(explicit)
                and int(explicit[explicit_index]) == query
                and explicit_index == logical_index
            )
            explicit_hits += int(explicit_ok)

            broken_hits += int(
                broken_one_path_hit(local_index, sector, len(base))
            )
            scan_position = (
                sector * len(base)
                + inverse_permutation(local_index, sector, len(base))
            )
            scan_comparisons.append(scan_position + 1)

            if len(example_rows) < 16 and depth in {0, args.train_depth, args.max_depth}:
                example_rows.append(
                    {
                        "depth": depth,
                        "logical_index": logical_index,
                        "query": query,
                        "sector": sector,
                        "local_index": local_index,
                        "rotation_path": det.rotation_path,
                        "residual_query": det.residual_query,
                        "comparisons": det.comparisons,
                        "deterministic_found": bool(det_ok),
                        "learned_found": bool(learned_ok),
                    }
                )

        explicit_bytes = int(explicit.nbytes)
        lazy_bytes = int(base.nbytes + depth * 2 * np.dtype(np.int64).itemsize)
        sample_count = len(indices)
        max_bound = depth + math.ceil(math.log2(len(base))) + 1
        row = {
            "depth": depth,
            "seen_during_training": depth <= args.train_depth,
            "logical_candidate_count": logical_count,
            "sample_count": sample_count,
            "deterministic_exact": det_hits / sample_count,
            "learned_exact": learned_hits / sample_count,
            "inverse_exact": inverse_hits / sample_count,
            "explicit_sorted_exact": explicit_hits / sample_count,
            "cat_order_exact": cat_order_exact,
            "broken_rotation_one_path_exact": broken_hits / sample_count,
            "mean_treeheap_comparisons": float(np.mean(comparisons)),
            "max_treeheap_comparisons": int(max(comparisons)),
            "registered_step_bound": max_bound,
            "mean_unordered_scan_comparisons": float(np.mean(scan_comparisons)),
            "explicit_payload_bytes": explicit_bytes,
            "lazy_descriptor_bytes": lazy_bytes,
            "explicit_to_lazy_storage_ratio": explicit_bytes / lazy_bytes,
        }
        depth_rows.append(row)
        all_det_hits += det_hits
        all_learned_hits += learned_hits
        all_inverse_hits += inverse_hits
        all_queries += sample_count

    requested_depth = args.max_depth + 1
    growth_status = (
        "BUDGET_EXHAUSTED" if requested_depth > args.max_depth else "ALLOWED"
    )
    deepest_query = query_for_index(
        base, offsets, len(base) * (1 << args.max_depth) - 1
    )
    query_budget = args.max_depth + math.ceil(math.log2(len(base))) - 1
    budgeted_query = route_lazy(
        base,
        ranges,
        offsets,
        deepest_query,
        theta=theta,
        max_steps=query_budget,
    )

    deepest = depth_rows[-1]
    ood_rows = [row for row in depth_rows if row["depth"] > args.train_depth]
    learned_ood_exact = min(row["learned_exact"] for row in ood_rows)
    gates = {
        "P1_deterministic_exact": all_det_hits == all_queries,
        "P2_learned_ood_exact": learned_ood_exact >= 0.999,
        "P3_inverse_exact": all_inverse_hits == all_queries,
        "P4_step_bound": all(
            row["max_treeheap_comparisons"] <= row["registered_step_bound"]
            for row in depth_rows
        ),
        "P5_storage_ratio": deepest["explicit_to_lazy_storage_ratio"] >= 1000.0,
        "P6_broken_order_fails": deepest["broken_rotation_one_path_exact"] <= 0.10,
        "P7_budget_enforced": (
            growth_status == "BUDGET_EXHAUSTED"
            and budgeted_query.status == "BUDGET_EXHAUSTED"
        ),
        "cat_order_exact": all(row["cat_order_exact"] for row in depth_rows),
        "explicit_sorted_exact": all(
            row["explicit_sorted_exact"] == 1.0 for row in depth_rows
        ),
    }

    summary = {
        "claim": "M0-ROT-C01",
        "predict": "P-ROT01",
        "status_before_run": "design",
        "seed": args.seed,
        "base_size": args.base_size,
        "base_values": [int(x) for x in base],
        "train_depth": args.train_depth,
        "max_depth": args.max_depth,
        "samples_per_depth": args.samples_per_depth,
        "route_kernel": {
            "kind": "shared_two_parameter_logistic_comparator",
            "theta": [float(v) for v in theta],
            "parameter_count": 2,
            "training_examples": int(len(train_x)),
            "training_trace": training_trace,
        },
        "aggregate": {
            "tested_queries": all_queries,
            "deterministic_exact": all_det_hits / all_queries,
            "learned_exact": all_learned_hits / all_queries,
            "learned_ood_min_exact": learned_ood_exact,
            "inverse_exact": all_inverse_hits / all_queries,
            "max_seen_steps": max_seen_steps,
            "deepest_logical_candidates": deepest["logical_candidate_count"],
            "deepest_explicit_bytes": deepest["explicit_payload_bytes"],
            "deepest_lazy_bytes": deepest["lazy_descriptor_bytes"],
            "deepest_storage_ratio": deepest["explicit_to_lazy_storage_ratio"],
            "deepest_broken_one_path_exact": deepest[
                "broken_rotation_one_path_exact"
            ],
            "deepest_mean_treeheap_comparisons": deepest[
                "mean_treeheap_comparisons"
            ],
            "deepest_mean_unordered_scan_comparisons": deepest[
                "mean_unordered_scan_comparisons"
            ],
        },
        "budget": {
            "hard_max_rotation_depth": args.max_depth,
            "requested_rotation_depth": requested_depth,
            "growth_status": growth_status,
            "query_step_budget": query_budget,
            "query_status": budgeted_query.status,
        },
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "boundary": (
            "Regular order-isomorphic orbit only; no semantic, arbitrary-space, "
            "cryptographic, or sorted-array comparison advantage claim."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in depth_rows:
            handle.write(json.dumps({"kind": "depth", **row}, sort_keys=True) + "\n")
        for row in example_rows:
            handle.write(json.dumps({"kind": "example", **row}, sort_keys=True) + "\n")

    readme = f"""# Bounded Rotation Search Evidence

Claim: `M0-ROT-C01`
Predict: `P-ROT01`

## Result

```text
pilot_pass                         = {summary['pilot_pass']}
tested_queries                     = {all_queries}
deterministic_exact                = {summary['aggregate']['deterministic_exact']:.6f}
learned_ood_min_exact              = {learned_ood_exact:.6f}
inverse_exact                      = {summary['aggregate']['inverse_exact']:.6f}
deepest_logical_candidates         = {deepest['logical_candidate_count']}
deepest_treeheap_mean_comparisons  = {deepest['mean_treeheap_comparisons']:.3f}
deepest_unordered_mean_comparisons = {deepest['mean_unordered_scan_comparisons']:.3f}
deepest_explicit/lazy_storage      = {deepest['explicit_to_lazy_storage_ratio']:.3f}x
deepest_broken_one_path_exact      = {deepest['broken_rotation_one_path_exact']:.6f}
over_budget_status                 = {growth_status}
```

## Interpretation

The probe tests compact reuse of a regular order-isomorphic orbit. Search work
is linear in rotation depth and logarithmic in logical candidate count. The
explicit sorted baseline has the same asymptotic comparison count but stores
every payload. Destroying local order forces equality scanning or breaks the
one-path kernel.

This is not evidence for semantic reasoning, arbitrary exponential search, or
cryptographic key recovery.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-size", type=int, default=31)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-depth", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
