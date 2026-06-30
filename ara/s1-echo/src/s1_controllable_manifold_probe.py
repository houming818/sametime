#!/usr/bin/env python3
"""SPR-037 controllable fold manifold probe.

This is a small CPU-only toy proof.

Question:
    If TreeHeap fold is controlled by convolution-kernel knobs, can the output
    move from noisy blocks toward product-like blocks in a measurable way?

The probe does not claim language understanding. It only tests whether a weak
relation field plus an order prior can form a controllable surface over fold
quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


@dataclass(frozen=True)
class SentenceCase:
    name: str
    tokens: tuple[str, ...]
    gold_spans: tuple[tuple[int, int], ...]


CASES: tuple[SentenceCase, ...] = (
    SentenceCase(
        name="cat_running_for_car",
        tokens=("the", "cat", "is", "running", "for", "a", "car"),
        gold_spans=((0, 2), (2, 4), (5, 7), (4, 7), (1, 7)),
    ),
    SentenceCase(
        name="dog_chasing_ball",
        tokens=("a", "dog", "is", "chasing", "the", "ball"),
        gold_spans=((0, 2), (2, 4), (4, 6), (1, 6)),
    ),
    SentenceCase(
        name="relative_clause_book",
        tokens=("the", "book", "that", "i", "bought", "yesterday", "is", "expensive"),
        gold_spans=((0, 2), (3, 6), (2, 6), (0, 6), (6, 8)),
    ),
    SentenceCase(
        name="kids_playing_park",
        tokens=("some", "kids", "are", "playing", "in", "the", "park"),
        gold_spans=((0, 2), (2, 4), (5, 7), (4, 7), (1, 7)),
    ),
)


def span_set(span: tuple[int, int]) -> frozenset[int]:
    start, end = span
    return frozenset(range(start, end))


def gold_sets(case: SentenceCase) -> set[frozenset[int]]:
    full = frozenset(range(len(case.tokens)))
    return {span_set(span) for span in case.gold_spans if len(span_set(span)) > 1 and span_set(span) != full}


def build_relation_field(case: SentenceCase) -> list[list[float]]:
    """Build a weak latent relation field from toy block annotations.

    Smaller blocks receive stronger pair attraction. This imitates an observed
    relation layout, not a grammar parser.
    """

    n = len(case.tokens)
    field = [[0.0 for _ in range(n)] for _ in range(n)]
    for start, end in case.gold_spans:
        width = end - start
        if width <= 1:
            continue
        weight = 1.0 / math.sqrt(width)
        for i in range(start, end):
            for j in range(i + 1, end):
                field[i][j] += weight
                field[j][i] += weight
    max_value = max(max(row) for row in field) or 1.0
    return [[value / max_value for value in row] for row in field]


def build_order_field(n: int) -> list[list[float]]:
    """Linear-neighborhood prior."""

    return [[0.0 if i == j else math.exp(-abs(i - j) / 2.0) for j in range(n)] for i in range(n)]


def cross_mean(field: list[list[float]], left: frozenset[int], right: frozenset[int]) -> float:
    values = [field[i][j] for i in left for j in right]
    return sum(values) / len(values)


def fold_once(
    case: SentenceCase,
    relation_weight: float,
    order_weight: float,
    noise_scale: float,
    rng: random.Random,
) -> tuple[list[frozenset[int]], list[dict[str, object]]]:
    n = len(case.tokens)
    relation = build_relation_field(case)
    order = build_order_field(n)
    clusters: list[frozenset[int]] = [frozenset([i]) for i in range(n)]
    merges: list[frozenset[int]] = []
    trace: list[dict[str, object]] = []

    while len(clusters) > 1:
        best = None
        best_score = -1e9
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                left, right = clusters[a], clusters[b]
                relation_score = cross_mean(relation, left, right)
                order_score = cross_mean(order, left, right)
                balance_penalty = abs(len(left) - len(right)) / max(1, len(left) + len(right))
                noise = rng.gauss(0.0, noise_scale)
                score = (
                    relation_weight * relation_score
                    + order_weight * order_score
                    - 0.15 * balance_penalty
                    + noise
                )
                if score > best_score:
                    best_score = score
                    best = (a, b, relation_score, order_score, balance_penalty, noise)

        assert best is not None
        a, b, relation_score, order_score, balance_penalty, noise = best
        merged = clusters[a] | clusters[b]
        merges.append(merged)
        trace.append(
            {
                "merge": sorted(merged),
                "merge_tokens": [case.tokens[i] for i in sorted(merged)],
                "relation_score": relation_score,
                "order_score": order_score,
                "balance_penalty": balance_penalty,
                "noise": noise,
                "score": best_score,
            }
        )
        next_clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in (a, b)]
        next_clusters.append(merged)
        clusters = next_clusters

    return merges, trace


def f1(predicted: Iterable[frozenset[int]], gold: set[frozenset[int]], n: int) -> float:
    full = frozenset(range(n))
    pred = {item for item in predicted if len(item) > 1 and item != full}
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = len(pred & gold)
    precision = hit / len(pred)
    recall = hit / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    weights = [float(x) for x in args.weights.split(",")]
    cells = []
    best_trace = None
    best_cell = None
    diagonal = []

    for relation_weight in weights:
        for order_weight in weights:
            scores = []
            exact = []
            for seed in range(args.seeds):
                rng = random.Random(args.seed + seed)
                per_case = []
                for case in CASES:
                    merges, trace = fold_once(case, relation_weight, order_weight, args.noise, rng)
                    score = f1(merges, gold_sets(case), len(case.tokens))
                    per_case.append(score)
                    if best_trace is None or score > best_trace["f1"]:
                        best_trace = {
                            "case": case.name,
                            "tokens": case.tokens,
                            "relation_weight": relation_weight,
                            "order_weight": order_weight,
                            "f1": score,
                            "trace": trace,
                            "gold_blocks": [
                                [case.tokens[i] for i in range(start, end)] for start, end in case.gold_spans
                            ],
                        }
                scores.append(mean(per_case))
                exact.append(1.0 if all(x >= 0.999 for x in per_case) else 0.0)

            cell = {
                "relation_weight": relation_weight,
                "order_weight": order_weight,
                "mean_f1": mean(scores),
                "std_f1": pstdev(scores),
                "exact_rate": mean(exact),
            }
            cells.append(cell)
            if relation_weight == order_weight:
                diagonal.append(cell)
            if best_cell is None or cell["mean_f1"] > best_cell["mean_f1"]:
                best_cell = cell

    low = next(cell for cell in cells if cell["relation_weight"] == 0.0 and cell["order_weight"] == 0.0)
    high = max(cells, key=lambda c: c["relation_weight"] + c["order_weight"])
    diagonal_sorted = sorted(diagonal, key=lambda c: c["relation_weight"])
    diagonal_gain = diagonal_sorted[-1]["mean_f1"] - diagonal_sorted[0]["mean_f1"]
    product_cells = [cell for cell in cells if cell["mean_f1"] >= args.product_threshold]

    return {
        "config": {
            "seeds": args.seeds,
            "seed": args.seed,
            "weights": weights,
            "noise": args.noise,
            "product_threshold": args.product_threshold,
            "cases": [case.name for case in CASES],
        },
        "claim": "S1-MANIFOLD-C01",
        "metrics": {
            "low_control_mean_f1": low["mean_f1"],
            "max_control_mean_f1": best_cell["mean_f1"],
            "high_sum_control_mean_f1": high["mean_f1"],
            "diagonal_gain": diagonal_gain,
            "product_cell_count": len(product_cells),
            "pilot_pass": best_cell["mean_f1"] >= args.product_threshold and diagonal_gain >= args.min_diagonal_gain,
        },
        "best_cell": best_cell,
        "cells": cells,
        "diagonal": diagonal_sorted,
        "best_trace": best_trace,
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for cell in summary["cells"]:
            fh.write(json.dumps(cell, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"best_trace": summary["best_trace"]}, ensure_ascii=False) + "\n")
    metrics = summary["metrics"]
    best = summary["best_cell"]
    readme = f"""# S1 controllable manifold probe

Claim: `S1-MANIFOLD-C01`

This CPU-only toy proof sweeps two TreeHeap kernel controls:

- `relation_weight`: how strongly the fold kernel trusts the latent relation field.
- `order_weight`: how strongly the fold kernel trusts linear neighborhood/order.

The task uses four short sentence cases and asks whether predicted merge blocks
match weak gold blocks such as `the cat`, `is running`, `a car`, and the larger
predicate block.

## Result

- low-control mean F1: `{metrics["low_control_mean_f1"]:.4f}`
- best mean F1: `{metrics["max_control_mean_f1"]:.4f}`
- high-sum-control mean F1: `{metrics["high_sum_control_mean_f1"]:.4f}`
- diagonal gain: `{metrics["diagonal_gain"]:.4f}`
- product cells: `{metrics["product_cell_count"]}`
- pilot pass: `{metrics["pilot_pass"]}`

Best cell:

```json
{json.dumps(best, ensure_ascii=False, indent=2)}
```

## Boundary

This is not a language-understanding proof and not a WMT proof. It only shows
whether fold quality can be controlled by kernel knobs on a transparent toy
relation field.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_controllable_manifold_probe")
    parser.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--product-threshold", type=float, default=0.62)
    parser.add_argument("--min-diagonal-gain", type=float, default=0.25)
    args = parser.parse_args()
    summary = run_sweep(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
