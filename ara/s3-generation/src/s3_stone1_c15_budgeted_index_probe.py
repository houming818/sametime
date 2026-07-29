#!/usr/bin/env python3
"""C15: test TreeHeap placement learning with a finite retrieval budget."""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import random
from statistics import mean


GROUPS = {
    "food": ("rice", "noodles", "apple", "bread"),
    "medicine": ("ibuprofen", "amoxicillin", "aspirin", "insulin"),
    "vehicle": ("car", "bus", "bicycle", "train"),
    "place": ("beach", "museum", "mountain", "hotel"),
}
QUERY_GROUP = {
    "eat": "food", "cook": "food", "taste": "food",
    "prescribe": "medicine", "dose": "medicine", "treat": "medicine",
    "drive": "vehicle", "ride": "vehicle", "repair": "vehicle",
    "visit": "place", "book": "place", "photograph": "place",
}
VALUES = tuple(value for group in GROUPS.values() for value in group)
QUERIES = tuple(QUERY_GROUP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=75035)
    parser.add_argument("--train-per-query", type=int, default=600)
    parser.add_argument("--test-per-query", type=int, default=300)
    parser.add_argument("--layouts", type=int, default=64)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--swap-rounds", type=int, default=32)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--evidence-dir", default="")
    return parser.parse_args()


def query_weights(query: str) -> list[float]:
    primary = QUERY_GROUP[query]
    query_offset = QUERIES.index(query) % 4
    primary_weights = (0.42, 0.29, 0.18, 0.11)
    result = []
    for value in VALUES:
        group = next(name for name, rows in GROUPS.items() if value in rows)
        if group == primary:
            position = GROUPS[group].index(value)
            result.append(0.91 * primary_weights[(position - query_offset) % 4])
        else:
            result.append(0.09 / (len(VALUES) - len(GROUPS[primary])))
    total = sum(result)
    return [value / total for value in result]


def sample_counts(seed: int, rows_per_query: int) -> dict[str, Counter]:
    rng = random.Random(seed)
    counts = {}
    for query in QUERIES:
        draws = rng.choices(VALUES, weights=query_weights(query), k=rows_per_query)
        counts[query] = Counter(draws)
    return counts


def shuffled_pair(train, test, seed: int):
    rng = random.Random(seed)
    shuffled_train, shuffled_test = {}, {}
    for query in QUERIES:
        labels = list(VALUES)
        rng.shuffle(labels)
        mapping = dict(zip(VALUES, labels))
        shuffled_train[query] = Counter({mapping[value]: count
                                         for value, count in train[query].items()})
        shuffled_test[query] = Counter({mapping[value]: count
                                        for value, count in test[query].items()})
    return shuffled_train, shuffled_test


class CountTree:
    """A complete fixed-capacity TreeHeap with exact query mass at each node."""

    def __init__(self, layout: tuple[str, ...], counts, smoothing: float = 0.0):
        if len(layout) != len(VALUES) or len(layout) & (len(layout) - 1):
            raise ValueError("layout must fill a power-of-two TreeHeap")
        self.layout = layout
        self.width = len(layout)
        self.leaf_start = self.width - 1
        self.nodes = 2 * self.width - 1
        self.mass = {}
        self.position = {value: index for index, value in enumerate(layout)}
        for query in QUERIES:
            row = [0.0] * self.nodes
            for offset, value in enumerate(layout):
                row[self.leaf_start + offset] = counts[query].get(value, 0) + smoothing
            for node in range(self.leaf_start - 1, -1, -1):
                row[node] = row[2 * node + 1] + row[2 * node + 2]
            self.mass[query] = row

    def path_probability(self, query: str, value: str) -> float:
        node = self.leaf_start + self.position[value]
        path = []
        while node:
            path.append(node)
            node = (node - 1) // 2
        probability = 1.0
        parent = 0
        for child in reversed(path):
            probability *= self.mass[query][child] / self.mass[query][parent]
            parent = child
        return probability

    def search(self, query: str, budget: int, top_k: int):
        queue = [(-self.mass[query][0], 0)]
        visited, returned, trace = 0, [], []
        while queue and visited < budget and len(returned) < top_k:
            negative_mass, node = heapq.heappop(queue)
            visited += 1
            trace.append({"node": node, "mass": -negative_mass})
            if node >= self.leaf_start:
                returned.append(self.layout[node - self.leaf_start])
                continue
            left, right = 2 * node + 1, 2 * node + 2
            heapq.heappush(queue, (-self.mass[query][left], left))
            heapq.heappush(queue, (-self.mass[query][right], right))
        return returned, trace


def weighted_nll(index_counts, eval_counts, layout, smoothing: float) -> float:
    tree = CountTree(layout, index_counts, smoothing)
    total_loss = total = 0.0
    for query in QUERIES:
        for value, count in eval_counts[query].items():
            total_loss -= count * math.log(tree.path_probability(query, value))
            total += count
    return total_loss / total


def weighted_hit(index_counts, eval_counts, layout, budget: int, top_k: int) -> float:
    tree = CountTree(layout, index_counts)
    hit = total = 0
    for query in QUERIES:
        returned, _ = tree.search(query, budget, top_k)
        hit += sum(eval_counts[query].get(value, 0) for value in returned)
        total += sum(eval_counts[query].values())
    return hit / total


def flat_hit(index_counts, eval_counts, top_k: int) -> float:
    hit = total = 0
    for query in QUERIES:
        returned = [value for value, _ in index_counts[query].most_common(top_k)]
        hit += sum(eval_counts[query].get(value, 0) for value in returned)
        total += sum(eval_counts[query].values())
    return hit / total


def random_layout(rng: random.Random) -> tuple[str, ...]:
    layout = list(VALUES)
    rng.shuffle(layout)
    return tuple(layout)


def optimize_layout(counts, start: tuple[str, ...], budget: int, top_k: int,
                    rounds: int):
    current = start
    current_score = weighted_hit(counts, counts, current, budget, top_k)
    trace = [{"round": 0, "train_hit": current_score}]
    for round_index in range(1, rounds + 1):
        best_layout, best_score = current, current_score
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                candidate = list(current)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                score = weighted_hit(counts, counts, tuple(candidate), budget, top_k)
                if score > best_score + 1e-12:
                    best_layout, best_score = tuple(candidate), score
        if best_layout == current:
            break
        current, current_score = best_layout, best_score
        trace.append({"round": round_index, "train_hit": current_score})
    return current, current_score, trace


def run_condition(train, test, args, seed: int):
    rng = random.Random(seed)
    layouts = [random_layout(rng) for _ in range(args.layouts)]
    random_train = [weighted_hit(train, train, layout, args.budget, args.top_k)
                    for layout in layouts]
    random_test = [weighted_hit(train, test, layout, args.budget, args.top_k)
                   for layout in layouts]
    start = layouts[max(range(len(layouts)), key=random_train.__getitem__)]
    optimized, train_hit, trace = optimize_layout(
        train, start, args.budget, args.top_k, args.swap_rounds,
    )
    test_hit = weighted_hit(train, test, optimized, args.budget, args.top_k)
    return {
        "random_train_mean": mean(random_train),
        "random_test_mean": mean(random_test),
        "random_test_min": min(random_test),
        "random_test_max": max(random_test),
        "optimized_train_hit": train_hit,
        "optimized_test_hit": test_hit,
        "optimized_advantage": test_hit - mean(random_test),
        "train_test_gap": train_hit - test_hit,
        "layout": optimized,
        "optimization_trace": trace,
    }


def main() -> None:
    args = parse_args()
    if args.budget < 1 or args.top_k < 1:
        raise ValueError("budget and top-k must be positive")
    train = sample_counts(args.seed, args.train_per_query)
    test = sample_counts(args.seed + 1, args.test_per_query)
    rng = random.Random(args.seed + 2)
    layouts = [random_layout(rng) for _ in range(args.layouts)]
    exact_nll = [weighted_nll(train, test, layout, args.smoothing)
                 for layout in layouts]
    structured = run_condition(train, test, args, args.seed + 3)
    shuffled_train, shuffled_test = shuffled_pair(train, test, args.seed + 4)
    shuffled = run_condition(shuffled_train, shuffled_test, args, args.seed + 5)
    tree = CountTree(tuple(structured["layout"]), train)
    eat_values, eat_trace = tree.search("eat", args.budget, args.top_k)
    flat = flat_hit(train, test, args.top_k)
    summary = {
        "claim": "S3-BUDGETED-CONDITIONAL-INDEX-C15",
        "status": "probe_complete",
        "config": vars(args),
        "exact": {
            "nll_min": min(exact_nll),
            "nll_max": max(exact_nll),
            "nll_range": max(exact_nll) - min(exact_nll),
            "flat_count_entries": len(QUERIES) * len(VALUES),
            "tree_count_entries": len(QUERIES) * (2 * len(VALUES) - 1),
        },
        "structured": structured,
        "shuffled": shuffled,
        "advantage_drop_after_shuffle": (
            structured["optimized_advantage"] - shuffled["optimized_advantage"]
        ),
        "flat_exact_topk_hit": flat,
        "eat_trace": {"returned": eat_values, "visited": eat_trace},
    }
    summary["gates"] = {
        "exact_nll_invariant": summary["exact"]["nll_range"] < 1e-12,
        "tree_storage_exceeds_flat": (
            summary["exact"]["tree_count_entries"]
            > summary["exact"]["flat_count_entries"]
        ),
        "optimized_beats_random": structured["optimized_advantage"] >= 0.05,
        "heldout_transfer": abs(structured["train_test_gap"]) < 0.10,
        "shuffle_reduces_advantage": summary["advantage_drop_after_shuffle"] >= 0.02,
        "eat_returns_multiple": len(eat_values) >= 2,
    }
    if args.evidence_dir:
        output = Path(args.evidence_dir)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
