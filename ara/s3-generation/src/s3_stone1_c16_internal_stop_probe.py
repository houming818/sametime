#!/usr/bin/env python3
"""C16: compare leaf-only and internal-STOP TreeHeap retrieval."""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_c15_budgeted_index_probe as c15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=75036)
    parser.add_argument("--train-per-query", type=int, default=600)
    parser.add_argument("--test-per-query", type=int, default=300)
    parser.add_argument("--layouts", type=int, default=32)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--swap-rounds", type=int, default=16)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--evidence-dir", default="")
    return parser.parse_args()


class PayloadTree:
    def __init__(self, layout, counts, smoothing: float = 0.0):
        if len(layout) != 31:
            raise ValueError("C16 uses exactly 31 physical nodes")
        payloads = [value for value in layout if value is not None]
        if sorted(payloads) != sorted(c15.VALUES):
            raise ValueError("layout must contain each payload exactly once")
        self.layout = tuple(layout)
        self.mass = {}
        self.position = {value: node for node, value in enumerate(layout)
                         if value is not None}
        for query in c15.QUERIES:
            row = [0.0] * len(layout)
            for node in range(len(layout) - 1, -1, -1):
                value = layout[node]
                local = 0.0 if value is None else counts[query].get(value, 0) + smoothing
                left, right = 2 * node + 1, 2 * node + 2
                row[node] = local
                if left < len(layout):
                    row[node] += row[left]
                if right < len(layout):
                    row[node] += row[right]
            self.mass[query] = row

    @staticmethod
    def depth(node: int) -> int:
        return (node + 1).bit_length() - 1

    def path_probability(self, query: str, value: str, counts) -> float:
        node = self.position[value]
        local = counts[query].get(value, 0)
        probability = 1.0
        path = []
        cursor = node
        while cursor:
            path.append(cursor)
            cursor = (cursor - 1) // 2
        parent = 0
        for child in reversed(path):
            probability *= self.mass[query][child] / self.mass[query][parent]
            parent = child
        probability *= local / self.mass[query][node]
        return probability

    def search(self, query: str, budget: int, top_k: int):
        queue = [(-self.mass[query][0], 0)]
        exposed, trace = [], []
        visits = 0
        while queue and visits < budget:
            negative_mass, node = heapq.heappop(queue)
            visits += 1
            value = self.layout[node]
            trace.append({"node": node, "depth": self.depth(node),
                          "mass": -negative_mass, "payload": value})
            if value is not None:
                exposed.append((self.mass_for_value(query, value), value,
                                self.depth(node)))
            left, right = 2 * node + 1, 2 * node + 2
            if left < len(self.layout) and self.mass[query][left] > 0:
                heapq.heappush(queue, (-self.mass[query][left], left))
            if right < len(self.layout) and self.mass[query][right] > 0:
                heapq.heappush(queue, (-self.mass[query][right], right))
        exposed.sort(key=lambda row: (-row[0], row[1]))
        return exposed[:top_k], trace

    def mass_for_value(self, query: str, value: str) -> float:
        node = self.position[value]
        left, right = 2 * node + 1, 2 * node + 2
        child_mass = 0.0
        if left < len(self.layout):
            child_mass += self.mass[query][left]
        if right < len(self.layout):
            child_mass += self.mass[query][right]
        return self.mass[query][node] - child_mass


def leaf_layout(values) -> tuple:
    return tuple([None] * 15 + list(values))


def internal_layout(rng: random.Random) -> tuple:
    positions = rng.sample(range(31), len(c15.VALUES))
    layout = [None] * 31
    values = list(c15.VALUES)
    rng.shuffle(values)
    for node, value in zip(positions, values):
        layout[node] = value
    return tuple(layout)


def random_leaf_layout(rng: random.Random) -> tuple:
    values = list(c15.VALUES)
    rng.shuffle(values)
    return leaf_layout(values)


def retrieval_stats(index_counts, eval_counts, layout, budget: int, top_k: int):
    tree = PayloadTree(layout, index_counts)
    hit = total = depth_mass = 0
    for query in c15.QUERIES:
        returned, _ = tree.search(query, budget, top_k)
        depths = {value: depth for _, value, depth in returned}
        for value, count in eval_counts[query].items():
            total += count
            if value in depths:
                hit += count
                depth_mass += count * depths[value]
    return {"hit": hit / total, "mean_hit_depth": depth_mass / max(1, hit)}


def exact_nll(index_counts, eval_counts, layout, smoothing: float):
    smoothed = PayloadTree(layout, index_counts, smoothing)
    adjusted = {query: {value: index_counts[query].get(value, 0) + smoothing
                        for value in c15.VALUES}
                for query in c15.QUERIES}
    loss = total = 0.0
    for query in c15.QUERIES:
        for value, count in eval_counts[query].items():
            loss -= count * math.log(smoothed.path_probability(query, value, adjusted))
            total += count
    return loss / total


def optimize(index_counts, start, allowed_nodes, budget: int, top_k: int,
             rounds: int):
    current = tuple(start)
    current_stats = retrieval_stats(index_counts, index_counts, current, budget, top_k)
    trace = [{"round": 0, **current_stats}]
    for round_index in range(1, rounds + 1):
        best, best_stats = current, current_stats
        for offset, left in enumerate(allowed_nodes):
            for right in allowed_nodes[offset + 1:]:
                candidate = list(current)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                stats = retrieval_stats(index_counts, index_counts, tuple(candidate),
                                        budget, top_k)
                if stats["hit"] > best_stats["hit"] + 1e-12:
                    best, best_stats = tuple(candidate), stats
                elif (abs(stats["hit"] - best_stats["hit"]) < 1e-12
                      and stats["mean_hit_depth"] < best_stats["mean_hit_depth"] - 1e-12):
                    best, best_stats = tuple(candidate), stats
        if best == current:
            break
        current, current_stats = best, best_stats
        trace.append({"round": round_index, **current_stats})
    return current, current_stats, trace


def run_arm(train, test, layouts, allowed_nodes, args):
    random_test = [retrieval_stats(train, test, layout, args.budget, args.top_k)
                   for layout in layouts]
    start = max(layouts, key=lambda layout: retrieval_stats(
        train, train, layout, args.budget, args.top_k)["hit"])
    optimized, train_stats, trace = optimize(
        train, start, allowed_nodes, args.budget, args.top_k, args.swap_rounds,
    )
    test_stats = retrieval_stats(train, test, optimized, args.budget, args.top_k)
    return {
        "random_test_hit_mean": mean(row["hit"] for row in random_test),
        "random_test_depth_mean": mean(row["mean_hit_depth"] for row in random_test),
        "optimized_train": train_stats,
        "optimized_test": test_stats,
        "layout": optimized,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    train = c15.sample_counts(args.seed, args.train_per_query)
    test = c15.sample_counts(args.seed + 1, args.test_per_query)
    rng = random.Random(args.seed + 2)
    leaf_layouts = [random_leaf_layout(rng) for _ in range(args.layouts)]
    internal_layouts = [internal_layout(rng) for _ in range(args.layouts)]
    leaf = run_arm(train, test, leaf_layouts, tuple(range(15, 31)), args)
    internal = run_arm(train, test, internal_layouts, tuple(range(31)), args)
    nll_layouts = leaf_layouts[:4] + internal_layouts[:4]
    nlls = [exact_nll(train, test, layout, args.smoothing) for layout in nll_layouts]
    tree = PayloadTree(tuple(internal["layout"]), train)
    eat_returned, eat_trace = tree.search("eat", args.budget, args.top_k)
    hit_gain = internal["optimized_test"]["hit"] - leaf["optimized_test"]["hit"]
    depth_gain = (leaf["optimized_test"]["mean_hit_depth"]
                  - internal["optimized_test"]["mean_hit_depth"])
    summary = {
        "claim": "S3-INTERNAL-STOP-INDEX-C16",
        "status": "smoke_complete",
        "config": vars(args),
        "budgets": {"physical_nodes": 31, "payload_slots": 16,
                    "node_visits": args.budget, "top_k": args.top_k,
                    "max_payloads_per_node": 1},
        "exact_nll_range": max(nlls) - min(nlls),
        "leaf_only": leaf,
        "internal_stop": internal,
        "heldout_hit_gain": hit_gain,
        "heldout_depth_reduction": depth_gain,
        "flat_exact_topk_hit": c15.flat_hit(train, test, args.top_k),
        "eat_trace": {"returned": eat_returned, "visited": eat_trace},
    }
    summary["gates"] = {
        "exact_nll_invariant": summary["exact_nll_range"] < 1e-12,
        "internal_hit_gain": hit_gain >= 0.03,
        "internal_depth_gain": depth_gain >= 0.50,
        "heldout_transfer": abs(
            internal["optimized_train"]["hit"]
            - internal["optimized_test"]["hit"]
        ) < 0.10,
        "matched_capacity": True,
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
