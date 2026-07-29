#!/usr/bin/env python3
"""C17: online payload displacement by collision-split insertion."""
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
    parser.add_argument("--seed", type=int, default=75037)
    parser.add_argument("--train-per-query", type=int, default=600)
    parser.add_argument("--test-per-query", type=int, default=300)
    parser.add_argument("--orders", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--evidence-dir", default="")
    return parser.parse_args()


def cosine(left, right) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / max(1e-12, left_norm * right_norm)


class PressureTree:
    def __init__(self, counts, max_depth: int, mode: str, seed: int):
        self.counts = counts
        self.max_depth = max_depth
        self.capacity = 2 ** (max_depth + 1) - 1
        self.mode = mode
        self.rng = random.Random(seed)
        self.payload = {}
        self.internal = set()
        self.signature_sum = {}
        self.leaf_count = {}
        self.query_mass = {}
        self.splits = 0
        self.moves = 0

    def value_signature(self, value: str):
        vector = [self.counts[query].get(value, 0) for query in c15.QUERIES]
        total = sum(vector)
        return tuple(component / max(1, total) for component in vector)

    @staticmethod
    def depth(node: int) -> int:
        return (node + 1).bit_length() - 1

    def occupied(self, node: int) -> bool:
        return node in self.payload or node in self.internal

    def insert(self, value: str) -> None:
        if value in self.payload.values():
            raise ValueError(f"duplicate value: {value}")
        if not self.occupied(0):
            self.payload[0] = value
            self.recompute()
            return
        node = 0
        while node not in self.payload:
            left, right = 2 * node + 1, 2 * node + 2
            child_capacity = 2 ** (self.max_depth - self.depth(node) - 1)
            left_full = self.leaf_count[left] >= child_capacity
            right_full = self.leaf_count[right] >= child_capacity
            if left_full and right_full:
                raise RuntimeError("entered a full fixed-capacity subheap")
            if self.mode == "random":
                choices = [child for child, full in ((left, left_full), (right, right_full))
                           if not full]
                node = self.rng.choice(choices)
            else:
                signature = self.value_signature(value)
                left_score = -math.inf if left_full else cosine(signature, self.centroid(left))
                right_score = -math.inf if right_full else cosine(signature, self.centroid(right))
                if abs(left_score - right_score) < 1e-12:
                    node = left if self.leaf_count[left] <= self.leaf_count[right] else right
                else:
                    node = left if left_score > right_score else right
        if self.depth(node) >= self.max_depth:
            raise RuntimeError("fixed TreeHeap depth exhausted")
        old = self.payload.pop(node)
        self.internal.add(node)
        left, right = 2 * node + 1, 2 * node + 2
        if self.mode == "random" and self.rng.random() < 0.5:
            left, right = right, left
        self.payload[left] = old
        self.payload[right] = value
        self.splits += 1
        self.moves += 1
        self.recompute()

    def recompute(self) -> None:
        self.signature_sum.clear()
        self.leaf_count.clear()
        self.query_mass.clear()

        def visit(node: int):
            if node in self.payload:
                value = self.payload[node]
                signature = self.value_signature(value)
                mass = tuple(self.counts[query].get(value, 0) for query in c15.QUERIES)
                self.signature_sum[node] = signature
                self.leaf_count[node] = 1
                self.query_mass[node] = mass
                return signature, 1, mass
            left, right = 2 * node + 1, 2 * node + 2
            if not self.occupied(left) and not self.occupied(right):
                return None
            left_state, right_state = visit(left), visit(right)
            states = [state for state in (left_state, right_state) if state is not None]
            signature = tuple(sum(state[0][i] for state in states)
                              for i in range(len(c15.QUERIES)))
            count = sum(state[1] for state in states)
            mass = tuple(sum(state[2][i] for state in states)
                         for i in range(len(c15.QUERIES)))
            self.signature_sum[node] = signature
            self.leaf_count[node] = count
            self.query_mass[node] = mass
            return signature, count, mass

        if self.payload:
            visit(0)

    def centroid(self, node: int):
        count = self.leaf_count[node]
        return tuple(value / count for value in self.signature_sum[node])

    def mean_payload_depth(self) -> float:
        return mean(self.depth(node) for node in self.payload)

    def search(self, query: str, budget: int, top_k: int):
        query_index = c15.QUERIES.index(query)
        queue = [(-self.query_mass[0][query_index], 0)]
        returned, visited = [], 0
        while queue and visited < budget:
            negative_mass, node = heapq.heappop(queue)
            visited += 1
            if node in self.payload:
                value = self.payload[node]
                returned.append((self.counts[query].get(value, 0), value,
                                 self.depth(node)))
                continue
            left, right = 2 * node + 1, 2 * node + 2
            for child in (left, right):
                if child in self.query_mass:
                    heapq.heappush(queue, (-self.query_mass[child][query_index], child))
        returned.sort(key=lambda row: (-row[0], row[1]))
        return returned[:top_k]

    def validate(self) -> dict:
        active_nodes = len(self.internal) + len(self.payload)
        mass_ok = True
        for node in self.internal:
            left, right = 2 * node + 1, 2 * node + 2
            expected = tuple(self.query_mass[left][i] + self.query_mass[right][i]
                             for i in range(len(c15.QUERIES)))
            mass_ok &= expected == self.query_mass[node]
        return {
            "payloads": len(self.payload),
            "active_nodes": active_nodes,
            "expected_active_nodes": 2 * len(self.payload) - 1,
            "all_payloads_are_leaves": all(
                node not in self.internal for node in self.payload
            ),
            "parent_mass_exact": mass_ok,
            "within_fixed_capacity": max(self.leaf_count, default=0) < self.capacity,
            "splits": self.splits,
            "moves": self.moves,
        }


def hit_at(tree: PressureTree, test, budget: int, top_k: int) -> float:
    hit = total = 0
    for query in c15.QUERIES:
        returned = {value for _, value, _ in tree.search(query, budget, top_k)}
        for value, count in test[query].items():
            total += count
            if value in returned:
                hit += count
    return hit / total


def build(counts, order, mode: str, max_depth: int, seed: int):
    tree = PressureTree(counts, max_depth, mode, seed)
    depth_history = []
    payload_depth_history = []
    root_history = []
    for value in order:
        tree.insert(value)
        depth_history.append(tree.mean_payload_depth())
        payload_depth_history.append({payload: tree.depth(node)
                                      for node, payload in tree.payload.items()})
        root_history.append(tree.payload.get(0, "aggregate"))
    return tree, depth_history, payload_depth_history, root_history


def main() -> None:
    args = parse_args()
    train = c15.sample_counts(args.seed, args.train_per_query)
    test = c15.sample_counts(args.seed + 1, args.test_per_query)
    rng = random.Random(args.seed + 2)
    semantic_curves, random_curves, order_rows = [], [], []
    canonical = None
    for order_index in range(args.orders):
        order = list(c15.VALUES)
        rng.shuffle(order)
        semantic, depths, payload_depths, roots = build(
            train, order, "signature", args.max_depth, args.seed + 10 + order_index,
        )
        random_tree, _, _, _ = build(
            train, order, "random", args.max_depth, args.seed + 100 + order_index,
        )
        semantic_curve = [hit_at(semantic, test, budget, args.top_k)
                          for budget in range(1, args.max_budget + 1)]
        random_curve = [hit_at(random_tree, test, budget, args.top_k)
                        for budget in range(1, args.max_budget + 1)]
        semantic_curves.append(semantic_curve)
        random_curves.append(random_curve)
        no_payload_moves_up = all(
            current.get(value, previous_depth) >= previous_depth
            for previous, current in zip(payload_depths, payload_depths[1:])
            for value, previous_depth in previous.items()
        )
        order_rows.append({
            "order": order,
            "depth_history": depths,
            "first_payload_depth_after_two": payload_depths[1][order[0]],
            "no_payload_moves_up": no_payload_moves_up,
            "root_history": roots,
            "semantic_auc": mean(semantic_curve),
            "random_auc": mean(random_curve),
        })
        canonical = canonical or semantic
    semantic_mean = [mean(curve[index] for curve in semantic_curves)
                     for index in range(args.max_budget)]
    random_mean = [mean(curve[index] for curve in random_curves)
                   for index in range(args.max_budget)]
    full_budget = canonical.capacity
    full_hit = hit_at(canonical, test, full_budget, args.top_k)
    flat_hit = c15.flat_hit(train, test, args.top_k)
    canonical_row = order_rows[0]
    validation = canonical.validate()
    summary = {
        "claim": "S3-PRESSURE-SPLIT-INDEX-C17",
        "status": "smoke_complete",
        "config": vars(args),
        "canonical": {
            **validation,
            "first_root": canonical_row["root_history"][0],
            "second_root": canonical_row["root_history"][1],
            "first_payload_depth_after_two": canonical_row["first_payload_depth_after_two"],
            "no_payload_moves_up": all(row["no_payload_moves_up"] for row in order_rows),
            "depth_history": canonical_row["depth_history"],
            "eat_top3": canonical.search("eat", args.max_budget, args.top_k),
        },
        "budget_curve": {
            "budgets": list(range(1, args.max_budget + 1)),
            "signature_hit_mean": semantic_mean,
            "random_hit_mean": random_mean,
            "signature_auc": mean(semantic_mean),
            "random_auc": mean(random_mean),
            "auc_gain": mean(semantic_mean) - mean(random_mean),
        },
        "full_retrieval": {"tree_hit": full_hit, "flat_hit": flat_hit},
        "orders": order_rows,
    }
    depth_history = summary["canonical"]["depth_history"]
    summary["gates"] = {
        "root_payload_moves": (
            summary["canonical"]["first_root"] != "aggregate"
            and summary["canonical"]["second_root"] == "aggregate"
            and summary["canonical"]["first_payload_depth_after_two"] >= 1
        ),
        "active_node_closure": validation["active_nodes"] == validation["expected_active_nodes"],
        "payloads_only_at_leaves": validation["all_payloads_are_leaves"],
        "parent_mass_exact": validation["parent_mass_exact"],
        "fixed_capacity": validation["within_fixed_capacity"],
        "depth_nondecreasing": all(right + 1e-12 >= left
                                    for left, right in zip(depth_history, depth_history[1:])),
        "individual_payloads_never_move_up": summary["canonical"]["no_payload_moves_up"],
        "full_retrieval_matches_flat": abs(full_hit - flat_hit) < 1e-12,
        "signature_beats_random": summary["budget_curve"]["auc_gain"] > 0,
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
