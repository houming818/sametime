#!/usr/bin/env python3
"""Structural C05 probe.

This is a small, deterministic NumPy experiment for the ARA M0 TreeHeap line.

Question:
  Does C05 depend on TreeHeap structure, or can a flat address scorer do the
  same thing?

Toy:
  Generate full binary trees. Inject a local 3-node subheap pattern at exactly
  one address. Candidate scorers must recover the target address.

Train:
  Shallow trees only.

Test:
  Deeper trees with unseen target addresses.

Variants:
  C0 flat_address
  C1 path_only
  C2 subheap_kernel
  C3 path_subheap_kernel
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Callable

import numpy as np


PATTERN = np.array([5.0, -4.0, 3.0], dtype=np.float64)
VARIANTS = ("flat_address", "path_only", "subheap_kernel", "path_subheap_kernel")


@dataclass
class HeapExample:
    depth: int
    target_path: str
    tree: dict[str, float]
    candidates: list[str]


def all_paths(max_depth: int) -> list[str]:
    paths = [""]
    frontier = [""]
    for _ in range(max_depth):
        nxt = []
        for p in frontier:
            nxt.extend([p + "L", p + "R"])
        paths.extend(nxt)
        frontier = nxt
    return paths


def candidate_paths(depth: int) -> list[str]:
    # Candidates must have both children, so they stop one level above leaves.
    return [p for p in all_paths(depth - 1)]


def path_to_id(path: str) -> int:
    # Heap-like array id. root=0, L=1, R=2, LL=3, LR=4, ...
    idx = 0
    for ch in path:
        idx = 2 * idx + (1 if ch == "L" else 2)
    return idx


def make_heap(rng: np.random.Generator, depth: int, target_depth: int) -> HeapExample:
    paths = all_paths(depth)
    tree = {p: float(rng.normal(0.0, 1.0)) for p in paths}
    target_candidates = [p for p in candidate_paths(depth) if len(p) == target_depth]
    target_path = str(rng.choice(target_candidates))

    tree[target_path] = float(PATTERN[0])
    tree[target_path + "L"] = float(PATTERN[1])
    tree[target_path + "R"] = float(PATTERN[2])

    return HeapExample(
        depth=depth,
        target_path=target_path,
        tree=tree,
        candidates=candidate_paths(depth),
    )


def make_dataset(
    rng: np.random.Generator,
    n: int,
    depths: tuple[int, ...],
    target_depth_mode: str,
) -> list[HeapExample]:
    examples = []
    for _ in range(n):
        depth = int(rng.choice(depths))
        if target_depth_mode == "shallow":
            possible = [d for d in range(0, depth) if d <= 2]
        elif target_depth_mode == "deep":
            possible = [depth - 1]
        else:
            raise ValueError(f"unknown target_depth_mode={target_depth_mode}")
        target_depth = int(rng.choice(possible))
        examples.append(make_heap(rng, depth=depth, target_depth=target_depth))
    return examples


def path_features(path: str, max_depth: int) -> np.ndarray:
    # Prefix-aware path features. Padding is zero; L=-1, R=+1.
    bits = np.zeros(max_depth, dtype=np.float64)
    for i, ch in enumerate(path[:max_depth]):
        bits[i] = -1.0 if ch == "L" else 1.0
    depth = len(path) / max_depth
    n_l = path.count("L") / max(1, len(path))
    n_r = path.count("R") / max(1, len(path))
    return np.concatenate([np.array([depth, n_l, n_r], dtype=np.float64), bits])


def subheap_features(tree: dict[str, float], path: str) -> np.ndarray:
    patch = np.array([tree[path], tree[path + "L"], tree[path + "R"]], dtype=np.float64)
    diff = patch - PATTERN
    return np.concatenate(
        [
            patch,
            diff,
            np.abs(diff),
            diff * diff,
            np.array([float(np.linalg.norm(diff))], dtype=np.float64),
        ]
    )


def feature_fn(name: str, train_address_ids: dict[int, int], max_depth: int) -> Callable:
    def flat_address(ex: HeapExample, path: str) -> np.ndarray:
        v = np.zeros(len(train_address_ids), dtype=np.float64)
        slot = train_address_ids.get(path_to_id(path))
        if slot is not None:
            v[slot] = 1.0
        return v

    def path_only(ex: HeapExample, path: str) -> np.ndarray:
        return path_features(path, max_depth=max_depth)

    def subheap_kernel(ex: HeapExample, path: str) -> np.ndarray:
        return subheap_features(ex.tree, path)

    def path_subheap_kernel(ex: HeapExample, path: str) -> np.ndarray:
        return np.concatenate([path_features(path, max_depth=max_depth), subheap_features(ex.tree, path)])

    fns = {
        "flat_address": flat_address,
        "path_only": path_only,
        "subheap_kernel": subheap_kernel,
        "path_subheap_kernel": path_subheap_kernel,
    }
    return fns[name]


def flatten_examples(examples: list[HeapExample], f: Callable) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for ex in examples:
        for path in ex.candidates:
            xs.append(f(ex, path))
            ys.append(1.0 if path == ex.target_path else 0.0)
    return np.vstack(xs).astype(np.float64), np.array(ys, dtype=np.float64)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma, {"mean": mu.tolist(), "std": sigma.tolist()}


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, list[dict]]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    weights = np.where(y > 0.5, neg / pos, 1.0)
    trace = []

    for epoch in range(epochs + 1):
        p = sigmoid(x @ w)
        err = (p - y) * weights
        grad = (x.T @ err) / len(y) + l2 * w
        if epoch > 0:
            w -= lr * grad
        if epoch in {0, 1, 5, 25, 100, 250, 500, epochs}:
            eps = 1e-9
            loss = -np.mean(weights * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            trace.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "grad_norm": float(np.linalg.norm(grad)),
                    "mean_prob_pos": float(p[y > 0.5].mean()),
                    "mean_prob_neg": float(p[y <= 0.5].mean()),
                }
            )
    return w, trace


def evaluate_examples(examples: list[HeapExample], f: Callable, norm: dict, w: np.ndarray) -> dict:
    ranks = []
    top1 = 0
    hit3 = 0
    rows = []
    for ex_i, ex in enumerate(examples):
        x = np.vstack([f(ex, p) for p in ex.candidates])
        mu = np.array(norm["mean"], dtype=np.float64)
        sigma = np.array(norm["std"], dtype=np.float64)
        x = add_bias((x - mu) / sigma)
        scores = x @ w
        order = np.argsort(-scores)
        ranked_paths = [ex.candidates[i] for i in order]
        rank = ranked_paths.index(ex.target_path) + 1
        pred = ranked_paths[0]
        ranks.append(rank)
        top1 += int(rank == 1)
        hit3 += int(rank <= 3)
        if ex_i < 8:
            rows.append(
                {
                    "depth": ex.depth,
                    "target_path": ex.target_path or "ROOT",
                    "pred_path": pred or "ROOT",
                    "rank": int(rank),
                    "target_score": float(scores[ex.candidates.index(ex.target_path)]),
                    "pred_score": float(scores[order[0]]),
                    "candidate_count": len(ex.candidates),
                }
            )

    n = len(examples)
    return {
        "accuracy": top1 / n,
        "hit_at_3": hit3 / n,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "route_executable_rate": 1.0,
        "examples": rows,
    }


def run_probe(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    train = make_dataset(rng, args.train_examples, depths=(2, 3), target_depth_mode="shallow")
    test = make_dataset(rng, args.test_examples, depths=(5, 6), target_depth_mode="deep")

    train_ids = sorted({path_to_id(p) for ex in train for p in ex.candidates})
    train_address_ids = {addr: i for i, addr in enumerate(train_ids)}
    max_depth = 6

    results = {}
    traces = {}
    for name in VARIANTS:
        f = feature_fn(name, train_address_ids=train_address_ids, max_depth=max_depth)
        x_train, y_train = flatten_examples(train, f)
        x_test, y_test = flatten_examples(test, f)
        x_train_s, x_test_s, norm = standardize_train_test(x_train, x_test)
        del x_test_s, y_test  # Evaluation is grouped by heap, so it reuses norm directly.
        x_train_s = add_bias(x_train_s)
        w, trace = train_logistic(x_train_s, y_train, epochs=args.epochs, lr=args.lr, l2=args.l2)
        train_eval = evaluate_examples(train, f, norm, w)
        test_eval = evaluate_examples(test, f, norm, w)
        results[name] = {
            "feature_dim": int(x_train.shape[1]),
            "train": {k: v for k, v in train_eval.items() if k != "examples"},
            "test": {k: v for k, v in test_eval.items() if k != "examples"},
            "test_examples": test_eval["examples"],
            "trace": trace,
        }
        traces[name] = trace

    c0 = results["flat_address"]["test"]["accuracy"]
    c1 = results["path_only"]["test"]["accuracy"]
    c2 = results["subheap_kernel"]["test"]["accuracy"]
    c3 = results["path_subheap_kernel"]["test"]["accuracy"]

    pilot_pass = bool(c2 >= args.pass_accuracy and c3 >= args.pass_accuracy and c0 <= args.fail_ceiling and c1 <= args.fail_ceiling)

    return {
        "experiment": "structural_c05_probe",
        "seed": args.seed,
        "train_examples": args.train_examples,
        "test_examples": args.test_examples,
        "train_depths": [2, 3],
        "test_depths": [5, 6],
        "pattern": PATTERN.tolist(),
        "variants": list(VARIANTS),
        "pilot_pass": pilot_pass,
        "pass_rule": {
            "subheap_kernel_accuracy_min": args.pass_accuracy,
            "path_subheap_kernel_accuracy_min": args.pass_accuracy,
            "flat_address_accuracy_max": args.fail_ceiling,
            "path_only_accuracy_max": args.fail_ceiling,
        },
        "results": results,
        "interpretation": (
            "This is a structural toy proof. It tests whether subheap features carry "
            "unseen-depth relocation signal beyond flat address or path-only baselines. "
            "It does not prove language understanding or WMT performance."
        ),
    }


def write_outputs(summary: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for name, result in summary["results"].items():
            f.write(json.dumps({"variant": name, "trace": result["trace"]}, ensure_ascii=False) + "\n")
            for row in result["test_examples"]:
                f.write(json.dumps({"variant": name, "test_example": row}, ensure_ascii=False) + "\n")

    lines = [
        "# Structural C05 Probe Evidence",
        "",
        f"Verdict: `pilot_pass = {summary['pilot_pass']}`",
        "",
        "## What Was Tested",
        "",
        "A local 3-node subheap pattern was injected into synthetic TreeHeaps.",
        "Models trained on shallow trees and were tested on deeper, unseen target addresses.",
        "",
        "## Result Table",
        "",
        "| Variant | Train acc | Test acc | Hit@3 | Mean rank | Feature dim |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        r = summary["results"][name]
        lines.append(
            "| {name} | {tr:.3f} | {te:.3f} | {h3:.3f} | {mr:.2f} | {fd} |".format(
                name=name,
                tr=r["train"]["accuracy"],
                te=r["test"]["accuracy"],
                h3=r["test"]["hit_at_3"],
                mr=r["test"]["mean_rank"],
                fd=r["feature_dim"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `subheap_kernel` and `path_subheap_kernel` succeed while `flat_address`",
            "and `path_only` fail, the relocation signal is carried by local subheap",
            "structure rather than absolute memory slots.",
            "",
            "This supports `M0-SOFT-C07` as a structural pilot. It does not by itself",
            "upgrade `M0-SOFT-C05`, because C05 still needs the full write-mechanism",
            "ablation against naive memory write and generic encoder soft plus.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/m0-treeheap-math/evidence/structural_c05_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-examples", type=int, default=900)
    parser.add_argument("--test-examples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--pass-accuracy", type=float, default=0.95)
    parser.add_argument("--fail-ceiling", type=float, default=0.25)
    args = parser.parse_args()

    summary = run_probe(args)
    write_outputs(summary, pathlib.Path(args.out))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
