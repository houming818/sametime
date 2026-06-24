#!/usr/bin/env python3
"""Kernel convolution operator probe.

This probe tests the architectural claim that TreeHeap operations can be
defined as kernel convolutions over a tree-shaped state field.

It is not a language experiment and it is not a learned-kernel proof. It is a
deterministic toy that checks whether search, plus/write, and conjugate mirror
can share the same convolutional definition:

  local patch -> kernel score/update -> full-tree state map.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np


QUERY = np.array([2.0, 5.0, -3.0], dtype=np.float64)
WRITE = np.array([0.75, -0.50, 0.25], dtype=np.float64)


@dataclass
class ConvResult:
    scores: dict[str, float]
    probs: dict[str, float]
    hit_path: str
    target_prob: float
    entropy: float


def all_paths(depth: int) -> list[str]:
    paths = [""]
    frontier = [""]
    for _ in range(depth):
        nxt: list[str] = []
        for p in frontier:
            nxt.extend([p + "L", p + "R"])
        paths.extend(nxt)
        frontier = nxt
    return paths


def candidate_paths(depth: int) -> list[str]:
    return all_paths(depth - 1)


def mirror_path(path: str) -> str:
    return "".join("R" if ch == "L" else "L" for ch in path)


def mirror_patch(patch: np.ndarray) -> np.ndarray:
    return patch[[0, 2, 1]]


def make_heap(rng: np.random.Generator, depth: int, target_path: str) -> dict[str, np.ndarray]:
    tree = {p: np.array([rng.normal(0.0, 0.35)], dtype=np.float64) for p in all_paths(depth)}
    tree[target_path] = np.array([QUERY[0]], dtype=np.float64)
    tree[target_path + "L"] = np.array([QUERY[1]], dtype=np.float64)
    tree[target_path + "R"] = np.array([QUERY[2]], dtype=np.float64)
    return tree


def patch(tree: dict[str, np.ndarray], path: str) -> np.ndarray:
    return np.array(
        [
            float(tree[path][0]),
            float(tree[path + "L"][0]),
            float(tree[path + "R"][0]),
        ],
        dtype=np.float64,
    )


def patch_nodes(path: str) -> set[str]:
    return {path, path + "L", path + "R"}


def patches_overlap(a: str, b: str) -> bool:
    return bool(patch_nodes(a) & patch_nodes(b))


def kernel_score(query: np.ndarray, local_patch: np.ndarray, temperature: float = 1.0) -> float:
    # A fixed convolution kernel: high score means local subheap matches query.
    diff = local_patch - query
    return -float(diff @ diff) / temperature


def softmax(scores: dict[str, float], beta: float) -> dict[str, float]:
    keys = list(scores)
    values = np.array([scores[k] for k in keys], dtype=np.float64) * beta
    values -= values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def entropy(probs: dict[str, float]) -> float:
    ps = np.array(list(probs.values()), dtype=np.float64)
    return float(-(ps * np.log(ps + 1e-12)).sum())


def convolve_search(
    tree: dict[str, np.ndarray],
    depth: int,
    query: np.ndarray,
    *,
    beta: float,
    mode: str = "direct",
) -> ConvResult:
    scores: dict[str, float] = {}
    for p in candidate_paths(depth):
        local = patch(tree, p)
        if mode == "direct":
            q = query
            x = local
        elif mode == "conjugate_unmirror_patch":
            q = query
            x = mirror_patch(local)
        elif mode == "mirrored_query":
            q = mirror_patch(query)
            x = local
        else:
            raise ValueError(f"unknown mode={mode}")
        scores[p] = kernel_score(q, x)
    probs = softmax(scores, beta=beta)
    hit_path = max(scores, key=scores.get)
    return ConvResult(
        scores=scores,
        probs=probs,
        hit_path=hit_path,
        target_prob=0.0,
        entropy=entropy(probs),
    )


def apply_plus_convolution(
    tree: dict[str, np.ndarray],
    depth: int,
    query: np.ndarray,
    write: np.ndarray,
    *,
    beta: float,
) -> tuple[dict[str, np.ndarray], ConvResult, dict[str, float]]:
    conv = convolve_search(tree, depth, query, beta=beta)
    new_tree = {p: v.copy() for p, v in tree.items()}
    before = {p: patch(tree, p) for p in candidate_paths(depth)}
    for p, prob in conv.probs.items():
        new_tree[p] = new_tree[p] + prob * write[0]
        new_tree[p + "L"] = new_tree[p + "L"] + prob * write[1]
        new_tree[p + "R"] = new_tree[p + "R"] + prob * write[2]
    after = {p: patch(new_tree, p) for p in candidate_paths(depth)}
    update_norms = {p: float(np.linalg.norm(after[p] - before[p])) for p in candidate_paths(depth)}
    return new_tree, conv, update_norms


def mirror_tree(tree: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {mirror_path(p): v.copy() for p, v in tree.items()}


def map_equiv_error(original_scores: dict[str, float], mirrored_scores: dict[str, float]) -> float:
    errors = []
    for p, s in original_scores.items():
        errors.append(abs(s - mirrored_scores[mirror_path(p)]))
    return float(max(errors))


def run_probe(out: pathlib.Path, seed: int, depth: int, beta: float) -> dict:
    rng = np.random.default_rng(seed)
    target = "LRL"
    if len(target) >= depth:
        raise ValueError("depth must be greater than target depth")

    tree = make_heap(rng, depth=depth, target_path=target)
    candidates = candidate_paths(depth)

    search = convolve_search(tree, depth, QUERY, beta=beta)
    search.target_prob = search.probs[target]

    new_tree, plus_conv, update_norms = apply_plus_convolution(tree, depth, QUERY, WRITE, beta=beta)
    plus_conv.target_prob = plus_conv.probs[target]
    update_hit = max(update_norms, key=update_norms.get)
    disjoint_update_norms = {
        p: v for p, v in update_norms.items() if p != target and not patches_overlap(p, target)
    }
    target_before = patch(tree, target)
    target_after = patch(new_tree, target)
    target_expected = target_before + WRITE
    target_update_error = float(np.linalg.norm(target_after - target_expected))

    mirrored = mirror_tree(tree)
    mirrored_target = mirror_path(target)
    raw_mirror = convolve_search(mirrored, depth, QUERY, beta=beta, mode="direct")
    conjugate = convolve_search(mirrored, depth, QUERY, beta=beta, mode="conjugate_unmirror_patch")
    conjugate.target_prob = conjugate.probs[mirrored_target]
    equiv_error = map_equiv_error(search.scores, conjugate.scores)

    mirrored_query = convolve_search(mirrored, depth, mirror_patch(QUERY), beta=beta, mode="direct")
    mirrored_query.target_prob = mirrored_query.probs[mirrored_target]

    summary = {
        "seed": seed,
        "depth": depth,
        "beta": beta,
        "target_path": target,
        "mirrored_target_path": mirrored_target,
        "candidate_count": len(candidates),
        "claim": "M0-SOFT-C08",
        "predict": "P-SOFT04",
        "search": {
            "hit_path": search.hit_path,
            "hit_at_1": search.hit_path == target,
            "target_prob": search.target_prob,
            "entropy": search.entropy,
        },
        "plus_convolution": {
            "write_hit_path": update_hit,
            "write_hit_at_1": update_hit == target,
            "target_prob": plus_conv.target_prob,
            "target_update_error": target_update_error,
            "target_update_norm": float(np.linalg.norm(target_after - target_before)),
            "max_nontarget_update_norm": float(
                max(v for p, v in update_norms.items() if p != target)
            ),
            "max_disjoint_nontarget_update_norm": float(max(disjoint_update_norms.values())),
            "n_overlap_nontarget_patches": int(
                sum(1 for p in update_norms if p != target and patches_overlap(p, target))
            ),
        },
        "conjugate_mirror": {
            "raw_mirror_hit_path": raw_mirror.hit_path,
            "raw_mirror_hit_at_1": raw_mirror.hit_path == mirrored_target,
            "mirrored_query_hit_path": mirrored_query.hit_path,
            "mirrored_query_hit_at_1": mirrored_query.hit_path == mirrored_target,
            "conjugate_hit_path": conjugate.hit_path,
            "conjugate_hit_at_1": conjugate.hit_path == mirrored_target,
            "conjugate_target_prob": conjugate.target_prob,
            "score_map_equiv_max_error": equiv_error,
        },
    }
    summary["pilot_pass"] = bool(
        summary["search"]["hit_at_1"]
        and summary["plus_convolution"]["write_hit_at_1"]
        and summary["plus_convolution"]["target_update_error"] < 0.05
        and summary["conjugate_mirror"]["conjugate_hit_at_1"]
        and summary["conjugate_mirror"]["score_map_equiv_max_error"] < 1e-9
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    trace_rows = []
    for p in candidates:
        trace_rows.append(
            {
                "path": p,
                "mirror_path": mirror_path(p),
                "patch": patch(tree, p).tolist(),
                "search_score": search.scores[p],
                "search_prob": search.probs[p],
                "plus_update_norm": update_norms[p],
                "conjugate_score_at_mirror": conjugate.scores[mirror_path(p)],
            }
        )
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    readme = f"""# Kernel Convolution Ops Probe Evidence

Verdict: `pilot_pass = {summary['pilot_pass']}`

## What Was Tested

This deterministic toy tests the claim that TreeHeap operations can be defined
as kernel convolutions over the whole tree state:

```text
search:    kernel scans every local subheap and emits a score map
plus:      the same score map becomes a soft write/update field
conjugate: a mirrored kernel recovers the mirrored score map
```

## Key Metrics

| Check | Result |
|---|---:|
| search hit@1 | {summary['search']['hit_at_1']} |
| plus write hit@1 | {summary['plus_convolution']['write_hit_at_1']} |
| plus target update error | {summary['plus_convolution']['target_update_error']:.6f} |
| max non-target update norm | {summary['plus_convolution']['max_nontarget_update_norm']:.6f} |
| max disjoint non-target update norm | {summary['plus_convolution']['max_disjoint_nontarget_update_norm']:.6f} |
| overlapping non-target patches | {summary['plus_convolution']['n_overlap_nontarget_patches']} |
| raw mirror hit@1 | {summary['conjugate_mirror']['raw_mirror_hit_at_1']} |
| conjugate mirror hit@1 | {summary['conjugate_mirror']['conjugate_hit_at_1']} |
| score-map equiv max error | {summary['conjugate_mirror']['score_map_equiv_max_error']:.6e} |

## Interpretation

The positive result supports `M0-SOFT-C08` as a toy operator-semantics pilot:
search, plus/write, and conjugate mirror can be expressed as TreeHeap kernel
convolutions that produce full-tree maps.

The non-target update norm includes overlapping TreeHeap patches. A parent patch
can observe a child update even when it was not selected as the write target.
For localization, use `max_disjoint_nontarget_update_norm`.

This is not a learned-kernel proof and not language evidence. It is a clean
operator-level proof target for the next learned C05/C06 experiments.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--beta", type=float, default=8.0)
    args = parser.parse_args()
    summary = run_probe(args.out, seed=args.seed, depth=args.depth, beta=args.beta)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
