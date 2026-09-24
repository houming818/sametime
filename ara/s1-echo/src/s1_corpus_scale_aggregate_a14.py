"""Aggregate fixed-test corpus-scale TreeHeap embedding runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch


EPS = 1e-12


def parse_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def co_cluster_f1(a: Sequence[int], b: Sequence[int]) -> float:
    tp = fp = fn = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            same_a = a[i] == a[j]
            same_b = b[i] == b[j]
            tp += int(same_a and same_b)
            fp += int((not same_a) and same_b)
            fn += int(same_a and (not same_b))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, EPS)


def topk_neighbor_jaccard(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a = a / a.norm(dim=1, keepdim=True).clamp_min(EPS)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(EPS)
    sim_a = a @ a.t()
    sim_b = b @ b.t()
    sim_a.fill_diagonal_(-float("inf"))
    sim_b.fill_diagonal_(-float("inf"))
    near_a = sim_a.topk(k, dim=1).indices
    near_b = sim_b.topk(k, dim=1).indices
    values = []
    for row in range(a.shape[0]):
        left = set(near_a[row].tolist())
        right = set(near_b[row].tolist())
        values.append(len(left & right) / len(left | right))
    return sum(values) / len(values)


def frequency_r2(counts: torch.Tensor, assignment: Sequence[int]) -> float:
    y = counts.sum(dim=1).clamp_min(1.0).log()
    labels = torch.tensor(assignment, dtype=torch.long)
    prediction = torch.zeros_like(y)
    for label in torch.unique(labels):
        members = labels == label
        prediction[members] = y[members].mean()
    total = ((y - y.mean()) ** 2).sum()
    residual = ((y - prediction) ** 2).sum()
    return float((1.0 - residual / total.clamp_min(EPS)).item())


def context_coordinates(counts: torch.Tensor, fit_ratio: float, split_seed: int, alpha: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    fit = torch.binomial(counts, torch.full_like(counts, fit_ratio), generator=generator)
    probability = (fit + alpha) / (fit.sum(dim=1, keepdim=True) + alpha * fit.shape[1])
    return probability.sqrt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scales", default="50000,100000,200000,500000,1000000")
    parser.add_argument("--depths", default="3,5")
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--fit-ratio", type=float, default=0.8)
    parser.add_argument("--neighbor-k", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = parse_ints(args.scales)
    depths = parse_ints(args.depths)

    rows: List[Dict[str, object]] = []
    assignments: Dict[tuple, List[int]] = {}
    coordinates: Dict[int, torch.Tensor] = {}
    sealed_hashes = set()
    target_ids = None
    context_ids = None
    for scale in scales:
        counts_path = root / "counts" / f"train_{scale:07d}" / "context_counts.pt"
        payload = torch.load(counts_path, map_location="cpu", weights_only=False)
        counts = payload["train"].to(torch.float64)
        coordinates[scale] = context_coordinates(counts, args.fit_ratio, args.split_seed, args.alpha)
        target_ids = target_ids or payload["target_ids"]
        context_ids = context_ids or payload["context_ids"]
        if target_ids != payload["target_ids"] or context_ids != payload["context_ids"]:
            raise RuntimeError("vocabulary identity changed across scale arms")
        count_manifest = json.loads((counts_path.parent / "manifest.json").read_text(encoding="utf-8"))
        sealed_hashes.add(count_manifest["sealed_test_sha256"])
        for depth in depths:
            run_dir = root / "runs" / f"train_{scale:07d}_depth_{depth}"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            assignment = [int(value) for value in summary["embedding_index"]["token_leaf"]]
            assignments[(scale, depth)] = assignment
            row = {
                "train_lines": scale,
                "depth": depth,
                "leaves": summary["shape"]["leaves"],
                "fit_pairs": summary["counts"]["fit"],
                "dev_pairs": summary["counts"]["search_dev"],
                "sealed_test_pairs": summary["counts"]["sealed_test"],
                "initial_dev_nll": summary["search"]["initial_dev_nll"],
                "best_dev_nll": summary["search"]["best_dev_nll"],
                "dev_gain": summary["search"]["dev_gain"],
                "initial_test_nll": summary["sealed_evaluation"]["initial_test_nll"],
                "best_test_nll": summary["sealed_evaluation"]["best_test_nll"],
                "test_gain": -summary["sealed_evaluation"]["best_minus_initial"],
                "random_test_nll": summary["sealed_evaluation"]["random_test_nll"],
                "leaf_utilization": summary["structure"]["leaf_utilization"],
                "occupancy_entropy": summary["structure"]["occupancy_entropy"],
                "frequency_r2_by_leaf": frequency_r2(counts, assignment),
                "mechanical_gates_pass": all(summary["gates"].values()),
                "assignment_sha256": summary["embedding_index"]["assignment_sha256"],
            }
            rows.append(row)
    if len(sealed_hashes) != 1:
        raise RuntimeError(f"sealed test identity changed: {sorted(sealed_hashes)}")

    adjacent = []
    for previous, current in zip(scales, scales[1:]):
        neighbor = topk_neighbor_jaccard(coordinates[previous], coordinates[current], args.neighbor_k)
        for depth in depths:
            before = next(row for row in rows if row["train_lines"] == previous and row["depth"] == depth)
            after = next(row for row in rows if row["train_lines"] == current and row["depth"] == depth)
            adjacent.append({
                "from_train_lines": previous,
                "to_train_lines": current,
                "depth": depth,
                "test_nll_delta": after["best_test_nll"] - before["best_test_nll"],
                "test_gain_delta": after["test_gain"] - before["test_gain"],
                "partition_co_cluster_f1": co_cluster_f1(assignments[(previous, depth)], assignments[(current, depth)]),
                "top10_context_neighbor_jaccard": neighbor,
                "frequency_r2_delta": after["frequency_r2_by_leaf"] - before["frequency_r2_by_leaf"],
            })

    by_depth = {}
    for depth in depths:
        depth_rows = [row for row in rows if row["depth"] == depth]
        best = min(depth_rows, key=lambda row: row["best_test_nll"])
        nll_values = [float(row["best_test_nll"]) for row in depth_rows]
        by_depth[str(depth)] = {
            "best_scale": best["train_lines"],
            "best_test_nll": best["best_test_nll"],
            "strictly_nonincreasing_test_nll": all(b <= a for a, b in zip(nll_values, nll_values[1:])),
            "endpoint_test_nll_delta": nll_values[-1] - nll_values[0],
        }

    report = {
        "claim": "S1-F-SCALE-A14-C01",
        "boundary": "Fixed-vocabulary context-field embedding scale audit; not generation, translation, READ, or Decoder quality.",
        "contract": {
            "scales": scales,
            "depths": depths,
            "split_seed": args.split_seed,
            "alpha": args.alpha,
            "fit_ratio": args.fit_ratio,
            "neighbor_k": args.neighbor_k,
            "sealed_test_sha256": next(iter(sealed_hashes)),
        },
        "rows": rows,
        "adjacent_scale": adjacent,
        "by_depth": by_depth,
        "mechanical_contract_pass": all(bool(row["mechanical_gates_pass"]) for row in rows),
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "scale_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
