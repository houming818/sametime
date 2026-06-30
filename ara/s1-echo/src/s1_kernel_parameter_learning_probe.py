#!/usr/bin/env python3
"""SPR-039 parameter TreeHeap kernel learning proof.

SPR-038 updated the activation / memory TreeHeap H while keeping the kernel
fixed. This proof does the opposite: it keeps input heaps fixed per sample and
learns the parameter TreeHeap Theta of a local convolution kernel.

Minimal hidden rule on a 7-node complete binary heap:

    y1 = h1 + h2 + h3
    y2 = h2 + h4 + h5
    y3 = h3 + h6 + h7

The model must learn Theta ~= [1, 1, 1] from scalar loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


Triples = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

CORRECT_TRIPLES: Triples = ((0, 1, 2), (1, 3, 4), (2, 5, 6))
WRONG_TRIPLES: Triples = ((0, 3, 4), (1, 5, 6), (2, 1, 2))
HIDDEN_THETA = [1.0, 1.0, 1.0]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mse(preds: list[list[float]], targets: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for pred, target in zip(preds, targets):
        for p, t in zip(pred, target):
            total += (p - t) ** 2
            count += 1
    return total / count


def make_dataset(n: int, rng: random.Random, low: float, high: float) -> list[list[float]]:
    return [[rng.uniform(low, high) for _ in range(7)] for _ in range(n)]


def conv_targets(heap: list[float], triples: Triples = CORRECT_TRIPLES) -> list[float]:
    return [sum(heap[j] for j in triple) for triple in triples]


def shared_kernel_predict(heap: list[float], theta: list[float], triples: Triples) -> list[float]:
    return [dot(theta, [heap[j] for j in triple]) for triple in triples]


def eval_shared(theta: list[float], heaps: list[list[float]], triples: Triples) -> dict[str, float]:
    preds = [shared_kernel_predict(h, theta, triples) for h in heaps]
    targets = [conv_targets(h) for h in heaps]
    return {"mse": mse(preds, targets)}


def train_shared_kernel(
    heaps: list[list[float]],
    triples: Triples,
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    rng = random.Random(seed)
    theta = [rng.uniform(-0.8, 0.8) for _ in range(3)]
    theta_initial = list(theta)
    targets = [conv_targets(h) for h in heaps]
    trace = []

    for epoch in range(epochs + 1):
        preds = [shared_kernel_predict(h, theta, triples) for h in heaps]
        loss = mse(preds, targets)
        if epoch in {0, 1, 2, 5, 10, 25, 50, 100, 250, 500, epochs}:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "theta": list(theta),
                    "theta_l2_error": math.sqrt(sum((t - 1.0) ** 2 for t in theta)),
                }
            )
        if epoch == epochs:
            break

        grad = [0.0, 0.0, 0.0]
        denom = len(heaps) * 3
        for heap, target in zip(heaps, targets):
            for out_idx, triple in enumerate(triples):
                feats = [heap[j] for j in triple]
                err = dot(theta, feats) - target[out_idx]
                for k in range(3):
                    grad[k] += 2.0 * err * feats[k] / denom
        for k in range(3):
            theta[k] -= lr * grad[k]

    return {
        "theta_initial": theta_initial,
        "theta_final": theta,
        "theta_delta_l2": math.sqrt(sum((a - b) ** 2 for a, b in zip(theta, theta_initial))),
        "theta_l2_error": math.sqrt(sum((t - 1.0) ** 2 for t in theta)),
        "trace": trace,
    }


def train_index_specific_kernel(
    heaps: list[list[float]],
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    """A stronger non-shared baseline: one 3-weight kernel per internal node.

    It still uses correct local triples, but does not share the parameter
    TreeHeap kernel across addresses. It should solve the toy with 9 parameters,
    while the TreeHeap shared kernel solves it with 3.
    """

    rng = random.Random(seed)
    weights = [[rng.uniform(-0.8, 0.8) for _ in range(3)] for _ in range(3)]
    targets = [conv_targets(h) for h in heaps]

    for _ in range(epochs):
        grad = [[0.0, 0.0, 0.0] for _ in range(3)]
        denom = len(heaps) * 3
        for heap, target in zip(heaps, targets):
            for out_idx, triple in enumerate(CORRECT_TRIPLES):
                feats = [heap[j] for j in triple]
                err = dot(weights[out_idx], feats) - target[out_idx]
                for k in range(3):
                    grad[out_idx][k] += 2.0 * err * feats[k] / denom
        for out_idx in range(3):
            for k in range(3):
                weights[out_idx][k] -= lr * grad[out_idx][k]

    def predict(heap: list[float]) -> list[float]:
        return [
            dot(weights[out_idx], [heap[j] for j in triple])
            for out_idx, triple in enumerate(CORRECT_TRIPLES)
        ]

    return {
        "parameters": 9,
        "weights_final": weights,
        "train_mse": mse([predict(h) for h in heaps], targets),
        "predict": predict,
    }


def train_global_flat_baseline(
    heaps: list[list[float]],
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    """A matched-size flat baseline with no subheap address structure.

    Features per node are [current_node_value, global_heap_mean, bias].
    It has 3 shared parameters like the TreeHeap kernel but cannot inspect the
    correct left/right subheap.
    """

    rng = random.Random(seed)
    theta = [rng.uniform(-0.8, 0.8) for _ in range(3)]
    targets = [conv_targets(h) for h in heaps]

    def feats(heap: list[float], out_idx: int) -> list[float]:
        root_idx = CORRECT_TRIPLES[out_idx][0]
        return [heap[root_idx], mean(heap), 1.0]

    def predict(heap: list[float]) -> list[float]:
        return [dot(theta, feats(heap, i)) for i in range(3)]

    for _ in range(epochs):
        grad = [0.0, 0.0, 0.0]
        denom = len(heaps) * 3
        for heap, target in zip(heaps, targets):
            for out_idx in range(3):
                f = feats(heap, out_idx)
                err = dot(theta, f) - target[out_idx]
                for k in range(3):
                    grad[k] += 2.0 * err * f[k] / denom
        for k in range(3):
            theta[k] -= lr * grad[k]

    return {"parameters": 3, "theta_final": theta, "train_mse": mse([predict(h) for h in heaps], targets), "predict": predict}


def run(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    train = make_dataset(args.train, rng, -3.0, 3.0)
    test = make_dataset(args.test, rng, -3.0, 3.0)
    ood = make_dataset(args.ood, rng, -10.0, 10.0)

    learned = train_shared_kernel(train, CORRECT_TRIPLES, args.seed + 1, args.epochs, args.lr)
    theta = learned["theta_final"]
    no_learning_theta = learned["theta_initial"]

    wrong = train_shared_kernel(train, WRONG_TRIPLES, args.seed + 2, args.epochs, args.lr)
    wrong_theta = wrong["theta_final"]

    index_baseline = train_index_specific_kernel(train, args.seed + 3, args.epochs, args.lr)
    global_baseline = train_global_flat_baseline(train, args.seed + 4, args.epochs, args.lr)

    def eval_predictor(predict, heaps: list[list[float]]) -> dict[str, float]:
        return {"mse": mse([predict(h) for h in heaps], [conv_targets(h) for h in heaps])}

    no_learning = {
        "parameters": 3,
        "theta": no_learning_theta,
        "train": eval_shared(no_learning_theta, train, CORRECT_TRIPLES),
        "test": eval_shared(no_learning_theta, test, CORRECT_TRIPLES),
        "ood": eval_shared(no_learning_theta, ood, CORRECT_TRIPLES),
    }

    treeheap = {
        "parameters": 3,
        "theta_initial": learned["theta_initial"],
        "theta_final": theta,
        "theta_delta_l2": learned["theta_delta_l2"],
        "theta_l2_error": learned["theta_l2_error"],
        "train": eval_shared(theta, train, CORRECT_TRIPLES),
        "test": eval_shared(theta, test, CORRECT_TRIPLES),
        "ood": eval_shared(theta, ood, CORRECT_TRIPLES),
        "trace": learned["trace"],
    }

    wrong_address = {
        "parameters": 3,
        "theta_final": wrong_theta,
        "theta_l2_error_to_hidden": wrong["theta_l2_error"],
        "train": eval_shared(wrong_theta, train, WRONG_TRIPLES),
        "test": eval_shared(wrong_theta, test, WRONG_TRIPLES),
        "ood": eval_shared(wrong_theta, ood, WRONG_TRIPLES),
    }

    index_specific = {
        "parameters": 9,
        "weights_final": index_baseline["weights_final"],
        "train": {"mse": index_baseline["train_mse"]},
        "test": eval_predictor(index_baseline["predict"], test),
        "ood": eval_predictor(index_baseline["predict"], ood),
        "meaning": "stronger baseline using correct local triples but no shared kernel",
    }

    flat_global = {
        "parameters": 3,
        "theta_final": global_baseline["theta_final"],
        "train": {"mse": global_baseline["train_mse"]},
        "test": eval_predictor(global_baseline["predict"], test),
        "ood": eval_predictor(global_baseline["predict"], ood),
        "meaning": "matched-size flat baseline with no left/right subheap access",
    }

    example_heap = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    example = {
        "heap": example_heap,
        "target": conv_targets(example_heap) + example_heap[3:],
        "treeheap_internal_pred": shared_kernel_predict(example_heap, theta, CORRECT_TRIPLES),
        "wrong_address_internal_pred": shared_kernel_predict(example_heap, wrong_theta, WRONG_TRIPLES),
    }

    pass_checks = {
        "theta_updates": treeheap["theta_delta_l2"] > args.min_theta_delta,
        "theta_recovers_hidden": treeheap["theta_l2_error"] < args.max_theta_error,
        "test_mse_low": treeheap["test"]["mse"] < args.max_mse,
        "ood_mse_low": treeheap["ood"]["mse"] < args.max_mse,
        "wrong_address_materially_worse": wrong_address["test"]["mse"] > max(args.min_wrong_mse, treeheap["test"]["mse"] * args.min_wrong_ratio),
        "flat_global_materially_worse": flat_global["test"]["mse"] > max(args.min_wrong_mse, treeheap["test"]["mse"] * args.min_wrong_ratio),
    }

    return {
        "claim": "S1-KERNEL-LEARN-C01",
        "predict": "P-S1-KERNEL39",
        "host": args.host_label,
        "config": {
            "seed": args.seed,
            "train": args.train,
            "test": args.test,
            "ood": args.ood,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_theta": HIDDEN_THETA,
            "correct_triples": CORRECT_TRIPLES,
            "wrong_triples": WRONG_TRIPLES,
        },
        "models": {
            "treeheap_shared_kernel": treeheap,
            "no_learning": no_learning,
            "wrong_address_shared_kernel": wrong_address,
            "flat_global_matched_size": flat_global,
            "index_specific_local_kernel": index_specific,
        },
        "example": example,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": "If pilot_pass is true, parameter TreeHeap Theta learned a local convolution rule distinct from SPR-038 state relaxation.",
            "not_proved": [
                "not language understanding",
                "not WMT translation",
                "not superiority over every larger flat model",
                "not multi-kernel specialization",
                "not learned relation-field semantics",
            ],
        },
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trace = summary["models"]["treeheap_shared_kernel"]["trace"]
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(row) + "\n")

    metrics = summary["models"]["treeheap_shared_kernel"]
    wrong = summary["models"]["wrong_address_shared_kernel"]
    flat = summary["models"]["flat_global_matched_size"]
    readme = f"""# S1 Kernel Parameter Learning Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
learned_theta = {metrics['theta_final']}
theta_l2_error = {metrics['theta_l2_error']:.6g}
theta_delta_l2 = {metrics['theta_delta_l2']:.6g}
treeheap_test_mse = {metrics['test']['mse']:.6g}
treeheap_ood_mse = {metrics['ood']['mse']:.6g}
wrong_address_test_mse = {wrong['test']['mse']:.6g}
flat_global_test_mse = {flat['test']['mse']:.6g}
```

## Meaning

This proof updates `Theta`, the parameter TreeHeap / shared local kernel. It
does not update only `H`, the per-sample heap state. That is the key distinction
from SPR-038.

## Boundary

This does not prove language understanding, WMT translation, or superiority
over every larger flat model.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_kernel_parameter_learning_probe")
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--test", type=int, default=256)
    parser.add_argument("--ood", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max-theta-error", type=float, default=1e-4)
    parser.add_argument("--max-mse", type=float, default=1e-8)
    parser.add_argument("--min-theta-delta", type=float, default=0.5)
    parser.add_argument("--min-wrong-mse", type=float, default=0.05)
    parser.add_argument("--min-wrong-ratio", type=float, default=1000.0)
    parser.add_argument("--host-label", default="local")
    args = parser.parse_args()

    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
