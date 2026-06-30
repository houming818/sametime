#!/usr/bin/env python3
"""SPR-040 TreeHeap conjugate kernel symmetry proof.

This script tests the mirror-conjugation law:

    M(K_theta(H)) == K_conj(theta)(M(H))

where conj([root,left,right]) = [root,right,left].

It also trains a mirrored kernel from mirrored targets and checks whether it
recovers the conjugate parameter TreeHeap.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


INTERNAL = (0, 1, 2)
TRIPLES = ((0, 1, 2), (1, 3, 4), (2, 5, 6))
MIRROR_PERM = (0, 2, 1, 6, 5, 4, 3)
THETA = [0.5, 1.25, -0.75]
THETA_CONJ = [0.5, -0.75, 1.25]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mirror_heap(heap: list[float]) -> list[float]:
    return [heap[j] for j in MIRROR_PERM]


def conj_theta(theta: list[float]) -> list[float]:
    return [theta[0], theta[2], theta[1]]


def conv_internal(heap: list[float], theta: list[float]) -> list[float]:
    return [dot(theta, [heap[j] for j in triple]) for triple in TRIPLES]


def conv_full(heap: list[float], theta: list[float]) -> list[float]:
    out = list(heap)
    vals = conv_internal(heap, theta)
    for idx, val in zip(INTERNAL, vals):
        out[idx] = val
    return out


def mse(preds: list[list[float]], targets: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for pred, target in zip(preds, targets):
        for p, t in zip(pred, target):
            total += (p - t) ** 2
            count += 1
    return total / count


def max_abs(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def make_dataset(n: int, rng: random.Random, low: float, high: float) -> list[list[float]]:
    return [[rng.uniform(low, high) for _ in range(7)] for _ in range(n)]


def equivariance_errors(heaps: list[list[float]], theta: list[float]) -> dict[str, float]:
    theta_c = conj_theta(theta)
    flipped_errors = []
    unflipped_errors = []
    for heap in heaps:
        left = mirror_heap(conv_full(heap, theta))
        right = conv_full(mirror_heap(heap), theta_c)
        wrong = conv_full(mirror_heap(heap), theta)
        flipped_errors.append(max_abs(left, right))
        unflipped_errors.append(max_abs(left, wrong))
    return {
        "max_flipped_error": max(flipped_errors),
        "mean_flipped_error": mean(flipped_errors),
        "min_unflipped_error": min(unflipped_errors),
        "mean_unflipped_error": mean(unflipped_errors),
    }


def train_kernel(
    heaps: list[list[float]],
    targets: list[list[float]],
    seed: int,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    rng = random.Random(seed)
    theta = [rng.uniform(-0.8, 0.8) for _ in range(3)]
    initial = list(theta)
    trace = []
    for epoch in range(epochs + 1):
        preds = [conv_internal(h, theta) for h in heaps]
        loss = mse(preds, targets)
        if epoch in {0, 1, 2, 5, 10, 25, 50, 100, 250, 500, epochs}:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "theta": list(theta),
                    "theta_conj_l2_error": math.sqrt(sum((a - b) ** 2 for a, b in zip(theta, THETA_CONJ))),
                }
            )
        if epoch == epochs:
            break
        grad = [0.0, 0.0, 0.0]
        denom = len(heaps) * 3
        for heap, target in zip(heaps, targets):
            for out_idx, triple in enumerate(TRIPLES):
                feats = [heap[j] for j in triple]
                err = dot(theta, feats) - target[out_idx]
                for k in range(3):
                    grad[k] += 2.0 * err * feats[k] / denom
        for k in range(3):
            theta[k] -= lr * grad[k]
    return {
        "theta_initial": initial,
        "theta_final": theta,
        "theta_delta_l2": math.sqrt(sum((a - b) ** 2 for a, b in zip(theta, initial))),
        "theta_conj_l2_error": math.sqrt(sum((a - b) ** 2 for a, b in zip(theta, THETA_CONJ))),
        "train_mse": mse([conv_internal(h, theta) for h in heaps], targets),
        "trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    train = make_dataset(args.train, rng, -3.0, 3.0)
    test = make_dataset(args.test, rng, -3.0, 3.0)
    ood = make_dataset(args.ood, rng, -10.0, 10.0)

    # Mirrored training data: input is M(H), target is the internal part of
    # M(K_theta(H)). A learned kernel should recover conj(theta).
    mirrored_train = [mirror_heap(h) for h in train]
    mirrored_targets = [conv_internal(mirror_heap(conv_full(h, THETA)), [1.0, 0.0, 0.0]) for h in train]
    # The line above extracts internal nodes of M(K_theta(H)) through the same
    # conv_internal interface with root-only theta. This keeps target indexing
    # explicit and avoids a separate helper.

    learned = train_kernel(mirrored_train, mirrored_targets, args.seed + 1, args.epochs, args.lr)
    theta_learned = learned["theta_final"]

    def eval_learned(heaps: list[list[float]]) -> dict[str, float]:
        inputs = [mirror_heap(h) for h in heaps]
        targets = [conv_internal(mirror_heap(conv_full(h, THETA)), [1.0, 0.0, 0.0]) for h in heaps]
        return {"mse": mse([conv_internal(h, theta_learned) for h in inputs], targets)}

    errors_test = equivariance_errors(test, THETA)
    errors_ood = equivariance_errors(ood, THETA)

    example_heap = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    example = {
        "heap": example_heap,
        "theta": THETA,
        "theta_conj": THETA_CONJ,
        "conv": conv_full(example_heap, THETA),
        "mirror_conv": mirror_heap(conv_full(example_heap, THETA)),
        "conv_mirror_with_conj": conv_full(mirror_heap(example_heap), THETA_CONJ),
        "conv_mirror_with_unflipped": conv_full(mirror_heap(example_heap), THETA),
    }

    pass_checks = {
        "deductive_test_equivariance": errors_test["max_flipped_error"] < args.max_equiv_error,
        "deductive_ood_equivariance": errors_ood["max_flipped_error"] < args.max_equiv_error,
        "unflipped_kernel_fails": errors_test["mean_unflipped_error"] > args.min_unflipped_error,
        "learned_conjugate_theta": learned["theta_conj_l2_error"] < args.max_theta_error,
        "learned_test_mse_low": eval_learned(test)["mse"] < args.max_mse,
        "learned_ood_mse_low": eval_learned(ood)["mse"] < args.max_mse,
    }

    return {
        "claim": "S1-KERNEL-CONJ-C01",
        "predict": "P-S1-KERNEL40",
        "host": args.host_label,
        "config": {
            "seed": args.seed,
            "train": args.train,
            "test": args.test,
            "ood": args.ood,
            "epochs": args.epochs,
            "lr": args.lr,
            "theta": THETA,
            "theta_conj": THETA_CONJ,
            "mirror_perm": MIRROR_PERM,
        },
        "deductive": {
            "test": errors_test,
            "ood": errors_ood,
        },
        "learned_conjugate": {
            **learned,
            "test": eval_learned(test),
            "ood": eval_learned(ood),
        },
        "example": example,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": "If pilot_pass is true, TreeHeap local convolution supports mirror conjugation and the mirrored kernel can be learned from mirrored data.",
            "not_proved": [
                "not language understanding",
                "not WMT translation",
                "not learned semantic conjugacy",
                "not arbitrary group equivariance",
                "not superiority over all flat models",
            ],
        },
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["learned_conjugate"]["trace"]:
            f.write(json.dumps(row) + "\n")
    readme = f"""# S1 Conjugate Kernel Symmetry Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
theta = {summary['config']['theta']}
theta_conj = {summary['config']['theta_conj']}
deductive_test_max_flipped_error = {summary['deductive']['test']['max_flipped_error']:.6g}
deductive_test_mean_unflipped_error = {summary['deductive']['test']['mean_unflipped_error']:.6g}
learned_theta = {summary['learned_conjugate']['theta_final']}
learned_theta_conj_l2_error = {summary['learned_conjugate']['theta_conj_l2_error']:.6g}
learned_test_mse = {summary['learned_conjugate']['test']['mse']:.6g}
learned_ood_mse = {summary['learned_conjugate']['ood']['mse']:.6g}
```

## Meaning

This proof tests both a deductive conjugation identity and an inductive learned
mirrored kernel.

## Boundary

It does not prove language understanding, WMT translation, or arbitrary group
equivariance.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_conjugate_kernel_symmetry_probe")
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--test", type=int, default=256)
    parser.add_argument("--ood", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max-equiv-error", type=float, default=1e-10)
    parser.add_argument("--max-theta-error", type=float, default=1e-5)
    parser.add_argument("--max-mse", type=float, default=1e-8)
    parser.add_argument("--min-unflipped-error", type=float, default=0.5)
    parser.add_argument("--host-label", default="local")
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
