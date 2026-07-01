#!/usr/bin/env python3
"""SPR-040 TreeHeap mirror / chiral flip kernel proof.

This script tests a precise algebraic implementation of geometric mirror:

    P_m K_theta(H) == K_{P_lr theta}(P_m H)

where:

- P_m mirrors heap addresses.
- P_lr swaps the local kernel's left/right slots.

This is not complex conjugation. It is a left/right mirror, also called a
chiral flip in this ARA note.
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
THETA_MIRROR = [0.5, -0.75, 1.25]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mirror_heap(heap: list[float]) -> list[float]:
    """Apply the heap-address mirror permutation P_m."""
    return [heap[j] for j in MIRROR_PERM]


def mirror_theta(theta: list[float]) -> list[float]:
    """Apply the local kernel-slot mirror permutation P_lr."""
    return [theta[0], theta[2], theta[1]]


def internal_values(heap: list[float]) -> list[float]:
    return [heap[i] for i in INTERNAL]


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
    theta_m = mirror_theta(theta)
    flipped_errors = []
    unflipped_errors = []
    for heap in heaps:
        left = mirror_heap(conv_full(heap, theta))
        right = conv_full(mirror_heap(heap), theta_m)
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
                    "theta_mirror_l2_error": math.sqrt(
                        sum((a - b) ** 2 for a, b in zip(theta, THETA_MIRROR))
                    ),
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
        "theta_mirror_l2_error": math.sqrt(sum((a - b) ** 2 for a, b in zip(theta, THETA_MIRROR))),
        "train_mse": mse([conv_internal(h, theta) for h in heaps], targets),
        "trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    train = make_dataset(args.train, rng, -3.0, 3.0)
    test = make_dataset(args.test, rng, -3.0, 3.0)
    ood = make_dataset(args.ood, rng, -10.0, 10.0)

    # Mirrored training data: input is P_m H, target is the internal part of
    # P_m K_theta(H). A learned kernel should recover P_lr theta.
    mirrored_train = [mirror_heap(h) for h in train]
    mirrored_targets = [internal_values(mirror_heap(conv_full(h, THETA))) for h in train]

    learned = train_kernel(mirrored_train, mirrored_targets, args.seed + 1, args.epochs, args.lr)
    theta_learned = learned["theta_final"]

    def eval_learned(heaps: list[list[float]]) -> dict[str, float]:
        inputs = [mirror_heap(h) for h in heaps]
        targets = [internal_values(mirror_heap(conv_full(h, THETA))) for h in heaps]
        return {"mse": mse([conv_internal(h, theta_learned) for h in inputs], targets)}

    errors_test = equivariance_errors(test, THETA)
    errors_ood = equivariance_errors(ood, THETA)

    example_heap = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    example = {
        "heap": example_heap,
        "theta": THETA,
        "theta_mirror": THETA_MIRROR,
        "conv": conv_full(example_heap, THETA),
        "mirror_conv": mirror_heap(conv_full(example_heap, THETA)),
        "conv_mirror_with_mirrored_theta": conv_full(mirror_heap(example_heap), THETA_MIRROR),
        "conv_mirror_with_unflipped_theta": conv_full(mirror_heap(example_heap), THETA),
    }

    learned_test = eval_learned(test)
    learned_ood = eval_learned(ood)
    pass_checks = {
        "deductive_test_equivariance": errors_test["max_flipped_error"] < args.max_equiv_error,
        "deductive_ood_equivariance": errors_ood["max_flipped_error"] < args.max_equiv_error,
        "unflipped_kernel_fails": errors_test["mean_unflipped_error"] > args.min_unflipped_error,
        "learned_mirrored_theta": learned["theta_mirror_l2_error"] < args.max_theta_error,
        "learned_test_mse_low": learned_test["mse"] < args.max_mse,
        "learned_ood_mse_low": learned_ood["mse"] < args.max_mse,
    }
    structure_assignment = {
        "claim": "kernel slots are structural directions, not anonymous scalar positions",
        "root_slot_stays_root": abs(theta_learned[0] - THETA[0]),
        "left_slot_learns_original_right": abs(theta_learned[1] - THETA[2]),
        "right_slot_learns_original_left": abs(theta_learned[2] - THETA[1]),
        "no_rotation_or_3d_fold_claim": True,
    }

    return {
        "claim": "S1-KERNEL-MIRROR-C01",
        "predict": "P-S1-KERNEL40",
        "host": args.host_label,
        "terminology": {
            "preferred": ["mirror", "chiral flip", "left/right mirror"],
            "retired": ["conjugate"],
            "reason": "This proof uses permutation of heap addresses and local kernel slots, not complex conjugation.",
        },
        "config": {
            "seed": args.seed,
            "train": args.train,
            "test": args.test,
            "ood": args.ood,
            "epochs": args.epochs,
            "lr": args.lr,
            "theta": THETA,
            "theta_mirror": THETA_MIRROR,
            "mirror_perm": MIRROR_PERM,
            "kernel_slot_perm": [0, 2, 1],
        },
        "deductive": {
            "law": "P_m K_theta(H) = K_{P_lr theta}(P_m H)",
            "test": errors_test,
            "ood": errors_ood,
        },
        "learned_mirror": {
            **learned,
            "test": learned_test,
            "ood": learned_ood,
        },
        "structure_assignment": structure_assignment,
        "example": example,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": "If pilot_pass is true, TreeHeap local convolution supports mirror/chiral flip equivariance and the mirrored root/left/right slot assignment can be learned from scalar loss.",
            "not_proved": [
                "not language understanding",
                "not WMT translation",
                "not learned semantic mirror in real corpora",
                "not learned rotation angle",
                "not full 3D fold",
                "not arbitrary group equivariance",
                "not superiority over all flat models",
            ],
        },
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["learned_mirror"]["trace"]:
            f.write(json.dumps(row) + "\n")
    readme = f"""# S1 Mirror / Chiral Kernel Flip Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
theta = {summary['config']['theta']}
theta_mirror = {summary['config']['theta_mirror']}
deductive_law = {summary['deductive']['law']}
deductive_test_max_flipped_error = {summary['deductive']['test']['max_flipped_error']:.6g}
deductive_test_mean_unflipped_error = {summary['deductive']['test']['mean_unflipped_error']:.6g}
learned_theta = {summary['learned_mirror']['theta_final']}
learned_theta_mirror_l2_error = {summary['learned_mirror']['theta_mirror_l2_error']:.6g}
left_slot_learns_original_right_error = {summary['structure_assignment']['left_slot_learns_original_right']:.6g}
right_slot_learns_original_left_error = {summary['structure_assignment']['right_slot_learns_original_left']:.6g}
learned_test_mse = {summary['learned_mirror']['test']['mse']:.6g}
learned_ood_mse = {summary['learned_mirror']['ood']['mse']:.6g}
```

## Meaning

This proof tests both a deductive mirror identity and an inductive learned
mirrored kernel. The retired word `conjugate` is intentionally avoided here:
the operation is a left/right address permutation plus a left/right kernel-slot
permutation.

The learned part is specifically a slot-assignment proof:

```text
root stays root
left learns the original right coefficient
right learns the original left coefficient
```

It is not a rotation-angle proof or a full 3D fold proof.

## Boundary

It does not prove language understanding, WMT translation, or arbitrary group
equivariance.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_mirror_kernel_symmetry_probe")
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
