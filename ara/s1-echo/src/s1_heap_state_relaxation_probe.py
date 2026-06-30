#!/usr/bin/env python3
"""SPR-038 Heap-state relaxation proof.

This is a CPU-only proof for Houming818's hypothesis:

    A TreeHeap gradient does not have to update only model parameters. It can
    update the heap state H itself, relaxing the current structure toward a
    lower-energy equilibrium while the kernel/rules remain fixed.

The script intentionally does not train theta. It only updates arr[i].
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


def scalar_relaxation(steps: int, lr: float) -> dict[str, object]:
    """Relax [root, left, right] using only state gradients.

    Energy:
        E = (left - right)^2 + (root - (left + right) / 2)^2

    This is a minimal analytic proof that a heap state can move from
    [2, 1, 3] toward a balanced state without changing any external parameter.
    """

    root, left, right = 2.0, 1.0, 3.0
    trace = []

    def energy(r: float, l: float, rr: float) -> float:
        return (l - rr) ** 2 + (r - (l + rr) / 2.0) ** 2

    for step in range(steps + 1):
        e = energy(root, left, right)
        if step in {0, 1, 2, 5, 10, 25, 50, 100, steps}:
            trace.append({"step": step, "root": root, "left": left, "right": right, "energy": e})
        if step == steps:
            break

        # Analytic gradients of E.
        mean_lr = (left + right) / 2.0
        diff = left - right
        parent_err = root - mean_lr

        d_root = 2.0 * parent_err
        d_left = 2.0 * diff - parent_err
        d_right = -2.0 * diff - parent_err

        root -= lr * d_root
        left -= lr * d_left
        right -= lr * d_right

    initial = trace[0]
    final = trace[-1]
    return {
        "name": "scalar_balance",
        "initial": initial,
        "final": final,
        "energy_drop": initial["energy"] - final["energy"],
        "energy_ratio": final["energy"] / initial["energy"],
        "state_delta": {
            "root": final["root"] - initial["root"],
            "left": final["left"] - initial["left"],
            "right": final["right"] - initial["right"],
        },
        "trace": trace,
    }


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def vec_mul(s: float, a: list[float]) -> list[float]:
    return [s * x for x in a]


def vec_mean(a: list[float], b: list[float]) -> list[float]:
    return [(x + y) / 2.0 for x, y in zip(a, b)]


def vec_sqnorm(a: list[float]) -> float:
    return sum(x * x for x in a)


def complete_tree_spans(max_leaf: int = 4) -> dict[int, tuple[int, int]]:
    return {
        1: (0, max_leaf),
        2: (0, max_leaf // 2),
        3: (max_leaf // 2, max_leaf),
        4: (0, 1),
        5: (1, 2),
        6: (2, 3),
        7: (3, 4),
    }


def centroid(points: list[list[float]], start: int, end: int) -> list[float]:
    dims = len(points[0])
    total = [0.0] * dims
    for point in points[start:end]:
        total = vec_add(total, point)
    return vec_mul(1.0 / (end - start), total)


def vector_energy_and_grad(
    state: dict[int, list[float]],
    leaves: dict[int, list[float]],
    relation_anchor: dict[int, list[float]],
    relation_weight: float,
    smooth_weight: float,
) -> tuple[float, dict[int, list[float]], dict[str, float]]:
    """Energy over a 7-node TreeHeap.

    Fixed rules:
        - leaves are observed token/world vectors and are not updated;
        - internal node should equal mean(left_child, right_child);
        - internal node should also stay near a relation anchor.

    The relation anchors are used to define the energy field. They are not
    trainable parameters in this probe.
    """

    dims = len(next(iter(state.values())))
    grad = {idx: [0.0] * dims for idx in state}
    total = 0.0
    consistency_energy = 0.0
    relation_energy = 0.0

    def node_value(idx: int) -> list[float]:
        return leaves[idx] if idx in leaves else state[idx]

    for parent, left, right in ((1, 2, 3), (2, 4, 5), (3, 6, 7)):
        parent_v = state[parent]
        mean_lr = vec_mean(node_value(left), node_value(right))
        err = vec_sub(parent_v, mean_lr)
        term = smooth_weight * vec_sqnorm(err)
        total += term
        consistency_energy += term

        # d ||p - (l+r)/2||^2 / dp
        grad[parent] = vec_add(grad[parent], vec_mul(2.0 * smooth_weight, err))

        # Internal children also receive gradients. Leaves are fixed observed
        # inputs and do not receive updates.
        for child in (left, right):
            if child in state:
                grad[child] = vec_add(grad[child], vec_mul(-smooth_weight, err))

    for idx, anchor in relation_anchor.items():
        err = vec_sub(state[idx], anchor)
        term = relation_weight * vec_sqnorm(err)
        total += term
        relation_energy += term
        grad[idx] = vec_add(grad[idx], vec_mul(2.0 * relation_weight, err))

    return total, grad, {"consistency": consistency_energy, "relation": relation_energy}


def vector_relaxation(seed: int, steps: int, lr: float) -> dict[str, object]:
    """Relax a 7-node vector TreeHeap with fixed rules and fixed leaves."""

    rng = random.Random(seed)
    leaves = {
        4: [0.0, 0.0],  # the
        5: [0.2, 0.1],  # cat
        6: [2.0, 1.8],  # is/running
        7: [2.2, 2.1],  # action complement
    }
    spans = complete_tree_spans(4)
    leaf_points = [leaves[i] for i in (4, 5, 6, 7)]

    relation_anchor = {
        idx: centroid(leaf_points, start, end)
        for idx, (start, end) in spans.items()
        if idx in (1, 2, 3)
    }

    # Bad initial internal heap state: intentionally far from child centroids.
    state = {
        idx: [rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)]
        for idx in (1, 2, 3)
    }

    relation_weight = 0.35
    smooth_weight = 1.0
    trace = []

    def centroid_error() -> float:
        return mean(
            math.sqrt(vec_sqnorm(vec_sub(state[idx], relation_anchor[idx])))
            for idx in (1, 2, 3)
        )

    for step in range(steps + 1):
        e, grad, parts = vector_energy_and_grad(state, leaves, relation_anchor, relation_weight, smooth_weight)
        if step in {0, 1, 2, 5, 10, 25, 50, 100, steps}:
            trace.append(
                {
                    "step": step,
                    "energy": e,
                    "consistency_energy": parts["consistency"],
                    "relation_energy": parts["relation"],
                    "centroid_error": centroid_error(),
                    "state": {str(k): v for k, v in state.items()},
                }
            )
        if step == steps:
            break
        for idx in state:
            state[idx] = vec_sub(state[idx], vec_mul(lr, grad[idx]))

    initial = trace[0]
    final = trace[-1]
    return {
        "name": "vector_treeheap_relaxation",
        "seed": seed,
        "fixed_kernel_parameters": {"relation_weight": relation_weight, "smooth_weight": smooth_weight},
        "leaves_fixed": {str(k): v for k, v in leaves.items()},
        "relation_anchor_not_trainable": {str(k): v for k, v in relation_anchor.items()},
        "initial": initial,
        "final": final,
        "energy_drop": initial["energy"] - final["energy"],
        "energy_ratio": final["energy"] / initial["energy"],
        "centroid_error_drop": initial["centroid_error"] - final["centroid_error"],
        "trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    scalar = scalar_relaxation(args.scalar_steps, args.scalar_lr)
    vector_runs = [vector_relaxation(args.seed + offset, args.vector_steps, args.vector_lr) for offset in range(args.runs)]
    energy_ratios = [run["energy_ratio"] for run in vector_runs]
    centroid_drops = [run["centroid_error_drop"] for run in vector_runs]
    passes = [
        run["energy_ratio"] < args.max_energy_ratio and run["centroid_error_drop"] > args.min_centroid_drop
        for run in vector_runs
    ]
    summary = {
        "claim": "S1-RELAX-C01",
        "config": vars(args),
        "design": {
            "theta_updated": False,
            "heap_state_updated": True,
            "target_heap_used_in_loss": False,
            "loss_type": "energy over current heap state",
        },
        "scalar": scalar,
        "vector_runs": vector_runs,
        "metrics": {
            "scalar_energy_ratio": scalar["energy_ratio"],
            "scalar_left_delta": scalar["state_delta"]["left"],
            "scalar_right_delta": scalar["state_delta"]["right"],
            "mean_vector_energy_ratio": mean(energy_ratios),
            "max_vector_energy_ratio": max(energy_ratios),
            "mean_centroid_error_drop": mean(centroid_drops),
            "pass_rate": sum(passes) / len(passes),
            "pilot_pass": scalar["energy_ratio"] < args.max_energy_ratio and all(passes),
        },
    }
    return summary


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"scalar": summary["scalar"]}, ensure_ascii=False) + "\n")
        for run in summary["vector_runs"]:
            fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    metrics = summary["metrics"]
    readme = f"""# S1 heap-state relaxation probe

Claim: `S1-RELAX-C01`

This proof tests Houming818's heap-state gradient hypothesis:

```text
The gradient does not have to update only kernel parameters theta. It can update
the current TreeHeap state H so that the heap relaxes to a lower-energy
equilibrium.
```

## Design

- theta updated: `{summary["design"]["theta_updated"]}`
- heap state updated: `{summary["design"]["heap_state_updated"]}`
- target heap used in loss: `{summary["design"]["target_heap_used_in_loss"]}`
- loss type: `{summary["design"]["loss_type"]}`

## Metrics

- scalar energy ratio: `{metrics["scalar_energy_ratio"]:.8f}`
- scalar left delta: `{metrics["scalar_left_delta"]:.4f}`
- scalar right delta: `{metrics["scalar_right_delta"]:.4f}`
- mean vector energy ratio: `{metrics["mean_vector_energy_ratio"]:.8f}`
- max vector energy ratio: `{metrics["max_vector_energy_ratio"]:.8f}`
- mean centroid error drop: `{metrics["mean_centroid_error_drop"]:.4f}`
- pass rate: `{metrics["pass_rate"]:.4f}`
- pilot pass: `{metrics["pilot_pass"]}`

## Boundary

This does not prove translation, language understanding, or unsupervised world
model learning. It proves only that a differentiable energy over the current
heap state can generate gradients that move `arr[i]` toward a lower-energy
state while fixed kernel/rule parameters stay unchanged.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_heap_state_relaxation_probe")
    parser.add_argument("--scalar-steps", type=int, default=120)
    parser.add_argument("--scalar-lr", type=float, default=0.08)
    parser.add_argument("--vector-steps", type=int, default=160)
    parser.add_argument("--vector-lr", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3801)
    parser.add_argument("--max-energy-ratio", type=float, default=0.02)
    parser.add_argument("--min-centroid-drop", type=float, default=0.5)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
