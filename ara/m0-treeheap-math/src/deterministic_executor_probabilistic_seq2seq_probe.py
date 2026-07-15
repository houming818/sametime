#!/usr/bin/env python3
"""Verify the deterministic algebra / probabilistic selection boundary."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from pathlib import Path

import numpy as np

import algebraic_operator_codec_probe as base


RESPONSES = {
    "earth": ["earth", "is", "round"],
    "weather": ["weather", "is", "clear"],
    "memory": ["memory", "has", "changed"],
}
FIRST_TOKENS = ["earth", "weather", "memory"]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def first_token_probs(heap: np.ndarray) -> np.ndarray:
    """A deterministic toy decoder conditioned on persistent TreeHeap state."""
    if int(round(float(heap[1]))) % 2:
        logits = np.array([2.0, 1.4, 0.2], dtype=np.float64)
    else:
        logits = np.array([0.6, 0.2, 2.4], dtype=np.float64)
    return softmax(logits)


def decode(heap: np.ndarray, mode: str, rng: np.random.Generator | None = None) -> list[str]:
    probs = first_token_probs(heap)
    if mode == "greedy":
        first = int(probs.argmax())
    elif mode == "sample":
        if rng is None:
            raise ValueError("sample mode requires rng")
        first = int(rng.choice(len(FIRST_TOKENS), p=probs))
    else:
        raise ValueError(mode)
    # Once the first response branch is selected, its sequence is read
    # deterministically. No random number enters the TreeHeap executor.
    return RESPONSES[FIRST_TOKENS[first]]


def random_program(rng: random.Random, tree_depth: int) -> tuple[int, int, int, int]:
    depth = rng.randint(1, min(5, tree_depth))
    max_level = tree_depth - depth
    target_level = rng.randint(0, max_level)
    address = rng.randint(2 ** target_level, 2 ** (target_level + 1) - 1)
    op = rng.randrange(len(base.OPS))
    delta_index = rng.randrange(len(base.DELTAS))
    return op, address, depth, delta_index


def execute(heap, program, size):
    op, address, depth, delta_index = program
    if op == 0:
        return base.mirror_subtree(heap, address, depth, size)
    return base.plus_subtree(heap, address, depth, base.DELTAS[delta_index], size)


def run(args):
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    deterministic_exact = 0
    inverse_exact = 0
    depth_buckets = {str(depth): {"count": 0, "repeat_exact": 0, "inverse_exact": 0} for depth in range(1, 6)}
    for _ in range(args.operator_trials):
        tree_depth = py_rng.randint(5, base.MAX_TREE_DEPTH)
        heap = base.legal_heap(tree_depth, py_rng.randint(1, 7))
        size = 2 ** (tree_depth + 1) - 1
        program = random_program(py_rng, tree_depth)
        first = execute(heap, program, size)
        second = execute(heap, program, size)
        repeat_ok = np.array_equal(first[:size + 1], second[:size + 1])
        deterministic_exact += int(repeat_ok)

        op, address, depth, delta_index = program
        if op == 0:
            restored = base.mirror_subtree(first, address, depth, size)
        else:
            restored = base.plus_subtree(first, address, depth, -base.DELTAS[delta_index], size)
        inverse_ok = np.array_equal(heap[:size + 1], restored[:size + 1])
        inverse_exact += int(inverse_ok)
        bucket = depth_buckets[str(depth)]
        bucket["count"] += 1
        bucket["repeat_exact"] += int(repeat_ok)
        bucket["inverse_exact"] += int(inverse_ok)

    # A probability bucket chooses a program once. Conditional execution is
    # still a deterministic function of the selected program and heap.
    selector_heap = base.legal_heap(6, 3)
    selector_size = 2 ** 7 - 1
    programs = [
        (0, 1, 2, 0), (0, 3, 3, 0),
        (1, 2, 2, 1), (1, 5, 1, 3),
    ]
    program_probs = softmax(np.array([2.0, 1.2, 0.5, -0.2]))
    sampled_ids = np_rng.choice(len(programs), size=args.sample_trials, p=program_probs)
    conditional_exact = 0
    for program_id in sampled_ids:
        program = programs[int(program_id)]
        a = execute(selector_heap, program, selector_size)
        b = execute(selector_heap, program, selector_size)
        conditional_exact += int(np.array_equal(a[:selector_size + 1], b[:selector_size + 1]))

    # Fixed state + greedy collapse must be one sequence.
    state_a = base.legal_heap(4, 1)
    size_a = 2 ** 5 - 1
    greedy_a = [tuple(decode(state_a, "greedy")) for _ in range(args.decode_repeats)]
    state_b = base.plus_subtree(state_a, 1, 2, 1, size_a)
    greedy_b = [tuple(decode(state_b, "greedy")) for _ in range(args.decode_repeats)]

    sampled = [tuple(decode(state_a, "sample", np_rng)) for _ in range(args.sample_trials)]
    counts = {token: 0 for token in FIRST_TOKENS}
    for sequence in sampled:
        counts[sequence[0]] += 1
    empirical = np.array([counts[token] / args.sample_trials for token in FIRST_TOKENS])
    declared = first_token_probs(state_a)
    kl = float(np.sum(empirical * np.log((empirical + 1e-12) / declared)))

    metrics = {
        "executor_repeat_exact": deterministic_exact / args.operator_trials,
        "inverse_restore_exact": inverse_exact / args.operator_trials,
        "executor_by_recursive_depth": {
            depth: {
                "count": values["count"],
                "repeat_exact": values["repeat_exact"] / max(values["count"], 1),
                "inverse_exact": values["inverse_exact"] / max(values["count"], 1),
            }
            for depth, values in depth_buckets.items()
        },
        "program_selector_declared_probs": program_probs.tolist(),
        "sampled_program_unique": int(len(set(int(x) for x in sampled_ids))),
        "sampled_program_conditional_exact": conditional_exact / args.sample_trials,
        "fixed_state_greedy_unique": len(set(greedy_a)),
        "fixed_state_greedy_output": list(greedy_a[0]),
        "sampled_unique_sequences": len(set(sampled)),
        "sampled_sequence_counts": {" ".join(RESPONSES[token]): counts[token] for token in FIRST_TOKENS},
        "declared_first_token_probs": dict(zip(FIRST_TOKENS, declared.tolist())),
        "empirical_first_token_probs": dict(zip(FIRST_TOKENS, empirical.tolist())),
        "sampled_first_token_kl": kl,
        "state_a_root": float(state_a[1]),
        "state_b_root": float(state_b[1]),
        "state_a_greedy_unique": len(set(greedy_a)),
        "state_b_greedy_unique": len(set(greedy_b)),
        "state_a_greedy_output": list(greedy_a[0]),
        "state_b_greedy_output": list(greedy_b[0]),
        "state_outputs_differ": greedy_a[0] != greedy_b[0],
    }
    gates = {
        "deterministic_executor": metrics["executor_repeat_exact"] == 1.0,
        "inverse_executor": metrics["inverse_restore_exact"] == 1.0,
        "conditional_after_probability_collapse": metrics["sampled_program_conditional_exact"] == 1.0,
        "greedy_fixed_state": metrics["fixed_state_greedy_unique"] == 1,
        "sampling_diversity": metrics["sampled_unique_sequences"] > 1 and kl < 0.02,
        "deterministic_state_dependence": metrics["state_a_greedy_unique"] == 1 and metrics["state_b_greedy_unique"] == 1 and metrics["state_outputs_differ"],
    }
    return metrics, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/m0-treeheap-math/evidence/deterministic_executor_probabilistic_seq2seq")
    ap.add_argument("--operator-trials", type=int, default=10000)
    ap.add_argument("--sample-trials", type=int, default=10000)
    ap.add_argument("--decode-repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()
    started = time.time()
    metrics, gates = run(args)
    summary = {
        "claim": "M0-DETSEQ-C01", "host": socket.gethostname(),
        "seconds": time.time() - started, "config": vars(args),
        "gates": gates, "all_gates_pass": all(gates.values()), "metrics": metrics,
        "scope": "Execution-boundary proof; not a learned language codec.",
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (out / "README.md").write_text("# Deterministic Executor / Probabilistic Seq2Seq Proof\n\nSee `summary.json`.\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
