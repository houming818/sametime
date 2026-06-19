#!/usr/bin/env python3
"""Primitive + plus probe for addressable TreeHeap.

This probe turns the proof sketch into a synthetic experiment:

  TreeHeap state = addressable arr + summary
  plus(H, primitive) -> H'
  address = (cursor + 1) mod base

It tests whether plus can act as successor, information gain, and cyclic
folding operator before we use it for echo or language tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm < EPS:
        return x.copy()
    return x / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < EPS:
        return 0.0
    return float(np.dot(a, b) / den)


def stable_vec(name: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(f"primitive-plus:{name}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=dim))


@dataclass(frozen=True)
class Node:
    name: str
    value: np.ndarray
    occupied: bool = True
    generation: int = 0

    def short(self) -> dict:
        return {
            "name": self.name,
            "occupied": self.occupied,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class TreeHeapState:
    arr: tuple[Node | None, ...]
    base: int
    cursor: int
    summary: np.ndarray
    step: int

    @property
    def root(self) -> Node | None:
        return self.arr[0]


def empty_state(base: int, dim: int) -> TreeHeapState:
    return TreeHeapState(
        arr=tuple([None] * base),
        base=base,
        cursor=-1,
        summary=np.zeros(dim, dtype=np.float64),
        step=0,
    )


def mod_addr(state: TreeHeapState, index: int) -> int:
    return index % state.base


def parent(index: int) -> int | None:
    if index == 0:
        return None
    return (index - 1) // 2


def left(index: int) -> int:
    return 2 * index + 1


def right(index: int) -> int:
    return 2 * index + 2


def summarize(arr: tuple[Node | None, ...], dim: int) -> np.ndarray:
    acc = np.zeros(dim, dtype=np.float64)
    for i, node in enumerate(arr):
        if node is None:
            continue
        # Address-sensitive summary. It is a projection of arr, not the primary
        # state.
        shift = (i * 7 + 3) % dim
        acc = acc + np.roll(node.value, shift)
    return normalize(acc)


def info_count(state: TreeHeapState) -> int:
    return sum(1 for node in state.arr if node is not None)


def plus(state: TreeHeapState, primitive: Node) -> TreeHeapState:
    target = mod_addr(state, state.cursor + 1)
    arr = list(state.arr)
    arr[target] = Node(
        name=primitive.name,
        value=primitive.value.copy(),
        occupied=True,
        generation=state.step + 1,
    )
    new_arr = tuple(arr)
    return TreeHeapState(
        arr=new_arr,
        base=state.base,
        cursor=target,
        summary=summarize(new_arr, primitive.value.shape[0]),
        step=state.step + 1,
    )


def window(state: TreeHeapState, center: int, offsets: tuple[int, ...]) -> list[Node | None]:
    return [state.arr[mod_addr(state, center + offset)] for offset in offsets]


def kernel_score(patch: list[Node | None], kernel: tuple[Node, ...]) -> float:
    if len(patch) != len(kernel):
        raise ValueError("patch/kernel length mismatch")
    score = 0.0
    for got, want in zip(patch, kernel):
        if got is None:
            score -= 1.0
        else:
            score += cosine(got.value, want.value)
    return score / len(kernel)


def match_cyclic_kernel(state: TreeHeapState, kernel: tuple[Node, ...]) -> list[dict]:
    offsets = tuple(range(-(len(kernel) // 2), len(kernel) // 2 + 1))
    rows = []
    for center in range(state.base):
        patch = window(state, center, offsets)
        rows.append(
            {
                "center": center,
                "score": kernel_score(patch, kernel),
                "patch": [node.name if node is not None else None for node in patch],
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["center"]))
    return rows


def state_distance(a: TreeHeapState, b: TreeHeapState) -> float:
    return float(np.linalg.norm(a.summary - b.summary))


def run(seed: int, dim: int, base: int, out_dir: Path) -> dict:
    _ = np.random.default_rng(seed)
    primitives = [Node(name=f"p{i}", value=stable_vec(f"p{i}", dim)) for i in range(base + 2)]

    states = [empty_state(base=base, dim=dim)]
    write_trace = []
    for primitive in primitives:
        before = states[-1]
        after = plus(before, primitive)
        states.append(after)
        write_trace.append(
            {
                "step": after.step,
                "primitive": primitive.name,
                "target": after.cursor,
                "target_expected": (before.cursor + 1) % base,
                "info_before": info_count(before),
                "info_after": info_count(after),
                "root": after.root.name if after.root is not None else None,
            }
        )

    pre_base_pairs = write_trace[:base]
    overflow_pairs = write_trace[base:]

    closure_ok = all(isinstance(state, TreeHeapState) and len(state.arr) == base for state in states)
    root_ok = states[1].root is not None and states[1].root.name == "p0"
    successor_ok = all(row["target"] == row["target_expected"] for row in write_trace)
    info_gain_pre_base = all(row["info_after"] == row["info_before"] + 1 for row in pre_base_pairs)
    info_saturates_after_base = all(row["info_after"] == base for row in overflow_pairs)
    mod_fold_targets = [row["target"] for row in overflow_pairs]
    mod_fold_ok = mod_fold_targets == [0, 1]

    full_state = states[base]
    after_wrap = states[base + 1]
    overwritten_root_ok = after_wrap.root is not None and after_wrap.root.name == "p8"

    # Kernel around center 1 before wrap: [p0, p1, p2].
    kernel = tuple(primitives[:3])
    matches = match_cyclic_kernel(full_state, kernel)
    kernel_hit_at_1 = 1.0 if matches and matches[0]["center"] == 1 else 0.0

    # After wrap, [p0, p1, p2] should no longer be a perfect window because p0
    # at address 0 has been overwritten by p8.
    matches_after_wrap = match_cyclic_kernel(after_wrap, kernel)
    wrap_breaks_old_kernel = matches_after_wrap[0]["score"] < matches[0]["score"]

    # Address law checks.
    address_laws_ok = True
    for i in range(base):
        l = left(i)
        r = right(i)
        if l < base and parent(l) != i:
            address_laws_ok = False
        if r < base and parent(r) != i:
            address_laws_ok = False

    summary_consistency_ok = all(
        np.allclose(state.summary, summarize(state.arr, dim), atol=1e-12) for state in states
    )

    # The orbit is generated by plus over addresses. The state after base writes
    # is not equal to empty, but the next target folds to address 0.
    cycle_address_error = abs(write_trace[base]["target"] - 0) if len(write_trace) > base else math.inf
    summary_change_after_each_plus = [
        state_distance(states[i], states[i + 1]) for i in range(len(states) - 1)
    ]
    summary_moves_ok = all(delta > 1e-9 for delta in summary_change_after_each_plus)

    pass_gates = {
        "closure": closure_ok,
        "root_reference": root_ok,
        "successor": successor_ok,
        "info_gain_pre_base": info_gain_pre_base,
        "info_saturation_after_base": info_saturates_after_base,
        "mod_fold": mod_fold_ok,
        "overwritten_root": overwritten_root_ok,
        "cyclic_kernel": kernel_hit_at_1 >= 1.0,
        "wrap_breaks_old_kernel": wrap_breaks_old_kernel,
        "address_laws": address_laws_ok,
        "summary_consistency": summary_consistency_ok,
        "summary_moves": summary_moves_ok,
    }

    summary = {
        "run": {
            "seed": seed,
            "dim": dim,
            "base": base,
            "steps": len(primitives),
        },
        "metrics": {
            "closure_ok": closure_ok,
            "root_ok": root_ok,
            "successor_ok": successor_ok,
            "info_gain_pre_base": info_gain_pre_base,
            "info_saturates_after_base": info_saturates_after_base,
            "mod_fold_targets": mod_fold_targets,
            "mod_fold_ok": mod_fold_ok,
            "overwritten_root_ok": overwritten_root_ok,
            "kernel_hit_at_1": kernel_hit_at_1,
            "kernel_top_score": matches[0]["score"],
            "kernel_after_wrap_top_score": matches_after_wrap[0]["score"],
            "wrap_breaks_old_kernel": wrap_breaks_old_kernel,
            "address_laws_ok": address_laws_ok,
            "summary_consistency_ok": summary_consistency_ok,
            "summary_min_delta": min(summary_change_after_each_plus),
            "cycle_address_error": cycle_address_error,
        },
        "pass": pass_gates,
        "pilot_pass": all(pass_gates.values()),
        "write_trace": write_trace,
        "kernel_matches": matches[:5],
        "kernel_matches_after_wrap": matches_after_wrap[:5],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in write_trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    return summary


def render_readme(summary: dict) -> str:
    m = summary["metrics"]
    p = summary["pass"]
    lines = [
        "# Primitive Plus Probe Evidence",
        "",
        "This is synthetic M0 evidence for `P-MATH02`.",
        "",
        "The experiment treats TreeHeap as an addressable array:",
        "",
        "```text",
        "arr[0] = root",
        "next(i) = (i + 1) mod base",
        "plus(H, primitive) writes primitive to next address",
        "```",
        "",
        "## Verdict",
        "",
        f"`pilot_pass = {summary['pilot_pass']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in m.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Gates", "", "| Gate | Pass |", "|---|---:|"])
    for key, value in p.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The toy supports the narrow claim that an addressable TreeHeap can use",
            "`plus` as a successor operation with information gain before the base",
            "is full, and as a modular overwrite/fold operation after the base is full.",
            "",
            "This is not language evidence. It only validates the next mathematical",
            "toolbox layer needed before TreeHeap-object echo.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "primitive_plus_probe",
    )
    args = parser.parse_args()
    summary = run(seed=args.seed, dim=args.dim, base=args.base, out_dir=args.out_dir)
    print(json.dumps({"pilot_pass": summary["pilot_pass"], "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
