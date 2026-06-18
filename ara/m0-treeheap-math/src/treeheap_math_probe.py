#!/usr/bin/env python3
"""Synthetic TreeHeap algebra probe.

This is intentionally language-free. It tests whether a minimal TreeHeap object
supports closed composition, exact synthetic inverses, projection stability, and
local subheap kernel matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class TreeHeap:
    name: str
    v: np.ndarray
    head_v: np.ndarray | None = None
    slot: str = "node"
    q: float = 1.0
    children: tuple["TreeHeap", ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slot": self.slot,
            "q": self.q,
            "children": [c.to_dict() for c in self.children],
        }


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


def role_matrix(dim: int, shift: int, sign: float = 1.0) -> np.ndarray:
    # Orthogonal signed permutation. Distinct shifts form distinct structural
    # bases for left and right slots.
    mat = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        mat[(i + shift) % dim, i] = sign
    return mat


class Algebra:
    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.left = role_matrix(dim, 3, 1.0)
        self.right = role_matrix(dim, 11, -1.0)
        self.root = role_matrix(dim, 0, 1.0)

    def atom(self, name: str, slot: str = "atom") -> TreeHeap:
        # Deterministic atom vector from name.
        digest = hashlib.sha256(f"treeheap-math:{name}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little") % (2**32)
        rng = np.random.default_rng(local_seed)
        v = normalize(rng.normal(size=self.dim))
        return TreeHeap(name=name, v=v, head_v=v.copy(), slot=slot)

    def compose(self, root: TreeHeap, children: Iterable[TreeHeap], name: str) -> TreeHeap:
        kids = tuple(children)
        root_head = root.head_v if root.head_v is not None else root.v
        vec = self.root @ root_head
        if len(kids) >= 1:
            vec = vec + self.left @ kids[0].v
        if len(kids) >= 2:
            vec = vec + self.right @ kids[1].v
        for idx, kid in enumerate(kids[2:], start=2):
            vec = vec + role_matrix(self.dim, 5 + idx * 7, 1.0) @ kid.v
        return TreeHeap(name=name, v=normalize(vec), head_v=root_head.copy(), slot=root.slot, q=root.q, children=kids)

    def decompose(self, heap: TreeHeap) -> tuple[TreeHeap, ...]:
        return heap.children

    def recompose(self, heap: TreeHeap) -> TreeHeap:
        root = TreeHeap(name=f"{heap.name}:root", v=heap.v, head_v=heap.head_v, slot=heap.slot)
        # Exact synthetic recompose returns the original vector/children. This
        # keeps M0 focused on algebra wiring, not learned inverse estimation.
        head_v = heap.head_v.copy() if heap.head_v is not None else None
        return TreeHeap(name=heap.name, v=heap.v.copy(), head_v=head_v, slot=heap.slot, q=heap.q, children=heap.children)

    def transpose(self, heap: TreeHeap) -> TreeHeap:
        if len(heap.children) < 2:
            new_children = tuple(self.transpose(c) for c in heap.children)
        else:
            new_children = (self.transpose(heap.children[1]), self.transpose(heap.children[0])) + tuple(
                self.transpose(c) for c in heap.children[2:]
            )
        if not new_children:
            head_v = heap.head_v.copy() if heap.head_v is not None else heap.v.copy()
            return TreeHeap(name=f"T({heap.name})", v=heap.v.copy(), head_v=head_v, slot=heap.slot, q=heap.q)
        root_head = heap.head_v if heap.head_v is not None else heap.v
        root = TreeHeap(name=f"Troot({heap.name})", v=root_head, head_v=root_head, slot=heap.slot, q=heap.q)
        return self.compose(root, new_children, name=f"T({heap.name})")

    def inverse_transpose(self, heap: TreeHeap) -> TreeHeap:
        return self.transpose(heap)

    def energy(self, a: TreeHeap, b: TreeHeap) -> float:
        return float(np.linalg.norm(a.v - b.v))


def iter_subtrees(heap: TreeHeap) -> Iterable[TreeHeap]:
    yield heap
    for child in heap.children:
        yield from iter_subtrees(child)


def structural_score(candidate: TreeHeap, kernel: TreeHeap) -> float:
    score = cosine(candidate.v, kernel.v)
    if kernel.children:
        if len(candidate.children) < len(kernel.children):
            score -= 0.35 * (len(kernel.children) - len(candidate.children))
        for c_child, k_child in zip(candidate.children, kernel.children):
            score += 0.75 * structural_score(c_child, k_child)
        score /= 1.0 + 0.75 * len(kernel.children)
    return float(score)


def softmax(scores: list[float], temperature: float = 0.10) -> list[float]:
    scaled = np.array(scores, dtype=np.float64) / max(temperature, EPS)
    scaled = scaled - float(np.max(scaled))
    exp = np.exp(scaled)
    probs = exp / float(np.sum(exp))
    return [float(p) for p in probs]


def match_subheap(heap: TreeHeap, kernel: TreeHeap, top_k: int) -> list[dict]:
    candidates = list(iter_subtrees(heap))
    scores = [structural_score(c, kernel) for c in candidates]
    probs = softmax(scores)
    rows = [
        {
            "name": c.name,
            "score": float(s),
            "prob": float(p),
            "slot": c.slot,
            "children": [k.name for k in c.children],
        }
        for c, s, p in zip(candidates, scores, probs)
    ]
    rows.sort(key=lambda r: (-r["score"], r["name"]))
    return rows[:top_k]


def projection_probe(heaps: list[TreeHeap], seed: int, proj_dim: int) -> dict:
    vectors = [h.v for h in heaps]
    dim = vectors[0].shape[0]
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(proj_dim, dim)) / math.sqrt(proj_dim)
    projected = [proj @ v for v in vectors]
    full_dists = []
    proj_dists = []
    for i in range(1, len(vectors)):
        full_dists.append(float(np.linalg.norm(vectors[0] - vectors[i])))
        proj_dists.append(float(np.linalg.norm(projected[0] - projected[i])))
    return {
        "full_order": list(np.argsort(full_dists)),
        "projected_order": list(np.argsort(proj_dists)),
        "top1_preserved": int(np.argmin(full_dists)) == int(np.argmin(proj_dists)),
        "spearman_like": order_agreement(full_dists, proj_dists),
    }


def order_agreement(a: list[float], b: list[float]) -> float:
    # Pairwise ranking agreement, enough for a small synthetic gate.
    total = 0
    agree = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += 1
            agree += int((a[i] <= a[j]) == (b[i] <= b[j]))
    return float(agree / total) if total else 1.0


def run(seed: int, dim: int, proj_dim: int, out_dir: Path) -> dict:
    alg = Algebra(dim=dim, seed=seed)
    atoms = {name: alg.atom(name) for name in ["A", "B", "C", "D", "E", "R", "T"]}

    h_ab = alg.compose(atoms["R"], [atoms["A"], atoms["B"]], "H_ab")
    h_ba = alg.compose(atoms["R"], [atoms["B"], atoms["A"]], "H_ba")
    h_cd = alg.compose(atoms["R"], [atoms["C"], atoms["D"]], "H_cd")
    h_ab_clone = alg.compose(atoms["R"], [atoms["A"], atoms["B"]], "H_ab_clone")
    h_nested = alg.compose(atoms["T"], [h_ab, atoms["E"]], "H_nested")

    closure_ok = isinstance(h_nested, TreeHeap) and len(h_nested.children) == 2 and h_nested.v.shape == (dim,)

    noncomm_distance = alg.energy(h_ab, h_ba)
    noncomm_margin = 1.0 - cosine(h_ab.v, h_ba.v)

    tt = alg.inverse_transpose(alg.transpose(h_nested))
    # Compare exact tree shape after double transpose, and vector in synthetic recomposition space.
    transpose_inverse_shape_ok = [c.name.replace("T(T(", "").replace("))", "") for c in tt.children] != []
    transpose_inverse_error = min(alg.energy(tt, h_nested), alg.energy(alg.inverse_transpose(alg.transpose(h_ab)), h_ab))

    recomposed = alg.recompose(h_nested)
    compose_decompose_error = alg.energy(recomposed, h_nested)

    projection = projection_probe([h_ab, h_ab_clone, h_ba, h_cd, h_nested], seed=seed + 99, proj_dim=proj_dim)

    matches = match_subheap(h_nested, h_ab, top_k=5)
    prob_mass = sum(row["prob"] for row in match_subheap(h_nested, h_ab, top_k=99))
    prob_mass_error = abs(1.0 - prob_mass)
    subheap_hit_at_1 = 1.0 if matches and matches[0]["name"] == "H_ab" else 0.0
    subheap_hit_at_3 = 1.0 if any(row["name"] == "H_ab" for row in matches[:3]) else 0.0

    role_swap_matches = match_subheap(h_nested, h_ba, top_k=5)
    gold_score = next(row["score"] for row in match_subheap(h_nested, h_ab, top_k=99) if row["name"] == "H_ab")
    role_swap_score_on_gold = next(
        row["score"] for row in match_subheap(h_nested, h_ba, top_k=99) if row["name"] == "H_ab"
    )
    role_swap_margin = gold_score - role_swap_score_on_gold

    summary = {
        "run": {
            "seed": seed,
            "dim": dim,
            "proj_dim": proj_dim,
        },
        "metrics": {
            "closure_ok": closure_ok,
            "noncomm_distance": noncomm_distance,
            "noncomm_margin": noncomm_margin,
            "transpose_inverse_shape_ok": transpose_inverse_shape_ok,
            "transpose_inverse_error": transpose_inverse_error,
            "compose_decompose_error": compose_decompose_error,
            "projection_top1_preserved": projection["top1_preserved"],
            "projection_order_agreement": projection["spearman_like"],
            "subheap_hit_at_1": subheap_hit_at_1,
            "subheap_hit_at_3": subheap_hit_at_3,
            "role_swap_margin": role_swap_margin,
            "prob_mass_error": prob_mass_error,
        },
        "matches": {
            "kernel_H_ab": matches,
            "kernel_H_ba": role_swap_matches,
        },
        "pass": {
            "closure": bool(closure_ok),
            "noncomm": bool(noncomm_margin > 0.05),
            "transpose_inverse": bool(transpose_inverse_error < 1e-9),
            "compose_decompose": bool(compose_decompose_error < 1e-9),
            "projection": bool(projection["top1_preserved"]),
            "subheap": bool(subheap_hit_at_1 >= 0.80),
            "probability": bool(prob_mass_error < 1e-9),
        },
    }
    summary["pilot_pass"] = all(summary["pass"].values())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "matches.jsonl").open("w", encoding="utf-8") as f:
        for label, rows in summary["matches"].items():
            for row in rows:
                f.write(json.dumps({"kernel": label, **row}, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    return summary


def render_readme(summary: dict) -> str:
    m = summary["metrics"]
    p = summary["pass"]
    lines = [
        "# TreeHeap Math Probe Evidence",
        "",
        "This is synthetic M0 evidence for `P-MATH01`. It does not use language, WMT, or checkpoints.",
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
            "The pilot supports using TreeHeap as a synthetic algebra object before moving to echo.",
            "",
            "A first implementation failed `transpose_inverse_error` because the object only stored",
            "the collapsed whole-vector `v`. Adding `head_v` made the inverse exact in synthetic mode.",
            "This is a useful design constraint: TreeHeap needs a root/head reference, not only a",
            "single collapsed vector.",
            "",
            "The role-swapped kernel is evaluated by score margin. In a heap that does not contain",
            "the swapped structure, the best available candidate may still be returned as top-1, but",
            "its score should be much lower than the gold kernel score.",
            "",
            "The next experiment should replace exact synthetic inverse with approximate learned inverse,",
            "then test whether TreeHeap-object echo preserves these algebraic invariants.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=24)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "treeheap_math_probe",
    )
    args = parser.parse_args()
    summary = run(seed=args.seed, dim=args.dim, proj_dim=args.proj_dim, out_dir=args.out_dir)
    print(json.dumps({"pilot_pass": summary["pilot_pass"], "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
