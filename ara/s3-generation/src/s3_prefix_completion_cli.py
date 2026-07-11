#!/usr/bin/env python3
"""Interactive readout of the current frozen S1 -> S3 prefix-bucket gate.

This is intentionally a small, honest demo.  It does not generate arbitrary
text.  Given a phrase containing one of the 18 synthetic objects used by the
S1 encoder probe, it shows the learned TreeHeap prefix and the S3 probability
bucket over the six surface category labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_best_assignment(s1_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((s1_dir / "summary.json").read_text(encoding="utf-8"))
    results = json.loads((s1_dir / "results.json").read_text(encoding="utf-8"))
    structured = [row for row in results if row["mode"] == "structured"]
    if not structured:
        raise ValueError("No structured S1 assignments were found.")
    best = max(
        structured,
        key=lambda row: float(row["cluster_purity"]) + float(row["heldout_mrr"]),
    )
    return summary["vocab"], best


def build_bucket(vocab: dict[str, Any], assignment: dict[str, Any]) -> dict[int, Counter[int]]:
    buckets: dict[int, Counter[int]] = defaultdict(Counter)
    for prefix, category in zip(assignment["assignments"], vocab["object_category"]):
        buckets[int(prefix)][int(category)] += 1
    return buckets


def decode_object(
    name: str,
    vocab: dict[str, Any],
    assignment: dict[str, Any],
    buckets: dict[int, Counter[int]],
    alpha: float,
) -> dict[str, Any]:
    objects = [str(item) for item in vocab["objects"]]
    categories = [str(item) for item in vocab["categories"]]
    index = objects.index(name)
    prefix = int(assignment["assignments"][index])
    counts = buckets[prefix]
    denominator = sum(counts.values()) + alpha * len(categories)
    scored = [
        (category, (counts.get(category_index, 0) + alpha) / denominator)
        for category_index, category in enumerate(categories)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return {
        "object": name,
        "prefix": prefix,
        "top": scored[0][0],
        "candidates": scored,
    }


def find_object(text: str, objects: list[str]) -> str | None:
    words = set(text.lower().replace("?", " ").replace(".", " ").split())
    matches = [name for name in objects if name in words]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 frozen prefix-bucket completion demo")
    parser.add_argument(
        "--s1-evidence-dir",
        default="ara/s1-echo/evidence/s1_encoder_minimal_observer_probe_026",
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()

    vocab, assignment = load_best_assignment(Path(args.s1_evidence_dir))
    objects = [str(item) for item in vocab["objects"]]
    buckets = build_bucket(vocab, assignment)

    print("S3 frozen TreeHeap prefix completion demo")
    print("Scope: category completion over the 18-object synthetic S1 vocabulary.")
    print("Commands: :objects, :quit.  Example input: 'I take amoxicillin'.")
    print(
        f"Using best frozen S1 assignment: seed={assignment['seed']}, k={assignment['k']}."
    )

    while True:
        try:
            text = input("\nquery> ").strip()
        except EOFError:
            print()
            break
        if text in {":quit", ":q", "exit"}:
            break
        if text == ":objects":
            print(", ".join(objects))
            continue

        object_name = find_object(text, objects)
        if object_name is None:
            print("I can only read one known object from this POC. Try :objects.")
            continue

        result = decode_object(object_name, vocab, assignment, buckets, args.alpha)
        print(f"TreeHeap read: object={result['object']} -> prefix={result['prefix']}")
        print("Probability bucket:")
        for label, probability in result["candidates"]:
            print(f"  {label:10s} {probability:.3f}")
        print(f"Completion: '{object_name}' is most likely {result['top']}.")


if __name__ == "__main__":
    main()
