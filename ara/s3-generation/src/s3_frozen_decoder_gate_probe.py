#!/usr/bin/env python3
"""S3 frozen TreeHeap decoder gate.

This probe deliberately avoids a big joint S3 model.

Input:
  A saved S1 minimal encoder run.  Each row contains a frozen object ->
  TreeHeap prefix assignment learned without gold category labels.

Decoder:
  For every frozen assignment row, build a small probability bucket:

      prefix_id -> P(surface_label | prefix_id)

  The "surface labels" in this first gate are category words such as food,
  medicine, beverage.  They are targets for decoder supervision, not inputs to
  the encoder.

Evaluation:
  Leave one object out, build the prefix bucket from the other objects, then
  ask whether the held-out object's prefix predicts the held-out surface label.

Why this matters:
  If structured S1 encoder output is better than shuffled encoder output, then
  S3 has evidence that frozen internal TreeHeap subheaps are readable.  If not,
  S3 should stay blocked.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


def entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12), 2) for p in probs if p > 0)


def rank_of_gold(probs: Sequence[float], gold: int) -> int:
    ranked = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    return ranked.index(gold) + 1


def decode_row(
    row: Dict[str, object],
    object_category: Sequence[int],
    category_names: Sequence[str],
) -> Dict[str, object]:
    assignments = [int(x) for x in row["assignments"]]
    n_cat = len(category_names)
    records = []

    for heldout_obj, gold_cat in enumerate(object_category):
        prefix_counts: Dict[int, Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()
        for obj_idx, cat in enumerate(object_category):
            if obj_idx == heldout_obj:
                continue
            prefix = assignments[obj_idx]
            prefix_counts[prefix][cat] += 1
            global_counts[cat] += 1

        heldout_prefix = assignments[heldout_obj]
        counts = prefix_counts.get(heldout_prefix)
        if not counts:
            counts = global_counts

        alpha = 0.25
        denom = sum(counts.values()) + alpha * n_cat
        probs = [(counts.get(ci, 0) + alpha) / denom for ci in range(n_cat)]
        pred = max(range(n_cat), key=lambda ci: (probs[ci], -ci))
        rank = rank_of_gold(probs, int(gold_cat))
        records.append(
            {
                "object_index": heldout_obj,
                "prefix": heldout_prefix,
                "gold": category_names[int(gold_cat)],
                "pred": category_names[pred],
                "top1": 1.0 if pred == int(gold_cat) else 0.0,
                "mrr": 1.0 / rank,
                "entropy_bits": entropy(probs),
                "bucket": {
                    category_names[ci]: round(probs[ci], 6)
                    for ci in range(n_cat)
                },
            }
        )

    return {
        "mode": row["mode"],
        "seed": row["seed"],
        "k": row["k"],
        "source_cluster_purity": row["cluster_purity"],
        "source_pairwise_f1": row["pairwise_f1"],
        "source_heldout_mrr": row["heldout_mrr"],
        "decoder_top1": mean(float(r["top1"]) for r in records),
        "decoder_mrr": mean(float(r["mrr"]) for r in records),
        "decoder_entropy_bits": mean(float(r["entropy_bits"]) for r in records),
        "examples": records[:6],
    }


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = ["decoder_top1", "decoder_mrr", "decoder_entropy_bits"]
    out: Dict[str, object] = {}
    for mode in sorted({str(r["mode"]) for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            metric: {
                "mean": mean(float(r[metric]) for r in mode_rows),
                "std": pstdev(float(r[metric]) for r in mode_rows),
                "n": len(mode_rows),
            }
            for metric in metrics
        }
    if "structured" in out and "shuffled" in out:
        out["structured_minus_shuffled"] = {
            metric: out["structured"][metric]["mean"] - out["shuffled"][metric]["mean"]
            for metric in metrics
        }
    return out


def best_examples(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    best = max(
        (r for r in rows if r["mode"] == "structured"),
        key=lambda r: float(r["decoder_mrr"]) + float(r["source_cluster_purity"]),
    )
    return {
        "mode": best["mode"],
        "seed": best["seed"],
        "k": best["k"],
        "decoder_top1": best["decoder_top1"],
        "decoder_mrr": best["decoder_mrr"],
        "examples": best["examples"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--s1-evidence-dir",
        default="ara/s1-echo/evidence/s1_encoder_minimal_observer_probe_026",
    )
    ap.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_frozen_decoder_gate_probe",
    )
    args = ap.parse_args()

    s1_dir = Path(args.s1_evidence_dir)
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((s1_dir / "summary.json").read_text(encoding="utf-8"))
    results = json.loads((s1_dir / "results.json").read_text(encoding="utf-8"))
    vocab = summary["vocab"]
    object_category = [int(x) for x in vocab["object_category"]]
    category_names = [str(x) for x in vocab["categories"]]

    rows = [decode_row(row, object_category, category_names) for row in results]
    agg = aggregate(rows)
    output = {
        "claim": "S3-FROZEN-DECODER-C01",
        "experiment": "P-S3-FROZEN-DECODER01",
        "source_s1_evidence": str(s1_dir),
        "method": "leave-one-object-out prefix probability bucket decoder over frozen S1 assignments",
        "surface_target": "category surface label",
        "summary": agg,
        "decision_hint": {
            "support_if": "structured frozen encoder decoder beats shuffled decoder on top1 and mrr",
            "not_proved": "exact sentence generation, WMT translation, or advantage over strong lexical semantic models",
        },
        "best_structured_examples": best_examples(rows),
    }

    (evidence_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (evidence_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (evidence_dir / "README.md").write_text(
        "\n".join(
            [
                "# S3 Frozen Decoder Gate Probe",
                "",
                "This evidence reads frozen S1 encoder assignments and tests whether",
                "internal TreeHeap prefix buckets can decode surface category labels.",
                "",
                "It is a minimal S3 gate, not a WMT proof.",
                "",
                "```json",
                json.dumps(output["summary"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
