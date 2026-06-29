#!/usr/bin/env python3
"""SPR-035 ordered fold kernel probe.

This is a pure toy proof for the question raised after SPR-034:

    If a 1D ordered token array is folded into a TreeHeap, which part is
    necessary for natural internal readout?

The probe compares three deterministic encoders:

1. ordered_tree_fold:
   Complete binary TreeHeap. Leaf addresses preserve the original order.
   Internal nodes compose exact subheap summaries. This should read natural
   subheap attributes exactly.

2. bag_root_fold:
   Collapses the whole sequence into one root-like bag/global summary. This
   keeps some global information but destroys address/path locality.

3. modulo_fold:
   Folds leaf positions into a small cyclic address base before reading. This
   is a diagnostic for the user's modulo/folding idea. It is expected to alias
   positions and fail natural subheap readout when used too early.

No learning and no GPU are used. The goal is not to prove translation. The
goal is to separate a deductive structural property:

    natural internal readout requires order-preserving path/address structure;
    modulo/residue should be a separate folding operator, not the first S1
    readout claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PAD = 0


@dataclass(frozen=True)
class Summary:
    length: int
    first: int
    last: int
    prefix0: int
    prefix1: int


def target_summary(seq: List[int], left: int, right: int) -> Summary:
    values = [tok for tok in seq[left:right] if tok != PAD]
    if not values:
        return Summary(0, PAD, PAD, PAD, PAD)
    return Summary(
        length=len(values),
        first=values[0],
        last=values[-1],
        prefix0=values[0],
        prefix1=values[1] if len(values) > 1 else PAD,
    )


def compose(left: Summary, right: Summary) -> Summary:
    length = left.length + right.length
    if length == 0:
        return Summary(0, PAD, PAD, PAD, PAD)

    first = left.first if left.length else right.first
    last = right.last if right.length else left.last

    prefix: List[int] = []
    if left.length:
        prefix.append(left.prefix0)
        if left.length > 1:
            prefix.append(left.prefix1)
    if len(prefix) < 2 and right.length:
        prefix.append(right.prefix0)
        if len(prefix) < 2 and right.length > 1:
            prefix.append(right.prefix1)

    while len(prefix) < 2:
        prefix.append(PAD)

    return Summary(length, first, last, prefix[0], prefix[1])


def node_span(node: int, max_len: int) -> Tuple[int, int]:
    """Return [left, right) leaf span for a 1-indexed complete binary heap."""
    path_bits: List[int] = []
    cur = node
    while cur > 1:
        path_bits.append(cur & 1)
        cur //= 2
    left, right = 0, max_len
    for bit in reversed(path_bits):
        mid = (left + right) // 2
        if bit == 0:
            right = mid
        else:
            left = mid
    return left, right


def ordered_tree_fold(seq: List[int], max_len: int) -> Dict[int, Summary]:
    """Build exact TreeHeap summaries for all nodes."""
    tree: Dict[int, Summary] = {}
    leaf_base = max_len
    for pos in range(max_len):
        tree[leaf_base + pos] = target_summary(seq, pos, pos + 1)
    for node in range(leaf_base - 1, 0, -1):
        tree[node] = compose(tree[node * 2], tree[node * 2 + 1])
    return tree


def bag_root_fold(seq: List[int], max_len: int) -> Dict[int, Summary]:
    """Every query receives the same global summary, losing subheap locality."""
    global_summary = target_summary(seq, 0, max_len)
    return {node: global_summary for node in range(1, max_len)}


def modulo_fold(seq: List[int], max_len: int, base: int) -> Dict[int, Summary]:
    """Fold positions by modulo class before answering subheap queries.

    This intentionally aliases positions. For a queried span [l, r), we collect
    every token whose position has a residue that appears inside [l, r). This
    is a crude cyclic-address fold: useful as a diagnostic, but not a natural
    subheap readout.
    """
    answers: Dict[int, Summary] = {}
    for node in range(1, max_len):
        left, right = node_span(node, max_len)
        residues = {pos % base for pos in range(left, right)}
        pseudo = [tok if (pos % base in residues) else PAD for pos, tok in enumerate(seq)]
        answers[node] = target_summary(pseudo, 0, max_len)
    return answers


def score(pred: Summary, target: Summary) -> Dict[str, int]:
    return {
        "length": int(pred.length == target.length),
        "first": int(pred.first == target.first),
        "last": int(pred.last == target.last),
        "prefix0": int(pred.prefix0 == target.prefix0),
        "prefix1": int(pred.prefix1 == target.prefix1),
        "exact_all": int(pred == target),
    }


def random_sequence(rng: random.Random, max_len: int, vocab: int) -> List[int]:
    length = rng.randint(max_len // 2, max_len)
    seq = [rng.randint(1, vocab - 1) for _ in range(length)]
    return seq + [PAD] * (max_len - length)


def empty_metrics() -> Dict[str, int]:
    return {k: 0 for k in ["length", "first", "last", "prefix0", "prefix1", "exact_all", "count"]}


def normalize(metrics: Dict[str, int]) -> Dict[str, float]:
    count = metrics["count"]
    out = {k: (metrics[k] / count if count else 0.0) for k in metrics if k != "count"}
    out["mean_natural"] = sum(out[k] for k in ["length", "first", "last", "prefix0", "prefix1"]) / 5.0
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    rng = random.Random(args.seed)
    model_metrics = {
        "ordered_tree_fold": empty_metrics(),
        "bag_root_fold": empty_metrics(),
        f"modulo_fold_base{args.mod_base}": empty_metrics(),
    }

    examples = []
    internal_nodes = list(range(1, args.max_len))
    for sample_idx in range(args.samples):
        seq = random_sequence(rng, args.max_len, args.vocab)
        ordered = ordered_tree_fold(seq, args.max_len)
        bag = bag_root_fold(seq, args.max_len)
        mod = modulo_fold(seq, args.max_len, args.mod_base)
        models = {
            "ordered_tree_fold": ordered,
            "bag_root_fold": bag,
            f"modulo_fold_base{args.mod_base}": mod,
        }

        for node in internal_nodes:
            left, right = node_span(node, args.max_len)
            target = target_summary(seq, left, right)
            if target.length == 0:
                continue
            for name, answers in models.items():
                s = score(answers[node], target)
                for key, value in s.items():
                    model_metrics[name][key] += value
                model_metrics[name]["count"] += 1

        if sample_idx < 3:
            node = rng.choice(internal_nodes)
            left, right = node_span(node, args.max_len)
            examples.append(
                {
                    "seq": seq,
                    "query_node": node,
                    "span": [left, right],
                    "target": target_summary(seq, left, right).__dict__,
                    "ordered_tree_fold": ordered[node].__dict__,
                    "bag_root_fold": bag[node].__dict__,
                    f"modulo_fold_base{args.mod_base}": mod[node].__dict__,
                }
            )

    normalized = {name: normalize(metrics) for name, metrics in model_metrics.items()}
    ordered_mean = normalized["ordered_tree_fold"]["mean_natural"]
    bag_mean = normalized["bag_root_fold"]["mean_natural"]
    mod_mean = normalized[f"modulo_fold_base{args.mod_base}"]["mean_natural"]

    return {
        "claim": "S1-FOLD-C01",
        "predict": "P-S1-FOLD01",
        "seed": args.seed,
        "samples": args.samples,
        "max_len": args.max_len,
        "vocab": args.vocab,
        "mod_base": args.mod_base,
        "models": normalized,
        "derived": {
            "ordered_minus_bag_mean_natural": ordered_mean - bag_mean,
            "ordered_minus_mod_mean_natural": ordered_mean - mod_mean,
        },
        "pilot_pass": ordered_mean == 1.0 and (ordered_mean - bag_mean) >= 0.35 and (ordered_mean - mod_mean) >= 0.25,
        "examples": examples,
        "interpretation": {
            "supported": "Order-preserving TreeHeap fold exactly preserves natural internal readout; bag/root and early modulo fold lose subheap locality.",
            "not_proved": [
                "not translation",
                "not learned semantic routing",
                "not a neural baseline battle",
                "not proof that modulo is useless",
                "not long real syntax",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ara/s1-echo/evidence/s1_ordered_fold_kernel_probe"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--mod-base", type=int, default=4)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = run(args)
    summary["elapsed_sec"] = round(time.time() - start, 3)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# S1 ordered fold kernel probe",
                "",
                "SPR-035 tests whether natural internal readout requires an",
                "order-preserving TreeHeap fold.",
                "",
                "Decision: `S1-FOLD-C01 -> supported pilot` if `pilot_pass=true`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (args.out / "trace.jsonl").write_text(
        json.dumps({"event": "complete", "elapsed_sec": summary["elapsed_sec"], "pilot_pass": summary["pilot_pass"]}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
