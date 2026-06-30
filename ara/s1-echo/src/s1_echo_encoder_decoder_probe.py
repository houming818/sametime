#!/usr/bin/env python3
"""S1 explicit Echo Encoder / Decoder proof.

Question:
    Can we implement a clean TreeHeap echo encoder and decoder?

This proof is deliberately algebraic and CPU-only:

    Encoder:
        token sequence -> ordered TreeHeap leaves -> internal summaries

    Decoder:
        root length + path-addressed leaf reads -> reconstructed sequence
        internal node path -> subheap span decode

No neural parameters are trained. The goal is to pin down the interface before
we ask a learned kernel to approximate it.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PAD = 0


@dataclass(frozen=True)
class NodeState:
    length: int
    first: int
    last: int
    checksum: int


@dataclass
class EncodedHeap:
    max_len: int
    leaf_base: int
    arr: dict[int, NodeState]
    leaves: dict[int, int]


def checksum(values: Iterable[int], mod: int = 1_000_003) -> int:
    acc = 0
    for idx, value in enumerate(values):
        acc = (acc + (idx + 1) * int(value)) % mod
    return acc


def leaf_state(token: int) -> NodeState:
    if token == PAD:
        return NodeState(0, PAD, PAD, 0)
    return NodeState(1, token, token, checksum([token]))


def compose(left: NodeState, right: NodeState) -> NodeState:
    length = left.length + right.length
    if length == 0:
        return NodeState(0, PAD, PAD, 0)
    first = left.first if left.length else right.first
    last = right.last if right.length else left.last
    # Ordered checksum: right side is shifted by left length, so order matters.
    combined_checksum = (left.checksum + right.checksum * (left.length + 1)) % 1_000_003
    return NodeState(length, first, last, combined_checksum)


def next_power_of_two(n: int) -> int:
    out = 1
    while out < n:
        out *= 2
    return out


def node_span(node: int, leaf_count: int) -> tuple[int, int]:
    left, right = 0, leaf_count
    path = []
    cur = node
    while cur > 1:
        path.append(cur & 1)
        cur //= 2
    for bit in reversed(path):
        mid = (left + right) // 2
        if bit == 0:
            right = mid
        else:
            left = mid
    return left, right


def leaf_node_for_position(pos: int, leaf_count: int) -> int:
    return leaf_count + pos


class TreeHeapEchoEncoder:
    """Ordered hard TreeHeap echo encoder."""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.leaf_count = next_power_of_two(max_len)
        self.leaf_base = self.leaf_count

    def encode(self, tokens: list[int]) -> EncodedHeap:
        if len(tokens) > self.max_len:
            raise ValueError(f"tokens length {len(tokens)} exceeds max_len {self.max_len}")
        padded = tokens + [PAD] * (self.leaf_count - len(tokens))
        arr: dict[int, NodeState] = {}
        leaves: dict[int, int] = {}
        for pos, token in enumerate(padded):
            node = self.leaf_base + pos
            leaves[node] = token
            arr[node] = leaf_state(token)
        for node in range(self.leaf_base - 1, 0, -1):
            arr[node] = compose(arr[node * 2], arr[node * 2 + 1])
        return EncodedHeap(self.max_len, self.leaf_base, arr, leaves)


class TreeHeapEchoDecoder:
    """Path-addressed TreeHeap echo decoder."""

    def decode_leaf(self, heap: EncodedHeap, pos: int) -> int:
        node = leaf_node_for_position(pos, heap.leaf_base)
        return heap.leaves.get(node, PAD)

    def decode_sequence(self, heap: EncodedHeap) -> list[int]:
        length = heap.arr[1].length
        return [self.decode_leaf(heap, pos) for pos in range(length)]

    def decode_subheap(self, heap: EncodedHeap, node: int) -> list[int]:
        left, right = node_span(node, heap.leaf_base)
        values = [self.decode_leaf(heap, pos) for pos in range(left, right)]
        return [value for value in values if value != PAD]

    def read_summary(self, heap: EncodedHeap, node: int) -> NodeState:
        return heap.arr[node]


def random_sequence(rng: random.Random, min_len: int, max_len: int, vocab: int) -> list[int]:
    length = rng.randint(min_len, max_len)
    return [rng.randint(1, vocab - 1) for _ in range(length)]


def load_wmt_sequences(args: argparse.Namespace) -> tuple[list[list[int]], dict[str, object]]:
    """Try to load WMT SentencePiece sequences; fall back to synthetic data."""

    wmt_path = Path(args.wmt_path)
    spm_path = Path(args.spm_model)
    if not wmt_path.exists() or not spm_path.exists():
        return [], {
            "source": "synthetic_fallback",
            "reason": "WMT or SentencePiece model not found",
            "wmt_path": str(wmt_path),
            "spm_model": str(spm_path),
        }

    try:
        import sentencepiece as spm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on remote env
        return [], {
            "source": "synthetic_fallback",
            "reason": f"sentencepiece import failed: {exc}",
            "wmt_path": str(wmt_path),
            "spm_model": str(spm_path),
        }

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    seqs: list[list[int]] = []
    examples: list[str] = []
    with wmt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.split("\t", 1)[0].strip()
            if not text:
                continue
            ids = [int(i) + 1 for i in sp.encode(text, out_type=int)]
            ids = [i for i in ids if 0 < i < args.vocab_limit]
            if args.min_len <= len(ids) <= args.max_len:
                seqs.append(ids)
                if len(examples) < 5:
                    examples.append(text)
            if len(seqs) >= args.samples:
                break
    return seqs, {
        "source": "wmt_sentencepiece",
        "wmt_path": str(wmt_path),
        "spm_model": str(spm_path),
        "examples": examples,
    }


def evaluate_sequences(seqs: list[list[int]], max_len: int, sample_nodes: int, seed: int) -> dict[str, object]:
    encoder = TreeHeapEchoEncoder(max_len)
    decoder = TreeHeapEchoDecoder()
    rng = random.Random(seed)
    seq_exact = 0
    leaf_ok = 0
    leaf_count = 0
    subheap_ok = 0
    subheap_count = 0
    summary_ok = 0
    summary_count = 0
    examples = []

    for idx, seq in enumerate(seqs):
        heap = encoder.encode(seq)
        decoded = decoder.decode_sequence(heap)
        seq_exact += int(decoded == seq)

        for pos, token in enumerate(seq):
            leaf_ok += int(decoder.decode_leaf(heap, pos) == token)
            leaf_count += 1

        internal_nodes = list(range(1, heap.leaf_base))
        rng.shuffle(internal_nodes)
        for node in internal_nodes[:sample_nodes]:
            decoded_sub = decoder.decode_subheap(heap, node)
            left, right = node_span(node, heap.leaf_base)
            target_sub = [seq[pos] for pos in range(left, min(right, len(seq)))]
            subheap_ok += int(decoded_sub == target_sub)
            subheap_count += 1

            summary = decoder.read_summary(heap, node)
            target_summary = encoder.encode(target_sub).arr[1] if target_sub else NodeState(0, PAD, PAD, 0)
            summary_ok += int(summary == target_summary)
            summary_count += 1

        if idx < 3:
            node = internal_nodes[0]
            examples.append(
                {
                    "tokens": seq,
                    "decoded": decoded,
                    "query_node": node,
                    "query_span": list(node_span(node, heap.leaf_base)),
                    "decoded_subheap": decoder.decode_subheap(heap, node),
                    "root_summary": asdict(heap.arr[1]),
                    "query_summary": asdict(heap.arr[node]),
                }
            )

    return {
        "sequence_exact": seq_exact / max(1, len(seqs)),
        "leaf_acc": leaf_ok / max(1, leaf_count),
        "subheap_exact": subheap_ok / max(1, subheap_count),
        "summary_exact": summary_ok / max(1, summary_count),
        "counts": {
            "sequences": len(seqs),
            "leaf_queries": leaf_count,
            "subheap_queries": subheap_count,
            "summary_queries": summary_count,
        },
        "examples": examples,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    seqs, dataset_meta = load_wmt_sequences(args)
    if not seqs:
        rng = random.Random(args.seed)
        seqs = [random_sequence(rng, args.min_len, args.max_len, args.vocab_limit) for _ in range(args.samples)]

    rng = random.Random(args.seed)
    rng.shuffle(seqs)
    split_a = int(len(seqs) * 0.8)
    split_b = int(len(seqs) * 0.9)
    train = seqs[:split_a]
    test = seqs[split_a:split_b]
    ood = seqs[split_b:]
    max_len = next_power_of_two(args.max_len)

    train_metrics = evaluate_sequences(train, max_len, args.sample_nodes, args.seed)
    test_metrics = evaluate_sequences(test, max_len, args.sample_nodes, args.seed + 1)
    ood_metrics = evaluate_sequences(ood, max_len, args.sample_nodes, args.seed + 2)

    pilot_pass = all(
        metrics[key] == 1.0
        for metrics in (train_metrics, test_metrics, ood_metrics)
        for key in ("sequence_exact", "leaf_acc", "subheap_exact", "summary_exact")
    )

    return {
        "claim": "S1-ECHO-ED-C01",
        "design": {
            "encoder": "token sequence -> ordered TreeHeap leaves -> internal NodeState summaries",
            "decoder": "root length + path-addressed leaf/subheap reads -> token sequence",
            "learned_parameters": 0,
            "uses_target_heap_in_decoder": False,
            "max_len": args.max_len,
            "leaf_count": max_len,
        },
        "dataset": {
            **dataset_meta,
            "samples": len(seqs),
            "train": len(train),
            "test": len(test),
            "ood": len(ood),
            "min_len": args.min_len,
            "max_len": args.max_len,
            "vocab_limit": args.vocab_limit,
        },
        "metrics": {
            "train": {k: v for k, v in train_metrics.items() if k != "examples"},
            "test": {k: v for k, v in test_metrics.items() if k != "examples"},
            "ood": {k: v for k, v in ood_metrics.items() if k != "examples"},
        },
        "examples": {
            "train": train_metrics["examples"],
            "test": test_metrics["examples"],
            "ood": ood_metrics["examples"],
        },
        "pilot_pass": pilot_pass,
        "interpretation": {
            "supported": "Explicit TreeHeap echo encoder/decoder interface is closed." if pilot_pass else "Echo encoder/decoder interface did not close.",
            "not_proved": [
                "not translation",
                "not learned semantic encoding",
                "not compression",
                "not superiority over neural baselines",
                "not noisy channel correction",
            ],
        },
    }


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for split, examples in summary["examples"].items():
            for row in examples:
                fh.write(json.dumps({"split": split, **row}, ensure_ascii=False) + "\n")
    metrics = summary["metrics"]
    readme = f"""# S1 explicit echo encoder/decoder probe

Claim: `S1-ECHO-ED-C01`

This proof implements an explicit TreeHeap echo encoder and decoder.

## Design

- learned parameters: `{summary["design"]["learned_parameters"]}`
- encoder: `{summary["design"]["encoder"]}`
- decoder: `{summary["design"]["decoder"]}`
- uses target heap in decoder: `{summary["design"]["uses_target_heap_in_decoder"]}`

## Metrics

| Split | Sequence exact | Leaf acc | Subheap exact | Summary exact |
|---|---:|---:|---:|---:|
| train | {metrics["train"]["sequence_exact"]:.4f} | {metrics["train"]["leaf_acc"]:.4f} | {metrics["train"]["subheap_exact"]:.4f} | {metrics["train"]["summary_exact"]:.4f} |
| test | {metrics["test"]["sequence_exact"]:.4f} | {metrics["test"]["leaf_acc"]:.4f} | {metrics["test"]["subheap_exact"]:.4f} | {metrics["test"]["summary_exact"]:.4f} |
| ood | {metrics["ood"]["sequence_exact"]:.4f} | {metrics["ood"]["leaf_acc"]:.4f} | {metrics["ood"]["subheap_exact"]:.4f} | {metrics["ood"]["summary_exact"]:.4f} |

## Boundary

This proves the hard/algebraic echo interface closes. It does not prove
translation, learned semantics, compression, or noisy correction.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_echo_encoder_decoder_probe")
    parser.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--vocab-limit", type=int, default=1024)
    parser.add_argument("--sample-nodes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3901)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps({**summary["metrics"], "pilot_pass": summary["pilot_pass"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
