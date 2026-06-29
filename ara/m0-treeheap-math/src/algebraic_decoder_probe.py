#!/usr/bin/env python3
"""SPR-033 algebraic decoder probe for TreeHeap internal state.

This probe tests a corrected reading of SPR-032:

    An internal node state may already contain information, but it is a
    high-dimensional algebraic/hash-like state. We should not expect humans to
    read it directly. We need decoders.

The key question here is not "can an MLP learn a decoder?" but:

    Does the TreeHeap algebra itself provide decoders?

We model a finite-field TreeHeap where leaves are written into path-addressed
slots. An internal node state is the modular sum of descendant slots. The
result is a latent vector over Z_p; it is not visually interpretable. However,
because TreeHeap has address/path structure, algebraic decoders can recover:

    - projection to a path/subheap
    - decompose into left/right child states
    - residue/mod summaries over ordered leaves
    - conjugate/mirror symmetry

This is a mathematical toy, not a language result.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PRIME = 1_000_003
PAD = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def mod_hash_vec(token_id: int, dim: int, prime: int = PRIME) -> np.ndarray:
    """Deterministic primitive vector in a finite field."""
    rng = np.random.default_rng(token_id * 1_000_003 + 97)
    return rng.integers(1, prime, size=(dim,), dtype=np.int64)


@dataclass
class EncodedTree:
    tokens: List[int]
    state: np.ndarray


class FiniteFieldTreeHeap:
    """A path-addressed finite-field TreeHeap.

    State shape is [max_len, channel_dim]. Each leaf path owns one slot. The
    flattened array is the internal node state. Internal compose is modular
    addition over descendant path slots.
    """

    def __init__(self, max_len: int, channel_dim: int, vocab: int, prime: int = PRIME) -> None:
        assert max_len > 0 and (max_len & (max_len - 1) == 0), "max_len must be power of two"
        self.max_len = max_len
        self.channel_dim = channel_dim
        self.vocab = vocab
        self.prime = prime
        self.primitive_to_token = {
            tuple(mod_hash_vec(tok, self.channel_dim, self.prime).tolist()): tok
            for tok in range(1, vocab)
        }

    def zero(self) -> np.ndarray:
        return np.zeros((self.max_len, self.channel_dim), dtype=np.int64)

    def encode(self, tokens: List[int]) -> EncodedTree:
        st = self.zero()
        for i, tok in enumerate(tokens[: self.max_len]):
            if tok != PAD:
                st[i, :] = mod_hash_vec(tok, self.channel_dim, self.prime)
        return EncodedTree(tokens=tokens[: self.max_len], state=st)

    def span(self, node: int) -> Tuple[int, int]:
        start = node
        end = node
        while start < self.max_len:
            start *= 2
            end = end * 2 + 1
        return start - self.max_len, end - self.max_len + 1

    def project(self, state: np.ndarray, node: int) -> np.ndarray:
        """Algebraic projection decoder: keep only one subheap span."""
        s, e = self.span(node)
        out = self.zero()
        out[s:e, :] = state[s:e, :]
        return out

    def compose(self, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        """Closure under modular addition of disjoint or overlapping components."""
        return (left_state + right_state) % self.prime

    def decompose(self, state: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """Algebraic decompose decoder for a node's left/right child subheaps."""
        left = self.project(state, node * 2)
        right = self.project(state, node * 2 + 1)
        return left, right

    def mirror(self, state: np.ndarray) -> np.ndarray:
        """Conjugate decoder: reverse the ordered leaf slots."""
        return state[::-1, :].copy()

    def residue(self, state: np.ndarray, modulus: int) -> List[np.ndarray]:
        """Mod decoder: aggregate leaves by address residue class."""
        buckets = [np.zeros((self.channel_dim,), dtype=np.int64) for _ in range(modulus)]
        for i in range(self.max_len):
            buckets[i % modulus] = (buckets[i % modulus] + state[i]) % self.prime
        return buckets

    def norm0(self, state: np.ndarray) -> int:
        """Count non-empty leaf slots. This is an algebraic length decoder."""
        return int(np.count_nonzero(np.any(state != 0, axis=1)))

    def decode_token_from_slot(self, state: np.ndarray, slot: int) -> int:
        """Nearest exact primitive-vector decoder for one leaf slot."""
        v = state[slot]
        if not np.any(v):
            return PAD
        return self.primitive_to_token.get(tuple(v.tolist()), -1)

    def decode_ordered_tokens(self, state: np.ndarray) -> List[int]:
        return [self.decode_token_from_slot(state, i) for i in range(self.max_len)]

    def bag_checksum(self, state: np.ndarray) -> int:
        """A natural algebraic checksum over all channels and slots."""
        return int(state.sum() % self.prime)


def random_tokens(rng: random.Random, max_len: int, vocab: int) -> List[int]:
    n = rng.randint(1, max_len)
    toks = [rng.randint(1, vocab - 1) for _ in range(n)]
    return toks + [PAD] * (max_len - n)


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs((a - b) % PRIME)
    return float(diff.sum())


def run_probe(samples: int, max_len: int, channel_dim: int, vocab: int, seed: int) -> Dict:
    rng = random.Random(seed)
    th = FiniteFieldTreeHeap(max_len=max_len, channel_dim=channel_dim, vocab=vocab)
    trace = []

    projection_ok = 0
    decompose_ok = 0
    mirror_ok = 0
    residue_ok = 0
    length_ok = 0
    ordered_decode_ok = 0
    checksum_stable_ok = 0

    for idx in range(samples):
        toks = random_tokens(rng, max_len, vocab)
        enc = th.encode(toks)
        st = enc.state

        valid_nodes = []
        for candidate in range(1, max_len):
            cs, ce = th.span(candidate)
            if any(tok != PAD for tok in toks[cs:ce]):
                valid_nodes.append(candidate)
        node = rng.choice(valid_nodes)
        s, e = th.span(node)

        projected = th.project(st, node)
        manual = th.zero()
        manual[s:e, :] = st[s:e, :]
        projection_ok += int(rel_error(projected, manual) == 0.0)

        left, right = th.decompose(st, node)
        recomposed = th.compose(left, right)
        decompose_ok += int(rel_error(recomposed, projected) == 0.0)

        mirrored = th.mirror(st)
        # Conjugate law: mirror(project(node)) equals project(mirror node) after
        # reversing the slot order. At state level, mirror is an involution.
        mirror_ok += int(rel_error(th.mirror(mirrored), st) == 0.0)

        residue = th.residue(st, 2)
        even_manual = np.zeros((channel_dim,), dtype=np.int64)
        odd_manual = np.zeros((channel_dim,), dtype=np.int64)
        for i in range(max_len):
            if i % 2 == 0:
                even_manual = (even_manual + st[i]) % PRIME
            else:
                odd_manual = (odd_manual + st[i]) % PRIME
        residue_ok += int(np.array_equal(residue[0], even_manual) and np.array_equal(residue[1], odd_manual))

        expected_len = sum(1 for x in toks if x != PAD)
        length_ok += int(th.norm0(st) == expected_len)

        decoded = th.decode_ordered_tokens(st)
        ordered_decode_ok += int(decoded == toks)

        # Checksum is not an arbitrary learned target; it is induced by the
        # finite-field representation and therefore stable under projection.
        checksum_full = th.bag_checksum(projected)
        checksum_children = th.bag_checksum(recomposed)
        checksum_stable_ok += int(checksum_full == checksum_children)

        if idx < 5:
            trace.append(
                {
                    "tokens": toks,
                    "query_node": node,
                    "span": [s, e],
                    "decoded_tokens": decoded,
                    "subheap_length": th.norm0(projected),
                    "subheap_checksum": checksum_full,
                }
            )

    def rate(x: int) -> float:
        return x / max(1, samples)

    pass_gate = all(
        rate(x) == 1.0
        for x in [
            projection_ok,
            decompose_ok,
            mirror_ok,
            residue_ok,
            length_ok,
            ordered_decode_ok,
            checksum_stable_ok,
        ]
    )

    return {
        "claim": "M0-DEC-C01",
        "predict": "P-MATH-DEC01",
        "settings": {
            "samples": samples,
            "max_len": max_len,
            "channel_dim": channel_dim,
            "vocab": vocab,
            "field": f"Z_{PRIME}",
            "seed": seed,
        },
        "metrics": {
            "projection_exact": rate(projection_ok),
            "decompose_recompose_exact": rate(decompose_ok),
            "mirror_involution_exact": rate(mirror_ok),
            "mod_residue_exact": rate(residue_ok),
            "length_decode_exact": rate(length_ok),
            "ordered_token_decode_exact": rate(ordered_decode_ok),
            "checksum_stability_exact": rate(checksum_stable_ok),
        },
        "pilot_pass": pass_gate,
        "interpretation": {
            "supported": "If pilot_pass is true, finite-field TreeHeap internal state has algebraic decoders for path, subheap, mod residue, mirror, and ordered leaf slots.",
            "not_proved": [
                "not language semantics",
                "not learned decoder superiority",
                "not WMT translation",
                "not compression optimality",
                "not robustness under noisy learned encoders",
            ],
        },
        "trace_examples": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/m0-treeheap-math/evidence/algebraic_decoder_probe")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--channel-dim", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=257)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = run_probe(args.samples, args.max_len, args.channel_dim, args.vocab, args.seed)
    summary["elapsed_sec"] = round(time.time() - start, 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["trace_examples"]:
            f.write(json.dumps(row) + "\n")
    (out / "README.md").write_text(
        "# Algebraic decoder probe\n\n"
        "SPR-033 tests whether TreeHeap internal states have mathematical decoders "
        "before asking for learned semantic decoders.\n\n"
        f"Decision: `M0-DEC-C01 -> {'supported pilot' if summary['pilot_pass'] else 'open/rejected pilot'}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
