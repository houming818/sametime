#!/usr/bin/env python3
"""S1 shallow TreeHeap representation probe, numpy-only.

Question:
  Can real short sentences be encoded into a shallow TreeHeap memory and then
  queried by root/subject/object/subheap probes?

This deliberately avoids torch so it can run on ni with only numpy.  The key
S1 idea tested here is copy-by-address:

  input token at position p -> learned soft slot write -> TreeHeap memory slot

If the model learns the shallow write rule, it can copy OOD lexical items into
subject/root/object slots even when those words were never output labels in
training.  Flat linear baselines do not get this copy mechanism.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SLOTS = ["root", "subject", "object"]
ROOT, SUBJECT, OBJECT = 0, 1, 2
SEQ_LEN = 3
N_SLOTS = 3


@dataclass(frozen=True)
class Example:
    tokens: Tuple[str, str, str]
    root: str
    subject: str
    object: str
    split: str
    pattern: str


def build_examples(seed: int) -> Tuple[List[Example], List[Example], List[Example]]:
    rng = random.Random(seed)

    names = ["alice", "bob", "carol", "dave", "erin", "frank"]
    objects = ["door", "key", "book", "map", "box", "cup"]
    verbs = ["opens", "finds", "moves", "carries", "reads", "draws"]

    train: List[Example] = []
    test: List[Example] = []
    ood: List[Example] = []

    # Real-word shallow SVO sentences.  The split holds out combinations, not
    # just examples.
    for si, subj in enumerate(names[:4]):
        for vi, verb in enumerate(verbs[:4]):
            for oi, obj in enumerate(objects[:4]):
                split = "test" if (si + 2 * vi + 3 * oi) % 5 == 0 else "train"
                target = test if split == "test" else train
                target.append(Example((subj, verb, obj), verb, subj, obj, split, "svo_seen_words"))

    # Same bag-of-words, different subject/object direction.
    pair_verbs = ["sees", "helps", "calls"]
    people = ["alice", "bob", "carol", "dave"]
    for i, left in enumerate(people):
        for j, right in enumerate(people):
            if i == j:
                continue
            verb = pair_verbs[(i + j) % len(pair_verbs)]
            split = "test" if (i + j) % 3 == 0 else "train"
            target = test if split == "test" else train
            target.append(Example((left, verb, right), verb, left, right, split, "svo_swapped"))

    # OOD lexical items: the copy mechanism should still place them into slots.
    for subj in names[4:]:
        for verb in verbs[4:]:
            for obj in objects[4:]:
                ood.append(Example((subj, verb, obj), verb, subj, obj, "ood", "new_words"))

    # Short-drama lines: ordinary words, still shallow.
    drama = [
        ("guard", "locks", "gate"),
        ("child", "drops", "toy"),
        ("pilot", "checks", "map"),
        ("artist", "paints", "wall"),
        ("nurse", "brings", "water"),
        ("teacher", "holds", "book"),
    ]
    for idx, (subj, verb, obj) in enumerate(drama):
        if idx < 4:
            train.append(Example((subj, verb, obj), verb, subj, obj, "train", "short_drama"))
        else:
            ood.append(Example((subj, verb, obj), verb, subj, obj, "ood", "short_drama_new_words"))

    rng.shuffle(train)
    rng.shuffle(test)
    rng.shuffle(ood)
    return train, test, ood


def vocab_from(examples: Iterable[Example]) -> Dict[str, int]:
    vocab = sorted({tok for ex in examples for tok in ex.tokens})
    return {tok: idx for idx, tok in enumerate(vocab)}


def encode_targets(examples: Sequence[Example], vocab: Dict[str, int]) -> np.ndarray:
    return np.array([[vocab[ex.root], vocab[ex.subject], vocab[ex.object]] for ex in examples], dtype=np.int64)


def encode_tokens(examples: Sequence[Example], vocab: Dict[str, int]) -> np.ndarray:
    return np.array([[vocab[tok] for tok in ex.tokens] for ex in examples], dtype=np.int64)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class LinearSlotClassifier:
    """Independent linear softmax probes for root/subject/object."""

    def __init__(self, dim: int, vocab_size: int, rng: np.random.Generator):
        self.w = rng.normal(0.0, 0.01, size=(N_SLOTS, dim, vocab_size))
        self.b = np.zeros((N_SLOTS, vocab_size), dtype=np.float64)

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        return np.einsum("bd,sdv->bsv", features, self.w) + self.b[None, :, :]

    def train(self, features: np.ndarray, target: np.ndarray, lr: float, epochs: int) -> List[Dict[str, float]]:
        trace = []
        n = features.shape[0]
        for epoch in range(epochs):
            logits = self.predict_logits(features)
            prob = softmax(logits, axis=-1)
            loss = 0.0
            grad = prob
            for s in range(N_SLOTS):
                loss += -np.log(prob[np.arange(n), s, target[:, s]] + 1e-12).mean()
                grad[np.arange(n), s, target[:, s]] -= 1.0
            loss /= N_SLOTS
            grad /= (n * N_SLOTS)
            dw = np.einsum("bd,bsv->sdv", features, grad)
            db = grad.sum(axis=0)
            self.w -= lr * dw
            self.b -= lr * db
            if epoch in {0, epochs // 2, epochs - 1}:
                trace.append({"epoch": epoch, "loss": float(loss)})
        return trace

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_logits(features).argmax(axis=-1)


class SoftTreeHeapWriter:
    """Learned soft write from sequence positions to TreeHeap slots.

    Parameters are only position->slot logits.  The memory distribution is:

      M[s, v] = sum_p P(slot=s | position=p) * 1[token_p = v]

    This is not a full language model.  It is the smallest S1 check that a
    learned TreeHeap write can copy real sentence tokens into queryable slots.
    """

    def __init__(self, rng: np.random.Generator):
        self.slot_logits = rng.normal(0.0, 0.01, size=(SEQ_LEN, N_SLOTS))

    def slot_probs(self) -> np.ndarray:
        return softmax(self.slot_logits, axis=-1)

    def memory(self, tokens: np.ndarray, vocab_size: int) -> np.ndarray:
        q = self.slot_probs()  # p,s
        n = tokens.shape[0]
        mem = np.zeros((n, N_SLOTS, vocab_size), dtype=np.float64)
        for p in range(SEQ_LEN):
            mem[np.arange(n), :, tokens[:, p]] += q[p][None, :]
        return mem

    def train(self, tokens: np.ndarray, target: np.ndarray, vocab_size: int, lr: float, epochs: int) -> List[Dict[str, float]]:
        trace = []
        n = tokens.shape[0]
        for epoch in range(epochs):
            q = self.slot_probs()
            mem = self.memory(tokens, vocab_size)
            loss = 0.0
            gq = np.zeros_like(q)
            for i in range(n):
                for s in range(N_SLOTS):
                    denom = mem[i, s, target[i, s]] + 1e-12
                    loss += -np.log(denom)
                    for p in range(SEQ_LEN):
                        if tokens[i, p] == target[i, s]:
                            gq[p, s] += -(1.0 / denom)
            loss /= (n * N_SLOTS)
            gq /= (n * N_SLOTS)

            # softmax gradient per position.
            grad_logits = np.zeros_like(q)
            for p in range(SEQ_LEN):
                dot = float((gq[p] * q[p]).sum())
                grad_logits[p] = q[p] * (gq[p] - dot)
            self.slot_logits -= lr * grad_logits

            if epoch in {0, epochs // 2, epochs - 1}:
                trace.append({"epoch": epoch, "loss": float(loss)})
        return trace

    def predict(self, tokens: np.ndarray, vocab_size: int) -> np.ndarray:
        return self.memory(tokens, vocab_size).argmax(axis=-1)


def bow_features(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    x = np.zeros((tokens.shape[0], vocab_size), dtype=np.float64)
    for p in range(tokens.shape[1]):
        x[np.arange(tokens.shape[0]), tokens[:, p]] += 1.0
    return x


def seq_features(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    x = np.zeros((tokens.shape[0], SEQ_LEN * vocab_size), dtype=np.float64)
    for p in range(tokens.shape[1]):
        x[np.arange(tokens.shape[0]), p * vocab_size + tokens[:, p]] = 1.0
    return x


def evaluate_predictions(pred: np.ndarray, target: np.ndarray, examples: Sequence[Example], inv_vocab: Dict[int, str]) -> Dict:
    exact_rows = pred == target
    slot_acc = exact_rows.mean(axis=0)
    exact = exact_rows.all(axis=1).mean()
    verb_object = (exact_rows[:, ROOT] & exact_rows[:, OBJECT]).mean()
    swapped_idx = [i for i, ex in enumerate(examples) if ex.pattern == "svo_swapped"]
    out = {
        "exact": float(exact),
        "root_acc": float(slot_acc[ROOT]),
        "subject_acc": float(slot_acc[SUBJECT]),
        "object_acc": float(slot_acc[OBJECT]),
        "verb_object_subheap_acc": float(verb_object),
        "samples": [],
    }
    if swapped_idx:
        out["swapped_exact"] = float(exact_rows[swapped_idx].all(axis=1).mean())
    for i, ex in enumerate(examples[:5]):
        out["samples"].append(
            {
                "sentence": " ".join(ex.tokens),
                "gold": {slot: inv_vocab[int(target[i, s])] for s, slot in enumerate(SLOTS)},
                "pred": {slot: inv_vocab[int(pred[i, s])] for s, slot in enumerate(SLOTS)},
            }
        )
    return out


def evaluate_model(kind: str, model, train, test, ood, vocab: Dict[str, int]) -> Dict:
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    result = {}
    for split_name, examples in [("train", train), ("test", test), ("ood", ood)]:
        tokens = encode_tokens(examples, vocab)
        target = encode_targets(examples, vocab)
        if kind == "bow_linear":
            pred = model.predict(bow_features(tokens, vocab_size))
        elif kind == "seq_linear":
            pred = model.predict(seq_features(tokens, vocab_size))
        elif kind == "soft_treeheap":
            pred = model.predict(tokens, vocab_size)
        else:
            raise ValueError(kind)
        result[split_name] = evaluate_predictions(pred, target, examples, inv_vocab)
    if kind == "soft_treeheap":
        result["slot_write_probs_by_position"] = model.slot_probs().tolist()
        result["slot_write_argmax_by_position"] = [SLOTS[int(i)] for i in model.slot_probs().argmax(axis=1)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.8)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test, ood = build_examples(args.seed)
    vocab = vocab_from([*train, *test, *ood])
    vocab_size = len(vocab)

    train_tokens = encode_tokens(train, vocab)
    train_target = encode_targets(train, vocab)

    bow = LinearSlotClassifier(vocab_size, vocab_size, rng)
    bow_trace = bow.train(bow_features(train_tokens, vocab_size), train_target, lr=args.lr, epochs=args.epochs)

    seq = LinearSlotClassifier(SEQ_LEN * vocab_size, vocab_size, rng)
    seq_trace = seq.train(seq_features(train_tokens, vocab_size), train_target, lr=args.lr, epochs=args.epochs)

    tree = SoftTreeHeapWriter(rng)
    tree_trace = tree.train(train_tokens, train_target, vocab_size, lr=args.lr, epochs=args.epochs)

    models = {
        "bow_linear": evaluate_model("bow_linear", bow, train, test, ood, vocab),
        "seq_linear": evaluate_model("seq_linear", seq, train, test, ood, vocab),
        "soft_treeheap": evaluate_model("soft_treeheap", tree, train, test, ood, vocab),
    }

    s1_pass = (
        models["soft_treeheap"]["test"]["exact"] >= 0.95
        and models["soft_treeheap"]["ood"]["exact"] >= 0.95
        and models["soft_treeheap"]["ood"]["verb_object_subheap_acc"] >= 0.95
    )

    summary = {
        "claim": "S1-C30",
        "predict": "P-S1-REAL-SHALLOW-01",
        "seed": args.seed,
        "epochs": args.epochs,
        "dataset": {
            "kind": "curated real-word short sentences",
            "train_examples": len(train),
            "test_examples": len(test),
            "ood_examples": len(ood),
            "vocab_size": vocab_size,
            "slots": SLOTS,
            "note": "OOD contains lexical items unseen as train outputs; TreeHeap copy-by-address can still place them into slots.",
        },
        "models": models,
        "pilot_pass": bool(s1_pass),
        "interpretation": {
            "supported": "A learned shallow TreeHeap write can encode real short sentences into queryable root/subject/object slots."
            if s1_pass
            else "The learned shallow TreeHeap write did not meet the S1 pilot gate.",
            "not_proved": [
                "not WMT translation",
                "not full syntax",
                "not deep TreeHeap",
                "not Transformer comparison",
                "curated short sentences are still a small S1 pilot",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }

    trace = {
        "bow_linear": bow_trace,
        "seq_linear": seq_trace,
        "soft_treeheap": tree_trace,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps({"model": k, **row}, ensure_ascii=False) for k, rows in trace.items() for row in rows)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "dataset.json").write_text(
        json.dumps(
            {
                "train": [asdict(ex) for ex in train],
                "test": [asdict(ex) for ex in test],
                "ood": [asdict(ex) for ex in ood],
                "vocab": vocab,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# S1 shallow TreeHeap probe",
                "",
                "Question: can real short sentences be encoded into a shallow TreeHeap memory and queried by probes?",
                "",
                "This pilot uses numpy-only training on curated real-word short sentences.",
                "",
                "Artifacts:",
                "",
                "- `summary.json`: metrics and claim decision",
                "- `dataset.json`: train/test/OOD sentence split",
                "- `trace.jsonl`: training loss trace",
                "",
                "This is a first S1 proof, not a WMT or full syntax proof.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
