#!/usr/bin/env python3
"""S1 corpus-trained world-coordinate + structured TreeHeap kernel probe.

Unlike the frozen-embedding probe, this experiment trains a small embedding
space from a local compound-word corpus.  The target coordinate system is
therefore derived from corpus co-occurrence rather than from an external model.

The TreeHeap model is also deliberately constrained:

  left token  -> left child
  right token -> right child
  root        -> compose kernel(left_child, right_child)

This tests whether a TreeHeap-shaped kernel can learn composition in the small
corpus coordinate system without using an unconstrained all-node reader.
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
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Compound:
    left: str
    right: str
    target: str
    split: str
    family: str


def compounds() -> List[Compound]:
    rows = [
        ("foot", "ball", "football", "train", "ball"),
        ("basket", "ball", "basketball", "train", "ball"),
        ("base", "ball", "baseball", "train", "ball"),
        ("snow", "ball", "snowball", "test", "ball"),
        ("hand", "ball", "handball", "ood", "ball"),
        ("rain", "coat", "raincoat", "train", "coat"),
        ("over", "coat", "overcoat", "train", "coat"),
        ("trench", "coat", "trenchcoat", "test", "coat"),
        ("book", "shelf", "bookshelf", "train", "storage"),
        ("book", "case", "bookcase", "train", "storage"),
        ("cup", "board", "cupboard", "test", "board"),
        ("key", "board", "keyboard", "ood", "board"),
        ("black", "board", "blackboard", "train", "board"),
        ("white", "board", "whiteboard", "test", "board"),
        ("note", "book", "notebook", "train", "book"),
        ("guide", "book", "guidebook", "train", "book"),
        ("text", "book", "textbook", "test", "book"),
        ("scrap", "book", "scrapbook", "ood", "book"),
        ("sun", "light", "sunlight", "train", "light"),
        ("moon", "light", "moonlight", "train", "light"),
        ("star", "light", "starlight", "test", "light"),
        ("flash", "light", "flashlight", "ood", "light"),
        ("bed", "room", "bedroom", "train", "room"),
        ("bath", "room", "bathroom", "train", "room"),
        ("class", "room", "classroom", "test", "room"),
        ("tooth", "brush", "toothbrush", "train", "brush"),
        ("hair", "brush", "hairbrush", "test", "brush"),
        ("paint", "brush", "paintbrush", "ood", "brush"),
        ("head", "phone", "headphone", "train", "phone"),
        ("ear", "phone", "earphone", "test", "phone"),
        ("door", "bell", "doorbell", "train", "door"),
        ("door", "way", "doorway", "ood", "door"),
    ]
    return [Compound(*r) for r in rows]


def build_corpus(rows: Sequence[Compound], repeat: int, seed: int) -> List[List[str]]:
    """Build a tiny local corpus with co-occurrence signals.

    The corpus intentionally uses only our curated rows and family words.  It is
    not external model distillation.
    """

    rng = random.Random(seed)
    corpus: List[List[str]] = []
    for r in rows:
        templates = [
            [r.target, "means", r.left, r.right],
            [r.target, "uses", r.left, "and", r.right],
            [r.left, r.right, "forms", r.target],
            [r.target, "belongs", "to", r.family],
            [r.left, "combines", "with", r.right, "as", r.target],
            [r.family, "contains", r.target],
        ]
        for _ in range(repeat):
            sent = list(rng.choice(templates))
            if rng.random() < 0.15:
                rng.shuffle(sent)
            corpus.append(sent)

    # Add family contrast sentences so right-side substructures have reusable
    # neighborhoods.
    by_family: Dict[str, List[str]] = {}
    for r in rows:
        by_family.setdefault(r.family, []).append(r.target)
    for fam, targets in by_family.items():
        for _ in range(max(4, repeat // 2)):
            corpus.append([fam, "family", *targets[:5]])
    rng.shuffle(corpus)
    return corpus


def make_vocab(corpus: Sequence[Sequence[str]]) -> Dict[str, int]:
    words = sorted({w for s in corpus for w in s})
    return {w: i for i, w in enumerate(words)}


def skipgram_pairs(corpus: Sequence[Sequence[str]], vocab: Dict[str, int], window: int) -> Tuple[np.ndarray, np.ndarray]:
    centers, contexts = [], []
    for sent in corpus:
        ids = [vocab[w] for w in sent]
        for i, c in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    centers.append(c)
                    contexts.append(ids[j])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.in_emb.weight, std=0.05)
        nn.init.normal_(self.out_emb.weight, std=0.05)

    def forward(self, center, pos_context, neg_context):
        v = self.in_emb(center)
        pos = self.out_emb(pos_context)
        neg = self.out_emb(neg_context)
        pos_loss = F.logsigmoid(torch.sum(v * pos, dim=-1))
        neg_loss = F.logsigmoid(-torch.einsum("bd,bkd->bk", v, neg)).sum(dim=-1)
        return -(pos_loss + neg_loss).mean()


def train_sgns(corpus, vocab, args, device) -> Tuple[np.ndarray, List[Dict]]:
    centers, contexts = skipgram_pairs(corpus, vocab, args.window)
    rng = np.random.default_rng(args.seed)
    counts = np.ones(len(vocab), dtype=np.float64)
    for c in contexts:
        counts[c] += 1
    noise = counts ** 0.75
    noise /= noise.sum()
    model = SGNS(len(vocab), args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.embed_lr)
    trace = []
    n = len(centers)
    for epoch in range(args.embed_epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            c = torch.tensor(centers[idx], dtype=torch.long, device=device)
            p = torch.tensor(contexts[idx], dtype=torch.long, device=device)
            neg_np = rng.choice(len(vocab), size=(len(idx), args.negatives), p=noise)
            neg = torch.tensor(neg_np, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = model(c, p, neg)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        if epoch in {0, args.embed_epochs // 2, args.embed_epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / n})
    vec = model.in_emb.weight.detach().cpu().numpy()
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec.astype(np.float32), trace


def split(rows, name):
    return [r for r in rows if r.split == name]


def arrays(rows, vocab, emb):
    left = np.stack([emb[vocab[r.left]] for r in rows])
    right = np.stack([emb[vocab[r.right]] for r in rows])
    target = np.stack([emb[vocab[r.target]] for r in rows])
    return left.astype(np.float32), right.astype(np.float32), target.astype(np.float32)


def norm_np(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def candidate_matrix(rows, vocab, emb):
    names = [r.target for r in rows]
    mat = norm_np(np.stack([emb[vocab[n]] for n in names]).astype(np.float32))
    return names, mat


def metrics(pred, target, candidates, names, gold):
    pred = norm_np(pred)
    target = norm_np(target)
    cos = np.sum(pred * target, axis=1)
    sims = pred @ candidates.T
    tops = sims.argmax(axis=1)
    top_names = [names[i] for i in tops]
    ranks = []
    for row, g in zip(sims, gold):
        order = np.argsort(-row)
        ranks.append(int(np.where(order == names.index(g))[0][0]) + 1)
    return {
        "mean_cosine": float(cos.mean()),
        "top1": float(np.mean([a == b for a, b in zip(top_names, gold)])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "examples": [
            {"gold": g, "pred_top1": p, "rank": int(r), "cosine": float(c)}
            for g, p, r, c in zip(gold[:8], top_names[:8], ranks[:8], cos[:8])
        ],
    }


class ConcatMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))

    def forward(self, l, r):
        return F.normalize(self.net(torch.cat([l, r, l * r, torch.abs(l - r)], dim=-1)), dim=-1)


class StructuredTreeHeapKernel(nn.Module):
    """Constrained TreeHeap kernel: write children, compose root."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.left_write = nn.Linear(dim, dim, bias=False)
        self.right_write = nn.Linear(dim, dim, bias=False)
        self.compose = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, l, r):
        left_child = self.left_write(l)
        right_child = self.right_write(r)
        root = self.compose(torch.cat([left_child, right_child, left_child * right_child, torch.abs(left_child - right_child)], dim=-1))
        return F.normalize(root, dim=-1)


def train_composer(model, arr, args, device):
    l, r, t = [torch.tensor(x, dtype=torch.float32, device=device) for x in arr]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.compose_lr, weight_decay=args.weight_decay)
    trace = []
    for epoch in range(args.compose_epochs):
        opt.zero_grad()
        pred = model(l, r)
        loss = (1.0 - torch.sum(pred * t, dim=-1)).mean()
        loss.backward()
        opt.step()
        if epoch in {0, args.compose_epochs // 2, args.compose_epochs - 1}:
            trace.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    return model.cpu(), trace


def predict(model, arr):
    l, r, _ = arr
    with torch.no_grad():
        return model(torch.tensor(l), torch.tensor(r)).numpy()


def pcount(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--repeat", type=int, default=24)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--embed-epochs", type=int, default=220)
    ap.add_argument("--embed-lr", type=float, default=0.025)
    ap.add_argument("--compose-epochs", type=int, default=900)
    ap.add_argument("--compose-lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows = compounds()
    corpus = build_corpus(rows, args.repeat, args.seed)
    vocab = make_vocab(corpus)
    emb, emb_trace = train_sgns(corpus, vocab, args, device)
    train_rows, test_rows, ood_rows = split(rows, "train"), split(rows, "test"), split(rows, "ood")
    tr, te, oo = arrays(train_rows, vocab, emb), arrays(test_rows, vocab, emb), arrays(ood_rows, vocab, emb)
    names, cand = candidate_matrix(rows, vocab, emb)

    models = {}
    traces = {"sgns": emb_trace}

    def eval_model(name, pred_train, pred_test, pred_ood, params):
        models[name] = {
            "parameters": params,
            "train": metrics(pred_train, tr[2], cand, names, [r.target for r in train_rows]),
            "test": metrics(pred_test, te[2], cand, names, [r.target for r in test_rows]),
            "ood": metrics(pred_ood, oo[2], cand, names, [r.target for r in ood_rows]),
        }

    eval_model("vector_add", norm_np(tr[0] + tr[1]), norm_np(te[0] + te[1]), norm_np(oo[0] + oo[1]), 0)

    mlp = ConcatMLP(args.dim, args.hidden)
    mlp, traces["concat_mlp"] = train_composer(mlp, tr, args, device)
    eval_model("concat_mlp", predict(mlp, tr), predict(mlp, te), predict(mlp, oo), pcount(mlp))

    th = StructuredTreeHeapKernel(args.dim, args.hidden)
    th, traces["structured_treeheap_kernel"] = train_composer(th, tr, args, device)
    eval_model("structured_treeheap_kernel", predict(th, tr), predict(th, te), predict(th, oo), pcount(th))

    tree_ood = models["structured_treeheap_kernel"]["ood"]["mean_cosine"]
    add_ood = models["vector_add"]["ood"]["mean_cosine"]
    mlp_ood = models["concat_mlp"]["ood"]["mean_cosine"]
    supported = bool(tree_ood > add_ood and tree_ood >= mlp_ood - 0.02)

    summary = {
        "claim": "S1-WM-C02",
        "predict": "P-S1-WM02",
        "host": "io.grepcode.cn",
        "seed": args.seed,
        "coordinate_source": {
            "kind": "local SGNS corpus embedding",
            "external_model": False,
            "distillation_guard": "no pretrained embedding model is used; coordinates are trained from the local curated compound corpus",
            "vocab_size": len(vocab),
            "corpus_sentences": len(corpus),
            "skipgram_pairs": int(len(skipgram_pairs(corpus, vocab, args.window)[0])),
        },
        "dataset": {
            "train": len(train_rows),
            "test": len(test_rows),
            "ood": len(ood_rows),
            "rows": [asdict(r) for r in rows],
        },
        "kernel_design": "structured TreeHeap kernel: left write -> left child, right write -> right child, root compose kernel",
        "models": models,
        "traces": traces,
        "pilot_pass": supported,
        "interpretation": {
            "supported": "Structured TreeHeap kernel beats vector_add on OOD in the local corpus coordinate system." if supported else "Structured TreeHeap kernel does not beat vector_add under the current local corpus coordinate setup.",
            "not_proved": [
                "not WMT",
                "not full language world model",
                "not superiority over all MLP/Transformer variants",
                "small curated corpus may encode the task too directly or too weakly",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "dataset.json").write_text(json.dumps({"rows": [asdict(r) for r in rows], "corpus": corpus, "vocab": vocab}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for model, rows2 in traces.items():
            for row in rows2:
                f.write(json.dumps({"model": model, **row}, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(
        "# S1 corpus embedding kernel probe\n\nCoordinates are trained from a local SGNS corpus, not from a pretrained embedding model.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
