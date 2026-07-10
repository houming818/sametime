#!/usr/bin/env python3
"""S1 mask-kernel structure extraction proof.

The task is not echo.  Given a masked sentence such as:

    I cooked [MASK]

the model outputs a probability bucket over possible fillers.  Some
context-object pairs are held out, so the model must transfer from related
contexts rather than only memorize exact pairs.

This is a controlled toy proof for SPR-049 / S1-MASK-KERNEL-C01.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_TOKENS = ["I", "ate", "cooked", "drank", "poured", "visited", "toured", "[MASK]", "[PAD]"]
OBJECTS = [
    "rice",
    "noodles",
    "apple",
    "bread",
    "fish",
    "water",
    "milk",
    "juice",
    "Paris",
    "Beijing",
    "museum",
    "park",
    "medicine",
    "book",
    "shirt",
    "car",
]
IN_IDX = {x: i for i, x in enumerate(INPUT_TOKENS)}
OBJ_IDX = {x: i for i, x in enumerate(OBJECTS)}

OBJECT_CLASS = {
    "rice": "food",
    "noodles": "food",
    "apple": "food",
    "bread": "food",
    "fish": "food",
    "water": "drink",
    "milk": "drink",
    "juice": "drink",
    "Paris": "place",
    "Beijing": "place",
    "museum": "place",
    "park": "place",
    "medicine": "buyable",
    "book": "buyable",
    "shirt": "buyable",
    "car": "buyable",
}

CONTEXT_CLASS = {
    "ate": "food",
    "cooked": "food",
    "drank": "drink",
    "poured": "drink",
    "visited": "place",
    "toured": "place",
}


@dataclass(frozen=True)
class Example:
    verb: str
    obj: str
    split: str
    note: str

    def tokens(self) -> list[int]:
        return [IN_IDX["I"], IN_IDX[self.verb], IN_IDX["[MASK]"], IN_IDX["[PAD]"]]

    def target(self) -> int:
        return OBJ_IDX[self.obj]

    def bucket(self) -> str:
        return CONTEXT_CLASS[self.verb]


def build_examples() -> list[Example]:
    rows = [
        # Food: enough overlap to transfer between eat and cook.
        ("ate", "rice", "train", "seen"),
        ("ate", "noodles", "train", "seen"),
        ("ate", "apple", "train", "seen"),
        ("ate", "bread", "train", "seen"),
        ("ate", "fish", "train", "seen"),
        ("cooked", "rice", "train", "bridge"),
        ("cooked", "fish", "train", "bridge"),
        ("cooked", "noodles", "heldout", "transfer_food"),
        ("cooked", "apple", "heldout", "transfer_food"),
        ("cooked", "bread", "heldout", "transfer_food"),
        # Drink: pour and drink overlap partly.
        ("drank", "water", "train", "seen"),
        ("drank", "milk", "train", "seen"),
        ("drank", "juice", "train", "seen"),
        ("poured", "water", "train", "bridge"),
        ("poured", "juice", "train", "bridge"),
        ("poured", "milk", "heldout", "transfer_drink"),
        # Place: visit and tour overlap partly.
        ("visited", "Paris", "train", "seen"),
        ("visited", "museum", "train", "seen"),
        ("visited", "park", "train", "seen"),
        ("toured", "Beijing", "train", "bridge"),
        ("toured", "park", "train", "bridge"),
        ("toured", "Paris", "heldout", "transfer_place"),
        ("toured", "museum", "heldout", "transfer_place"),
    ]
    return [Example(*row) for row in rows]


def split_examples(rows: list[Example], split: str) -> list[Example]:
    return [r for r in rows if r.split == split]


def tensorize(rows: list[Example], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([r.tokens() for r in rows], dtype=torch.long, device=device)
    y = torch.tensor([r.target() for r in rows], dtype=torch.long, device=device)
    return x, y


class BoWMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(len(INPUT_TOKENS), dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, len(OBJECTS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x).mean(dim=1)
        return self.net(h)


class FlatSeqMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(len(INPUT_TOKENS), dim)
        self.net = nn.Sequential(nn.Linear(dim * 4, hidden), nn.Tanh(), nn.Linear(hidden, len(OBJECTS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x).reshape(x.shape[0], -1)
        return self.net(h)


class TreeHeapMaskKernel(nn.Module):
    """4-leaf masked sentence -> 7-node TreeHeap -> mask probability bucket."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(len(INPUT_TOKENS), dim)
        self.path = nn.Parameter(torch.randn(7, dim) * 0.02)
        self.compose = nn.Sequential(nn.Linear(dim * 2, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        self.read = nn.Sequential(nn.Linear(dim * 4, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        self.obj_emb = nn.Parameter(torch.randn(len(OBJECTS), dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(len(OBJECTS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf = self.emb(x)
        nodes = [None] * 7
        # Complete binary heap: leaves are nodes 3,4,5,6.
        for k in range(4):
            nodes[3 + k] = leaf[:, k, :] + self.path[3 + k]
        nodes[1] = self.compose(torch.cat([nodes[3], nodes[4]], dim=-1)) + self.path[1]
        nodes[2] = self.compose(torch.cat([nodes[5], nodes[6]], dim=-1)) + self.path[2]
        nodes[0] = self.compose(torch.cat([nodes[1], nodes[2]], dim=-1)) + self.path[0]
        mask_state = nodes[5]
        right_state = nodes[2]
        state = self.read(torch.cat([nodes[0], nodes[1], right_state, mask_state], dim=-1))
        return state @ self.obj_emb.t() + self.bias


def train_model(
    model: nn.Module,
    train_rows: list[Example],
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model.to(device)
    x, y = tensorize(train_rows, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for epoch in range(epochs + 1):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if epoch in {0, 1, 2, 5, 10, 25, 50, 100, 250, 500, epochs}:
            trace.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch == epochs:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.cpu(), trace


def train_shuffled(rows: list[Example], seed: int) -> list[Example]:
    rng = random.Random(seed)
    objs = [r.obj for r in rows]
    rng.shuffle(objs)
    return [Example(r.verb, obj, r.split, "shuffled_target") for r, obj in zip(rows, objs)]


def probs_for_model(model: nn.Module, rows: list[Example]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x, _ = tensorize(rows, torch.device("cpu"))
        return F.softmax(model(x), dim=-1)


def pair_memory_probs(train_rows: list[Example], rows: list[Example], alpha: float = 0.05) -> torch.Tensor:
    global_counts = Counter(r.obj for r in train_rows)
    by_verb: dict[str, Counter] = defaultdict(Counter)
    for r in train_rows:
        by_verb[r.verb][r.obj] += 1
    out = []
    for r in rows:
        counts = by_verb.get(r.verb, Counter())
        scores = []
        for obj in OBJECTS:
            scores.append(counts[obj] + alpha * global_counts[obj] + alpha)
        t = torch.tensor(scores, dtype=torch.float32)
        out.append(t / t.sum())
    return torch.stack(out)


def entropy(p: torch.Tensor) -> float:
    return float((-(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)).mean())


def metrics(name: str, probs: torch.Tensor, rows: list[Example], topk: int = 5) -> dict[str, object]:
    ranks = []
    top1 = 0
    top5 = 0
    purity_vals = []
    examples = []
    for i, r in enumerate(rows):
        p = probs[i]
        order = torch.argsort(p, descending=True).tolist()
        gold = r.target()
        rank = order.index(gold) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        top5 += int(rank <= topk)
        bucket = r.bucket()
        top_objs = [OBJECTS[j] for j in order[:topk]]
        purity = sum(1 for obj in top_objs if OBJECT_CLASS[obj] == bucket) / topk
        purity_vals.append(purity)
        examples.append(
            {
                "verb": r.verb,
                "masked": f"I {r.verb} [MASK]",
                "gold": r.obj,
                "bucket": bucket,
                "rank": rank,
                "top5": top_objs,
                "top5_probs": [float(p[j]) for j in order[:topk]],
                "top5_bucket_purity": purity,
            }
        )
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    return {
        "name": name,
        "n": len(rows),
        "top1": top1 / len(rows),
        "top5": top5 / len(rows),
        "mrr": mrr,
        "mean_rank": sum(ranks) / len(ranks),
        "entropy": entropy(probs),
        "bucket_purity_top5": sum(purity_vals) / len(purity_vals),
        "examples": examples,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = build_examples()
    train_rows = split_examples(rows, "train")
    heldout_rows = split_examples(rows, "heldout")

    tree = TreeHeapMaskKernel(args.dim, args.hidden)
    tree, tree_trace = train_model(tree, train_rows, device, args.epochs, args.lr, args.seed + 1)

    bow = BoWMLP(args.dim, args.hidden)
    bow, bow_trace = train_model(bow, train_rows, device, args.epochs, args.lr, args.seed + 2)

    flat = FlatSeqMLP(args.dim, args.hidden)
    flat, flat_trace = train_model(flat, train_rows, device, args.epochs, args.lr, args.seed + 3)

    shuffled_rows = train_shuffled(train_rows, args.seed + 4)
    shuffled_tree = TreeHeapMaskKernel(args.dim, args.hidden)
    shuffled_tree, shuffled_trace = train_model(shuffled_tree, shuffled_rows, device, args.epochs, args.lr, args.seed + 5)

    pair_train = pair_memory_probs(train_rows, train_rows)
    pair_heldout = pair_memory_probs(train_rows, heldout_rows)

    model_metrics = {
        "pair_memory": {
            "parameters": 0,
            "train": metrics("pair_memory_train", pair_train, train_rows),
            "heldout": metrics("pair_memory_heldout", pair_heldout, heldout_rows),
        },
        "treeheap_mask_kernel": {
            "parameters": parameter_count(tree),
            "train": metrics("treeheap_train", probs_for_model(tree, train_rows), train_rows),
            "heldout": metrics("treeheap_heldout", probs_for_model(tree, heldout_rows), heldout_rows),
        },
        "bow_mlp": {
            "parameters": parameter_count(bow),
            "train": metrics("bow_train", probs_for_model(bow, train_rows), train_rows),
            "heldout": metrics("bow_heldout", probs_for_model(bow, heldout_rows), heldout_rows),
        },
        "flat_seq_mlp": {
            "parameters": parameter_count(flat),
            "train": metrics("flat_train", probs_for_model(flat, train_rows), train_rows),
            "heldout": metrics("flat_heldout", probs_for_model(flat, heldout_rows), heldout_rows),
        },
        "shuffled_treeheap": {
            "parameters": parameter_count(shuffled_tree),
            "train_on_shuffled": metrics("shuffled_train", probs_for_model(shuffled_tree, train_rows), train_rows),
            "heldout": metrics("shuffled_heldout", probs_for_model(shuffled_tree, heldout_rows), heldout_rows),
        },
    }

    th = model_metrics["treeheap_mask_kernel"]["heldout"]
    pair = model_metrics["pair_memory"]["heldout"]
    shuf = model_metrics["shuffled_treeheap"]["heldout"]
    bow_h = model_metrics["bow_mlp"]["heldout"]
    flat_h = model_metrics["flat_seq_mlp"]["heldout"]

    pass_checks = {
        "treeheap_mrr_beats_pair_memory": th["mrr"] > pair["mrr"] + args.min_mrr_gap,
        "treeheap_mrr_beats_shuffled": th["mrr"] > shuf["mrr"] + args.min_mrr_gap,
        "treeheap_bucket_purity_beats_shuffled": th["bucket_purity_top5"] > shuf["bucket_purity_top5"] + args.min_purity_gap,
        "treeheap_top5_nontrivial": th["top5"] >= args.min_top5,
        "treeheap_entropy_bucket_not_delta": th["entropy"] >= args.min_entropy,
    }
    baseline_warnings = {
        "bow_mrr_beats_or_matches_treeheap": bow_h["mrr"] >= th["mrr"],
        "flat_mrr_beats_or_matches_treeheap": flat_h["mrr"] >= th["mrr"],
    }
    if all(pass_checks.values()) and not any(baseline_warnings.values()):
        decision = "supported pilot"
    elif all(pass_checks.values()):
        decision = "weak positive / baseline-contested"
    elif pass_checks["treeheap_mrr_beats_shuffled"] and pass_checks["treeheap_bucket_purity_beats_shuffled"]:
        decision = "mixed pilot"
    else:
        decision = "open / failed pilot"

    summary = {
        "claim": "S1-MASK-KERNEL-C01",
        "predict": "P-S1-MASK-KERNEL01",
        "host": args.host_label,
        "device": str(device),
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "dim": args.dim,
            "hidden": args.hidden,
            "input_tokens": INPUT_TOKENS,
            "objects": OBJECTS,
            "object_class": OBJECT_CLASS,
            "context_class": CONTEXT_CLASS,
            "train_examples": [asdict(r) for r in train_rows],
            "heldout_examples": [asdict(r) for r in heldout_rows],
            "gold_classes_used_for_training": False,
        },
        "models": model_metrics,
        "traces": {
            "treeheap_mask_kernel": tree_trace,
            "bow_mlp": bow_trace,
            "flat_seq_mlp": flat_trace,
            "shuffled_treeheap": shuffled_trace,
        },
        "pass_checks": pass_checks,
        "baseline_warnings": baseline_warnings,
        "pilot_pass": decision in {"supported pilot", "weak positive / baseline-contested"},
        "decision": decision,
        "interpretation": {
            "supported_if": "TreeHeap mask kernel beats pair memory and shuffled control on held-out MRR and top-k bucket purity.",
            "not_proved": [
                "not WMT translation",
                "not natural-language understanding",
                "not proof over Transformer",
                "not unsupervised ontology discovery",
            ],
        },
    }
    return summary


def write_outputs(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for model, rows in summary["traces"].items():
            for row in rows:
                f.write(json.dumps({"model": model, **row}, ensure_ascii=False) + "\n")
    topk = {
        name: data["heldout"]["examples"]
        for name, data in summary["models"].items()
        if "heldout" in data
    }
    (out_dir / "topk_examples.json").write_text(json.dumps(topk, indent=2, ensure_ascii=False), encoding="utf-8")

    th = summary["models"]["treeheap_mask_kernel"]["heldout"]
    pair = summary["models"]["pair_memory"]["heldout"]
    shuf = summary["models"]["shuffled_treeheap"]["heldout"]
    bow = summary["models"]["bow_mlp"]["heldout"]
    flat = summary["models"]["flat_seq_mlp"]["heldout"]
    readme = f"""# S1 Mask Kernel Structure Extraction Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`
Device: `{summary['device']}`

## Result

decision: `{summary['decision']}`
pilot_pass: `{summary['pilot_pass']}`

```text
treeheap heldout MRR/top5/purity/entropy = {th['mrr']:.4f} / {th['top5']:.4f} / {th['bucket_purity_top5']:.4f} / {th['entropy']:.4f}
pair     heldout MRR/top5/purity/entropy = {pair['mrr']:.4f} / {pair['top5']:.4f} / {pair['bucket_purity_top5']:.4f} / {pair['entropy']:.4f}
shuffled heldout MRR/top5/purity/entropy = {shuf['mrr']:.4f} / {shuf['top5']:.4f} / {shuf['bucket_purity_top5']:.4f} / {shuf['entropy']:.4f}
bow      heldout MRR/top5/purity/entropy = {bow['mrr']:.4f} / {bow['top5']:.4f} / {bow['bucket_purity_top5']:.4f} / {bow['entropy']:.4f}
flat     heldout MRR/top5/purity/entropy = {flat['mrr']:.4f} / {flat['top5']:.4f} / {flat['bucket_purity_top5']:.4f} / {flat['entropy']:.4f}
```

## Boundary

This is a controlled masked-corpus proof. Gold bucket classes are used only for
audit metrics, not for training. This is not WMT translation or natural-language
understanding.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_mask_kernel_structure_extraction_probe")
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--min-mrr-gap", type=float, default=0.02)
    parser.add_argument("--min-purity-gap", type=float, default=0.05)
    parser.add_argument("--min-top5", type=float, default=0.75)
    parser.add_argument("--min-entropy", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host-label", default="local")
    args = parser.parse_args()

    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["baseline_warnings"], indent=2, ensure_ascii=False))
    print(f"decision={summary['decision']}")
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
