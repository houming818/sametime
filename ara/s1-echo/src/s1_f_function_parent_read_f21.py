"""Matched FOLD-topology and frozen-parent READ bridge on short WMT windows.

This is not a translation benchmark.  It separates a fixed-capacity FOLD
operator from a fresh linear parent readout:

  4 token leaves -> candidate FOLD -> parent -> token 5

Each candidate has the same embedding size and exactly three distinct local
transform modules.  After encoder pretraining, the encoder is frozen and a
new linear readout is trained from the parent only.  A mean-leaf linear
readout is recorded as the no-FOLD control.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as nnf


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")
TOPOLOGIES = {
    "left_deep": (((0, 1), 2), 3),
    "balanced": ((0, 1), (2, 3)),
    "right_deep": (0, (1, (2, 3))),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_windows(path: Path, scan_lines: int, vocab_size: int, max_examples: int, seed: int):
    sentences = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle):
            if scan_lines and line_no >= scan_lines:
                break
            if "\t" not in line:
                continue
            left, right = line.rstrip("\n").split("\t", 1)
            left_cjk = sum("\u4e00" <= c <= "\u9fff" for c in left)
            right_cjk = sum("\u4e00" <= c <= "\u9fff" for c in right)
            tokens = [token.lower() for token in TOKEN_RE.findall(right if left_cjk > right_cjk else left)]
            if len(tokens) >= 5:
                sentences.append(tokens)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    cut = max(1, int(0.8 * len(sentences)))
    train_sentences, test_sentences = sentences[:cut], sentences[cut:]
    counts = Counter(token for sentence in train_sentences for token in sentence)
    vocab = ["<pad>", "<unk>"] + [token for token, _ in counts.most_common(vocab_size - 2)]
    stoi = {token: index for index, token in enumerate(vocab)}

    def make_rows(source):
        rows = []
        for sentence in source:
            ids = [stoi.get(token, 1) for token in sentence]
            for offset in range(len(ids) - 4):
                rows.append((ids[offset : offset + 4], ids[offset + 4]))
                if max_examples and len(rows) >= max_examples:
                    return rows
        return rows

    return make_rows(train_sentences), make_rows(test_sentences), {
        "sentences": len(sentences),
        "train_sentences": len(train_sentences),
        "vocab": len(vocab),
    }


class FoldEncoder(nn.Module):
    def __init__(self, kind: str, vocab_size: int, dim: int):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, dim)
        # Every arm owns the same number and shape of transform parameters.
        self.steps = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    @staticmethod
    def local_step(value: torch.Tensor, transform: nn.Linear) -> torch.Tensor:
        return nnf.normalize(value + 0.5 * torch.tanh(transform(value)), dim=-1)

    def _fold(self, node, leaves: torch.Tensor, index: int):
        if isinstance(node, int):
            return leaves[:, node], index
        children = []
        for child in node:
            value, index = self._fold(child, leaves, index)
            children.append(value)
        parent = self.local_step(torch.stack(children, dim=1).mean(dim=1), self.steps[index])
        return parent, index + 1

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        leaves = self.embedding(tokens)
        if self.kind == "bag_3step":
            parent = leaves.mean(dim=1)
            for transform in self.steps:
                parent = self.local_step(parent, transform)
            return parent
        parent, used = self._fold(TOPOLOGIES[self.kind], leaves, 0)
        if used != 3:
            raise RuntimeError(f"expected three FOLD applications, got {used}")
        return parent

    def mean_leaf(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens).mean(dim=1)


def tensor_rows(rows, device: torch.device):
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
        torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
    )


def fit_head(features, labels, vocab_size: int, dim: int, steps: int, seed: int):
    seed_all(seed)
    head = nn.Linear(dim, vocab_size).to(features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    for _ in range(steps):
        index = torch.randint(len(features), (min(256, len(features)),), generator=generator, device="cpu").to(features.device)
        loss = nnf.cross_entropy(head(features[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
    return head


def metrics(logits: torch.Tensor, labels: torch.Tensor):
    nll = float(nnf.cross_entropy(logits, labels).item())
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "top1": float((logits.argmax(dim=-1) == labels).float().mean().item()),
        "finite": bool(torch.isfinite(logits).all().item()),
    }


def run_arm(kind, train_x, train_y, test_x, test_y, vocab_size, args, device):
    seed_all(args.seed)
    encoder = FoldEncoder(kind, vocab_size, args.dim).to(device)
    pretrain_head = nn.Linear(args.dim, vocab_size).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(pretrain_head.parameters()), lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    for _ in range(args.pretrain_steps):
        index = torch.randint(len(train_x), (min(args.batch_size, len(train_x)),), generator=generator, device="cpu").to(device)
        loss = nnf.cross_entropy(pretrain_head(encoder.encode(train_x[index])), train_y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(pretrain_head.parameters()), 5.0)
        optimizer.step()

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_parent = encoder.encode(train_x)
        test_parent = encoder.encode(test_x)
        train_mean = encoder.mean_leaf(train_x)
        test_mean = encoder.mean_leaf(test_x)
        pretrain_test = metrics(pretrain_head(test_parent), test_y)

    parent_head = fit_head(train_parent, train_y, vocab_size, args.dim, args.read_steps, args.seed + 1001)
    mean_head = fit_head(train_mean, train_y, vocab_size, args.dim, args.read_steps, args.seed + 1001)
    with torch.no_grad():
        parent = metrics(parent_head(test_parent), test_y)
        mean_leaf = metrics(mean_head(test_mean), test_y)
    return {
        "arm": kind,
        "parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "pretrain_parent": pretrain_test,
        "fresh_parent_read": parent,
        "fresh_mean_leaf_read": mean_leaf,
        "parent_vs_mean_leaf_nll": parent["nll"] - mean_leaf["nll"],
        "parent_retention_nll": parent["nll"] - pretrain_test["nll"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scan-lines", type=int, default=50_000)
    parser.add_argument("--max-examples", type=int, default=12_000)
    parser.add_argument("--test-examples", type=int, default=3_000)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--pretrain-steps", type=int, default=1_500)
    parser.add_argument("--read-steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_rows, test_rows, data_meta = read_windows(Path(args.data), args.scan_lines, args.vocab, args.max_examples, args.seed)
    test_rows = test_rows[: args.test_examples]
    if len(train_rows) < 100 or len(test_rows) < 100:
        raise RuntimeError(f"insufficient data train={len(train_rows)} test={len(test_rows)}")
    train_x, train_y = tensor_rows(train_rows, device)
    test_x, test_y = tensor_rows(test_rows, device)
    arms = ["left_deep", "balanced", "right_deep", "bag_3step"]
    rows = []
    for arm in arms:
        row = run_arm(arm, train_x, train_y, test_x, test_y, data_meta["vocab"], args, device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "claim": "S1-F-1/F-2",
        "boundary": "Fixed-topology FOLD and frozen-parent readout only; no READ routing, translation, or generation claim.",
        "seed": args.seed,
        "device": str(device),
        "data": args.data,
        "data_meta": data_meta,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "pretrain_steps": args.pretrain_steps,
        "read_steps": args.read_steps,
        "arms": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "arms": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
