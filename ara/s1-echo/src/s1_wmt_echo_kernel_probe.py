#!/usr/bin/env python3
"""S1 WMT echo kernel probe.

This is not a translation experiment. It asks whether real WMT SentencePiece
token sequences can be written into a TreeHeap-shaped memory and read back.

Kernel design:
  token ids -> shared token embedding
  positions -> fixed heap leaves
  bottom-up shared compose kernel builds internal nodes
  shared leaf decoder reads token ids back from leaf states

Baselines:
  bow_linear: orderless bag-of-words readout
  seq_mlp: flat position-aware MLP
  treeheap_kernel_echo: structured write/compose/read TreeHeap kernel
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0


def read_wmt_english(path: Path, limit_lines: int) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            en = line.split("\t", 1)[0].strip()
            if en:
                out.append(en)
            if len(out) >= limit_lines:
                break
    return out


def build_dataset(args) -> Tuple[np.ndarray, Dict]:
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    texts = read_wmt_english(Path(args.wmt_path), args.scan_lines)
    encoded = []
    for text in texts:
        ids = sp.encode(text, out_type=int)
        ids = [i + 1 for i in ids if 0 <= i < args.vocab_limit - 1]
        if args.min_len <= len(ids) <= args.max_len:
            encoded.append(ids)
        if len(encoded) >= args.samples:
            break
    if len(encoded) < args.samples:
        raise RuntimeError(f"only collected {len(encoded)} examples")
    arr = np.zeros((len(encoded), args.max_len), dtype=np.int64)
    for i, ids in enumerate(encoded):
        arr[i, : len(ids)] = ids
    meta = {
        "wmt_path": args.wmt_path,
        "spm_model": args.spm_model,
        "samples": len(encoded),
        "max_len": args.max_len,
        "min_len": args.min_len,
        "vocab_limit_including_pad": args.vocab_limit,
        "avg_nonpad_len": float(np.mean([len(x) for x in encoded])),
        "examples": texts[:5],
    }
    return arr, meta


def split_data(arr: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(arr))
    n_train = int(0.8 * len(arr))
    n_test = int(0.1 * len(arr))
    train = arr[idx[:n_train]]
    test = arr[idx[n_train:n_train + n_test]]
    ood = arr[idx[n_train + n_test:]]
    return train, test, ood


class BowLinear(nn.Module):
    def __init__(self, vocab: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.out = nn.Linear(vocab, max_len * vocab)
        self.vocab = vocab

    def forward(self, x):
        oh = F.one_hot(x, self.vocab).float().sum(dim=1)
        return self.out(oh).view(x.shape[0], self.max_len, self.vocab)


class SeqMLP(nn.Module):
    def __init__(self, vocab: int, max_len: int, hidden: int):
        super().__init__()
        self.max_len = max_len
        self.vocab = vocab
        self.net = nn.Sequential(
            nn.Linear(max_len * vocab, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_len * vocab),
        )

    def forward(self, x):
        flat = F.one_hot(x, self.vocab).float().reshape(x.shape[0], self.max_len * self.vocab)
        return self.net(flat).view(x.shape[0], self.max_len, self.vocab)


class TreeHeapEcho(nn.Module):
    def __init__(self, vocab: int, max_len: int, dim: int):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.leaf_count = 1
        while self.leaf_count < max_len:
            self.leaf_count *= 2
        self.node_count = 2 * self.leaf_count - 1
        self.leaf_start = self.leaf_count - 1
        self.emb = nn.Embedding(vocab, dim)
        self.compose = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.decoder = nn.Linear(dim, vocab)

    def forward(self, x):
        b = x.shape[0]
        dim = self.emb.embedding_dim
        states = x.new_zeros((b, self.node_count, dim), dtype=torch.float32)
        tok = self.emb(x)
        states[:, self.leaf_start:self.leaf_start + self.max_len, :] = tok
        for i in range(self.leaf_start - 1, -1, -1):
            left = states[:, 2 * i + 1, :]
            right = states[:, 2 * i + 2, :]
            states[:, i, :] = self.compose(torch.cat([left, right], dim=-1))
        leaves = states[:, self.leaf_start:self.leaf_start + self.max_len, :]
        return self.decoder(leaves)


def loss_fn(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)


def train(model, train_arr, args, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_all = torch.tensor(train_arr, dtype=torch.long, device=device)
    trace = []
    n = x_all.shape[0]
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            x = x_all[torch.tensor(idx, dtype=torch.long, device=device)]
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, x)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch in {0, args.epochs // 2, args.epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / seen})
    return model.cpu(), trace


def evaluate(model, arr, device):
    model.to(device)
    x = torch.tensor(arr, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1).cpu().numpy()
    mask = arr != PAD
    token_acc = float((pred[mask] == arr[mask]).mean()) if mask.any() else 0.0
    exact = float(((pred == arr) | ~mask).all(axis=1).mean())
    nonpad = mask.sum(axis=1)
    by_len = {}
    for L in sorted(set(nonpad.tolist())):
        idx = nonpad == L
        by_len[str(int(L))] = float(((pred[idx] == arr[idx]) | ~mask[idx]).all(axis=1).mean())
    return {"token_acc": token_acc, "exact": exact, "by_len_exact": by_len}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--scan-lines", type=int, default=80000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--vocab-limit", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    arr, data_meta = build_dataset(args)
    train_arr, test_arr, ood_arr = split_data(arr, args.seed)
    device = torch.device(args.device)

    models = {}
    traces = {}
    specs = [
        ("bow_linear", BowLinear(args.vocab_limit, args.max_len)),
        ("seq_mlp", SeqMLP(args.vocab_limit, args.max_len, args.hidden)),
        ("treeheap_kernel_echo", TreeHeapEcho(args.vocab_limit, args.max_len, args.dim)),
    ]
    for name, model in specs:
        model, trace = train(model, train_arr, args, device)
        traces[name] = trace
        models[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train": evaluate(model, train_arr, device),
            "test": evaluate(model, test_arr, device),
            "ood": evaluate(model, ood_arr, device),
        }

    tree = models["treeheap_kernel_echo"]["ood"]
    seq = models["seq_mlp"]["ood"]
    supported = bool(tree["token_acc"] >= 0.97 and tree["exact"] >= 0.75 and tree["exact"] >= seq["exact"] - 0.05)
    summary = {
        "claim": "S1-WMT-ECHO-C01",
        "predict": "P-S1-WMT-ECHO01",
        "host": "io.grepcode.cn",
        "device": args.device,
        "dataset": {
            **data_meta,
            "train": len(train_arr),
            "test": len(test_arr),
            "ood": len(ood_arr),
        },
        "kernel_design": "fixed position-to-heap-leaf write, shared bottom-up compose kernel, shared leaf decoder",
        "models": models,
        "traces": traces,
        "pilot_pass": supported,
        "interpretation": {
            "supported": "TreeHeap kernel echo can write/read real WMT SentencePiece short sequences." if supported else "TreeHeap kernel echo did not meet the WMT echo gate.",
            "not_proved": [
                "not translation",
                "not semantic world model",
                "not compression",
                "not long-sequence syntax",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps({"model": k, **r}) for k, rows in traces.items() for r in rows) + "\n", encoding="utf-8")
    (out / "README.md").write_text("# S1 WMT echo kernel probe\n\nReal WMT English SentencePiece short-sequence echo.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
