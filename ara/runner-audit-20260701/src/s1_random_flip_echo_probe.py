#!/usr/bin/env python3
"""Modified SPR-043: random flip with learned inverse gate.

Original: 100% of samples are flipped. Model always learns to unflip.
Fixed:     Random ~50% of samples are flipped. Model must learn:
          1. Detect whether input was flipped (gate probability)
          2. Apply inverse route when flipped, identity when not
          3. Decode to original tokens

This tests whether TreeHeap S1 can handle MIXED flip/no-flip inputs
without a given task flag.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0
UNK = 1
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,!?;:]")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def detok(tokens: Iterable[str]) -> str:
    return " ".join(x for x in tokens if x not in ("<pad>", "<unk>"))

def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1,
                d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return d[n][m]


class TreeNode:
    __slots__ = ("value", "left", "right")
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def build_tree(tokens: list[int]) -> TreeNode:
    if not tokens:
        return TreeNode(value=PAD)
    if len(tokens) == 1:
        return TreeNode(value=tokens[0])
    mid = len(tokens) // 2
    return TreeNode(left=build_tree(tokens[:mid]), right=build_tree(tokens[mid:]))

def flip_tree(node: TreeNode, depth: int | None = None) -> TreeNode:
    if node.value is not None:
        return TreeNode(value=node.value)
    if depth is not None and depth <= 0:
        return clone_tree(node)
    next_depth = None if depth is None else depth - 1
    assert node.left is not None and node.right is not None
    return TreeNode(left=flip_tree(node.right, next_depth), right=flip_tree(node.left, next_depth))

def clone_tree(node: TreeNode) -> TreeNode:
    if node.value is not None:
        return TreeNode(value=node.value)
    return TreeNode(left=clone_tree(node.left), right=clone_tree(node.right))

def leaves(node: TreeNode) -> list[int]:
    if node.value is not None:
        return [node.value]
    assert node.left is not None and node.right is not None
    return leaves(node.left) + leaves(node.right)

def treeheap_full_flip(tokens: list[int]) -> list[int]:
    return leaves(flip_tree(build_tree(tokens), depth=None))


class RandomFlipInverseGate(nn.Module):
    """Learned inverse gate: detect flip + apply inverse route when needed."""

    def __init__(self, vocab: int, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.decoder = nn.Linear(dim, vocab)

        # Inverse route: length-conditioned routing matrix (unflip)
        self.inv_route_logits = nn.Parameter(torch.zeros(max_len + 1, max_len, max_len))

        # Flip detector gate: binary classifier over pooled tree state
        self.gate_proj = nn.Linear(dim, 1)

        with torch.no_grad():
            self.inv_route_logits.normal_(mean=0.0, std=0.02)

    def forward(self, observed: torch.Tensor, lengths: torch.Tensor):
        B, L = observed.shape
        leaf = self.emb(observed)  # [B, L, D]

        # Flip detection gate
        gate_logit = self.gate_proj(leaf.mean(dim=1))  # [B, 1]
        gate_prob = torch.sigmoid(gate_logit)  # [B, 1]  -- prob that input IS flipped

        # Inverse route (unflip)
        inv_route = F.softmax(self.inv_route_logits[lengths], dim=-1)  # [B, L, L]
        inv_state = torch.einsum("bij,bjd->bid", inv_route, leaf)  # [B, L, D]

        # Identity route (pass through)
        id_state = leaf  # [B, L, D]

        # Soft gate: mix inv_state and id_state
        state = gate_prob.unsqueeze(-1) * inv_state + (1 - gate_prob.unsqueeze(-1)) * id_state

        return self.decoder(state), state, inv_route, gate_prob


class NoFlipBaseline(nn.Module):
    """Flat baseline: just decode from embedding, no route."""

    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.decoder = nn.Linear(dim, vocab)

    def forward(self, observed: torch.Tensor, lengths: torch.Tensor):
        return self.decoder(self.emb(observed))


def loss_fn(model_output, target, target_state, lengths, args):
    if isinstance(model_output, tuple):
        logits, state, inv_route, gate_prob = model_output
    else:
        logits = model_output

    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)

    if isinstance(model_output, tuple):
        # State loss
        mask = (target != PAD).float().unsqueeze(-1)
        state_loss = (((state - target_state.detach()) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

        # Entropy regularizer on route
        entropy = -(inv_route.clamp_min(1e-9) * inv_route.clamp_min(1e-9).log()).sum(dim=-1)
        pos = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
        active = (pos < lengths.unsqueeze(1)).float()
        entropy_loss = (entropy * active).sum() / active.sum().clamp_min(1.0)

        return ce + args.state_weight * state_loss + args.entropy_weight * entropy_loss
    else:
        return ce


def make_dataset(args, stoi, itos, raw_sentences):
    rng = np.random.default_rng(args.seed)
    encoded = []
    raw = []
    for sent in raw_sentences:
        ids = [stoi.get(tok, UNK) for tok in sent]
        if args.drop_unk and UNK in ids:
            continue
        encoded.append(ids)
        raw.append(sent)
        if len(encoded) >= args.samples:
            break

    arr = np.zeros((len(encoded), args.max_len), dtype=np.int64)
    obs = np.zeros((len(encoded), args.max_len), dtype=np.int64)
    flipped_mask = np.zeros((len(encoded),), dtype=np.int64)
    lengths = np.zeros((len(encoded),), dtype=np.int64)

    flip_rng = np.random.default_rng(args.seed + 1)
    examples = []
    n_flipped = 0

    for i, ids in enumerate(encoded):
        flipped = treeheap_full_flip(ids)
        should_flip = flip_rng.random() < args.flip_ratio

        lengths[i] = len(ids)
        arr[i, :len(ids)] = ids
        if should_flip:
            obs[i, :len(flipped)] = flipped
            flipped_mask[i] = 1
            n_flipped += 1
        else:
            obs[i, :len(ids)] = ids
            flipped_mask[i] = 0

        if len(examples) < args.examples:
            examples.append({
                "target": detok([itos[x] for x in ids]),
                "observed": detok([itos.get(x, "<unk>") for x in (flipped if should_flip else ids)]),
                "flipped": bool(should_flip),
                "length": len(ids),
            })

    meta = {
        "wmt_path": args.wmt_path,
        "samples": len(encoded),
        "n_flipped": n_flipped,
        "flip_ratio": args.flip_ratio,
        "max_len": args.max_len,
        "vocab_size": len(stoi),
        "examples": examples,
    }
    return arr, obs, flipped_mask, lengths, stoi, itos, meta


def train_model(model, train_data, args, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    target, observed, lengths = [torch.tensor(x, dtype=torch.long, device=device) for x in train_data]
    rng = np.random.default_rng(args.seed)
    trace = []
    n = target.shape[0]
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, args.batch):
            sel = torch.tensor(order[start:start + args.batch], dtype=torch.long, device=device)
            opt.zero_grad()
            model_out = model(observed[sel], lengths[sel])
            target_state = model.emb(target[sel])
            loss = loss_fn(model_out, target[sel], target_state, lengths[sel], args)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * sel.numel()
            seen += sel.numel()
        if epoch in {0, 1, 2, 5, 10, args.epochs // 2, args.epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / seen})
    return model.cpu(), trace


def evaluate(model, split, flips, itos, device, max_examples=12):
    model.to(device)
    target_np, observed_np, lengths_np = split
    flip_np = flips
    target = torch.tensor(target_np, dtype=torch.long, device=device)
    observed = torch.tensor(observed_np, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths_np, dtype=torch.long, device=device)

    with torch.no_grad():
        model_out = model(observed, lengths)
        if isinstance(model_out, tuple):
            logits, state, inv_route, gate_prob = model_out
            gate_pred = (gate_prob > 0.5).long().cpu().numpy().flatten()
            gate_accuracy = float((gate_pred == flip_np).mean())
        else:
            logits = model_out
            gate_accuracy = 0.0
            gate_pred = np.zeros_like(flip_np)

        pred_np = logits.argmax(dim=-1).cpu().numpy()

    mask = target_np != PAD
    token_acc = float((pred_np[mask] == target_np[mask]).mean()) if mask.any() else 0.0
    exact_rows = ((pred_np == target_np) | ~mask).all(axis=1)
    exact = float(exact_rows.mean())

    # Per-group breakdown
    flipped_idx = flip_np == 1
    unflipped_idx = flip_np == 0

    flip_exact = float(exact_rows[flipped_idx.astype(bool)].mean()) if flipped_idx.astype(bool).any() else 0.0
    unflip_exact = float(exact_rows[unflipped_idx.astype(bool)].mean()) if unflipped_idx.astype(bool).any() else 0.0

    flip_mask = mask & flipped_idx[:, None]
    unflip_mask = mask & unflipped_idx[:, None]
    flip_token = float((pred_np[flip_mask] == target_np[flip_mask]).mean()) if flip_mask.any() else 0.0
    unflip_token = float((pred_np[unflip_mask] == target_np[unflip_mask]).mean()) if unflip_mask.any() else 0.0

    examples = []
    for i, L in enumerate(lengths_np.tolist()):
        if len(examples) < max_examples:
            examples.append({
                "length": L,
                "flipped": int(flip_np[i]),
                "gate_prob": float(gate_prob[i].item()) if isinstance(model_out, tuple) else 0.0,
                "observed": detok([itos.get(x, "<unk>") for x in observed_np[i, :L].tolist()]),
                "restored": detok([itos.get(x, "<unk>") for x in pred_np[i, :L].tolist()]),
                "target": detok([itos.get(x, "<unk>") for x in target_np[i, :L].tolist()]),
                "exact": pred_np[i, :L].tolist() == target_np[i, :L].tolist(),
            })

    return {
        "exact_match": exact,
        "token_acc": token_acc,
        "flipped_exact": flip_exact,
        "unflipped_exact": unflip_exact,
        "flipped_token_acc": flip_token,
        "unflipped_token_acc": unflip_token,
        "flip_detection_acc": gate_accuracy,
        "n_flipped": int(flips.sum()),
        "n_unflipped": int((1 - flips).sum()),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/random_flip_probe")
    parser.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--scan-lines", type=int, default=50000)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--flip-ratio", type=float, default=0.5, help="Ratio of samples to flip")
    parser.add_argument("--drop-unk", action="store_true", default=True)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Load sentences
    with open(args.wmt_path) as f:
        lines = [line.strip() for line in f.readlines()[: args.scan_lines]]
    en_sents = []
    for line in lines:
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[1].strip() in ("en", "en-US", "en-GB"):
                en_sents.append(tokenize(parts[2]))
            elif len(parts) > 2:
                en_sents.append(tokenize(parts[2]))
            elif len(parts) > 1:
                en_sents.append(tokenize(parts[1]))
            else:
                en_sents.append(tokenize(parts[0]))
        else:
            en_sents.append(tokenize(line))

    # Filter by length
    en_sents = [s for s in en_sents if args.min_len <= len(s) <= args.max_len]

    # Build vocab
    counter = Counter()
    for s in en_sents:
        counter.update(s)
    stoi = {w: i + 2 for i, (w, _) in enumerate(counter.most_common())}
    stoi["<pad>"] = PAD
    stoi["<unk>"] = UNK
    itos = {v: k for k, v in stoi.items()}

    # Make dataset with RANDOM flipping
    arr, obs, flipped_mask, lengths, stoi, itos, meta = make_dataset(args, stoi, itos, en_sents)

    # Split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(arr))
    n_train = int(0.8 * len(arr))
    n_test = int(0.1 * len(arr))
    splits = {}
    for name, sel in {
        "train": idx[:n_train],
        "test": idx[n_train:n_train + n_test],
        "ood": idx[n_train + n_test:],
    }.items():
        splits[name] = {
            "data": (arr[sel], obs[sel], lengths[sel]),
            "flips": flipped_mask[sel],
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # RandomFlip model
    model = RandomFlipInverseGate(len(stoi), args.max_len, args.dim)
    train_data = splits["train"]["data"]
    model, trace = train_model(model, train_data, args, device)

    # Evaluate
    results = {}
    for split_name in ["train", "test", "ood"]:
        s = splits[split_name]
        results[split_name] = evaluate(model, s["data"], s["flips"], itos, device, args.examples)

    results["train"]["flipped"] = int(splits["train"]["flips"].sum())
    results["train"]["unflipped"] = int((1 - splits["train"]["flips"]).sum())
    results["test"]["flipped"] = int(splits["test"]["flips"].sum())
    results["test"]["unflipped"] = int((1 - splits["test"]["flips"]).sum())
    results["ood"]["flipped"] = int(splits["ood"]["flips"].sum())
    results["ood"]["unflipped"] = int((1 - splits["ood"]["flips"]).sum())

    # Baseline
    baseline = NoFlipBaseline(len(stoi), args.dim)
    bl_trace = None
    bl_results = {}
    for split_name in ["train", "test", "ood"]:
        s = splits[split_name]
        bl_data = (s["data"][0], s["data"][1], s["data"][2])
        bl_model, bl_trace = train_model(baseline, bl_data, args, device)
        bl_results[split_name] = evaluate(bl_model, bl_data, s["flips"], itos, device)

    # Pilot pass: flip detection must work AND flipped recovery must beat baseline
    ood = results["ood"]
    bl_ood = bl_results["ood"]
    gate_ok = ood["flip_detection_acc"] >= 0.75
    recovery_ok = ood["flipped_exact"] >= bl_ood["flipped_exact"] + 0.1
    pilot_pass = gate_ok and recovery_ok

    summary = {
        "experiment": "random_flip_inverse_gate",
        "claim": "S1-RANDOM-FLIP-C01",
        "predict": "P-S1-RANDOM-FLIP",
        "seed": args.seed,
        "flip_ratio": args.flip_ratio,
        "device": device,
        "dataset": meta,
        "pilot_pass": pilot_pass,
        "pass_gates": {
            "flip_detection_acc_gte_0.75": gate_ok,
            "flipped_recovery_beats_baseline": recovery_ok,
            "flip_detection_acc": ood["flip_detection_acc"],
            "flipped_exact": ood["flipped_exact"],
            "baseline_flipped_exact": bl_ood["flipped_exact"],
        },
        "treeheap_random_flip": {
            "trace": trace,
            "results": results,
        },
        "no_flip_baseline": {
            "results": bl_results,
        },
        "interpretation": {
            "supported": "Random flip inverse gate learns detect-unflip-recover from mixed flip/no-flip inputs.",
            "not_proved": ["not WMT translation", "not natural trigger discovery", "not full syntax"],
        },
    }

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (out_path / "trace.jsonl").open("w") as f:
        for row in trace:
            f.write(json.dumps({"model": "random_flip", **row}, ensure_ascii=False) + "\n")

    (out_path / "command.sh").write_text(
        "python3 src/s1_random_flip_echo_probe.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items()) + "\n"
    )

    print(json.dumps({
        "pilot_pass": pilot_pass,
        "gate_ok": gate_ok,
        "recovery_ok": recovery_ok,
        "flip_detection_acc": ood["flip_detection_acc"],
        "flipped_exact": ood["flipped_exact"],
        "unflipped_exact": ood["unflipped_exact"],
        "baseline_flipped_exact": bl_ood["flipped_exact"],
        "n_flipped": ood["n_flipped"],
        "n_unflipped": ood["n_unflipped"],
    }, indent=2))


if __name__ == "__main__":
    main()
