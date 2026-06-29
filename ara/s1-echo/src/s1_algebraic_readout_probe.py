#!/usr/bin/env python3
"""SPR-034 algebraic readout probe.

SPR-032 proved that a read kernel can collapse a query path from arr[1] to
stop/left/right. SPR-033 proved that a path-addressed TreeHeap can have
algebraic decoders. SPR-034 connects them:

    If a query reaches an internal node, read natural algebraic attributes of
    that subheap instead of forcing an arbitrary checksum label.

This experiment uses real WMT17 English SentencePiece token sequences. It is
not translation. It tests whether internal-node readout becomes easier when the
target is a TreeHeap-shaped attribute:

    length, first token, last token, mod residue, prefix[0], prefix[1].

Models:
    root_query_decoder:
        root state + query node id -> multitask targets.

    routed_state_decoder:
        target node state + query node id -> multitask targets. This assumes
        the route collapse from SPR-032 has selected the node.

    algebraic_oracle:
        deterministic span decoder from the TreeHeap address. This is the
        mathematical upper bound, not a trainable model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PAD = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_wmt_english(path: Path, limit: int) -> Iterable[str]:
    seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if seen >= limit:
                break
            line = line.strip()
            if not line:
                continue
            text = line.split("\t", 1)[0].strip()
            if text:
                seen += 1
                yield text


def collect_sequences(
    wmt_path: Path,
    spm_model: Path,
    samples: int,
    min_len: int,
    max_len: int,
    vocab_limit: int,
    scan_lines: int,
) -> Tuple[List[List[int]], List[str]]:
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model))
    seqs: List[List[int]] = []
    examples: List[str] = []
    for text in read_wmt_english(wmt_path, scan_lines):
        ids = [int(i) + 1 for i in sp.encode(text, out_type=int)]
        ids = [i for i in ids if 0 < i < vocab_limit]
        if min_len <= len(ids) <= max_len:
            seqs.append(ids)
            if len(examples) < 5:
                examples.append(text)
            if len(seqs) >= samples:
                break
    if len(seqs) < samples:
        raise RuntimeError(f"collected {len(seqs)} sequences, need {samples}")
    return seqs, examples


def node_span(node: int, max_len: int) -> Tuple[int, int]:
    start = node
    end = node
    while start < max_len:
        start *= 2
        end = end * 2 + 1
    return start - max_len, end - max_len + 1


def valid_nodes(padded: List[int], max_len: int) -> List[int]:
    out = []
    for node in range(1, max_len * 2):
        s, e = node_span(node, max_len)
        if any(x != PAD for x in padded[s:e]):
            out.append(node)
    return out


def algebraic_targets(
    padded: List[int],
    node: int,
    max_len: int,
    vocab_limit: int,
    residue_buckets: int,
) -> Dict[str, int]:
    s, e = node_span(node, max_len)
    vals = [int(x) for x in padded[s:e] if x != PAD]
    length = len(vals)
    first = vals[0] if vals else PAD
    last = vals[-1] if vals else PAD
    prefix0 = vals[0] if len(vals) > 0 else PAD
    prefix1 = vals[1] if len(vals) > 1 else PAD
    acc = 0
    for i, val in enumerate(vals):
        acc = (acc + (i + 1) * val) % residue_buckets
    return {
        "length": length,
        "first": min(first, vocab_limit - 1),
        "last": min(last, vocab_limit - 1),
        "residue": acc,
        "prefix0": min(prefix0, vocab_limit - 1),
        "prefix1": min(prefix1, vocab_limit - 1),
    }


@dataclass
class Item:
    tokens: torch.Tensor
    query_node: torch.Tensor
    is_internal: torch.Tensor
    length: torch.Tensor
    first: torch.Tensor
    last: torch.Tensor
    residue: torch.Tensor
    prefix0: torch.Tensor
    prefix1: torch.Tensor


class AlgebraicReadDataset(Dataset):
    def __init__(
        self,
        seqs: List[List[int]],
        max_len: int,
        vocab_limit: int,
        residue_buckets: int,
        train_mode: bool,
        seed: int,
        internal_only: bool = False,
    ) -> None:
        self.seqs = seqs
        self.max_len = max_len
        self.vocab_limit = vocab_limit
        self.residue_buckets = residue_buckets
        self.train_mode = train_mode
        self.internal_only = internal_only
        self.rng = random.Random(seed)
        self.eval_index: List[Tuple[int, int]] = []
        if not train_mode:
            for i, seq in enumerate(seqs):
                padded = self._pad(seq)
                for node in valid_nodes(padded, max_len):
                    if internal_only and node >= max_len:
                        continue
                    self.eval_index.append((i, node))

    def _pad(self, seq: List[int]) -> List[int]:
        return seq + [PAD] * (self.max_len - len(seq))

    def __len__(self) -> int:
        return len(self.seqs) if self.train_mode else len(self.eval_index)

    def make_item(self, seq: List[int], node: int) -> Item:
        padded = self._pad(seq)
        y = algebraic_targets(padded, node, self.max_len, self.vocab_limit, self.residue_buckets)
        return Item(
            tokens=torch.tensor(padded, dtype=torch.long),
            query_node=torch.tensor(node, dtype=torch.long),
            is_internal=torch.tensor(node < self.max_len, dtype=torch.bool),
            length=torch.tensor(y["length"], dtype=torch.long),
            first=torch.tensor(y["first"], dtype=torch.long),
            last=torch.tensor(y["last"], dtype=torch.long),
            residue=torch.tensor(y["residue"], dtype=torch.long),
            prefix0=torch.tensor(y["prefix0"], dtype=torch.long),
            prefix1=torch.tensor(y["prefix1"], dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Item:
        if self.train_mode:
            seq = self.seqs[idx]
            padded = self._pad(seq)
            nodes = valid_nodes(padded, self.max_len)
            if self.internal_only:
                nodes = [n for n in nodes if n < self.max_len] or nodes
            node = self.rng.choice(nodes)
        else:
            seq_idx, node = self.eval_index[idx]
            seq = self.seqs[seq_idx]
        return self.make_item(seq, node)


def collate(items: List[Item]) -> Dict[str, torch.Tensor]:
    keys = ["tokens", "query_node", "is_internal", "length", "first", "last", "residue", "prefix0", "prefix1"]
    return {k: torch.stack([getattr(x, k) for x in items]) for k in keys}


class Compose(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class TreeEncoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, dim: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.nodes = max_len * 2 - 1
        self.token_emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.compose = Compose(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        device = tokens.device
        states = torch.zeros(bsz, self.nodes + 1, self.token_emb.embedding_dim, device=device)
        pos = torch.arange(self.max_len, device=device)
        leaves = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for j in range(self.max_len):
            states[:, self.max_len + j, :] = leaves[:, j, :]
        for node in range(self.max_len - 1, 0, -1):
            states[:, node, :] = self.compose(states[:, node * 2, :], states[:, node * 2 + 1, :])
        return states


class MultiTaskHeads(nn.Module):
    def __init__(self, in_dim: int, hidden: int, vocab: int, max_len: int, residue_buckets: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.length = nn.Linear(hidden, max_len + 1)
        self.first = nn.Linear(hidden, vocab)
        self.last = nn.Linear(hidden, vocab)
        self.residue = nn.Linear(hidden, residue_buckets)
        self.prefix0 = nn.Linear(hidden, vocab)
        self.prefix1 = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "length": self.length(h),
            "first": self.first(h),
            "last": self.last(h),
            "residue": self.residue(h),
            "prefix0": self.prefix0(h),
            "prefix1": self.prefix1(h),
        }


class ReadoutModel(nn.Module):
    def __init__(self, mode: str, vocab: int, max_len: int, dim: int, residue_buckets: int) -> None:
        super().__init__()
        assert mode in {"root", "routed"}
        self.mode = mode
        self.max_len = max_len
        self.encoder = TreeEncoder(vocab, max_len, dim)
        self.query_emb = nn.Embedding(max_len * 2, dim)
        self.heads = MultiTaskHeads(dim * 2, dim * 2, vocab, max_len, residue_buckets)

    def forward(self, tokens: torch.Tensor, query_node: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = self.encoder(tokens)
        q = self.query_emb(query_node)
        if self.mode == "root":
            read_state = states[:, 1, :]
        else:
            b = torch.arange(tokens.shape[0], device=tokens.device)
            read_state = states[b, query_node, :]
        return self.heads(torch.cat([read_state, q], dim=-1))


TASKS = ["length", "first", "last", "residue", "prefix0", "prefix1"]


def multitask_loss(logits: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(F.cross_entropy(logits[k], batch[k]) for k in TASKS)


def train_model(
    model: ReadoutModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> List[Dict[str, float]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trace: List[Dict[str, float]] = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = multitask_loss(model(batch["tokens"], batch["query_node"]), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * batch["tokens"].shape[0]
            n += batch["tokens"].shape[0]
        if ep in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": ep, "loss": total / max(1, n)})
    return trace


@torch.no_grad()
def eval_model(model: ReadoutModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    counts = {k: 0 for k in TASKS}
    ok = {k: 0 for k in TASKS}
    counts_i = {k: 0 for k in TASKS}
    ok_i = {k: 0 for k in TASKS}
    exact_all = 0
    total = 0
    exact_i = 0
    total_i = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["tokens"], batch["query_node"])
        all_correct = torch.ones(batch["tokens"].shape[0], dtype=torch.bool, device=device)
        for k in TASKS:
            pred = logits[k].argmax(dim=-1)
            correct = pred == batch[k]
            ok[k] += int(correct.sum().item())
            counts[k] += int(correct.numel())
            is_i = batch["is_internal"]
            ok_i[k] += int((correct & is_i).sum().item())
            counts_i[k] += int(is_i.sum().item())
            all_correct = all_correct & correct
        exact_all += int(all_correct.sum().item())
        total += int(all_correct.numel())
        exact_i += int((all_correct & batch["is_internal"]).sum().item())
        total_i += int(batch["is_internal"].sum().item())
    out = {f"{k}_acc": ok[k] / max(1, counts[k]) for k in TASKS}
    out.update({f"internal_{k}_acc": ok_i[k] / max(1, counts_i[k]) for k in TASKS})
    out["mean_acc"] = sum(out[f"{k}_acc"] for k in TASKS) / len(TASKS)
    out["internal_mean_acc"] = sum(out[f"internal_{k}_acc"] for k in TASKS) / len(TASKS)
    out["exact_all"] = exact_all / max(1, total)
    out["internal_exact_all"] = exact_i / max(1, total_i)
    return out


def oracle_metrics(loader: DataLoader) -> Dict[str, float]:
    # Targets are produced by the algebraic span decoder itself, so this is the
    # deterministic upper bound for the chosen attributes.
    return {**{f"{k}_acc": 1.0 for k in TASKS}, **{f"internal_{k}_acc": 1.0 for k in TASKS}, "mean_acc": 1.0, "internal_mean_acc": 1.0, "exact_all": 1.0, "internal_exact_all": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmt", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_algebraic_readout_probe")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--scan-lines", type=int, default=150000)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--vocab-limit", type=int, default=513)
    ap.add_argument("--residue-buckets", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device.strip() if torch.cuda.is_available() else "cpu")

    seqs, examples = collect_sequences(
        Path(args.wmt),
        Path(args.spm),
        args.samples,
        args.min_len,
        args.max_len,
        args.vocab_limit,
        args.scan_lines,
    )
    random.Random(args.seed).shuffle(seqs)
    n_train = int(len(seqs) * 0.8)
    n_test = int(len(seqs) * 0.1)
    train_seqs = seqs[:n_train]
    test_seqs = seqs[n_train : n_train + n_test]
    ood_seqs = seqs[n_train + n_test :]

    train_ds = AlgebraicReadDataset(train_seqs, args.max_len, args.vocab_limit, args.residue_buckets, True, args.seed)
    test_ds = AlgebraicReadDataset(test_seqs, args.max_len, args.vocab_limit, args.residue_buckets, False, args.seed)
    ood_ds = AlgebraicReadDataset(ood_seqs, args.max_len, args.vocab_limit, args.residue_buckets, False, args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    root = ReadoutModel("root", args.vocab_limit, args.max_len, args.dim, args.residue_buckets)
    routed = ReadoutModel("routed", args.vocab_limit, args.max_len, args.dim, args.residue_buckets)

    root_trace = train_model(root, train_loader, device, args.epochs, args.lr)
    routed_trace = train_model(routed, train_loader, device, args.epochs, args.lr)

    root_test = eval_model(root, test_loader, device)
    root_ood = eval_model(root, ood_loader, device)
    routed_test = eval_model(routed, test_loader, device)
    routed_ood = eval_model(routed, ood_loader, device)
    oracle = oracle_metrics(ood_loader)

    improvements = {k: routed_ood[k] - root_ood[k] for k in routed_ood if isinstance(routed_ood[k], float) and k in root_ood}
    pass_gate = (
        rooted := True
    ) and routed_ood["internal_length_acc"] >= 0.98 and routed_ood["internal_first_acc"] >= 0.85 and routed_ood["internal_last_acc"] >= 0.85 and routed_ood["internal_residue_acc"] >= 0.65 and routed_ood["internal_mean_acc"] - root_ood["internal_mean_acc"] >= 0.20

    summary = {
        "claim": "S1-READ-C02",
        "predict": "P-S1-READ02",
        "host": "io.grepcode.cn",
        "device": str(device),
        "dataset": {
            "wmt_path": args.wmt,
            "spm_model": args.spm,
            "samples": len(seqs),
            "train": len(train_seqs),
            "test": len(test_seqs),
            "ood": len(ood_seqs),
            "min_len": args.min_len,
            "max_len": args.max_len,
            "vocab_limit_including_pad": args.vocab_limit,
            "residue_buckets": args.residue_buckets,
            "examples": examples,
        },
        "targets": TASKS,
        "models": {
            "root_query_decoder": {
                "parameters": sum(p.numel() for p in root.parameters()),
                "trace": root_trace,
                "test": root_test,
                "ood": root_ood,
            },
            "routed_state_decoder": {
                "parameters": sum(p.numel() for p in routed.parameters()),
                "trace": routed_trace,
                "test": routed_test,
                "ood": routed_ood,
            },
            "algebraic_oracle": {
                "parameters": 0,
                "ood": oracle,
                "meaning": "deterministic TreeHeap span decoder upper bound",
            },
        },
        "derived": {
            "routed_minus_root_ood": improvements,
            "parameter_ratio_routed_over_root": sum(p.numel() for p in routed.parameters()) / max(1, sum(p.numel() for p in root.parameters())),
        },
        "pilot_pass": bool(pass_gate),
        "interpretation": {
            "supported": "S1-READ-C02 is supported as a pilot if routed internal algebraic targets beat root bottleneck and key attributes pass gates.",
            "not_proved": [
                "not translation",
                "not semantic phrase meaning",
                "not unsupervised route discovery",
                "not long-sequence syntax",
                "not superiority over Transformer/pointer baselines",
            ],
        },
        "elapsed_sec": round(time.time() - start, 3),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in root_trace:
            f.write(json.dumps({"model": "root_query_decoder", **row}) + "\n")
        for row in routed_trace:
            f.write(json.dumps({"model": "routed_state_decoder", **row}) + "\n")
    (out_dir / "README.md").write_text(
        "# S1 algebraic readout probe\n\n"
        "SPR-034 tests internal-node algebraic readout targets over WMT short BPE sequences.\n\n"
        f"Decision: `S1-READ-C02 -> {'supported pilot' if pass_gate else 'open/rejected pilot'}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
