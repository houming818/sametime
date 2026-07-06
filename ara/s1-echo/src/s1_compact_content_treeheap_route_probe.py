#!/usr/bin/env python3
"""SPR-047 compact content-aware TreeHeap route proof.

SPR-046 proved content-aware routing with dense vocab-count summaries, but the
proof materialized q/arr[i]/arr[left]/arr[right] as 1024D dense tensors.

This version keeps the same mechanism boundary while making it scalable:

    leaf token id -> fixed 64D token vector
    arr[i]       -> sum of token vectors under node i
    route record -> (sentence_id, query_token_id, node, label)

The batch loader gathers compact q/current/left/right vectors on demand.
No target interval flags or left/right answer bits are provided.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = "<pad>"
UNK = "<unk>"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def tok_en(text: str) -> list[str]:
    return text.strip().lower().split()


def read_wmt_english(path: Path, *, samples: int, min_len: int, max_len: int, seed: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            a, b = line.rstrip("\n").split("\t", 1)
            en = b if sum("\u4e00" <= ch <= "\u9fff" for ch in a) > sum("\u4e00" <= ch <= "\u9fff" for ch in b) else a
            toks = tok_en(en)
            if min_len <= len(toks) <= max_len:
                rows.append(toks[:max_len])
                if len(rows) >= samples:
                    break
    if len(rows) < samples:
        raise RuntimeError(f"collected {len(rows)} rows, need {samples}")
    random.Random(seed).shuffle(rows)
    return rows


def build_vocab(rows: list[list[str]], vocab_size: int) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        counter.update(row)
    vocab = [PAD, UNK] + [w for w, _ in counter.most_common(vocab_size - 2)]
    return {w: i for i, w in enumerate(vocab)}


def encode_rows(rows: list[list[str]], stoi: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    data, lengths = [], []
    for row in rows:
        ids = [stoi.get(w, stoi[UNK]) for w in row[:max_len]]
        lengths.append(len(ids))
        ids = ids + [0] * (max_len - len(ids))
        data.append(ids)
    return torch.tensor(data, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


class HeapLayout:
    def __init__(self, max_len: int):
        self.max_len = next_power_of_two(max_len)
        self.leaf_base = self.max_len
        self.node_count = self.max_len * 2

    def mirrored_leaf_for_canonical_pos(self, pos: int) -> int:
        return self.leaf_base + (self.max_len - 1 - pos)

    def path_to_leaf(self, leaf: int) -> list[int]:
        bits = []
        node = leaf
        while node > 1:
            bits.append(1 if node % 2 == 0 else 2)  # 1=left, 2=right
            node //= 2
        return list(reversed(bits))


def fixed_token_vectors(vocab: int, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    vec = torch.randn(vocab, dim, generator=gen)
    vec = F.normalize(vec, dim=-1)
    vec[0].zero_()
    return vec


def build_compact_heap(ids: torch.Tensor, layout: HeapLayout, token_vec: torch.Tensor) -> torch.Tensor:
    bsz = ids.shape[0]
    dim = token_vec.shape[1]
    states = torch.zeros((bsz, layout.node_count, dim), dtype=torch.float32)
    mirrored = torch.flip(ids[:, : layout.max_len], dims=[1])
    states[:, layout.leaf_base : layout.leaf_base + layout.max_len] = token_vec[mirrored]
    for node in range(layout.leaf_base - 1, 0, -1):
        states[:, node] = states[:, node * 2] + states[:, node * 2 + 1]
    return states


def unique_query_positions(ids: torch.Tensor, length: int, max_queries: int) -> list[int]:
    vals = ids[:length].tolist()
    counts = Counter(vals)
    positions = [i for i, t in enumerate(vals) if t != 0 and t != 1 and counts[t] == 1]
    return positions[:max_queries]


def build_route_records(ids: torch.Tensor, lengths: torch.Tensor, layout: HeapLayout, max_queries: int) -> tuple[torch.Tensor, list[dict]]:
    rows = []
    meta = []
    for b in range(ids.shape[0]):
        length = int(lengths[b].item())
        for pos in unique_query_positions(ids[b], length, max_queries):
            token = int(ids[b, pos].item())
            leaf = layout.mirrored_leaf_for_canonical_pos(pos)
            node = 1
            for action in layout.path_to_leaf(leaf):
                rows.append((b, token, node, action))
                if len(meta) < 12:
                    meta.append({"sentence": b, "pos": pos, "token": token, "node": node, "label": action})
                node = node * 2 if action == 1 else node * 2 + 1
            rows.append((b, token, node, 0))
            if len(meta) < 12:
                meta.append({"sentence": b, "pos": pos, "token": token, "node": node, "label": 0})
    if not rows:
        raise RuntimeError("no route records")
    return torch.tensor(rows, dtype=torch.long), meta


class CompactRouteKernel(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 7, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q, cur, left, right):
        x = torch.cat([q, cur, left, right, q * cur, q * left, q * right], dim=-1)
        return self.net(x)


def gather_batch(records: torch.Tensor, states: torch.Tensor, token_vec: torch.Tensor, sel: torch.Tensor):
    rec = records[sel]
    sent = rec[:, 0]
    token = rec[:, 1]
    node = rec[:, 2]
    y = rec[:, 3]
    q = token_vec[token]
    cur = states[sent, node]
    left_node = torch.clamp(node * 2, max=states.shape[1] - 1)
    right_node = torch.clamp(node * 2 + 1, max=states.shape[1] - 1)
    has_child = node * 2 < states.shape[1]
    left = states[sent, left_node] * has_child.float().unsqueeze(-1)
    right = states[sent, right_node] * has_child.float().unsqueeze(-1)
    return q, cur, left, right, y


def train_kernel(records, states, token_vec, args):
    model = CompactRouteKernel(token_vec.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trace = []
    started = time.time()
    for epoch in range(args.epochs):
        epoch_started = time.time()
        order = torch.randperm(records.shape[0])
        total_loss, total_ok = 0.0, 0
        for i in range(0, records.shape[0], args.batch):
            sel = order[i : i + args.batch]
            q, cur, left, right, y = gather_batch(records, states, token_vec, sel)
            logits = model(q, cur, left, right)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * sel.numel()
            total_ok += int((logits.argmax(-1) == y).sum().item())
        epoch_sec = time.time() - epoch_started
        eta_sec = ((time.time() - started) / (epoch + 1)) * (args.epochs - epoch - 1)
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / records.shape[0],
            "step_acc": total_ok / records.shape[0],
            "epoch_sec": epoch_sec,
            "eta_sec": eta_sec,
        }
        trace.append(row)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            log(
                f"compact_epoch {epoch + 1}/{args.epochs} "
                f"loss={row['loss']:.6f} step_acc={row['step_acc']:.4f} "
                f"epoch_sec={epoch_sec:.2f} eta_min={eta_sec / 60:.1f}"
            )
    return model, trace


def eval_kernel(model, records, states, token_vec, meta_seed):
    with torch.no_grad():
        preds = []
        ys = []
        examples = []
        for i in range(0, records.shape[0], 4096):
            sel = torch.arange(i, min(i + 4096, records.shape[0]))
            q, cur, left, right, y = gather_batch(records, states, token_vec, sel)
            pred = model(q, cur, left, right).argmax(-1)
            preds.append(pred)
            ys.append(y)
            if len(examples) < 8:
                rec = records[sel]
                for j in range(min(8 - len(examples), rec.shape[0])):
                    examples.append({
                        "sentence": int(rec[j, 0].item()),
                        "token": int(rec[j, 1].item()),
                        "node": int(rec[j, 2].item()),
                        "label": int(y[j].item()),
                        "pred": int(pred[j].item()),
                        "left_dot": float((q[j] * left[j]).sum().item()),
                        "right_dot": float((q[j] * right[j]).sum().item()),
                        "cur_dot": float((q[j] * cur[j]).sum().item()),
                    })
        pred = torch.cat(preds)
        y = torch.cat(ys)
    step_acc = float((pred == y).float().mean().item())
    grouped: dict[tuple[int, int], list[bool]] = {}
    for i, rec in enumerate(records):
        key = (int(rec[0].item()), int(rec[1].item()))
        grouped.setdefault(key, []).append(bool(pred[i].item() == y[i].item()))
    route_exact = sum(all(v) for v in grouped.values()) / len(grouped)
    return {"steps": int(records.shape[0]), "routes": len(grouped), "step_acc": step_acc, "route_exact": route_exact, "examples": examples}


class FlatLengthRoute(nn.Module):
    def __init__(self, max_len: int):
        super().__init__()
        self.route_logits = nn.Parameter(torch.zeros(max_len + 1, max_len, max_len))

    def forward(self, lengths):
        return self.route_logits[lengths]


def flat_baseline(lengths, train_mask, ood_mask, layout, args):
    model = FlatLengthRoute(layout.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.flat_lr)
    rows = torch.arange(layout.max_len)
    train_lengths = lengths[train_mask]
    for _ in range(args.flat_epochs):
        order = torch.randperm(train_lengths.shape[0])
        for i in range(0, train_lengths.shape[0], args.batch):
            sel = order[i : i + args.batch]
            batch_lengths = train_lengths[sel]
            logits = model(batch_lengths)
            target = torch.flip(rows, dims=[0]).unsqueeze(0).expand(sel.numel(), -1)
            mask = rows.unsqueeze(0) < batch_lengths.unsqueeze(1)
            loss = F.cross_entropy(logits[mask], target[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def score(mask_sel):
        batch_lengths = lengths[mask_sel]
        logits = model(batch_lengths)
        target = torch.flip(rows, dims=[0]).unsqueeze(0).expand(batch_lengths.shape[0], -1)
        pred = logits.argmax(-1)
        mask = rows.unsqueeze(0) < batch_lengths.unsqueeze(1)
        return {
            "token_acc": float((pred[mask] == target[mask]).float().mean().item()),
            "exact": float(((pred == target) | ~mask).all(dim=1).float().mean().item()),
        }

    return {"train": score(train_mask), "ood": score(ood_mask)}


def run(args):
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(
        f"start samples={args.samples} vocab={args.vocab} dim={args.dim} "
        f"max_len={args.max_len} epochs={args.epochs}"
    )
    rows = read_wmt_english(Path(args.wmt_path), samples=args.samples, min_len=args.min_len, max_len=args.max_len, seed=args.seed)
    stoi = build_vocab(rows, args.vocab)
    ids, lengths = encode_rows(rows, stoi, args.max_len)
    layout = HeapLayout(args.max_len)
    token_vec = fixed_token_vectors(args.vocab, args.dim, args.seed + 1009)
    train_mask = lengths <= args.train_max_len
    ood_mask = lengths > args.train_max_len
    phase = time.time()
    train_states = build_compact_heap(ids[train_mask], layout, token_vec)
    ood_states = build_compact_heap(ids[ood_mask], layout, token_vec)
    log(f"build_compact_heap sec={time.time() - phase:.2f}")
    train_records, _ = build_route_records(ids[train_mask], lengths[train_mask], layout, args.max_queries_per_sentence)
    ood_records, _ = build_route_records(ids[ood_mask], lengths[ood_mask], layout, args.max_queries_per_sentence)
    compact_bytes = (
        train_states.numel() * train_states.element_size()
        + ood_states.numel() * ood_states.element_size()
        + train_records.numel() * train_records.element_size()
        + ood_records.numel() * ood_records.element_size()
        + token_vec.numel() * token_vec.element_size()
    )
    complexity = {
        "train_route_steps": int(train_records.shape[0]),
        "ood_route_steps": int(ood_records.shape[0]),
        "compact_memory_mb": compact_bytes / (1024 * 1024),
        "dense_prior_memory_mb": 6191.25,
        "memory_reduction_x_vs_dense_prior": 6191.25 / (compact_bytes / (1024 * 1024)),
        "batches_per_epoch": math.ceil(train_records.shape[0] / args.batch),
    }
    log("complexity_estimate " + json.dumps(complexity, ensure_ascii=False))
    model, trace = train_kernel(train_records, train_states, token_vec, args)
    train_metrics = eval_kernel(model, train_records, train_states, token_vec, 0)
    ood_metrics = eval_kernel(model, ood_records, ood_states, token_vec, 1)
    flat = flat_baseline(lengths, train_mask, ood_mask, layout, args)
    pass_checks = {
        "compact_memory_under_512mb": complexity["compact_memory_mb"] < 512,
        "ood_route_exact_ge_0_99": ood_metrics["route_exact"] >= 0.99,
        "ood_step_acc_ge_0_99": ood_metrics["step_acc"] >= 0.99,
        "flat_length_matrix_fails_unseen_lengths": flat["ood"]["exact"] < 0.50,
    }
    summary = {
        "claim": "S1-COMPACT-CONTENT-ROUTE-C01",
        "predict": "P-S1-COMPACT-CONTENT-ROUTE01",
        "purpose": "compact subheap embedding route without dense vocab-count feature materialization",
        "samples": args.samples,
        "vocab": args.vocab,
        "dim": args.dim,
        "max_len": layout.max_len,
        "train_rows": int(train_mask.sum().item()),
        "ood_rows": int(ood_mask.sum().item()),
        "complexity": complexity,
        "metrics": {
            "train": {k: v for k, v in train_metrics.items() if k != "examples"},
            "ood": {k: v for k, v in ood_metrics.items() if k != "examples"},
            "flat_length_matrix": flat,
            "trace": trace,
        },
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "examples": {"ood": ood_metrics["examples"]},
        "limits": [
            "fixed random token vectors, not learned semantic embeddings",
            "query token is supervised",
            "unique-token query positions only",
            "not translation",
            "not unsupervised span discovery",
        ],
        "elapsed_sec": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(
        "# S1 Compact Content TreeHeap Route Probe\n\n"
        "Claim: `S1-COMPACT-CONTENT-ROUTE-C01`\n\n"
        f"Compact {args.dim}D subheap states replace dense vocab-count route features.\n\n"
        "```json\n" + json.dumps({k: summary[k] for k in ["complexity", "metrics", "pass_checks", "pilot_pass", "limits"]}, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    (out / "command.sh").write_text("python3 " + " ".join(sys.argv) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv")
    p.add_argument("--out", default="ara/s1-echo/evidence/s1_compact_content_treeheap_route_probe")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--train-max-len", type=int, default=24)
    p.add_argument("--max-queries-per-sentence", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--flat-epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--flat-lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()
    summary = run(args)
    print(json.dumps({
        "pilot_pass": summary["pilot_pass"],
        "pass_checks": summary["pass_checks"],
        "complexity": summary["complexity"],
        "ood": summary["metrics"]["ood"],
        "flat_ood": summary["metrics"]["flat_length_matrix"]["ood"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
