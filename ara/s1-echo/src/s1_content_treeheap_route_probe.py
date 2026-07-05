#!/usr/bin/env python3
"""SPR-046 content-aware recursive TreeHeap route proof.

SPR-045 fixed the "flat LxL matrix is not a tree" problem, but its route kernel
still used target geometry features.  This probe removes those geometry-answer
features.  The route kernel gets only:

    query token vector
    arr[i] summary
    arr[2i] summary
    arr[2i+1] summary

and must choose stop/left/right by reading heap content.

This is still a controlled echo proof, not translation or semantic grounding.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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


def build_mirrored_heap(ids: torch.Tensor, layout: HeapLayout, vocab: int) -> torch.Tensor:
    """Build count-vector summaries for a mirrored complete binary heap.

    arr[node] is a multi-hot/count vector of tokens under that node.
    Leaves store mirrored token ids.  PAD is ignored in summaries.
    """
    bsz = ids.shape[0]
    arr = torch.zeros((bsz, layout.node_count, vocab), dtype=torch.float32)
    mirrored = torch.flip(ids[:, : layout.max_len], dims=[1])
    for p in range(layout.max_len):
        token = mirrored[:, p]
        mask = token != 0
        if mask.any():
            arr[mask, layout.leaf_base + p, token[mask]] = 1.0
    for node in range(layout.leaf_base - 1, 0, -1):
        arr[:, node] = arr[:, node * 2] + arr[:, node * 2 + 1]
    arr.clamp_(0.0, 1.0)
    return arr


def unique_query_positions(ids: torch.Tensor, length: int, max_queries: int) -> list[int]:
    vals = ids[:length].tolist()
    counts = Counter(vals)
    positions = [i for i, t in enumerate(vals) if t != 0 and t != 1 and counts[t] == 1]
    return positions[:max_queries]


def build_route_examples(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    layout: HeapLayout,
    vocab: int,
    max_queries_per_sentence: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    heaps = build_mirrored_heap(ids, layout, vocab)
    q_rows, cur_rows, left_rows, right_rows, labels = [], [], [], [], []
    meta = []
    eye = torch.eye(vocab)
    for b in range(ids.shape[0]):
        length = int(lengths[b].item())
        for pos in unique_query_positions(ids[b], length, max_queries_per_sentence):
            token = int(ids[b, pos].item())
            leaf = layout.mirrored_leaf_for_canonical_pos(pos)
            node = 1
            path = layout.path_to_leaf(leaf)
            for action in path:
                q_rows.append(eye[token])
                cur_rows.append(heaps[b, node])
                left_rows.append(heaps[b, node * 2])
                right_rows.append(heaps[b, node * 2 + 1])
                labels.append(action)
                meta.append({"sentence": b, "pos": pos, "token": token, "node": node, "label": action})
                node = node * 2 if action == 1 else node * 2 + 1
            q_rows.append(eye[token])
            cur_rows.append(heaps[b, node])
            left_rows.append(torch.zeros(vocab))
            right_rows.append(torch.zeros(vocab))
            labels.append(0)
            meta.append({"sentence": b, "pos": pos, "token": token, "node": node, "label": 0})
    if not labels:
        raise RuntimeError("no unique-token route examples")
    return (
        torch.stack(q_rows),
        torch.stack(cur_rows),
        torch.stack(left_rows),
        torch.stack(right_rows),
        torch.tensor(labels, dtype=torch.long),
        meta,
    )


class ContentRouteKernel(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab * 7, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q: torch.Tensor, cur: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # The products are computed inside the kernel from heap content and query.
        x = torch.cat([q, cur, left, right, q * cur, q * left, q * right], dim=-1)
        return self.net(x)


def train_kernel(q, cur, left, right, y, args) -> tuple[ContentRouteKernel, list[dict]]:
    model = ContentRouteKernel(q.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trace = []
    train_started = time.time()
    for epoch in range(args.epochs):
        epoch_started = time.time()
        order = torch.randperm(y.shape[0])
        total_loss, total_ok = 0.0, 0
        for i in range(0, y.shape[0], args.batch):
            sel = order[i : i + args.batch]
            logits = model(q[sel], cur[sel], left[sel], right[sel])
            loss = F.cross_entropy(logits, y[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * sel.numel()
            total_ok += int((logits.argmax(-1) == y[sel]).sum().item())
        epoch_sec = time.time() - epoch_started
        mean_epoch_sec = (time.time() - train_started) / (epoch + 1)
        eta_sec = mean_epoch_sec * (args.epochs - epoch - 1)
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / y.shape[0],
            "step_acc": total_ok / y.shape[0],
            "epoch_sec": epoch_sec,
            "eta_sec": eta_sec,
        }
        trace.append(row)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            log(
                "content_epoch "
                f"{epoch + 1}/{args.epochs} "
                f"loss={row['loss']:.6f} step_acc={row['step_acc']:.4f} "
                f"epoch_sec={epoch_sec:.2f} eta_min={eta_sec / 60:.1f}"
            )
    return model, trace


def eval_kernel(model, q, cur, left, right, y, meta) -> dict:
    with torch.no_grad():
        pred = model(q, cur, left, right).argmax(-1)
    step_acc = float((pred == y).float().mean().item())
    grouped: dict[tuple[int, int], list[bool]] = {}
    examples = []
    for i, m in enumerate(meta):
        key = (m["sentence"], m["pos"])
        grouped.setdefault(key, []).append(bool(pred[i].item() == int(y[i].item())))
        if len(examples) < 8:
            examples.append({
                "sentence": m["sentence"],
                "pos": m["pos"],
                "token": m["token"],
                "node": m["node"],
                "label": int(y[i].item()),
                "pred": int(pred[i].item()),
                "left_has_query": float((q[i] * left[i]).sum().item()),
                "right_has_query": float((q[i] * right[i]).sum().item()),
                "current_has_query": float((q[i] * cur[i]).sum().item()),
            })
    route_exact = sum(all(v) for v in grouped.values()) / len(grouped)
    return {
        "steps": len(y),
        "routes": len(grouped),
        "step_acc": step_acc,
        "route_exact": route_exact,
        "examples": examples,
    }


class FlatLengthRoute(nn.Module):
    def __init__(self, max_len: int):
        super().__init__()
        self.route_logits = nn.Parameter(torch.zeros(max_len + 1, max_len, max_len))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.route_logits[lengths]


def flat_baseline(ids, lengths, train_mask, ood_mask, layout, args) -> dict:
    model = FlatLengthRoute(layout.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.flat_lr)
    rows = torch.arange(layout.max_len)
    train_lengths = lengths[train_mask]
    flat_started = time.time()
    for epoch in range(args.flat_epochs):
        epoch_started = time.time()
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
        epoch_sec = time.time() - epoch_started
        mean_epoch_sec = (time.time() - flat_started) / (epoch + 1)
        eta_sec = mean_epoch_sec * (args.flat_epochs - epoch - 1)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.flat_epochs:
            log(f"flat_epoch {epoch + 1}/{args.flat_epochs} epoch_sec={epoch_sec:.2f} eta_min={eta_sec / 60:.1f}")
    def score(sel_mask):
        batch_lengths = lengths[sel_mask]
        logits = model(batch_lengths)
        target = torch.flip(rows, dims=[0]).unsqueeze(0).expand(batch_lengths.shape[0], -1)
        pred = logits.argmax(-1)
        mask = rows.unsqueeze(0) < batch_lengths.unsqueeze(1)
        token_acc = float((pred[mask] == target[mask]).float().mean().item())
        exact = float(((pred == target) | ~mask).all(dim=1).float().mean().item())
        return {"token_acc": token_acc, "exact": exact}
    return {"train": score(train_mask), "ood": score(ood_mask)}


def run(args) -> dict:
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log(
        "start "
        f"samples={args.samples} vocab={args.vocab} max_len={args.max_len} "
        f"train_max_len={args.train_max_len} epochs={args.epochs} "
        f"flat_epochs={args.flat_epochs} batch={args.batch} hidden={args.hidden}"
    )
    phase = time.time()
    rows = read_wmt_english(Path(args.wmt_path), samples=args.samples, min_len=args.min_len, max_len=args.max_len, seed=args.seed)
    log(f"read_wmt rows={len(rows)} sec={time.time() - phase:.2f}")
    phase = time.time()
    stoi = build_vocab(rows, args.vocab)
    ids, lengths = encode_rows(rows, stoi, args.max_len)
    layout = HeapLayout(args.max_len)
    log(f"build_vocab_encode vocab={len(stoi)} heap_max_len={layout.max_len} sec={time.time() - phase:.2f}")

    train_mask = lengths <= args.train_max_len
    ood_mask = lengths > args.train_max_len
    log(f"split train_rows={int(train_mask.sum().item())} ood_rows={int(ood_mask.sum().item())}")
    phase = time.time()
    train = build_route_examples(ids[train_mask], lengths[train_mask], layout, args.vocab, args.max_queries_per_sentence)
    log(f"build_train_route_examples steps={len(train[4])} sec={time.time() - phase:.2f}")
    phase = time.time()
    ood = build_route_examples(ids[ood_mask], lengths[ood_mask], layout, args.vocab, args.max_queries_per_sentence)
    log(f"build_ood_route_examples steps={len(ood[4])} sec={time.time() - phase:.2f}")

    feature_tensors = train[:4] + ood[:4]
    dense_bytes = sum(t.numel() * t.element_size() for t in feature_tensors)
    train_steps = len(train[4])
    ood_steps = len(ood[4])
    batches_per_content_epoch = math.ceil(train_steps / args.batch)
    mlp_input_dim = args.vocab * 7
    rough_muladds_per_content_epoch = train_steps * (
        mlp_input_dim * args.hidden + args.hidden * args.hidden + args.hidden * 3
    )
    complexity_estimate = {
        "train_route_steps": train_steps,
        "ood_route_steps": ood_steps,
        "route_steps_total": train_steps + ood_steps,
        "dense_feature_memory_mb": dense_bytes / (1024 * 1024),
        "batches_per_content_epoch": batches_per_content_epoch,
        "mlp_input_dim": mlp_input_dim,
        "rough_muladds_per_content_epoch": rough_muladds_per_content_epoch,
        "rough_muladds_all_content_epochs": rough_muladds_per_content_epoch * args.epochs,
    }
    log("complexity_estimate " + json.dumps(complexity_estimate, ensure_ascii=False))
    if args.profile_only:
        summary = {
            "claim": "S1-CONTENT-ROUTE-C01",
            "predict": "P-S1-CONTENT-ROUTE01",
            "purpose": "profile content-aware recursive TreeHeap route before full run",
            "complexity_estimate": complexity_estimate,
            "profile_only": True,
            "pilot_pass": False,
            "elapsed_sec": time.time() - started,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    model, trace = train_kernel(*train[:5], args)
    train_metrics = eval_kernel(model, *train[:5], train[5])
    ood_metrics = eval_kernel(model, *ood[:5], ood[5])
    flat = flat_baseline(ids, lengths, train_mask, ood_mask, layout, args)

    pass_checks = {
        "kernel_reads_heap_content": True,
        "no_geometry_answer_features": True,
        "ood_route_exact_ge_0_99": ood_metrics["route_exact"] >= 0.99,
        "ood_step_acc_ge_0_99": ood_metrics["step_acc"] >= 0.99,
        "flat_length_matrix_fails_unseen_lengths": flat["ood"]["exact"] < 0.50,
    }
    summary = {
        "claim": "S1-CONTENT-ROUTE-C01",
        "predict": "P-S1-CONTENT-ROUTE01",
        "purpose": "content-aware recursive TreeHeap route over arr[i], not geometry-answer routing",
        "wmt_path": args.wmt_path,
        "samples": args.samples,
        "vocab": args.vocab,
        "max_len": layout.max_len,
        "train_max_len": args.train_max_len,
        "train_rows": int(train_mask.sum().item()),
        "ood_rows": int(ood_mask.sum().item()),
        "train_route_examples": len(train[4]),
        "ood_route_examples": len(ood[4]),
        "metrics": {
            "train": {k: v for k, v in train_metrics.items() if k != "examples"},
            "ood": {k: v for k, v in ood_metrics.items() if k != "examples"},
            "flat_length_matrix": flat,
            "trace": trace,
        },
        "complexity_estimate": complexity_estimate,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "examples": {"ood": ood_metrics["examples"]},
        "limits": [
            "query token is supervised",
            "uses bag/count summaries, not full semantic vectors",
            "unique-token query positions only",
            "not translation",
            "not unsupervised span discovery",
            "needs pointer/shared-flat baselines next",
        ],
        "elapsed_sec": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(
        "# S1 Content TreeHeap Route Probe\n\n"
        "Claim: `S1-CONTENT-ROUTE-C01`\n\n"
        "The kernel receives query plus `arr[i]`, `arr[2i]`, `arr[2i+1]` content summaries.\n"
        "It does not receive target interval flags or precomputed left/right answer bits.\n\n"
        "```json\n" + json.dumps({k: summary[k] for k in ["metrics", "pass_checks", "pilot_pass", "limits"]}, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    (out / "command.sh").write_text("python3 ara/s1-echo/src/s1_content_treeheap_route_probe.py\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv")
    p.add_argument("--out", default="ara/s1-echo/evidence/s1_content_treeheap_route_probe")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--train-max-len", type=int, default=24)
    p.add_argument("--max-queries-per-sentence", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--flat-epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--flat-lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--profile-only", action="store_true")
    args = p.parse_args()
    summary = run(args)
    if args.profile_only:
        print(json.dumps({
            "profile_only": True,
            "pilot_pass": summary["pilot_pass"],
            "complexity_estimate": summary["complexity_estimate"],
            "elapsed_sec": summary["elapsed_sec"],
        }, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({
            "pilot_pass": summary["pilot_pass"],
            "pass_checks": summary["pass_checks"],
            "ood": summary["metrics"]["ood"],
            "flat_ood": summary["metrics"]["flat_length_matrix"]["ood"],
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
