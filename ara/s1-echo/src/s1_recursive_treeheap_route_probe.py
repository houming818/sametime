#!/usr/bin/env python3
"""SPR-045 recursive TreeHeap route proof.

This probe replaces the old flat `L x L` inverse route with a real TreeHeap
reader:

    i = 1
    while not stop:
        logits = K_theta(q, S_i, address_i)
        i = 2*i or 2*i+1

The task is deliberately narrow.  A sentence is written into max_len heap
leaves, the whole heap is mirrored, and the reader must recover the canonical
sentence by recursively walking from arr[1] to the mirrored leaf address.

This proves the mechanism boundary, not translation quality.
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


def tok_en(text: str) -> list[str]:
    return text.strip().lower().split()


def read_wmt_english(path: Path, *, samples: int, min_len: int, max_len: int, seed: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            a, b = line.rstrip("\n").split("\t", 1)
            # Accept both EN<TAB>ZH and ZH<TAB>EN files.
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
        self.intervals: dict[int, tuple[int, int]] = {}
        for i in range(1, self.node_count):
            self.intervals[i] = self._interval(i)

    def _interval(self, node: int) -> tuple[int, int]:
        depth = int(math.floor(math.log2(node)))
        offset = node - (1 << depth)
        width = self.max_len // (1 << depth)
        start = offset * width
        return start, start + width

    def leaf_node(self, pos: int) -> int:
        return self.leaf_base + pos

    def mirrored_leaf_for_canonical_pos(self, pos: int) -> int:
        return self.leaf_node(self.max_len - 1 - pos)

    def path_to_leaf(self, leaf: int) -> list[int]:
        bits = []
        node = leaf
        while node > 1:
            bits.append(1 if node % 2 == 0 else 2)  # action id: 1=left, 2=right
            node //= 2
        return list(reversed(bits))

    def mirror_leaf_index(self, pos: int) -> int:
        return self.max_len - 1 - pos


def build_mirrored_leaves(ids: torch.Tensor, layout: HeapLayout) -> torch.Tensor:
    # A full TreeHeap mirror maps leaf position p to max_len-1-p.
    return torch.flip(ids[:, : layout.max_len], dims=[1])


def route_features(layout: HeapLayout, target_leaf_pos: int, node: int, step: int) -> list[float]:
    start, end = layout.intervals[node]
    left_start, left_end = layout.intervals.get(node * 2, (start, start))
    right_start, right_end = layout.intervals.get(node * 2 + 1, (end, end))
    width = end - start
    center = (start + end - 1) / 2.0
    target = target_leaf_pos
    denom = float(layout.max_len)
    return [
        target / denom,
        start / denom,
        end / denom,
        center / denom,
        width / denom,
        left_start / denom,
        left_end / denom,
        right_start / denom,
        right_end / denom,
        step / max(1.0, math.log2(layout.max_len)),
        1.0 if left_start <= target < left_end else 0.0,
        1.0 if right_start <= target < right_end else 0.0,
    ]


def make_route_training(layout: HeapLayout, positions: list[int], repeat: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    feats, labels = [], []
    for _ in range(repeat):
        rng.shuffle(positions)
        for pos in positions:
            target_leaf_pos = layout.mirror_leaf_index(pos)
            node = 1
            for step, action in enumerate(layout.path_to_leaf(layout.mirrored_leaf_for_canonical_pos(pos))):
                feats.append(route_features(layout, target_leaf_pos, node, step))
                labels.append(action)
                node = node * 2 if action == 1 else node * 2 + 1
            feats.append(route_features(layout, target_leaf_pos, node, int(math.log2(layout.max_len))))
            labels.append(0)  # stop
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class RecursiveRouteKernel(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_recursive_kernel(layout: HeapLayout, train_positions: list[int], args) -> tuple[RecursiveRouteKernel, dict]:
    x, y = make_route_training(layout, train_positions, args.route_repeat, args.seed)
    model = RecursiveRouteKernel(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trace = []
    for epoch in range(args.epochs):
        order = torch.randperm(x.shape[0])
        total_loss = 0.0
        total_ok = 0
        for i in range(0, x.shape[0], args.batch):
            sel = order[i : i + args.batch]
            logits = model(x[sel])
            loss = F.cross_entropy(logits, y[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * sel.numel()
            total_ok += int((logits.argmax(-1) == y[sel]).sum().item())
        trace.append({"epoch": epoch + 1, "loss": total_loss / x.shape[0], "step_acc": total_ok / x.shape[0]})
    return model, {"trace": trace, "final_step_acc": trace[-1]["step_acc"], "final_loss": trace[-1]["loss"]}


def recursive_read(model: RecursiveRouteKernel, layout: HeapLayout, pos: int, max_steps: int | None = None) -> tuple[int, list[int]]:
    node = 1
    actions = []
    target_leaf_pos = layout.mirror_leaf_index(pos)
    max_steps = max_steps or (int(math.log2(layout.max_len)) + 2)
    for step in range(max_steps):
        feat = torch.tensor([route_features(layout, target_leaf_pos, node, step)], dtype=torch.float32)
        action = int(model(feat).argmax(-1).item())
        actions.append(action)
        if action == 0:
            break
        if node >= layout.leaf_base:
            break
        node = node * 2 if action == 1 else node * 2 + 1
    return node, actions


def hard_oracle_read(layout: HeapLayout, pos: int) -> int:
    return layout.mirrored_leaf_for_canonical_pos(pos)


class FlatLengthRoute(nn.Module):
    def __init__(self, max_len: int):
        super().__init__()
        self.route_logits = nn.Parameter(torch.zeros(max_len + 1, max_len, max_len))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.route_logits[lengths]


def train_flat_length_route(train_ids: torch.Tensor, train_lengths: torch.Tensor, layout: HeapLayout, args) -> tuple[FlatLengthRoute, dict]:
    model = FlatLengthRoute(layout.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.flat_lr)
    rows = torch.arange(layout.max_len)
    trace = []
    for epoch in range(args.flat_epochs):
        total_loss, total_ok, total_n = 0.0, 0, 0
        order = torch.randperm(train_ids.shape[0])
        for i in range(0, train_ids.shape[0], args.batch):
            sel = order[i : i + args.batch]
            lengths = train_lengths[sel]
            logits = model(lengths)
            target = torch.flip(rows, dims=[0]).unsqueeze(0).expand(sel.numel(), -1)
            # Only canonical token positions inside each sentence count.
            mask = rows.unsqueeze(0) < lengths.unsqueeze(1)
            loss = F.cross_entropy(logits[mask], target[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pred = logits.argmax(-1)
            total_loss += float(loss.item()) * int(mask.sum().item())
            total_ok += int((pred[mask] == target[mask]).sum().item())
            total_n += int(mask.sum().item())
        trace.append({"epoch": epoch + 1, "loss": total_loss / total_n, "route_acc": total_ok / total_n})
    return model, {"trace": trace, "final_route_acc": trace[-1]["route_acc"], "final_loss": trace[-1]["loss"]}


def eval_routes(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    layout: HeapLayout,
    recursive_model: RecursiveRouteKernel,
    flat_model: FlatLengthRoute,
) -> dict:
    mirrored = build_mirrored_leaves(ids, layout)
    rows = torch.arange(layout.max_len)
    flat_logits = flat_model(lengths)
    flat_pred = flat_logits.argmax(-1)
    sent_total = ids.shape[0]
    oracle_exact = 0
    recursive_exact = 0
    flat_exact = 0
    recursive_token_ok = 0
    flat_token_ok = 0
    token_total = 0
    examples = []
    for b in range(ids.shape[0]):
        length = int(lengths[b].item())
        oracle_row, recursive_row, flat_row = [], [], []
        action_examples = []
        for p in range(length):
            target = int(ids[b, p].item())
            oracle_leaf = hard_oracle_read(layout, p) - layout.leaf_base
            oracle_row.append(int(mirrored[b, oracle_leaf].item()))
            node, actions = recursive_read(recursive_model, layout, p)
            rec_leaf = node - layout.leaf_base if node >= layout.leaf_base else -1
            rec_token = int(mirrored[b, rec_leaf].item()) if 0 <= rec_leaf < layout.max_len else -999
            recursive_row.append(rec_token)
            flat_leaf = int(flat_pred[b, p].item())
            flat_token = int(mirrored[b, flat_leaf].item()) if 0 <= flat_leaf < layout.max_len else -999
            flat_row.append(flat_token)
            recursive_token_ok += int(rec_token == target)
            flat_token_ok += int(flat_token == target)
            token_total += 1
            if b < 3 and p < min(5, length):
                action_examples.append({"pos": p, "target_leaf": oracle_leaf, "actions": actions, "node": node})
        target_row = [int(x) for x in ids[b, :length].tolist()]
        oracle_exact += int(oracle_row == target_row)
        recursive_exact += int(recursive_row == target_row)
        flat_exact += int(flat_row == target_row)
        if b < 3:
            examples.append({
                "length": length,
                "target_ids": target_row[:12],
                "oracle_ids": oracle_row[:12],
                "recursive_ids": recursive_row[:12],
                "flat_ids": flat_row[:12],
                "recursive_actions": action_examples,
            })
    return {
        "sentences": sent_total,
        "tokens": token_total,
        "oracle_exact": oracle_exact / sent_total,
        "recursive_exact": recursive_exact / sent_total,
        "flat_exact": flat_exact / sent_total,
        "recursive_token_acc": recursive_token_ok / token_total,
        "flat_token_acc": flat_token_ok / token_total,
        "examples": examples,
    }


def write_readme(out: Path, summary: dict) -> None:
    lines = [
        "# S1 Recursive TreeHeap Route Probe",
        "",
        "Claim: `S1-RECURSIVE-ROUTE-C01`",
        "",
        "This proof intentionally separates TreeHeap routing from a flat `L x L` route matrix.",
        "",
        "A valid TreeHeap route here means:",
        "",
        "```text",
        "i = 1",
        "K_theta(q, S_i, address_i) -> stop/left/right",
        "i = 2*i or 2*i+1 until stop",
        "```",
        "",
        "It does not claim translation, semantics, or discovered natural-language syntax.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(summary["metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- `oracle_exact` checks the hard TreeHeap mirror algebra.",
        "- `recursive_exact` checks learned stop/left/right traversal.",
        "- `flat_exact` is the old length-indexed route matrix baseline.",
        "- This proof is only about the mechanism boundary: tree route vs matrix route.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_wmt_english(Path(args.wmt_path), samples=args.samples, min_len=args.min_len, max_len=args.max_len, seed=args.seed)
    stoi = build_vocab(rows, args.vocab)
    ids, lengths = encode_rows(rows, stoi, args.max_len)
    layout = HeapLayout(args.max_len)

    # Train on short lengths, test on longer lengths to stress flat length-table rows.
    train_mask = lengths <= args.train_max_len
    ood_mask = lengths > args.train_max_len
    if int(train_mask.sum()) < 100 or int(ood_mask.sum()) < 100:
        raise RuntimeError("not enough train/OOD rows; adjust samples or train_max_len")
    train_ids, train_lengths = ids[train_mask], lengths[train_mask]
    ood_ids, ood_lengths = ids[ood_mask], lengths[ood_mask]

    train_positions = list(range(layout.max_len))
    recursive_model, recursive_train = train_recursive_kernel(layout, train_positions, args)
    flat_model, flat_train = train_flat_length_route(train_ids, train_lengths, layout, args)

    train_metrics = eval_routes(train_ids[: args.eval_rows], train_lengths[: args.eval_rows], layout, recursive_model, flat_model)
    ood_metrics = eval_routes(ood_ids[: args.eval_rows], ood_lengths[: args.eval_rows], layout, recursive_model, flat_model)
    metrics = {
        "train": {k: v for k, v in train_metrics.items() if k != "examples"},
        "ood": {k: v for k, v in ood_metrics.items() if k != "examples"},
        "recursive_train": recursive_train,
        "flat_train": flat_train,
    }
    pass_checks = {
        "oracle_ood_exact": ood_metrics["oracle_exact"] == 1.0,
        "recursive_ood_exact_ge_0_99": ood_metrics["recursive_exact"] >= 0.99,
        "recursive_uses_step_actions": all(
            len(item["actions"]) >= int(math.log2(layout.max_len))
            for ex in ood_metrics["examples"]
            for item in ex["recursive_actions"]
        ),
        "flat_length_matrix_fails_unseen_lengths": ood_metrics["flat_exact"] < 0.50,
    }
    summary = {
        "claim": "S1-RECURSIVE-ROUTE-C01",
        "predict": "P-S1-RECURSIVE-ROUTE01",
        "purpose": "prove recursive stop/left/right TreeHeap route, not flat route matrix",
        "wmt_path": args.wmt_path,
        "samples": args.samples,
        "max_len": layout.max_len,
        "train_max_len": args.train_max_len,
        "train_rows": int(train_mask.sum().item()),
        "ood_rows": int(ood_mask.sum().item()),
        "metrics": metrics,
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "examples": {
            "ood": ood_metrics["examples"],
        },
        "limits": [
            "not translation",
            "not semantic grounding",
            "route target position is supervised",
            "does not discover which subheap should be flipped",
            "flat shared route baselines remain a future control",
        ],
        "elapsed_sec": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in recursive_train["trace"]:
            f.write(json.dumps({"phase": "recursive_route", **row}, ensure_ascii=False) + "\n")
        for row in flat_train["trace"]:
            f.write(json.dumps({"phase": "flat_length_route", **row}, ensure_ascii=False) + "\n")
    (out / "command.sh").write_text(" ".join(["python3", "ara/s1-echo/src/s1_recursive_treeheap_route_probe.py"]) + "\n", encoding="utf-8")
    write_readme(out, summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv")
    p.add_argument("--out", default="ara/s1-echo/evidence/s1_recursive_treeheap_route_probe")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--train-max-len", type=int, default=24)
    p.add_argument("--eval-rows", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--flat-epochs", type=int, default=80)
    p.add_argument("--route-repeat", type=int, default=64)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--flat-lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    summary = run(args)
    print(json.dumps({
        "pilot_pass": summary["pilot_pass"],
        "pass_checks": summary["pass_checks"],
        "ood": summary["metrics"]["ood"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
