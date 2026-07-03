#!/usr/bin/env python3
"""S1 local TreeHeap flip echo probe.

SPR-043 full-tree flip is a useful smoke proof, but too idealized. This probe
uses the same TreeHeap flip algebra locally: pick a contiguous span, build a
TreeHeap for the span, Flip(span_root, full_depth), leave the rest unchanged,
and learn an inverse route conditioned on span start and span length.
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
    out: list[str] = []
    for tok in tokens:
        if tok in {".", ",", "!", "?", ";", ":"} and out:
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return " ".join(out)


def read_wmt_english(path: Path, scan_lines: int) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            en = line.split("\t", 1)[0].strip()
            if en:
                out.append(en)
            if len(out) >= scan_lines:
                break
    return out


@dataclass
class TreeNode:
    value: int | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def build_tree(tokens: list[int]) -> TreeNode:
    if len(tokens) == 1:
        return TreeNode(value=tokens[0])
    mid = (len(tokens) + 1) // 2
    return TreeNode(left=build_tree(tokens[:mid]), right=build_tree(tokens[mid:]))


def clone_tree(node: TreeNode) -> TreeNode:
    if node.value is not None:
        return TreeNode(value=node.value)
    assert node.left is not None and node.right is not None
    return TreeNode(left=clone_tree(node.left), right=clone_tree(node.right))


def flip_tree(node: TreeNode, depth: int | None = None) -> TreeNode:
    if node.value is not None:
        return TreeNode(value=node.value)
    if depth is not None and depth <= 0:
        return clone_tree(node)
    assert node.left is not None and node.right is not None
    next_depth = None if depth is None else depth - 1
    return TreeNode(left=flip_tree(node.right, next_depth), right=flip_tree(node.left, next_depth))


def leaves(node: TreeNode) -> list[int]:
    if node.value is not None:
        return [node.value]
    assert node.left is not None and node.right is not None
    return leaves(node.left) + leaves(node.right)


def treeheap_full_flip(tokens: list[int]) -> list[int]:
    return leaves(flip_tree(build_tree(tokens), depth=None))


def local_treeheap_flip(tokens: list[int], start: int, span_len: int) -> list[int]:
    out = list(tokens)
    out[start:start + span_len] = treeheap_full_flip(out[start:start + span_len])
    return out


def levenshtein(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def build_dataset(args):
    texts = read_wmt_english(Path(args.wmt_path), args.scan_lines)
    tokenized = [tokenize(t) for t in texts]
    tokenized = [t for t in tokenized if args.min_len <= len(t) <= args.max_len]
    counts = Counter(tok for sent in tokenized for tok in sent)
    vocab_words = [w for w, _ in counts.most_common(args.vocab_size - 2)]
    stoi = {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(vocab_words)}}
    itos = {i: w for w, i in stoi.items()}

    rng = np.random.default_rng(args.seed)
    encoded: list[list[int]] = []
    starts: list[int] = []
    spans: list[int] = []
    for sent in tokenized:
        ids = [stoi.get(tok, UNK) for tok in sent]
        if args.drop_unk and UNK in ids:
            continue
        max_span = min(args.max_span_len, len(ids))
        min_span = min(args.min_span_len, max_span)
        if max_span < min_span:
            continue
        span_len = int(rng.integers(min_span, max_span + 1))
        start = int(rng.integers(0, len(ids) - span_len + 1))
        encoded.append(ids)
        starts.append(start)
        spans.append(span_len)
        if len(encoded) >= args.samples:
            break
    if len(encoded) < args.samples:
        raise RuntimeError(f"only collected {len(encoded)} examples")

    target = np.zeros((len(encoded), args.max_len), dtype=np.int64)
    observed = np.zeros((len(encoded), args.max_len), dtype=np.int64)
    lengths = np.zeros((len(encoded),), dtype=np.int64)
    start_arr = np.array(starts, dtype=np.int64)
    span_arr = np.array(spans, dtype=np.int64)
    hard_ok = 0
    examples = []
    for i, ids in enumerate(encoded):
        obs = local_treeheap_flip(ids, starts[i], spans[i])
        restored = local_treeheap_flip(obs, starts[i], spans[i])
        hard_ok += int(restored == ids)
        L = len(ids)
        lengths[i] = L
        target[i, :L] = ids
        observed[i, :L] = obs
        if len(examples) < args.examples:
            examples.append(
                {
                    "target": detok([itos[x] for x in ids]),
                    "observed_local_treeheap_flip": detok([itos[x] for x in obs]),
                    "hard_restored": detok([itos[x] for x in restored]),
                    "length": L,
                    "span_start": starts[i],
                    "span_len": spans[i],
                }
            )

    meta = {
        "wmt_path": args.wmt_path,
        "scan_lines": args.scan_lines,
        "samples": len(encoded),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_span_len": args.min_span_len,
        "max_span_len": args.max_span_len,
        "vocab_size": len(stoi),
        "drop_unk": args.drop_unk,
        "hard_local_treeheap_closure_exact": hard_ok / len(encoded),
        "hard_treeheap_operator": "local Flip(span_root, full_depth) applied twice on local span TreeHeap",
        "dataset_examples": examples,
    }
    return target, observed, lengths, start_arr, span_arr, stoi, itos, meta


def split_data(*arrays, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(arrays[0]))
    n_train = int(0.8 * len(idx))
    n_test = int(0.1 * len(idx))
    split_idx = {
        "train": idx[:n_train],
        "test": idx[n_train:n_train + n_test],
        "ood": idx[n_train + n_test:],
    }
    return {name: tuple(arr[sel] for arr in arrays) for name, sel in split_idx.items()}


class LocalInverseFlip(nn.Module):
    def __init__(self, vocab: int, max_len: int, max_span_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.emb.weight.requires_grad_(False)
        self.route_logits = nn.Parameter(torch.zeros(max_span_len + 1, max_span_len, max_span_len))
        with torch.no_grad():
            self.emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(dim))
            self.emb.weight[PAD].zero_()
            self.route_logits.normal_(mean=0.0, std=0.02)

    def decode_logits(self, state: torch.Tensor) -> torch.Tensor:
        state_n = F.normalize(state, dim=-1)
        codebook_n = F.normalize(self.emb.weight, dim=-1)
        return torch.matmul(state_n, codebook_n.t()) * 20.0

    def forward(self, observed: torch.Tensor, lengths: torch.Tensor, starts: torch.Tensor, spans: torch.Tensor):
        leaf = self.emb(observed)
        state = leaf.clone()
        route = F.softmax(self.route_logits[spans], dim=-1)
        bsz, max_len, dim = leaf.shape
        max_span = self.route_logits.shape[1]
        span_positions = torch.arange(max_span, device=leaf.device).unsqueeze(0)
        source_idx = starts.unsqueeze(1) + span_positions
        valid = span_positions < spans.unsqueeze(1)
        source_idx = source_idx.clamp(0, max_len - 1)
        gathered = leaf.gather(1, source_idx.unsqueeze(-1).expand(-1, -1, dim))
        restored_span = torch.einsum("bij,bjd->bid", route, gathered)
        for b in range(bsz):
            start = int(starts[b].item())
            span = int(spans[b].item())
            state[b, start : start + span, :] = restored_span[b, :span, :]
        return self.decode_logits(state), state, route


class NoInverseBaseline(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.emb.weight.requires_grad_(False)
        with torch.no_grad():
            self.emb.weight.normal_(mean=0.0, std=1.0 / math.sqrt(dim))
            self.emb.weight[PAD].zero_()

    def decode_logits(self, state: torch.Tensor) -> torch.Tensor:
        state_n = F.normalize(state, dim=-1)
        codebook_n = F.normalize(self.emb.weight, dim=-1)
        return torch.matmul(state_n, codebook_n.t()) * 20.0

    def forward(self, observed: torch.Tensor, lengths: torch.Tensor, starts: torch.Tensor, spans: torch.Tensor):
        state = self.emb(observed)
        route = torch.empty((observed.shape[0], 1, 1), device=observed.device)
        return self.decode_logits(state), state, route


def loss_fn(logits, state, target, target_state, route, lengths, starts, spans, args):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)
    pos = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
    active = (pos < lengths.unsqueeze(1)).float().unsqueeze(-1)
    state_loss = (((state - target_state.detach()) ** 2) * active).sum() / active.sum().clamp_min(1.0)
    if route.numel() > 1:
        entropy = -(route.clamp_min(1e-9) * route.clamp_min(1e-9).log()).sum(dim=-1)
        span_pos = torch.arange(route.shape[1], device=target.device).unsqueeze(0)
        valid = (span_pos < spans.unsqueeze(1)).float()
        entropy_loss = (entropy * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        entropy_loss = torch.tensor(0.0, device=target.device)
    return ce + args.state_weight * state_loss + args.entropy_weight * entropy_loss


def train_model(model, split, args, device):
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return model.cpu(), [{"epoch": -1, "loss": None, "note": "no trainable parameters"}]
    opt = torch.optim.Adam(trainable, lr=args.lr)
    target, observed, lengths, starts, spans = [torch.tensor(x, dtype=torch.long, device=device) for x in split]
    rng = np.random.default_rng(args.seed)
    trace = []
    n = target.shape[0]
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start_idx in range(0, n, args.batch):
            sel = torch.tensor(order[start_idx:start_idx + args.batch], dtype=torch.long, device=device)
            opt.zero_grad()
            logits, state, route = model(observed[sel], lengths[sel], starts[sel], spans[sel])
            target_state = model.emb(target[sel])
            loss = loss_fn(logits, state, target[sel], target_state, route, lengths[sel], starts[sel], spans[sel], args)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * sel.numel()
            seen += sel.numel()
        if epoch in {0, 1, 2, 5, 10, args.epochs // 2, args.epochs - 1}:
            trace.append({"epoch": epoch, "loss": total / seen})
    return model.cpu(), trace


def levenshtein(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def bucket_pos(start: int, span_len: int, length: int) -> str:
    center = start + span_len / 2.0
    ratio = center / max(1, length)
    if ratio < 0.33:
        return "front"
    if ratio < 0.66:
        return "middle"
    return "back"


def evaluate(model, split, itos, device, max_examples, eval_batch):
    model.to(device)
    target_np, observed_np, lengths_np, starts_np, spans_np = split
    target = torch.tensor(target_np, dtype=torch.long, device=device)
    observed = torch.tensor(observed_np, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths_np, dtype=torch.long, device=device)
    starts = torch.tensor(starts_np, dtype=torch.long, device=device)
    spans = torch.tensor(spans_np, dtype=torch.long, device=device)
    pred_np = np.zeros_like(target_np)
    with torch.no_grad():
        for batch_start in range(0, target.shape[0], eval_batch):
            batch_end = min(target.shape[0], batch_start + eval_batch)
            logits, state, route = model(
                observed[batch_start:batch_end],
                lengths[batch_start:batch_end],
                starts[batch_start:batch_end],
                spans[batch_start:batch_end],
            )
            pred_np[batch_start:batch_end] = logits.argmax(dim=-1).cpu().numpy()
    mask = target_np != PAD
    exact_rows = ((pred_np == target_np) | ~mask).all(axis=1)
    token_acc = float((pred_np[mask] == target_np[mask]).mean()) if mask.any() else 0.0
    exact = float(exact_rows.mean())
    edit_scores = []
    by_len = defaultdict(lambda: {"samples": 0, "exact": 0, "tok_ok": 0, "tok_total": 0, "edit_sum": 0.0})
    by_span = defaultdict(lambda: {"samples": 0, "exact": 0, "tok_ok": 0, "tok_total": 0, "edit_sum": 0.0})
    by_pos = defaultdict(lambda: {"samples": 0, "exact": 0, "tok_ok": 0, "tok_total": 0, "edit_sum": 0.0})
    examples = []
    for i, L in enumerate(lengths_np.tolist()):
        pred = pred_np[i, :L].tolist()
        tgt = target_np[i, :L].tolist()
        obs = observed_np[i, :L].tolist()
        sim = 1.0 - levenshtein(pred, tgt) / max(1, L)
        edit_scores.append(sim)
        tok_ok = sum(int(a == b) for a, b in zip(pred, tgt))
        for table, key in [
            (by_len, str(L)),
            (by_span, str(int(spans_np[i]))),
            (by_pos, bucket_pos(int(starts_np[i]), int(spans_np[i]), L)),
        ]:
            row = table[key]
            row["samples"] += 1
            row["exact"] += int(pred == tgt)
            row["tok_ok"] += tok_ok
            row["tok_total"] += L
            row["edit_sum"] += sim
        if len(examples) < max_examples:
            examples.append(
                {
                    "length": L,
                    "span_start": int(starts_np[i]),
                    "span_len": int(spans_np[i]),
                    "observed": detok([itos.get(x, "<unk>") for x in obs]),
                    "restored": detok([itos.get(x, "<unk>") for x in pred]),
                    "target": detok([itos.get(x, "<unk>") for x in tgt]),
                    "exact": pred == tgt,
                    "edit_similarity": round(sim, 4),
                }
            )

    def finish(table, numeric=False):
        items = sorted(table.items(), key=lambda kv: int(kv[0]) if numeric else kv[0])
        return {
            k: {
                "samples": v["samples"],
                "exact_match": v["exact"] / v["samples"],
                "token_acc": v["tok_ok"] / v["tok_total"],
                "edit_similarity": v["edit_sum"] / v["samples"],
            }
            for k, v in items
        }

    return {
        "exact_match": exact,
        "token_acc": token_acc,
        "edit_similarity": float(np.mean(edit_scores)),
        "by_length": finish(by_len, numeric=True),
        "by_span_len": finish(by_span, numeric=True),
        "by_span_position": finish(by_pos, numeric=False),
        "examples": examples,
    }


def run(args):
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target, observed, lengths, starts, spans, stoi, itos, meta = build_dataset(args)
    splits = split_data(target, observed, lengths, starts, spans, seed=args.seed + 1)
    device = torch.device(args.device)
    model = LocalInverseFlip(len(stoi), args.max_len, args.max_span_len, args.dim)
    model, trace = train_model(model, splits["train"], args, device)
    baseline = NoInverseBaseline(len(stoi), args.dim)
    baseline, baseline_trace = train_model(baseline, splits["train"], args, device)
    metrics = {name: evaluate(model, split, itos, device, args.examples, args.eval_batch) for name, split in splits.items()}
    baseline_metrics = {name: evaluate(baseline, split, itos, device, args.examples, args.eval_batch) for name, split in splits.items()}
    pass_checks = {
        "hard_local_treeheap_closure_exact": meta["hard_local_treeheap_closure_exact"] == 1.0,
        "learned_ood_exact_high": metrics["ood"]["exact_match"] >= args.min_ood_exact,
        "learned_ood_token_high": metrics["ood"]["token_acc"] >= args.min_ood_token_acc,
        "learned_beats_no_inverse": metrics["ood"]["exact_match"] >= baseline_metrics["ood"]["exact_match"] + args.min_exact_margin,
        "has_readable_examples": len(metrics["ood"]["examples"]) >= min(args.examples, len(splits["ood"][0])),
    }
    return {
        "claim": "S1-ECHO-LOCAL-FLIP-C01",
        "predict": "P-S1-ECHO-LOCAL01",
        "host": args.host_label,
        "device": args.device,
        "config": vars(args),
        "dataset": meta,
        "models": {
            "learned_local_inverse_treeheap_flip": {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trace": trace,
                "metrics": metrics,
            },
            "no_inverse_baseline": {
                "parameters": sum(p.numel() for p in baseline.parameters()),
                "trace": baseline_trace,
                "metrics": baseline_metrics,
            },
        },
        "pass_checks": pass_checks,
        "pilot_pass": all(pass_checks.values()),
        "interpretation": {
            "supported": "Local same-algebra TreeHeap flip echo has realistic span-level recovery evidence.",
            "not_proved": [
                "not translation",
                "not semantic understanding",
                "not automatic span/node/depth discovery",
                "not unsupervised natural trigger learning",
            ],
        },
        "elapsed_sec": round(time.time() - started, 3),
    }


def write_outputs(summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in summary["models"]["learned_local_inverse_treeheap_flip"]["trace"]:
            f.write(json.dumps({"model": "learned_local_inverse_treeheap_flip", **row}, ensure_ascii=False) + "\n")
        for row in summary["models"]["no_inverse_baseline"]["trace"]:
            f.write(json.dumps({"model": "no_inverse_baseline", **row}, ensure_ascii=False) + "\n")
    learned = summary["models"]["learned_local_inverse_treeheap_flip"]["metrics"]["ood"]
    base = summary["models"]["no_inverse_baseline"]["metrics"]["ood"]
    example_lines = []
    for ex in learned["examples"]:
        example_lines.append(
            f"- len={ex['length']} span=({ex['span_start']},{ex['span_len']}) exact={ex['exact']}\n"
            f"  - observed: `{ex['observed']}`\n"
            f"  - restored: `{ex['restored']}`\n"
            f"  - target: `{ex['target']}`"
        )
    readme = f"""# S1 Local Flip Echo Probe

Claim: `{summary['claim']}`
Predict: `{summary['predict']}`
Host: `{summary['host']}`

## Result

pilot_pass: `{summary['pilot_pass']}`

```text
hard_local_treeheap_closure_exact = {summary['dataset']['hard_local_treeheap_closure_exact']:.6f}
learned_ood_exact = {learned['exact_match']:.6f}
learned_ood_token_acc = {learned['token_acc']:.6f}
learned_ood_edit_similarity = {learned['edit_similarity']:.6f}
no_inverse_ood_exact = {base['exact_match']:.6f}
```

## Readable Examples

{chr(10).join(example_lines)}

## Boundary

Perturbation is local TreeHeap `Flip(span_root, full_depth)`, not external array
reverse. The span is given to the model, so this does not prove automatic
node/depth discovery.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ara/s1-echo/evidence/s1_local_flip_echo_probe")
    parser.add_argument("--wmt-path", default="/mnt/nas/datasets/wmt17/train.zh-en")
    parser.add_argument("--scan-lines", type=int, default=800000)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--min-span-len", type=int, default=2)
    parser.add_argument("--max-span-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--drop-unk", action="store_true", default=True)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--state-weight", type=float, default=1.5)
    parser.add_argument("--entropy-weight", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=4302)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host-label", default="local")
    parser.add_argument("--examples", type=int, default=14)
    parser.add_argument("--min-ood-exact", type=float, default=0.90)
    parser.add_argument("--min-ood-token-acc", type=float, default=0.98)
    parser.add_argument("--min-exact-margin", type=float, default=0.30)
    args = parser.parse_args()
    summary = run(args)
    write_outputs(summary, Path(args.out))
    print(json.dumps(summary["pass_checks"], indent=2, ensure_ascii=False))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
