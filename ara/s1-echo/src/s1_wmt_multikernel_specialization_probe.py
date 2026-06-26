#!/usr/bin/env python3
"""S1 WMT multi-kernel specialization probe.

This probe asks a narrow question:

    Does a TreeHeap kernel bank naturally specialize under structural
    perturbation tasks, in a way that a single shared kernel cannot?

The data are real WMT English SentencePiece token sequences. The tasks are
controlled structure tasks over those sequences:

    echo         : reconstruct the full sequence
    mask_restore : reconstruct the original sequence after one token is masked
    left_query   : read the left subheap
    right_query  : read the right subheap
    mirror       : reconstruct the reversed sequence

This is not translation and not a semantic world-model proof. It is a kernel
specialization proof for S1 write/read machinery.
"""

from __future__ import annotations

import argparse
import json
import math
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


TASKS = ["echo", "mask_restore", "left_query", "right_query", "mirror"]
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
            # The local WMT17 file is tab-separated EN/ZH in current storage.
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
        # Reserve vocab_limit - 1 for MASK. PAD is 0, real SP ids are shifted by +1.
        ids = [i for i in ids if 0 < i < vocab_limit - 1]
        if min_len <= len(ids) <= max_len:
            seqs.append(ids)
            if len(examples) < 5:
                examples.append(text)
            if len(seqs) >= samples:
                break
    if len(seqs) < samples:
        raise RuntimeError(f"only collected {len(seqs)} sequences, need {samples}")
    return seqs, examples


@dataclass
class BatchSpec:
    tokens: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    task: torch.Tensor
    mask_pos: torch.Tensor


class StructuralTaskDataset(Dataset):
    def __init__(
        self,
        seqs: List[List[int]],
        max_len: int,
        vocab_limit: int,
        train_mode: bool,
        seed: int,
    ) -> None:
        self.seqs = seqs
        self.max_len = max_len
        self.vocab_limit = vocab_limit
        self.mask_token = vocab_limit - 1
        self.train_mode = train_mode
        self.rng = random.Random(seed)

        # Eval is deterministic and contains every task for every sequence.
        self.eval_index: List[Tuple[int, int]] = []
        if not train_mode:
            for i in range(len(seqs)):
                for t in range(len(TASKS)):
                    self.eval_index.append((i, t))

    def __len__(self) -> int:
        return len(self.seqs) if self.train_mode else len(self.eval_index)

    def _pad(self, ids: List[int]) -> List[int]:
        return ids + [PAD] * (self.max_len - len(ids))

    def _make(self, seq: List[int], task_id: int, mask_pos: int) -> BatchSpec:
        padded = self._pad(seq)
        inp = list(padded)
        target = [PAD] * self.max_len
        loss_mask = [0] * self.max_len

        if task_id == 0:  # echo
            target = list(padded)
            loss_mask = [1 if x != PAD else 0 for x in padded]
        elif task_id == 1:  # mask_restore
            inp[mask_pos] = self.mask_token
            target = list(padded)
            loss_mask = [1 if x != PAD else 0 for x in padded]
        elif task_id == 2:  # left_query
            left = padded[: self.max_len // 2]
            target[: len(left)] = left
            loss_mask[: len(left)] = [1 if x != PAD else 0 for x in left]
        elif task_id == 3:  # right_query
            right = padded[self.max_len // 2 :]
            target[: len(right)] = right
            loss_mask[: len(right)] = [1 if x != PAD else 0 for x in right]
        elif task_id == 4:  # mirror
            nonpad = [x for x in padded if x != PAD]
            rev = list(reversed(nonpad))
            target[: len(rev)] = rev
            loss_mask[: len(rev)] = [1] * len(rev)
        else:
            raise ValueError(task_id)

        return BatchSpec(
            tokens=torch.tensor(inp, dtype=torch.long),
            targets=torch.tensor(target, dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
            task=torch.tensor(task_id, dtype=torch.long),
            mask_pos=torch.tensor(mask_pos, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> BatchSpec:
        if self.train_mode:
            seq = self.seqs[idx]
            task_id = self.rng.randrange(len(TASKS))
            mask_pos = self.rng.randrange(len(seq))
        else:
            seq_idx, task_id = self.eval_index[idx]
            seq = self.seqs[seq_idx]
            mask_pos = min(len(seq) - 1, len(seq) // 2)
        return self._make(seq, task_id, mask_pos)


def collate(batch: List[BatchSpec]) -> Dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([b.tokens for b in batch]),
        "targets": torch.stack([b.targets for b in batch]),
        "loss_mask": torch.stack([b.loss_mask for b in batch]),
        "task": torch.stack([b.task for b in batch]),
        "mask_pos": torch.stack([b.mask_pos for b in batch]),
    }


class KernelMLP(nn.Module):
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


class MultiKernelTreeHeap(nn.Module):
    def __init__(self, vocab: int, max_len: int, dim: int, kernels: int, tasks: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.dim = dim
        self.kernels = kernels
        self.tasks = tasks
        self.start = max_len - 1
        self.nodes = max_len * 2 - 1

        self.token_emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.task_emb = nn.Embedding(tasks, dim)
        self.kernel_bank = nn.ModuleList([KernelMLP(dim) for _ in range(kernels)])
        self.gate = nn.Linear(dim, kernels)
        self.read = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, max_len * vocab),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        task: torch.Tensor,
        ablate_kernel: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.shape[0]
        device = tokens.device
        pos = torch.arange(self.max_len, device=device)
        leaves = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]

        states: List[torch.Tensor | None] = [None for _ in range(self.nodes)]
        for j in range(self.max_len):
            states[self.start + j] = leaves[:, j, :]

        task_vec = self.task_emb(task)
        gate_logits = self.gate(task_vec)
        if ablate_kernel is not None and self.kernels > 1:
            gate_logits[:, ablate_kernel] = -1e9
        weights = torch.softmax(gate_logits, dim=-1)

        for idx in range(self.start - 1, -1, -1):
            left = states[2 * idx + 1]
            right = states[2 * idx + 2]
            assert left is not None and right is not None
            cands = torch.stack([k(left, right) for k in self.kernel_bank], dim=1)
            mixed = torch.sum(cands * weights[:, :, None], dim=1)
            states[idx] = mixed

        # Select task-relevant node:
        # full tasks use root; subheap queries use left/right child.
        root = states[0]
        left_child = states[1]
        right_child = states[2]
        assert root is not None and left_child is not None and right_child is not None
        selected = root.clone()
        selected = torch.where((task == 2)[:, None], left_child, selected)
        selected = torch.where((task == 3)[:, None], right_child, selected)

        logits = self.read(torch.cat([selected, task_vec], dim=-1)).view(
            bsz, self.max_len, self.vocab
        )
        return logits, weights


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


@torch.no_grad()
def evaluate(
    model: MultiKernelTreeHeap,
    loader: DataLoader,
    device: torch.device,
    ablate_kernel: int | None = None,
) -> Dict[str, object]:
    model.eval()
    task_stats = {
        name: {"tokens": 0, "token_ok": 0, "seq": 0, "seq_ok": 0}
        for name in TASKS
    }
    gate_sum = torch.zeros(len(TASKS), model.kernels, device=device)
    gate_count = torch.zeros(len(TASKS), device=device)

    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["loss_mask"].to(device)
        task = batch["task"].to(device)
        logits, weights = model(tokens, task, ablate_kernel=ablate_kernel)
        pred = logits.argmax(dim=-1)
        ok = (pred == targets) & mask
        for tid, name in enumerate(TASKS):
            rows = task == tid
            if not rows.any():
                continue
            row_mask = mask[rows]
            row_ok = ok[rows]
            task_stats[name]["tokens"] += int(row_mask.sum().item())
            task_stats[name]["token_ok"] += int(row_ok.sum().item())
            per_seq = (row_ok | ~row_mask).all(dim=1)
            task_stats[name]["seq"] += int(rows.sum().item())
            task_stats[name]["seq_ok"] += int(per_seq.sum().item())
            gate_sum[tid] += weights[rows].sum(dim=0)
            gate_count[tid] += int(rows.sum().item())

    out: Dict[str, object] = {}
    for name, st in task_stats.items():
        out[name] = {
            "token_acc": st["token_ok"] / max(1, st["tokens"]),
            "exact": st["seq_ok"] / max(1, st["seq"]),
        }
    gate_mean = gate_sum / gate_count.clamp_min(1)[:, None]
    out["gate_mean_by_task"] = {
        TASKS[i]: [float(x) for x in gate_mean[i].detach().cpu()]
        for i in range(len(TASKS))
    }
    ent = -(gate_mean.clamp_min(1e-9) * gate_mean.clamp_min(1e-9).log()).sum(dim=1)
    out["gate_entropy_by_task"] = {
        TASKS[i]: float(ent[i].detach().cpu()) for i in range(len(TASKS))
    }
    return out


def train_model(
    model: MultiKernelTreeHeap,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    entropy_weight: float,
    balance_weight: float,
) -> Tuple[MultiKernelTreeHeap, List[Dict[str, float]]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trace: List[Dict[str, float]] = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["loss_mask"].to(device)
            task = batch["task"].to(device)
            logits, weights = model(tokens, task)
            loss = masked_ce(logits, targets, mask)
            if model.kernels > 1:
                entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=1).mean()
                mean_gate = weights.mean(dim=0)
                balance = ((mean_gate - (1.0 / model.kernels)) ** 2).sum()
                loss = loss + entropy_weight * entropy + balance_weight * balance
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * tokens.shape[0]
            n += tokens.shape[0]
        if ep in {0, epochs // 2, epochs - 1}:
            ev = evaluate(model, test_loader, device)
            mean_exact = np.mean([ev[t]["exact"] for t in TASKS])  # type: ignore[index]
            trace.append({"epoch": ep, "loss": total / max(1, n), "test_mean_exact": float(mean_exact)})
    return model, trace


def mean_exact(metrics: Dict[str, object]) -> float:
    return float(np.mean([metrics[t]["exact"] for t in TASKS]))  # type: ignore[index]


def ablation_table(
    model: MultiKernelTreeHeap,
    loader: DataLoader,
    device: torch.device,
    base: Dict[str, object],
) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    for k in range(model.kernels):
        ev = evaluate(model, loader, device, ablate_kernel=k)
        drops = {
            t: float(base[t]["exact"] - ev[t]["exact"])  # type: ignore[index]
            for t in TASKS
        }
        rows[f"drop_kernel_{k}"] = drops
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmt", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--scan-lines", type=int, default=100000)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--vocab-limit", type=int, default=2049)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--kernels", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entropy-weight", type=float, default=0.01)
    ap.add_argument("--balance-weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
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

    train_ds = StructuralTaskDataset(train_seqs, args.max_len, args.vocab_limit, True, args.seed)
    test_ds = StructuralTaskDataset(test_seqs, args.max_len, args.vocab_limit, False, args.seed)
    ood_ds = StructuralTaskDataset(ood_seqs, args.max_len, args.vocab_limit, False, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    models: Dict[str, Dict[str, object]] = {}
    for name, kernels in [("single_kernel_treeheap", 1), ("multi_kernel_treeheap", args.kernels)]:
        model = MultiKernelTreeHeap(args.vocab_limit, args.max_len, args.dim, kernels, len(TASKS))
        model, trace = train_model(
            model,
            train_loader,
            test_loader,
            device,
            args.epochs,
            args.lr,
            args.entropy_weight,
            args.balance_weight,
        )
        test_metrics = evaluate(model, test_loader, device)
        ood_metrics = evaluate(model, ood_loader, device)
        entry: Dict[str, object] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trace": trace,
            "test": test_metrics,
            "ood": ood_metrics,
            "test_mean_exact": mean_exact(test_metrics),
            "ood_mean_exact": mean_exact(ood_metrics),
        }
        if kernels > 1:
            entry["ood_ablation_exact_drop"] = ablation_table(model, ood_loader, device, ood_metrics)
        models[name] = entry

    single = models["single_kernel_treeheap"]
    multi = models["multi_kernel_treeheap"]
    improvement = float(multi["ood_mean_exact"] - single["ood_mean_exact"])
    gates = multi["ood"]["gate_mean_by_task"]  # type: ignore[index]
    task_argmax = {t: int(np.argmax(v)) for t, v in gates.items()}  # type: ignore[union-attr]
    unique_heads = len(set(task_argmax.values()))
    ablations = multi["ood_ablation_exact_drop"]  # type: ignore[index]
    max_drop = 0.0
    for row in ablations.values():  # type: ignore[union-attr]
        max_drop = max(max_drop, max(float(x) for x in row.values()))

    pilot_pass = (
        improvement >= 0.05
        and float(multi["ood_mean_exact"]) >= 0.65
        and unique_heads >= 2
        and max_drop >= 0.10
    )

    summary = {
        "claim": "S1-MK-C01",
        "predict": "P-S1-MK01",
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
            "vocab_limit_including_pad_and_mask": args.vocab_limit,
            "examples": examples,
        },
        "tasks": TASKS,
        "predict_statement": (
            "If structural perturbation tasks create useful gradient pressure, "
            "a TreeHeap kernel bank should outperform a single shared kernel, "
            "and kernel ablation/gates should show task-dependent specialization."
        ),
        "models": models,
        "derived": {
            "ood_mean_exact_improvement_multi_minus_single": improvement,
            "task_argmax_kernel": task_argmax,
            "unique_argmax_kernels": unique_heads,
            "max_ood_ablation_exact_drop": max_drop,
        },
        "pilot_pass": pilot_pass,
        "interpretation": {
            "supported": (
                "Multi-kernel TreeHeap specialization is supported if pilot_pass is true."
            ),
            "not_proved": [
                "not translation",
                "not semantic world model",
                "not natural syntax induction",
                "not proof that Transformer heads always specialize",
            ],
        },
        "elapsed_sec": round(time.time() - start, 3),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for name, entry in models.items():
            for row in entry["trace"]:  # type: ignore[index]
                f.write(json.dumps({"model": name, **row}) + "\n")
    (out_dir / "README.md").write_text(
        "# S1 WMT multi-kernel specialization probe\n\n"
        "Controlled structural perturbation tasks over real WMT English "
        "SentencePiece short sequences.\n\n"
        f"Decision: `S1-MK-C01 -> {'supported pilot' if pilot_pass else 'open/rejected pilot'}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
