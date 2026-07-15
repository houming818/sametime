#!/usr/bin/env python3
"""WMT translation-only test of multi-scale TreeHeap memory masking."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
from s3_residual_treeheap_forest_pretrain import ParameterTreeHeapKernel


VARIANTS = ("plain", "structured_mask", "random_mask")


class EmergentMaskTreeHeap(base.Seq2SeqBase):
    def __init__(
        self,
        vocab: int,
        dim: int,
        hidden: int,
        rank: int,
        heads: int,
        variant: str,
        mask_probability: float,
        mask_cuts: int,
        max_mask_depth: int,
    ):
        super().__init__(vocab, dim, hidden)
        self.embedding = nn.Embedding(vocab, dim)
        self.kernel = ParameterTreeHeapKernel(dim, rank, heads)
        self.norm = nn.LayerNorm(dim)
        self.variant = variant
        self.mask_probability = mask_probability
        self.mask_cuts = mask_cuts
        self.max_mask_depth = max_mask_depth

    def tree(self, src: torch.Tensor, length: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        width = src.shape[1]
        node = self.embedding(src)
        node_mask = torch.arange(width, device=src.device)[None] < length[:, None]
        levels = [node]
        masks = [node_mask]
        while width > 1:
            if width % 2:
                node = torch.cat((node, torch.zeros_like(node[:, :1])), dim=1)
                node_mask = torch.cat((node_mask, torch.zeros_like(node_mask[:, :1])), dim=1)
            left, right = node[:, 0::2], node[:, 1::2]
            left_mask, right_mask = node_mask[:, 0::2], node_mask[:, 1::2]
            initial = (left + right) * math.sqrt(0.5)
            parent, _, _ = self.kernel(initial, left, right)
            parent = self.norm(parent)
            node = torch.where(
                (left_mask & right_mask)[:, :, None],
                parent,
                torch.where(left_mask[:, :, None], left, right),
            )
            node_mask = left_mask | right_mask
            levels.append(node)
            masks.append(node_mask)
            width = node.shape[1]
        return levels, masks

    def mask_program(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        masks: Sequence[torch.Tensor],
        control: str,
        zero_selected_roots: bool,
        stochastic: bool,
    ) -> List[torch.Tensor]:
        visible = [mask.clone() for mask in masks]
        original = [mask.clone() for mask in masks]
        batch = src.shape[0]
        for row in range(batch):
            source_hash = int((src[row].long() * (torch.arange(src.shape[1], device=src.device) + 1)).sum())
            seed = source_hash + 7919 * (row + 1)
            if stochastic:
                seed += int(torch.randint(0, 2**20, (1,), device=src.device))
            rng = random.Random(seed)
            if stochastic and rng.random() > self.mask_probability:
                continue
            structured = [mask[row].clone() for mask in masks]
            selected_roots: List[Tuple[int, int]] = []
            maximum = min(self.max_mask_depth, len(masks) - 1, int(math.log2(max(2, int(length[row])))))
            if maximum < 1:
                continue
            for _ in range(self.mask_cuts):
                depth = rng.randint(1, maximum)
                span = 1 << depth
                root_count = max(1, int(length[row]) // span)
                root_index = rng.randrange(root_count)
                start = root_index * span
                for level in range(depth):
                    first = start >> level
                    count = 1 << (depth - level)
                    structured[level][first:first + count] = False
                selected_roots.append((depth, root_index))

            if control == "structured":
                for level in range(len(visible)):
                    visible[level][row] = structured[level]
            elif control == "random":
                for level in range(len(visible) - 1):
                    hidden_count = int((original[level][row] & ~structured[level]).sum())
                    candidates = torch.where(original[level][row])[0].tolist()
                    rng.shuffle(candidates)
                    for index in candidates[:hidden_count]:
                        visible[level][row, index] = False
            else:
                raise ValueError(control)

            if zero_selected_roots:
                for depth, root_index in selected_roots:
                    visible[depth][row, root_index] = False
        return visible

    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        levels, masks = self.tree(src, length)
        stochastic = self.training and mode == "full" and self.variant != "plain"
        if stochastic:
            control = "structured" if self.variant == "structured_mask" else "random"
            masks = self.mask_program(src, length, masks, control, False, True)
        elif mode in ("cut", "cut_root_zero"):
            masks = self.mask_program(src, length, masks, "structured", mode == "cut_root_zero", False)
        elif mode != "full":
            raise ValueError(f"unknown encode mode: {mode}")
        return torch.cat(levels, dim=1), torch.cat(masks, dim=1)


def make_loaders(rows, cfg, pad: int):
    splits = [
        rows[: cfg.train_samples],
        rows[cfg.train_samples : cfg.train_samples + cfg.valid_samples],
        rows[cfg.train_samples + cfg.valid_samples :],
    ]
    loaders = []
    for index, split in enumerate(splits):
        generator = torch.Generator().manual_seed(cfg.seed + index)
        loaders.append(
            DataLoader(
                base.ParallelDataset(split),
                batch_size=cfg.batch_size,
                shuffle=index == 0,
                generator=generator,
                num_workers=cfg.num_workers,
                collate_fn=base.collate(pad),
                pin_memory=cfg.device.startswith("cuda"),
            )
        )
    return loaders


def finite_gradients(model: nn.Module) -> bool:
    values = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(values) and all(bool(torch.isfinite(value).all()) for value in values)


def train_variant(variant, initial_state, rows, cfg, vocab, pad, bos, eos, sp, args, out):
    train_loader, valid_loader, test_loader = make_loaders(rows, cfg, pad)
    model = EmergentMaskTreeHeap(
        vocab,
        cfg.dim,
        cfg.hidden,
        args.rank,
        args.heads,
        variant,
        args.mask_probability,
        args.mask_cuts,
        args.max_mask_depth,
    ).to(cfg.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    use_amp = cfg.device.startswith("cuda") and args.precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trace = []
    best_nll = float("inf")
    best = None
    gradients_ok = True
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = count = 0
        for src, length, target, _ in train_loader:
            src, length, target = src.to(cfg.device), length.to(cfg.device), target.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, length, target, bos)
                loss = base.ce(logits, target, pad)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_ok = gradients_ok and finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        valid = base.evaluate(model, valid_loader, cfg, pad, bos, eos, sp)
        row = {
            "epoch": epoch,
            "train_nll": total / max(1, count),
            **{key: value for key, value in valid.items() if key != "examples"},
        }
        trace.append(row)
        print(f"[{variant}] epoch={epoch} train={row['train_nll']:.4f} valid={row['nll']:.4f} bleu={row['token_bleu4']:.3f}", flush=True)
        if valid["nll"] < best_nll:
            best_nll = float(valid["nll"])
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best)
    checkpoint = out / f"checkpoint_{variant}.pt"
    torch.save({"variant": variant, "state_dict": best, "config": asdict(cfg), "args": vars(args)}, checkpoint)
    evaluations = {
        mode: base.evaluate(model, test_loader, cfg, pad, bos, eos, sp, mode)
        for mode in ("full", "cut", "cut_root_zero")
    }
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": checkpoint.name,
        "seconds": time.time() - started,
        "finite_gradients": gradients_ok,
        "trace": trace,
        "evaluation": evaluations,
        "root_contribution": evaluations["cut_root_zero"]["nll"] - evaluations["cut"]["nll"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_treeheap_emergent_mask_translation")
    ap.add_argument("--seed", type=int, default=71511)
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--valid-samples", type=int, default=500)
    ap.add_argument("--test-samples", type=int, default=500)
    ap.add_argument("--max-scan", type=int, default=100000)
    ap.add_argument("--min-len", type=int, default=9)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mask-probability", type=float, default=0.5)
    ap.add_argument("--mask-cuts", type=int, default=2)
    ap.add_argument("--max-mask-depth", type=int, default=3)
    ap.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    cfg = base.Config(
        data=args.data,
        spm_model=args.spm_model,
        evidence_dir=args.evidence_dir,
        model="emergent_mask",
        seed=args.seed,
        train_samples=args.train_samples,
        valid_samples=args.valid_samples,
        test_samples=args.test_samples,
        max_scan=args.max_scan,
        min_len=args.min_len,
        max_len=args.max_len,
        dim=args.dim,
        hidden=args.hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    rows, pieces = base.load_rows(cfg, sp)
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    prototype = EmergentMaskTreeHeap(
        vocab, cfg.dim, cfg.hidden, args.rank, args.heads, "plain",
        args.mask_probability, args.mask_cuts, args.max_mask_depth,
    )
    initial_state = copy.deepcopy(prototype.state_dict())
    del prototype

    out = Path(args.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {}
    for variant in VARIANTS:
        torch.manual_seed(args.seed)
        results[variant] = train_variant(
            variant, initial_state, rows, cfg, vocab, pad, bos, eos, sp, args, out
        )

    plain = results["plain"]
    structured = results["structured_mask"]
    random_control = results["random_mask"]
    clean_gain = plain["evaluation"]["full"]["nll"] - structured["evaluation"]["full"]["nll"]
    cut_gain = plain["evaluation"]["cut"]["nll"] - structured["evaluation"]["cut"]["nll"]
    random_cut_gain = random_control["evaluation"]["cut"]["nll"] - structured["evaluation"]["cut"]["nll"]
    root_gain = structured["root_contribution"] - plain["root_contribution"]
    gates = {
        "P1_clean_s2_gain": clean_gain >= 0.03,
        "P2_structured_cut_gain": cut_gain >= 0.10,
        "P3_upper_state_causal_gain": root_gain >= 0.05,
        "P4_beats_equal_random_mask": random_cut_gain >= 0.05,
        "P5_finite_and_nonempty": all(
            result["finite_gradients"] and result["evaluation"]["full"]["token_bleu4"] > 0
            for result in results.values()
        ),
    }
    summary = {
        "claim": "S2-TREEHEAP-EMERGENT-MASK-C01",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {"direction": "en_to_zh", "rows": len(rows), "vocab": vocab},
        "results": results,
        "derived": {
            "clean_nll_gain_over_plain": clean_gain,
            "cut_nll_gain_over_plain": cut_gain,
            "cut_nll_gain_over_random_mask": random_cut_gain,
            "root_contribution_gain_over_plain": root_gain,
        },
        "gates": gates,
        "decision": "supported_smoke" if all(gates.values()) else "partial" if any(list(gates.values())[:4]) else "not_supported",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "trace.jsonl").write_text(
        "\n".join(json.dumps({"variant": variant, **row}) for variant, result in results.items() for row in result["trace"]) + "\n",
        encoding="utf-8",
    )
    (out / "examples.json").write_text(
        json.dumps({variant: result["evaluation"]["full"]["examples"] for variant, result in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# S2 TreeHeap Emergent-Mask Translation\n\n```json\n"
        + json.dumps({"derived": summary["derived"], "gates": gates, "decision": summary["decision"]}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": summary["derived"], "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
