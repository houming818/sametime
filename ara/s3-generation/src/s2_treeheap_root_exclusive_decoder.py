#!/usr/bin/env python3
"""WMT proof for a root-exclusive variable-resolution TreeHeap decoder."""
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
from typing import List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
from s2_treeheap_emergent_mask_translation import (
    EmergentMaskTreeHeap,
    finite_gradients,
    make_loaders,
)


VARIANTS = ("full_tree", "exclusive_tree", "exclusive_flat")


class RootExclusiveTreeHeap(EmergentMaskTreeHeap):
    def __init__(self, vocab, dim, hidden, rank, heads, variant, max_frontier_depth):
        super().__init__(vocab, dim, hidden, rank, heads, variant, 1.0, 1, max_frontier_depth)
        self.variant = variant
        self.max_frontier_depth = max_frontier_depth
        self.flat_project = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def flat_levels(self, src: torch.Tensor, length: torch.Tensor) -> List[torch.Tensor]:
        width = src.shape[1]
        leaf = self.embedding(src)
        valid = (torch.arange(width, device=src.device)[None] < length[:, None]).to(leaf.dtype)
        total = leaf * valid[:, :, None]
        count = valid
        levels = [leaf]
        while width > 1:
            if width % 2:
                total = torch.cat((total, torch.zeros_like(total[:, :1])), dim=1)
                count = torch.cat((count, torch.zeros_like(count[:, :1])), dim=1)
            total = total[:, 0::2] + total[:, 1::2]
            count = count[:, 0::2] + count[:, 1::2]
            mean = total / count.clamp_min(1)[:, :, None]
            levels.append(self.flat_project(mean))
            width = total.shape[1]
        return levels

    def frontier_visibility(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        masks: Sequence[torch.Tensor],
        stochastic: bool,
        remove_internal_roots: bool,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        visible = [torch.zeros_like(mask) for mask in masks]
        internal_count = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
        frontier_count = torch.zeros_like(internal_count)
        positions = torch.arange(src.shape[1], device=src.device) + 1
        for row in range(src.shape[0]):
            source_hash = int((src[row].long() * positions).sum())
            if stochastic:
                source_hash += int(torch.randint(0, 2**20, (1,), device=src.device))
            rng = random.Random(source_hash + 104729 * (row + 1))
            pos = 0
            size = int(length[row])
            while pos < size:
                maximum = 0
                while maximum < self.max_frontier_depth:
                    next_depth = maximum + 1
                    span = 1 << next_depth
                    if pos % span or pos + span > size:
                        break
                    maximum = next_depth
                depth = rng.randint(0, maximum) if maximum else 0
                index = pos >> depth
                if not (remove_internal_roots and depth > 0):
                    visible[depth][row, index] = True
                frontier_count[row] += 1
                internal_count[row] += int(depth > 0)
                pos += 1 << depth
        return visible, frontier_count, internal_count

    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full"):
        tree_levels, tree_masks = self.tree(src, length)
        flat_levels = None

        if mode == "leaf_only":
            visible = [torch.zeros_like(mask) for mask in tree_masks]
            visible[0] = tree_masks[0].clone()
            levels = tree_levels
        elif mode == "full" and self.training:
            if self.variant == "full_tree":
                visible = [mask.clone() for mask in tree_masks]
                levels = tree_levels
            else:
                visible, _, _ = self.frontier_visibility(src, length, tree_masks, True, False)
                if self.variant == "exclusive_flat":
                    flat_levels = self.flat_levels(src, length)
                levels = flat_levels if flat_levels is not None else tree_levels
        elif mode == "full":
            if self.variant == "full_tree":
                visible = [mask.clone() for mask in tree_masks]
                levels = tree_levels
            else:
                visible, _, _ = self.frontier_visibility(src, length, tree_masks, False, False)
                if self.variant == "exclusive_flat":
                    flat_levels = self.flat_levels(src, length)
                levels = flat_levels if flat_levels is not None else tree_levels
        elif mode in ("frontier", "frontier_flat", "frontier_root_zero"):
            visible, _, _ = self.frontier_visibility(
                src, length, tree_masks, False, mode == "frontier_root_zero"
            )
            use_flat = mode == "frontier_flat" or (
                mode == "frontier_root_zero" and self.variant == "exclusive_flat"
            )
            levels = self.flat_levels(src, length) if use_flat else tree_levels
        else:
            raise ValueError(f"unknown mode: {mode}")

        memory = torch.cat(levels, dim=1)
        mask = torch.cat(visible, dim=1)
        memory = memory * mask[:, :, None]
        empty = ~mask.any(-1)
        if bool(empty.any()):
            mask[empty, 0] = True
        return memory, mask

    @torch.no_grad()
    def compression(self, src: torch.Tensor, length: torch.Tensor):
        _, masks = self.tree(src, length)
        _, frontier_count, internal_count = self.frontier_visibility(src, length, masks, False, False)
        return frontier_count, internal_count


def train_variant(variant, initial_state, rows, cfg, vocab, pad, bos, eos, sp, args, out):
    train_loader, valid_loader, test_loader = make_loaders(rows, cfg, pad)
    model = RootExclusiveTreeHeap(
        vocab, cfg.dim, cfg.hidden, args.rank, args.heads, variant, args.max_frontier_depth
    ).to(cfg.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
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
            logits = model(src, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_ok = gradients_ok and finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
        for mode in ("full", "frontier", "frontier_flat", "frontier_root_zero", "leaf_only")
    }
    frontier_total = internal_total = token_total = 0
    for src, length, _, _ in test_loader:
        src, length = src.to(cfg.device), length.to(cfg.device)
        frontier, internal = model.compression(src, length)
        frontier_total += int(frontier.sum())
        internal_total += int(internal.sum())
        token_total += int(length.sum())
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": checkpoint.name,
        "seconds": time.time() - started,
        "finite_gradients": gradients_ok,
        "trace": trace,
        "evaluation": evaluations,
        "frontier_states_per_token": frontier_total / max(1, token_total),
        "internal_roots_per_frontier": internal_total / max(1, frontier_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm-model", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_treeheap_root_exclusive_decoder")
    ap.add_argument("--seed", type=int, default=71521)
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
    ap.add_argument("--max-frontier-depth", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    cfg = base.Config(
        data=args.data, spm_model=args.spm_model, evidence_dir=args.evidence_dir,
        model="root_exclusive", seed=args.seed, train_samples=args.train_samples,
        valid_samples=args.valid_samples, test_samples=args.test_samples,
        max_scan=args.max_scan, min_len=args.min_len, max_len=args.max_len,
        dim=args.dim, hidden=args.hidden, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, device=args.device, num_workers=args.num_workers,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    rows, pieces = base.load_rows(cfg, sp)
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    prototype = RootExclusiveTreeHeap(
        vocab, cfg.dim, cfg.hidden, args.rank, args.heads, "full_tree", args.max_frontier_depth
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

    full = results["full_tree"]
    tree = results["exclusive_tree"]
    flat = results["exclusive_flat"]
    clean_gain = full["evaluation"]["full"]["nll"] - tree["evaluation"]["full"]["nll"]
    recursive_gain = flat["evaluation"]["full"]["nll"] - tree["evaluation"]["full"]["nll"]
    root_damage = tree["evaluation"]["frontier_root_zero"]["nll"] - tree["evaluation"]["full"]["nll"]
    full_forced_damage = (
        full["evaluation"]["frontier_root_zero"]["nll"]
        - full["evaluation"]["frontier"]["nll"]
    )
    root_training_gain = root_damage - full_forced_damage
    gates = {
        "P1_clean_s2_gain": clean_gain >= 0.03,
        "P2_recursive_over_flat": recursive_gain >= 0.05,
        "P3_compressed_roots_causal": root_damage >= 0.10,
        "P4_exclusive_training_causal_gain": root_training_gain >= 0.05,
        "P5_finite_and_nonempty": all(
            result["finite_gradients"] and result["evaluation"]["full"]["token_bleu4"] > 0
            for result in results.values()
        ),
    }
    summary = {
        "claim": "S2-TREEHEAP-ROOT-EXCLUSIVE-C01",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {"direction": "en_to_zh", "rows": len(rows), "vocab": vocab},
        "results": results,
        "derived": {
            "clean_nll_gain_over_full_tree": clean_gain,
            "recursive_nll_gain_over_flat_span": recursive_gain,
            "exclusive_tree_compressed_root_damage": root_damage,
            "full_tree_forced_frontier_root_damage": full_forced_damage,
            "exclusive_training_root_causal_gain": root_training_gain,
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
        "# S2 TreeHeap Root-Exclusive Decoder\n\n```json\n"
        + json.dumps({"derived": summary["derived"], "gates": gates, "decision": summary["decision"]}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": summary["derived"], "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
