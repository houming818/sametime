#!/usr/bin/env python3
"""Compare reversible adaptive/alternating lifting pumps on WMT massive."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
import s2_lifting_pump_wmt as prior


VARIANTS = (
    "flat_seq",
    "old_recursive",
    "learned_update",
    "alternate_fixed",
    "adaptive_alternate",
)


class LearnedUpdate(nn.Module):
    """Starts at A(D)=0.5D, then learns a bounded residual update."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, detail: torch.Tensor) -> torch.Tensor:
        return 0.5 * detail + 0.5 * torch.tanh(self.net(detail))

    def delta(self, detail: torch.Tensor) -> torch.Tensor:
        return self.forward(detail) - 0.5 * detail


class AdaptiveLiftingEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        heap_width: int,
        pad: int,
        learned_update: bool,
        alternate: bool,
    ):
        super().__init__()
        if heap_width < 2 or heap_width & (heap_width - 1):
            raise ValueError("heap_width must be a power of two")
        self.embedding = nn.Embedding(vocab, dim)
        self.predictor = prior.SharedPredictor(dim)
        self.update_kernel = LearnedUpdate(dim) if learned_update else None
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad = pad
        self.learned_update = learned_update
        self.alternate = alternate

    def update(self, detail: torch.Tensor) -> torch.Tensor:
        return self.update_kernel(detail) if self.update_kernel is not None else 0.5 * detail

    def update_delta(self, detail: torch.Tensor) -> torch.Tensor:
        if self.update_kernel is None:
            return torch.zeros_like(detail)
        return self.update_kernel.delta(detail)

    def right_anchor(self, depth: int) -> bool:
        return self.alternate and depth % 2 == 1

    def fold(self, src: torch.Tensor, length: torch.Tensor, pair_break_depth: int = -1):
        if src.shape[1] > self.heap_width:
            raise ValueError(f"source width {src.shape[1]} exceeds heap width {self.heap_width}")
        padded = torch.full(
            (src.shape[0], self.heap_width), self.pad,
            dtype=src.dtype, device=src.device,
        )
        padded[:, : src.shape[1]] = src
        leaf_mask = torch.arange(self.heap_width, device=src.device)[None] < length[:, None]
        leaf = self.embedding(padded) * leaf_mask[:, :, None]
        node, node_mask = leaf, leaf_mask
        details: List[torch.Tensor] = []
        masks: List[torch.Tensor] = [leaf_mask]
        for depth in range(self.depths):
            left, right = node[:, 0::2], node[:, 1::2]
            lm, rm = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == pair_break_depth:
                if self.right_anchor(depth):
                    left, lm = left.roll(1, dims=0), lm.roll(1, dims=0)
                else:
                    right, rm = right.roll(1, dims=0), rm.roll(1, dims=0)
            if self.right_anchor(depth):
                detail = left - self.predictor(right)
                parent = right + self.update(detail)
            else:
                detail = right - self.predictor(left)
                parent = left + self.update(detail)
            node_mask = lm | rm
            parent = parent * node_mask[:, :, None]
            detail = detail * node_mask[:, :, None]
            details.append(detail)
            masks.append(node_mask)
            node = parent
        return leaf, node[:, 0], details, masks

    def unfold(
        self,
        root: torch.Tensor,
        details: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        intervention: str = "native",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        node = root
        local_details = list(details)
        local_masks = list(masks)
        if intervention == "source_shuffle":
            node = node.roll(1, dims=0)
            local_details = [row.roll(1, dims=0) for row in local_details]
            local_masks = [row.roll(1, dims=0) for row in local_masks]
        elif intervention == "root_shuffle":
            node = node.roll(1, dims=0)
        elif intervention.startswith("detail_shuffle_"):
            depth = int(intervention.rsplit("_", 1)[1])
            local_details[depth] = local_details[depth].roll(1, dims=0)
        elif intervention != "native":
            raise ValueError(intervention)
        levels = [node[:, None]]
        level_masks = [local_masks[-1]]
        for depth in range(len(local_details) - 1, -1, -1):
            detail = local_details[depth]
            anchor = levels[-1] - self.update(detail)
            predicted = detail + self.predictor(anchor)
            if self.right_anchor(depth):
                left, right = predicted, anchor
            else:
                left, right = anchor, predicted
            expanded = torch.empty(
                left.shape[0], left.shape[1] * 2, left.shape[2],
                device=left.device, dtype=left.dtype,
            )
            expanded[:, 0::2], expanded[:, 1::2] = left, right
            expanded = expanded * local_masks[depth][:, :, None]
            levels.append(expanded)
            level_masks.append(local_masks[depth])
        return levels, level_masks

    def states(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        intervention: str = "native",
        pair_break_depth: int = -1,
    ):
        leaf, root, details, masks = self.fold(src, length, pair_break_depth)
        levels, level_masks = self.unfold(root, details, masks, intervention)
        return leaf, root, details, levels, level_masks


class AdaptiveRecursive(prior.S2Model):
    def __init__(
        self,
        vocab: int,
        dim: int,
        hidden: int,
        heap_width: int,
        pad: int,
        learned_update: bool,
        alternate: bool,
    ):
        super().__init__()
        self.encoder = AdaptiveLiftingEncoder(
            vocab, dim, heap_width, pad, learned_update, alternate,
        )
        self.decoder = prior.RecursiveDecoder(vocab, dim, hidden, self.encoder.depths)

    def states(self, src, length, intervention="native", pair_break_depth=-1):
        return self.encoder.states(src, length, intervention, pair_break_depth)

    def teacher(
        self, src, length, target, bos,
        intervention="native", route_mode="native", pair_break_depth=-1,
    ):
        _, _, _, levels, masks = self.states(src, length, intervention, pair_break_depth)
        return self.decoder.teacher(levels, masks, target, bos, route_mode)

    def greedy(self, src, length, bos, eos, max_len, intervention="native", route_mode="native"):
        _, _, _, levels, masks = self.states(src, length, intervention)
        return self.decoder.greedy(levels, masks, bos, eos, max_len, route_mode)


def make_model(name: str, vocab: int, dim: int, hidden: int, heap_width: int, pad: int):
    if name == "flat_seq":
        return prior.FlatModel(vocab, dim, hidden)
    if name == "old_recursive":
        return prior.LiftingRecursive(vocab, dim, hidden, heap_width, pad)
    if name == "learned_update":
        return AdaptiveRecursive(vocab, dim, hidden, heap_width, pad, True, False)
    if name == "alternate_fixed":
        return AdaptiveRecursive(vocab, dim, hidden, heap_width, pad, False, True)
    if name == "adaptive_alternate":
        return AdaptiveRecursive(vocab, dim, hidden, heap_width, pad, True, True)
    raise ValueError(name)


def load_rows(args, sp) -> Tuple[List[Tuple[List[int], List[int]]], dict]:
    required = args.train_samples + args.valid_samples + args.test_samples
    reservoir: List[Tuple[List[int], List[int]]] = []
    eligible = 0
    scanned = 0
    rng = random.Random(args.seed + 7103)
    with open(args.data, encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= args.max_scan:
                break
            scanned += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(args.source_col, args.target_col):
                continue
            source = sp.encode(parts[args.source_col], out_type=int)
            target = sp.encode(parts[args.target_col], out_type=int)
            if not (
                args.min_len <= len(source) <= args.max_len
                and args.min_len <= len(target) <= args.max_len
            ):
                continue
            row = (source + [sp.eos_id()], target + [sp.eos_id()])
            eligible += 1
            if len(reservoir) < required:
                reservoir.append(row)
            else:
                index = rng.randrange(eligible)
                if index < required:
                    reservoir[index] = row
    if len(reservoir) < required:
        raise RuntimeError(
            f"only {len(reservoir)} usable rows from {scanned}; need {required}"
        )
    rng.shuffle(reservoir)
    return reservoir, {
        "scanned": scanned,
        "eligible": eligible,
        "selected": len(reservoir),
        "sampling": "deterministic_reservoir_then_shuffle",
    }


@torch.no_grad()
def closure_audit(model, loader, device) -> dict:
    if not hasattr(model, "states"):
        return {}
    src, length, _, _ = next(iter(loader))
    src, length = src.to(device), length.to(device)
    leaf, _, _, levels, _ = model.states(src, length)
    difference = levels[-1] - leaf
    return {
        "state_mse": float(difference.square().mean()),
        "state_max_abs": float(difference.abs().max()),
    }


@torch.no_grad()
def update_audit(model, loader, device) -> dict:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, AdaptiveLiftingEncoder) or not encoder.learned_update:
        return {"delta_rms": 0.0, "delta_to_detail_rms": 0.0}
    src, length, _, _ = next(iter(loader))
    src, length = src.to(device), length.to(device)
    _, _, details, _ = encoder.fold(src, length)
    delta_square = detail_square = count = 0.0
    for detail in details:
        delta = encoder.update_delta(detail)
        delta_square += float(delta.square().sum())
        detail_square += float(detail.square().sum())
        count += detail.numel()
    delta_rms = math.sqrt(delta_square / max(1.0, count))
    detail_rms = math.sqrt(detail_square / max(1.0, count))
    return {
        "delta_rms": delta_rms,
        "detail_rms": detail_rms,
        "delta_to_detail_rms": delta_rms / max(1e-12, detail_rms),
    }


def train_variant(
    name, train_loader, valid_loader, test_loader,
    args, vocab, pad, bos, eos, sp, output,
):
    torch.manual_seed(args.seed + sum(map(ord, name)))
    model = make_model(name, vocab, args.dim, args.hidden, args.heap_width, pad).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trace = []
    best_nll, best = float("inf"), None
    gradients_ok = True
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = steps = 0
        for src, length, target, _ in train_loader:
            src = src.to(args.device, non_blocking=True)
            length = length.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits, _ = model.teacher(src, length, target, bos)
            loss = base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_ok = gradients_ok and prior.finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        valid = prior.evaluate(model, valid_loader, args, pad, bos, eos, sp)
        row = {
            "model": name,
            "epoch": epoch,
            "train_nll": total / max(1, steps),
            "valid_nll": valid["nll"],
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best)
    checkpoint = output / f"checkpoint_{name}.pt"
    torch.save({"name": name, "state_dict": best, "config": vars(args)}, checkpoint)
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": gradients_ok,
        "seconds": time.time() - started,
        "trace": trace,
        "test": prior.evaluate(model, test_loader, args, pad, bos, eos, sp, generate=True),
        "checkpoint": checkpoint.name,
        "closure": closure_audit(model, test_loader, args.device),
        "update": update_audit(model, test_loader, args.device),
    }
    if name == args.audit_variant:
        result["interventions"] = {
            "source_shuffle": prior.evaluate(
                model, test_loader, args, pad, bos, eos, sp,
                intervention="source_shuffle",
            ),
            "root_shuffle": prior.evaluate(
                model, test_loader, args, pad, bos, eos, sp,
                intervention="root_shuffle",
            ),
            "force_root": prior.evaluate(
                model, test_loader, args, pad, bos, eos, sp,
                route_mode="force_root",
            ),
            "force_leaf": prior.evaluate(
                model, test_loader, args, pad, bos, eos, sp,
                route_mode="force_leaf",
            ),
            "detail_shuffle": [
                prior.evaluate(
                    model, test_loader, args, pad, bos, eos, sp,
                    intervention=f"detail_shuffle_{depth}",
                )
                for depth in range(model.encoder.depths)
            ],
            "pair_break": [
                prior.evaluate(
                    model, test_loader, args, pad, bos, eos, sp,
                    pair_break_depth=depth,
                )
                for depth in range(model.encoder.depths)
            ],
        }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def build_gates(args, results: Dict[str, dict]) -> Tuple[dict, dict]:
    old_nll = results["old_recursive"]["test"]["nll"]
    candidate = results[args.candidate_variant]
    candidate_nll = candidate["test"]["nll"]
    recursive = [result for name, result in results.items() if name != "flat_seq"]
    derived = {
        "old_nll": old_nll,
        "candidate_variant": args.candidate_variant,
        "candidate_nll": candidate_nll,
        "candidate_gain_over_old": old_nll - candidate_nll,
        "candidate_update_delta_rms": candidate["update"]["delta_rms"],
    }
    if args.stage == "ablation":
        adaptive_nll = results["adaptive_alternate"]["test"]["nll"]
        learned_nll = results["learned_update"]["test"]["nll"]
        alternate_nll = results["alternate_fixed"]["test"]["nll"]
        better_single = min(learned_nll, alternate_nll)
        derived.update({
            "adaptive_nll": adaptive_nll,
            "adaptive_gain_over_old": old_nll - adaptive_nll,
            "learned_update_gain_over_old": old_nll - learned_nll,
            "alternate_fixed_gain_over_old": old_nll - alternate_nll,
            "adaptive_update_delta_rms": results["adaptive_alternate"]["update"]["delta_rms"],
        })
        gates = {
            "P1_all_closed_finite": all(
                result["closure"].get("state_mse", 0.0) < 1e-10
                and result["finite_gradients"]
                for result in recursive
            ),
            "P2_one_change_improves": max(
                old_nll - learned_nll, old_nll - alternate_nll,
                old_nll - adaptive_nll,
            ) >= 0.03,
            "P3_combination_not_antagonistic": adaptive_nll <= better_single + 0.03,
            "P4_update_actually_learned": derived["adaptive_update_delta_rms"] >= 1e-3,
        }
        return derived, gates

    flat_nll = results["flat_seq"]["test"]["nll"]
    intervention = candidate["interventions"]
    detail_damage = [row["nll"] - candidate_nll for row in intervention["detail_shuffle"]]
    pair_damage = [row["nll"] - candidate_nll for row in intervention["pair_break"]]
    route_mass = candidate["test"].get("route_depth_mass", [])
    old_gap = old_nll - flat_nll
    new_gap = candidate_nll - flat_nll
    gap_closed = (old_gap - new_gap) / max(1e-12, old_gap)
    derived.update({
        "flat_nll": flat_nll,
        "old_gap_to_flat": old_gap,
        "candidate_gap_to_flat": new_gap,
        "flat_gap_fraction_closed": gap_closed,
        "source_shuffle_damage": intervention["source_shuffle"]["nll"] - candidate_nll,
        "root_shuffle_damage": intervention["root_shuffle"]["nll"] - candidate_nll,
        "detail_shuffle_damage": detail_damage,
        "pair_break_damage": pair_damage,
        "force_root_damage": intervention["force_root"]["nll"] - candidate_nll,
        "force_leaf_damage": intervention["force_leaf"]["nll"] - candidate_nll,
    })
    gates = {
        "P5_candidate_beats_old": derived["candidate_gain_over_old"] >= 0.05,
        "P6_closes_flat_gap": gap_closed >= 0.25,
        "P7_source_causal": derived["source_shuffle_damage"] >= 0.50,
        "P8_root_and_details_causal": (
            derived["root_shuffle_damage"] >= 0.05
            and sum(value >= 0.05 for value in detail_damage) >= 3
        ),
        "P9_pairs_causal": sum(value >= 0.05 for value in pair_damage) >= 3,
        "P10_multiresolution": (
            sum(value >= 0.05 for value in route_mass) >= 2
            and (route_mass[-1] if route_mass else 1.0) <= 0.90
        ),
        "P11_closed_finite_nonempty": (
            candidate["closure"].get("state_mse", 1.0) < 1e-10
            and candidate["finite_gradients"]
            and candidate["test"].get("nonempty", 0.0) > 0.0
        ),
    }
    return derived, gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_adaptive_lifting_wmt_ablation")
    parser.add_argument("--stage", choices=("ablation", "scale"), default="ablation")
    parser.add_argument("--seed", type=int, default=71659)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--valid-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--max-scan", type=int, default=300000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--audit-variant", choices=VARIANTS, default="adaptive_alternate")
    parser.add_argument("--candidate-variant", choices=VARIANTS, default="adaptive_alternate")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    required = {"old_recursive", "adaptive_alternate"}
    if args.stage == "scale":
        required.remove("adaptive_alternate")
        required.add("flat_seq")
        required.add(args.candidate_variant)
    if not required.issubset(args.variants):
        raise ValueError(f"{args.stage} requires variants {sorted(required)}")
    if args.audit_variant not in args.variants:
        raise ValueError("audit variant must be trained")
    if args.stage == "scale" and args.audit_variant != args.candidate_variant:
        raise ValueError("scale audit variant must equal candidate variant")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, sampling = load_rows(args, sp)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    loaders = [
        DataLoader(
            base.ParallelDataset(split),
            batch_size=args.batch_size,
            shuffle=index == 0,
            num_workers=args.num_workers,
            collate_fn=base.collate(pad),
            pin_memory=args.device.startswith("cuda"),
        )
        for index, split in enumerate(splits)
    ]
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: Dict[str, dict] = {}
    for name in args.variants:
        results[name] = train_variant(
            name, *loaders, args, vocab, pad, bos, eos, sp, output,
        )
    derived, gates = build_gates(args, results)
    decision = "supported" if gates and all(gates.values()) else "partial" if any(gates.values()) else "not_supported"
    summary = {
        "claim": "S2-ADAPTIVE-LIFT-WMT-C01",
        "predict": "P-S2-ADAPTIVE-LIFT-WMT-01",
        "stage": args.stage,
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {
            "direction": "en_to_zh",
            "rows": len(rows),
            "vocab": vocab,
            **sampling,
        },
        "models": results,
        "derived": derived,
        "gates": gates,
        "decision": decision,
        "boundary": (
            "Comparative WMT mechanism proof. No full-corpus convergence, "
            "production BLEU, compression, sparse compute, world knowledge, "
            "or consciousness claim."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "trace.jsonl").write_text(
        "\n".join(json.dumps(row) for result in results.values() for row in result["trace"]) + "\n",
        encoding="utf-8",
    )
    (output / "examples.json").write_text(
        json.dumps(
            {name: result["test"].get("examples", []) for name, result in results.items()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# S2 Adaptive Lifting WMT\n\n```json\n"
        + json.dumps({"derived": derived, "gates": gates, "decision": decision}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": derived, "gates": gates, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
