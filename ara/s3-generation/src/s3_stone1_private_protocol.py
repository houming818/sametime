#!/usr/bin/env python3
"""Owner: SameTime S3 generation.
Authors: Houming818 and Codex Review.
Created: 2026-07-21.
Updated: 2026-07-21.
Purpose: Train and audit the STONE-1 fixed-capacity TreeHeap translation PoC.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as adaptive
import s2_lifting_pump_wmt as prior
import s3_private_protocol_data_dose as data_dose
import s3_wmt_treeheap_seq2seq as base


Row = Tuple[List[int], List[int]]
VARIANTS = ("identity", "learned_structural", "frozen_random")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalDirectionKernel(nn.Module):
    """Shared local convolution that chooses a reversible pair orientation."""

    def __init__(self, dim: int, depths: int, random_seed: int = 7331):
        super().__init__()
        feature_dim = 4 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.depth_bias = nn.Parameter(torch.zeros(depths))
        self.random_seed = random_seed
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.net[-1].bias)

    def fixed_random(self, left: torch.Tensor, depth: int) -> torch.Tensor:
        position = torch.arange(left.shape[1], device=left.device)[None, :]
        pattern = (position * 17 + depth * 31 + self.random_seed) % 5
        return pattern.lt(2).to(left.dtype).expand(left.shape[0], -1)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        right_valid: torch.Tensor,
        depth: int,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "identity":
            probability = torch.ones_like(right_valid, dtype=left.dtype)
            gate = probability
        elif mode == "random":
            probability = self.fixed_random(left, depth)
            gate = probability
        elif mode == "learned":
            feature = torch.cat((left, right, left - right, left * right), dim=-1)
            probability = torch.sigmoid(
                self.net(feature).squeeze(-1) + self.depth_bias[depth]
            )
            hard = probability.ge(0.5).to(probability.dtype)
            gate = hard + probability - probability.detach()
        else:
            raise ValueError(f"unknown gate mode: {mode}")

        # Left-packed sequences may have one unmatched left child. It must stay
        # the anchor, otherwise padding would become a structural decision.
        gate = torch.where(right_valid, gate, torch.ones_like(gate))
        probability = torch.where(
            right_valid, probability, torch.ones_like(probability),
        )
        return gate, probability


class StructuralLiftingEncoder(nn.Module):
    """Exactly reversible TreeHeap FOLD with learned local anchor directions."""

    def __init__(
        self, vocab: int, dim: int, heap_width: int, pad: int, variant: str,
    ):
        super().__init__()
        if heap_width < 2 or heap_width & (heap_width - 1):
            raise ValueError("heap_width must be a power of two")
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.embedding = nn.Embedding(vocab, dim)
        self.predictor = prior.SharedPredictor(dim)
        self.update_kernel = adaptive.LearnedUpdate(dim)
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad = pad
        self.variant = variant
        self.direction = LocalDirectionKernel(dim, self.depths)

    def default_gate_mode(self) -> str:
        return {
            "identity": "identity",
            "learned_structural": "learned",
            "frozen_random": "random",
        }[self.variant]

    def fold(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        gate_override: str | None = None,
        pair_break_depth: int = -1,
    ):
        if src.shape[1] > self.heap_width:
            raise ValueError(
                f"source width {src.shape[1]} exceeds heap width {self.heap_width}"
            )
        padded = torch.full(
            (src.shape[0], self.heap_width), self.pad,
            dtype=src.dtype, device=src.device,
        )
        padded[:, : src.shape[1]] = src
        leaf_mask = (
            torch.arange(self.heap_width, device=src.device)[None] < length[:, None]
        )
        leaf = self.embedding(padded) * leaf_mask[:, :, None]
        node, node_mask = leaf, leaf_mask
        details: List[torch.Tensor] = []
        masks: List[torch.Tensor] = [leaf_mask]
        gates: List[torch.Tensor] = []
        probabilities: List[torch.Tensor] = []
        mode = gate_override or self.default_gate_mode()

        for depth in range(self.depths):
            left, right = node[:, 0::2], node[:, 1::2]
            left_mask, right_mask = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == pair_break_depth:
                right = right.roll(1, dims=0)
                right_mask = right_mask.roll(1, dims=0)

            gate, probability = self.direction(
                left, right, right_mask, depth, mode,
            )
            gate3 = gate[:, :, None]
            left_detail = right - self.predictor(left)
            right_detail = left - self.predictor(right)
            left_parent = left + self.update_kernel(left_detail)
            right_parent = right + self.update_kernel(right_detail)
            detail = gate3 * left_detail + (1.0 - gate3) * right_detail
            parent = gate3 * left_parent + (1.0 - gate3) * right_parent

            node_mask = left_mask | right_mask
            detail = detail * node_mask[:, :, None]
            parent = parent * node_mask[:, :, None]
            details.append(detail)
            masks.append(node_mask)
            gates.append(gate)
            probabilities.append(probability)
            node = parent

        return leaf, node[:, 0], details, masks, gates, probabilities

    def unfold(
        self,
        root: torch.Tensor,
        details: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        gates: Sequence[torch.Tensor],
        intervention: str = "native",
    ):
        node = root
        local_details = list(details)
        local_masks = list(masks)
        local_gates = list(gates)
        if intervention == "source_shuffle":
            node = node.roll(1, dims=0)
            local_details = [row.roll(1, dims=0) for row in local_details]
            local_masks = [row.roll(1, dims=0) for row in local_masks]
            local_gates = [row.roll(1, dims=0) for row in local_gates]
        elif intervention == "root_shuffle":
            node = node.roll(1, dims=0)
        elif intervention.startswith("detail_shuffle_"):
            depth = int(intervention.rsplit("_", 1)[1])
            local_details[depth] = local_details[depth].roll(1, dims=0)
        elif intervention not in {"native", "address_swap"}:
            raise ValueError(intervention)

        levels = [node[:, None]]
        level_masks = [local_masks[-1]]
        for depth in range(len(local_details) - 1, -1, -1):
            detail = local_details[depth]
            gate = local_gates[depth][:, :, None]
            anchor = levels[-1] - self.update_kernel(detail)
            predicted = detail + self.predictor(anchor)
            left = gate * anchor + (1.0 - gate) * predicted
            right = gate * predicted + (1.0 - gate) * anchor
            expanded = torch.empty(
                left.shape[0], left.shape[1] * 2, left.shape[2],
                device=left.device, dtype=left.dtype,
            )
            expanded[:, 0::2], expanded[:, 1::2] = left, right
            expanded = expanded * local_masks[depth][:, :, None]
            levels.append(expanded)
            level_masks.append(local_masks[depth])

        if intervention == "address_swap":
            for index in range(1, len(levels)):
                if levels[index].shape[1] < 2:
                    continue
                swapped = levels[index].clone()
                swapped[:, 0::2] = levels[index][:, 1::2]
                swapped[:, 1::2] = levels[index][:, 0::2]
                swapped_mask = level_masks[index].clone()
                swapped_mask[:, 0::2] = level_masks[index][:, 1::2]
                swapped_mask[:, 1::2] = level_masks[index][:, 0::2]
                levels[index], level_masks[index] = swapped, swapped_mask
        return levels, level_masks

    def states(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        intervention: str = "native",
        gate_override: str | None = None,
        pair_break_depth: int = -1,
    ):
        packed = self.fold(src, length, gate_override, pair_break_depth)
        leaf, root, details, masks, gates, probabilities = packed
        levels, level_masks = self.unfold(
            root, details, masks, gates, intervention,
        )
        return leaf, root, details, levels, level_masks, gates, probabilities, masks


class StoneTreeHeap(prior.S2Model):
    def __init__(
        self,
        vocab: int,
        dim: int,
        hidden: int,
        heap_width: int,
        pad: int,
        variant: str,
        leaf_cut: int,
    ):
        super().__init__()
        self.variant = variant
        self.leaf_cut = leaf_cut
        self.encoder = StructuralLiftingEncoder(
            vocab, dim, heap_width, pad, variant,
        )
        self.decoder = prior.RecursiveDecoder(
            vocab, dim, hidden, self.encoder.depths,
        )

    def visible(self, levels, masks):
        stop = len(levels) - self.leaf_cut
        if stop < 1:
            raise ValueError("leaf_cut removes every TreeHeap level")
        return levels[:stop], masks[:stop]

    def states(self, src, length, **kwargs):
        return self.encoder.states(src, length, **kwargs)

    def teacher(
        self,
        src,
        length,
        target,
        bos,
        intervention="native",
        gate_override=None,
        pair_break_depth=-1,
        route_mode="native",
    ):
        state = self.states(
            src, length, intervention=intervention,
            gate_override=gate_override, pair_break_depth=pair_break_depth,
        )
        levels, masks = self.visible(state[3], state[4])
        return self.decoder.teacher(levels, masks, target, bos, route_mode)

    def greedy(
        self,
        src,
        length,
        bos,
        eos,
        max_len,
        intervention="native",
        gate_override=None,
        route_mode="native",
    ):
        state = self.states(
            src, length, intervention=intervention,
            gate_override=gate_override,
        )
        levels, masks = self.visible(state[3], state[4])
        return self.decoder.greedy(levels, masks, bos, eos, max_len, route_mode)


def make_model(variant: str, args, vocab: int, pad: int) -> StoneTreeHeap:
    return StoneTreeHeap(
        vocab, args.dim, args.hidden, args.heap_width, pad,
        variant, args.leaf_cut,
    )


def severe_repetition(tokens: Sequence[int]) -> bool:
    if len(tokens) < 8:
        return False
    runs = 1
    longest_run = 1
    for index in range(1, len(tokens)):
        runs = runs + 1 if tokens[index] == tokens[index - 1] else 1
        longest_run = max(longest_run, runs)
    bigrams = list(zip(tokens, tokens[1:]))
    diversity = len(set(bigrams)) / max(1, len(bigrams))
    return longest_run >= 4 or diversity < 0.45


@torch.no_grad()
def evaluate(
    model,
    loader,
    args,
    pad: int,
    bos: int,
    eos: int,
    sp,
    generate: bool = False,
    intervention: str = "native",
    gate_override: str | None = None,
):
    model.eval()
    loss_sum = tokens = exact = nonempty = repeated = count = 0
    hypotheses: List[List[int]] = []
    references: List[List[int]] = []
    examples = []
    for source, length, target, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        logits, _ = model.teacher(
            source, length, target, bos,
            intervention=intervention, gate_override=gate_override,
        )
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(valid.sum())
        if not generate:
            continue
        predicted, _ = model.greedy(
            source, length, bos, eos, target.shape[1],
            intervention=intervention, gate_override=gate_override,
        )
        predicted = predicted.cpu()
        source_cpu, target_cpu = source.cpu(), target.cpu()
        for index in range(source.shape[0]):
            hyp = base.clean(predicted[index].tolist(), eos, pad)
            ref = base.clean(target_cpu[index].tolist(), eos, pad)
            src = base.clean(source_cpu[index].tolist(), eos, pad)
            hypotheses.append(hyp)
            references.append(ref)
            exact += int(hyp == ref)
            nonempty += int(bool(hyp))
            repeated += int(severe_repetition(hyp))
            count += 1
            if len(examples) < 20:
                examples.append({
                    "en": sp.decode(src),
                    "reference_zh": sp.decode(ref),
                    "hypothesis_zh": sp.decode(hyp),
                    "severe_repetition": severe_repetition(hyp),
                })
    nll = loss_sum / max(1, tokens)
    result = {"nll": nll, "ppl": math.exp(min(20.0, nll)), "tokens": tokens}
    if generate:
        result.update({
            "exact": exact / max(1, count),
            "nonempty": nonempty / max(1, count),
            "severe_repetition_rate": repeated / max(1, count),
            "token_bleu4": base.bleu4(hypotheses, references),
            "examples": examples,
        })
    return result


@torch.no_grad()
def structure_audit(model: StoneTreeHeap, loader, args) -> dict:
    source, length, _, _ = next(iter(loader))
    source, length = source.to(args.device), length.to(args.device)
    state = model.states(source, length)
    leaf, levels, gates, probabilities = state[0], state[3], state[5], state[6]
    closure = levels[-1] - leaf
    depth_rows = []
    fold_masks = state[7]
    for depth, (gate, probability) in enumerate(zip(gates, probabilities)):
        valid = fold_masks[depth + 1]
        selected = probability[valid]
        entropy = -(
            selected.clamp(1e-7, 1 - 1e-7) * selected.clamp(1e-7, 1 - 1e-7).log()
            + (1 - selected).clamp(1e-7, 1 - 1e-7)
            * (1 - selected).clamp(1e-7, 1 - 1e-7).log()
        )
        depth_rows.append({
            "depth": depth,
            "probability_mean": float(selected.mean()),
            "probability_std": float(selected.std(unbiased=False)),
            "hard_left_fraction": float(gate.detach().ge(0.5).float().mean()),
            "entropy_mean": float(entropy.mean()),
        })
    return {
        "closure_mse": float(closure.square().mean()),
        "closure_max_abs": float(closure.abs().max()),
        "depths": depth_rows,
    }


@torch.no_grad()
def latency_audit(model, loader, args, bos: int, eos: int) -> dict:
    source, length, _, _ = next(iter(loader))
    source, length = source[:1].to(args.device), length[:1].to(args.device)
    for _ in range(3):
        model.greedy(source, length, bos, eos, args.cli_max_new_tokens)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.latency_repeats):
        started = time.perf_counter()
        model.greedy(source, length, bos, eos, args.cli_max_new_tokens)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "samples": len(samples),
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def append_trace(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path, model, variant: str, seed: int, args, vocab: int, pad: int,
    tokenizer_sha256: str,
) -> dict:
    payload = {
        "format": "treeheap-stone1-v1",
        "variant": variant,
        "seed": seed,
        "model_config": {
            "vocab": vocab,
            "pad": pad,
            "dim": args.dim,
            "hidden": args.hidden,
            "heap_width": args.heap_width,
            "leaf_cut": args.leaf_cut,
        },
        "tokenizer": {
            "path": args.spm_model,
            "sha256": tokenizer_sha256,
        },
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
    }


def train_arm(
    variant: str,
    seed: int,
    rows: Sequence[Row],
    valid_loader,
    test_loader,
    args,
    vocab: int,
    pad: int,
    bos: int,
    eos: int,
    sp,
    output: Path,
    checkpoint_dir: Path,
    tokenizer_sha256: str,
):
    set_seed(seed)
    model = make_model(variant, args, vocab, pad).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, valid_loader, args, pad, bos, eos, sp)
    best_nll = initial["nll"]
    best_step = 0
    best_state = copy.deepcopy({
        key: value.detach().cpu() for key, value in model.state_dict().items()
    })
    trace_path = output / "trace.jsonl"
    append_trace(trace_path, {
        "event": "evaluation", "variant": variant, "seed": seed,
        "step": 0, "valid_nll": best_nll, "train_nll_window": None,
    })
    batches = data_dose.infinite_batches(rows, args, pad, seed + args.train_samples)
    window_loss = 0.0
    window_steps = 0
    finite = True
    started = time.time()
    for step in range(1, args.fixed_steps + 1):
        _, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        model.train()
        logits, _ = model.teacher(source, length, target, bos)
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1
        if step % args.eval_interval and step != args.fixed_steps:
            continue
        valid = evaluate(model, valid_loader, args, pad, bos, eos, sp)
        row = {
            "event": "evaluation", "variant": variant, "seed": seed,
            "step": step, "sample_exposures": step * args.batch_size,
            "train_nll_window": window_loss / max(1, window_steps),
            "valid_nll": valid["nll"], "elapsed_sec": time.time() - started,
        }
        append_trace(trace_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        window_loss = 0.0
        window_steps = 0
        if math.isfinite(valid["nll"]) and valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best_step = step
            best_state = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, args, pad, bos, eos, sp, generate=True)
    structure = structure_audit(model, test_loader, args)
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
    )
    checkpoint = None
    if variant == "learned_structural":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = save_checkpoint(
            checkpoint_dir / f"stone1_{variant}_seed{seed}.pt",
            model, variant, seed, args, vocab, pad, tokenizer_sha256,
        )
    result = {
        "variant": variant,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_step": best_step,
        "best_valid_nll": best_nll,
        "test": test,
        "structure": structure,
        "seconds": time.time() - started,
        "peak_vram_bytes": peak_vram,
        "finite_gradients": finite,
        "checkpoint": checkpoint,
    }
    return result, best_state if variant == "learned_structural" else None


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def aggregate_results(results: Sequence[dict]) -> dict:
    aggregate: Dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        nll = [row["test"]["nll"] for row in rows]
        bleu = [row["test"]["token_bleu4"] for row in rows]
        nonempty = [row["test"]["nonempty"] for row in rows]
        repetition = [row["test"]["severe_repetition_rate"] for row in rows]
        nll_mean, nll_std = mean_std(nll)
        aggregate[variant] = {
            "nll_mean": nll_mean,
            "nll_std": nll_std,
            "bleu4_mean": statistics.mean(bleu),
            "nonempty_mean": statistics.mean(nonempty),
            "severe_repetition_mean": statistics.mean(repetition),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
            "parameters": rows[0]["parameters"],
            "peak_vram_max": max(row["peak_vram_bytes"] for row in rows),
        }
    return aggregate


def audit_learned(
    state, seed: int, test_loader, args, vocab, pad, bos, eos, sp,
):
    model = make_model("learned_structural", args, vocab, pad).to(args.device)
    model.load_state_dict(state)
    normal = evaluate(model, test_loader, args, pad, bos, eos, sp)
    identity = evaluate(
        model, test_loader, args, pad, bos, eos, sp, gate_override="identity",
    )
    random_score = evaluate(
        model, test_loader, args, pad, bos, eos, sp, gate_override="random",
    )
    address = evaluate(
        model, test_loader, args, pad, bos, eos, sp,
        intervention="address_swap",
    )
    latency = latency_audit(model, test_loader, args, bos, eos)
    result = {
        "seed": seed,
        "normal": normal,
        "force_identity": identity,
        "force_random": random_score,
        "address_swap": address,
        "damage_nll": {
            "force_identity": identity["nll"] - normal["nll"],
            "force_random": random_score["nll"] - normal["nll"],
            "address_swap": address["nll"] - normal["nll"],
        },
        "latency": latency,
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def decide(results, aggregate, intervention, args) -> Tuple[dict, str]:
    learned = aggregate["learned_structural"]
    identity = aggregate["identity"]
    random_arm = aggregate["frozen_random"]
    learned_rows = [row for row in results if row["variant"] == "learned_structural"]
    checkpoint_max = max(row["checkpoint"]["bytes"] for row in learned_rows)
    closure_max = max(row["structure"]["closure_max_abs"] for row in learned_rows)
    damage = intervention["damage_nll"]
    gates = {
        "Q1_nll_at_most_3_90": learned["nll_mean"] <= 3.90,
        "Q2_bleu4_at_least_13_5": learned["bleu4_mean"] >= 13.5,
        "Q3_nll_std_at_most_0_05": learned["nll_std"] <= 0.05,
        "Q4_nonempty_is_one": learned["nonempty_mean"] >= 1.0,
        "Q5_repetition_at_most_0_10": learned["severe_repetition_mean"] <= 0.10,
        "S1_learned_beats_identity_0_05": identity["nll_mean"] - learned["nll_mean"] >= 0.05,
        "S2_learned_beats_random_0_10": random_arm["nll_mean"] - learned["nll_mean"] >= 0.10,
        "S3_force_identity_damage_0_10": damage["force_identity"] >= 0.10,
        "S4_force_random_damage_0_10": damage["force_random"] >= 0.10,
        "S5_address_damage_0_10": damage["address_swap"] >= 0.10,
        "S6_closure_below_1e_5": closure_max < 1e-5,
        "E1_latency_p50_at_most_1000ms": intervention["latency"]["p50_ms"] <= 1000.0,
        "E2_vram_at_most_4gib": learned["peak_vram_max"] <= 4 * 2**30,
        "E3_checkpoint_at_most_300mib": checkpoint_max <= 300 * 2**20,
        "E4_finite": all(row["finite_gradients"] for row in results),
        "E5_checkpoint_created": all(row["checkpoint"] for row in learned_rows),
    }
    quality = all(value for key, value in gates.items() if key.startswith("Q"))
    structure = all(value for key, value in gates.items() if key.startswith("S"))
    engineering = all(value for key, value in gates.items() if key.startswith("E"))
    if args.smoke:
        status = "smoke_only"
    elif quality and structure and engineering:
        status = "supported_stone1_complete"
    elif structure and engineering:
        status = "mechanism_poc_only"
    elif quality and engineering:
        status = "seq2seq_demo_only"
    else:
        status = "not_supported_under_recipe"
    return gates, status


def render_report(summary: dict, results: Sequence[dict]) -> str:
    lines = [
        "# STONE-1 Private Protocol Translation Report", "",
        "## Experiment Card", "",
        "| Field | Value |", "|---|---|",
        f"| Status | `{summary['status']}` |",
        f"| Host / device | `{summary['host']}` / `{summary['device_name']}` |",
        f"| Runtime | {summary['seconds'] / 3600:.2f} h |",
        f"| Git commit | `{summary['git_commit']}` |", "",
        "## Results", "",
        "| Variant | Seed | Best step | Test NLL | PPL | BLEU-4 | Nonempty | Repetition | Time | VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        test = row["test"]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['best_step']:,} | "
            f"{test['nll']:.4f} | {test['ppl']:.1f} | {test['token_bleu4']:.3f} | "
            f"{test['nonempty']:.3f} | {test['severe_repetition_rate']:.3f} | "
            f"{row['seconds'] / 60:.1f} min | {row['peak_vram_bytes'] / 2**30:.2f} GiB |"
        )
    lines.extend([
        "", "## Aggregate", "", "```json",
        json.dumps(summary["aggregate"], indent=2, ensure_ascii=False), "```",
        "", "## Structural Intervention", "", "```json",
        json.dumps(summary["intervention"], indent=2, ensure_ascii=False), "```",
        "", "## Decision Gates", "", "```json",
        json.dumps(summary["gates"], indent=2, ensure_ascii=False), "```",
        "", "## Boundary", "",
        summary["boundary"], "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[71901, 71902, 71903])
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--baseline-max-scan", type=int, default=300_000)
    parser.add_argument("--pool-max-scan", type=int, default=3_000_000)
    parser.add_argument("--source-rows", type=int, default=14_170_275)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71900)
    parser.add_argument("--pool-seed", type=int, default=72003)
    parser.add_argument("--model-seed", type=int, default=71901)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fixed-steps", type=int, default=15_625)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--cli-max-new-tokens", type=int, default=64)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    required = set(VARIANTS)
    if not required.issubset(args.variants):
        raise ValueError(f"variants must include {sorted(required)}")
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    if args.leaf_cut < 1 or args.leaf_cut >= int(math.log2(args.heap_width)) + 1:
        raise ValueError("leaf_cut must hide at least the leaf and retain root")
    if args.smoke:
        args.train_samples = min(args.train_samples, 4096)
        args.valid_samples = min(args.valid_samples, 128)
        args.test_samples = min(args.test_samples, 128)
        args.fixed_steps = min(args.fixed_steps, 100)
        args.eval_interval = min(args.eval_interval, 50)
        args.seeds = args.seeds[:1]
        args.baseline_max_scan = min(args.baseline_max_scan, 100_000)
        args.pool_max_scan = min(args.pool_max_scan, 200_000)
        args.latency_repeats = min(args.latency_repeats, 5)

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output / "checkpoints"
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    config = vars(args).copy()
    config["checkpoint_dir"] = str(checkpoint_dir)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    command = shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]])
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n",
        encoding="utf-8",
    )
    (output / "trace.jsonl").write_text("", encoding="utf-8")

    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)
    tokenizer_sha256 = manifest["tokenizer"]["sha256"]

    results = []
    learned_states = {}
    for seed in args.seeds:
        for variant in args.variants:
            result, state = train_arm(
                variant, seed, rows[: args.train_samples], valid_loader, test_loader,
                args, vocab, pad, bos, eos, sp, output, checkpoint_dir,
                tokenizer_sha256,
            )
            results.append(result)
            if state is not None:
                learned_states[seed] = state
            (output / "runs.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    audit_seed = min(
        (row for row in results if row["variant"] == "learned_structural"),
        key=lambda row: row["best_valid_nll"],
    )["seed"]
    intervention = audit_learned(
        learned_states[audit_seed], audit_seed, test_loader, args,
        vocab, pad, bos, eos, sp,
    )
    (output / "interventions.json").write_text(
        json.dumps(intervention, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    aggregate = aggregate_results(results)
    gates, status = decide(results, aggregate, intervention, args)
    best_checkpoint = min(
        (row for row in results if row["variant"] == "learned_structural"),
        key=lambda row: row["best_valid_nll"],
    )["checkpoint"]
    summary = {
        "experiment_id": "s3_stone1_private_protocol_smoke" if args.smoke else "s3_stone1_private_protocol",
        "claim": "S3-STONE1-PRIVATE-PROTOCOL-C01",
        "predict": "P-S3-STONE1-PRIVATE-PROTOCOL-01",
        "status": status,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "git_commit": args.code_commit or git_revision(),
        "seconds": time.time() - started,
        "config": config,
        "dataset": manifest,
        "aggregate": aggregate,
        "intervention": intervention,
        "gates": gates,
        "demo_checkpoint": best_checkpoint,
        "boundary": (
            "This test can support a fixed-capacity TreeHeap translation PoC. "
            "It cannot establish dialogue, world knowledge, semantic rotation, "
            "or superiority over industry-scale Transformers."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        render_report(summary, results), encoding="utf-8",
    )
    (output / "cli_smoke.json").write_text(
        json.dumps({
            "checkpoint": best_checkpoint,
            "examples": min(
                (row for row in results if row["variant"] == "learned_structural"),
                key=lambda row: row["best_valid_nll"],
            )["test"]["examples"][:5],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
