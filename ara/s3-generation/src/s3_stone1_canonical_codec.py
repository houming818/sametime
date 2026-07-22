#!/usr/bin/env python3
"""Train and audit STONE-1 C02's canonical 0.4/0.6 TreeHeap codec."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shlex
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_lifting_pump_wmt as prior
import s3_private_protocol_data_dose as data_dose
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


Row = Tuple[List[int], List[int]]
VARIANTS = ("canonical_algebraic", "canonical_learned", "canonical_frozen")
BASE_RIGHT_WEIGHT = 0.6
RESIDUAL_SCALE = 0.25


class CanonicalCodec(nn.Module):
    """Fixed handedness with optional continuous residuals around lifting."""

    def __init__(self, dim: int, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.predict_net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.update_net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        if variant in {"canonical_algebraic", "canonical_learned"}:
            nn.init.zeros_(self.predict_net[-1].weight)
            nn.init.zeros_(self.predict_net[-1].bias)
            nn.init.zeros_(self.update_net[-1].weight)
            nn.init.zeros_(self.update_net[-1].bias)
        else:
            # Keep embedding/decoder initialization matched across arms. The
            # frozen codec receives deterministic random residuals without
            # advancing the model's global RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7331)
                nn.init.normal_(self.predict_net[-1].weight, std=0.02)
                nn.init.zeros_(self.predict_net[-1].bias)
                nn.init.normal_(self.update_net[-1].weight, std=0.02)
                nn.init.zeros_(self.update_net[-1].bias)

        if variant != "canonical_learned":
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def use_residual(self, override: str | None) -> bool:
        if override is None or override == "native":
            return self.variant != "canonical_algebraic"
        if override == "algebraic":
            return False
        raise ValueError(f"unknown codec override: {override}")

    def predict_delta(self, left: torch.Tensor, override: str | None = None):
        if not self.use_residual(override):
            return torch.zeros_like(left)
        return RESIDUAL_SCALE * torch.tanh(self.predict_net(left))

    def update_delta(self, detail: torch.Tensor, override: str | None = None):
        if not self.use_residual(override):
            return torch.zeros_like(detail)
        return RESIDUAL_SCALE * torch.tanh(self.update_net(detail))

    def predict(self, left: torch.Tensor, override: str | None = None):
        return left + self.predict_delta(left, override)

    def update(self, detail: torch.Tensor, override: str | None = None):
        return BASE_RIGHT_WEIGHT * detail + self.update_delta(detail, override)

    def learned_output_parameter_norm(self) -> float:
        parameters = (
            self.predict_net[-1].weight, self.predict_net[-1].bias,
            self.update_net[-1].weight, self.update_net[-1].bias,
        )
        total = sum(parameter.detach().square().sum() for parameter in parameters)
        return float(total.sqrt())

    def learned_output_parameter_rms(self) -> float:
        parameters = (
            self.predict_net[-1].weight, self.predict_net[-1].bias,
            self.update_net[-1].weight, self.update_net[-1].bias,
        )
        total = sum(parameter.detach().square().sum() for parameter in parameters)
        count = sum(parameter.numel() for parameter in parameters)
        return float((total / count).sqrt())


class CanonicalLiftingEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int, heap_width: int, pad: int, variant: str):
        super().__init__()
        if heap_width < 2 or heap_width & (heap_width - 1):
            raise ValueError("heap_width must be a power of two")
        self.embedding = nn.Embedding(vocab, dim)
        self.codec = CanonicalCodec(dim, variant)
        self.heap_width = heap_width
        self.depths = int(math.log2(heap_width))
        self.pad = pad
        self.variant = variant

    def fold(
        self,
        src: torch.Tensor,
        length: torch.Tensor,
        codec_override: str | None = None,
        pair_break_depth: int = -1,
        fold_mirror_depth: int = -1,
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

        for depth in range(self.depths):
            left, right = node[:, 0::2], node[:, 1::2]
            left_mask, right_mask = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == fold_mirror_depth:
                left, right = right, left
                left_mask, right_mask = right_mask, left_mask
            if depth == pair_break_depth:
                right = right.roll(1, dims=0)
                right_mask = right_mask.roll(1, dims=0)
            detail = right - self.codec.predict(left, codec_override)
            parent = left + self.codec.update(detail, codec_override)
            paired = right_mask[:, :, None]
            detail = torch.where(paired, detail, torch.zeros_like(detail))
            parent = torch.where(paired, parent, left)
            node_mask = left_mask | right_mask
            detail = detail * node_mask[:, :, None]
            parent = parent * node_mask[:, :, None]
            details.append(detail)
            masks.append(node_mask)
            node = parent
        return leaf, node[:, 0], details, masks

    def unfold(
        self,
        root: torch.Tensor,
        details: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        codec_override: str | None = None,
        intervention: str = "native",
    ):
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
        elif intervention not in {"native", "address_swap"}:
            raise ValueError(intervention)

        levels = [node[:, None]]
        level_masks = [local_masks[-1]]
        for depth in range(len(local_details) - 1, -1, -1):
            detail = local_details[depth]
            paired = local_masks[depth][:, 1::2, None]
            left = torch.where(
                paired,
                levels[-1] - self.codec.update(detail, codec_override),
                levels[-1],
            )
            right = torch.where(
                paired,
                detail + self.codec.predict(left, codec_override),
                torch.zeros_like(left),
            )
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
        codec_override: str | None = None,
        pair_break_depth: int = -1,
        fold_mirror_depth: int = -1,
    ):
        leaf, root, details, masks = self.fold(
            src, length, codec_override, pair_break_depth, fold_mirror_depth,
        )
        levels, level_masks = self.unfold(
            root, details, masks, codec_override, intervention,
        )
        return leaf, root, details, levels, level_masks, masks


class CanonicalTreeHeap(prior.S2Model):
    def __init__(
        self, vocab: int, dim: int, hidden: int, heap_width: int,
        pad: int, variant: str, leaf_cut: int,
    ):
        super().__init__()
        self.variant = variant
        self.leaf_cut = leaf_cut
        self.encoder = CanonicalLiftingEncoder(vocab, dim, heap_width, pad, variant)
        self.decoder = prior.RecursiveDecoder(vocab, dim, hidden, self.encoder.depths)

    def visible(self, levels, masks, max_visible_levels: int | None = None):
        stop = len(levels) - self.leaf_cut
        if stop < 1:
            raise ValueError("leaf_cut removes every TreeHeap level")
        if max_visible_levels is not None:
            stop = min(stop, max_visible_levels)
        return levels[:stop], masks[:stop]

    def states(self, src, length, **kwargs):
        return self.encoder.states(src, length, **kwargs)

    def teacher(
        self, src, length, target, bos, intervention="native",
        codec_override=None, max_visible_levels=None, pair_break_depth=-1,
        fold_mirror_depth=-1,
        route_mode="native", gate_override=None,
    ):
        state = self.states(
            src, length, intervention=intervention,
            codec_override=codec_override, pair_break_depth=pair_break_depth,
            fold_mirror_depth=fold_mirror_depth,
        )
        levels, masks = self.visible(state[3], state[4], max_visible_levels)
        return self.decoder.teacher(levels, masks, target, bos, route_mode)

    def greedy(
        self, src, length, bos, eos, max_len, intervention="native",
        codec_override=None, max_visible_levels=None, route_mode="native",
        gate_override=None,
    ):
        state = self.states(
            src, length, intervention=intervention, codec_override=codec_override,
        )
        levels, masks = self.visible(state[3], state[4], max_visible_levels)
        return self.decoder.greedy(levels, masks, bos, eos, max_len, route_mode)


def make_model(variant: str, args, vocab: int, pad: int) -> CanonicalTreeHeap:
    return CanonicalTreeHeap(
        vocab, args.dim, args.hidden, args.heap_width, pad, variant, args.leaf_cut,
    )


@torch.no_grad()
def evaluate(
    model, loader, args, pad: int, bos: int, eos: int, sp,
    generate: bool = False, intervention: str = "native",
    codec_override: str | None = None, max_visible_levels: int | None = None,
    fold_mirror_depth: int = -1,
):
    model.eval()
    loss_sum = tokens = exact = nonempty = repeated = count = 0
    route_sum = None
    route_batches = 0
    hypotheses: List[List[int]] = []
    references: List[List[int]] = []
    examples = []
    for source, length, target, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        logits, route = model.teacher(
            source, length, target, bos, intervention=intervention,
            codec_override=codec_override, max_visible_levels=max_visible_levels,
            fold_mirror_depth=fold_mirror_depth,
        )
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(valid.sum())
        if route is not None:
            route_cpu = route.detach().float().cpu()
            route_sum = route_cpu if route_sum is None else route_sum + route_cpu
            route_batches += 1
        if not generate:
            continue
        predicted, _ = model.greedy(
            source, length, bos, eos, target.shape[1], intervention=intervention,
            codec_override=codec_override, max_visible_levels=max_visible_levels,
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
            repeated += int(c01.severe_repetition(hyp))
            count += 1
            if len(examples) < 20:
                examples.append({
                    "en": sp.decode(src),
                    "reference_zh": sp.decode(ref),
                    "hypothesis_zh": sp.decode(hyp),
                    "severe_repetition": c01.severe_repetition(hyp),
                })
    nll = loss_sum / max(1, tokens)
    result = {"nll": nll, "ppl": math.exp(min(20.0, nll)), "tokens": tokens}
    if route_sum is not None:
        result["route_mass_by_level"] = (
            route_sum / max(1, route_batches)
        ).tolist()
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
def structure_audit(model: CanonicalTreeHeap, loader, args) -> dict:
    source, length, _, _ = next(iter(loader))
    source, length = source.to(args.device), length.to(args.device)
    state = model.states(source, length)
    closure = state[3][-1] - state[0]
    detail_rows = []
    node = state[0]
    node_mask = state[5][0]
    for depth, detail in enumerate(state[2]):
        valid = state[5][depth + 1]
        selected = detail[valid]
        left, right = node[:, 0::2], node[:, 1::2]
        right_mask = node_mask[:, 1::2]
        paired = right_mask[:, :, None]
        predict_delta = model.encoder.codec.predict_delta(left)
        update_delta = model.encoder.codec.update_delta(detail)
        predict_selected = predict_delta[paired.expand_as(predict_delta)]
        update_selected = update_delta[paired.expand_as(update_delta)]
        base_update = (BASE_RIGHT_WEIGHT * detail)[paired.expand_as(detail)]
        parent = left + model.encoder.codec.update(detail)
        parent = torch.where(paired, parent, left)
        node_mask = state[5][depth + 1]
        parent = parent * node_mask[:, :, None]
        detail_rows.append({
            "depth": depth,
            "detail_rms": float(selected.square().mean().sqrt()),
            "detail_abs_mean": float(selected.abs().mean()),
            "predict_delta_rms": (
                float(predict_selected.square().mean().sqrt())
                if predict_selected.numel() else 0.0
            ),
            "update_delta_rms": (
                float(update_selected.square().mean().sqrt())
                if update_selected.numel() else 0.0
            ),
            "base_update_rms": (
                float(base_update.square().mean().sqrt())
                if base_update.numel() else 0.0
            ),
        })
        node = parent
    return {
        "closure_mse": float(closure.square().mean()),
        "closure_max_abs": float(closure.abs().max()),
        "learned_output_parameter_norm": (
            model.encoder.codec.learned_output_parameter_norm()
        ),
        "learned_output_parameter_rms": (
            model.encoder.codec.learned_output_parameter_rms()
        ),
        "depths": detail_rows,
    }


def save_checkpoint(
    path: Path, model, variant: str, seed: int, args, vocab: int, pad: int,
    tokenizer_sha256: str,
) -> dict:
    payload = {
        "format": "treeheap-stone1-canonical-v1",
        "variant": variant,
        "seed": seed,
        "model_config": {
            "vocab": vocab, "pad": pad, "dim": args.dim,
            "hidden": args.hidden, "heap_width": args.heap_width,
            "leaf_cut": args.leaf_cut,
        },
        "codec": {
            "detail": "R - L at zero residual",
            "parent": "0.4 L + 0.6 R at zero residual",
            "base_right_weight": BASE_RIGHT_WEIGHT,
            "residual_scale": RESIDUAL_SCALE,
        },
        "tokenizer": {"path": args.spm_model, "sha256": tokenizer_sha256},
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": c01.file_digest(path),
    }


def train_arm(
    variant: str, seed: int, rows: Sequence[Row], valid_loader, test_loader,
    args, vocab: int, pad: int, bos: int, eos: int, sp, output: Path,
    checkpoint_dir: Path, tokenizer_sha256: str,
):
    c01.set_seed(seed)
    model = make_model(variant, args, vocab, pad).to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, valid_loader, args, pad, bos, eos, sp)
    best_nll = initial["nll"]
    best_step = 0
    best_state = copy.deepcopy({
        key: value.detach().cpu() for key, value in model.state_dict().items()
    })
    trace_path = output / "trace.jsonl"
    c01.append_trace(trace_path, {
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
        c01.append_trace(trace_path, row)
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
    if variant == "canonical_learned":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = save_checkpoint(
            checkpoint_dir / f"stone1_c02_{variant}_seed{seed}.pt",
            model, variant, seed, args, vocab, pad, tokenizer_sha256,
        )
    result = {
        "variant": variant, "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "best_step": best_step, "best_valid_nll": best_nll,
        "test": test, "structure": structure,
        "seconds": time.time() - started, "peak_vram_bytes": peak_vram,
        "finite_gradients": finite, "checkpoint": checkpoint,
    }
    return result, best_state if variant == "canonical_learned" else None


def aggregate_results(results: Sequence[dict]) -> dict:
    aggregate: Dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        nll = [row["test"]["nll"] for row in rows]
        aggregate[variant] = {
            "nll_mean": statistics.mean(nll),
            "nll_std": statistics.pstdev(nll),
            "bleu4_mean": statistics.mean(row["test"]["token_bleu4"] for row in rows),
            "nonempty_mean": statistics.mean(row["test"]["nonempty"] for row in rows),
            "severe_repetition_mean": statistics.mean(
                row["test"]["severe_repetition_rate"] for row in rows
            ),
            "seconds_mean": statistics.mean(row["seconds"] for row in rows),
            "parameters": rows[0]["parameters"],
            "trainable_parameters": rows[0]["trainable_parameters"],
            "peak_vram_max": max(row["peak_vram_bytes"] for row in rows),
        }
    return aggregate


def audit_learned(
    state, seed: int, test_loader, args, vocab, pad, bos, eos, sp,
):
    model = make_model("canonical_learned", args, vocab, pad).to(args.device)
    model.load_state_dict(state)
    normal = evaluate(model, test_loader, args, pad, bos, eos, sp)
    algebraic = evaluate(
        model, test_loader, args, pad, bos, eos, sp, codec_override="algebraic",
    )
    address = evaluate(
        model, test_loader, args, pad, bos, eos, sp,
        intervention="address_swap",
    )
    visible_levels = model.encoder.depths + 1 - model.leaf_cut
    depth_growth = []
    for level_count in range(1, visible_levels + 1):
        score = evaluate(
            model, test_loader, args, pad, bos, eos, sp,
            max_visible_levels=level_count,
        )
        depth_growth.append({"visible_levels": level_count, **score})
    latency = c01.latency_audit(model, test_loader, args, bos, eos)
    result = {
        "seed": seed, "normal": normal, "force_algebraic": algebraic,
        "address_swap": address,
        "damage_nll": {
            "force_algebraic": algebraic["nll"] - normal["nll"],
            "address_swap": address["nll"] - normal["nll"],
        },
        "depth_growth": depth_growth,
        "root_to_full_gain_nll": depth_growth[0]["nll"] - depth_growth[-1]["nll"],
        "improving_depth_transitions": sum(
            right["nll"] < left["nll"]
            for left, right in zip(depth_growth, depth_growth[1:])
        ),
        "learned_output_parameter_norm": (
            model.encoder.codec.learned_output_parameter_norm()
        ),
        "learned_output_parameter_rms": (
            model.encoder.codec.learned_output_parameter_rms()
        ),
        "latency": latency,
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def decide(results, aggregate, intervention, args):
    learned = aggregate["canonical_learned"]
    algebraic = aggregate["canonical_algebraic"]
    frozen = aggregate["canonical_frozen"]
    learned_rows = [row for row in results if row["variant"] == "canonical_learned"]
    checkpoint_max = max(row["checkpoint"]["bytes"] for row in learned_rows)
    closure_max = max(row["structure"]["closure_max_abs"] for row in learned_rows)
    damage = intervention["damage_nll"]
    gates = {
        "Q1_nll_at_most_3_90": learned["nll_mean"] <= 3.90,
        "Q2_bleu4_at_least_13_5": learned["bleu4_mean"] >= 13.5,
        "Q3_nll_std_at_most_0_05": learned["nll_std"] <= 0.05,
        "Q4_nonempty_is_one": learned["nonempty_mean"] >= 1.0,
        "Q5_repetition_at_most_0_10": learned["severe_repetition_mean"] <= 0.10,
        "S1_learned_beats_algebraic_0_05": (
            algebraic["nll_mean"] - learned["nll_mean"] >= 0.05
        ),
        "S2_learned_beats_frozen_0_10": (
            frozen["nll_mean"] - learned["nll_mean"] >= 0.10
        ),
        "S3_force_algebraic_damage_0_10": damage["force_algebraic"] >= 0.10,
        "S4_address_damage_0_10": damage["address_swap"] >= 0.10,
        "S5_root_to_full_gain_0_50": intervention["root_to_full_gain_nll"] >= 0.50,
        "S6_four_of_five_depths_improve": (
            intervention["improving_depth_transitions"] >= 4
        ),
        "S7_closure_below_1e_5": closure_max < 1e-5,
        "E1_latency_p50_at_most_1000ms": intervention["latency"]["p50_ms"] <= 1000,
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
        status = "codec_mechanism_poc_only"
    elif quality and engineering:
        status = "seq2seq_demo_only"
    else:
        status = "c02_not_supported_stone1_incomplete"
    return gates, status


def render_report(summary: dict, results: Sequence[dict]) -> str:
    lines = [
        "# STONE-1 C02 Canonical Codec Report", "", "## Experiment Card", "",
        "| Field | Value |", "|---|---|",
        f"| Status | `{summary['status']}` |",
        f"| Host / device | `{summary['host']}` / `{summary['device_name']}` |",
        f"| Runtime | {summary['seconds'] / 3600:.2f} h |",
        f"| Git commit | `{summary['git_commit']}` |", "", "## Results", "",
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
        "", "## Codec Intervention and Depth Growth", "", "```json",
        json.dumps(summary["intervention"], indent=2, ensure_ascii=False), "```",
        "", "## Decision Gates", "", "```json",
        json.dumps(summary["gates"], indent=2, ensure_ascii=False), "```",
        "", "## Boundary", "", summary["boundary"], "",
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

    if not set(VARIANTS).issubset(args.variants):
        raise ValueError(f"variants must include {sorted(VARIANTS)}")
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
    config["canonical_codec"] = {
        "detail": "R-L", "parent": "0.4L+0.6R",
        "base_right_weight": BASE_RIGHT_WEIGHT,
        "residual_scale": RESIDUAL_SCALE,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    command = shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]])
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8",
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
        (row for row in results if row["variant"] == "canonical_learned"),
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
    best_learned = min(
        (row for row in results if row["variant"] == "canonical_learned"),
        key=lambda row: row["best_valid_nll"],
    )
    summary = {
        "experiment_id": (
            "s3_stone1_canonical_codec_smoke" if args.smoke
            else "s3_stone1_canonical_codec"
        ),
        "claim": "S3-STONE1-PRIVATE-PROTOCOL-C02",
        "predict": "P-S3-STONE1-PRIVATE-PROTOCOL-02",
        "milestone": "STONE-1",
        "status": status,
        "host": socket.gethostname(),
        "device_name": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
        ),
        "git_commit": args.code_commit or c01.git_revision(),
        "seconds": time.time() - started,
        "config": config, "dataset": manifest, "aggregate": aggregate,
        "intervention": intervention, "gates": gates,
        "demo_checkpoint": best_learned["checkpoint"],
        "historical_c01_identity": {
            "nll_mean": 4.071877074325834,
            "bleu4_mean": 11.173549989485323,
            "note": "context only; not rerun inside C02",
        },
        "boundary": (
            "C02 can support or reject one canonical codec recipe inside the "
            "unfinished STONE-1 milestone. It cannot establish dialogue, world "
            "knowledge, human-readable depth semantics, or industry superiority."
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
            "checkpoint": best_learned["checkpoint"],
            "examples": best_learned["test"]["examples"][:5],
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
