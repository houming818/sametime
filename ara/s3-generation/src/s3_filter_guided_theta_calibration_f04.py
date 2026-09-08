#!/usr/bin/env python3
"""Absorb temporary TreeHeap node-filter corrections into trainable FOLD theta."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402
from treeheap_epoch_translate_cli import DEPTHS, load_runtime  # noqa: E402


CLAIM = "S3-FILTER-GUIDED-THETA-CALIBRATION-F04"
NATIVE_SCALE = math.sqrt(0.5)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RadialAnnealingFold(nn.Module):
    """Content-conditioned bounded parent gain, identity at theta == 0."""

    def __init__(self, dim: int, rank: int, max_merges: int = 5, gain_limit: float = 1.5):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.max_merges = max_merges
        self.log_limit = math.log(gain_limit)
        self.projection = nn.Linear(3 * dim, rank, bias=False)
        self.head = nn.Parameter(torch.zeros(max_merges, rank))
        self.bias = nn.Parameter(torch.zeros(max_merges))
        self.last_gain_rows: list[torch.Tensor] = []
        self.last_log_gain_rows: list[torch.Tensor] = []
        self.last_gain_masks: list[torch.Tensor] = []

    def forward(
        self,
        slots: torch.Tensor,
        slot_mask: torch.Tensor,
        filter_u: torch.Tensor | None = None,
    ):
        levels = [slots]
        masks = [slot_mask]
        node, valid = slots, slot_mask
        gain_rows = []
        log_gain_rows = []
        gain_masks = []
        filter_cursor = 0
        merge = 0
        while node.shape[1] > 1:
            if merge >= self.max_merges:
                raise RuntimeError(f"F04 fold width requires more than {self.max_merges} merges")
            left, right = node[:, 0::2], node[:, 1::2]
            left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
            both = left_valid & right_valid
            native = (left + right) * NATIVE_SCALE
            feature = torch.cat((
                F.layer_norm(left, (self.dim,)),
                F.layer_norm(right, (self.dim,)),
                F.layer_norm(left - right, (self.dim,)),
            ), dim=-1)
            hidden = torch.tanh(self.projection(feature))
            raw = self.bias[merge] + (
                hidden * self.head[merge][None, None]
            ).sum(-1) / math.sqrt(self.rank)
            log_gain = self.log_limit * torch.tanh(raw)
            gain = torch.exp(log_gain)
            filter_gain = 1.0
            if filter_u is not None:
                width = native.shape[1]
                local_filter = filter_u[:, filter_cursor:filter_cursor + width]
                if local_filter.shape[1] != width:
                    raise RuntimeError("filter coordinate count does not cover recursive parents")
                filter_gain = torch.exp(self.log_limit * torch.tanh(local_filter))
                filter_cursor += width
            parent = torch.where(
                both[:, :, None],
                native * gain[:, :, None] * filter_gain[:, :, None],
                torch.where(left_valid[:, :, None], left, right),
            )
            valid = left_valid | right_valid
            parent = parent * valid[:, :, None]
            gain_rows.append(gain[both])
            log_gain_rows.append(log_gain)
            gain_masks.append(both)
            levels.append(parent)
            masks.append(valid)
            node = parent
            merge += 1
        if filter_u is not None and filter_cursor != filter_u.shape[1]:
            raise RuntimeError(f"unused recursive filter coordinates: {filter_u.shape[1] - filter_cursor}")
        self.last_gain_rows = gain_rows
        self.last_log_gain_rows = log_gain_rows
        self.last_gain_masks = gain_masks
        return list(reversed(levels)), list(reversed(masks))

    def gain_summary(self) -> dict:
        available = [row.detach().flatten() for row in self.last_gain_rows if row.numel()]
        if not available:
            return {"mean": 1.0, "min": 1.0, "max": 1.0}
        values = torch.cat(available)
        return {
            "mean": float(values.mean().cpu()),
            "min": float(values.min().cpu()),
            "max": float(values.max().cpu()),
        }

    def gain_vector(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.last_log_gain_rows:
            raise RuntimeError("gain_vector requested before FOLD")
        return torch.cat(self.last_log_gain_rows, dim=1), torch.cat(self.last_gain_masks, dim=1)


class AnnealingTheta(nn.Module):
    def __init__(self, base_dim: int, extra_dim: int, rank: int):
        super().__init__()
        self.base = RadialAnnealingFold(base_dim, rank)
        self.extra = RadialAnnealingFold(extra_dim, rank)

    def choose(self, dim: int) -> RadialAnnealingFold:
        if dim == self.base.dim:
            return self.base
        if dim == self.extra.dim:
            return self.extra
        raise ValueError(f"unexpected protocol dim {dim}")

    def gain_summary(self) -> dict:
        return {"base": self.base.gain_summary(), "extra": self.extra.gain_summary()}


class FoldRouter:
    def __init__(self, theta: AnnealingTheta, filter_u: torch.Tensor | None = None):
        self.theta = theta
        self.filter_u = filter_u
        self.calls = 0

    def __call__(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        levels, masks = self.theta.choose(slots.shape[-1])(
            slots, slot_mask, self.filter_u,
        )
        self.calls += 1
        return levels, masks


@contextmanager
def routed_fold(theta: AnnealingTheta, filter_u: torch.Tensor | None = None):
    original = d07.fold_protocol
    router = FoldRouter(theta, filter_u)
    d07.fold_protocol = router
    try:
        yield router
    finally:
        d07.fold_protocol = original


def load_specimens(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload.get("concepts")
    rows = payload.get("rows")
    if not isinstance(concepts, dict) or not isinstance(rows, list):
        raise ValueError("invalid F04 specimen file")
    if {row.get("split") for row in rows} != {"train", "test"}:
        raise ValueError("F04 requires train and test rows")
    return concepts, rows


def encode_rows(rows: list[dict], sp, pieces: int, eos: int, pad: int, device: str):
    encoded = []
    for row in rows:
        raw = sp.encode(row["source"], out_type=int)
        if not raw or len(raw) > 32:
            raise ValueError(f"F04 source outside 1..32 pieces: {row['id']}={len(raw)}")
        encoded.append([pieces + 2, *raw, eos])
    width = max(map(len, encoded))
    source = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long, device=device)
    for index, row in enumerate(encoded):
        source[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return source, lengths


def compile_concepts(concepts: dict, sp, device: str) -> dict[str, list[torch.Tensor]]:
    compiled = {}
    for name, alternatives in concepts.items():
        sequences = []
        for text in alternatives:
            ids = sp.encode(text, out_type=int)
            if ids:
                sequences.append(torch.tensor(ids, dtype=torch.long, device=device))
        if not sequences:
            raise ValueError(f"concept {name} has no tokenized alternatives")
        compiled[name] = sequences
    return compiled


def phrase_coverage(probability: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    steps = probability.shape[0]
    width = int(token_ids.numel())
    if width > steps:
        return probability.new_tensor(0.0)
    occurrences = []
    for start in range(steps - width + 1):
        value = probability.new_tensor(1.0)
        for offset, token in enumerate(token_ids):
            value = value * probability[start + offset, token]
        occurrences.append(value)
    occurrence = torch.stack(occurrences).clamp(0.0, 1.0 - 1e-7)
    return 1.0 - torch.prod(1.0 - occurrence)


def concept_coverage(probability: torch.Tensor, alternatives: list[torch.Tensor]) -> torch.Tensor:
    values = torch.stack([phrase_coverage(probability, ids) for ids in alternatives])
    values = values.clamp(0.0, 1.0 - 1e-7)
    return 1.0 - torch.prod(1.0 - values)


def behavior_loss(
    logits: torch.Tensor,
    rows: list[dict],
    concepts: dict[str, list[torch.Tensor]],
    eos: int,
) -> tuple[torch.Tensor, dict]:
    probability = F.softmax(logits.float(), dim=-1)
    positive_losses = []
    negative_losses = []
    positive_values = []
    negative_values = []
    per_row = []
    for row_index, row in enumerate(rows):
        local_positive = {}
        local_negative = {}
        for name in row["positive"]:
            coverage = concept_coverage(probability[row_index], concepts[name]).clamp_min(1e-7)
            positive_losses.append(-coverage.log())
            positive_values.append(coverage)
            local_positive[name] = float(coverage.detach().cpu())
        for name in row["negative"]:
            coverage = concept_coverage(probability[row_index], concepts[name]).clamp(0.0, 1.0 - 1e-7)
            negative_losses.append(-torch.log1p(-coverage))
            negative_values.append(coverage)
            local_negative[name] = float(coverage.detach().cpu())
        per_row.append({"id": row["id"], "positive": local_positive, "negative": local_negative})
    positive = torch.stack(positive_losses).mean()
    negative = torch.stack(negative_losses).mean()
    eos_coverage = 1.0 - torch.prod(1.0 - probability[:, :, eos].clamp_max(1.0 - 1e-7), dim=1)
    eos_loss = -eos_coverage.clamp_min(1e-7).log().mean()
    repetition = (probability[:, 1:] * probability[:, :-1]).sum(-1).mean()
    total = positive + 0.25 * negative + 0.10 * eos_loss + 0.20 * repetition
    metrics = {
        "loss": float(total.detach().cpu()),
        "positive_coverage": float(torch.stack(positive_values).mean().detach().cpu()),
        "negative_activation": float(torch.stack(negative_values).mean().detach().cpu()),
        "eos_coverage": float(eos_coverage.mean().detach().cpu()),
        "expected_repetition": float(repetition.detach().cpu()),
        "rows": per_row,
    }
    return total, metrics


def decode_logits(
    model,
    theta: AnnealingTheta,
    source: torch.Tensor,
    lengths: torch.Tensor,
    bos: int,
    depth: int,
    steps: int,
    filter_u: torch.Tensor | None = None,
    prefixes: torch.Tensor | None = None,
):
    with routed_fold(theta, filter_u) as router:
        base_tree, base_masks, budgets, _, _, extra = model.protocol(source, lengths, depth, "native")
    if router.calls != 2:
        raise RuntimeError(f"expected base and extra fold calls, got {router.calls}")
    base_tree = model.reconstructor.convolve(base_tree, base_masks)
    base_hidden = base_tree[0].new_zeros((base_tree[0].shape[0], model.reconstructor.hidden))
    extra_tree, extra_masks, _, _ = extra
    extra_tree = model.extra.reconstructor.convolve(extra_tree, extra_masks)
    extra_hidden = extra_tree[0].new_zeros((extra_tree[0].shape[0], model.extra.reconstructor.hidden))
    previous = torch.full((source.shape[0],), bos, dtype=torch.long, device=source.device)
    logits_rows = []
    predicted = []
    for step in range(steps):
        base_context, _ = model.reconstructor.read(base_hidden, base_tree, base_masks)
        base_hidden = model.reconstructor.cell(
            torch.cat((model.reconstructor.embedding(previous), base_context), dim=-1), base_hidden,
        )
        logits = model.reconstructor.output(torch.cat((base_hidden, base_context), dim=-1))
        extra_context, _ = model.extra.reconstructor.read(extra_hidden, extra_tree, extra_masks)
        extra_hidden = model.extra.reconstructor.cell(
            torch.cat((model.extra.reconstructor.embedding(previous), extra_context), dim=-1), extra_hidden,
        )
        extra_logits = model.extra.reconstructor.output(torch.cat((extra_hidden, extra_context), dim=-1))
        logits = logits + torch.tanh(model.extra_logit_gain) * extra_logits
        token = logits.argmax(-1)
        logits_rows.append(logits)
        predicted.append(token)
        previous = (prefixes[:, step] if prefixes is not None else token).detach()
    return torch.stack(logits_rows, dim=1), torch.stack(predicted, dim=1), budgets


def internal_filter_coordinates(max_slots: int) -> int:
    return max_slots - 1


def find_filter(
    model,
    theta: AnnealingTheta,
    source: torch.Tensor,
    lengths: torch.Tensor,
    rows: list[dict],
    concepts,
    bos: int,
    eos: int,
    depth: int,
    steps: int,
    inner_steps: int,
    inner_lr: float,
):
    previous_requires_grad = [parameter.requires_grad for parameter in theta.parameters()]
    for parameter in theta.parameters():
        parameter.requires_grad_(False)
    filter_u = nn.Parameter(torch.zeros(
        source.shape[0], internal_filter_coordinates(32), device=source.device,
    ))
    optimizer = torch.optim.Adam([filter_u], lr=inner_lr)
    before = after = None
    try:
        for inner in range(inner_steps):
            logits, _, _ = decode_logits(
                model, theta, source, lengths, bos, depth, steps, filter_u=filter_u,
            )
            loss, metrics = behavior_loss(logits, rows, concepts, eos)
            if before is None:
                before = metrics
            regularizer = 0.01 * torch.tanh(filter_u).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (loss + regularizer).backward()
            if filter_u.grad is None or not bool(torch.isfinite(filter_u.grad).all()):
                raise RuntimeError("non-finite inner filter gradient")
            torch.nn.utils.clip_grad_norm_([filter_u], 1.0)
            optimizer.step()
        logits, prefixes, _ = decode_logits(
            model, theta, source, lengths, bos, depth, steps, filter_u=filter_u,
        )
        _, after = behavior_loss(logits, rows, concepts, eos)
    finally:
        for parameter, requires_grad in zip(theta.parameters(), previous_requires_grad):
            parameter.requires_grad_(requires_grad)
    return filter_u.detach(), before, after, prefixes.detach()


def clean(ids: list[int], eos: int, pieces: int) -> list[int]:
    return d10.wmt.clean(ids, eos, pieces)


@torch.no_grad()
def evaluate_rows(
    model, theta, source, lengths, rows, concepts, sp, pieces, eos, bos,
    behavior_steps: int, generation_steps: int,
):
    per_depth = {}
    for depth in DEPTHS:
        logits, _, _ = decode_logits(
            model, theta, source, lengths, bos, depth, behavior_steps,
        )
        _, metrics = behavior_loss(logits, rows, concepts, eos)
        _, generated, budgets = decode_logits(
            model, theta, source, lengths, bos, depth, generation_steps,
        )
        outputs = []
        for index, row in enumerate(rows):
            ids = clean(generated[index].tolist(), eos, pieces)
            outputs.append({
                "id": row["id"], "source": row["source"],
                "generation": sp.decode(ids), "output_pieces": len(ids),
                "budget": int(budgets[index]),
            })
        metrics["outputs"] = outputs
        per_depth[str(depth)] = metrics
    keys = ("loss", "positive_coverage", "negative_activation", "eos_coverage", "expected_repetition")
    mean = {key: sum(per_depth[str(depth)][key] for depth in DEPTHS) / len(DEPTHS) for key in keys}
    return {"mean": mean, "per_depth": per_depth}


def filter_regret(
    model, theta, source, lengths, rows, concepts, bos, eos,
    behavior_steps: int, inner_steps: int, inner_lr: float,
):
    rows_out = []
    for depth in DEPTHS:
        filter_u, before, after, _ = find_filter(
            model, theta, source, lengths, rows, concepts, bos, eos, depth,
            behavior_steps, inner_steps, inner_lr,
        )
        rows_out.append({
            "depth": depth, "native_loss": before["loss"], "filtered_loss": after["loss"],
            "regret": before["loss"] - after["loss"],
            "filter_abs_mean": float(torch.tanh(filter_u).abs().mean().cpu()),
        })
    return {
        "per_depth": rows_out,
        "mean_regret": sum(row["regret"] for row in rows_out) / len(rows_out),
    }


def freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def frozen_exact(model, expected: dict[str, torch.Tensor]) -> bool:
    state = model.state_dict()
    return all(name in state and torch.equal(state[name].detach().cpu(), value) for name, value in expected.items())


def theta_finite(theta: AnnealingTheta) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in theta.parameters())


def absorption_loss(theta: AnnealingTheta, filter_u: torch.Tensor) -> torch.Tensor:
    """Move native recursive gains toward the local causal-filter correction."""
    correction = math.log(1.5) * torch.tanh(filter_u)
    losses = []
    for fold in (theta.base, theta.extra):
        current, valid = fold.gain_vector()
        if current.shape != correction.shape:
            raise RuntimeError(
                f"theta/filter shape mismatch: {tuple(current.shape)} != {tuple(correction.shape)}"
            )
        target = current.detach() + correction
        squared = (current - target).square()
        losses.append((squared * valid.to(squared.dtype)).sum() / valid.sum().clamp_min(1))
    return torch.stack(losses).mean()


def train_arm(
    name: str,
    model,
    theta: AnnealingTheta,
    train_source,
    train_lengths,
    train_rows,
    concepts,
    bos: int,
    eos: int,
    args,
    trace_path: Path,
):
    optimizer = torch.optim.AdamW(theta.parameters(), lr=args.outer_lr, weight_decay=1e-4)
    nonzero_gradient = False
    inner_first = None
    started = time.time()
    for step in range(1, args.outer_steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        filter_metrics = None
        filter_u = None
        if name == "filter-guided-theta":
            filter_u, before, after, _ = find_filter(
                model, theta, train_source, train_lengths, train_rows, concepts,
                bos, eos, depth, args.behavior_steps, args.inner_steps, args.inner_lr,
            )
            filter_metrics = {
                "native_loss": before["loss"], "filtered_loss": after["loss"],
                "regret": before["loss"] - after["loss"],
                "filter_abs_mean": float(torch.tanh(filter_u).abs().mean().cpu()),
            }
            if inner_first is None:
                inner_first = filter_metrics
        logits, _, _ = decode_logits(
            model, theta, train_source, train_lengths, bos, depth,
            args.behavior_steps,
        )
        behavioral, metrics = behavior_loss(logits, train_rows, concepts, eos)
        absorption = logits.new_tensor(0.0)
        if filter_u is not None:
            absorption = absorption_loss(theta, filter_u)
        loss = behavioral + args.absorb_weight * absorption
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in theta.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"non-finite or absent theta gradient in {name} step {step}")
        nonzero_gradient |= any(float(gradient.abs().max()) > 1e-12 for gradient in gradients)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(theta.parameters(), 1.0))
        optimizer.step()
        if not theta_finite(theta):
            raise RuntimeError(f"non-finite theta in {name} step {step}")
        if step == 1 or step % args.log_every == 0 or step == args.outer_steps:
            event = {
                "event": "train", "arm": name, "step": step, "depth": depth,
                "loss": float(loss.detach().cpu()), "behavior": metrics,
                "absorption": float(absorption.detach().cpu()), "grad_norm": grad_norm,
                "filter": filter_metrics, "gains": theta.gain_summary(),
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, event)
            print(json.dumps({
                "event": "train", "arm": name, "step": step, "depth": depth,
                "loss": event["loss"], "positive": metrics["positive_coverage"],
                "negative": metrics["negative_activation"], "filter": filter_metrics,
            }, ensure_ascii=False), flush=True)
    return {"nonzero_gradient": nonzero_gradient, "inner_first": inner_first}


def native_outputs(model, source, lengths, rows, sp, pieces, eos, bos, max_len: int):
    outputs = []
    with torch.no_grad():
        for depth in DEPTHS:
            generated, _, _, _, _ = model.greedy(source, lengths, bos, eos, max_len, depth)
            for index, row in enumerate(rows):
                ids = clean(generated[index].tolist(), eos, pieces)
                outputs.append((row["id"], depth, ids, sp.decode(ids)))
    return outputs


def theta_outputs(model, theta, source, lengths, rows, sp, pieces, eos, bos, max_len: int):
    outputs = []
    with torch.no_grad():
        for depth in DEPTHS:
            _, generated, _ = decode_logits(model, theta, source, lengths, bos, depth, max_len)
            for index, row in enumerate(rows):
                ids = clean(generated[index].tolist(), eos, pieces)
                outputs.append((row["id"], depth, ids, sp.decode(ids)))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--specimens", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11501)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--outer-steps", type=int, default=300)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--outer-lr", type=float, default=2e-3)
    parser.add_argument("--inner-lr", type=float, default=0.15)
    parser.add_argument("--absorb-weight", type=float, default=1.0)
    parser.add_argument("--behavior-steps", type=int, default=24)
    parser.add_argument("--generation-steps", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "smoke":
        args.outer_steps = min(args.outer_steps, 30)
        args.inner_steps = min(args.inner_steps, 3)
        args.eval_rows = min(args.eval_rows, 16)
        args.log_every = min(args.log_every, 10)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    checkpoint = Path(args.checkpoint)
    specimen_path = Path(args.specimens)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    concepts_raw, rows = load_specimens(specimen_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    runtime = load_runtime(checkpoint, args.device)
    payload, base_args, sp, model, pieces, eos, bos, source_hash = runtime
    if model.extra is None:
        raise RuntimeError("F04 requires the TreeHeap-106M extra channel")
    freeze_model(model)
    expected_state = payload["trainable_state_dict"]
    args.pad, args.vocab = pieces, pieces + 3
    base_args.device = args.device
    base_args.eval_batch = args.eval_batch
    base_args.wmt_data = args.eval_wmt_data
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    concepts = compile_concepts(concepts_raw, sp, args.device)
    train_source, train_lengths = encode_rows(train_rows, sp, pieces, eos, args.pad, args.device)
    test_source, test_lengths = encode_rows(test_rows, sp, pieces, eos, args.pad, args.device)
    all_source, all_lengths = encode_rows(rows, sp, pieces, eos, args.pad, args.device)

    template = AnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
    direct_theta = copy.deepcopy(template)
    guided_theta = copy.deepcopy(template)
    native_direct = native_outputs(
        model, all_source, all_lengths, rows, sp, pieces, eos, bos, args.generation_steps,
    )
    theta_direct = theta_outputs(
        model, template, all_source, all_lengths, rows, sp, pieces, eos, bos, args.generation_steps,
    )
    step0_text_exact = all(left[:3] == right[:3] for left, right in zip(native_direct, theta_direct))

    valid_rows, _, _ = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )
    native_valid = d10.valid_summary(model, valid_rows, base_args, args.pad, bos)
    with routed_fold(template):
        theta_valid = d10.valid_summary(model, valid_rows, base_args, args.pad, bos)
    step0_nll_delta = abs(native_valid["mean_nll"] - theta_valid["mean_nll"])

    initial = {
        "train": evaluate_rows(
            model, template, train_source, train_lengths, train_rows, concepts,
            sp, pieces, eos, bos, args.behavior_steps, args.generation_steps,
        ),
        "test": evaluate_rows(
            model, template, test_source, test_lengths, test_rows, concepts,
            sp, pieces, eos, bos, args.behavior_steps, args.generation_steps,
        ),
    }
    initial_regret = filter_regret(
        model, template, train_source, train_lengths, train_rows, concepts, bos, eos,
        args.behavior_steps, args.inner_steps, args.inner_lr,
    )
    contract = {
        "claim": CLAIM, "mode": args.mode, "host": socket.gethostname(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_sha256": source_hash,
        "specimens": str(specimen_path), "specimens_sha256": sha256(specimen_path),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "theta_parameters": sum(parameter.numel() for parameter in template.parameters()),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": vars(args),
        "step0": {
            "text_exact": step0_text_exact, "nll_delta": step0_nll_delta,
            "native_valid": native_valid, "theta_valid": theta_valid,
            "initial": initial, "filter_regret": initial_regret,
        },
    }
    write_json(output / "contract.json", contract)

    direct_trace = output / "direct_theta.trace.jsonl"
    guided_trace = output / "filter_guided_theta.trace.jsonl"
    for path in (direct_trace, guided_trace):
        if path.exists():
            path.unlink()
    direct_train = train_arm(
        "direct-theta", model, direct_theta, train_source, train_lengths, train_rows,
        concepts, bos, eos, args, direct_trace,
    )
    guided_train = train_arm(
        "filter-guided-theta", model, guided_theta, train_source, train_lengths, train_rows,
        concepts, bos, eos, args, guided_trace,
    )

    arms = {}
    for name, theta, train_state in (
        ("direct-theta", direct_theta, direct_train),
        ("filter-guided-theta", guided_theta, guided_train),
    ):
        train_eval = evaluate_rows(
            model, theta, train_source, train_lengths, train_rows, concepts,
            sp, pieces, eos, bos, args.behavior_steps, args.generation_steps,
        )
        test_eval = evaluate_rows(
            model, theta, test_source, test_lengths, test_rows, concepts,
            sp, pieces, eos, bos, args.behavior_steps, args.generation_steps,
        )
        regret = filter_regret(
            model, theta, train_source, train_lengths, train_rows, concepts, bos, eos,
            args.behavior_steps, args.inner_steps, args.inner_lr,
        )
        with routed_fold(theta):
            valid = d10.valid_summary(model, valid_rows, base_args, args.pad, bos)
        checkpoint_path = output / f"{name}.theta.pt"
        torch.save({
            "claim": CLAIM, "arm": name, "theta_state_dict": theta.state_dict(),
            "config": vars(args), "checkpoint_state_sha256": payload["trainable_state_sha256"],
        }, checkpoint_path)
        reloaded = AnnealingTheta(256, model.extra_dim, args.rank).to(args.device)
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        reloaded.load_state_dict(saved["theta_state_dict"])
        reload_eval = evaluate_rows(
            model, reloaded, test_source, test_lengths, test_rows, concepts,
            sp, pieces, eos, bos, args.behavior_steps, args.generation_steps,
        )
        arms[name] = {
            "train_state": train_state, "train": train_eval, "test": test_eval,
            "filter_regret": regret, "valid": valid,
            "theta_finite": theta_finite(theta),
            "reload_test_loss_delta": abs(
                test_eval["mean"]["loss"] - reload_eval["mean"]["loss"]
            ),
            "gains": theta.gain_summary(),
        }

    initial_train = initial["train"]["mean"]
    initial_test = initial["test"]["mean"]
    direct = arms["direct-theta"]
    guided = arms["filter-guided-theta"]
    guided_train_mean = guided["train"]["mean"]
    guided_test_mean = guided["test"]["mean"]
    gates = {
        "P0_strict_origin": step0_text_exact and step0_nll_delta <= 1e-9,
        "P1_filter_local_direction": initial_regret["mean_regret"] > 0.0,
        "P2_finite_gradient_frozen_reload": (
            direct_train["nonzero_gradient"] and guided_train["nonzero_gradient"]
            and direct["theta_finite"] and guided["theta_finite"]
            and frozen_exact(model, expected_state)
            and direct["reload_test_loss_delta"] <= 1e-7
            and guided["reload_test_loss_delta"] <= 1e-7
        ),
        "P3_native_train_conservation": (
            guided_train_mean["positive_coverage"] > initial_train["positive_coverage"]
            and guided_train_mean["negative_activation"] <= initial_train["negative_activation"] + 0.05
            and guided_train_mean["expected_repetition"] <= initial_train["expected_repetition"] + 0.05
        ),
        "P4_heldout_increment": (
            guided_test_mean["positive_coverage"] > initial_test["positive_coverage"]
            and guided_test_mean["positive_coverage"]
            >= direct["test"]["mean"]["positive_coverage"] + 0.02
        ),
        "P5_filter_regret_absorbed": (
            guided["filter_regret"]["mean_regret"] < initial_regret["mean_regret"]
        ),
    }
    summary = {
        "claim": CLAIM, "mode": args.mode, "host": socket.gethostname(),
        "initial": initial, "initial_filter_regret": initial_regret,
        "arms": arms, "gates": gates,
        "frozen_exact": frozen_exact(model, expected_state),
        "seconds": time.time() - started,
        "claim_boundary": "lexical conservation under controlled probes; no full-sentence or role-binding claim",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "event": "complete", "mode": args.mode, "gates": gates,
        "initial_train_positive": initial_train["positive_coverage"],
        "direct_test_positive": direct["test"]["mean"]["positive_coverage"],
        "guided_test_positive": guided_test_mean["positive_coverage"],
        "initial_regret": initial_regret["mean_regret"],
        "guided_regret": guided["filter_regret"]["mean_regret"],
        "seconds": summary["seconds"],
    }, ensure_ascii=False), flush=True)
    safety = gates["P0_strict_origin"] and gates["P2_finite_gradient_frozen_reload"]
    if not safety:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
