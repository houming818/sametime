#!/usr/bin/env python3
"""Exact per-depth task-gradient Gram audit for STONE-2 C03."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


GROUPS = {
    "coarse": (0, 1, 2),
    "middle": (3, 4, 5),
    "fine": (6, 7),
}
PAIRS = (("coarse", "middle"), ("coarse", "fine"), ("middle", "fine"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flat_grad(module: nn.Module) -> torch.Tensor:
    parts = []
    for parameter in module.parameters():
        if parameter.grad is None:
            parts.append(torch.zeros_like(parameter).reshape(-1))
        else:
            parts.append(parameter.grad.detach().reshape(-1))
    return torch.cat(parts)


def vector_sum(vectors) -> torch.Tensor:
    if not vectors:
        raise ValueError("cannot sum an empty gradient list")
    result = torch.zeros_like(vectors[0])
    for vector in vectors:
        result = result + vector
    return result


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def make_untied_read():
    def read(self, hidden, tree, masks, mode="native", ablate_depth=-1):
        del mode, ablate_depth
        base_query = self.query(hidden)
        query = base_query
        frontier = masks[0].to(query.dtype)
        entropies = []
        gain = torch.sigmoid(self.read_gain_logit)
        last_depth = len(tree) - 1
        for depth, (nodes, valid) in enumerate(zip(tree, masks)):
            frontier = frontier * valid.to(frontier.dtype)
            frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
            local = (frontier[:, :, None] * nodes).sum(1)
            depth_state = self.depth_embedding.weight[depth][None].expand_as(local)
            query = query + gain * self.read_kernel_by_depth[depth](query, local, depth_state)
            entropies.append(
                -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1).mean()
            )
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            branch_query = (
                self.branch_by_depth[depth](hidden) + gain * (query - base_query)
            )[:, None, None]
            scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
            scores = scores.masked_fill(~child_valid, -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * child_valid.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return local + (query - base_query), torch.stack(entropies)

    return read


def install_untied_modules(model) -> None:
    decoder = model.decoder
    depths = int(decoder.depth_embedding.num_embeddings)
    decoder.read_kernel_by_depth = nn.ModuleList(
        copy.deepcopy(decoder.read_kernel) for _ in range(depths)
    )
    decoder.branch_by_depth = nn.ModuleList(
        copy.deepcopy(decoder.branch) for _ in range(max(1, depths - 1))
    )
    decoder.read = MethodType(make_untied_read(), decoder)


def build_model(payload, device, vocab, pad):
    config = SimpleNamespace(**payload["config"])
    config.device = device
    model = integrated.build_integrated_model(config, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def loss_for_batch(model, batch, pad, bos, device):
    source, length, target = c10.collate_rows(batch, pad, device)
    logits, _ = model.teacher(source, length, target, bos)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad,
    )
    return logits, loss


def grouped(vectors, group_name, available):
    selected = [vectors[depth] for depth in GROUPS[group_name] if depth < available]
    return vector_sum(selected)


def module_metrics(native_vector, clone_vectors, available):
    clone_vectors = clone_vectors[:available]
    reconstructed = vector_sum(clone_vectors)
    relative_error = float(
        (reconstructed - native_vector).norm() / native_vector.norm().clamp_min(1e-12)
    )
    groups = {name: grouped(clone_vectors, name, available) for name in GROUPS}
    pair_metrics = {}
    for left, right in PAIRS:
        pair_metrics[f"{left}:{right}"] = {
            "cosine": cosine(groups[left], groups[right]),
            "dot": float(torch.dot(groups[left], groups[right])),
        }
    sum_group = vector_sum(list(groups.values()))
    norm_sum = sum(float(vector.norm()) for vector in groups.values())
    return {
        "native_norm": float(native_vector.norm()),
        "reconstructed_norm": float(reconstructed.norm()),
        "reconstruction_relative_error": relative_error,
        "depth_norms": [float(vector.norm()) for vector in clone_vectors],
        "group_norms": {name: float(vector.norm()) for name, vector in groups.items()},
        "pairs": pair_metrics,
        "cancellation_ratio": float(sum_group.norm()) / max(1e-12, norm_sum),
    }


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def quantile(values, q):
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def aggregate(batch_rows, module_name):
    result = {"pairs": {}}
    for pair in (f"{a}:{b}" for a, b in PAIRS):
        values = [row[module_name]["pairs"][pair]["cosine"] for row in batch_rows]
        result["pairs"][pair] = {
            "cosine_mean": sum(values) / len(values),
            "cosine_median": quantile(values, 0.5),
            "cosine_min": min(values),
            "negative_fraction": sum(value < 0.0 for value in values) / len(values),
        }
    ratios = [row[module_name]["cancellation_ratio"] for row in batch_rows]
    errors = [row[module_name]["reconstruction_relative_error"] for row in batch_rows]
    result["cancellation_ratio_mean"] = sum(ratios) / len(ratios)
    result["cancellation_ratio_median"] = quantile(ratios, 0.5)
    result["max_reconstruction_relative_error"] = max(errors)
    return result


def conflict_gate(module_summary) -> bool:
    return any(
        row["cosine_median"] <= -0.05 and row["negative_fraction"] >= 0.50
        for row in module_summary["pairs"].values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=16301)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-source-pieces", type=int, default=65)
    parser.add_argument("--max-source-pieces", type=int, default=128)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.task_eval_rows = max(1024, args.batches * args.batch_size * 4)
    sp = spm.SentencePieceProcessor(model_file=config.spm_model)
    pieces = sp.get_piece_size()
    pad, bos, eos = pieces, sp.bos_id(), sp.eos_id()
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    eligible = [
        row for row in test_rows
        if args.min_source_pieces <= len(row[0]) <= args.max_source_pieces
    ]
    random.shuffle(eligible)
    required = args.batches * args.batch_size
    if len(eligible) < required:
        print(json.dumps(
            {
                "event": "insufficient_full_depth_rows",
                "eligible": len(eligible),
                "required": required,
                "min_source_pieces": args.min_source_pieces,
                "max_source_pieces": args.max_source_pieces,
                "maximum_complete_batches": len(eligible) // args.batch_size,
            },
            ensure_ascii=False,
        ), flush=True)
        raise RuntimeError(
            f"needed {required} full-depth rows, found {len(eligible)}; "
            "reduce --batches without duplicating samples"
        )
    selected = eligible[:required]
    if len(selected) != required:
        raise RuntimeError(f"needed {required} rows, found {len(selected)}")

    native, _ = build_model(payload, args.device, pieces + 3, pad)
    untied, _ = build_model(payload, args.device, pieces + 3, pad)
    install_untied_modules(untied)
    batches = []
    for batch_index in range(args.batches):
        batch = selected[batch_index * args.batch_size:(batch_index + 1) * args.batch_size]
        native.zero_grad(set_to_none=True)
        native_logits, native_loss = loss_for_batch(native, batch, pad, bos, args.device)
        native_loss.backward()
        native_read = flat_grad(native.decoder.read_kernel)
        native_branch = flat_grad(native.decoder.branch)

        untied.zero_grad(set_to_none=True)
        untied_logits, untied_loss = loss_for_batch(untied, batch, pad, bos, args.device)
        untied_loss.backward()
        required_width = max(len(item[0]) for item in batch)
        active_width = 1 << (required_width - 1).bit_length()
        levels = int(math.log2(active_width)) + 1
        read_vectors = [flat_grad(module) for module in untied.decoder.read_kernel_by_depth]
        branch_vectors = [flat_grad(module) for module in untied.decoder.branch_by_depth]
        active_read = min(levels, len(read_vectors))
        active_branch = min(levels - 1, len(branch_vectors))
        row = {
            "batch": batch_index,
            "source_lengths": [len(item[0]) for item in batch],
            "native_loss": float(native_loss.detach()),
            "untied_loss": float(untied_loss.detach()),
            "loss_abs_delta": abs(float(native_loss.detach() - untied_loss.detach())),
            "logits_max_abs_delta": float(
                (native_logits - untied_logits).detach().abs().max()
            ),
            "active_read_depths": active_read,
            "active_branch_depths": active_branch,
            "read": module_metrics(native_read, read_vectors, active_read),
            "branch": module_metrics(native_branch, branch_vectors, active_branch),
        }
        batches.append(row)

    read_summary = aggregate(batches, "read")
    branch_summary = aggregate(batches, "branch")
    max_logits = max(row["logits_max_abs_delta"] for row in batches)
    max_loss = max(row["loss_abs_delta"] for row in batches)
    all_finite = finite_numbers(batches)
    full_depth_coverage = all(
        row["active_read_depths"] == 8 and row["active_branch_depths"] == 7
        for row in batches
    )
    a0 = (
        all_finite
        and full_depth_coverage
        and max_logits < 1e-6
        and max_loss < 1e-7
        and read_summary["max_reconstruction_relative_error"] < 1e-5
        and branch_summary["max_reconstruction_relative_error"] < 1e-5
    )
    read_conflict = conflict_gate(read_summary)
    branch_conflict = conflict_gate(branch_summary)
    strong_cancellation = min(
        read_summary["cancellation_ratio_median"],
        branch_summary["cancellation_ratio_median"],
    ) <= 0.80
    if not a0:
        decision = "invalid_implementation_stop"
    elif read_conflict and branch_conflict:
        decision = "register_two_factor_matched_smoke"
    elif read_conflict:
        decision = "register_grouped_read_matched_smoke"
    elif branch_conflict:
        decision = "register_grouped_branch_matched_smoke"
    else:
        decision = "do_not_split_shared_kernels"

    source_digest = hashlib.sha256()
    for row in selected:
        source_digest.update(json.dumps(row[0], separators=(",", ":")).encode("ascii"))
    result = {
        "audit": "S3-STONE2-C03-D02-TASK-GRADIENT-GRAM",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rows": len(selected),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "selected_source_ids_sha256": source_digest.hexdigest(),
        "groups": {name: list(depths) for name, depths in GROUPS.items()},
        "batch_results": batches,
        "aggregate": {"read": read_summary, "branch": branch_summary},
        "gates": {
            "A0_exact_untie_and_finite": a0,
            "A0_full_depth_coverage": full_depth_coverage,
            "A1_read_conflict": read_conflict,
            "A2_branch_conflict": branch_conflict,
            "A3_strong_cancellation": strong_cancellation,
        },
        "decision": decision,
        "formal_training_authorized": False,
    }
    write_json(args.output, result)
    print(json.dumps({
        "event": "complete",
        "gates": result["gates"],
        "decision": decision,
        "read": read_summary,
        "branch": branch_summary,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
