#!/usr/bin/env python3
"""A16 zero-training multiresolution Bayesian READ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sentencepiece as spm
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidate(raw: str) -> Tuple[str, Path, Path]:
    name, paths = raw.split("=", 1)
    checkpoint, counts = paths.split("::", 1)
    return name, Path(checkpoint), Path(counts)


def iter_english(path: Path) -> Iterable[Tuple[int, bytes, str]]:
    with path.open("rb") as handle:
        for raw_index, raw in enumerate(handle):
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError:
                continue
            if len(fields) >= 2 and fields[1].strip():
                yield raw_index, raw, fields[1]


def extract_examples(
    data_path: Path,
    spm_path: Path,
    target_ids: Sequence[int],
    context_ids: Sequence[int],
    start_valid_line: int,
    scan_valid_lines: int,
    max_examples: int,
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    target_lookup = {int(token): index for index, token in enumerate(target_ids)}
    context_lookup = {int(token): index for index, token in enumerate(context_ids)}
    max_context = 2 * window
    targets: List[int] = []
    contexts: List[List[int]] = []
    valid_index = 0
    digest = hashlib.sha256()
    first_raw = None
    last_raw = None

    for raw_index, raw, text in iter_english(data_path):
        if valid_index < start_valid_line:
            valid_index += 1
            continue
        if valid_index >= start_valid_line + scan_valid_lines or len(targets) >= max_examples:
            break
        digest.update(raw)
        first_raw = raw_index if first_raw is None else first_raw
        last_raw = raw_index
        token_ids = sp.encode(text, out_type=int)
        for position, token_id in enumerate(token_ids):
            target = target_lookup.get(token_id)
            if target is None:
                continue
            bag = []
            left = max(0, position - window)
            right = min(len(token_ids), position + window + 1)
            for context_position in range(left, right):
                if context_position == position:
                    continue
                context = context_lookup.get(token_ids[context_position])
                if context is not None:
                    bag.append(context)
            if bag:
                targets.append(target)
                contexts.append(bag[:max_context])
                if len(targets) >= max_examples:
                    break
        valid_index += 1

    if len(targets) < max_examples:
        raise RuntimeError(f"collected {len(targets)} examples, need {max_examples}")
    context_tensor = torch.full((len(contexts), max_context), -1, dtype=torch.long)
    for index, bag in enumerate(contexts):
        context_tensor[index, : len(bag)] = torch.tensor(bag, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    metadata = {
        "start_valid_line": start_valid_line,
        "scan_valid_lines": scan_valid_lines,
        "first_raw_line": first_raw,
        "last_raw_line": last_raw,
        "region_sha256": digest.hexdigest(),
        "examples": len(targets),
        "window": window,
        "mean_context_tokens": float((context_tensor >= 0).sum(dim=1).float().mean()),
    }
    return target_tensor, context_tensor, metadata


def normalize_probability(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = value.to(torch.float64).clamp_min(0)
    value = value + epsilon
    return value / value.sum(dim=1, keepdim=True)


def node_ids_at_depth(path_bits: torch.Tensor, depth: int) -> torch.Tensor:
    nodes = torch.zeros(path_bits.shape[0], dtype=torch.long)
    for level in range(depth):
        nodes = 2 * nodes + 1 + path_bits[:, level].to(torch.long)
    return nodes


def make_fields(checkpoint: Dict, epsilon: float) -> Tuple[Dict[str, torch.Tensor], Dict]:
    depth = int(checkpoint["depth"])
    bits = checkpoint["token_path_bits"].to(torch.long)
    node_probability = checkpoint["node_probability"].to(torch.float64)
    node_residual = checkpoint["node_residual"].to(torch.float64)
    full = checkpoint["token_context_sqrt_probability"].to(torch.float64).square()
    full = normalize_probability(full, epsilon)
    fields: Dict[str, torch.Tensor] = {}
    node_ids: Dict[str, torch.Tensor] = {}
    for level in range(depth + 1):
        ids = node_ids_at_depth(bits, level)
        node_ids[f"depth_{level}"] = ids
        fields[f"depth_{level}"] = normalize_probability(node_probability[ids], epsilon)
    fields["full_token"] = full

    residual_closure = 0.0
    residual_mass = 0.0
    for level in range(depth + 1):
        ids = node_ids[f"depth_{level}"]
        reconstructed = node_probability[0].repeat(bits.shape[0], 1)
        if level:
            for step in range(1, level + 1):
                step_ids = node_ids[f"depth_{step}"]
                reconstructed = reconstructed + node_residual[step_ids]
        residual_closure = max(residual_closure, float((reconstructed - node_probability[ids]).abs().max()))
    for node in range(1, len(node_residual)):
        residual_mass = max(residual_mass, abs(float(node_residual[node].sum())))

    leaf = fields[f"depth_{depth}"]
    token_residual = full - leaf
    token_closure = float((leaf + token_residual - full).abs().max())
    token_residual_mass = float(token_residual.sum(dim=1).abs().max())
    simplex_error = max(float((field.sum(dim=1) - 1).abs().max()) for field in fields.values())
    audit = {
        "depth": depth,
        "simplex_max_abs": simplex_error,
        "node_residual_closure_max_abs": residual_closure,
        "node_residual_mass_max_abs": residual_mass,
        "token_residual_closure_max_abs": token_closure,
        "token_residual_mass_max_abs": token_residual_mass,
    }
    return fields, {"audit": audit, "node_ids": node_ids}


@torch.no_grad()
def evaluate(
    probability: torch.Tensor | None,
    log_prior: torch.Tensor,
    targets: torch.Tensor,
    contexts: torch.Tensor,
    batch: int,
    device: torch.device,
) -> Dict[str, float]:
    log_prior = log_prior.to(device)
    log_probability = None if probability is None else probability.to(device).log()
    total_nll = 0.0
    top1 = 0
    top5 = 0
    reciprocal_rank = 0.0
    count = 0
    for start in range(0, len(targets), batch):
        target = targets[start : start + batch].to(device)
        context = contexts[start : start + batch].to(device)
        if log_probability is None:
            score = log_prior.unsqueeze(0).expand(len(target), -1)
        else:
            valid = context >= 0
            safe = context.clamp_min(0)
            selected = log_probability[:, safe]
            selected = selected * valid.unsqueeze(0)
            score = log_prior.unsqueeze(0) + selected.sum(dim=2).transpose(0, 1)
        posterior_nll = -score[torch.arange(len(target), device=device), target] + torch.logsumexp(score, dim=1)
        total_nll += float(posterior_nll.sum())
        order = torch.argsort(score, dim=1, descending=True)
        matches = order == target.unsqueeze(1)
        ranks = matches.to(torch.int64).argmax(dim=1) + 1
        top1 += int((ranks == 1).sum())
        top5 += int((ranks <= 5).sum())
        reciprocal_rank += float((1.0 / ranks.to(torch.float64)).sum())
        count += len(target)
    return {
        "nll": total_nll / count,
        "top1": top1 / count,
        "top5": top5 / count,
        "mrr": reciprocal_rank / count,
        "examples": count,
    }


def permutation(size: int, seed: int) -> torch.Tensor:
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed))


def metric_max_abs(left: Dict[str, float], right: Dict[str, float]) -> float:
    return max(abs(left[key] - right[key]) for key in ["nll", "top1", "top5", "mrr"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="NAME=CHECKPOINT::COUNTS")
    parser.add_argument("--data", required=True)
    parser.add_argument("--spm-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-valid-line", type=int, default=1100000)
    parser.add_argument("--scan-valid-lines", type=int, default=100000)
    parser.add_argument("--examples", type=int, default=50000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--shuffle-seeds", default="20261001,20261002,20261003")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [parse_candidate(raw) for raw in args.candidate]
    loaded = {}
    for name, checkpoint_path, counts_path in specs:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        counts = torch.load(counts_path, map_location="cpu", weights_only=False)
        loaded[name] = (checkpoint_path, counts_path, checkpoint, counts)

    first_checkpoint = next(iter(loaded.values()))[2]
    target_ids = [int(value) for value in first_checkpoint["target_ids"]]
    context_ids = [int(value) for value in first_checkpoint["context_ids"]]
    for name, (_, _, checkpoint, counts) in loaded.items():
        if [int(value) for value in checkpoint["target_ids"]] != target_ids:
            raise RuntimeError(f"target vocabulary mismatch: {name}")
        if [int(value) for value in checkpoint["context_ids"]] != context_ids:
            raise RuntimeError(f"context vocabulary mismatch: {name}")
        if [int(value) for value in counts["target_ids"]] != target_ids:
            raise RuntimeError(f"count target order mismatch: {name}")
        if [int(value) for value in counts["context_ids"]] != context_ids:
            raise RuntimeError(f"count context order mismatch: {name}")

    targets, contexts, data_meta = extract_examples(
        Path(args.data), Path(args.spm_model), target_ids, context_ids,
        args.start_valid_line, args.scan_valid_lines, args.examples, args.window,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    shuffle_seeds = [int(value) for value in args.shuffle_seeds.split(",") if value]
    summary: Dict = {
        "claim": "S1-BAYES-READ-A16-C01",
        "format": "zero_training_multiresolution_bayesian_read_a16_v1",
        "contract": vars(args),
        "trainable_parameter_count": 0,
        "data": data_meta,
        "data_size_bytes": Path(args.data).stat().st_size,
        "spm_sha256": sha256(Path(args.spm_model)),
        "candidates": {},
    }

    reference_data_signature = hashlib.sha256(targets.numpy().tobytes() + contexts.numpy().tobytes()).hexdigest()
    summary["evaluation_tensor_sha256"] = reference_data_signature
    for name, (checkpoint_path, counts_path, checkpoint, counts) in loaded.items():
        fields, structure = make_fields(checkpoint, args.epsilon)
        train_counts = counts["train"].to(torch.float64)
        prior = train_counts.sum(dim=1) + 1.0
        prior = prior / prior.sum()
        log_prior = prior.log()
        prior_result = evaluate(None, log_prior, targets, contexts, args.batch, device)
        native = {field_name: evaluate(field, log_prior, targets, contexts, args.batch, device)
                  for field_name, field in fields.items()}
        controls = {}
        for seed in shuffle_seeds:
            order = permutation(len(target_ids), seed)
            controls[str(seed)] = {
                "row_shuffle_full": evaluate(fields["full_token"][order], log_prior, targets, contexts, args.batch, device),
                "path_shuffle": {
                    field_name: evaluate(field[order], log_prior, targets, contexts, args.batch, device)
                    for field_name, field in fields.items() if field_name.startswith("depth_")
                },
            }
            print(json.dumps({
                "candidate": name,
                "shuffle_seed": seed,
                "native_full_top1": native["full_token"]["top1"],
                "shuffle_full_top1": controls[str(seed)]["row_shuffle_full"]["top1"],
            }), flush=True)

        depth_zero_error = metric_max_abs(prior_result, native["depth_0"])
        audit = structure["audit"]
        finite = all(
            math.isfinite(metric)
            for result in [prior_result, *native.values()]
            for key, metric in result.items() if key != "examples"
        )
        shuffled_full_top1 = [controls[str(seed)]["row_shuffle_full"]["top1"] for seed in shuffle_seeds]
        shuffled_full_nll = [controls[str(seed)]["row_shuffle_full"]["nll"] for seed in shuffle_seeds]
        aggregate = {
            "native_full_minus_shuffle_top1": native["full_token"]["top1"] - float(np.median(shuffled_full_top1)),
            "shuffle_minus_native_full_nll": float(np.median(shuffled_full_nll)) - native["full_token"]["nll"],
            "field_causal_predictive_information": (
                native["full_token"]["top1"] > float(np.median(shuffled_full_top1))
                and native["full_token"]["nll"] < float(np.median(shuffled_full_nll))
            ),
        }
        gates = {
            "zero_trainable_parameters": True,
            "finite": finite,
            "simplex": audit["simplex_max_abs"] <= 1e-6,
            "node_residual_closure": audit["node_residual_closure_max_abs"] <= 1e-6,
            "token_residual_closure": audit["token_residual_closure_max_abs"] <= 1e-6,
            "depth_zero_equals_prior": depth_zero_error <= 1e-8,
            "same_evaluation_examples": True,
        }
        gates["pass"] = all(gates.values())
        summary["candidates"][name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "counts": str(counts_path),
            "counts_sha256": sha256(counts_path),
            "prior_only": prior_result,
            "native": native,
            "controls": controls,
            "audit": audit,
            "depth_zero_prior_metric_max_abs": depth_zero_error,
            "aggregate": aggregate,
            "mechanical_gates": gates,
        }
    summary["claim_result"] = {
        "mechanical_gates_all": all(candidate["mechanical_gates"]["pass"] for candidate in summary["candidates"].values()),
        "field_causal_predictive_information_all": all(
            candidate["aggregate"]["field_causal_predictive_information"]
            for candidate in summary["candidates"].values()
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["claim_result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
