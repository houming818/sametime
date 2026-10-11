#!/usr/bin/env python3
"""Inspect multiscale structure in a frozen TreeHeap probability field."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch

import s1_multiscale_probability_embedding_t0 as t0
import s1_treeheap_background_landmarks as legacy


EPS = 1e-12


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def tree_coordinates(probability: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    leaves = probability.shape[1]
    depth = int(math.log2(leaves))
    if 2**depth != leaves:
        raise ValueError("leaf count must be a power of two")
    masses: List[torch.Tensor] = []
    splits: List[torch.Tensor] = []
    details: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    for level in range(depth):
        nodes = 2**level
        width = leaves // nodes
        parent = probability.reshape(len(probability), nodes, width).sum(dim=2)
        left = probability.reshape(len(probability), nodes, width)[:, :, : width // 2].sum(dim=2)
        right = parent - left
        split = left / parent.clamp_min(EPS)
        masses.append(parent)
        splits.append(split)
        details.append(2.0 * split - 1.0)
        logits.append((left.clamp_min(EPS) / right.clamp_min(EPS)).log())
    return {"masses": masses, "splits": splits, "details": details, "logits": logits}


def reconstruct_from_splits(
    splits: Sequence[torch.Tensor], keep: int | None = None
) -> torch.Tensor:
    rows = splits[0].shape[0]
    flat_splits = torch.cat(list(splits), dim=1)
    if keep is not None and keep < flat_splits.shape[1]:
        flat_masses = internal_masses_from_splits(splits)
        score = flat_masses * (2.0 * flat_splits - 1.0).abs()
        chosen = torch.topk(score, keep, dim=1).indices
        masked = torch.full_like(flat_splits, 0.5)
        masked.scatter_(1, chosen, flat_splits.gather(1, chosen))
        level_splits = []
        cursor = 0
        for level in range(len(splits)):
            width = 2**level
            level_splits.append(masked[:, cursor : cursor + width])
            cursor += width
    else:
        level_splits = list(splits)
    mass = torch.ones((rows, 1), dtype=flat_splits.dtype, device=flat_splits.device)
    for split in level_splits:
        left = mass * split
        right = mass * (1.0 - split)
        mass = torch.stack((left, right), dim=2).reshape(rows, -1)
    return mass


def internal_masses_from_splits(splits: Sequence[torch.Tensor]) -> torch.Tensor:
    rows = splits[0].shape[0]
    mass = torch.ones((rows, 1), dtype=splits[0].dtype, device=splits[0].device)
    output = []
    for split in splits:
        output.append(mass)
        mass = torch.stack((mass * split, mass * (1.0 - split)), dim=2).reshape(rows, -1)
    return torch.cat(output, dim=1)


def binary_entropy(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(EPS, 1.0 - EPS)
    return -value * value.log() - (1.0 - value) * (1.0 - value).log()


def js_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.clamp_min(EPS)
    right = right.clamp_min(EPS)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=1)
        + (right * (right.log() - middle.log())).sum(dim=1)
    )


def hierarchy_audit(
    train_probability: torch.Tensor,
    test_counts: torch.Tensor,
    prior: torch.Tensor,
    keep_values: Sequence[int],
) -> Dict[str, object]:
    coord = tree_coordinates(train_probability)
    reconstructed = reconstruct_from_splits(coord["splits"])
    mass_error = max(
        float((mass.sum(dim=1) - 1.0).abs().max()) for mass in coord["masses"]
    )
    reconstruction_error = float((reconstructed - train_probability).abs().max())

    leaf_entropy = -(train_probability.clamp_min(EPS) * train_probability.clamp_min(EPS).log()).sum(dim=1)
    entropy_by_depth = [
        (mass * binary_entropy(split)).sum(dim=1)
        for mass, split in zip(coord["masses"], coord["splits"])
    ]
    entropy_chain = torch.stack(entropy_by_depth, dim=1).sum(dim=1)

    valid = test_counts.sum(dim=1) > 0
    empirical = test_counts[valid].to(torch.float64)
    empirical /= empirical.sum(dim=1, keepdim=True).clamp_min(EPS)
    predicted = train_probability[valid]
    target_coord = tree_coordinates(empirical)
    predicted_coord = {
        key: [item[valid] for item in value] for key, value in coord.items()
    }
    leaf_ce = -(empirical * predicted.clamp_min(EPS).log()).sum(dim=1)
    ce_by_depth = []
    for target_mass, target_split, predicted_split in zip(
        target_coord["masses"], target_coord["splits"], predicted_coord["splits"]
    ):
        local = -target_split * predicted_split.clamp_min(EPS).log()
        local -= (1.0 - target_split) * (1.0 - predicted_split).clamp_min(EPS).log()
        ce_by_depth.append((target_mass * local).sum(dim=1))
    ce_chain = torch.stack(ce_by_depth, dim=1).sum(dim=1)

    normalized_prior = prior / prior.sum().clamp_min(EPS)
    energy = [
        float((normalized_prior * (mass * detail.square()).sum(dim=1)).sum())
        for mass, detail in zip(coord["masses"], coord["details"])
    ]
    entropy_contribution = [
        float((normalized_prior * value).sum()) for value in entropy_by_depth
    ]
    valid_prior = normalized_prior[valid]
    valid_prior /= valid_prior.sum().clamp_min(EPS)
    ce_contribution = [float((valid_prior * value).sum()) for value in ce_by_depth]

    compression = []
    for keep in keep_values:
        approximation = reconstruct_from_splits(coord["splits"], keep)
        approximation_valid = approximation[valid]
        heldout_ce = -(empirical * approximation_valid.clamp_min(EPS).log()).sum(dim=1)
        compression.append({
            "kept_details": int(keep),
            "mean_js": float((normalized_prior * js_rows(train_probability, approximation)).sum()),
            "heldout_cross_entropy": float((valid_prior * heldout_ce).sum()),
            "max_abs": float((train_probability - approximation).abs().max()),
        })

    logits = torch.cat(coord["logits"], dim=1)
    centered = logits - (normalized_prior[:, None] * logits).sum(dim=0, keepdim=True)
    weighted = centered * normalized_prior.sqrt()[:, None]
    singular = torch.linalg.svdvals(weighted)
    variance = singular.square()
    variance /= variance.sum().clamp_min(EPS)
    cumulative = variance.cumsum(dim=0)

    return {
        "mass_conservation_max_abs": mass_error,
        "full_reconstruction_max_abs": reconstruction_error,
        "entropy_chain_max_abs": float((leaf_entropy - entropy_chain).abs().max()),
        "heldout_ce_chain_max_abs": float((leaf_ce - ce_chain).abs().max()),
        "valid_test_rows": int(valid.sum()),
        "detail_energy_by_depth": energy,
        "entropy_contribution_by_depth": entropy_contribution,
        "heldout_ce_contribution_by_depth": ce_contribution,
        "compression": compression,
        "logit_spectrum": {
            "singular_values": singular.cpu().tolist(),
            "cumulative_variance_top1": float(cumulative[0]),
            "cumulative_variance_top2": float(cumulative[min(1, len(cumulative) - 1)]),
            "cumulative_variance_top4": float(cumulative[min(3, len(cumulative) - 1)]),
            "cumulative_variance_top8": float(cumulative[min(7, len(cumulative) - 1)]),
        },
    }


def polynomial_features(parent_logit: torch.Tensor, parent_mass: torch.Tensor, degree: int) -> torch.Tensor:
    log_mass = parent_mass.clamp_min(EPS).log()
    columns = [torch.ones_like(parent_logit), parent_logit, log_mass]
    if degree >= 2:
        columns.extend([parent_logit.square(), log_mass.square(), parent_logit * log_mass])
    return torch.stack(columns, dim=1)


def ridge_fit_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    identity = torch.eye(x_train.shape[1], dtype=x_train.dtype, device=x_train.device)
    identity[0, 0] = 0.0
    beta = torch.linalg.solve(x_train.t() @ x_train + ridge * identity, x_train.t() @ y_train)
    return x_test @ beta


def recurrence_audit(probability: torch.Tensor, degree: int, ridge: float) -> Dict[str, float]:
    coord = tree_coordinates(probability)
    rows = torch.arange(len(probability), device=probability.device)
    train_mask = rows % 5 != 0
    test_mask = ~train_mask
    predictions = []
    targets = []
    baselines = []
    for level in range(len(coord["logits"]) - 1):
        parent_logits = coord["logits"][level]
        parent_masses = coord["masses"][level]
        child_logits = coord["logits"][level + 1]
        for parent_index in range(parent_logits.shape[1]):
            for side in range(2):
                child_index = 2 * parent_index + side
                x = polynomial_features(
                    parent_logits[:, parent_index], parent_masses[:, parent_index], degree
                )
                y = child_logits[:, child_index]
                prediction = ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask], ridge)
                predictions.append(prediction)
                targets.append(y[test_mask])
                baselines.append(torch.full_like(y[test_mask], float(y[train_mask].mean())))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    baseline = torch.cat(baselines)
    ss_res = (target - prediction).square().sum()
    ss_tot = (target - baseline).square().sum().clamp_min(EPS)
    return {
        "degree": degree,
        "train_token_rows": int(train_mask.sum()),
        "test_token_rows": int(test_mask.sum()),
        "edges": len(predictions),
        "r2": float(1.0 - ss_res / ss_tot),
        "rmse": float((ss_res / len(target)).sqrt()),
    }


def arm(
    name: str,
    train: torch.Tensor,
    test: torch.Tensor,
    leaf: torch.Tensor,
    leaves: int,
    alpha: float,
    prior: torch.Tensor,
    keep_values: Sequence[int],
    ridge: float,
) -> Dict[str, object]:
    train_probability = t0.aggregate_by_leaf(train, leaf, leaves, alpha)
    test_counts = t0.aggregate_counts(test, leaf, leaves)
    return {
        "name": name,
        "leaf_counts": torch.bincount(leaf, minlength=leaves).cpu().tolist(),
        "field_sha256": tensor_sha256(train_probability),
        "hierarchy": hierarchy_audit(train_probability, test_counts, prior, keep_values),
        "recurrence": {
            "linear": recurrence_audit(train_probability, 1, ridge),
            "quadratic": recurrence_audit(train_probability, 2, ridge),
        },
    }


def metric_at_keep(row: Dict[str, object], keep: int) -> Dict[str, float]:
    return next(item for item in row["hierarchy"]["compression"] if item["kept_details"] == keep)


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--target-vocab", type=int, default=2_048)
    parser.add_argument("--context-vocab", type=int, default=1_024)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--power-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--keep", default="1,2,4,8,15")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    leaves = 2**args.depth
    keep_values = sorted({int(value) for value in args.keep.split(",")})
    if keep_values[-1] != leaves - 1:
        raise ValueError("keep list must include all internal details")
    t0.seed_all(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    processor = spm.SentencePieceProcessor(model_file=args.spm_model)
    cfg = legacy.Config(
        data=args.data,
        spm_model=args.spm_model,
        simlex="",
        max_scan_lines=args.max_lines,
        target_vocab=args.target_vocab,
        context_vocab=args.context_vocab,
        window=args.window,
        max_sentence_tokens=96,
        test_mod=10,
        depth=args.depth,
        bootstrap_seeds=1,
        seed_start=args.seed,
        temperatures=[1.0],
        em_steps=1,
        power_steps=args.power_steps,
        alpha=args.alpha,
        sgns_dim=1,
        sgns_steps=1,
        sgns_batch=1,
        sgns_negatives=1,
        sgns_lr=0.01,
        device=args.device,
    )
    vocabulary = legacy.choose_vocab(cfg, processor)
    vocabulary.pop("counts")
    target_ids = vocabulary.pop("target_ids")
    context_ids = vocabulary.pop("context_ids")
    train_cpu, test_cpu, count_stats = legacy.build_counts(cfg, processor, target_ids, context_ids)
    train = train_cpu.to(device)
    test = test_cpu.to(device)
    features, context_mass = t0.context_features(train, args.alpha)
    learned_leaf, split_trace = t0.learned_leaf_assignment(
        features, context_mass, args.depth, args.power_steps
    )
    prior = train.sum(dim=1).to(torch.float64).clamp_min(1.0)
    prior /= prior.sum()

    learned = arm(
        "learned", train, test, learned_leaf, leaves, args.alpha, prior, keep_values, args.ridge
    )
    random_arms = []
    permutation_arms = []
    for index in range(args.controls):
        random_leaf = t0.random_leaf_assignment(
            args.context_vocab, leaves, args.seed + 100 + index, device
        )
        random_arms.append(
            arm(
                f"random_{index}", train, test, random_leaf, leaves, args.alpha,
                prior, keep_values, args.ridge,
            )
        )
        generator = torch.Generator().manual_seed(args.seed + 200 + index)
        permutation = torch.randperm(leaves, generator=generator).to(device)
        permuted_leaf = permutation[learned_leaf]
        permutation_arms.append(
            arm(
                f"leaf_permutation_{index}", train, test, permuted_leaf, leaves,
                args.alpha, prior, keep_values, args.ridge,
            )
        )

    learned_k4 = metric_at_keep(learned, 4)["mean_js"]
    random_k4 = mean([metric_at_keep(row, 4)["mean_js"] for row in random_arms])
    permutation_k4 = mean([metric_at_keep(row, 4)["mean_js"] for row in permutation_arms])
    learned_r2 = learned["recurrence"]["quadratic"]["r2"]
    random_r2 = mean([row["recurrence"]["quadratic"]["r2"] for row in random_arms])
    permutation_r2 = mean([
        row["recurrence"]["quadratic"]["r2"] for row in permutation_arms
    ])
    integrity_rows = [learned, *random_arms, *permutation_arms]
    integrity = {
        "mass_conservation_le_1e_8": all(
            row["hierarchy"]["mass_conservation_max_abs"] <= 1e-8 for row in integrity_rows
        ),
        "entropy_chain_le_1e_8": all(
            row["hierarchy"]["entropy_chain_max_abs"] <= 1e-8 for row in integrity_rows
        ),
        "heldout_ce_chain_le_1e_8": all(
            row["hierarchy"]["heldout_ce_chain_max_abs"] <= 1e-8 for row in integrity_rows
        ),
        "full_reconstruction_le_1e_8": all(
            row["hierarchy"]["full_reconstruction_max_abs"] <= 1e-8 for row in integrity_rows
        ),
        "balanced_leaves": all(len(set(row["leaf_counts"])) == 1 for row in integrity_rows),
    }
    discovery = {
        "c01_learned_k4_js": learned_k4,
        "c01_random_mean_k4_js": random_k4,
        "c01_permutation_mean_k4_js": permutation_k4,
        "c01_below_both_controls_by_5pct": (
            learned_k4 <= 0.95 * random_k4 and learned_k4 <= 0.95 * permutation_k4
        ),
        "c02_learned_quadratic_r2": learned_r2,
        "c02_random_mean_quadratic_r2": random_r2,
        "c02_permutation_mean_quadratic_r2": permutation_r2,
        "c02_r2_ge_0_10_and_margin_ge_0_02": (
            learned_r2 >= 0.10
            and learned_r2 >= random_r2 + 0.02
            and learned_r2 >= permutation_r2 + 0.02
        ),
    }
    result = {
        "claim": ["S1-MS-MICROSCOPE-M1-I01", "S1-MS-MICROSCOPE-M1-C01", "S1-MS-MICROSCOPE-M1-C02"],
        "config": vars(args),
        "inputs": {
            "data_sha256": legacy.sha256(args.data),
            "spm_sha256": legacy.sha256(args.spm_model),
            **vocabulary,
            "count_stats": count_stats,
        },
        "structure": {
            "split_trace": split_trace,
            "learned_leaf_sha256": tensor_sha256(learned_leaf),
        },
        "learned": learned,
        "random_controls": random_arms,
        "leaf_permutation_controls": permutation_arms,
        "integrity_gates": integrity,
        "integrity_pass": all(integrity.values()),
        "discovery_flags": discovery,
    }
    result["all_finite"] = finite_tree(result)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "integrity_pass": result["integrity_pass"],
        "all_finite": result["all_finite"],
        "discovery_flags": discovery,
        "learned_energy": learned["hierarchy"]["detail_energy_by_depth"],
        "learned_entropy": learned["hierarchy"]["entropy_contribution_by_depth"],
        "learned_heldout_ce": learned["hierarchy"]["heldout_ce_contribution_by_depth"],
    }, indent=2), flush=True)
    if not result["integrity_pass"] or not result["all_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

