#!/usr/bin/env python3
"""Observe scale-by-scale refinement of the dynamic T1 token posterior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import sentencepiece as spm
import torch

import s1_multiscale_context_posterior_t1 as t1
import s1_multiscale_probability_embedding_t0 as t0
import s1_treeheap_background_landmarks as legacy


EPS = 1e-12


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def log_posterior(
    probability: torch.Tensor,
    observations: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    log_likelihood = probability.clamp_min(EPS).log().to(torch.float32)
    score = prior.clamp_min(EPS).log().to(torch.float32).unsqueeze(0).expand(
        len(observations), -1
    ).clone()
    for index in range(observations.shape[1]):
        score += log_likelihood[:, observations[:, index]].t()
    return score - torch.logsumexp(score, dim=1, keepdim=True)


def spectrum(residual: torch.Tensor) -> Dict[str, float]:
    residual = residual - residual.mean(dim=1, keepdim=True)
    gram = residual @ residual.t() / max(1, residual.shape[1])
    eigen = torch.linalg.eigvalsh(gram.to(torch.float64)).clamp_min(0).flip(0)
    variance = eigen / eigen.sum().clamp_min(EPS)
    cumulative = variance.cumsum(dim=0)
    return {
        "events": residual.shape[0],
        "top1": float(cumulative[0]),
        "top4": float(cumulative[min(3, len(cumulative) - 1)]),
        "top16": float(cumulative[min(15, len(cumulative) - 1)]),
        "effective_rank": float(torch.exp(-(variance * variance.clamp_min(EPS).log()).sum())),
    }


@torch.no_grad()
def transition_audit(
    coarse_probability: torch.Tensor,
    fine_probability: torch.Tensor,
    coarse_observation: torch.Tensor,
    fine_observation: torch.Tensor,
    gold: torch.Tensor,
    prior: torch.Tensor,
    batch_size: int,
    spectrum_events: int,
) -> Dict[str, object]:
    coarse_log_field = coarse_probability.clamp_min(EPS).log().to(torch.float32)
    fine_log_field = fine_probability.clamp_min(EPS).log().to(torch.float32)
    total = 0
    totals = {
        "max_abs_log_closure": 0.0,
        "kl_fine_to_coarse": 0.0,
        "js": 0.0,
        "entropy_drop": 0.0,
        "gold_log_probability_gain": 0.0,
        "gold_gain_positive": 0,
        "residual_energy": 0.0,
    }
    residual_samples: List[torch.Tensor] = []
    for start in range(0, len(gold), batch_size):
        stop = min(start + batch_size, len(gold))
        batch_gold = gold[start:stop]
        coarse_obs = coarse_observation[start:stop]
        fine_obs = fine_observation[start:stop]
        coarse_log = log_posterior(coarse_probability, coarse_obs, prior)
        fine_log = log_posterior(fine_probability, fine_obs, prior)
        correction = torch.zeros_like(coarse_log)
        for index in range(fine_obs.shape[1]):
            correction += (
                fine_log_field[:, fine_obs[:, index]].t()
                - coarse_log_field[:, coarse_obs[:, index]].t()
            )
        reconstructed = coarse_log + correction
        reconstructed -= torch.logsumexp(reconstructed, dim=1, keepdim=True)
        totals["max_abs_log_closure"] = max(
            totals["max_abs_log_closure"], float((reconstructed - fine_log).abs().max())
        )
        coarse = coarse_log.exp()
        fine = fine_log.exp()
        middle = 0.5 * (coarse + fine)
        kl = (fine * (fine_log - coarse_log)).sum(dim=1)
        js = 0.5 * (
            (fine * (fine_log - middle.clamp_min(EPS).log())).sum(dim=1)
            + (coarse * (coarse_log - middle.clamp_min(EPS).log())).sum(dim=1)
        )
        coarse_entropy = -(coarse * coarse_log).sum(dim=1)
        fine_entropy = -(fine * fine_log).sum(dim=1)
        row = torch.arange(len(batch_gold), device=gold.device)
        gold_gain = fine_log[row, batch_gold] - coarse_log[row, batch_gold]
        centered = correction - (fine * correction).sum(dim=1, keepdim=True)
        energy = (fine * centered.square()).sum(dim=1)
        count = len(batch_gold)
        totals["kl_fine_to_coarse"] += float(kl.sum())
        totals["js"] += float(js.sum())
        totals["entropy_drop"] += float((coarse_entropy - fine_entropy).sum())
        totals["gold_log_probability_gain"] += float(gold_gain.sum())
        totals["gold_gain_positive"] += int((gold_gain > 0).sum())
        totals["residual_energy"] += float(energy.sum())
        total += count
        remaining = spectrum_events - sum(len(item) for item in residual_samples)
        if remaining > 0:
            residual_samples.append(correction[:remaining].cpu())
    return {
        "events": total,
        "max_abs_log_closure": totals["max_abs_log_closure"],
        "mean_kl_fine_to_coarse": totals["kl_fine_to_coarse"] / total,
        "mean_js": totals["js"] / total,
        "mean_entropy_drop": totals["entropy_drop"] / total,
        "mean_gold_log_probability_gain": totals["gold_log_probability_gain"] / total,
        "gold_gain_positive_fraction": totals["gold_gain_positive"] / total,
        "mean_residual_energy": totals["residual_energy"] / total,
        "residual_spectrum": spectrum(torch.cat(residual_samples, dim=0)),
    }


def arm_audit(
    name: str,
    train: torch.Tensor,
    leaf: torch.Tensor,
    contexts: torch.Tensor,
    gold: torch.Tensor,
    prior: torch.Tensor,
    leaves: int,
    alpha: float,
    context_steps: Sequence[int],
    batch_size: int,
    spectrum_events: int,
) -> Dict[str, object]:
    finest = t0.aggregate_by_leaf(train, leaf, leaves, alpha)
    scales = {bins: t0.coarsen(finest, bins) for bins in (1, 2, 4, 8, 16)}
    output = []
    for count in context_steps:
        selected_contexts = contexts[:, :count]
        for coarse_bins, fine_bins in ((1, 2), (2, 4), (4, 8), (8, 16)):
            coarse_obs = t1.context_bins(leaf, selected_contexts, coarse_bins)
            fine_obs = t1.context_bins(leaf, selected_contexts, fine_bins)
            row = transition_audit(
                scales[coarse_bins],
                scales[fine_bins],
                coarse_obs,
                fine_obs,
                gold,
                prior,
                batch_size,
                spectrum_events,
            )
            row.update({
                "contexts": count,
                "coarse_bins": coarse_bins,
                "fine_bins": fine_bins,
            })
            output.append(row)
    frontloading = []
    for count in context_steps:
        rows = sorted(
            [row for row in output if row["contexts"] == count],
            key=lambda row: row["coarse_bins"],
        )
        gains = [row["mean_gold_log_probability_gain"] for row in rows]
        total_gain = sum(gains)
        through_k4 = sum(gains[:2])
        frontloading.append({
            "contexts": count,
            "total_gold_log_probability_gain_k1_to_k16": total_gain,
            "gold_gain_through_k2": gains[0],
            "gold_gain_through_k4": through_k4,
            "frontloading_fraction_through_k2": gains[0] / max(EPS, total_gain),
            "frontloading_fraction_through_k4": through_k4 / max(EPS, total_gain),
        })
    return {"name": name, "transitions": output, "frontloading": frontloading}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--target-vocab", type=int, default=2_048)
    parser.add_argument("--context-vocab", type=int, default=1_024)
    parser.add_argument("--max-events", type=int, default=5_000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--power-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--context-steps", default="1,2,4")
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--spectrum-events", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    leaves = 2**args.depth
    context_steps = sorted({int(value) for value in args.context_steps.split(",")})
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
    train_cpu, _test_cpu, count_stats = legacy.build_counts(cfg, processor, target_ids, context_ids)
    train = train_cpu.to(device)
    features, context_mass = t0.context_features(train, args.alpha)
    learned_leaf, split_trace = t0.learned_leaf_assignment(
        features, context_mass, args.depth, args.power_steps
    )
    gold_cpu, contexts_cpu, event_stats = t1.held_out_events(
        cfg, processor, target_ids, context_ids, context_steps[-1], args.max_events
    )
    gold = gold_cpu.to(device)
    contexts = contexts_cpu.to(device)
    prior = train.sum(dim=1).to(torch.float64).clamp_min(1.0)
    prior /= prior.sum()

    learned = arm_audit(
        "learned", train, learned_leaf, contexts, gold, prior, leaves, args.alpha,
        context_steps, args.batch_size, args.spectrum_events,
    )
    random_controls = []
    permutation_controls = []
    for index in range(args.controls):
        random_leaf = t0.random_leaf_assignment(
            args.context_vocab, leaves, args.seed + 100 + index, device
        )
        random_controls.append(arm_audit(
            f"random_{index}", train, random_leaf, contexts, gold, prior, leaves,
            args.alpha, context_steps, args.batch_size, args.spectrum_events,
        ))
        generator = torch.Generator().manual_seed(args.seed + 200 + index)
        permutation = torch.randperm(leaves, generator=generator).to(device)
        permutation_controls.append(arm_audit(
            f"leaf_permutation_{index}", train, permutation[learned_leaf], contexts,
            gold, prior, leaves, args.alpha, context_steps, args.batch_size,
            args.spectrum_events,
        ))

    arms = [learned, *random_controls, *permutation_controls]
    max_closure = max(
        transition["max_abs_log_closure"]
        for arm in arms for transition in arm["transitions"]
    )
    learned_m4 = next(row for row in learned["frontloading"] if row["contexts"] == 4)
    random_m4 = [
        next(row for row in arm["frontloading"] if row["contexts"] == 4)
        for arm in random_controls
    ]
    permutation_m4 = [
        next(row for row in arm["frontloading"] if row["contexts"] == 4)
        for arm in permutation_controls
    ]
    random_frontloading = sum(row["frontloading_fraction_through_k4"] for row in random_m4) / len(random_m4)
    permutation_frontloading = sum(
        row["frontloading_fraction_through_k4"] for row in permutation_m4
    ) / len(permutation_m4)
    formal_gate = {
        "learned_total_gold_gain_ge_0_10": (
            learned_m4["total_gold_log_probability_gain_k1_to_k16"] >= 0.10
        ),
        "learned_frontloading_fraction_through_k4": learned_m4["frontloading_fraction_through_k4"],
        "random_mean_frontloading_fraction_through_k4": random_frontloading,
        "permutation_mean_frontloading_fraction_through_k4": permutation_frontloading,
        "learned_minus_random_ge_0_15": (
            learned_m4["frontloading_fraction_through_k4"] - random_frontloading >= 0.15
        ),
        "learned_minus_permutation_ge_0_10": (
            learned_m4["frontloading_fraction_through_k4"] - permutation_frontloading >= 0.10
        ),
    }
    formal_gate["c03_pass"] = (
        formal_gate["learned_total_gold_gain_ge_0_10"]
        and formal_gate["learned_minus_random_ge_0_15"]
        and formal_gate["learned_minus_permutation_ge_0_10"]
    )
    result = {
        "claim": ["S1-MS-MICROSCOPE-M1-I02", "S1-MS-MICROSCOPE-M1-C03"],
        "config": vars(args),
        "inputs": {
            "data_sha256": legacy.sha256(args.data),
            "spm_sha256": legacy.sha256(args.spm_model),
            **vocabulary,
            "count_stats": count_stats,
            "event_stats": event_stats,
            "events": len(gold),
        },
        "structure": {"split_trace": split_trace},
        "learned": learned,
        "random_controls": random_controls,
        "leaf_permutation_controls": permutation_controls,
        "integrity": {
            "max_abs_log_closure": max_closure,
            "posterior_refinement_le_2e_5": max_closure <= 2e-5,
            "all_16_leaves_used": int((torch.bincount(learned_leaf, minlength=16) > 0).sum()) == 16,
        },
        "formal_discovery_gate": formal_gate,
    }
    result["all_finite"] = finite_tree(result)
    result["integrity_pass"] = all(result["integrity"].values()) and result["all_finite"]
    (output / "posterior_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "integrity_pass": result["integrity_pass"],
        "max_abs_log_closure": max_closure,
        "formal_discovery_gate": formal_gate,
        "learned_transitions": learned["transitions"],
    }, indent=2), flush=True)
    if not result["integrity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

