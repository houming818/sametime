#!/usr/bin/env python3
"""T1 held-out context-posterior annealing over a frozen T0 tree."""

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
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def held_out_events(
    cfg: legacy.Config,
    processor: spm.SentencePieceProcessor,
    target_ids: Sequence[int],
    context_ids: Sequence[int],
    required_contexts: int,
    max_events: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    target_map = {token: index for index, token in enumerate(target_ids)}
    context_map = {token: index for index, token in enumerate(context_ids)}
    gold: List[int] = []
    observed: List[List[int]] = []
    test_lines = 0
    candidate_positions = 0

    for line_index, _raw, text in legacy.iter_english(cfg.data, cfg.max_scan_lines):
        if line_index % cfg.test_mod:
            continue
        test_lines += 1
        ids = [token for token in processor.encode(text, out_type=int) if token >= 4]
        ids = ids[: cfg.max_sentence_tokens]
        for position, token in enumerate(ids):
            target = target_map.get(token)
            if target is None:
                continue
            candidate_positions += 1
            neighbours: List[Tuple[int, int, int]] = []
            for other in range(max(0, position - cfg.window), min(len(ids), position + cfg.window + 1)):
                if other == position:
                    continue
                context = context_map.get(ids[other])
                if context is None:
                    continue
                distance = abs(other - position)
                side = 0 if other < position else 1
                neighbours.append((distance, side, context))
            neighbours.sort(key=lambda item: (item[0], item[1]))
            if len(neighbours) < required_contexts:
                continue
            gold.append(target)
            observed.append([item[2] for item in neighbours[:required_contexts]])
            if len(gold) >= max_events:
                return (
                    torch.tensor(gold, dtype=torch.long),
                    torch.tensor(observed, dtype=torch.long),
                    {
                        "test_lines_scanned": test_lines,
                        "candidate_positions": candidate_positions,
                        "paired_events": len(gold),
                    },
                )
    return (
        torch.tensor(gold, dtype=torch.long),
        torch.tensor(observed, dtype=torch.long),
        {
            "test_lines_scanned": test_lines,
            "candidate_positions": candidate_positions,
            "paired_events": len(gold),
        },
    )


def context_bins(leaf: torch.Tensor, contexts: torch.Tensor, bins: int) -> torch.Tensor:
    divisor = len(torch.unique(leaf)) // bins
    if divisor < 1:
        raise ValueError("bins exceed leaf count")
    return leaf[contexts] // divisor


@torch.no_grad()
def posterior_metrics(
    probability: torch.Tensor,
    observation: torch.Tensor,
    gold: torch.Tensor,
    prior: torch.Tensor,
    context_steps: Sequence[int],
    batch_size: int,
    keep_nll_at: int | None = None,
) -> Tuple[Dict[str, Dict[str, float]], torch.Tensor | None]:
    log_probability = probability.clamp_min(EPS).log().to(torch.float32)
    log_prior = prior.clamp_min(EPS).log().to(torch.float32)
    steps = set(context_steps)
    totals = {
        count: {"nll": 0.0, "entropy": 0.0, "top1": 0, "top5": 0, "mrr": 0.0, "events": 0}
        for count in context_steps
    }
    kept: List[torch.Tensor] = []
    for start in range(0, len(gold), batch_size):
        batch_gold = gold[start : start + batch_size]
        batch_obs = observation[start : start + batch_size]
        score = log_prior.unsqueeze(0).expand(len(batch_gold), -1).clone()
        for index in range(batch_obs.shape[1]):
            score += log_probability[:, batch_obs[:, index]].t()
            count = index + 1
            if count not in steps:
                continue
            log_posterior = score - torch.logsumexp(score, dim=1, keepdim=True)
            row = torch.arange(len(batch_gold), device=score.device)
            event_nll = -log_posterior[row, batch_gold]
            posterior = log_posterior.exp()
            entropy = -(posterior * log_posterior).sum(dim=1)
            order = torch.argsort(score, dim=1, descending=True)
            rank = (order == batch_gold[:, None]).to(torch.int64).argmax(dim=1) + 1
            bucket = totals[count]
            bucket["nll"] += float(event_nll.sum())
            bucket["entropy"] += float(entropy.sum())
            bucket["top1"] += int((rank == 1).sum())
            bucket["top5"] += int((rank <= 5).sum())
            bucket["mrr"] += float((1.0 / rank.to(torch.float32)).sum())
            bucket["events"] += len(batch_gold)
            if keep_nll_at == count:
                kept.append(event_nll.cpu())
    result: Dict[str, Dict[str, float]] = {}
    for count, bucket in totals.items():
        events = max(1, int(bucket["events"]))
        result[str(count)] = {
            "events": int(bucket["events"]),
            "nll": bucket["nll"] / events,
            "entropy": bucket["entropy"] / events,
            "top1": bucket["top1"] / events,
            "top5": bucket["top5"] / events,
            "mrr": bucket["mrr"] / events,
        }
    return result, torch.cat(kept) if kept else None


def paired_bootstrap_margin(
    tree_nll: torch.Tensor,
    random_nll: Sequence[torch.Tensor],
    seed: int,
    samples: int,
) -> Dict[str, float]:
    random_mean = torch.stack(list(random_nll), dim=0).mean(dim=0)
    delta = random_mean - tree_nll
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(samples):
        index = torch.randint(0, len(delta), (len(delta),), generator=generator)
        estimates.append(float(delta[index].mean()))
    ordered = sorted(estimates)
    return {
        "mean": float(delta.mean()),
        "lower_95": ordered[int(0.025 * (samples - 1))],
        "upper_95": ordered[int(0.975 * (samples - 1))],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--target-vocab", type=int, default=2_048)
    parser.add_argument("--context-vocab", type=int, default=1_024)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--power-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--context-steps", default="1,2,4")
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.depth != 4:
        raise ValueError("T1 contract fixes depth=4")
    context_steps = sorted({int(value) for value in args.context_steps.split(",")})
    if not context_steps or context_steps[-1] > 2 * args.window:
        raise ValueError("invalid context steps")
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
    random_leaves = [
        t0.random_leaf_assignment(args.context_vocab, 16, args.seed + 100 + index, device)
        for index in range(args.random_replicates)
    ]
    gold_cpu, contexts_cpu, event_stats = held_out_events(
        cfg,
        processor,
        target_ids,
        context_ids,
        context_steps[-1],
        args.max_events,
    )
    if not len(gold_cpu):
        raise RuntimeError("no paired held-out events")
    gold = gold_cpu.to(device)
    contexts = contexts_cpu.to(device)
    prior = train.sum(dim=1).to(torch.float64).clamp_min(1.0)
    prior /= prior.sum()

    scales: List[Dict[str, object]] = []
    comparison_tree_nll = None
    comparison_random_nll: List[torch.Tensor] = []
    for bins in (1, 2, 4, 8, 16):
        learned_probability = t0.coarsen(
            t0.aggregate_by_leaf(train, learned_leaf, 16, args.alpha), bins
        )
        learned_observation = context_bins(learned_leaf, contexts, bins)
        learned_metrics, learned_nll = posterior_metrics(
            learned_probability,
            learned_observation,
            gold,
            prior,
            context_steps,
            args.batch_size,
            keep_nll_at=context_steps[-1] if bins == 16 else None,
        )
        random_metrics = []
        for random_leaf in random_leaves:
            random_probability = t0.coarsen(
                t0.aggregate_by_leaf(train, random_leaf, 16, args.alpha), bins
            )
            random_observation = context_bins(random_leaf, contexts, bins)
            metrics, event_nll = posterior_metrics(
                random_probability,
                random_observation,
                gold,
                prior,
                context_steps,
                args.batch_size,
                keep_nll_at=context_steps[-1] if bins == 16 else None,
            )
            random_metrics.append(metrics)
            if bins == 16 and event_nll is not None:
                comparison_random_nll.append(event_nll)
        if bins == 16:
            comparison_tree_nll = learned_nll
        scales.append({
            "bins": bins,
            "learned": learned_metrics,
            "random_replicates": random_metrics,
        })
        print(json.dumps(scales[-1]), flush=True)

    full_probability = t0.normalize_rows(train.to(torch.float64), args.alpha)
    full_metrics, _ = posterior_metrics(
        full_probability,
        contexts,
        gold,
        prior,
        context_steps,
        args.batch_size,
    )
    if comparison_tree_nll is None or not comparison_random_nll:
        raise RuntimeError("missing registered comparison arrays")
    bootstrap = paired_bootstrap_margin(
        comparison_tree_nll,
        comparison_random_nll,
        args.seed + 500,
        args.bootstrap_samples,
    )

    def scale(bins: int) -> Dict[str, object]:
        return next(row for row in scales if row["bins"] == bins)

    def mean_random_nll(row: Dict[str, object], step: int) -> float:
        return sum(
            replicate[str(step)]["nll"] for replicate in row["random_replicates"]
        ) / len(row["random_replicates"])

    finest = scale(16)
    coarse = scale(4)
    max_step = context_steps[-1]
    finest_one = finest["learned"][str(context_steps[0])]
    finest_many = finest["learned"][str(max_step)]
    c01_margin = mean_random_nll(finest, max_step) - finest_many["nll"]
    c03_margin = mean_random_nll(coarse, max_step) - coarse["learned"][str(max_step)]["nll"]
    claims = {
        "c01_random_minus_tree_nll_ge_0_02": c01_margin >= 0.02,
        "c01_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
        "c02_entropy_drop_ge_0_05": finest_one["entropy"] - finest_many["entropy"] >= 0.05,
        "c02_nll_drop_ge_0_05": finest_one["nll"] - finest_many["nll"] >= 0.05,
        "c03_coarse_random_minus_tree_nll_ge_0_01": c03_margin >= 0.01,
    }
    simplex_error = float(
        (t0.aggregate_by_leaf(train, learned_leaf, 16, args.alpha).sum(dim=1) - 1.0)
        .abs()
        .max()
    )
    result = {
        "claim": ["S1-MS-CONTEXT-T1-C01", "S1-MS-CONTEXT-T1-C02", "S1-MS-CONTEXT-T1-C03"],
        "config": vars(args),
        "inputs": {
            "data_sha256": legacy.sha256(args.data),
            "spm_sha256": legacy.sha256(args.spm_model),
            **vocabulary,
            "count_stats": count_stats,
            "event_stats": event_stats,
        },
        "structure": {
            "learned_leaf_counts": torch.bincount(learned_leaf, minlength=16).cpu().tolist(),
            "random_leaf_counts": [
                torch.bincount(leaf, minlength=16).cpu().tolist() for leaf in random_leaves
            ],
            "split_trace": split_trace,
            "simplex_max_abs": simplex_error,
        },
        "scales": scales,
        "uncompressed_context_reference": full_metrics,
        "registered_margins": {
            "finest_random_minus_tree_nll": c01_margin,
            "coarse_random_minus_tree_nll": c03_margin,
            "finest_entropy_drop": finest_one["entropy"] - finest_many["entropy"],
            "finest_nll_drop": finest_one["nll"] - finest_many["nll"],
            "paired_bootstrap": bootstrap,
        },
        "claim_gates": claims,
    }
    result["integrity_gates"] = {
        "all_16_leaves_used": int((torch.bincount(learned_leaf, minlength=16) > 0).sum()) == 16,
        "simplex_le_1e_6": simplex_error <= 1e-6,
        "paired_events_ge_2000": len(gold) >= 2_000,
        "held_out_lines_present": event_stats["test_lines_scanned"] > 0,
        "all_finite": finite_tree(result),
    }
    result["integrity_pass"] = all(result["integrity_gates"].values())
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "event_contract.json").write_text(
        json.dumps({
            "gold_sha256": tensor_sha256(gold_cpu),
            "contexts_sha256": tensor_sha256(contexts_cpu),
            "events": len(gold_cpu),
            "required_contexts": context_steps[-1],
            "selection": "distance ascending, left before right on ties",
            "split": "source line index modulo 10 equals zero",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "integrity_pass": result["integrity_pass"],
        "claim_gates": claims,
        "margins": result["registered_margins"],
    }, indent=2), flush=True)
    if not result["integrity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
