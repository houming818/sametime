#!/usr/bin/env python3
"""Multiscale probability-tree embedding gate over frozen corpus counts.

Owner: Nio / Trinity
Author: GPT-5.6 Sol
Created: 2026-10-07
Updated: 2026-10-07
Responsibility: construct a 16-leaf context tree and compare multiscale
probability fields with random, frequency and SGNS controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

import s1_treeheap_background_landmarks as legacy


EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(value: torch.Tensor, alpha: float) -> torch.Tensor:
    value = value.to(torch.float64).clamp_min(0) + alpha
    return value / value.sum(dim=1, keepdim=True).clamp_min(EPS)


def context_features(counts: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    context_mass = counts.sum(dim=0).clamp_min(1.0)
    probability = (counts + alpha) / (counts.sum(dim=0, keepdim=True) + alpha * counts.shape[0]).clamp_min(EPS)
    return probability.sqrt().t().contiguous(), context_mass


def principal_projection(local: torch.Tensor, weights: torch.Tensor, steps: int) -> torch.Tensor:
    center = (weights[:, None] * local).sum(dim=0) / weights.sum().clamp_min(EPS)
    centered = local - center
    index = torch.arange(len(local), device=local.device, dtype=torch.int64)
    signs = (((index * 1103515245 + 12345) >> 8) & 1).to(local.dtype) * 2.0 - 1.0
    axis = centered.t() @ (signs * weights.sqrt())
    if float(axis.norm()) < EPS:
        axis = centered[0].clone()
    axis = axis / axis.norm().clamp_min(EPS)
    for _ in range(steps):
        projected = centered @ axis
        axis = centered.t() @ (weights * projected)
        axis = axis / axis.norm().clamp_min(EPS)
    pivot = int(axis.abs().argmax())
    if axis[pivot] < 0:
        axis = -axis
    return centered @ axis


def learned_leaf_assignment(
    features: torch.Tensor, weights: torch.Tensor, depth: int, power_steps: int
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    nodes = [torch.arange(len(features), device=features.device)]
    trace: List[Dict[str, float]] = []
    for level in range(depth):
        children = []
        for node_index, indices in enumerate(nodes):
            local = features[indices]
            local_weights = weights[indices]
            projection = principal_projection(local, local_weights, power_steps)
            order = torch.argsort(projection, stable=True)
            half = max(1, len(order) // 2)
            left = indices[order[:half]]
            right = indices[order[half:]]
            if not len(left) or not len(right):
                raise RuntimeError("empty learned tree child")
            children.extend([left, right])
            trace.append({
                "level": level,
                "node": node_index,
                "contexts": len(indices),
                "left": len(left),
                "right": len(right),
                "projection_std": float(projection.std(unbiased=False)),
            })
        nodes = children
    leaf = torch.empty(len(features), dtype=torch.long, device=features.device)
    for leaf_index, indices in enumerate(nodes):
        leaf[indices] = leaf_index
    return leaf, trace


def random_leaf_assignment(contexts: int, leaves: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(contexts, generator=generator)
    leaf = torch.empty(contexts, dtype=torch.long)
    leaf[order] = torch.arange(contexts) * leaves // contexts
    return leaf.to(device)


def aggregate_by_leaf(counts: torch.Tensor, leaf: torch.Tensor, leaves: int, alpha: float) -> torch.Tensor:
    output = torch.zeros((counts.shape[0], leaves), dtype=torch.float64, device=counts.device)
    output.index_add_(1, leaf, counts.to(torch.float64))
    return normalize_rows(output, alpha)


def aggregate_counts(counts: torch.Tensor, leaf: torch.Tensor, leaves: int) -> torch.Tensor:
    output = torch.zeros((counts.shape[0], leaves), dtype=torch.float64, device=counts.device)
    output.index_add_(1, leaf, counts.to(torch.float64))
    return output


def coarsen(value: torch.Tensor, bins: int) -> torch.Tensor:
    if value.shape[1] % bins:
        raise ValueError("bins must divide leaf count")
    return value.reshape(value.shape[0], bins, value.shape[1] // bins).sum(dim=2)


def js_pair(probability: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    p = probability[left].clamp_min(EPS)
    q = probability[right].clamp_min(EPS)
    middle = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - middle.log())).sum(dim=1) + (q * (q.log() - middle.log())).sum(dim=1))


def simlex_rows(
    path: Path, sp: spm.SentencePieceProcessor, target_map: Dict[int, int]
) -> List[Dict[str, object]]:
    return legacy.load_simlex(str(path), sp, target_map)


def spearman_for_field(rows: Sequence[Dict[str, object]], probability: torch.Tensor) -> float:
    if len(rows) < 2:
        return float("nan")
    left = torch.tensor([int(row["i"]) for row in rows], device=probability.device)
    right = torch.tensor([int(row["j"]) for row in rows], device=probability.device)
    score = (-js_pair(probability, left, right)).cpu().tolist()
    return legacy.spearman([float(row["human"]) for row in rows], score)


def spearman_for_vectors(rows: Sequence[Dict[str, object]], embedding: torch.Tensor) -> float:
    score = [float((embedding[int(row["i"])] * embedding[int(row["j"])]).sum()) for row in rows]
    return legacy.spearman([float(row["human"]) for row in rows], score)


@torch.no_grad()
def retrieval(
    train_probability: torch.Tensor,
    test_counts: torch.Tensor,
    prior: torch.Tensor,
    batch: int,
) -> Dict[str, float]:
    valid_rows = torch.where(test_counts.sum(dim=1) > 0)[0]
    log_probability = train_probability.clamp_min(EPS).log()
    log_prior = prior.clamp_min(EPS).log()
    total_nll = 0.0
    top1 = 0
    top5 = 0
    reciprocal = 0.0
    total = 0
    for start in range(0, len(valid_rows), batch):
        row_ids = valid_rows[start : start + batch]
        query = test_counts[row_ids]
        score = query @ log_probability.t() + log_prior.unsqueeze(0)
        gold = score[torch.arange(len(row_ids), device=score.device), row_ids]
        total_nll += float((-gold + torch.logsumexp(score, dim=1)).sum())
        order = torch.argsort(score, dim=1, descending=True)
        rank = (order == row_ids[:, None]).to(torch.int64).argmax(dim=1) + 1
        top1 += int((rank == 1).sum())
        top5 += int((rank <= 5).sum())
        reciprocal += float((1.0 / rank.to(torch.float64)).sum())
        total += len(row_ids)
    return {
        "rows": total,
        "nll": total_nll / max(1, total),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "mrr": reciprocal / max(1, total),
    }


def clean_piece(piece: str) -> str:
    return piece.replace("▁", "").strip()


def teacher_items(
    rows: Sequence[Dict[str, object]],
    pieces: Sequence[str],
    learned: torch.Tensor,
    random_field: torch.Tensor,
    sgns: torch.Tensor,
    limit: int,
    seed: int,
) -> List[Dict[str, object]]:
    anchors = []
    for row in sorted(rows, key=lambda item: -float(item["human"])):
        for index in (int(row["i"]), int(row["j"])):
            word = clean_piece(pieces[index])
            if word.isalpha() and index not in anchors:
                anchors.append(index)
    generator = random.Random(seed)
    output = []
    learned_distance = torch.cdist(learned.sqrt().to(torch.float32), learned.sqrt().to(torch.float32))
    random_distance = torch.cdist(random_field.sqrt().to(torch.float32), random_field.sqrt().to(torch.float32))
    sgns_similarity = sgns @ sgns.t()
    for anchor in anchors:
        candidates = {}
        for name, score, largest in (
            ("tree", learned_distance[anchor], False),
            ("random", random_distance[anchor], False),
            ("sgns", sgns_similarity[anchor], True),
        ):
            order = torch.argsort(score, descending=largest)
            chosen = None
            for candidate in order.tolist():
                word = clean_piece(pieces[candidate])
                if candidate != anchor and word.isalpha() and len(word) > 1:
                    chosen = candidate
                    break
            if chosen is not None:
                candidates[name] = chosen
        if len(candidates) != 3:
            continue
        for control in ("random", "sgns"):
            pair = [("tree", candidates["tree"]), (control, candidates[control])]
            generator.shuffle(pair)
            output.append({
                "kind": "blind_neighbor",
                "anchor": clean_piece(pieces[anchor]),
                "option_a": clean_piece(pieces[pair[0][1]]),
                "option_b": clean_piece(pieces[pair[1][1]]),
                "hidden_a": pair[0][0],
                "hidden_b": pair[1][0],
            })
            if len(output) >= limit:
                return output
    return output


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--simlex", default="ara/s1-echo/data/simlex999/SimLex-999.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--target-vocab", type=int, default=2_048)
    parser.add_argument("--context-vocab", type=int, default=1_024)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--power-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sgns-dim", type=int, default=64)
    parser.add_argument("--sgns-steps", type=int, default=500)
    parser.add_argument("--sgns-batch", type=int, default=1024)
    parser.add_argument("--sgns-negatives", type=int, default=8)
    parser.add_argument("--sgns-lr", type=float, default=0.02)
    parser.add_argument("--teacher-items", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.depth != 4:
        raise ValueError("T0 contract fixes depth=4 (16 leaves)")
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    cfg = legacy.Config(
        data=args.data,
        spm_model=args.spm_model,
        simlex=args.simlex,
        max_scan_lines=args.max_lines,
        target_vocab=args.target_vocab,
        context_vocab=args.context_vocab,
        window=args.window,
        max_sentence_tokens=96,
        test_mod=10,
        depth=4,
        bootstrap_seeds=1,
        seed_start=args.seed,
        temperatures=[1.0],
        em_steps=1,
        power_steps=args.power_steps,
        alpha=args.alpha,
        sgns_dim=args.sgns_dim,
        sgns_steps=args.sgns_steps,
        sgns_batch=args.sgns_batch,
        sgns_negatives=args.sgns_negatives,
        sgns_lr=args.sgns_lr,
        device=args.device,
    )

    vocab = legacy.choose_vocab(cfg, sp)
    corpus_counts = vocab.pop("counts")
    target_ids = vocab.pop("target_ids")
    context_ids = vocab.pop("context_ids")
    target_map = {token: index for index, token in enumerate(target_ids)}
    pieces = [sp.id_to_piece(token) for token in target_ids]
    train_cpu, test_cpu, count_stats = legacy.build_counts(cfg, sp, target_ids, context_ids)
    train = train_cpu.to(device)
    test = test_cpu.to(device)
    features, context_mass = context_features(train, args.alpha)
    learned_leaf, split_trace = learned_leaf_assignment(features, context_mass, args.depth, args.power_steps)
    random_leaf = random_leaf_assignment(args.context_vocab, 2**args.depth, args.seed + 1, device)
    learned_16 = aggregate_by_leaf(train, learned_leaf, 16, args.alpha)
    random_16 = aggregate_by_leaf(train, random_leaf, 16, args.alpha)
    learned_test_16 = aggregate_counts(test, learned_leaf, 16)
    random_test_16 = aggregate_counts(test, random_leaf, 16)
    prior = train.sum(dim=1).to(torch.float64).clamp_min(1.0)
    prior /= prior.sum()

    rows = simlex_rows(Path(args.simlex), sp, target_map)
    sgns, sgns_meta = legacy.train_sgns(train_cpu, cfg)
    sgns = sgns.to(device)
    scale_rows = []
    for bins in (1, 2, 4, 8, 16):
        learned_probability = coarsen(learned_16, bins)
        random_probability = coarsen(random_16, bins)
        learned_test = coarsen(learned_test_16, bins)
        random_test = coarsen(random_test_16, bins)
        scale_rows.append({
            "bins": bins,
            "simlex_tree_rho": spearman_for_field(rows, learned_probability),
            "simlex_random_rho": spearman_for_field(rows, random_probability),
            "tree_retrieval": retrieval(learned_probability, learned_test, prior, 256),
            "random_retrieval": retrieval(random_probability, random_test, prior, 256),
        })
        print(json.dumps(scale_rows[-1]), flush=True)

    frequency_score = [
        -abs(math.log(float(prior[int(row["i"])])) - math.log(float(prior[int(row["j"])])))
        for row in rows
    ]
    frequency_rho = legacy.spearman([float(row["human"]) for row in rows], frequency_score)
    sgns_rho = spearman_for_vectors(rows, sgns)
    leaf_counts = torch.bincount(learned_leaf, minlength=16)
    random_leaf_counts = torch.bincount(random_leaf, minlength=16)
    simplex_error = max(
        float((learned_16.sum(dim=1) - 1).abs().max()),
        float((random_16.sum(dim=1) - 1).abs().max()),
    )
    prior_retrieval = retrieval(torch.ones((args.target_vocab, 1), dtype=torch.float64, device=device), test.sum(dim=1, keepdim=True).to(torch.float64), prior, 256)
    full = scale_rows[-1]
    result = {
        "claim": ["S1-MS-PROB-EMBED-T0-C01", "S1-MS-PROB-EMBED-T0-C02"],
        "config": vars(args),
        "inputs": {
            "data_sha256": legacy.sha256(args.data),
            "spm_sha256": legacy.sha256(args.spm_model),
            "simlex_sha256": sha256(Path(args.simlex)),
            **vocab,
            "count_stats": count_stats,
        },
        "structure": {
            "learned_leaf_counts": leaf_counts.cpu().tolist(),
            "random_leaf_counts": random_leaf_counts.cpu().tolist(),
            "split_trace": split_trace,
            "simplex_max_abs": simplex_error,
        },
        "simlex": {
            "covered_pairs": len(rows),
            "scales": scale_rows,
            "sgns_rho": sgns_rho,
            "frequency_rho": frequency_rho,
        },
        "retrieval": {
            "prior": prior_retrieval,
            "scales": scale_rows,
        },
        "sgns": sgns_meta,
        "claim_gates": {
            "c01_tree_minus_random_rho_ge_0_03": full["simlex_tree_rho"] - full["simlex_random_rho"] >= 0.03,
            "c02_tree_nll_below_random_by_0_02": full["random_retrieval"]["nll"] - full["tree_retrieval"]["nll"] >= 0.02,
            "c02_tree_nll_below_prior": full["tree_retrieval"]["nll"] < prior_retrieval["nll"],
        },
    }
    result["integrity_gates"] = {
        "all_16_leaves_used": int((leaf_counts > 0).sum()) == 16,
        "simplex_le_1e_6": simplex_error <= 1e-6,
        "train_and_test_pairs_nonempty": count_stats["train_pairs"] > 0 and count_stats["test_pairs"] > 0,
        "simlex_coverage_ge_40": len(rows) >= 40,
        "sgns_completed": bool(sgns_meta["trace"]) and sgns_meta["trace"][-1]["step"] == args.sgns_steps,
        "all_finite": finite_tree(result),
    }
    result["integrity_pass"] = all(result["integrity_gates"].values())
    items = teacher_items(rows, pieces, learned_16, random_16, sgns, args.teacher_items, args.seed + 2)
    with (output / "teacher_items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "simlex_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# T0 multiscale probability embedding evidence\n\n```json\n"
        + json.dumps({"integrity": result["integrity_gates"], "claims": result["claim_gates"]}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"integrity_pass": result["integrity_pass"], "claim_gates": result["claim_gates"], "teacher_items": len(items)}, indent=2), flush=True)
    if not result["integrity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

