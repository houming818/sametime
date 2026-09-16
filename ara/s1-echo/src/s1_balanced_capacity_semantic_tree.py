#!/usr/bin/env python3
"""Balanced-capacity TreeHeap from real token-context probability laws."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sentencepiece as spm
import torch

import s1_annealed_fold_unfold_token_space as anneal
import s1_real_corpus_annealed_token_space as real


@dataclass
class Config:
    experiment: str
    data: str
    spm_model: str
    evidence_dir: str
    max_scan_lines: int
    target_vocab: int
    context_vocab: int
    window: int
    max_sentence_tokens: int
    test_mod: int
    depth: int
    iterations: int
    alpha: float
    neighbor_k: int
    seed: int
    device: str


def balanced_split(x: torch.Tensor, indices: torch.Tensor, iterations: int) -> Tuple[torch.Tensor, torch.Tensor]:
    local = x[indices]
    count = len(indices)
    if count % 2:
        raise ValueError(f"balanced split requires an even node size, got {count}")
    mean = local.mean(dim=0)
    axis = anneal.deterministic_axis(local, torch.ones(count, dtype=local.dtype, device=local.device))
    order = torch.argsort((local - mean) @ axis, stable=True)
    left_mask = torch.zeros(count, dtype=torch.bool, device=local.device)
    left_mask[order[: count // 2]] = True

    for _ in range(iterations):
        left_center = local[left_mask].mean(dim=0)
        right_center = local[~left_mask].mean(dim=0)
        delta = ((local - left_center) ** 2).sum(dim=1) - ((local - right_center) ** 2).sum(dim=1)
        order = torch.argsort(delta, stable=True)
        updated = torch.zeros_like(left_mask)
        updated[order[: count // 2]] = True
        if torch.equal(updated, left_mask):
            break
        left_mask = updated
    return indices[left_mask], indices[~left_mask]


def build_balanced_semantic_tree(x: torch.Tensor, depth: int, iterations: int) -> Dict[str, object]:
    token_count = x.shape[0]
    if token_count % (2**depth):
        raise ValueError("target vocabulary must be divisible by leaf count")
    routes = torch.zeros((token_count, depth), dtype=torch.long, device=x.device)
    nodes = [torch.arange(token_count, device=x.device)]
    max_imbalance = 0
    fold_error = 0.0
    prototypes: List[List[torch.Tensor]] = []

    for level in range(depth):
        next_nodes = []
        level_prototypes = []
        for node_index, indices in enumerate(nodes):
            left, right = balanced_split(x, indices, iterations)
            routes[right, level] = 1
            next_nodes.extend((left, right))
            max_imbalance = max(max_imbalance, abs(len(left) - len(right)))
            left_center = x[left].mean(dim=0)
            right_center = x[right].mean(dim=0)
            parent_center = x[indices].mean(dim=0)
            folded = (len(left) * left_center + len(right) * right_center) / len(indices)
            fold_error = max(fold_error, float((parent_center - folded).abs().max().item()))
            level_prototypes.extend((left_center, right_center))
        nodes = next_nodes
        prototypes.append(level_prototypes)

    powers = 2 ** torch.arange(depth - 1, -1, -1, device=x.device)
    leaf = (routes * powers).sum(dim=1)
    return {
        "routes": routes,
        "leaf": leaf,
        "nodes": nodes,
        "prototypes": prototypes,
        "max_capacity_imbalance": max_imbalance,
        "fold_conservation_max_abs": fold_error,
    }


def routes_from_leaf(leaf: torch.Tensor, depth: int) -> torch.Tensor:
    return torch.stack([((leaf >> shift) & 1) for shift in range(depth - 1, -1, -1)], dim=1)


def balanced_control(kind: str, frequency: torch.Tensor, depth: int, seed: int) -> Dict[str, torch.Tensor]:
    count = len(frequency)
    if kind == "frequency":
        order = torch.argsort(frequency, stable=True)
    elif kind == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(count, generator=generator).to(frequency.device)
    else:
        raise ValueError(kind)
    leaf = torch.empty(count, dtype=torch.long, device=frequency.device)
    capacity = count // (2**depth)
    leaf[order] = torch.arange(count, device=frequency.device) // capacity
    return {"leaf": leaf, "routes": routes_from_leaf(leaf, depth)}


def lcp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a == b).to(torch.long).cumprod(dim=1).sum(dim=1)


def topology_metrics(
    routes: torch.Tensor,
    heldout_x: torch.Tensor,
    neighbor_k: int,
    seed: int,
) -> Dict[str, float]:
    distance = torch.cdist(heldout_x, heldout_x) ** 2
    distance.fill_diagonal_(float("inf"))
    nearest = distance.topk(neighbor_k, largest=False).indices
    centers = torch.arange(len(routes), device=routes.device)[:, None].expand_as(nearest)
    near_lcp = lcp(routes[centers.reshape(-1)], routes[nearest.reshape(-1)]).to(torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    random_other = torch.randint(0, len(routes), centers.shape, generator=generator).to(routes.device)
    same = random_other == centers
    random_other[same] = (random_other[same] + 1) % len(routes)
    random_lcp = lcp(routes[centers.reshape(-1)], routes[random_other.reshape(-1)]).to(torch.float64)
    return {
        "heldout_neighbor_lcp": float(near_lcp.mean().item()),
        "random_pair_lcp": float(random_lcp.mean().item()),
        "lcp_gain": float((near_lcp.mean() - random_lcp.mean()).item()),
    }


def leaf_metrics(leaf: torch.Tensor, depth: int) -> Dict[str, float]:
    counts = torch.bincount(leaf, minlength=2**depth).to(torch.float64)
    occupied = counts[counts > 0]
    probability = occupied / occupied.sum()
    return {
        "leaf_utilization": float(len(occupied) / (2**depth)),
        "leaf_entropy": float((-(probability * probability.log()).sum() / math.log(2**depth)).item()),
        "leaf_count_min": int(counts.min().item()),
        "leaf_count_max": int(counts.max().item()),
    }


def heldout_nll(leaf: torch.Tensor, train: torch.Tensor, test: torch.Tensor, alpha: float) -> float:
    total_loss = 0.0
    total_count = float(test.sum().item())
    for leaf_id in leaf.unique().tolist():
        members = leaf == leaf_id
        prototype = train[members].sum(dim=0) + alpha
        prototype /= prototype.sum()
        total_loss -= float((test[members] * prototype.log()[None, :]).sum().item())
    return total_loss / max(total_count, 1.0)


def readable_leaves(leaf: torch.Tensor, pieces: Sequence[str], limit: int = 12) -> List[Dict[str, object]]:
    rows = []
    for leaf_id in leaf.unique(sorted=True).tolist()[:limit]:
        members = torch.nonzero(leaf == leaf_id, as_tuple=False).flatten().tolist()
        rows.append({"leaf": int(leaf_id), "pieces": [pieces[i] for i in members]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="P-S1-BALANCED-CAPACITY01")
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--max-scan-lines", type=int, default=50_000)
    parser.add_argument("--target-vocab", type=int, default=256)
    parser.add_argument("--context-vocab", type=int, default=512)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-sentence-tokens", type=int, default=96)
    parser.add_argument("--test-mod", type=int, default=10)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19401)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = Config(**vars(args))
    started = time.time()
    evidence = Path(cfg.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    vocab = real.choose_vocab(cfg, sp)
    train, test, corpus = real.build_counts(cfg, sp, vocab["target_ids"], vocab["context_ids"])
    train_probability = anneal.distributions(train, cfg.alpha)
    test_probability = anneal.distributions(test, cfg.alpha)
    x = train_probability.sqrt().to(cfg.device)
    heldout_x = test_probability.sqrt().to(cfg.device)
    frequency = train.sum(dim=1).to(cfg.device)
    pieces = [sp.id_to_piece(i) for i in vocab["target_ids"]]

    semantic = build_balanced_semantic_tree(x, cfg.depth, cfg.iterations)
    arms = {
        "semantic": semantic,
        "random": balanced_control("random", frequency, cfg.depth, cfg.seed),
        "frequency": balanced_control("frequency", frequency, cfg.depth, cfg.seed),
    }
    metrics: Dict[str, object] = {}
    for name, arm in arms.items():
        row = {
            **leaf_metrics(arm["leaf"], cfg.depth),
            **topology_metrics(arm["routes"], heldout_x, cfg.neighbor_k, cfg.seed),
            "heldout_context_nll": heldout_nll(arm["leaf"].cpu(), train, test, cfg.alpha),
            "readable_leaves": readable_leaves(arm["leaf"].cpu(), pieces),
        }
        if name == "semantic":
            row["max_capacity_imbalance"] = semantic["max_capacity_imbalance"]
            row["fold_conservation_max_abs"] = semantic["fold_conservation_max_abs"]
        metrics[name] = row

    s, r, f = metrics["semantic"], metrics["random"], metrics["frequency"]
    gates = {
        "exact_capacity": s["max_capacity_imbalance"] == 0,
        "full_leaf_use": s["leaf_utilization"] == 1.0 and abs(s["leaf_entropy"] - 1.0) < 1e-12,
        "nll_beats_random": s["heldout_context_nll"] < r["heldout_context_nll"],
        "neighbor_lcp_gain_ge_0_25": s["lcp_gain"] >= 0.25,
        "neighbor_lcp_beats_controls": s["heldout_neighbor_lcp"] > max(r["heldout_neighbor_lcp"], f["heldout_neighbor_lcp"]),
        "finite_fold": math.isfinite(s["fold_conservation_max_abs"]),
    }
    summary = {
        "claim": "S1-BALANCED-CAPACITY-C01",
        "experiment": cfg.experiment,
        "config": asdict(cfg),
        "corpus": {**corpus, "valid_lines": vocab["valid_lines"], "token_total": vocab["token_total"], "scan_sha256": vocab["scan_sha256"]},
        "metrics": metrics,
        "gates": gates,
        "smoke_pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
